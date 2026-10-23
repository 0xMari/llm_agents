from copy import deepcopy
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from models import HotelOffer, HotelSearchQuery
from providers.hotels import parse_hotel_offer, search_hotels
from providers.links import parse_redirect
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError


RECORDED_FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures" / "serpapi"


def raw_hotel(name="Test Stay", total="249.99", nightly="50.01", rating=4.4):
    # Synthetic API-shaped fixture; no live prices or account data.
    return {
        "type": "hotel", "name": name,
        "rate_per_night": {"extracted_lowest": nightly},
        "total_rate": {"extracted_lowest": total},
        "overall_rating": rating,
        "link": "https://hotel.example/stay",
    }


def detailed_raw_hotel(**kwargs):
    # API-shaped details for offline tests, not a new real-data recording.
    return {
        **raw_hotel(**kwargs),
        "property_token": "test-property-token",
        "description": "Hotel with a pool and restaurant.",
        "address": "Test street, Lisbon",
        "hotel_class": "4-star hotel",
        "extracted_hotel_class": 4,
        "reviews": 128,
        "gps_coordinates": {"latitude": 38.7223, "longitude": -9.1393},
        "check_in_time": "3:00 PM",
        "check_out_time": "11:00 AM",
        "amenities": ["Free Wi-Fi", "Pool"],
        "excluded_amenities": ["No beach access"],
        "essential_info": ["Private room"],
        "images": [{
            "thumbnail": "https://images.example/thumbnail.jpg",
            "original_image": "https://images.example/original.jpg",
        }],
        "serpapi_property_details_link": "https://serpapi.com/search.json?property_token=test-property-token",
    }


def fake_hotel_search(query, *, budget=None):
    if query.destination != "Lisbon":
        return []
    nights = (query.checkout_date - query.checkin_date).days
    return [
        HotelOffer(
            destination=query.destination, name=name,
            nightly_price_eur=rate, rating=rating, nights=nights,
            total_price_eur=rate * nights,
        )
        for name, rate, rating in [("Central Stay", 95, 4.4), ("Simple Rooms", 62, 3.9)]
    ]


class HotelProviderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.api = self.enterContext(patch("providers.hotels.search_serpapi"))
        self.api.return_value = {"properties": [raw_hotel()]}
        self.query = HotelSearchQuery(
            destination="Lisbon", location_name="Lisbon, Portugal",
            checkin_date=date(2026, 9, 15), checkout_date=date(2026, 9, 20), adults=2,
        )

    def test_request_uses_verified_location_dates_currency_and_guests(self):
        hotels = search_hotels(self.query)
        self.api.assert_called_once_with({
            "engine": "google_hotels", "q": "Hotels in Lisbon, Portugal",
            "check_in_date": "2026-09-15", "check_out_date": "2026-09-20",
            "adults": 2, "currency": "EUR", "hl": "en", "gl": "it",
        }, budget=None)
        self.assertEqual(len(hotels), 1)
        self.assertEqual(hotels[0].destination, "Lisbon")
        self.assertEqual(hotels[0].nights, 5)
        self.assertEqual(hotels[0].total_price_eur, Decimal("249.99"))

    def test_total_is_not_multiplied_by_nights_or_guests(self):
        hotel = search_hotels(self.query)[0]
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))
        self.assertNotEqual(hotel.total_price_eur, hotel.nightly_price_eur * hotel.nights)

    def test_price_filter_uses_explicit_stay_total_locally(self):
        self.api.return_value = {"properties": [raw_hotel(total="250"), raw_hotel(total="250.01")]}
        hotels = search_hotels(self.query.model_copy(update={"max_total_price_eur": 250}))
        self.assertEqual([hotel.total_price_eur for hotel in hotels], [Decimal("250")])
        self.assertNotIn("max_price", self.api.call_args.args[0])

    def test_empty_success_is_not_an_error(self):
        for response in (
            {"properties": []},
            {"search_metadata": {"status": "Success"}, "error": "No results for this query"},
        ):
            with self.subTest(response=response):
                self.api.return_value = response
                self.assertEqual(search_hotels(self.query), [])

    def test_missing_or_malformed_results_are_errors(self):
        for response in ({}, {"properties": None}, {"properties": {}}, {"properties": [None]},
                         {"error": "Service failed"}, {"non_matching_properties": [raw_hotel()]}):
            with self.subTest(response=response):
                self.api.return_value = response
                with self.assertRaises(SerpAPIResponseError):
                    search_hotels(self.query)

    def test_property_details_shape_is_supported(self):
        self.api.return_value = {
            **raw_hotel(),
            "search_information": {"hotels_results_state": "Showing results for property details"},
        }
        self.assertEqual(search_hotels(self.query)[0].name, "Test Stay")

    def test_non_matching_and_sponsored_results_are_not_mixed_in(self):
        self.api.return_value = {
            "properties": [raw_hotel()],
            "non_matching_properties": [raw_hotel(name="Not matching")],
            "ads": [raw_hotel(name="Advertisement")],
        }
        self.assertEqual([hotel.name for hotel in search_hotels(self.query)], ["Test Stay"])

    def test_search_is_bounded_to_one_page(self):
        self.api.return_value["serpapi_pagination"] = {"next_page_token": "next-page"}
        search_hotels(self.query)
        self.api.assert_called_once()

    def test_provider_exceptions_propagate_without_becoming_empty_lists(self):
        for error in (SerpAPIError("Quota"), SerpAPIResponseError("Invalid JSON")):
            with self.subTest(error=error):
                self.api.side_effect = error
                with self.assertRaises(SerpAPIError) as caught:
                    search_hotels(self.query)
                self.assertIs(caught.exception, error)

    def test_missing_total_fails_instead_of_inventing_a_price(self):
        self.api.return_value = {"properties": [raw_hotel(total=None)]}
        with self.assertRaises(SerpAPIResponseError):
            search_hotels(self.query)

    def test_missing_optional_rating_and_nightly_rate_are_not_fabricated(self):
        self.api.return_value = {"properties": [{"name": "No review yet", "total_rate": {"extracted_lowest": 100}}]}
        hotel = search_hotels(self.query)[0]
        self.assertIsNone(hotel.rating)
        self.assertIsNone(hotel.nightly_price_eur)
        self.assertIsNone(hotel.redirect)
        self.assertEqual(hotel.total_price_eur, 100)

    def test_malformed_hotel_fields_have_a_provider_exception(self):
        for fields in (
            {"name": None}, {"name": " "}, {"name": 123}, {"overall_rating": "bad"},
            {"overall_rating": 6}, {"overall_rating": "NaN"}, {"rate_per_night": []},
            {"total_rate": {"extracted_lowest": "unknown"}},
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(SerpAPIResponseError):
                    parse_hotel_offer({**raw_hotel(), **fields}, self.query)

    def test_invalid_query_fails_before_the_network(self):
        for fields in (
            {"destination": " "}, {"location_name": " "}, {"adults": 0},
            {"checkout_date": date(2026, 9, 15)}, {"checkout_date": date(2026, 9, 14)},
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(ValidationError):
                    HotelSearchQuery(**{**self.query.model_dump(), **fields})
        with self.assertRaises(ValueError):
            search_hotels(self.query.model_copy(update={"checkout_date": self.query.checkin_date}))
        self.api.assert_not_called()

    def test_parser_does_not_modify_response(self):
        payload = raw_hotel()
        original = deepcopy(payload)
        parse_hotel_offer(payload, self.query)
        self.assertEqual(payload, original)

    def test_public_website_has_explicit_redirect_kind(self):
        hotel = search_hotels(self.query)[0]
        self.assertEqual(hotel.redirect.kind, "property_website")
        self.assertEqual(hotel.redirect.label, "Open property page")
        self.assertEqual(hotel.model_dump(mode="json")["redirect"]["url"], "https://hotel.example/stay")

    def test_api_links_and_rpc_metadata_are_not_used_as_redirects(self):
        self.api.return_value = {
            "search_metadata": {"google_hotels_url": "https://www.google.com/_/TravelFrontendUi/data/batchexecute"},
            "properties": [{**raw_hotel(), "link": None, "serpapi_property_details_link": "https://serpapi.com/search.json?property_token=test"}],
        }
        offer = search_hotels(self.query)[0]
        self.assertIsNone(offer.redirect)
        self.assertIsNone(offer.search_redirect)

    def test_available_details_and_token_are_preserved_without_extra_requests(self):
        raw = detailed_raw_hotel()
        self.api.return_value = {"properties": [raw]}
        hotel = search_hotels(self.query)[0]
        self.api.assert_called_once()
        self.assertEqual(hotel.property_token, "test-property-token")
        self.assertEqual(hotel.rating, 4.4)
        self.assertEqual(hotel.details.property_type, "hotel")
        self.assertEqual(hotel.details.description, raw["description"])
        self.assertEqual(hotel.details.address, raw["address"])
        self.assertEqual(hotel.details.hotel_class, "4-star hotel")
        self.assertEqual(hotel.details.stars, 4)
        self.assertEqual(hotel.details.reviews, 128)
        self.assertEqual(hotel.details.gps_coordinates.model_dump(), raw["gps_coordinates"])
        self.assertEqual(hotel.details.check_in_time, "3:00 PM")
        self.assertEqual(hotel.details.check_out_time, "11:00 AM")
        self.assertEqual(hotel.details.amenities, ["Free Wi-Fi", "Pool"])
        self.assertEqual(hotel.details.excluded_amenities, ["No beach access"])
        self.assertEqual(hotel.details.essential_info, ["Private room"])
        self.assertEqual(hotel.details.images[0].model_dump(mode="json"), raw["images"][0])
        self.assertEqual(hotel.nights, 5)
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))
        self.assertEqual(hotel.vibe_match_status, "not_verified")
        self.assertEqual(HotelOffer.model_validate_json(hotel.model_dump_json()), hotel)

    def test_details_response_is_enriched_like_search_results(self):
        self.api.return_value = {
            **detailed_raw_hotel(),
            "search_information": {"hotels_results_state": "Showing results for property details"},
        }
        hotel = search_hotels(self.query)[0]
        self.assertEqual(hotel.property_token, "test-property-token")
        self.assertEqual(hotel.details.stars, 4)
        self.api.assert_called_once()

    def test_missing_details_are_unknown_not_fabricated(self):
        raw = {"name": "Beach Hotel", "total_rate": {"extracted_lowest": "100"}}
        hotel = parse_hotel_offer(raw, self.query)
        self.assertIsNone(hotel.property_token)
        self.assertTrue(all(value is None for value in hotel.details.model_dump().values()))
        self.assertEqual(hotel.vibe_match_status, "not_verified")

    def test_hotel_stars_are_not_inferred_from_rating_or_class_text(self):
        for raw in (raw_hotel(rating=5), {**raw_hotel(), "hotel_class": "5-star hotel"}):
            with self.subTest(raw=raw):
                hotel = parse_hotel_offer(raw, self.query)
                self.assertIsNone(hotel.details.stars)
                self.assertEqual(hotel.rating, raw["overall_rating"])

    def test_property_token_is_optional_opaque_text_not_an_api_url(self):
        hotel = parse_hotel_offer({**raw_hotel(), "property_token": " opaque_-token== "}, self.query)
        self.assertEqual(hotel.property_token, "opaque_-token==")
        for value in (None, "", "  ", 123, [], {"token": "test"}):
            with self.subTest(value=value):
                hotel = parse_hotel_offer({**raw_hotel(), "property_token": value}, self.query)
                self.assertIsNone(hotel.property_token)

    def test_malformed_optional_details_do_not_discard_valid_price(self):
        raw = {
            **raw_hotel(), "type": 12, "description": {}, "address": [],
            "hotel_class": False, "extracted_hotel_class": 6, "reviews": -1,
            "gps_coordinates": {"latitude": 91, "longitude": 10},
            "check_in_time": {}, "check_out_time": 11,
            "amenities": "Pool", "excluded_amenities": {}, "essential_info": [None],
            "images": [{"thumbnail": "not a URL"}],
        }
        hotel = parse_hotel_offer(raw, self.query)
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))
        self.assertTrue(all(value is None for value in hotel.details.model_dump().values()))

    def test_optional_integer_fields_do_not_coerce_booleans_or_strings(self):
        for value in (True, "4", 4.5, {}, None):
            with self.subTest(value=value):
                hotel = parse_hotel_offer({**raw_hotel(), "extracted_hotel_class": value, "reviews": value}, self.query)
                self.assertIsNone(hotel.details.stars)
                self.assertIsNone(hotel.details.reviews)
        hotel = parse_hotel_offer({**raw_hotel(), "reviews": 0, "extracted_hotel_class": 1}, self.query)
        self.assertEqual(hotel.details.reviews, 0)
        self.assertEqual(hotel.details.stars, 1)

    def test_invalid_or_partial_coordinates_are_omitted(self):
        for value in (
            None, [], {"latitude": 10}, {"latitude": 10, "longitude": 181},
            {"latitude": True, "longitude": 10}, {"latitude": float("nan"), "longitude": 10},
            {"latitude": 10, "longitude": float("inf")},
        ):
            with self.subTest(value=value):
                hotel = parse_hotel_offer({**raw_hotel(), "gps_coordinates": value}, self.query)
                self.assertIsNone(hotel.details.gps_coordinates)

    def test_optional_lists_keep_valid_items_without_mutating_response(self):
        raw = {
            **detailed_raw_hotel(), "amenities": [" Free Wi-Fi ", None, {}, "", "Pool"],
            "excluded_amenities": [], "essential_info": [False, "Private room"],
        }
        before = deepcopy(raw)
        hotel = parse_hotel_offer(raw, self.query)
        self.assertEqual(hotel.details.amenities, ["Free Wi-Fi", "Pool"])
        self.assertEqual(hotel.details.excluded_amenities, [])
        self.assertEqual(hotel.details.essential_info, ["Private room"])
        self.assertEqual(raw, before)
        hotel.details.amenities.append("Test change")
        self.assertEqual(raw, before)

    def test_missing_empty_and_invalid_detail_lists_remain_distinguishable(self):
        for key in ("amenities", "excluded_amenities", "essential_info", "images"):
            for value, expected in ((None, None), ([], []), ([None, {}], None)):
                with self.subTest(key=key, value=value):
                    hotel = parse_hotel_offer({**raw_hotel(), key: value}, self.query)
                    self.assertEqual(getattr(hotel.details, key), expected)

    def test_photo_urls_are_validated_without_fetching_them(self):
        raw = {**raw_hotel(), "images": [
            {"thumbnail": "https://images.example/good.jpg", "original_image": "javascript:alert(1)"},
            {"original_image": "https://images.example/original.jpg"},
            {"thumbnail": "https://serpapi.com/search.json?api_key=secret"},
            {"original_image": "https://user:password@images.example/photo.jpg"},
            {"original_image": "https://images.example/photo.jpg?API_KEY=secret"},
            {"thumbnail": "https://www.google.com/_/TravelFrontendUi/data/batchexecute"},
            None,
        ]}
        hotel = parse_hotel_offer(raw, self.query)
        self.assertEqual(len(hotel.details.images), 2)
        self.assertIsNone(hotel.details.images[0].original_image)
        self.assertIsNone(hotel.details.images[1].thumbnail)
        self.assertNotIn("secret", hotel.model_dump_json())
        self.api.assert_not_called()

    def test_vibe_is_not_verified_from_descriptions_coordinates_or_provider_flags(self):
        hotel = parse_hotel_offer({
            **detailed_raw_hotel(), "name": "Beachfront Hotel", "description": "Near a beach",
            "amenities": ["Beach access"], "vibe_match_status": "verified",
        }, self.query)
        self.assertEqual(hotel.vibe_match_status, "not_verified")
        with self.assertRaises(ValidationError):
            HotelOffer.model_validate({**hotel.model_dump(), "vibe_match_status": "verified"})

    def test_unknown_private_provider_fields_are_not_exported(self):
        hotel = parse_hotel_offer({
            **detailed_raw_hotel(), "api_key": "secret-key",
            "serpapi_property_details_link": "https://serpapi.com/search.json?api_key=secret-key",
            "search_metadata": {"json_endpoint": "private-search-url"},
            "gps_coordinates": {"latitude": 10, "longitude": 20, "api_key": "nested-secret"},
        }, self.query)
        encoded = hotel.model_dump_json()
        for value in ("secret-key", "nested-secret", "private-search-url", "serpapi_property_details_link"):
            self.assertNotIn(value, encoded)
        self.assertEqual(hotel.property_token, "test-property-token")

    def test_public_google_hotels_link_is_distinct_from_property_link(self):
        self.api.return_value = {
            "search_metadata": {"google_hotels_url": "https://www.google.com/travel/hotels?q=Lisbon"},
            "properties": [detailed_raw_hotel()],
        }
        hotel = search_hotels(self.query)[0]
        self.assertEqual(hotel.search_redirect.kind, "hotel_search")
        self.assertEqual(hotel.search_redirect.label, "Open hotel search")
        self.assertEqual(str(hotel.search_redirect.url), "https://www.google.com/travel/hotels?q=Lisbon")
        self.assertEqual(str(hotel.redirect.url), "https://hotel.example/stay")
        self.api.assert_called_once()

    def test_missing_property_link_does_not_promote_api_link_or_other_rate_link(self):
        self.api.return_value = {
            "search_metadata": {"google_hotels_url": "https://www.google.com/travel/search?q=Lisbon"},
            "properties": [{
                **detailed_raw_hotel(), "link": None,
                "prices": [{"link": "https://booking.example/other-rate", "total_rate": {"extracted_lowest": 1000}}],
            }],
        }
        hotel = search_hotels(self.query)[0]
        self.assertIsNone(hotel.redirect)
        self.assertEqual(hotel.search_redirect.kind, "hotel_search")
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))

    def test_missing_or_malformed_search_metadata_does_not_discard_offer(self):
        for metadata in (None, [], "invalid", {}, {"google_hotels_url": "https://serpapi.com/search"}):
            with self.subTest(metadata=metadata):
                self.api.return_value = {"search_metadata": metadata, "properties": [raw_hotel()]}
                hotel = search_hotels(self.query)[0]
                self.assertIsNone(hotel.search_redirect)
                self.assertEqual(hotel.total_price_eur, Decimal("249.99"))

    def test_enrichment_does_not_change_local_budget_filter(self):
        self.api.return_value = {"properties": [detailed_raw_hotel(total="250"), detailed_raw_hotel(total="250.01")]}
        offers = search_hotels(self.query.model_copy(update={"max_total_price_eur": 250}))
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].total_price_eur, Decimal("250"))
        self.assertEqual(offers[0].details.stars, 4)


class RecordedHotelFixtureTest(unittest.TestCase):
    def test_recorded_hotel_responses(self):
        paths = sorted(RECORDED_FIXTURE_DIRECTORY.glob("hotels_*.json"))
        if not paths:
            self.skipTest("Fixture reale assente: esegui manualmente record_serpapi.py.")
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")):
            for path in paths:
                with self.subTest(fixture=path.name):
                    fixture = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(fixture["source"]["provider"], "serpapi")
                    self.assertEqual(fixture["source"]["engine"], "google_hotels")
                    self.assertEqual(fixture["source"]["currency"], "EUR")
                    query = HotelSearchQuery.model_validate(fixture["query"])
                    raw_hotels = fixture["response"]["properties"]
                    self.assertTrue(raw_hotels, "La fixture deve coprire almeno un'offerta reale.")
                    with patch("providers.hotels.search_serpapi", return_value=fixture["response"]) as api:
                        offers = search_hotels(query)
                    api.assert_called_once_with({
                        "engine": "google_hotels",
                        "q": f"Hotels in {query.location_name or query.destination}",
                        "check_in_date": query.checkin_date.isoformat(),
                        "check_out_date": query.checkout_date.isoformat(),
                        "adults": query.adults,
                        "currency": "EUR", "hl": "en", "gl": "it",
                    }, budget=None)
                    self.assertEqual(len(offers), len(raw_hotels))
                    for offer, raw in zip(offers, raw_hotels):
                        self.assertEqual(offer.name, raw["name"].strip())
                        self.assertEqual(offer.destination, query.destination)
                        self.assertEqual(offer.nights, (query.checkout_date - query.checkin_date).days)
                        self.assertEqual(offer.total_price_eur, Decimal(str(raw["total_rate"]["extracted_lowest"])))
                        rating = raw.get("overall_rating")
                        self.assertEqual(offer.rating, float(rating) if rating is not None else None)
                        nightly = (raw.get("rate_per_night") or {}).get("extracted_lowest")
                        self.assertEqual(offer.nightly_price_eur, Decimal(str(nightly)) if nightly is not None else None)
                        self.assertIsNone(offer.redirect)
                        self.assertIsNone(offer.property_token)
                        self.assertIsNone(offer.details.amenities)
                        self.assertIsNone(offer.details.stars)
                        self.assertEqual(offer.vibe_match_status, "not_verified")


class RedirectTest(unittest.TestCase):
    def test_invalid_or_private_api_urls_are_omitted(self):
        for url in (
            None, 4, {}, "not a URL", "javascript:alert(1)", "file:///tmp/example",
            "https://serpapi.com/search.json", "https://api.serpapi.com/search",
            "https://serpapi.com./search.json", "https://api.serpapi.com./search",
            "https://user:password@hotel.example", "https://hotel.example?api_key=secret",
            "https://www.google.com/_/TravelFrontendUi/data/batchexecute",
        ):
            with self.subTest(url=url):
                self.assertIsNone(parse_redirect(url, kind="property_website"))

    def test_flight_link_requires_a_google_flights_page(self):
        for url in ("https://www.google.com/travel/flights?tfs=test", "https://www.google.it/travel/flights/search?tfs=test"):
            with self.subTest(url=url):
                redirect = parse_redirect(url, kind="search")
                self.assertEqual(redirect.kind, "search")
                self.assertEqual(redirect.label, "Open flight search")
        for url in (
            "https://not-google.example/travel/flights", "https://www.google.com/",
            "https://www.google.com.evil.example/travel/flights", "https://www.google.com/flights/unsupported",
        ):
            with self.subTest(url=url):
                self.assertIsNone(parse_redirect(url, kind="search"))

    def test_hotel_search_link_requires_a_public_google_hotels_page(self):
        for url in (
            "https://www.google.com/travel/hotels?q=Lisbon",
            "https://www.google.it/travel/search?q=Lisbon",
            "https://www.google.co.uk/travel/hotels/entity/test",
        ):
            with self.subTest(url=url):
                redirect = parse_redirect(url, kind="hotel_search")
                self.assertEqual(redirect.kind, "hotel_search")
                self.assertEqual(redirect.label, "Open hotel search")
                self.assertIsNone(parse_redirect(url, kind="search"))
        for url in (
            "https://www.google.com/travel/flights", "https://www.google.com/",
            "https://not-google.example/travel/hotels", "https://www.google.com.evil.example/travel/hotels",
            "https://www.google.com/travel/hotels-internal", "https://www.google.com/_/TravelFrontendUi/data/batchexecute",
            "https://serpapi.com/search.json", "https://www.google.com/travel/hotels?api_key=secret",
        ):
            with self.subTest(url=url):
                self.assertIsNone(parse_redirect(url, kind="hotel_search"))

    def test_property_link_label_does_not_promise_the_official_site_or_quoted_rate(self):
        redirect = parse_redirect("https://travel-provider.example/hotel", kind="property_website")
        self.assertEqual(redirect.label, "Open property page")


if __name__ == "__main__":
    unittest.main()

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
            "adults": 2, "currency": "EUR", "hl": "it", "gl": "it",
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
        self.assertEqual(hotel.redirect.label, "Visita il sito della struttura")
        self.assertEqual(hotel.model_dump(mode="json")["redirect"]["url"], "https://hotel.example/stay")

    def test_api_links_and_rpc_metadata_are_not_used_as_redirects(self):
        self.api.return_value = {
            "search_metadata": {"google_hotels_url": "https://www.google.com/_/TravelFrontendUi/data/batchexecute"},
            "properties": [{**raw_hotel(), "link": None, "serpapi_property_details_link": "https://serpapi.com/search.json?property_token=test"}],
        }
        self.assertIsNone(search_hotels(self.query)[0].redirect)


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
                        "currency": "EUR", "hl": "it", "gl": "it",
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


class RedirectTest(unittest.TestCase):
    def test_invalid_or_private_api_urls_are_omitted(self):
        for url in (
            None, 4, {}, "not a URL", "javascript:alert(1)", "file:///tmp/example",
            "https://serpapi.com/search.json", "https://api.serpapi.com/search",
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
                self.assertEqual(redirect.label, "Apri la ricerca voli")
        for url in ("https://not-google.example/travel/flights", "https://www.google.com/", "https://www.google.com.evil.example/travel/flights"):
            with self.subTest(url=url):
                self.assertIsNone(parse_redirect(url, kind="search"))


if __name__ == "__main__":
    unittest.main()

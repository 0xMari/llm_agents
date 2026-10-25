import contextlib
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from requests.exceptions import Timeout

from execution import RunBudget, SearchLimits
from models import DestinationPreference, FlightOffer, FlightSearchQuery, HotelSearchResult
from providers.flights import (
    FlightLocationError,
    _country_status,
    find_city_options,
    parse_flight_leg,
    resolve_city,
    select_city_candidate,
    search_flights,
    search_id_for,
)
from providers.location_verification import CountryMatch
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError, search


def raw_leg(origin="FCO", destination="LIS", day=date(2026, 9, 15), *, via=(), price=180, token=None):
    airports = [origin, *via, destination]
    segments = []
    for index, (departure, arrival) in enumerate(zip(airports, airports[1:])):
        start = datetime.combine(day, time(9)) + timedelta(hours=3 * index)
        segments.append({
            "departure_airport": {"id": departure, "time": start.isoformat(sep=" ")},
            "arrival_airport": {"id": arrival, "time": (start + timedelta(hours=2)).isoformat(sep=" ")},
            "airline": "Test Air", "flight_number": f"TEST {index + 1}",
        })
    result = {"flights": segments, "total_duration": 120 * len(segments) + 60 * (len(segments) - 1), "price": price}
    if token is not None:
        result["departure_token"] = token
    return result


def make_flight(price=180, *, start=date(2026, 9, 15), end=date(2026, 9, 20), via=(), airline="Test Air"):
    outbound = parse_flight_leg(raw_leg(day=start, via=via))
    inbound = parse_flight_leg(raw_leg("LIS", "FCO", end))
    return FlightOffer(
        origin="Rome", destination="Lisbon", outbound=outbound, inbound=inbound,
        departure_date=start, return_date=end, airline=airline,
        price_eur=price, stops=outbound.stops + inbound.stops,
    )


def fake_city(text, *, country=None, city_id=None, budget=None):
    name = text.split(",")[0]
    codes = {"Rome": ["FCO", "CIA"], "Lisbon": ["LIS"], "Milan": ["MXP", "LIN"]}.get(name, ["NRT"])
    return {
        "type": "city", "name": text,
        "id": city_id or f"/m/test_{name.lower().replace(' ', '_')}",
        "airports": [{"id": code} for code in codes],
    }


def recorded_rome_autocomplete():
    path = Path(__file__).resolve().parent / "fixtures" / "serpapi" / "autocomplete_rome.json"
    return json.loads(path.read_text(encoding="utf-8"))


def supplied_valencia_autocomplete():
    # User-supplied autocomplete fields; city type is restored from the filtered output.
    path = Path(__file__).resolve().parent / "fixtures" / "serpapi" / "autocomplete_valencia.json"
    return json.loads(path.read_text(encoding="utf-8"))


def supplied_naples_city():
    # Fields printed in the user's diagnostic, not a complete HTTP capture.
    return {"name": "Naples", "id": "/g/1233f22n", "type": "city", "description": None}


def fake_flight_search(query, *, max_outbound=3, budget=None):
    if (query.origin, query.destination) != ("Rome", "Lisbon"):
        return []
    prices = {
        (date(2026, 9, 15), date(2026, 9, 20)): (180, 135),
        (date(2026, 10, 6), date(2026, 10, 11)): (155, 115),
    }.get((query.departure_date, query.return_date))
    if prices is None:
        return []
    return [
        make_flight(prices[0], start=query.departure_date, end=query.return_date, airline="Demo Air"),
        make_flight(prices[1], start=query.departure_date, end=query.return_date, via=("MAD",), airline="Budget Wings"),
    ]


class FlightProviderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.api = self.enterContext(patch("providers.flights.search_serpapi"))
        self.query = FlightSearchQuery(
            origin="Rome", destination="Lisbon", origin_id="/m/test_rome", destination_id="/m/test_lisbon",
            departure_date=date(2026, 9, 15), return_date=date(2026, 9, 20), adults=2,
        )
        self.outbound = raw_leg(token="outbound-token")
        self.inbound = raw_leg("LIS", "FCO", date(2026, 9, 20), price="249.99")

    def search_pair(self, outbound=None, inbound=None, query=None):
        self.api.side_effect = [
            {"best_flights": [self.outbound if outbound is None else outbound]},
            {"other_flights": [self.inbound if inbound is None else inbound]},
        ]
        return search_flights(query or self.query, max_outbound=1)

    def test_round_trip_preserves_segments_and_uses_one_total(self):
        offer = self.search_pair()[0]
        self.assertEqual(offer.price_eur, Decimal("249.99"))
        self.assertEqual(offer.outbound.segments[0].departure_airport, "FCO")
        self.assertEqual(offer.inbound.segments[-1].arrival_airport, "FCO")
        first, second = [call.args[0] for call in self.api.call_args_list]
        self.assertEqual(second, {**first, "departure_token": "outbound-token"})
        self.assertEqual(first["departure_id"], self.query.origin_id)
        self.assertEqual(first["arrival_id"], self.query.destination_id)
        self.assertEqual(first["currency"], "EUR")
        self.assertEqual(first["adults"], 2)
        self.assertEqual(first["type"], 1)
        self.assertEqual(first["outbound_date"], "2026-09-15")
        self.assertEqual(first["return_date"], "2026-09-20")
        self.assertNotIn("max_price", first)

    def test_budget_filter_uses_complete_round_trip(self):
        query = self.query.model_copy(update={"max_price_eur": 250})
        self.assertEqual(len(self.search_pair(query=query)), 1)
        self.assertEqual(self.search_pair(inbound={**self.inbound, "price": "250.01"}, query=query), [])

    def test_missing_or_invalid_search_link_does_not_discard_flights(self):
        for metadata in (None, {}, {"google_flights_url": "https://serpapi.com/search.json"}):
            with self.subTest(metadata=metadata):
                self.api.side_effect = [
                    {"search_metadata": metadata, "best_flights": [self.outbound]},
                    {"other_flights": [self.inbound]},
                ]
                offer = search_flights(self.query)[0]
                self.assertEqual(offer.price_eur, Decimal("249.99"))
                self.assertIsNone(offer.redirect)

    def test_empty_results_are_not_provider_errors(self):
        for response in ({"other_flights": []}, {"search_metadata": {"status": "Success"}, "error": "No results"}):
            with self.subTest(response=response):
                self.api.reset_mock()
                self.api.return_value = response
                self.assertEqual(search_flights(self.query), [])
                self.api.assert_called_once()
        self.api.side_effect = [{"best_flights": [self.outbound]}, {"other_flights": []}]
        self.assertEqual(search_flights(self.query), [])

    def test_unexpected_results_do_not_become_empty_inventory(self):
        for response in ({}, {"best_flights": None}, {"other_flights": [None]}):
            with self.subTest(response=response):
                self.api.return_value = response
                with self.assertRaises(SerpAPIResponseError):
                    search_flights(self.query)

    def test_missing_token_is_reported(self):
        outbound = dict(self.outbound)
        outbound.pop("departure_token")
        self.api.return_value = {"best_flights": [outbound]}
        with self.assertRaisesRegex(SerpAPIResponseError, "departure_token"):
            search_flights(self.query)

    def test_missing_or_invalid_price_is_not_zero(self):
        for value in (None, -1, "NaN", "Infinity", "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(SerpAPIResponseError):
                    self.search_pair(inbound={**self.inbound, "price": value})
        inbound = dict(self.inbound)
        inbound.pop("price")
        with self.assertRaises(SerpAPIResponseError):
            self.search_pair(inbound=inbound)

    def test_malformed_segments_are_reported(self):
        for value in ({}, {"flights": []}, {"flights": None}, {"flights": [None]}):
            with self.subTest(value=value):
                with self.assertRaises(SerpAPIResponseError):
                    parse_flight_leg(value)

    def test_changed_departure_dates_are_rejected(self):
        with self.assertRaisesRegex(SerpAPIResponseError, "date"):
            self.search_pair(outbound=raw_leg(day=date(2026, 9, 16), token="wrong-date"))
        with self.assertRaisesRegex(SerpAPIResponseError, "date"):
            self.search_pair(inbound=raw_leg("LIS", "FCO", date(2026, 9, 21)))

    def test_city_scope_allows_different_return_airport(self):
        offer = self.search_pair(inbound=raw_leg("LIS", "CIA", date(2026, 9, 20)))[0]
        self.assertEqual(offer.outbound.segments[0].departure_airport, "FCO")
        self.assertEqual(offer.inbound.segments[-1].arrival_airport, "CIA")

    def test_explicit_airport_constraint_is_checked_on_both_legs(self):
        query = self.query.model_copy(update={"origin_id": "FCO", "destination_id": "LIS"})
        for outbound, inbound in (
            (raw_leg("CIA", "LIS", token="wrong-airport"), self.inbound),
            (self.outbound, raw_leg("LIS", "CIA", date(2026, 9, 20))),
        ):
            with self.subTest(outbound=outbound):
                with self.assertRaisesRegex(SerpAPIResponseError, "airport"):
                    self.search_pair(outbound=outbound, inbound=inbound, query=query)

    def test_outbound_limit_diversifies_airports_and_deduplicates_tokens(self):
        first = raw_leg("MXP", "LIS", token="mxp-1")
        same_route = raw_leg("MXP", "LIS", token="mxp-2")
        other_airport = raw_leg("LIN", "LIS", token="lin-1")
        self.api.side_effect = [
            {"best_flights": [first, same_route], "other_flights": [first, other_airport]},
            {"other_flights": [raw_leg("LIS", "MXP", date(2026, 9, 20))]},
            {"other_flights": [raw_leg("LIS", "LIN", date(2026, 9, 20))]},
        ]
        query = self.query.model_copy(update={"origin": "Milan", "origin_id": "/m/test_milan"})
        offers = search_flights(query, max_outbound=2)
        self.assertEqual([offer.outbound.segments[0].departure_airport for offer in offers], ["MXP", "LIN"])
        self.assertEqual(self.api.call_count, 3)
        self.assertEqual([call.args[0]["departure_token"] for call in self.api.call_args_list[1:]], ["mxp-1", "lin-1"])

    def test_all_returns_stay_paired_with_the_selected_outbound(self):
        self.api.side_effect = [
            {"other_flights": [self.outbound]},
            {"best_flights": [self.inbound], "other_flights": [raw_leg("LIS", "FCO", date(2026, 9, 20), via=("MAD",), price=200)]},
        ]
        offers = search_flights(self.query)
        self.assertEqual(len(offers), 2)
        self.assertEqual(offers[0].outbound, offers[1].outbound)
        self.assertEqual([offer.stops for offer in offers], [0, 1])

    def test_errors_are_not_swallowed(self):
        self.api.side_effect = [{"best_flights": [self.outbound]}, SerpAPIError("Quota")]
        with self.assertRaises(SerpAPIError):
            search_flights(self.query)

    def test_invalid_limits_do_not_call_api(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    search_flights(self.query, max_outbound=limit)
        self.api.assert_not_called()

    def test_queries_require_ids_and_valid_dates(self):
        fields = self.query.model_dump()
        for changes in ({"origin_id": "Rome"}, {"origin_id": ""}, {"return_date": self.query.departure_date}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValidationError):
                    FlightSearchQuery(**{**fields, **changes})
        fields.pop("origin_id")
        with self.assertRaises(ValidationError):
            FlightSearchQuery(**fields)

    def test_offers_require_both_legs_and_consistent_summaries(self):
        fields = make_flight().model_dump()
        for key in ("outbound", "inbound"):
            with self.subTest(key=key):
                with self.assertRaises(ValidationError):
                    FlightOffer(**{name: value for name, value in fields.items() if name != key})
        with self.assertRaises(ValidationError):
            FlightOffer(**{**fields, "stops": 5})

    def test_city_lookup_and_airport_override(self):
        city = fake_city("Milan, Italy")
        self.api.return_value = {"suggestions": [city]}
        self.assertEqual(find_city_options("Milan, Italy"), [city])
        self.assertTrue(self.api.call_args.args[0]["exclude_regions"])
        self.assertEqual(search_id_for(city), city["id"])
        self.assertEqual(search_id_for(city, "lin"), "LIN")
        with self.assertRaises(FlightLocationError):
            search_id_for(city, "FCO")

    def test_ambiguous_or_unmatched_cities_need_clarification(self):
        cities = [fake_city("Rome, Italy"), {**fake_city("Rome, Georgia"), "id": "/m/test_other_rome"}]
        self.api.return_value = {"suggestions": cities}
        with self.assertRaises(FlightLocationError) as caught:
            resolve_city("Rome")
        self.assertEqual(len(caught.exception.options), 2)
        self.assertEqual(resolve_city("Rome, Italy")["id"], cities[0]["id"])
        self.assertEqual(resolve_city("Rome", city_id=cities[1]["id"]), cities[1])
        with self.assertRaises(FlightLocationError):
            resolve_city("Rome", city_id="/m/not_in_response")
        self.api.return_value = {"suggestions": [fake_city("Lisbon")]}
        with self.assertRaises(FlightLocationError):
            resolve_city("Rome")

    def test_country_description_resolves_bare_city_name_without_extra_calls(self):
        city = {**fake_city("Florence", city_id="/m/031y2"), "description": "City in Italy"}
        self.api.return_value = {"suggestions": [city]}
        for text in ("Florence", "Florence, Italy", " Florence, ITALY "):
            with self.subTest(text=text):
                self.api.reset_mock()
                self.assertEqual(resolve_city(text, country=" Italy "), city)
                self.api.assert_called_once()
                self.assertEqual(self.api.call_args.args[0]["q"], "Florence, Italy")
        self.assertEqual(city["name"], "Florence")
        self.assertEqual(city["description"], "City in Italy")

    def test_explicit_english_country_descriptions_are_supported(self):
        for description in ("City in Italy", "Town in Italy", "Capital of Italy", "Municipality in Italy", " CITY IN ITALY. "):
            with self.subTest(description=description):
                city = {**fake_city("Florence"), "description": description}
                self.assertEqual(select_city_candidate([city], "Florence", country="Italy"), city)
        self.api.assert_not_called()

    def test_country_evidence_distinguishes_match_mismatch_and_unknown(self):
        for name, description, expected in (
            ("Naples, Italy", None, CountryMatch.MATCH),
            ("Naples, USA", "City in Italy", CountryMatch.MISMATCH),
            ("Naples", " City in ITALY. ", CountryMatch.MATCH),
            ("Naples", "City in USA", CountryMatch.MISMATCH),
            ("Naples", "City in Campania", CountryMatch.MISMATCH),
            ("Naples", "City in Italy and France", CountryMatch.MISMATCH),
            ("Naples", None, CountryMatch.UNKNOWN),
            ("Naples", "", CountryMatch.UNKNOWN),
            ("Naples", {}, CountryMatch.UNKNOWN),
            ("Naples", "City in ", CountryMatch.UNKNOWN),
            ("Naples", "Historic city", CountryMatch.UNKNOWN),
        ):
            with self.subTest(name=name, description=description):
                city = {**supplied_naples_city(), "name": name, "description": description}
                self.assertIs(_country_status(city, "Italy"), expected)
        self.api.assert_not_called()

    def test_supplied_naples_stays_unresolved_without_country_evidence(self):
        city = supplied_naples_city()
        self.api.return_value = {"suggestions": [city]}
        with self.assertRaises(FlightLocationError) as caught:
            resolve_city("Naples, Italy", country="Italy")
        self.assertIs(caught.exception.country_status, CountryMatch.UNKNOWN)
        self.assertEqual(caught.exception.options, ["Naples (/g/1233f22n)"])
        self.api.assert_called_once()
        self.assertEqual(self.api.call_args.args[0]["q"], "Naples, Italy")

    def test_verifier_can_confirm_only_a_unique_unknown_candidate(self):
        city = supplied_naples_city()
        self.api.return_value = {"suggestions": [city]}
        verifier = Mock()
        verifier.verify.return_value = CountryMatch.MATCH
        budget = RunBudget(SearchLimits())
        resolved = resolve_city(" Naples, Italy ", country=" Italy ", verifier=verifier, budget=budget)
        self.assertIs(resolved, city)
        verifier.verify.assert_called_once_with(city, "Italy", budget=budget)
        self.assertIsNot(verifier.verify.call_args.args[0], city)
        self.api.assert_called_once()

    def test_verifier_cannot_mutate_the_selected_candidate(self):
        city = {**supplied_naples_city(), "airports": [{"id": "NAP"}]}
        original = deepcopy(city)
        self.api.return_value = {"suggestions": [city]}
        verifier = Mock()

        def verify(candidate, expected_country, *, budget):
            candidate["id"] = "/m/other_city"
            candidate["airports"][0]["id"] = "FCO"
            return CountryMatch.MATCH

        verifier.verify.side_effect = verify
        self.assertEqual(resolve_city("Naples", country="Italy", verifier=verifier), original)
        self.assertEqual(city, original)

    def test_inconclusive_or_negative_verification_does_not_resolve_the_city(self):
        self.api.return_value = {"suggestions": [supplied_naples_city()]}
        for status in (CountryMatch.UNKNOWN, CountryMatch.MISMATCH):
            with self.subTest(status=status):
                verifier = Mock()
                verifier.verify.return_value = status
                with self.assertRaises(FlightLocationError) as caught:
                    resolve_city("Naples", country="Italy", verifier=verifier)
                self.assertIs(caught.exception.country_status, status)
                self.assertEqual(caught.exception.options, ["Naples (/g/1233f22n)"])
                verifier.verify.assert_called_once()

    def test_verifier_is_not_used_for_confirmed_or_explicitly_selected_cities(self):
        for city, kwargs in (
            ({**supplied_naples_city(), "name": "Naples, Italy"}, {"country": "Italy"}),
            ({**supplied_naples_city(), "description": "City in Italy"}, {"country": "Italy"}),
            (supplied_naples_city(), {"country": "Italy", "city_id": "/g/1233f22n"}),
            (supplied_naples_city(), {}),
        ):
            with self.subTest(city=city, kwargs=kwargs):
                self.api.return_value = {"suggestions": [city]}
                verifier = Mock()
                self.assertEqual(resolve_city("Naples", verifier=verifier, **kwargs), city)
                verifier.verify.assert_not_called()

    def test_country_conflicts_cannot_be_overridden_by_verifier_or_explicit_id(self):
        for city in (
            {**supplied_naples_city(), "name": "Naples, USA"},
            {**supplied_naples_city(), "description": "City in USA"},
        ):
            for city_id in (None, city["id"]):
                with self.subTest(city=city, city_id=city_id):
                    self.api.return_value = {"suggestions": [city]}
                    verifier = Mock()
                    verifier.verify.return_value = CountryMatch.MATCH
                    with self.assertRaises(FlightLocationError) as caught:
                        resolve_city("Naples", country="Italy", city_id=city_id, verifier=verifier)
                    self.assertIs(caught.exception.country_status, CountryMatch.MISMATCH)
                    verifier.verify.assert_not_called()

    def test_ambiguous_candidates_never_trigger_verification(self):
        city = supplied_naples_city()
        for other in (
            {**city, "id": "/m/test_naples"},
            {**city, "id": "/m/test_naples", "description": "City in Italy"},
        ):
            with self.subTest(other=other):
                self.api.return_value = {"suggestions": [city, other]}
                verifier = Mock()
                verifier.verify.return_value = CountryMatch.MATCH
                with self.assertRaises(FlightLocationError) as caught:
                    resolve_city("Naples", country="Italy", verifier=verifier)
                self.assertIsNone(caught.exception.country_status)
                self.assertEqual(len(caught.exception.options), 2)
                verifier.verify.assert_not_called()

    def test_wrong_name_missing_city_and_unknown_explicit_id_do_not_trigger_verification(self):
        for cities, city_id in (
            ([], None),
            ([fake_city("Rome, Italy")], None),
            ([supplied_naples_city()], "/m/not_returned"),
        ):
            with self.subTest(cities=cities, city_id=city_id):
                self.api.return_value = {"suggestions": cities}
                verifier = Mock()
                with self.assertRaises(FlightLocationError):
                    resolve_city("Naples", country="Italy", city_id=city_id, verifier=verifier)
                verifier.verify.assert_not_called()

    def test_wrong_country_candidates_do_not_prevent_verification_of_one_unknown_city(self):
        city = supplied_naples_city()
        self.api.return_value = {"suggestions": [
            {**city, "name": "Naples, USA", "id": "/m/test_us_naples"}, city,
        ]}
        verifier = Mock()
        verifier.verify.return_value = CountryMatch.MATCH
        self.assertIs(resolve_city("Naples", country="Italy", verifier=verifier), city)
        verifier.verify.assert_called_once_with(city, "Italy", budget=None)

    def test_invalid_verifier_results_do_not_authorize_a_city(self):
        self.api.return_value = {"suggestions": [supplied_naples_city()]}
        for value in ("match", True, None, {"status": "match"}):
            with self.subTest(value=value):
                verifier = Mock()
                verifier.verify.return_value = value
                with self.assertRaisesRegex(TypeError, "must return CountryMatch"):
                    resolve_city("Naples", country="Italy", verifier=verifier)

    def test_verifier_errors_propagate_without_acceptance_or_retry(self):
        self.api.return_value = {"suggestions": [supplied_naples_city()]}
        for error in (SerpAPIError("Timeout", code="TIMEOUT"), RuntimeError("Verifier bug")):
            with self.subTest(error=error):
                verifier = Mock()
                verifier.verify.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    resolve_city("Naples", country="Italy", verifier=verifier)
                self.assertIs(caught.exception, error)
                verifier.verify.assert_called_once()

    def test_country_is_not_inferred_from_query_or_regional_description(self):
        for description in (None, "", {}, "City in Tuscany", "City in Italy and France", "City in USA"):
            with self.subTest(description=description):
                city = {**fake_city("Florence"), "description": description}
                self.api.return_value = {"suggestions": [city]}
                with self.assertRaises(FlightLocationError) as caught:
                    resolve_city("Florence", country="Italy")
                self.assertEqual(caught.exception.options, [f"Florence ({city['id']})"])

    def test_country_filter_preserves_ambiguity_and_explicit_id_selection(self):
        first = {**fake_city("Florence", city_id="/m/031y2"), "description": "City in Italy"}
        second = {**first, "id": "/m/test_other_florence"}
        self.api.return_value = {"suggestions": [first, second]}
        with self.assertRaises(FlightLocationError) as caught:
            resolve_city("Florence", country="Italy")
        self.assertEqual(len(caught.exception.options), 2)
        self.assertEqual(resolve_city("Florence", country="Italy", city_id=first["id"]), first)
        with self.assertRaises(FlightLocationError):
            resolve_city("Florence", country="Italy", city_id="/m/not_in_response")

    def test_confirmed_city_id_can_select_city_without_country_evidence(self):
        city = fake_city("Florence", city_id="/m/031y2")
        self.api.return_value = {"suggestions": [city]}
        self.assertEqual(resolve_city("Florence", country="Italy", city_id=city["id"]), city)

    def test_country_filter_rejects_wrong_country_and_same_country_different_name(self):
        cities = [
            {**fake_city("Florence, USA"), "description": "City in Italy"},
            fake_city("Rome, Italy"),
        ]
        with self.assertRaises(FlightLocationError):
            select_city_candidate(cities, "Florence", country="Italy")

    def test_country_filter_accepts_qualified_name_and_normalized_accents(self):
        self.api.return_value = supplied_valencia_autocomplete()
        city = resolve_city("Valencia, Spain", country="Spain")
        self.assertEqual(city["id"], "/m/0h3tv")
        self.assertEqual(self.api.call_args.args[0]["q"], "Valencia, Spain")

    def test_empty_country_is_rejected_before_autocomplete(self):
        with self.assertRaises(FlightLocationError):
            resolve_city("Florence", country=" ")
        self.api.assert_not_called()

    def test_recorded_autocomplete_ignores_suggestions_without_city_type(self):
        response = recorded_rome_autocomplete()
        self.api.return_value = response
        cities = find_city_options("Rome")
        self.assertEqual([city["id"] for city in cities], ["/m/06c62", "/m/0rwgm", "/m/097xtf"])
        self.assertEqual(response, recorded_rome_autocomplete())

    def test_recorded_rome_requires_explicit_selection(self):
        self.api.return_value = recorded_rome_autocomplete()
        with self.assertRaises(FlightLocationError) as caught:
            resolve_city("Rome")
        self.assertIn("Rome (/m/06c62)", caught.exception.options)
        self.assertIn("Rome, Georgia, USA (/m/0rwgm)", caught.exception.options)
        city = resolve_city("Rome", city_id="/m/06c62")
        self.assertEqual(search_id_for(city), "/m/06c62")
        self.assertEqual(search_id_for(city, "FCO"), "FCO")
        with self.assertRaises(FlightLocationError):
            resolve_city("Rome", city_id="/m/018046")

    def test_valencia_accepts_transport_ids_and_resolves_unaccented_city_name(self):
        response = supplied_valencia_autocomplete()
        original = json.dumps(response, ensure_ascii=False)
        self.api.return_value = response
        self.assertEqual(len(find_city_options("Valencia, Spain")), 5)
        city = resolve_city("Valencia, Spain")
        self.assertEqual(city["name"], "Val\u00e8ncia, Spain")
        self.assertEqual(search_id_for(city), "/m/0h3tv")
        self.assertEqual(search_id_for(city, "vlc"), "VLC")
        self.assertIn("4GS", [place["id"] for place in city["airports"]])
        self.assertEqual(json.dumps(response, ensure_ascii=False), original)

    def test_transport_id_is_not_accepted_as_explicit_airport_or_flight_airport(self):
        city = supplied_valencia_autocomplete()["suggestions"][0]
        with self.assertRaises(FlightLocationError):
            search_id_for(city, "4GS")
        leg = raw_leg(destination="4GS")
        with self.assertRaises(SerpAPIResponseError):
            parse_flight_leg(leg)
        with self.assertRaises(ValidationError):
            FlightSearchQuery(**{**self.query.model_dump(), "destination_id": "4GS"})

    def test_accent_normalization_does_not_pick_a_city_when_multiple_ids_match(self):
        first = supplied_valencia_autocomplete()["suggestions"][0]
        second = {**first, "name": "Valencia, Spain", "id": "/m/test_other_valencia"}
        self.api.return_value = {"suggestions": [first, second]}
        with self.assertRaises(FlightLocationError) as caught:
            resolve_city("Valencia, Spain")
        self.assertEqual(len(caught.exception.options), 2)
        self.assertEqual(resolve_city("Valencia, Spain", city_id="/m/0h3tv"), first)

    def test_malformed_transport_locations_are_still_invalid(self):
        for locations in (None, {}, [None], [{"id": ""}], [{"id": "bad"}], [{"id": "ABCD"}], [{"id": 123}]):
            with self.subTest(locations=locations):
                self.api.return_value = {"suggestions": [{**fake_city("Rome"), "airports": locations}]}
                with self.assertRaises(SerpAPIResponseError) as caught:
                    find_city_options("Rome")
                self.assertEqual(caught.exception.kind, "invalid_city_locations")

    def test_autocomplete_without_cities_needs_clarification(self):
        for kind in (None, "region", "airport", "unknown"):
            with self.subTest(kind=kind):
                self.api.return_value = {"suggestions": [{**fake_city("Rome"), "type": kind}]}
                self.assertEqual(find_city_options("Rome"), [])
                with self.assertRaises(FlightLocationError):
                    resolve_city("Rome", city_id="/m/test_rome")

    def test_explicit_city_validation_is_not_relaxed(self):
        for changes in ({"name": None}, {"name": " "}, {"id": None}, {"id": "bad"}, {"airports": [{"id": "bad"}]}):
            with self.subTest(changes=changes):
                self.api.return_value = {"suggestions": [{**fake_city("Rome"), **changes}]}
                with self.assertRaises(SerpAPIResponseError):
                    find_city_options("Rome")

    def test_missing_city_and_malformed_autocomplete_are_distinct(self):
        self.api.return_value = {"suggestions": []}
        with self.assertRaises(FlightLocationError):
            resolve_city("Unknown")
        for response in ({}, {"suggestions": [None]}, {"suggestions": [{"type": "city", "name": "Rome", "id": "bad"}]}):
            with self.subTest(response=response):
                self.api.return_value = response
                with self.assertRaises(SerpAPIResponseError):
                    find_city_options("Rome")

    def test_open_destination_rejects_fixed_airport_constraints(self):
        with self.assertRaises(ValidationError):
            DestinationPreference(mode="open", airport_code="LIS")

    def test_demo_import_does_not_call_api(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_module("try_serpapi", run_name="import_test")
        self.assertEqual(output.getvalue(), "")
        self.api.assert_not_called()

    def test_demo_uses_provider_and_only_searches_hotels_on_request(self):
        import try_serpapi

        with patch("try_serpapi.resolve_city", side_effect=fake_city), patch(
            "try_serpapi.search_flights", return_value=[make_flight()]
        ) as flights, patch("try_serpapi.search_hotels", return_value=HotelSearchResult(offers=[], received_count=0, parsed_count=0)) as hotels:
            with contextlib.redirect_stdout(io.StringIO()):
                try_serpapi.main(["--origin", "Milan, Italy", "--origin-airport", "LIN"])
            query = flights.call_args.args[0]
            self.assertEqual(query.origin_id, "LIN")
            self.assertEqual(query.destination_id, "/m/test_lisbon")
            self.assertEqual(flights.call_args.kwargs["max_outbound"], 1)
            hotels.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()):
                try_serpapi.main(["--hotels", "--adults", "2"])
            hotels.assert_called_once()
            self.assertEqual(hotels.call_args.args[0].adults, 2)
            self.assertEqual(hotels.call_args.args[0].checkin_date, date(2026, 9, 15))


class SerpAPIClientTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}))
        self.get = self.enterContext(patch("providers.serpapi_client.requests.get"))
        self.get.return_value.status_code = 200
        self.get.return_value.json.return_value = {"search_metadata": {"status": "Success"}}

    def test_client_passes_key_without_mutating_params(self):
        params = {"engine": "google_flights"}
        self.assertEqual(search(params)["search_metadata"]["status"], "Success")
        self.get.assert_called_once_with("https://serpapi.com/search.json", params={**params, "api_key": "test-key"}, timeout=(10, 60))
        self.assertNotIn("api_key", params)

    def test_missing_key_does_not_call_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SerpAPIError):
                search({})
        self.get.assert_not_called()

    def test_http_errors_are_explicit(self):
        for status, code in ((401, "AUTHENTICATION"), (403, "AUTHENTICATION"), (429, "QUOTA_OR_RATE_LIMIT"), (500, "HTTP_ERROR")):
            with self.subTest(status=status):
                self.get.return_value.status_code = status
                with self.assertRaisesRegex(SerpAPIError, str(status)) as caught:
                    search({})
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.status_code, status)

    def test_network_errors_do_not_expose_request_url(self):
        self.get.side_effect = Timeout("https://serpapi.com/search?api_key=test-key")
        with self.assertRaises(SerpAPIError) as caught:
            search({})
        self.assertNotIn("test-key", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertEqual(caught.exception.code, "TIMEOUT")

    def test_invalid_json_and_metadata_are_reported(self):
        self.get.return_value.json.side_effect = ValueError("invalid JSON")
        with self.assertRaises(SerpAPIResponseError):
            search({})
        self.get.return_value.json.side_effect = None
        for response in ([], {}, {"search_metadata": None}):
            with self.subTest(response=response):
                self.get.return_value.json.return_value = response
                with self.assertRaises(SerpAPIResponseError):
                    search({})

    def test_unsuccessful_search_is_not_empty_success(self):
        for status in ("Error", "Processing", "Queued", None):
            with self.subTest(status=status):
                self.get.return_value.json.return_value = {"search_metadata": {"status": status}}
                with self.assertRaises(SerpAPIError):
                    search({})


if __name__ == "__main__":
    unittest.main()

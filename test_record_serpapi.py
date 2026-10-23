import contextlib
from copy import deepcopy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import providers.flights
import providers.hotels
import record_serpapi
from providers.serpapi_client import SerpAPIError
from test_flights import fake_city, raw_leg, recorded_rome_autocomplete
from test_hotels import raw_hotel
import test_hotels


class FixtureRecorderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.folder = Path(self.enterContext(TemporaryDirectory()))
        self.output = self.folder / "hotels_real.json"
        self.api = self.enterContext(patch("record_serpapi.live_search"))
        self.params = {
            "engine": "google_hotels", "q": "Hotels in Lisbon, Portugal",
            "check_in_date": "2027-05-10", "check_out_date": "2027-05-15",
            "adults": 2, "currency": "EUR", "api_key": "secret-test-key",
        }
        self.response = {"properties": [raw_hotel()]}
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_allowlist_removes_sensitive_fields_without_changing_prices(self):
        self.response.update(search_metadata={"id": "private-search", "json_endpoint": "private-url"})
        item = self.response["properties"][0]
        item.update(property_token="private-token", api_key="secret-test-key", description="private-description")
        item["total_rate"].update(extracted_before_taxes_fees="200.01", api_key="nested-secret")
        original = deepcopy(self.response)
        fixture = record_serpapi.build_hotel_fixture(self.params, self.response)
        encoded = json.dumps(fixture)
        for value in ("secret-test-key", "nested-secret", "private-search", "private-token", "private-url", "private-description", "https://"):
            self.assertNotIn(value, encoded)
        clean = fixture["response"]["properties"][0]
        self.assertEqual(clean["total_rate"], {"extracted_lowest": "249.99", "extracted_before_taxes_fees": "200.01"})
        self.assertEqual(clean["rate_per_night"]["extracted_lowest"], "50.01")
        self.assertEqual(fixture["query"]["adults"], 2)
        self.assertEqual(fixture["source"]["currency"], "EUR")
        self.assertEqual(self.response, original)

    def test_missing_optional_fields_stay_missing(self):
        self.response["properties"] = [{"name": "Minimal Stay", "total_rate": {"extracted_lowest": 100.25}}]
        fixture = record_serpapi.build_hotel_fixture(self.params, self.response)
        self.assertEqual(fixture["response"], self.response)

    def test_empty_or_unexpected_results_are_not_recorded_as_success(self):
        for response in ({}, {"properties": []}, {"properties": None}, {"properties": [None]}, {"name": "Property details"}):
            with self.subTest(response=response):
                with self.assertRaises(record_serpapi.FixtureRecordingError):
                    record_serpapi.build_hotel_fixture(self.params, response)
        for changes in ({"currency": "USD"}, {"engine": "google_flights"}, {"q": "unexpected"}):
            with self.subTest(changes=changes):
                with self.assertRaises(record_serpapi.FixtureRecordingError):
                    record_serpapi.build_hotel_fixture({**self.params, **changes}, self.response)

    def test_unexpected_nested_values_cannot_bypass_allowlist(self):
        for changes in (
            {"name": {"api_key": "secret"}},
            {"overall_rating": ["secret"]},
            {"total_rate": {"extracted_lowest": {"api_key": "secret"}}},
            {"rate_per_night": ["secret"]},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(record_serpapi.FixtureRecordingError):
                    record_serpapi.build_hotel_fixture(self.params, {"properties": [{**raw_hotel(), **changes}]})

    def test_records_once_and_returns_original_payload_to_provider(self):
        self.api.return_value = self.response
        flight_search = providers.flights.search_serpapi
        hotel_search = providers.hotels.search_serpapi

        def demo(args):
            self.assertEqual(args[-3:], ["--max-outbound", "1", "--hotels"])
            self.assertIs(providers.hotels.search_serpapi(self.params), self.response)

        with patch("record_serpapi.run_demo", side_effect=demo):
            self.assertEqual(record_serpapi.record_hotels(self.output, []), 1)
        self.assertEqual(json.loads(self.output.read_text())["response"]["properties"][0]["total_rate"]["extracted_lowest"], "249.99")
        self.assertIs(providers.flights.search_serpapi, flight_search)
        self.assertIs(providers.hotels.search_serpapi, hotel_search)

    def test_existing_file_stops_before_network(self):
        self.output.write_text("existing fixture", encoding="utf-8")
        with self.assertRaisesRegex(record_serpapi.FixtureRecordingError, "already exists"):
            record_serpapi.record_hotels(self.output, [])
        self.api.assert_not_called()
        self.assertEqual(self.output.read_text(), "existing fixture")

    def test_call_limit_stops_before_sixth_request(self):
        def demo(args):
            for _ in range(6):
                providers.flights.search_serpapi({"engine": "google_flights"})

        with patch("record_serpapi.run_demo", side_effect=demo):
            with self.assertRaisesRegex(record_serpapi.FixtureRecordingError, "limit"):
                record_serpapi.record_hotels(self.output, [])
        self.assertEqual(self.api.call_count, 5)
        self.assertFalse(self.output.exists())

    def test_no_hotel_search_is_not_reported_as_success(self):
        with patch("record_serpapi.run_demo"):
            with self.assertRaisesRegex(record_serpapi.FixtureRecordingError, "No fixture"):
                record_serpapi.record_hotels(self.output, [])
        self.assertFalse(self.output.exists())

    def test_provider_failure_does_not_create_fixture(self):
        self.api.side_effect = SerpAPIError("Quota")
        with patch("record_serpapi.run_demo", side_effect=lambda args: providers.hotels.search_serpapi(self.params)):
            with self.assertRaises(SerpAPIError):
                record_serpapi.record_hotels(self.output, [])
        self.assertFalse(self.output.exists())

    def test_nonfinite_numbers_do_not_leave_partial_file(self):
        self.response["properties"][0]["total_rate"]["extracted_lowest"] = float("nan")
        self.api.return_value = self.response
        with patch("record_serpapi.run_demo", side_effect=lambda args: providers.hotels.search_serpapi(self.params)):
            with self.assertRaisesRegex(record_serpapi.FixtureRecordingError, "JSON"):
                record_serpapi.record_hotels(self.output, [])
        self.assertFalse(self.output.exists())

    def test_cli_and_actual_provider_pipeline_record_and_replay_offline(self):
        start = date.today() + timedelta(days=45)
        end = start + timedelta(days=5)
        self.api.side_effect = [
            {"suggestions": [fake_city("Rome, Italy")]},
            {"suggestions": [fake_city("Lisbon, Portugal")]},
            {"best_flights": [raw_leg(day=start, token="outbound-token")]},
            {"other_flights": [raw_leg("LIS", "FCO", end, price="180.25")]},
            self.response,
        ]
        record_serpapi.main([
            "--departure-date", start.isoformat(), "--return-date", end.isoformat(),
            "--adults", "2", "--output", str(self.output),
        ])
        self.assertEqual(self.api.call_count, 5)
        fixture = json.loads(self.output.read_text())
        self.assertEqual(fixture["query"]["checkin_date"], start.isoformat())
        self.assertEqual(fixture["query"]["checkout_date"], end.isoformat())
        self.assertEqual(fixture["query"]["adults"], 2)
        # Synthetic payloads exist only in this temporary directory, never as real fixtures.
        with patch("test_hotels.RECORDED_FIXTURE_DIRECTORY", self.folder):
            result = unittest.TestResult()
            test_hotels.RecordedHotelFixtureTest("test_recorded_hotel_responses").run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(result.skipped, [])
        self.assertEqual(self.api.call_count, 5)

    def test_import_and_help_have_no_side_effects(self):
        with patch("record_serpapi.record_hotels") as record:
            with self.assertRaises(SystemExit) as caught:
                record_serpapi.main(["--help"])
            self.assertEqual(caught.exception.code, 0)
            record.assert_not_called()
        runpy.run_module("record_serpapi", run_name="import_test")
        self.api.assert_not_called()

    def test_cli_passes_verified_city_ids_through_actual_pipeline(self):
        start = date.today() + timedelta(days=45)
        end = start + timedelta(days=5)
        self.api.side_effect = [
            recorded_rome_autocomplete(),
            {"suggestions": [fake_city("Lisbon, Portugal"), fake_city("Lisbon, USA", city_id="/m/test_other_lisbon")]},
            {"best_flights": [raw_leg(day=start, token="outbound-token")]},
            {"other_flights": [raw_leg("LIS", "FCO", end, price="180.25")]},
            self.response,
        ]
        record_serpapi.main([
            "--origin", "Rome", "--origin-city-id", "/m/06c62",
            "--destination", "Lisbon", "--destination-city-id", "/m/test_lisbon",
            "--output", str(self.output),
        ])
        self.assertEqual(self.api.call_count, 5)
        flight_params = self.api.call_args_list[2].args[0]
        self.assertEqual(flight_params["departure_id"], "/m/06c62")
        self.assertEqual(flight_params["arrival_id"], "/m/test_lisbon")
        self.assertTrue(self.output.exists())


if __name__ == "__main__":
    unittest.main()

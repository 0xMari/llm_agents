from datetime import date, timedelta
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from requests.exceptions import Timeout

from execution import RunBudget, SearchLimitExceeded, SearchLimits
from llm.openrouter import OpenRouterError, call_openrouter_json
from models import DestinationPreference, EventType, PlanningError, TimePreference, TripRequest, TripSearchResult
from planning import build_proposal
from providers.flights import FlightLocationError, find_city_options, search_flights
from providers.serpapi_client import SerpAPIError, search
from test_flights import fake_city, raw_leg, recorded_rome_autocomplete
from test_hotels import raw_hotel


def http_response(payload, status=200):
    response = Mock(status_code=status, text="HTTP test error")
    response.json.return_value = payload
    return response


class RunBudgetTest(unittest.TestCase):
    def test_limits_reject_invalid_types_and_ranges(self):
        for name in SearchLimits.model_fields:
            for value in (True, "1", 1.5, -1, None):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValidationError):
                        SearchLimits(**{name: value})
        for name in ("max_destinations", "max_periods_per_destination", "max_outbound"):
            with self.subTest(name=name):
                with self.assertRaises(ValidationError):
                    SearchLimits(**{name: 0})
        with self.assertRaises(ValidationError):
            SearchLimits(max_destinations=4)

    def test_limits_are_immutable(self):
        limits = SearchLimits()
        with self.assertRaises(ValidationError):
            limits.max_llm_calls = 100

    def test_zero_disables_calls_without_incrementing(self):
        budget = RunBudget(SearchLimits(max_serpapi_calls=0, max_llm_calls=0))
        for service in ("serpapi", "llm"):
            with self.subTest(service=service):
                with self.assertRaises(SearchLimitExceeded) as caught:
                    budget.claim(service)
                self.assertEqual((caught.exception.service, caught.exception.used, caught.exception.limit), (service, 0, 0))
        self.assertEqual(budget.used, {"serpapi": 0, "llm": 0})

    def test_independent_services_allow_exactly_the_configured_attempts(self):
        budget = RunBudget(SearchLimits(max_serpapi_calls=2, max_llm_calls=1))
        budget.claim("serpapi")
        budget.claim("serpapi")
        budget.claim("llm")
        for service, limit in (("serpapi", 2), ("llm", 1)):
            with self.subTest(service=service):
                with self.assertRaises(SearchLimitExceeded) as caught:
                    budget.claim(service)
                self.assertEqual(caught.exception.used, limit)
                self.assertEqual(caught.exception.limit, limit)
                self.assertEqual(budget.used[service], limit)

    def test_each_run_gets_its_own_counters(self):
        limits = SearchLimits()
        first, second = RunBudget(limits), RunBudget(limits)
        first.claim("serpapi")
        first.claim("llm")
        self.assertEqual(second.used, {"serpapi": 0, "llm": 0})
        self.assertIsNot(first.used, second.used)


class HTTPBudgetTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict("os.environ", {
            "SERPAPI_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test-model",
        }))
        self.get = self.enterContext(patch("providers.serpapi_client.requests.get"))
        self.post = self.enterContext(patch("llm.openrouter.requests.post"))
        self.get.return_value = http_response({"search_metadata": {"status": "Success"}})
        self.post.return_value = http_response({"choices": [{"message": {"content": "{}"}}]})

    def test_serpapi_blocks_second_attempt_before_http(self):
        budget = RunBudget(SearchLimits(max_serpapi_calls=1))
        search({"engine": "google_flights"}, budget=budget)
        with self.assertRaises(SearchLimitExceeded):
            search({"engine": "google_hotels"}, budget=budget)
        self.get.assert_called_once()
        self.assertEqual(budget.used, {"serpapi": 1, "llm": 0})

    def test_openrouter_blocks_second_attempt_before_http(self):
        budget = RunBudget(SearchLimits(max_llm_calls=1))
        call_openrouter_json([], {}, budget=budget)
        with self.assertRaises(SearchLimitExceeded):
            call_openrouter_json([], {}, budget=budget)
        self.post.assert_called_once()
        self.assertEqual(budget.used, {"serpapi": 0, "llm": 1})

    def test_missing_configuration_is_not_counted(self):
        for key, call, exception in (
            ("SERPAPI_API_KEY", lambda budget: search({}, budget=budget), SerpAPIError),
            ("OPENROUTER_API_KEY", lambda budget: call_openrouter_json([], {}, budget=budget), OpenRouterError),
            ("OPENROUTER_MODEL", lambda budget: call_openrouter_json([], {}, budget=budget), OpenRouterError),
        ):
            with self.subTest(key=key), patch.dict("os.environ", {key: ""}):
                budget = RunBudget(SearchLimits(max_serpapi_calls=0, max_llm_calls=0))
                with self.assertRaises(exception):
                    call(budget)
                self.assertEqual(budget.used, {"serpapi": 0, "llm": 0})
        self.get.assert_not_called()
        self.post.assert_not_called()

    def test_failed_http_attempts_are_not_refunded(self):
        for service, transport, call, exception in (
            ("serpapi", self.get, lambda budget: search({}, budget=budget), SerpAPIError),
            ("llm", self.post, lambda budget: call_openrouter_json([], {}, budget=budget), OpenRouterError),
        ):
            for failure in ("timeout", "http", "json"):
                with self.subTest(service=service, failure=failure):
                    transport.reset_mock(side_effect=True)
                    if failure == "timeout":
                        transport.side_effect = Timeout("test timeout")
                    else:
                        transport.return_value = http_response({}, status=429 if failure == "http" else 200)
                        if failure == "json":
                            transport.return_value.json.side_effect = ValueError("invalid JSON")
                    budget = RunBudget(SearchLimits(max_serpapi_calls=1, max_llm_calls=1))
                    with self.assertRaises(exception):
                        call(budget)
                    with self.assertRaises(SearchLimitExceeded):
                        call(budget)
                    self.assertEqual(budget.used[service], 1)
                    transport.assert_called_once()

    def test_invalid_local_input_does_not_consume_calls(self):
        budget = RunBudget(SearchLimits())
        with self.assertRaises(FlightLocationError):
            find_city_options(" ", budget=budget)
        with self.assertRaises(ValueError):
            search_flights(None, max_outbound=0, budget=budget)
        self.assertEqual(budget.used, {"serpapi": 0, "llm": 0})
        self.get.assert_not_called()


class PlannerLimitsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict("os.environ", {
            "SERPAPI_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test-model",
        }))
        self.get = self.enterContext(patch("providers.serpapi_client.requests.get", side_effect=self.serpapi_response))
        self.post = self.enterContext(patch("llm.openrouter.requests.post", side_effect=self.llm_response))
        self.start = date(2027, 5, 10)
        self.end = self.start + timedelta(days=5)
        self.periods = [self.period(self.start, 80)]
        self.destinations = [self.destination("Lisbon", "Portugal", 90)]
        self.no_flights = False
        self.request = TripRequest(
            origin="Rome", origin_city_id="/m/06c62", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon", city_id="/m/04llb"),
            time_pref=TimePreference(mode="exact_dates", start_date=self.start, end_date=self.end),
        )

    def period(self, start, score):
        return {"start": start.isoformat(), "end": (start + timedelta(days=5)).isoformat(), "score": score, "reason": "Test period"}

    def destination(self, name, country, score):
        return {"name": name, "country": country, "match_score": score, "reason": "Test destination", "themes_matched": [], "caution": None}

    def flexible_request(self):
        return self.request.model_copy(update={"time_pref": TimePreference(mode="flexible_window", min_days=6, max_days=6)})

    def open_request(self):
        return self.request.model_copy(update={"destination_pref": DestinationPreference(mode="open")})

    def llm_response(self, url, **kwargs):
        name = kwargs["json"]["response_format"]["json_schema"]["name"]
        if name == "destination_advice":
            payload = {"destinations": self.destinations}
        else:
            self.assertEqual(name, "travel_window_advice")
            payload = {"selected_periods": self.periods, "warnings": [], "alternatives": [], "rationale": "Test advice"}
        return http_response({"choices": [{"message": {"content": json.dumps(payload)}}]})

    def serpapi_response(self, url, *, params, **kwargs):
        engine = params["engine"]
        if engine == "google_flights_autocomplete":
            if params["q"] == "Rome":
                payload = recorded_rome_autocomplete()
            else:
                payload = {"suggestions": [fake_city("Lisbon, Portugal", city_id="/m/04llb")]}
        elif engine == "google_flights":
            if "departure_token" in params:
                payload = {"other_flights": [
                    raw_leg("LIS", "FCO", date.fromisoformat(params["return_date"]), price=price)
                    for price in (180, 190)
                ]}
            else:
                payload = {"best_flights": [] if self.no_flights else [
                    raw_leg(origin=origin, day=date.fromisoformat(params["outbound_date"]), token=f"token-{index}")
                    for index, origin in enumerate(("FCO", "CIA", "FCO"))
                ]}
        else:
            self.assertEqual(engine, "google_hotels")
            payload = {"properties": [raw_hotel()]}
        return http_response({"search_metadata": {"status": "Success"}, **payload})

    def test_fixed_pipeline_counts_all_http_calls_and_reuses_hotel_cache(self):
        original_request = self.request.model_dump()
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(len(result.proposals), 2)
        self.assertEqual(self.get.call_count, 5)
        self.post.assert_called_once()
        self.assertEqual([call.kwargs["params"]["engine"] for call in self.get.call_args_list], [
            "google_flights_autocomplete", "google_flights_autocomplete",
            "google_flights", "google_flights", "google_hotels",
        ])
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 5)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
        self.assertEqual(self.request.model_dump(), original_request)

    def test_serpapi_limit_stops_before_each_stage_without_partial_success(self):
        for limit in range(5):
            with self.subTest(limit=limit):
                self.get.reset_mock()
                self.post.reset_mock()
                result = build_proposal(self.request, limits=SearchLimits(max_serpapi_calls=limit))
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
                self.assertEqual(self.get.call_count, limit)
                self.post.assert_called_once()
                details = result.trace.events[-1].details
                self.assertEqual((details["service"], details["used"], details["limit"]), ("serpapi", limit, limit))
                self.assertEqual(details["serpapi_calls"], limit)
                self.assertIn(EventType.REQUEST_ROUTED, [event.event_type for event in result.trace.events])
                self.assertNotIn(EventType.PROPOSALS_SELECTED, [event.event_type for event in result.trace.events])

    def test_zero_llm_limit_stops_before_any_http_call(self):
        result = build_proposal(self.request, limits=SearchLimits(max_llm_calls=0))
        self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
        self.assertEqual(result.trace.events[-1].details["service"], "llm")
        self.get.assert_not_called()
        self.post.assert_not_called()

    def test_open_destination_shares_llm_limit_with_travel_window(self):
        result = build_proposal(self.open_request(), limits=SearchLimits(max_llm_calls=1))
        self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
        self.assertEqual(result.trace.events[-1].details["service"], "llm")
        self.post.assert_called_once()
        self.get.assert_not_called()
        self.assertEqual(self.post.call_args.kwargs["json"]["response_format"]["json_schema"]["name"], "destination_advice")

    def test_destinations_are_ranked_and_capped_before_window_calls(self):
        self.destinations = [self.destination("Kyoto", "Japan", 50), self.destination("Lisbon", "Portugal", 90)]
        result = build_proposal(self.open_request(), limits=SearchLimits(max_llm_calls=2))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(self.get.call_count, 5)
        self.assertTrue(all(proposal.destination.name == "Lisbon" for proposal in result.proposals))
        scope = next(event for event in result.trace.events if event.event_type == EventType.SEARCH_SCOPE_REDUCED)
        self.assertEqual(scope.details["scope"], "destinations")
        self.assertEqual((scope.details["available_count"], scope.details["selected_count"]), (2, 1))
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 2)

    def test_periods_are_ranked_and_deduplicated_before_the_cap(self):
        second, third = self.start + timedelta(days=7), self.start + timedelta(days=14)
        self.periods = [self.period(second, 70), self.period(self.start, 10), self.period(self.start, 90), self.period(third, 40)]
        result = build_proposal(self.flexible_request(), limits=SearchLimits(max_periods_per_destination=2, max_serpapi_calls=8))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(self.get.call_count, 8)
        self.post.assert_called_once()
        queries = [call.kwargs["params"] for call in self.get.call_args_list]
        outbounds = [query for query in queries if query["engine"] == "google_flights" and "departure_token" not in query]
        self.assertEqual([query["outbound_date"] for query in outbounds], [self.start.isoformat(), second.isoformat()])
        self.assertEqual(sum(query["engine"] == "google_flights_autocomplete" for query in queries), 2)
        scope = next(event for event in result.trace.events if event.event_type == EventType.SEARCH_SCOPE_REDUCED)
        self.assertEqual((scope.details["available_count"], scope.details["unique_count"], scope.details["selected_count"]), (4, 3, 2))
        self.assertEqual(result.proposals[0].period.score, 90)

    def test_outbound_limit_applies_before_fetching_returns(self):
        result = build_proposal(self.request, limits=SearchLimits(max_outbound=2, max_serpapi_calls=6))
        self.assertIsInstance(result, TripSearchResult)
        queries = [call.kwargs["params"] for call in self.get.call_args_list]
        self.assertEqual([query["departure_token"] for query in queries if "departure_token" in query], ["token-0", "token-1"])
        self.assertEqual(sum(query["engine"] == "google_hotels" for query in queries), 1)
        self.assertEqual(self.get.call_count, 6)

    def test_limit_after_accepted_combinations_is_not_reported_as_success(self):
        self.periods.append(self.period(self.start + timedelta(days=7), 70))
        result = build_proposal(self.flexible_request(), limits=SearchLimits(max_periods_per_destination=2))
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
        self.assertEqual(self.get.call_count, 5)
        types = [event.event_type for event in result.trace.events]
        self.assertIn(EventType.COMBINATION_ACCEPTED, types)
        self.assertNotIn(EventType.PROPOSALS_SELECTED, types)

    def test_repeated_runs_start_from_zero_with_the_same_limits_object(self):
        limits = SearchLimits()
        for _ in range(2):
            result = build_proposal(self.request, limits=limits)
            self.assertIsInstance(result, TripSearchResult)
            self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 5)
            self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
        self.assertEqual(self.get.call_count, 10)
        self.assertEqual(self.post.call_count, 2)

    def test_empty_results_at_exact_limit_are_not_limit_exhaustion(self):
        self.no_flights = True
        result = build_proposal(self.request, limits=SearchLimits(max_serpapi_calls=3))
        self.assertEqual(result.code, "NO_FLIGHTS_FOUND")
        self.assertEqual(self.get.call_count, 3)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 3)

    def test_provider_failure_keeps_its_error_code_and_attempt_count(self):
        self.get.side_effect = Timeout("test timeout")
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertEqual(result.code, "FLIGHT_PROVIDER_ERROR")
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 1)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
        self.get.assert_called_once()

    def test_llm_failure_keeps_its_error_code_and_attempt_count(self):
        self.post.side_effect = Timeout("test timeout")
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertEqual(result.code, "LLM_PROVIDER_ERROR")
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 0)
        self.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()

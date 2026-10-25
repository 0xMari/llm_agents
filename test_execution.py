from datetime import date, timedelta
from decimal import Decimal
import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from requests.exceptions import Timeout

from execution import RunBudget, SearchLimitExceeded, SearchLimits
from llm.openrouter import OpenRouterError, call_openrouter_json
from models import DecisionEvent, DestinationPreference, EventType, PlanningError, ProposalComment, ReasonCode, TimePreference, TripRequest, TripSearchResult
from planning import build_proposal
from providers.flights import FlightLocationError, find_city_options, search_flights
from providers.serpapi_client import SerpAPIError, search
from providers.location_verification import CountryMatch
from test_flights import fake_city, raw_leg, recorded_rome_autocomplete, supplied_naples_city, supplied_valencia_autocomplete
from test_hotels import raw_hotel
from test_proposal_summary import fake_summary_response


def http_response(payload, status=200):
    response = Mock(status_code=status, text="HTTP test error")
    response.json.return_value = payload
    return response


class RunBudgetTest(unittest.TestCase):
    def test_trace_string_lists_round_trip_without_decimal_tuple_conversion(self):
        for values in ([], ["Lisbon"], ["Lisbon", "Valencia"], ["Lisbon", "Valencia", "Prague"], ["a", "b", "c", "d"]):
            with self.subTest(values=values):
                details = {"destinations": values, "price": Decimal("232.50"), "count": 3, "score": 1.5, "flag": True, "missing": None}
                event = DecisionEvent(event_type=EventType.DESTINATIONS_RESOLVED, reason_code=ReasonCode.OPEN_DESTINATION_SUGGESTED, details=details)
                comment = ProposalComment(reason_code=ReasonCode.OPEN_DESTINATION_SUGGESTED, comment="Test", details=details)
                self.assertEqual(DecisionEvent.model_validate_json(event.model_dump_json()), event)
                self.assertEqual(ProposalComment.model_validate_json(comment.model_dump_json()), comment)

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

    def test_shared_retry_reserves_one_call_and_cannot_exceed_either_limit(self):
        first = RunBudget(SearchLimits(max_llm_calls=3))
        first.claim("llm")
        self.assertTrue(first.claim_llm_retry())
        self.assertEqual(first.used["llm"], 2)
        self.assertEqual(first.llm_retries, 1)
        self.assertFalse(first.claim_llm_retry())
        self.assertEqual(first.used["llm"], 2)
        exhausted = RunBudget(SearchLimits(max_llm_calls=1))
        exhausted.claim("llm")
        self.assertFalse(exhausted.claim_llm_retry())
        self.assertEqual(exhausted.used["llm"], 1)
        self.assertEqual(exhausted.llm_retries, 0)
        self.assertEqual(RunBudget(first.limits).llm_retries, 0)


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
        self.sleep = self.enterContext(patch("llm.openrouter.time.sleep"))
        self.start = date(2027, 5, 10)
        self.end = self.start + timedelta(days=5)
        self.periods = [self.period(self.start, 80)]
        self.destinations = [self.destination("Lisbon", "Portugal", 90)]
        self.no_flights = False
        self.summary_content = None
        self.summary_transport_error = None
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
        elif name == "proposal_summaries":
            if self.summary_transport_error is not None:
                raise self.summary_transport_error
            content = self.summary_content
            if content is None:
                content = fake_summary_response(kwargs["json"]["messages"])
            return http_response({"choices": [{"message": {"content": content}}]})
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
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].selection_profiles, ["cheapest", "comfortable"])
        self.assertEqual(result.proposals[0].comparison.proposal_id, "p1")
        self.assertEqual(result.proposals[0].comparison.reference_id, "p1")
        self.assertEqual(result.proposals[0].comparison.extra_price_eur, 0)
        self.assertEqual(result.proposals[0].comparison.stops_delta, 0)
        self.assertEqual(result.proposals[0].comparison.duration_delta_minutes, 0)
        self.assertEqual(result.proposals[0].comparison.early_departures_delta, 0)
        self.assertEqual(result.trace.events[-1].details["selected_count"], 1)
        self.assertNotIn("selected_group_count", result.trace.events[-1].details)
        self.assertNotIn("selected_variant_count", result.trace.events[-1].details)
        self.assertEqual(self.get.call_count, 5)
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(result.summary_status, "generated")
        self.assertEqual(result.proposals[0].summary.proposal_id, "p1")
        self.assertEqual([call.kwargs["params"]["engine"] for call in self.get.call_args_list], [
            "google_flights_autocomplete", "google_flights_autocomplete",
            "google_flights", "google_flights", "google_hotels",
        ])
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 5)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 2)
        self.assertEqual(self.request.model_dump(), original_request)

    def mock_naples_destination(self):
        self.destinations = [self.destination("Naples", "Italy", 90)]

        def response(url, *, params, **kwargs):
            if params["engine"] == "google_flights_autocomplete" and params["q"] == "Naples, Italy":
                payload = {"suggestions": [supplied_naples_city()]}
            elif params["engine"] == "google_flights":
                if "departure_token" in params:
                    payload = {"other_flights": [
                        raw_leg("NAP", "FCO", date.fromisoformat(params["return_date"]), price=180),
                    ]}
                else:
                    payload = {"best_flights": [
                        raw_leg("FCO", "NAP", date.fromisoformat(params["outbound_date"]), token="naples-token"),
                    ]}
            else:
                return self.serpapi_response(url, params=params, **kwargs)
            return http_response({"search_metadata": {"status": "Success"}, **payload})

        self.get.side_effect = response

    def test_naples_without_verifier_reports_unknown_country_without_searching_flights(self):
        self.mock_naples_destination()
        result = build_proposal(self.open_request(), limits=SearchLimits(max_llm_calls=3))
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.FLIGHT_LOCATION_UNRESOLVED)
        failure = next(e for e in result.trace.events if e.event_type == EventType.DESTINATION_SEARCH_FAILED)
        self.assertEqual(failure.details["country_status"], "unknown")
        self.assertEqual(failure.details["location_options"], ["Naples (/g/1233f22n)"])
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 2)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 2)
        self.assertEqual(PlanningError.model_validate_json(result.model_dump_json()), result)

    def test_planner_uses_verifier_and_keeps_the_original_city_id(self):
        self.mock_naples_destination()
        verifier = Mock()
        verifier.verify.return_value = CountryMatch.MATCH
        result = build_proposal(
            self.open_request(), limits=SearchLimits(max_llm_calls=3), country_verifier=verifier,
        )
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(result.search_status, "complete")
        self.assertEqual(result.summary_status, "generated")
        self.assertEqual(result.proposals[0].destination_city_id, "/g/1233f22n")
        self.assertEqual(result.proposals[0].flight.outbound.segments[-1].arrival_airport, "NAP")
        verifier.verify.assert_called_once()
        self.assertEqual(verifier.verify.call_args.args, (supplied_naples_city(), "Italy"))
        self.assertIsInstance(verifier.verify.call_args.kwargs["budget"], RunBudget)
        self.assertEqual(self.get.call_count, 5)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 5)

    def test_counted_verification_is_cached_across_periods_and_uses_the_run_budget(self):
        self.mock_naples_destination()
        self.periods = [self.period(self.start, 90), self.period(self.start + timedelta(days=10), 80)]
        request = self.open_request().model_copy(update={
            "time_pref": TimePreference(mode="flexible_window", min_days=6, max_days=6),
        })
        verifier = Mock()

        def verify(city, country, *, budget):
            # Simulate one provider attempt; no concrete verification API is configured.
            budget.claim("serpapi")
            return CountryMatch.MATCH

        verifier.verify.side_effect = verify
        limits = SearchLimits(max_periods_per_destination=2, max_serpapi_calls=9, max_llm_calls=3)
        result = build_proposal(request, limits=limits, country_verifier=verifier)
        self.assertIsInstance(result, TripSearchResult)
        verifier.verify.assert_called_once()
        self.assertEqual(self.get.call_count, 8)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 9)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 3)

    def test_verification_cannot_bypass_the_global_call_limit(self):
        self.mock_naples_destination()
        verifier = Mock()

        def verify(city, country, *, budget):
            budget.claim("serpapi")
            return CountryMatch.MATCH

        verifier.verify.side_effect = verify
        result = build_proposal(
            self.open_request(), limits=SearchLimits(max_serpapi_calls=2, max_llm_calls=3),
            country_verifier=verifier,
        )
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.SEARCH_LIMIT_REACHED)
        self.assertEqual(result.trace.events[-1].details["service"], "serpapi")
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 2)
        self.assertEqual(self.get.call_count, 2)
        verifier.verify.assert_called_once()

    def test_failed_verification_does_not_hide_other_open_destinations(self):
        for status in (CountryMatch.UNKNOWN, CountryMatch.MISMATCH):
            with self.subTest(status=status):
                self.mock_three_cities()
                self.get.reset_mock()
                self.post.reset_mock()
                original = self.get.side_effect

                def response(url, *, params, **kwargs):
                    if params.get("q") == "Vienna, Austria":
                        city = fake_city("Vienna", city_id="/m/test_vienna")
                        return http_response({"search_metadata": {"status": "Success"}, "suggestions": [city]})
                    return original(url, params=params, **kwargs)

                self.get.side_effect = response
                verifier = Mock()
                verifier.verify.return_value = status
                result = build_proposal(
                    self.open_request(), limits=SearchLimits(
                        max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
                    ), country_verifier=verifier,
                )
                self.assertIsInstance(result, TripSearchResult)
                self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Paris"])
                self.assertEqual(result.search_status, "partial")
                self.assertEqual(result.search_issues[0].destination, "Vienna")
                failure = next(e for e in result.trace.events if e.event_type == EventType.DESTINATION_SEARCH_FAILED)
                self.assertEqual(failure.details["country_status"], status.value)
                self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 10)
                verifier.verify.assert_called_once()
                self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_verification_provider_error_remains_a_provider_error_not_unknown_inventory(self):
        self.mock_naples_destination()
        verifier = Mock()
        verifier.verify.side_effect = SerpAPIError("PRIVATE-ERROR", code="TIMEOUT")
        result = build_proposal(
            self.open_request(), limits=SearchLimits(max_llm_calls=3), country_verifier=verifier,
        )
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.FLIGHT_PROVIDER_ERROR)
        self.assertEqual(result.trace.events[-1].details["provider_error"], "TIMEOUT")
        self.assertNotIn("PRIVATE-ERROR", result.model_dump_json())
        self.assertEqual(self.get.call_count, 2)
        verifier.verify.assert_called_once()

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
        result = build_proposal(self.open_request(), limits=SearchLimits(max_llm_calls=3))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(self.post.call_count, 3)
        self.assertEqual(result.summary_status, "generated")
        self.assertEqual(self.get.call_count, 5)
        self.assertTrue(all(
            proposal.destination.name == "Lisbon"
            for proposal in result.proposals
        ))
        scope = next(event for event in result.trace.events if event.event_type == EventType.SEARCH_SCOPE_REDUCED)
        self.assertEqual(scope.details["scope"], "destinations")
        self.assertEqual((scope.details["available_count"], scope.details["selected_count"]), (2, 1))
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 3)

    def mock_three_cities(self, *, use_valencia=False):
        cities = {
            "Paris, France": ("/m/test_paris", "ORY", 100, (), 60),
            "Lisbon, Portugal": ("/m/04llb", "LIS", 150, ("MAD",), 100),
            "Vienna, Austria": ("/m/test_vienna", "VIE", 200, ("MAD", "BCN"), 90),
        }
        if use_valencia:
            del cities["Vienna, Austria"]
            cities["Valencia, Spain"] = ("/m/0h3tv", "VLC", 200, ("MAD", "BCN"), 90)
        self.destinations = [
            self.destination(*name.split(", "), data[4]) for name, data in cities.items()
        ]
        original = self.serpapi_response

        def response(url, *, params, **kwargs):
            engine = params["engine"]
            if engine == "google_flights_autocomplete":
                if params["q"] == "Rome":
                    return original(url, params=params, **kwargs)
                if use_valencia and params["q"] == "Valencia, Spain":
                    return http_response({"search_metadata": {"status": "Success"}, **supplied_valencia_autocomplete()})
                city_id, airport, *_ = cities[params["q"]]
                city = fake_city(params["q"], city_id=city_id)
                city["airports"] = [{"id": airport}]
                payload = {"suggestions": [city]}
            elif engine == "google_flights":
                city_id, airport, price, via, _ = next(data for data in cities.values() if data[0] == params["arrival_id"])
                if "departure_token" in params:
                    payload = {"other_flights": [
                        raw_leg(airport, "FCO", date.fromisoformat(params["return_date"]), price=price + extra)
                        for extra in (0, 10)
                    ]}
                else:
                    payload = {"best_flights": [raw_leg(
                        "FCO", airport, date.fromisoformat(params["outbound_date"]), via=via, token=f"token-{city_id}",
                    )]}
            else:
                self.assertEqual(engine, "google_hotels")
                payload = {"properties": [raw_hotel(name=params["q"], total="300")]}
            return http_response({"search_metadata": {"status": "Success"}, **payload})

        self.get.side_effect = response

    def test_three_city_pipeline_keeps_vibe_order_and_does_not_add_provider_calls(self):
        self.mock_three_cities()
        limits = SearchLimits(max_destinations=3, max_serpapi_calls=13, max_llm_calls=5)
        result = build_proposal(self.open_request(), limits=limits)
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Vienna", "Paris"])
        self.assertEqual([p.destination_city_id for p in result.proposals], ["/m/04llb", "/m/test_vienna", "/m/test_paris"])
        self.assertEqual([p.total_price_eur for p in result.proposals], [450, 500, 400])
        self.assertEqual([p.selection_profiles for p in result.proposals], [[], [], ["cheapest", "comfortable"]])
        self.assertEqual([p.comparison.reference_id for p in result.proposals], ["p3"] * 3)
        self.assertEqual([p.summary.proposal_id for p in result.proposals], ["p1", "p2", "p3"])
        self.assertEqual(result.summary_status, "generated")
        self.assertEqual(self.get.call_count, 13)
        self.assertEqual(self.post.call_count, 5)
        for call in self.get.call_args_list:
            params = call.kwargs["params"]
            self.assertEqual(params["hl"], "en")
            self.assertEqual(params["gl"], "it")
            if params["engine"] != "google_flights_autocomplete":
                self.assertEqual(params["currency"], "EUR")
        self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(
            [call.kwargs["json"]["response_format"]["json_schema"]["name"] for call in self.post.call_args_list],
            ["destination_advice", "travel_window_advice", "travel_window_advice", "travel_window_advice", "proposal_summaries"],
        )
        summary_payload = json.loads(self.post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertEqual(summary_payload["request"]["destination_mode"], "open")
        self.assertEqual([p["destination"]["name"] for p in summary_payload["proposals"]], ["Lisbon", "Vienna", "Paris"])
        event = result.trace.events[-1]
        self.assertEqual(event.reason_code, ReasonCode.SELECTED_BY_DESTINATION_MATCH)
        self.assertEqual(event.details["destination_mode"], "open")
        self.assertEqual(event.details["destination_city_ids"], [p.destination_city_id for p in result.proposals])
        self.assertEqual(event.details["serpapi_calls"], 13)
        self.assertEqual(event.details["llm_calls"], 5)

    def test_three_city_pipeline_can_recover_a_502_in_any_llm_stage(self):
        for stage in ("destination_advice", "travel_window_advice", "proposal_summaries"):
            with self.subTest(stage=stage):
                self.mock_three_cities()
                self.get.reset_mock()
                self.post.reset_mock()
                self.sleep.reset_mock()
                failed = False

                def response(url, **kwargs):
                    nonlocal failed
                    name = kwargs["json"]["response_format"]["json_schema"]["name"]
                    if name == stage and not failed:
                        failed = True
                        return http_response({"error": {"code": 502, "message": "PRIVATE-KEY"}}, 502)
                    return self.llm_response(url, **kwargs)

                self.post.side_effect = response
                result = build_proposal(self.open_request(), limits=SearchLimits(
                    max_destinations=3, max_serpapi_calls=13, max_llm_calls=6,
                ))
                self.assertIsInstance(result, TripSearchResult)
                self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Vienna", "Paris"])
                self.assertEqual(result.summary_status, "generated")
                self.assertEqual(self.get.call_count, 13)
                self.assertEqual(self.post.call_count, 6)
                self.sleep.assert_called_once_with(1.0)
                details = result.trace.events[-1].details
                self.assertEqual(details["llm_calls"], 6)
                self.assertEqual(details["llm_retries"], 1)
                self.assertEqual(details["llm_retry_error_kind"], "provider_unavailable")
                self.assertEqual(details["llm_retry_http_status"], 502)
                self.assertEqual(details["llm_retry_api_code"], 502)
                self.assertNotIn("PRIVATE-KEY", result.model_dump_json())
                self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_retry_used_by_destination_agent_is_not_available_to_travel_window(self):
        failed = False

        def response(url, **kwargs):
            nonlocal failed
            name = kwargs["json"]["response_format"]["json_schema"]["name"]
            if name == "destination_advice" and not failed:
                failed = True
                return http_response({"error": {"code": 502}}, 502)
            if name == "travel_window_advice":
                return http_response({"error": {"code": 503}}, 503)
            return self.llm_response(url, **kwargs)

        self.post.side_effect = response
        result = build_proposal(self.open_request(), limits=SearchLimits(max_llm_calls=10))
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.LLM_PROVIDER_ERROR)
        details = result.trace.events[-1].details
        self.assertEqual(details["llm_calls"], 3)
        self.assertEqual(details["llm_retries"], 1)
        self.assertEqual(details["llm_http_status"], 503)
        self.assertEqual(details["llm_retry_http_status"], 502)
        self.assertEqual(self.post.call_count, 3)
        self.get.assert_not_called()
        self.sleep.assert_called_once()

    def test_retry_does_not_increase_default_limits_or_discard_offers_if_summary_has_no_room(self):
        failed = False

        def response(url, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                return http_response({"error": {"code": 502}}, 502)
            return self.llm_response(url, **kwargs)

        self.post.side_effect = response
        result = build_proposal(self.request, limits=SearchLimits())
        self.assert_summary_unavailable(result, ReasonCode.SEARCH_LIMIT_REACHED)
        self.assertEqual(result.trace.events[-1].details["max_llm_calls"], 2)
        self.assertEqual(result.trace.events[-1].details["llm_retries"], 1)
        self.sleep.assert_called_once()

    def test_city_diversity_never_bypasses_budget_or_forces_three_results(self):
        self.mock_three_cities()
        request = self.open_request().model_copy(update={"budget_eur": 450, "budget_flexibility_pct": 0})
        result = build_proposal(request, limits=SearchLimits(max_destinations=3, max_serpapi_calls=13, max_llm_calls=5))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Paris"])
        self.assertTrue(all(p.total_price_eur <= 450 for p in result.proposals))
        self.assertEqual(result.trace.events[-1].details["selected_count"], 2)

    def test_supplied_valencia_response_completes_three_city_search(self):
        self.mock_three_cities(use_valencia=True)
        result = build_proposal(self.open_request(), limits=SearchLimits(
            max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
        ))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Valencia", "Paris"])
        self.assertEqual(result.proposals[1].destination_city_id, "/m/0h3tv")
        self.assertEqual(result.proposals[1].flight.outbound.segments[-1].arrival_airport, "VLC")
        self.assertEqual(result.search_status, "complete")
        self.assertEqual(result.search_issues, [])
        self.assertEqual(result.summary_status, "generated")
        self.assertEqual(self.get.call_count, 13)
        self.assertEqual(self.post.call_count, 5)

    def fail_middle_city(self, stage):
        self.mock_three_cities()
        original = self.get.side_effect

        def response(url, *, params, **kwargs):
            failure = (
                stage == "location_resolution" and params.get("q") == "Vienna, Austria"
                or stage == "flight_search" and params.get("arrival_id") == "/m/test_vienna"
                or stage == "hotel_search" and params.get("q") == "Hotels in Vienna, Austria"
            )
            if failure:
                return http_response({"search_metadata": {"status": "Success"}})
            return original(url, params=params, **kwargs)

        self.get.side_effect = response

    def test_mixed_hotel_results_preserve_the_destination_and_report_issues_once_per_stay(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                self.get.reset_mock()
                self.post.reset_mock()

                def response(url, *, params, **kwargs):
                    if params["engine"] == "google_hotels":
                        return http_response({"search_metadata": {"status": "Success"}, "properties": [
                            raw_hotel(name="Valid stay"),
                            raw_hotel(name="PRIVATE-HOTEL", total="PRIVATE-PRICE"),
                        ]})
                    return self.serpapi_response(url, params=params, **kwargs)

                self.get.side_effect = response
                request = self.request if mode == "fixed" else self.open_request()
                result = build_proposal(request, limits=SearchLimits(max_llm_calls=3))
                self.assertIsInstance(result, TripSearchResult)
                self.assertTrue(all(p.hotel.name == "Valid stay" for p in result.proposals))
                self.assertEqual(result.search_status, "partial")
                self.assertEqual(result.summary_status, "generated")
                self.assertEqual(len(result.search_issues), 1)
                self.assertEqual(result.search_issues[0].scope, "offer")
                self.assertEqual(result.search_issues[0].code, ReasonCode.HOTEL_OFFERS_DISCARDED)
                self.assertEqual(sum(e.event_type == EventType.HOTEL_OFFER_ISSUE for e in result.trace.events), 1)
                self.assertEqual(sum(e.event_type == EventType.HOTEL_SEARCH_PARTIAL for e in result.trace.events), 1)
                self.assertNotIn(EventType.DESTINATION_SEARCH_FAILED, [e.event_type for e in result.trace.events])
                issue = next(e for e in result.trace.events if e.event_type == EventType.HOTEL_OFFER_ISSUE)
                self.assertEqual(issue.details["offer_index"], 1)
                self.assertEqual(issue.details["field"], "total_rate")
                self.assertEqual(issue.details["issue_code"], "invalid_value")
                self.assertNotIn("PRIVATE", result.model_dump_json())
                self.assertEqual(self.get.call_count, 5)
                self.assertEqual(self.post.call_count, 2 if mode == "fixed" else 3)
                self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_ignored_optional_hotel_fields_do_not_mark_the_search_as_incomplete(self):
        def response(url, *, params, **kwargs):
            if params["engine"] == "google_hotels":
                return http_response({"search_metadata": {"status": "Success"}, "properties": [
                    {**raw_hotel(rating="PRIVATE"), "rate_per_night": []},
                ]})
            return self.serpapi_response(url, params=params, **kwargs)

        self.get.side_effect = response
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(result.search_status, "complete")
        self.assertEqual(result.search_issues, [])
        self.assertTrue(all(p.hotel.rating is None and p.hotel.nightly_price_eur is None for p in result.proposals))
        issues = [e for e in result.trace.events if e.event_type == EventType.HOTEL_OFFER_ISSUE]
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(e.reason_code == ReasonCode.HOTEL_FIELD_IGNORED for e in issues))
        self.assertNotIn("PRIVATE", result.model_dump_json())
        self.assertEqual(self.get.call_count, 5)

    def test_all_invalid_hotel_offers_are_not_reported_as_empty_inventory(self):
        def response(url, *, params, **kwargs):
            if params["engine"] == "google_hotels":
                return http_response({"search_metadata": {"status": "Success"}, "properties": [
                    None, {"name": "PRIVATE"}, raw_hotel(total=-1),
                ]})
            return self.serpapi_response(url, params=params, **kwargs)

        self.get.side_effect = response
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.HOTEL_OUTPUT_INVALID)
        self.assertEqual(result.trace.events[-1].details["response_error_kind"], "all_hotel_offers_invalid")
        self.assertEqual(sum(e.event_type == EventType.HOTEL_OFFER_ISSUE for e in result.trace.events), 3)
        self.assertNotIn(ReasonCode.NO_HOTELS_FOUND, [e.reason_code for e in result.trace.events])
        self.assertNotIn("PRIVATE", result.model_dump_json())
        self.assertEqual(self.get.call_count, 5)
        self.assertEqual(self.post.call_count, 1)

    def test_one_city_with_all_invalid_hotel_offers_does_not_remove_other_cities(self):
        self.mock_three_cities()
        original = self.get.side_effect

        def response(url, *, params, **kwargs):
            if params["engine"] == "google_hotels" and params["q"] == "Hotels in Vienna, Austria":
                return http_response({"search_metadata": {"status": "Success"}, "properties": [None, {"name": "Unpriced"}]})
            return original(url, params=params, **kwargs)

        self.get.side_effect = response
        result = build_proposal(self.open_request(), limits=SearchLimits(
            max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
        ))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Paris"])
        self.assertEqual(result.search_status, "partial")
        self.assertEqual(result.search_issues[0].scope, "destination")
        self.assertEqual(result.search_issues[0].code, ReasonCode.HOTEL_OUTPUT_INVALID)
        failure = next(e for e in result.trace.events if e.event_type == EventType.DESTINATION_SEARCH_FAILED)
        self.assertEqual(failure.details["response_error_kind"], "all_hotel_offers_invalid")
        self.assertEqual(sum(e.event_type == EventType.HOTEL_OFFER_ISSUE for e in result.trace.events), 2)
        self.assertEqual(self.get.call_count, 13)
        self.assertEqual(self.post.call_count, 5)

    def test_valid_but_over_budget_hotels_keep_the_budget_error_even_with_discarded_items(self):
        def response(url, *, params, **kwargs):
            if params["engine"] == "google_hotels":
                return http_response({"search_metadata": {"status": "Success"}, "properties": [raw_hotel(total=1000), None]})
            return self.serpapi_response(url, params=params, **kwargs)

        self.get.side_effect = response
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.NO_AFFORDABLE_PROPOSAL)
        self.assertIn(EventType.HOTEL_SEARCH_PARTIAL, [e.event_type for e in result.trace.events])
        self.assertNotIn(ReasonCode.NO_HOTELS_FOUND, [e.reason_code for e in result.trace.events])
        self.assertEqual(self.get.call_count, 5)

    def test_invalid_destination_data_preserves_other_cities_and_reports_partial_search(self):
        for stage, calls, code in (
            ("location_resolution", 10, ReasonCode.FLIGHT_OUTPUT_INVALID),
            ("flight_search", 11, ReasonCode.FLIGHT_OUTPUT_INVALID),
            ("hotel_search", 13, ReasonCode.HOTEL_OUTPUT_INVALID),
        ):
            with self.subTest(stage=stage):
                self.get.reset_mock()
                self.post.reset_mock()
                self.fail_middle_city(stage)
                result = build_proposal(self.open_request(), limits=SearchLimits(
                    max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
                ))
                self.assertIsInstance(result, TripSearchResult)
                self.assertEqual([p.destination.name for p in result.proposals], ["Lisbon", "Paris"])
                self.assertEqual(result.search_status, "partial")
                self.assertEqual(result.summary_status, "generated")
                self.assertEqual(len(result.search_issues), 1)
                issue = result.search_issues[0]
                self.assertEqual((issue.destination, issue.country, issue.stage, issue.code), ("Vienna", "Austria", stage, code))
                failures = [event for event in result.trace.events if event.event_type == EventType.DESTINATION_SEARCH_FAILED]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0].details["scope"], "destination")
                self.assertEqual(failures[0].details["stage"], stage)
                self.assertEqual(failures[0].details["response_error_kind"], "missing_hotel_results" if stage == "hotel_search" else "missing_results")
                self.assertEqual(result.trace.events[-1].details["search_status"], "partial")
                self.assertEqual(result.trace.events[-1].details["affected_destinations"], ["Vienna"])
                self.assertEqual(self.get.call_count, calls)
                self.assertEqual(self.post.call_count, 5)
                self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_all_destination_lookups_invalid_return_an_error_not_empty_inventory(self):
        self.mock_three_cities()
        original = self.get.side_effect

        def response(url, *, params, **kwargs):
            if params["engine"] == "google_flights_autocomplete" and params["q"] != "Rome":
                return http_response({"search_metadata": {"status": "Success"}})
            return original(url, params=params, **kwargs)

        self.get.side_effect = response
        result = build_proposal(self.open_request(), limits=SearchLimits(
            max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
        ))
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, "FLIGHT_OUTPUT_INVALID")
        self.assertEqual(sum(e.event_type == EventType.DESTINATION_SEARCH_FAILED for e in result.trace.events), 3)
        self.assertEqual(result.trace.events[-1].event_type, EventType.SEARCH_FAILED)
        self.assertEqual(self.get.call_count, 4)
        self.assertEqual(self.post.call_count, 4)

    def test_origin_lookup_failure_remains_fatal_for_open_destination(self):
        self.mock_three_cities()
        self.get.side_effect = None
        self.get.return_value = http_response({"search_metadata": {"status": "Success"}})
        result = build_proposal(self.open_request(), limits=SearchLimits(max_destinations=3, max_llm_calls=5))
        self.assertIsInstance(result, PlanningError)
        details = result.trace.events[-1].details
        self.assertEqual(details["stage"], "location_resolution")
        self.assertEqual(details["location_role"], "origin")
        self.assertEqual(details["response_error_kind"], "missing_results")
        self.get.assert_called_once()

    def test_authentication_and_remote_quota_errors_remain_fatal_after_valid_offers(self):
        for status in (401, 429):
            with self.subTest(status=status):
                self.get.reset_mock()
                self.post.reset_mock()
                self.mock_three_cities()
                original = self.get.side_effect

                def response(url, *, params, **kwargs):
                    if params.get("q") == "Vienna, Austria":
                        return http_response({}, status=status)
                    return original(url, params=params, **kwargs)

                self.get.side_effect = response
                result = build_proposal(self.open_request(), limits=SearchLimits(
                    max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
                ))
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, "FLIGHT_PROVIDER_ERROR")
                self.assertIn(EventType.COMBINATION_ACCEPTED, [e.event_type for e in result.trace.events])
                self.assertNotIn(EventType.PROPOSALS_SELECTED, [e.event_type for e in result.trace.events])
                self.assertEqual(self.get.call_count, 6)
                self.assertEqual(self.post.call_count, 3)

    def test_call_limits_remain_fatal_after_a_destination_is_skipped(self):
        for limits, service in (
            (SearchLimits(max_destinations=3, max_serpapi_calls=6, max_llm_calls=5), "serpapi"),
            (SearchLimits(max_destinations=3, max_serpapi_calls=13, max_llm_calls=3), "llm"),
        ):
            with self.subTest(service=service):
                self.get.reset_mock()
                self.post.reset_mock()
                self.fail_middle_city("location_resolution")
                result = build_proposal(self.open_request(), limits=limits)
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
                self.assertEqual(result.trace.events[-1].details["service"], service)

    def test_partial_search_and_summary_failure_have_independent_statuses(self):
        self.fail_middle_city("location_resolution")
        self.summary_content = "invalid JSON"
        result = build_proposal(self.open_request(), limits=SearchLimits(
            max_destinations=3, max_serpapi_calls=13, max_llm_calls=5,
        ))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(result.search_status, "partial")
        self.assertEqual(result.summary_status, "unavailable")
        self.assertEqual(len(result.proposals), 2)

    def test_periods_are_ranked_and_deduplicated_before_the_cap(self):
        second, third = self.start + timedelta(days=7), self.start + timedelta(days=14)
        self.periods = [self.period(second, 70), self.period(self.start, 10), self.period(self.start, 90), self.period(third, 40)]
        result = build_proposal(self.flexible_request(), limits=SearchLimits(max_periods_per_destination=2, max_serpapi_calls=8))
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(self.get.call_count, 8)
        self.assertEqual(self.post.call_count, 2)
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
            self.assertEqual(result.trace.events[-1].details["llm_calls"], 2)
        self.assertEqual(self.get.call_count, 10)
        self.assertEqual(self.post.call_count, 4)

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
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 2)
        self.assertEqual(result.trace.events[-1].details["llm_retries"], 1)
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 0)
        self.get.assert_not_called()

    def assert_summary_unavailable(self, result, reason, *, calls=2):
        self.assertIsInstance(result, TripSearchResult)
        self.assertEqual(result.summary_status, "unavailable")
        self.assertTrue(result.proposals)
        self.assertTrue(all(p.summary is None and p.comparison is not None for p in result.proposals))
        self.assertTrue(all(p.travel_context.rationale == "Test advice" for p in result.proposals))
        failure = result.trace.events[-2]
        self.assertEqual(failure.event_type, EventType.PROPOSAL_SUMMARY_FAILED)
        self.assertEqual(failure.reason_code, reason)
        self.assertEqual(failure.details["agent"], "proposal_summary")
        self.assertEqual(result.trace.events[-1].event_type, EventType.PROPOSALS_SELECTED)
        self.assertEqual(result.trace.events[-1].details["summary_status"], "unavailable")
        self.assertEqual(result.trace.events[-1].details["llm_calls"], calls)
        self.assertEqual(self.post.call_count, calls)
        self.assertEqual(self.get.call_count, 5)
        self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)
        return failure

    def test_summary_budget_exhaustion_preserves_offers_without_extra_http(self):
        for mode, calls in (("fixed", 1), ("open", 2)):
            with self.subTest(mode=mode):
                self.get.reset_mock()
                self.post.reset_mock()
                request = self.request if mode == "fixed" else self.open_request()
                result = build_proposal(request, limits=SearchLimits(max_llm_calls=calls))
                failure = self.assert_summary_unavailable(result, ReasonCode.SEARCH_LIMIT_REACHED, calls=calls)
                self.assertEqual(failure.details, {
                    "agent": "proposal_summary", "service": "llm", "used": calls, "limit": calls,
                })

    def test_summary_timeout_preserves_offers_and_safe_diagnostics(self):
        self.summary_transport_error = Timeout("PRIVATE-PROMPT PRIVATE-KEY")
        result = build_proposal(self.request, limits=SearchLimits())
        failure = self.assert_summary_unavailable(result, ReasonCode.LLM_PROVIDER_ERROR)
        self.assertEqual(failure.details["llm_error_kind"], "timeout")
        self.assertNotIn("PRIVATE", result.model_dump_json())

    def test_summary_remote_error_preserves_http_diagnostics_not_body(self):
        original = self.llm_response

        def response(url, **kwargs):
            if kwargs["json"]["response_format"]["json_schema"]["name"] == "proposal_summaries":
                return http_response({"error": {"code": 429, "message": "PRIVATE-KEY"}}, status=429)
            return original(url, **kwargs)

        self.post.side_effect = response
        result = build_proposal(self.request, limits=SearchLimits())
        failure = self.assert_summary_unavailable(result, ReasonCode.LLM_PROVIDER_ERROR)
        self.assertEqual(failure.details["llm_error_kind"], "rate_limit")
        self.assertEqual(failure.details["llm_http_status"], 429)
        self.assertEqual(failure.details["llm_api_code"], 429)
        self.assertNotIn("PRIVATE-KEY", result.model_dump_json())

    def test_invalid_summary_json_and_schema_preserve_offers(self):
        for content in ("PRIVATE-INVALID-JSON", json.dumps({"items": [], "private": "PRIVATE"})):
            with self.subTest(content=content):
                self.get.reset_mock()
                self.post.reset_mock()
                self.summary_content = content
                result = build_proposal(self.request, limits=SearchLimits())
                self.assert_summary_unavailable(result, ReasonCode.LLM_OUTPUT_INVALID)
                self.assertNotIn("PRIVATE", result.model_dump_json())

    def test_invalid_summary_ids_preserve_offers_without_partial_text(self):
        for ids in (["unknown"], ["p1", "p1"], ["p1", "unknown"]):
            with self.subTest(ids=ids):
                self.get.reset_mock()
                self.post.reset_mock()
                self.summary_content = json.dumps({"items": [
                    {"proposal_id": id_, "title": "PRIVATE", "advantage": "PRIVATE", "tradeoff": None}
                    for id_ in ids
                ]})
                result = build_proposal(self.request, limits=SearchLimits())
                self.assert_summary_unavailable(result, ReasonCode.LLM_CONSTRAINT_VIOLATION)
                self.assertNotIn("PRIVATE", result.model_dump_json())

    def test_summary_success_and_failure_leave_identical_offer_facts(self):
        success = build_proposal(self.request, limits=SearchLimits())
        self.summary_content = "invalid JSON"
        failure = build_proposal(self.request, limits=SearchLimits())
        self.assertEqual(
            [p.model_dump(exclude={"summary"}) for p in success.proposals],
            [p.model_dump(exclude={"summary"}) for p in failure.proposals],
        )
        self.assertEqual(success.summary_status, "generated")
        self.assertEqual(failure.summary_status, "unavailable")

    def test_summary_quality_failure_preserves_facts_and_reports_safe_reason_without_retry(self):
        success = build_proposal(self.request, limits=SearchLimits())
        for text, kind in (
            ("PRIVATE unverified details and", "incomplete_text"),
            ("PRIVATE suitability not_verified.", "internal_reference"),
            ("PRIVATE museums in the\u4e2d\u5fc3.", "unexpected_script"),
        ):
            with self.subTest(kind=kind):
                self.get.reset_mock()
                self.post.reset_mock()
                self.summary_content = json.dumps({"items": [
                    {**p.summary.model_dump(), "advantage": text}
                    for p in success.proposals
                ]})
                result = build_proposal(self.request, limits=SearchLimits())
                failure = self.assert_summary_unavailable(result, ReasonCode.LLM_CONSTRAINT_VIOLATION)
                self.assertEqual(failure.details["summary_quality_issue"], kind)
                self.assertEqual(failure.details["summary_field"], "advantage")
                self.assertNotIn("PRIVATE", result.model_dump_json())
                self.assertEqual(
                    [p.model_dump(exclude={"summary"}) for p in success.proposals],
                    [p.model_dump(exclude={"summary"}) for p in result.proposals],
                )

    def test_no_summary_attempt_without_selected_offers(self):
        self.no_flights = True
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertIsInstance(result, PlanningError)
        self.post.assert_called_once()
        self.assertNotIn(EventType.PROPOSAL_SUMMARY_FAILED, [e.event_type for e in result.trace.events])


if __name__ == "__main__":
    unittest.main()

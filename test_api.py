from copy import deepcopy
from datetime import date
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import create_app
from execution import SearchLimitExceeded
from models import (
    DecisionEvent, DecisionTrace, DestinationSearchIssue, EventType,
    PlanningError, ReasonCode, TripRequest, TripSearchResult,
)
from planning import build_proposal as run_planner
from providers.flights import FlightLocationError, find_city_options
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError
from test_execution import http_response
from test_flights import fake_city, raw_leg
from test_hotels import raw_hotel
from test_open_destination_selection import city_proposal
from test_proposal_summary import fake_summary_response


PRIVATE = "PRIVATE-API-KEY-AND-PROMPT"


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.find = self.enterContext(patch("api.find_city_options", return_value=[]))
        self.plan = self.enterContext(patch("api.build_proposal"))
        self.result = TripSearchResult(
            proposals=[city_proposal("Lisbon", "Portugal", "/m/04llb", "143.27")],
            trace=DecisionTrace(events=[]), summary_status="generated",
        )
        self.plan.return_value = self.result
        self.client = self.enterContext(TestClient(create_app(allowed_origins=[]), raise_server_exceptions=False))
        self.body = {
            "origin": "Rome", "origin_city_id": "/m/06c62", "adults": 1,
            "budget_eur": 900, "budget_flexibility_pct": 0,
            "destination_pref": {"mode": "fixed", "destination": "Lisbon", "city_id": "/m/04llb"},
            "time_pref": {"mode": "exact_dates", "start_date": "2027-05-10", "end_date": "2027-05-15"},
            "vibe": {"free_text": "Museums and local food, at a slow pace.", "themes": ["food", "museums"]},
        }

    def test_health_needs_no_provider_configuration_or_calls(self):
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.find.assert_not_called()
        self.plan.assert_not_called()

    def test_docs_and_openapi_are_available_without_provider_calls(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(set(schema["paths"]), {"/health", "/locations", "/proposals"})
        models = schema["components"]["schemas"]
        for name in ("ProposalRequest", "ApiDestinationPreference", "ApiTimePreference", "ApiTravelVibe"):
            self.assertFalse(models[name]["additionalProperties"])
        self.assertNotIn("reference_date", models["ProposalRequest"]["properties"])
        self.assertNotIn("limits", models["ProposalRequest"]["properties"])
        self.assertIn("422", schema["paths"]["/proposals"]["post"]["responses"])
        self.find.assert_not_called()
        self.plan.assert_not_called()

    def test_location_projection_preserves_transport_ids_without_raw_provider_fields(self):
        self.find.return_value = [{
            "name": "Naples", "id": "/g/1233f22n", "description": None, "api_key": PRIVATE,
            "airports": [
                {"id": "NAP", "name": "Naples International Airport", "private": PRIVATE},
                {"id": "INP", "name": "Naples Central"},
                {"id": "4GS", "name": {"private": PRIVATE}},
            ],
        }]
        response = self.client.get("/locations", params={"q": "  Naples, Italy  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"locations": [{
            "name": "Naples", "city_id": "/g/1233f22n", "description": None,
            "transport_locations": [
                {"id": "NAP", "name": "Naples International Airport"},
                {"id": "INP", "name": "Naples Central"},
                {"id": "4GS", "name": None},
            ],
        }]})
        self.assertNotIn(PRIVATE, response.text)
        self.assertEqual(self.find.call_args.args, ("Naples, Italy",))
        budget = self.find.call_args.kwargs["budget"]
        self.assertEqual((budget.limits.max_serpapi_calls, budget.limits.max_llm_calls), (1, 0))
        self.plan.assert_not_called()

    def test_locations_use_a_fresh_single_call_budget_for_each_request(self):
        budgets = []

        def find(query, *, budget):
            budget.claim("serpapi")
            budgets.append(budget)
            return []

        self.find.side_effect = find
        for _ in range(2):
            response = self.client.get("/locations", params={"q": "Rome"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"locations": []})
        self.assertIsNot(budgets[0], budgets[1])
        self.assertTrue(all(budget.used == {"serpapi": 1, "llm": 0} for budget in budgets))

    def test_invalid_location_queries_do_not_call_the_provider(self):
        for query in (None, "", "R", "  ", "\n\t", "x" * 121):
            with self.subTest(query=query):
                response = self.client.get("/locations", params={} if query is None else {"q": query})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["code"], "INVALID_REQUEST")
        self.find.assert_not_called()

    def test_location_provider_errors_are_sanitized_and_mapped(self):
        for error, status, code in (
            (SerpAPIResponseError(PRIVATE), 502, "LOCATION_OUTPUT_INVALID"),
            (SerpAPIError(PRIVATE, code="TIMEOUT"), 504, "LOCATION_PROVIDER_TIMEOUT"),
            (SerpAPIError(PRIVATE, code="CONFIGURATION"), 503, "LOCATION_PROVIDER_ERROR"),
            (SerpAPIError(PRIVATE, code="AUTHENTICATION"), 503, "LOCATION_PROVIDER_ERROR"),
            (SerpAPIError(PRIVATE, code="QUOTA_OR_RATE_LIMIT"), 503, "LOCATION_PROVIDER_ERROR"),
            (FlightLocationError(PRIVATE), 422, "INVALID_REQUEST"),
            (SearchLimitExceeded("serpapi", 1, 1), 503, "SEARCH_LIMIT_REACHED"),
        ):
            with self.subTest(error=error):
                self.find.side_effect = error
                response = self.client.get("/locations", params={"q": "Rome"})
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["code"], code)
                self.assertNotIn(PRIVATE, response.text)

    def test_proposals_pass_a_domain_request_without_rewriting_preferences(self):
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.result.model_dump(mode="json"))
        request = self.plan.call_args.args[0]
        self.assertIs(type(request), TripRequest)
        self.assertEqual(request, TripRequest.model_validate(self.body))
        self.assertEqual(set(self.plan.call_args.kwargs), {"limits"})
        self.assertEqual(response.json()["proposals"][0]["total_price_eur"], "699.27")

    def test_fixed_and_open_profiles_are_chosen_by_the_server(self):
        for mode, expected in (("fixed", (1, 5, 3)), ("open", (3, 13, 6))):
            with self.subTest(mode=mode):
                body = deepcopy(self.body)
                if mode == "open":
                    body["destination_pref"] = {"mode": "open"}
                self.assertEqual(self.client.post("/proposals", json=body).status_code, 200)
                limits = self.plan.call_args.kwargs["limits"]
                self.assertEqual((limits.max_destinations, limits.max_serpapi_calls, limits.max_llm_calls), expected)
                self.assertEqual((limits.max_outbound, limits.max_periods_per_destination), (1, 1))

    def test_unknown_top_level_controls_are_rejected_not_ignored(self):
        for field in ("limits", "reference_date", "country_verifier", "model", "api_key"):
            with self.subTest(field=field):
                response = self.client.post("/proposals", json={**self.body, field: PRIVATE})
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(PRIVATE, response.text)
                self.assertEqual(response.json()["issues"][0]["type"], "extra_forbidden")
        self.plan.assert_not_called()

    def test_unknown_nested_fields_are_rejected_without_echoing_values(self):
        for field in ("destination_pref", "time_pref", "vibe"):
            with self.subTest(field=field):
                body = deepcopy(self.body)
                body[field]["limits"] = PRIVATE
                response = self.client.post("/proposals", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(PRIVATE, response.text)
        self.plan.assert_not_called()

    def test_invalid_input_is_rejected_before_the_planner(self):
        for changes in (
            {"origin": ""}, {"origin": "   "}, {"origin": "x" * 161}, {"adults": 0},
            {"budget_eur": 0}, {"budget_flexibility_pct": 31},
            {"destination_pref": {"mode": "fixed", "destination": "   "}},
            {"destination_pref": {"mode": "open", "city_id": "/m/04llb"}},
            {"origin_city_id": "Rome"},
            {"time_pref": {"mode": "exact_dates", "start_date": "2027-05-15", "end_date": "2027-05-10"}},
            {"time_pref": {"mode": "month"}},
            {"time_pref": {"mode": "season", "season": "monsoon"}},
            {"time_pref": {"mode": "flexible_window", "min_days": 8, "max_days": 4}},
            {"vibe": {"free_text": "x" * 2001}},
            {"vibe": {"themes": ["museums"] * 21}},
        ):
            with self.subTest(changes=changes):
                response = self.client.post("/proposals", json={**self.body, **changes})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["code"], "INVALID_REQUEST")
                for issue in response.json()["issues"]:
                    self.assertEqual(set(issue), {"location", "type", "message"})
        self.plan.assert_not_called()

    def test_missing_fields_and_invalid_json_are_sanitized(self):
        responses = [
            self.client.post("/proposals", json={}),
            self.client.post("/proposals", content='{"api_key":"' + PRIVATE + '",', headers={"Content-Type": "application/json"}),
        ]
        for response in responses:
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(PRIVATE, response.text)
            self.assertNotIn('"input":', response.text)
            self.assertNotIn('"ctx":', response.text)
        self.plan.assert_not_called()

    def test_all_time_modes_and_optional_vibe_and_budget_are_supported(self):
        for preference in (
            self.body["time_pref"], {"mode": "month", "month": 5},
            {"mode": "season", "season": "spring"},
            {"mode": "flexible_window", "min_days": 5, "max_days": 7},
        ):
            with self.subTest(preference=preference):
                body = {key: value for key, value in self.body.items() if key not in ("budget_eur", "vibe")}
                body["time_pref"] = preference
                self.assertEqual(self.client.post("/proposals", json=body).status_code, 200)
                request = self.plan.call_args.args[0]
                self.assertIsNone(request.budget_eur)
                self.assertIsNone(request.vibe.free_text)
                self.assertEqual(request.vibe.themes, [])

    def test_partial_results_and_missing_summary_remain_successful(self):
        self.plan.return_value = self.result.model_copy(update={
            "search_status": "partial", "summary_status": "unavailable",
            "search_issues": [DestinationSearchIssue(
                destination="Valencia", country="Spain", stage="hotel_search",
                code=ReasonCode.HOTEL_OUTPUT_INVALID, message="No usable hotel response.",
            )],
        })
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["search_status"], "partial")
        self.assertEqual(response.json()["summary_status"], "unavailable")
        self.assertEqual(len(response.json()["proposals"]), 1)
        self.assertEqual(response.json()["search_issues"][0]["destination"], "Valencia")

    def test_planning_error_statuses_do_not_expose_internal_messages_or_trace(self):
        for code, status in (
            (ReasonCode.INVALID_TRAVEL_DATES, 422), (ReasonCode.FLIGHT_LOCATION_UNRESOLVED, 422),
            (ReasonCode.NO_FLIGHTS_FOUND, 200), (ReasonCode.NO_HOTELS_FOUND, 200),
            (ReasonCode.NO_AFFORDABLE_PROPOSAL, 200),
            (ReasonCode.LLM_OUTPUT_INVALID, 502), (ReasonCode.LLM_CONSTRAINT_VIOLATION, 502),
            (ReasonCode.FLIGHT_OUTPUT_INVALID, 502), (ReasonCode.HOTEL_OUTPUT_INVALID, 502),
            (ReasonCode.LLM_PROVIDER_ERROR, 503), (ReasonCode.FLIGHT_PROVIDER_ERROR, 503),
            (ReasonCode.HOTEL_PROVIDER_ERROR, 503), (ReasonCode.SEARCH_LIMIT_REACHED, 503),
        ):
            with self.subTest(code=code):
                self.plan.return_value = PlanningError(code=code, message=PRIVATE, trace=DecisionTrace(events=[DecisionEvent(
                    event_type=EventType.SEARCH_FAILED, reason_code=code, comment=PRIVATE,
                    details={"raw_error": PRIVATE},
                )]))
                response = self.client.post("/proposals", json=self.body)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["code"], code)
                self.assertNotIn(PRIVATE, response.text)
                self.assertNotIn("trace", response.json())

    def test_location_clarification_options_are_preserved(self):
        self.plan.return_value = PlanningError(
            code=ReasonCode.FLIGHT_LOCATION_UNRESOLVED, message=PRIVATE,
            trace=DecisionTrace(events=[DecisionEvent(
                event_type=EventType.SEARCH_FAILED, reason_code=ReasonCode.FLIGHT_LOCATION_UNRESOLVED,
                details={"location_options": ["Rome, Italy (/m/06c62)", "Rome, USA (/m/test_us)"]},
            )]),
        )
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(response.json()["location_options"]), 2)

    def test_provider_timeouts_are_distinct_from_invalid_output(self):
        for code, details in (
            (ReasonCode.FLIGHT_PROVIDER_ERROR, {"provider_error": "TIMEOUT"}),
            (ReasonCode.HOTEL_PROVIDER_ERROR, {"provider_error": "TIMEOUT"}),
            (ReasonCode.LLM_PROVIDER_ERROR, {"llm_error_kind": "timeout"}),
        ):
            with self.subTest(code=code):
                self.plan.return_value = PlanningError(code=code, message=PRIVATE, trace=DecisionTrace(events=[DecisionEvent(
                    event_type=EventType.SEARCH_FAILED, reason_code=code, details=details,
                )]))
                self.assertEqual(self.client.post("/proposals", json=self.body).status_code, 504)

    def test_unknown_planning_errors_and_unexpected_exceptions_return_safe_500(self):
        self.plan.return_value = PlanningError(code=PRIVATE, message=PRIVATE)
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "INTERNAL_ERROR")
        self.assertNotIn(PRIVATE, response.text)
        for mock, path, kwargs in (
            (self.plan, "/proposals", {"json": self.body}),
            (self.find, "/locations?q=Rome", {}),
        ):
            with self.subTest(path=path):
                mock.side_effect = RuntimeError(PRIVATE)
                response = self.client.request("POST" if path == "/proposals" else "GET", path, **kwargs)
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["code"], "INTERNAL_ERROR")
                self.assertNotIn(PRIVATE, response.text)

    def test_cors_only_allows_configured_frontend_origins(self):
        with TestClient(create_app(allowed_origins=["http://localhost:5173"])) as client:
            for origin, status in (("http://localhost:5173", 200), ("https://untrusted.example", 400)):
                with self.subTest(origin=origin):
                    response = client.options("/proposals", headers={
                        "Origin": origin, "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "Content-Type",
                    })
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.headers.get("access-control-allow-origin"), origin if status == 200 else None)
            self.assertEqual(client.get("/health", headers={"Origin": "http://localhost:5173"}).headers["access-control-allow-origin"], "http://localhost:5173")
        self.find.assert_not_called()
        self.plan.assert_not_called()

    def test_cors_is_disabled_by_default_and_can_be_configured_through_environment(self):
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertNotIn("access-control-allow-origin", response.headers)
        with patch.dict("os.environ", {"FRONTEND_ORIGINS": " http://localhost:5173,https://frontend.example "}):
            with TestClient(create_app()) as client:
                response = client.get("/health", headers={"Origin": "https://frontend.example"})
                self.assertEqual(response.headers["access-control-allow-origin"], "https://frontend.example")

    def test_cors_rejects_wildcards_and_null_origins(self):
        for origin in ("*", "null"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                create_app(allowed_origins=[origin])

    def enable_real_offline_pipeline(self):
        self.plan.side_effect = run_planner
        clock = self.enterContext(patch("planning.date", wraps=date))
        clock.today.return_value = date(2026, 10, 25)
        self.enterContext(patch.dict("os.environ", {
            "SERPAPI_API_KEY": PRIVATE, "OPENROUTER_API_KEY": PRIVATE, "OPENROUTER_MODEL": "test-model",
        }))

        def search_response(url, *, params, **kwargs):
            if params["engine"] == "google_flights_autocomplete":
                city = fake_city("Rome, Italy", city_id="/m/06c62") if params["q"] == "Rome" else fake_city("Lisbon, Portugal", city_id="/m/04llb")
                payload = {"suggestions": [city]}
            elif params["engine"] == "google_flights":
                if "departure_token" in params:
                    payload = {"other_flights": [raw_leg("LIS", "FCO", date.fromisoformat(params["return_date"]), price="180.25")]}
                else:
                    payload = {"best_flights": [raw_leg(day=date.fromisoformat(params["outbound_date"]), token="test-token")]}
            else:
                self.assertEqual(params["engine"], "google_hotels")
                payload = {"properties": [raw_hotel()]}
            return http_response({"search_metadata": {"status": "Success"}, **payload})

        def llm_response(url, *, json, **kwargs):
            schema = json["response_format"]["json_schema"]["name"]
            if schema == "destination_advice":
                content = '{"destinations":[{"name":"Lisbon","country":"Portugal","reason":"Test.","match_score":90,"themes_matched":[],"caution":null}]}'
            elif schema == "travel_window_advice":
                content = '{"selected_periods":[{"start":"2027-05-10","end":"2027-05-15","score":90,"reason":"Test."}],"alternatives":[],"warnings":[],"rationale":"Test."}'
            else:
                content = fake_summary_response(json["messages"])
            return http_response({"choices": [{"message": {"content": content}}]})

        get = self.enterContext(patch("providers.serpapi_client.requests.get", side_effect=search_response))
        post = self.enterContext(patch("llm.openrouter.requests.post", side_effect=llm_response))
        return get, post

    def test_real_planner_and_http_clients_complete_fixed_and_open_searches_offline(self):
        get, post = self.enable_real_offline_pipeline()
        for mode, llm_calls in (("fixed", 2), ("open", 3)):
            with self.subTest(mode=mode):
                get.reset_mock()
                post.reset_mock()
                body = deepcopy(self.body)
                if mode == "open":
                    body["destination_pref"] = {"mode": "open"}
                response = self.client.post("/proposals", json=body)
                self.assertEqual(response.status_code, 200)
                result = TripSearchResult.model_validate(response.json())
                self.assertEqual(result.proposals[0].destination_city_id, "/m/04llb")
                self.assertEqual(str(result.proposals[0].total_price_eur), "430.24")
                self.assertEqual(result.summary_status, "generated")
                self.assertEqual(get.call_count, 5)
                self.assertEqual(post.call_count, llm_calls)
                self.assertNotIn(PRIVATE, response.text)

    def test_real_past_input_returns_422_without_provider_calls(self):
        get, post = self.enable_real_offline_pipeline()
        self.body["time_pref"] = {"mode": "exact_dates", "start_date": "2026-10-24", "end_date": "2026-10-29"}
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INVALID_TRAVEL_DATES")
        get.assert_not_called()
        post.assert_not_called()

    def test_real_planner_rejects_an_unverified_city_id_before_searching_flights(self):
        get, post = self.enable_real_offline_pipeline()
        self.body["destination_pref"]["city_id"] = "/m/not_returned"
        response = self.client.post("/proposals", json=self.body)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "FLIGHT_LOCATION_UNRESOLVED")
        self.assertTrue(response.json()["location_options"])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(post.call_count, 1)

    def test_real_location_client_makes_one_counted_http_call(self):
        self.find.side_effect = find_city_options
        with patch.dict("os.environ", {"SERPAPI_API_KEY": PRIVATE}), patch(
            "providers.serpapi_client.requests.get", return_value=http_response({
                "search_metadata": {"status": "Success"},
                "suggestions": [fake_city("Rome, Italy", city_id="/m/06c62")],
            }),
        ) as get:
            response = self.client.get("/locations", params={"q": "Rome"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["locations"][0]["city_id"], "/m/06c62")
        get.assert_called_once()
        self.assertEqual(self.find.call_args.kwargs["budget"].used, {"serpapi": 1, "llm": 0})
        self.assertNotIn(PRIVATE, response.text)


if __name__ == "__main__":
    unittest.main()

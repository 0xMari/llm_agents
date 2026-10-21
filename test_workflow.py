import unittest
from unittest.mock import ANY, Mock, patch

import json

from requests.exceptions import Timeout
from llm.openrouter import OpenRouterError, call_openrouter_json
from planning import build_proposal, calculate_budget_ceiling
from execution import SearchLimits
from routing import route_request
from pydantic import ValidationError
from datetime import date, timedelta
from decimal import Decimal

from providers.flights import search_flights
from providers.flights import FlightLocationError, resolve_city
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError
from providers.hotels import parse_hotel_offer, search_hotels
from test_flights import fake_city, fake_flight_search, make_flight, raw_leg
from test_hotels import fake_hotel_search, raw_hotel

from scoring import score_proposal, select_top_proposals
from agents.destination_agent import suggest_destinations
from agents.travel_window import evaluate_travel_window, resolve_month_year

from models import (
    TripRequest,
    PlanningError,
    ReasonCode,
    DestinationPreference,
    TimePreference,
    TravelVibe,
    EventType,
    PlanningScenario,
    FlightOffer,
    FlightSearchQuery,
    HotelOffer,
    HotelSearchQuery,
    SuggestedDestination
)

def fake_travel_window_response(
    start="2026-09-15",
    end="2026-09-20",
    score=90,
    rationale="Settembre è una buona finestra di viaggio.",
):
    return json.dumps(
        {
            "selected_periods": [
                {
                    "start": start,
                    "end": end,
                    "reason": "Periodo adatto per clima e vibe.",
                    "score": score,
                }
            ],
            "warnings": [],
            "alternatives": [],
            "rationale": rationale,
        }
    )


def fake_flight_and_hotel(total_price: int | Decimal) -> tuple[FlightOffer, HotelOffer]:
    hotel = HotelOffer(
        destination="Lisbon", name="Test Stay",
        nightly_price_eur=50, nights=5, rating=4.4,
        total_price_eur=Decimal("250.00"),
    )
    flight = make_flight(total_price - hotel.total_price_eur)
    return flight, hotel


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.limits = SearchLimits(
            max_destinations=3, max_periods_per_destination=3,
            max_outbound=3, max_serpapi_calls=50, max_llm_calls=4,
        )
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.fake_resolve_city = self.enterContext(patch("planning.resolve_city", side_effect=fake_city))
        self.fake_flight_search = self.enterContext(patch("planning.search_flights", side_effect=fake_flight_search))
        self.fake_hotel_search = self.enterContext(patch("planning.search_hotels", side_effect=fake_hotel_search))
        self.openrouter_patcher = patch("agents.travel_window.call_openrouter_json")
        self.fake_openrouter_call = self.openrouter_patcher.start()
        self.addCleanup(self.openrouter_patcher.stop)

        self.fake_openrouter_call.return_value = fake_travel_window_response()

        self.destination_patcher = patch(
            "agents.destination_agent.call_openrouter_json"
        )
        self.fake_destination_call = self.destination_patcher.start()
        self.addCleanup(self.destination_patcher.stop)

        destination = SuggestedDestination(
            name="Lisbon",
            country="Portugal",
            reason="Destinazione simulata per verificare il workflow.",
            match_score=70,
        )

        self.fake_destination_call.return_value = json.dumps(
            {"destinations": [destination.model_dump(mode="json")]},
            ensure_ascii=False,
        )


    def build_proposal(self, request):
        # These tests exercise multiple candidates; smoke-test defaults are narrower.
        return build_proposal(request, limits=self.limits)

    def test_proposal_respects_budget(self):
        request = TripRequest(
            origin = "Rome",
            destination_pref = DestinationPreference(
                mode = "fixed",
                destination = "Lisbon",
            ),
            budget_eur = 900,
            time_pref = TimePreference(
                mode = "flexible_window",
                min_days = 4,
                max_days = 6,
            ),
            vibe = TravelVibe(
                themes = ["food"],
            ),
        )

        result = self.build_proposal(request)
        self.assertNotIsInstance(result, PlanningError)

        for proposal in result.proposals:
            self.assertLessEqual(proposal.total_price_eur, request.budget_eur)
        
        first_proposal = result.proposals[0]

        self.assertTrue(
            any(
                comment.reason_code == ReasonCode.WITHIN_BUDGET
                for comment in first_proposal.comments
            )
        )


    def test_invalid_days_raise_validation_error(self):
        with self.assertRaises(ValidationError):
            TripRequest(
                origin = "Rome",
                destination_pref = DestinationPreference(
                    mode = "fixed",
                    destination = "Lisbon",
                ),
                budget_eur = 900,
                time_pref = TimePreference(
                    mode = "flexible_window",
                    min_days = 8,
                    max_days = 4,
                ),
                vibe = TravelVibe(
                    themes = ["food"],
                ),
            )


    # def test_low_budget_returns_structured_error(self):
    #     request = TripRequest(
    #         origin = "Rome",
    #         destination_pref = DestinationPreference(
    #             mode = "fixed",
    #             destination = "Lisbon",
    #         ),
    #         budget_eur = 300,
    #         time_pref = TimePreference(
    #             mode = "flexible_window",
    #             min_days = 4,
    #             max_days = 6,
    #         ),
    #         vibe = TravelVibe(
    #             themes = ["food"],
    #         ),
    #     )

    #     result = build_proposal(request)

    #     self.assertIsInstance(result, PlanningError)
    #     self.assertEqual(result.code, "NO_AFFORDABLE_PROPOSAL")

    #     self.assertTrue(
    #         any(
    #             event.reason_code == ReasonCode.OVER_BUDGET
    #             for event in result.trace.events
    #         )
    #     )


    def test_returns_at_most_three_proposals(self):
        request = TripRequest(
            origin = "Rome",
            destination_pref = DestinationPreference(
                mode = "fixed",
                destination = "Lisbon",
            ),
            budget_eur = 900,
            time_pref = TimePreference(
                mode = "flexible_window",
                min_days = 4,
                max_days = 6,
            ),
            vibe = TravelVibe(
                themes = ["food"],
            ),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertLessEqual(len(result.proposals), 3)

    
    def test_exact_dates_use_user_dates(self):
        start = date(2026,9,15)
        end = date(2026,9,20)

        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start=start.isoformat(),
            end=end.isoformat(),
            score=100,
            rationale="Date esatte rispettate.",
        )

        request = TripRequest(
            origin = "Rome",
            destination_pref = DestinationPreference(
                mode = "fixed",
                destination = "Lisbon",
            ),
            budget_eur = 900,
            time_pref = TimePreference(
                mode = "exact_dates",
                start_date = start,
                end_date = end,
            ),
            vibe = TravelVibe(
                themes = ["food"],
            ),
        )

        result = self.build_proposal(request)
        self.assertNotIsInstance(result, PlanningError)
        for proposal in result.proposals:
            self.assertEqual(proposal.period.start, start)
            self.assertEqual(proposal.period.end, end)
            self.assertEqual(proposal.period.score, 100)

    
    def test_trace_contains_detected_scenario(self):
        request = TripRequest(
            origin = "Rome",
            budget_eur = 900,
            destination_pref = DestinationPreference(
                mode = "fixed",
                destination = "Lisbon",
            ),
            time_pref = TimePreference(
                mode = "flexible_window",
                min_days = 4,
                max_days = 6,
            ),
            vibe = TravelVibe(
                themes = ["food"],
            ),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)

        routing_events = [
            event
            for event in result.trace.events
            if event.event_type == EventType.REQUEST_ROUTED
            and event.reason_code == ReasonCode.SCENARIO_DETECTED
        ]

        self.assertEqual(len(routing_events), 1)
        self.assertEqual(
            routing_events[0].details["scenario"],
            PlanningScenario.FIXED_DESTINATION_FLEXIBLE_DATES.value,
        )

    
    def test_routes_fixed_destination_exact_dates(self):
        request = TripRequest(
            origin = "Rome",
            budget_eur = 900,
            destination_pref = DestinationPreference(mode = "fixed", destination = "Lisbon"),
            time_pref = TimePreference(
                mode = "exact_dates",
                start_date = date(2026, 9, 15),
                end_date = date(2026, 9, 20),
            ),
        )

        self.assertEqual(
            route_request(request),
            PlanningScenario.FIXED_DESTINATION_EXACT_DATES,
        )


    def test_routes_fixed_destination_flexible_dates(self):
        request = TripRequest(
            origin = "Rome",
            budget_eur = 900,
            destination_pref = DestinationPreference(mode = "fixed", destination = "Lisbon"),
            time_pref = TimePreference(
                mode = "flexible_window",
                min_days = 4,
                max_days = 8,
            ),
        )

        self.assertEqual(
            route_request(request),
            PlanningScenario.FIXED_DESTINATION_FLEXIBLE_DATES,
        )
    

    def test_routes_open_destination_exact_dates(self):
        request = TripRequest(
            origin = "Rome",
            budget_eur = 900,
            destination_pref = DestinationPreference(mode = "open"),
            time_pref = TimePreference(
                mode = "exact_dates",
                start_date = date(2026, 9, 15),
                end_date = date(2026, 9, 20),
            ),
        )

        self.assertEqual(
            route_request(request),
            PlanningScenario.OPEN_DESTINATION_EXACT_DATES,
        )

    
    def test_routes_open_destination_flexible_dates(self):
        request = TripRequest(
            origin = "Rome",
            budget_eur = 900,
            destination_pref = DestinationPreference(mode = "open"),
            time_pref = TimePreference(
                mode = "flexible_window",
                min_days = 4,
                max_days = 8,
            ),
        )

        self.assertEqual(
            route_request(request),
            PlanningScenario.OPEN_DESTINATION_FLEXIBLE_DATES,
        )


    def test_search_flights_uses_query_dates(self):
        query = FlightSearchQuery(
            origin = "Rome",
            destination = "Lisbon",
            origin_id = "/m/test_rome",
            destination_id = "/m/test_lisbon",
            departure_date = date(2026, 9, 15),
            return_date = date(2026, 9, 20),
        )

        with patch("providers.flights.search_serpapi", side_effect=[
            {"best_flights": [raw_leg(token="test-token")]},
            {"other_flights": [raw_leg("LIS", "FCO", date(2026, 9, 20))]},
        ]):
            flights = search_flights(query)

        self.assertTrue(len(flights) > 0)
        for flight in flights:
            self.assertEqual(flight.origin, query.origin)
            self.assertEqual(flight.destination, query.destination)
            self.assertEqual(flight.departure_date, query.departure_date)
            self.assertEqual(flight.return_date, query.return_date)


    def test_search_flights_filters_by_origin_and_destination(self):
        query = FlightSearchQuery(
            origin="Milan",
            destination="Lisbon",
            origin_id="/m/test_milan",
            destination_id="/m/test_lisbon",
            departure_date=date(2026, 9, 15),
            return_date=date(2026, 9, 20),
        )

        with patch("providers.flights.search_serpapi", return_value={"other_flights": []}) as api:
            flights = search_flights(query)

        self.assertEqual(flights, [])
        self.assertEqual(api.call_args.args[0]["departure_id"], query.origin_id)
        self.assertEqual(api.call_args.args[0]["arrival_id"], query.destination_id)

    
    def test_search_hotels_filters_by_max_total_price(self):
        query = HotelSearchQuery(
            destination = "Lisbon",
            checkin_date = date(2026, 9, 15),
            checkout_date = date(2026, 9, 20),
            max_total_price_eur = 500,
        )

        with patch("providers.hotels.search_serpapi", return_value={"properties": [raw_hotel(total=499), raw_hotel(total=501)]}):
            hotels = search_hotels(query)

        self.assertEqual(len(hotels), 1)
        for hotel in hotels:
            self.assertEqual(hotel.destination, query.destination)
            self.assertLessEqual(hotel.total_price_eur, 500)


    def test_search_hotels_filters_by_destination(self):
        query = HotelSearchQuery(
            destination="Madrid",
            checkin_date=date(2026, 9, 15),
            checkout_date=date(2026, 9, 20),
        )

        with patch("providers.hotels.search_serpapi", return_value={"properties": []}) as api:
            hotels = search_hotels(query)

        self.assertEqual(hotels, [])
        self.assertEqual(api.call_args.args[0]["q"], "Hotels in Madrid")


    def test_fixed_destination_becomes_proposal_destination(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.fake_destination_call.assert_not_called()
        for proposal in result.proposals:
            self.assertEqual(proposal.destination.name, "Lisbon")
            self.assertEqual(proposal.destination.match_score, 100)
    

    def test_open_destination_uses_suggested_destinations(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
            vibe=TravelVibe(themes=["nature"]),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertGreater(len(result.proposals), 0)
        for proposal in result.proposals:
            self.assertIsNotNone(proposal.destination.name)


    def test_destination_agent_returns_validated_candidates(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        candidate = json.loads(self.fake_destination_call.return_value)["destinations"][0]

        for names in (["Lisbon"], ["Lisbon", "Porto", "Coimbra"]):
            with self.subTest(names=names):
                self.fake_destination_call.return_value = json.dumps({
                    "destinations": [dict(candidate, name=name) for name in names],
                })

                destinations = suggest_destinations(request)

                self.assertEqual([destination.name for destination in destinations], names)
                for destination in destinations:
                    self.assertIsInstance(destination, SuggestedDestination)
                    self.assertEqual(destination.country, "Portugal")
                    self.assertEqual(destination.match_score, candidate["match_score"])
                self.assertEqual(
                    self.fake_destination_call.call_args.kwargs["schema_name"],
                    "destination_advice",
                )
        self.fake_openrouter_call.assert_not_called()


    def test_destination_agent_rejects_invalid_responses(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        candidate = json.loads(self.fake_destination_call.return_value)["destinations"][0]
        invalid_candidates = {
            "empty list": [],
            "duplicates": [candidate, dict(candidate, name=" lisbon ")],
            "score too low": [dict(candidate, match_score=-1)],
            "score too high": [dict(candidate, match_score=101)],
            "blank name": [dict(candidate, name=" ")],
            "missing country": [{k: v for k, v in candidate.items() if k != "country"}],
            "unexpected field": [dict(candidate, price_eur=100)],
            "too many candidates": [
                dict(candidate, name=name) for name in ("Lisbon", "Porto", "Coimbra", "Faro")
            ],
        }
        responses = {"invalid JSON": "not valid json"}
        responses.update({
            label: json.dumps({"destinations": candidates})
            for label, candidates in invalid_candidates.items()
        })

        for label, response in responses.items():
            with self.subTest(case=label):
                self.fake_destination_call.return_value = response
                with self.assertRaises(ValidationError):
                    suggest_destinations(request)


    def assert_destination_failure_stops_workflow(self, expected_reason):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        with patch("planning.search_flights") as flights, patch(
            "planning.search_hotels"
        ) as hotels:
            result = self.build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, expected_reason.value)
        self.assertEqual(
            [event.event_type for event in result.trace.events],
            [EventType.REQUEST_ROUTED, EventType.DESTINATION_RESOLUTION_FAILED],
        )
        event = result.trace.events[-1]
        self.assertEqual(event.reason_code, expected_reason)
        expected_details = {
            "agent": "destination", "serpapi_calls": 0, "llm_calls": 0,
            **self.limits.model_dump(),
        }
        if expected_reason == ReasonCode.LLM_PROVIDER_ERROR:
            expected_details.update(
                llm_error_kind="provider_error", llm_http_status=None, llm_api_code=None,
            )
        self.assertEqual(event.details, expected_details)
        self.fake_destination_call.assert_called_once()
        self.fake_openrouter_call.assert_not_called()
        flights.assert_not_called()
        hotels.assert_not_called()


    def test_build_proposal_handles_destination_provider_error(self):
        self.fake_destination_call.side_effect = OpenRouterError("Timeout simulato")

        self.assert_destination_failure_stops_workflow(ReasonCode.LLM_PROVIDER_ERROR)


    def test_build_proposal_handles_invalid_destination_json(self):
        self.fake_destination_call.return_value = "not valid json"

        self.assert_destination_failure_stops_workflow(ReasonCode.LLM_OUTPUT_INVALID)


    def test_trace_contains_resolved_destinations(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                event.event_type == EventType.DESTINATIONS_RESOLVED
                for event in result.trace.events
            )
        )

    
    def test_calculates_budget_ceiling(self):
        for budget, flexibility, expected in (
            (1000, 20, 1200),
            (None, 0, None),
            (None, 20, None),
            (None, 30, None),
            (800, 0, 800),
            (800, 20, 960),
            (800, 30, 1040),
            (100, 13, 113),
            (999, 20, Decimal("1198.80")),
        ):
            with self.subTest(budget=budget, flexibility=flexibility):
                request = TripRequest(
                    origin="Rome", budget_eur=budget,
                    budget_flexibility_pct=flexibility,
                    destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
                    time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                )

                self.assertEqual(calculate_budget_ceiling(request), expected)


    def test_budget_rejects_nonpositive_amounts(self):
        for budget in (0, -1):
            with self.subTest(budget=budget):
                with self.assertRaises(ValidationError):
                    TripRequest(
                        origin="Rome", budget_eur=budget,
                        destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
                        time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                    )


    def test_missing_budget_allows_offers_and_serializes_null_details(self):
        flight, hotel = fake_flight_and_hotel(total_price=20000)
        for mode, budget_fields in (
            ("fixed", {}),
            ("open", {"budget_eur": None, "budget_flexibility_pct": 30}),
        ):
            with self.subTest(mode=mode):
                request = TripRequest(
                    origin="Rome",
                    destination_pref=DestinationPreference(
                        mode=mode, destination="Lisbon" if mode == "fixed" else None,
                    ),
                    time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                    **budget_fields,
                )
                self.assertIsNone(request.budget_eur)
                with patch("planning.search_flights", return_value=[flight]), patch(
                    "planning.search_hotels", return_value=[hotel]
                ):
                    result = self.build_proposal(request)

                self.assertNotIsInstance(result, PlanningError)
                self.assertEqual(len(result.proposals), 1)
                self.assertEqual(result.proposals[0].total_price_eur, 20000)
                serialized = json.loads(result.model_dump_json())
                budget_comments = [
                    comment for comment in serialized["proposals"][0]["comments"]
                    if comment["reason_code"] == ReasonCode.NO_BUDGET_PROVIDED.value
                ]
                accepted_events = [
                    event for event in serialized["trace"]["events"]
                    if event["event_type"] == EventType.COMBINATION_ACCEPTED.value
                ]
                self.assertEqual(len(budget_comments), 1)
                self.assertEqual(len(accepted_events), 1)
                for entry in budget_comments + accepted_events:
                    self.assertEqual(entry["reason_code"], ReasonCode.NO_BUDGET_PROVIDED.value)
                    self.assertIsNone(entry["details"]["budget_eur"])
                    self.assertIsNone(entry["details"]["budget_ceiling_eur"])
                reasons = [event.reason_code for event in result.trace.events]
                self.assertNotIn(ReasonCode.OVER_MAX_BUDGET, reasons)
                self.assertNotIn(ReasonCode.WITHIN_BUDGET, reasons)
                self.assertNotIn(ReasonCode.OVER_PREFERRED_BUDGET, reasons)

                calls = [self.fake_openrouter_call]
                if mode == "open":
                    calls.append(self.fake_destination_call)
                for llm_call in calls:
                    payload = json.loads(llm_call.call_args.kwargs["messages"][1]["content"])
                    self.assertIsNone(payload["budget_eur"])
                    self.assertEqual(payload["budget_flexibility_pct"], request.budget_flexibility_pct)


    def test_missing_budget_preserves_unavailable_offer_errors(self):
        request = TripRequest(
            origin="Rome",
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        flight, _ = fake_flight_and_hotel(total_price=800)
        for missing_step, expected in (
            ("flights", ReasonCode.NO_FLIGHTS_FOUND),
            ("hotels", ReasonCode.NO_HOTELS_FOUND),
        ):
            with self.subTest(missing_step=missing_step):
                flights = [] if missing_step == "flights" else [flight]
                with patch("planning.search_flights", return_value=flights), patch(
                    "planning.search_hotels", return_value=[]
                ):
                    result = self.build_proposal(request)

                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, expected.value)


    def test_budget_limits_include_the_boundary_and_reject_one_euro_more(self):
        for flexibility, total, expected_reason in (
            (0, 800, ReasonCode.WITHIN_BUDGET),
            (0, 801, ReasonCode.OVER_MAX_BUDGET),
            (20, 800, ReasonCode.WITHIN_BUDGET),
            (20, 801, ReasonCode.OVER_PREFERRED_BUDGET),
            (20, 960, ReasonCode.OVER_PREFERRED_BUDGET),
            (20, 961, ReasonCode.OVER_MAX_BUDGET),
        ):
            with self.subTest(flexibility=flexibility, total=total):
                request = TripRequest(
                    origin="Rome", budget_eur=800, budget_flexibility_pct=flexibility,
                    destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
                    time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                )
                flight, hotel = fake_flight_and_hotel(total)
                with patch("planning.search_flights", return_value=[flight]), patch(
                    "planning.search_hotels", return_value=[hotel]
                ):
                    result = self.build_proposal(request)

                if expected_reason == ReasonCode.OVER_MAX_BUDGET:
                    self.assertIsInstance(result, PlanningError)
                    self.assertEqual(result.code, ReasonCode.NO_AFFORDABLE_PROPOSAL.value)
                    self.assertEqual(result.trace.events[-1].reason_code, expected_reason)
                else:
                    self.assertNotIsInstance(result, PlanningError)
                    self.assertEqual(len(result.proposals), 1)
                    proposal = result.proposals[0]
                    self.assertEqual(proposal.total_price_eur, total)
                    budget_comments = [
                        comment for comment in proposal.comments
                        if comment.reason_code == expected_reason
                    ]
                    self.assertEqual(len(budget_comments), 1)
                    self.assertEqual(budget_comments[0].details["budget_eur"], 800)
                    self.assertEqual(
                        budget_comments[0].details["budget_ceiling_eur"],
                        800 if flexibility == 0 else 960,
                    )

    
    def test_budget_boundary_preserves_cents_in_proposals_and_trace(self):
        request = TripRequest(
            origin="Rome", budget_eur=999, budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        for total in (Decimal("1198.80"), Decimal("1198.81")):
            with self.subTest(total=total):
                flight, hotel = fake_flight_and_hotel(total)
                with patch("planning.search_flights", return_value=[flight]), patch(
                    "planning.search_hotels", return_value=[hotel]
                ):
                    result = self.build_proposal(request)

                if total > Decimal("1198.80"):
                    self.assertIsInstance(result, PlanningError)
                    self.assertEqual(result.code, ReasonCode.NO_AFFORDABLE_PROPOSAL.value)
                    entries = [result.trace.events[-1]]
                else:
                    self.assertNotIsInstance(result, PlanningError)
                    proposal = result.proposals[0]
                    self.assertIsInstance(proposal.flight.price_eur, Decimal)
                    self.assertIsInstance(proposal.hotel.total_price_eur, Decimal)
                    self.assertIsInstance(proposal.total_price_eur, Decimal)
                    self.assertEqual(proposal.total_price_eur, total)
                    self.assertIsInstance(score_proposal(proposal, request)[-1], Decimal)
                    payload = json.loads(result.model_dump_json())
                    self.assertEqual(payload["proposals"][0]["total_price_eur"], "1198.80")
                    entries = [
                        event for event in result.trace.events
                        if event.event_type == EventType.COMBINATION_ACCEPTED
                    ] + [
                        comment for comment in proposal.comments
                        if comment.reason_code == ReasonCode.OVER_PREFERRED_BUDGET
                    ]
                    self.assertEqual(len(entries), 2)

                for entry in entries:
                    self.assertIsInstance(entry.details["total_price_eur"], Decimal)
                    self.assertEqual(entry.details["total_price_eur"], total)
                    self.assertIsInstance(entry.details["budget_ceiling_eur"], Decimal)
                    self.assertEqual(entry.details["budget_ceiling_eur"], Decimal("1198.80"))
                    serialized = json.loads(entry.model_dump_json())
                    self.assertEqual(serialized["details"]["total_price_eur"], str(total))


    def test_accepts_proposal_within_budget_flexibility(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=400,
            budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                comment.reason_code == ReasonCode.OVER_PREFERRED_BUDGET
                for proposal in result.proposals
                for comment in proposal.comments
            )
        )

    
    def test_rejects_proposals_over_budget_ceiling(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=300,
            budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                event.reason_code == ReasonCode.OVER_MAX_BUDGET
                for event in result.trace.events
            )
        )


    def test_no_flights_returns_specific_error(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Kyoto"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        with patch("planning.search_hotels") as hotels:
            result = self.build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.NO_FLIGHTS_FOUND.value)
        self.assertIn("Nessun volo", result.message)
        event = result.trace.events[-1]
        self.assertEqual(event.event_type, EventType.SEARCH_FAILED)
        self.assertEqual(event.reason_code, ReasonCode.NO_FLIGHTS_FOUND)
        self.assertEqual(event.details, {
            "destination": "Kyoto", "start": "2026-09-15", "end": "2026-09-20",
            "serpapi_calls": 0, "llm_calls": 0, **self.limits.model_dump(),
        })
        hotels.assert_not_called()


    def test_no_hotels_returns_specific_error(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        with patch("planning.search_hotels", return_value=[]):
            result = self.build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.NO_HOTELS_FOUND.value)
        self.assertIn("Nessun hotel", result.message)
        failures = [
            event for event in result.trace.events
            if event.event_type == EventType.SEARCH_FAILED
        ]
        self.assertTrue(failures)
        for event in failures:
            self.assertEqual(event.reason_code, ReasonCode.NO_HOTELS_FOUND)
            self.assertEqual({key: event.details[key] for key in ("destination", "start", "end")}, {
                "destination": "Lisbon", "start": "2026-09-15", "end": "2026-09-20",
            })
        self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 0)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 0)


    def test_over_budget_inventory_is_not_reported_as_missing(self):
        for budget in (100, 200):
            with self.subTest(budget=budget):
                request = TripRequest(
                    origin="Rome", budget_eur=budget,
                    destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
                    time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                )

                result = self.build_proposal(request)

                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, ReasonCode.NO_AFFORDABLE_PROPOSAL.value)
                self.assertIn("budget massimo", result.message)
                reasons = [event.reason_code for event in result.trace.events]
                self.assertIn(ReasonCode.OVER_MAX_BUDGET, reasons)
                self.assertNotIn(ReasonCode.NO_FLIGHTS_FOUND, reasons)
                self.assertNotIn(ReasonCode.NO_HOTELS_FOUND, reasons)


    def test_search_continues_after_period_without_offers(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        advice = json.loads(fake_travel_window_response())
        later_advice = json.loads(fake_travel_window_response(
            start="2026-10-06", end="2026-10-11",
        ))
        advice["selected_periods"].extend(later_advice["selected_periods"])
        self.fake_openrouter_call.return_value = json.dumps(advice)

        for missing_step, reason in (
            ("flights", ReasonCode.NO_FLIGHTS_FOUND),
            ("hotels", ReasonCode.NO_HOTELS_FOUND),
        ):
            with self.subTest(missing_step=missing_step):
                def available_flights(query, *, max_outbound=3, budget=None):
                    if missing_step == "flights" and query.departure_date == date(2026, 9, 15):
                        return []
                    return fake_flight_search(query)

                def available_hotels(query, *, budget=None):
                    if missing_step == "hotels" and query.checkin_date == date(2026, 9, 15):
                        return []
                    return fake_hotel_search(query)

                with patch("planning.search_flights", side_effect=available_flights), patch(
                    "planning.search_hotels", side_effect=available_hotels
                ):
                    result = self.build_proposal(request)

                self.assertNotIsInstance(result, PlanningError)
                self.assertTrue(result.proposals)
                for proposal in result.proposals:
                    self.assertEqual(proposal.period.start, date(2026, 10, 6))
                failures = [event for event in result.trace.events if event.reason_code == reason]
                self.assertTrue(failures)
                for event in failures:
                    self.assertEqual(event.details["destination"], "Lisbon")
                    self.assertEqual(event.details["start"], "2026-09-15")


    def test_open_destination_classification_uses_all_candidates(self):
        lisbon = json.loads(self.fake_destination_call.return_value)["destinations"][0]
        kyoto = dict(lisbon, name="Kyoto", country="Japan")
        for destinations in ([kyoto, lisbon], [lisbon, kyoto]):
            for budget in (100, 900):
                with self.subTest(first=destinations[0]["name"], budget=budget):
                    self.fake_destination_call.return_value = json.dumps({
                        "destinations": destinations,
                    })
                    request = TripRequest(
                        origin="Rome", budget_eur=budget,
                        destination_pref=DestinationPreference(mode="open"),
                        time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
                    )

                    result = self.build_proposal(request)

                    if budget == 100:
                        self.assertIsInstance(result, PlanningError)
                        self.assertEqual(result.code, ReasonCode.NO_AFFORDABLE_PROPOSAL.value)
                    else:
                        self.assertNotIsInstance(result, PlanningError)
                        self.assertTrue(result.proposals)
                        for proposal in result.proposals:
                            self.assertEqual(proposal.destination.name, "Lisbon")
                    failures = [
                        event for event in result.trace.events
                        if event.reason_code == ReasonCode.NO_FLIGHTS_FOUND
                    ]
                    self.assertEqual(len(failures), 1)
                    self.assertEqual(failures[0].details["destination"], "Kyoto")


    def test_proposals_are_sorted_by_score(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)

        scores = [
            score_proposal(proposal, request)
            for proposal in result.proposals
        ]

        self.assertEqual(scores, sorted(scores, reverse=True))


    def test_scoring_distinguishes_preferred_and_absent_budget(self):
        request = TripRequest(
            origin="Rome", budget_eur=800, budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )
        result = self.build_proposal(request)
        self.assertNotIsInstance(result, PlanningError)
        base = result.proposals[0]

        def proposal_with(total, match_score):
            total = Decimal(total)
            return base.model_copy(update={
                "destination": base.destination.model_copy(update={"match_score": match_score}),
                "flight": base.flight.model_copy(update={
                    "price_eur": total - base.hotel.total_price_eur,
                }),
                "total_price_eur": total,
            })

        within_budget = proposal_with(800, 60)
        better_match = proposal_with(900, 95)
        for budget, expected in (
            (800, [within_budget, better_match]),
            (None, [better_match, within_budget]),
        ):
            with self.subTest(budget=budget):
                current_request = request.model_copy(update={"budget_eur": budget})
                self.assertEqual(
                    select_top_proposals([better_match, within_budget], current_request),
                    expected,
                )
                self.assertEqual(
                    select_top_proposals([better_match, within_budget], current_request, limit=1),
                    expected[:1],
                )

        cheaper_same_match = proposal_with(800, 95)
        no_budget = request.model_copy(update={"budget_eur": None})
        self.assertEqual(
            select_top_proposals([better_match, cheaper_same_match], no_budget),
            [cheaper_same_match, better_match],
        )


    def test_workflow_prioritizes_proposals_within_preferred_budget(self):
        request = TripRequest(
            origin="Rome", budget_eur=550, budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(
            [proposal.total_price_eur <= request.budget_eur for proposal in result.proposals],
            [True, True, False],
        )
        self.assertLess(result.proposals[0].hotel.rating, result.proposals[-1].hotel.rating)
    

    def test_travel_window_exact_dates_uses_user_dates(self):
        start = date(2026, 9, 15)
        end = date(2026, 9, 20)

        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start=start.isoformat(),
            end=end.isoformat(),
            score=100,
            rationale="Date esatte rispettate.",
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(
                mode="exact_dates",
                start_date=start,
                end_date=end,
            ),
        )
        destination = SuggestedDestination(
            name="Lisbon",
            country="Portugal",
            reason="Destinazione test.",
            match_score=100,
        )

        advice = evaluate_travel_window(request, destination)

        self.assertEqual(advice.selected_periods[0].start, start)
        self.assertEqual(advice.selected_periods[0].end, end)
        self.assertEqual(advice.selected_periods[0].score, 100)


    def test_trace_contains_travel_window_evaluation(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        travel_window_events = [
            event
            for event in result.trace.events
            if event.event_type == EventType.TRAVEL_WINDOW_EVALUATED
        ]

        self.assertGreater(len(travel_window_events), 0)
        self.assertEqual(travel_window_events[0].details["destination"], "Lisbon")

        self.assertNotIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                event.event_type == EventType.TRAVEL_WINDOW_EVALUATED
                for event in result.trace.events
            )
        )


    def test_travel_window_trace_includes_destination(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = self.build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)

        travel_window_events = [
            event
            for event in result.trace.events
            if event.event_type == EventType.TRAVEL_WINDOW_EVALUATED
        ]

        self.assertEqual(len(travel_window_events), 1)
        self.assertEqual(travel_window_events[0].details["destination"], "Lisbon")
    

    def test_travel_window_uses_llm_response_without_calling_openrouter(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-09-15",
            end="2026-09-20",
            score=90,
            rationale="Settembre è una buona finestra di viaggio.",
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(
                mode="fixed",
                destination="Lisbon",
            ),
            time_pref=TimePreference(
                mode="flexible_window",
                min_days=4,
                max_days=6,
            ),
            vibe=TravelVibe(themes=["food"]),
        )

        destination = SuggestedDestination(
            name="Lisbon",
            country="Portugal",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        advice = evaluate_travel_window(request, destination)

        self.assertEqual(advice.selected_periods[0].start, date(2026, 9, 15))
        self.assertEqual(advice.selected_periods[0].end, date(2026, 9, 20))
        self.assertEqual(advice.selected_periods[0].score, 90)
        self.assertEqual(advice.rationale, "Settembre è una buona finestra di viaggio.")

        self.fake_openrouter_call.assert_called_once()


    def test_travel_window_rejects_invalid_llm_json(self):
        self.fake_openrouter_call.return_value = "not valid json"

        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        destination = SuggestedDestination(
            name="Lisbon",
            country="Portugal",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        with self.assertRaises(ValidationError):
            evaluate_travel_window(request, destination)


    def test_travel_window_rejects_llm_dates_that_violate_exact_dates(self):
        start = date(2026, 9, 15)
        end = date(2026, 9, 20)

        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-10-06",
            end="2026-10-11",
            score=90,
            rationale="Periodo alternativo proposto.",
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(
                mode="exact_dates",
                start_date=start,
                end_date=end,
            ),
        )

        destination = SuggestedDestination(
            name="Lisbon",
            country="Portugal",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        with self.assertRaises(ValueError):
            evaluate_travel_window(request, destination)


    def assert_build_proposal_llm_failure(self, expected_reason):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(
                mode="exact_dates",
                start_date=date(2026, 9, 15),
                end_date=date(2026, 9, 20),
            ),
        )

        with patch("planning.search_flights") as fake_flights, patch(
            "planning.search_hotels"
        ) as fake_hotels:
            result = self.build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, expected_reason.value)
        failure_events = [
            event for event in result.trace.events
            if event.event_type == EventType.TRAVEL_WINDOW_FAILED
        ]
        self.assertEqual(len(failure_events), 1)
        self.assertEqual(failure_events[0].reason_code, expected_reason)
        self.assertEqual(failure_events[0].details["destination"], "Lisbon")
        event_types = [event.event_type for event in result.trace.events]
        self.assertIn(EventType.REQUEST_ROUTED, event_types)
        self.assertIn(EventType.DESTINATIONS_RESOLVED, event_types)
        self.assertNotIn(EventType.TRAVEL_WINDOW_EVALUATED, event_types)
        self.fake_openrouter_call.assert_called_once()
        fake_flights.assert_not_called()
        fake_hotels.assert_not_called()


    def test_build_proposal_handles_llm_provider_error(self):
        self.fake_openrouter_call.side_effect = OpenRouterError("Timeout simulato")

        self.assert_build_proposal_llm_failure(ReasonCode.LLM_PROVIDER_ERROR)


    def test_build_proposal_handles_invalid_llm_json(self):
        self.fake_openrouter_call.return_value = "not valid json"

        self.assert_build_proposal_llm_failure(ReasonCode.LLM_OUTPUT_INVALID)


    def test_build_proposal_handles_llm_exact_date_violation(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-10-06",
            end="2026-10-11",
        )

        self.assert_build_proposal_llm_failure(ReasonCode.LLM_CONSTRAINT_VIOLATION)


    def test_resolve_month_year_uses_current_year_for_future_month(self):
        year = resolve_month_year(
            month=8,
            reference_date=date(2026, 6, 7),
        )

        self.assertEqual(year, 2026)


    def test_resolve_month_year_uses_next_year_for_past_month(self):
        year = resolve_month_year(
            month=4,
            reference_date=date(2026, 6, 7),
        )

        self.assertEqual(year, 2027)


    def test_travel_window_accepts_resolved_year_for_month_mode(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-08-01",
            end="2026-08-14",
            score=70,
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=1200,
            destination_pref=DestinationPreference(mode="fixed", destination="Japan"),
            time_pref=TimePreference(mode="month", month=8, min_days=10, max_days=14),
        )

        destination = SuggestedDestination(
            name="Japan",
            country="Japan",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        advice = evaluate_travel_window(
            request,
            destination,
            reference_date=date(2026, 6, 7),
        )

        self.assertEqual(advice.selected_periods[0].start.year, 2026)
        self.assertEqual(advice.selected_periods[0].start.month, 8)


    def test_travel_window_rejects_wrong_year_for_month_mode(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2025-08-01",
            end="2025-08-14",
            score=70,
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=1200,
            destination_pref=DestinationPreference(mode="fixed", destination="Japan"),
            time_pref=TimePreference(mode="month", month=8, min_days=10, max_days=14),
        )

        destination = SuggestedDestination(
            name="Japan",
            country="Japan",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        with self.assertRaises(ValueError):
            evaluate_travel_window(
                request,
                destination,
                reference_date=date(2026, 6, 7),
            )


    def test_travel_window_rejects_period_shorter_than_min_days(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-08-01",
            end="2026-08-05",
            score=70,
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=1200,
            destination_pref=DestinationPreference(mode="fixed", destination="Japan"),
            time_pref=TimePreference(mode="month", month=8, min_days=10, max_days=14),
        )

        destination = SuggestedDestination(
            name="Japan",
            country="Japan",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        with self.assertRaises(ValueError):
            evaluate_travel_window(
                request,
                destination,
                reference_date=date(2026, 6, 7),
            )


    def test_travel_window_rejects_period_longer_than_max_days(self):
        self.fake_openrouter_call.return_value = fake_travel_window_response(
            start="2026-08-01",
            end="2026-08-20",
            score=70,
        )

        request = TripRequest(
            origin="Rome",
            budget_eur=1200,
            destination_pref=DestinationPreference(mode="fixed", destination="Japan"),
            time_pref=TimePreference(mode="month", month=8, min_days=10, max_days=14),
        )

        destination = SuggestedDestination(
            name="Japan",
            country="Japan",
            reason="Destinazione scelta dall'utente.",
            match_score=100,
        )

        with self.assertRaises(ValueError):
            evaluate_travel_window(
                request,
                destination,
                reference_date=date(2026, 6, 7),
            )


    def test_hotel_nights_prices_and_budget(self):
        def search(checkout_day, budget=None):
            query = HotelSearchQuery(
                destination="Lisbon",
                checkin_date=date(2026, 9, 15),
                checkout_date=date(2026, 9, checkout_day),
                max_total_price_eur=budget,
            )
            totals = {18: (285, 186), 22: (665, 434)}[checkout_day]
            response = {"properties": [
                raw_hotel(name="Central Stay", total=totals[0]),
                raw_hotel(name="Simple Rooms", total=totals[1]),
            ]}
            with patch("providers.hotels.search_serpapi", return_value=response):
                return search_hotels(query)

        first = search(18)
        second = search(22)

        for offers, nights, price in [(first, 3, 285), (second, 7, 665)]:
            self.assertTrue(offers, "La ricerca deve trovare hotel.")
            for hotel in offers:
                self.assertEqual(hotel.nights, nights)
            by_name = {hotel.name: hotel for hotel in offers}
            self.assertIn("Central Stay", by_name)
            self.assertEqual(by_name["Central Stay"].total_price_eur, price)

        limited = search(22, budget=500)
        self.assertEqual(
            [(hotel.name, hotel.total_price_eur) for hotel in limited],
            [("Simple Rooms", 434)],
        )


    def test_hotels_reject_invalid_dates(self):
        for checkout_day in (15, 14):
            with self.subTest(checkout_day=checkout_day):
                with self.assertRaises(ValidationError):
                    HotelSearchQuery(
                        destination="Lisbon",
                        checkin_date=date(2026, 9, 15),
                        checkout_date=date(2026, 9, checkout_day),
                    )


    def test_travel_window_counts_inclusive_days(self):
        request = TripRequest(
            origin="Rome", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=6, max_days=6),
        )
        destination = SuggestedDestination(name="Lisbon", reason="Test", match_score=100)
        self.fake_openrouter_call.return_value = fake_travel_window_response(end="2026-09-20")
        advice = evaluate_travel_window(request, destination)
        self.assertEqual(advice.selected_periods[0].end, date(2026, 9, 20))

        self.fake_openrouter_call.return_value = fake_travel_window_response(end="2026-09-21")
        with self.assertRaisesRegex(ValueError, "durata massima"):
            evaluate_travel_window(request, destination)


    def flight_request(self, **fields):
        return TripRequest(**{
            "origin": "Rome",
            "budget_eur": 900,
            "destination_pref": DestinationPreference(mode="fixed", destination="Lisbon"),
            "time_pref": TimePreference(mode="flexible_window", min_days=4, max_days=6),
            **fields,
        })

    def test_flight_query_uses_city_ids_and_reuses_resolution_across_periods(self):
        advice = json.loads(fake_travel_window_response())
        advice["selected_periods"].extend(json.loads(fake_travel_window_response(
            start="2026-10-06", end="2026-10-11",
        ))["selected_periods"])
        self.fake_openrouter_call.return_value = json.dumps(advice)
        result = self.build_proposal(self.flight_request())
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(self.fake_resolve_city.call_count, 2)
        self.assertEqual(self.fake_flight_search.call_count, 2)
        for call in self.fake_flight_search.call_args_list:
            query = call.args[0]
            self.assertEqual(query.origin_id, "/m/test_rome")
            self.assertEqual(query.destination_id, "/m/test_lisbon")

    def test_workflow_uses_real_flight_pipeline_with_offline_api_responses(self):
        responses = [
            {"suggestions": [fake_city("Rome, Italy")]},
            {"suggestions": [fake_city("Lisbon, Portugal")]},
            {"best_flights": [raw_leg(token="round-trip-token")]},
            {"other_flights": [raw_leg("LIS", "FCO", date(2026, 9, 20), price="249.99")]},
        ]
        with patch("planning.resolve_city", resolve_city), patch(
            "planning.search_flights", search_flights
        ), patch("providers.flights.search_serpapi", side_effect=responses) as api:
            result = self.build_proposal(self.flight_request())
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(api.call_count, 4)
        self.assertEqual(api.call_args_list[2].args[0]["departure_id"], "/m/test_rome")
        self.assertEqual(api.call_args_list[2].args[0]["arrival_id"], "/m/test_lisbon")
        self.assertEqual(result.proposals[0].flight.price_eur, Decimal("249.99"))
        self.assertEqual(result.proposals[0].total_price_eur, Decimal("724.99"))

    def test_airports_only_restrict_queries_when_explicit(self):
        result = self.build_proposal(self.flight_request(
            origin_city_id="/m/chosen_rome",
            origin_airport_code="FCO",
            destination_pref=DestinationPreference(
                mode="fixed", destination="Lisbon", city_id="/m/chosen_lisbon", airport_code="LIS",
            ),
        ))
        self.assertNotIsInstance(result, PlanningError)
        self.fake_resolve_city.assert_any_call("Rome", city_id="/m/chosen_rome", budget=ANY)
        self.fake_resolve_city.assert_any_call("Lisbon", city_id="/m/chosen_lisbon", budget=ANY)
        query = self.fake_flight_search.call_args.args[0]
        self.assertEqual((query.origin_id, query.destination_id), ("FCO", "LIS"))

    def test_unverified_airport_is_not_silently_ignored(self):
        result = self.build_proposal(self.flight_request(origin_airport_code="LIN"))
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.FLIGHT_LOCATION_UNRESOLVED.value)
        self.fake_flight_search.assert_not_called()

    def test_ambiguous_city_returns_options_without_searching_flights(self):
        self.fake_resolve_city.side_effect = FlightLocationError(
            "Citta ambigua", options=["Rome, Italy (/m/test_rome)", "Rome, Georgia (/m/test_other)"],
        )
        result = self.build_proposal(self.flight_request())
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.FLIGHT_LOCATION_UNRESOLVED.value)
        self.assertEqual(len(result.trace.events[-1].details["location_options"]), 2)
        self.fake_flight_search.assert_not_called()

    def test_flight_provider_failures_are_not_missing_inventory(self):
        for error, reason in (
            (SerpAPIError("Quota"), ReasonCode.FLIGHT_PROVIDER_ERROR),
            (SerpAPIResponseError("Price"), ReasonCode.FLIGHT_OUTPUT_INVALID),
        ):
            with self.subTest(reason=reason):
                self.fake_flight_search.side_effect = error
                with patch("planning.search_hotels") as hotels:
                    result = self.build_proposal(self.flight_request())
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, reason.value)
                self.assertEqual(result.trace.events[-1].reason_code, reason)
                hotels.assert_not_called()

    def test_city_lookup_outage_has_provider_error_code(self):
        self.fake_resolve_city.side_effect = SerpAPIError("Timeout")
        result = self.build_proposal(self.flight_request())
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.FLIGHT_PROVIDER_ERROR.value)
        self.fake_flight_search.assert_not_called()

    def test_hotel_dates_follow_actual_arrival_and_same_stay_is_reused(self):
        flight = make_flight()
        flight.outbound.segments[-1].arrival_local += timedelta(days=1)
        self.fake_flight_search.side_effect = None
        self.fake_flight_search.return_value = [flight, flight.model_copy(update={"price_eur": 200})]
        with patch("planning.search_hotels", wraps=fake_hotel_search) as hotels:
            result = self.build_proposal(self.flight_request())
        self.assertNotIsInstance(result, PlanningError)
        hotels.assert_called_once()
        query = hotels.call_args.args[0]
        self.assertEqual(query.checkin_date, date(2026, 9, 16))
        self.assertEqual(query.checkout_date, date(2026, 9, 20))
        self.assertTrue(all(proposal.hotel.nights == 4 for proposal in result.proposals))

    def test_day_only_stay_does_not_call_hotel_provider(self):
        flight = make_flight()
        flight.outbound.segments[-1].arrival_local += timedelta(days=5)
        self.fake_flight_search.side_effect = None
        self.fake_flight_search.return_value = [flight]
        with patch("planning.search_hotels") as hotels:
            result = self.build_proposal(self.flight_request())
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, ReasonCode.NO_HOTELS_FOUND.value)
        hotels.assert_not_called()

    def test_hotel_query_uses_verified_city_and_same_guests_as_flights(self):
        def canonical_city(text, *, city_id=None, budget=None):
            name = {"Rome": "Rome, Italy", "Lisbon": "Lisbon, Portugal"}.get(text, text)
            return fake_city(name, city_id=city_id)

        self.fake_resolve_city.side_effect = canonical_city
        result = self.build_proposal(self.flight_request(adults=2))
        self.assertNotIsInstance(result, PlanningError)
        hotel_query = self.fake_hotel_search.call_args.args[0]
        self.assertEqual(hotel_query.destination, "Lisbon")
        self.assertEqual(hotel_query.location_name, "Lisbon, Portugal")
        self.assertEqual(hotel_query.adults, 2)
        self.assertEqual(self.fake_flight_search.call_args.args[0].adults, 2)
        self.assertIsNone(hotel_query.max_total_price_eur)

    def test_empty_hotel_search_is_cached_for_identical_stays(self):
        self.fake_hotel_search.side_effect = None
        self.fake_hotel_search.return_value = []
        result = self.build_proposal(self.flight_request())
        self.assertEqual(result.code, ReasonCode.NO_HOTELS_FOUND.value)
        self.fake_hotel_search.assert_called_once()

    def test_different_arrivals_produce_separate_hotel_searches(self):
        first, second = make_flight(), make_flight()
        second.outbound.segments[-1].arrival_local += timedelta(days=1)
        self.fake_flight_search.side_effect = None
        self.fake_flight_search.return_value = [first, second]
        result = self.build_proposal(self.flight_request())
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(self.fake_hotel_search.call_count, 2)
        self.assertEqual(
            {call.args[0].checkin_date for call in self.fake_hotel_search.call_args_list},
            {date(2026, 9, 15), date(2026, 9, 16)},
        )

    def test_hotel_cache_is_local_to_one_planning_run(self):
        for adults in (1, 2, 2):
            result = self.build_proposal(self.flight_request(adults=adults))
            self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(self.fake_hotel_search.call_count, 3)
        self.assertEqual([call.args[0].adults for call in self.fake_hotel_search.call_args_list], [1, 2, 2])

    def test_same_city_name_in_different_countries_does_not_share_hotels(self):
        destinations = [
            SuggestedDestination(name="Springfield", country=country, reason="Test", match_score=80)
            for country in ("United States", "Australia")
        ]
        flight, hotel = fake_flight_and_hotel(500)
        self.fake_flight_search.side_effect = lambda query, **kwargs: [flight.model_copy(update={"destination": query.destination})]
        self.fake_hotel_search.side_effect = lambda query, **kwargs: [hotel.model_copy(update={"destination": query.destination})]
        with patch("planning.resolve_destinations", return_value=destinations):
            result = self.build_proposal(self.flight_request(destination_pref=DestinationPreference(mode="open")))
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(self.fake_hotel_search.call_count, 2)
        self.assertEqual(
            {call.args[0].location_name for call in self.fake_hotel_search.call_args_list},
            {"Springfield, United States", "Springfield, Australia"},
        )

    def test_missing_rating_is_ranked_after_known_rating_without_false_comments(self):
        _, hotel = fake_flight_and_hotel(500)
        self.fake_flight_search.side_effect = None
        self.fake_flight_search.return_value = [make_flight()]
        self.fake_hotel_search.side_effect = None
        self.fake_hotel_search.return_value = [
            hotel.model_copy(update={"name": "Unrated", "rating": None}),
            hotel.model_copy(update={"name": "Rated", "rating": 0.0}),
        ]
        result = self.build_proposal(self.flight_request())
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual([proposal.hotel.name for proposal in result.proposals], ["Rated", "Unrated"])
        self.assertIsNone(result.proposals[-1].hotel.rating)
        for proposal in result.proposals:
            self.assertNotIn(ReasonCode.GOOD_HOTEL_RATING, [comment.reason_code for comment in proposal.comments])

    def test_hotel_provider_errors_keep_context_and_stop_planning(self):
        for error, reason in (
            (SerpAPIError("Sensitive raw message", code="TIMEOUT"), ReasonCode.HOTEL_PROVIDER_ERROR),
            (SerpAPIError("Sensitive raw message", code="QUOTA_OR_RATE_LIMIT", status_code=429), ReasonCode.HOTEL_PROVIDER_ERROR),
            (SerpAPIResponseError("Sensitive raw message"), ReasonCode.HOTEL_OUTPUT_INVALID),
        ):
            with self.subTest(reason=reason, error=error.code):
                self.fake_hotel_search.reset_mock()
                self.fake_hotel_search.side_effect = error
                result = self.build_proposal(self.flight_request())
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, reason.value)
                event = result.trace.events[-1]
                self.assertEqual(event.reason_code, reason)
                self.assertEqual(event.details["provider_error"], error.code)
                self.assertEqual(event.details["http_status"], error.status_code)
                self.assertEqual(event.details["checkin_date"], "2026-09-15")
                self.assertEqual(event.details["checkout_date"], "2026-09-20")
                self.assertNotIn("Sensitive raw message", result.model_dump_json())
                self.assertNotIn(ReasonCode.NO_HOTELS_FOUND, [item.reason_code for item in result.trace.events])
                self.fake_hotel_search.assert_called_once()

    def test_hotel_failure_after_an_accepted_proposal_is_not_silently_ignored(self):
        first, second = make_flight(), make_flight()
        second.outbound.segments[-1].arrival_local += timedelta(days=1)
        self.fake_flight_search.side_effect = None
        self.fake_flight_search.return_value = [first, second]
        _, hotel = fake_flight_and_hotel(500)
        self.fake_hotel_search.side_effect = [[hotel], SerpAPIError("Timeout", code="TIMEOUT")]
        result = self.build_proposal(self.flight_request())
        self.assertEqual(result.code, ReasonCode.HOTEL_PROVIDER_ERROR.value)
        self.assertIn(EventType.COMBINATION_ACCEPTED, [item.event_type for item in result.trace.events])
        self.assertNotIn(EventType.PROPOSALS_SELECTED, [item.event_type for item in result.trace.events])

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"})
    def test_hotel_http_errors_reach_planning_error_through_the_real_provider(self):
        for status, code in ((401, "AUTHENTICATION"), (429, "QUOTA_OR_RATE_LIMIT"), (500, "HTTP_ERROR")):
            with self.subTest(status=status):
                response = Mock(status_code=status)
                with patch("planning.search_hotels", search_hotels), patch(
                    "providers.serpapi_client.requests.get", return_value=response
                ) as get:
                    result = self.build_proposal(self.flight_request())
                self.assertEqual(result.code, ReasonCode.HOTEL_PROVIDER_ERROR.value)
                self.assertEqual(result.trace.events[-1].details["provider_error"], code)
                self.assertEqual(result.trace.events[-1].details["http_status"], status)
                get.assert_called_once()

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"})
    def test_hotel_timeout_and_invalid_json_reach_planning_error(self):
        with patch("planning.search_hotels", search_hotels), patch(
            "providers.serpapi_client.requests.get", side_effect=Timeout("URL with secret")
        ):
            result = self.build_proposal(self.flight_request())
        self.assertEqual(result.code, ReasonCode.HOTEL_PROVIDER_ERROR.value)
        self.assertEqual(result.trace.events[-1].details["provider_error"], "TIMEOUT")
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("Invalid JSON")
        with patch("planning.search_hotels", search_hotels), patch(
            "providers.serpapi_client.requests.get", return_value=response
        ):
            result = self.build_proposal(self.flight_request())
        self.assertEqual(result.code, ReasonCode.HOTEL_OUTPUT_INVALID.value)

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"})
    def test_hotel_failed_search_status_is_not_confused_with_empty_inventory(self):
        for status, expected in (("Error", ReasonCode.HOTEL_PROVIDER_ERROR), ("Success", ReasonCode.NO_HOTELS_FOUND)):
            with self.subTest(status=status):
                response = Mock(status_code=200)
                response.json.return_value = {"search_metadata": {"status": status}, "error": "Message from service"}
                with patch("planning.search_hotels", search_hotels), patch(
                    "providers.serpapi_client.requests.get", return_value=response
                ):
                    result = self.build_proposal(self.flight_request())
                self.assertEqual(result.code, expected.value)

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"})
    def test_full_provider_pipeline_preserves_prices_dates_and_redirects(self):
        outbound = raw_leg(token="test-outbound")
        outbound["flights"][-1]["arrival_airport"]["time"] = "2026-09-16 01:00:00"
        outbound["total_duration"] = 900
        payloads = [
            {"suggestions": [fake_city("Rome, Italy")]},
            {"suggestions": [fake_city("Lisbon, Portugal")]},
            {"best_flights": [outbound]},
            {"other_flights": [raw_leg("LIS", "FCO", date(2026, 9, 20), price="249.99")]},
            {"properties": [raw_hotel(total="249.99", nightly="62.50")]},
        ]
        responses = []
        for payload in payloads:
            response = Mock(status_code=200)
            response.json.return_value = {
                "search_metadata": {"status": "Success", "google_flights_url": "https://www.google.com/travel/flights?tfs=test"},
                **payload,
            }
            responses.append(response)
        with patch("planning.resolve_city", resolve_city), patch(
            "planning.search_flights", search_flights
        ), patch("planning.search_hotels", search_hotels), patch(
            "providers.serpapi_client.requests.get", side_effect=responses
        ) as get:
            result = self.build_proposal(self.flight_request(budget_eur=500, budget_flexibility_pct=0, adults=2))
        self.assertNotIsInstance(result, PlanningError)
        self.assertEqual(get.call_count, 5)
        proposal = result.proposals[0]
        self.assertEqual(proposal.flight.price_eur, Decimal("249.99"))
        self.assertEqual(proposal.hotel.total_price_eur, Decimal("249.99"))
        self.assertEqual(proposal.total_price_eur, Decimal("499.98"))
        self.assertEqual(proposal.hotel.nights, 4)
        self.assertEqual(proposal.flight.redirect.kind, "search")
        self.assertEqual(proposal.hotel.redirect.kind, "property_website")
        params = get.call_args_list[-1].kwargs["params"]
        self.assertEqual(params["q"], "Hotels in Lisbon, Portugal")
        self.assertEqual(params["check_in_date"], "2026-09-16")
        self.assertEqual(params["check_out_date"], "2026-09-20")
        self.assertEqual(params["adults"], 2)


class HotelOfferTest(unittest.TestCase):
    def setUp(self):
        self.query = HotelSearchQuery(
            destination="Lisbon",
            checkin_date=date(2026, 9, 15),
            checkout_date=date(2026, 9, 20),
        )
        self.raw_hotel = {
            "name": "Test Stay",
            "rate_per_night": {"extracted_lowest": 50.01},
            "total_rate": {"extracted_lowest": 249.99},
            "overall_rating": 4.4,
        }

    def test_total_is_explicit_and_not_recalculated(self):
        fields = dict(
            destination="Lisbon", name="Test Stay",
            nightly_price_eur="50.00", nights=5, rating=4.4,
        )
        with self.assertRaises(ValidationError):
            HotelOffer(**fields)
        hotel = HotelOffer(**fields, total_price_eur="249.99")
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))
        self.assertNotEqual(hotel.total_price_eur, hotel.nightly_price_eur * hotel.nights)

    def test_parser_preserves_prices_and_derives_nights(self):
        hotel = parse_hotel_offer(self.raw_hotel, self.query)
        self.assertEqual(hotel.destination, self.query.destination)
        self.assertEqual(hotel.name, "Test Stay")
        self.assertEqual(hotel.nights, 5)
        self.assertEqual(hotel.rating, 4.4)
        self.assertIsInstance(hotel.nightly_price_eur, Decimal)
        self.assertIsInstance(hotel.total_price_eur, Decimal)
        self.assertEqual(hotel.nightly_price_eur, Decimal("50.01"))
        self.assertEqual(hotel.total_price_eur, Decimal("249.99"))

    def test_parser_rejects_missing_total(self):
        cases = [
            {}, {"total_rate": None}, {"total_rate": {}},
            {"total_rate": {"extracted_lowest": None}}, {"total_rate": []},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                raw_hotel = {key: value for key, value in self.raw_hotel.items() if key != "total_rate"}
                raw_hotel.update(fields)
                with self.assertRaisesRegex(SerpAPIResponseError, "prezzo totale"):
                    parse_hotel_offer(raw_hotel, self.query)

    def test_parser_rejects_negative_or_nonfinite_prices(self):
        for field in ("rate_per_night", "total_rate"):
            for value in (-1, "NaN", "Infinity"):
                with self.subTest(field=field, value=value):
                    raw_hotel = {**self.raw_hotel, field: {"extracted_lowest": value}}
                    with self.assertRaises(SerpAPIResponseError):
                        parse_hotel_offer(raw_hotel, self.query)

    def test_parser_rejects_invalid_stay_dates(self):
        for checkout_day in (14, 15):
            with self.subTest(checkout_day=checkout_day):
                query = self.query.model_copy(update={"checkout_date": date(2026, 9, checkout_day)})
                with self.assertRaises(SerpAPIResponseError):
                    parse_hotel_offer(self.raw_hotel, query)


@patch.dict("os.environ", {
    "OPENROUTER_API_KEY": "test-api-key",
    "OPENROUTER_MODEL": "test-model",
})
class OpenRouterClientTest(unittest.TestCase):
    @patch("llm.openrouter.requests.post")
    def test_timeout_becomes_openrouter_error(self, fake_post):
        timeout = Timeout("Timeout simulato")
        fake_post.side_effect = timeout

        with self.assertRaises(OpenRouterError) as caught:
            call_openrouter_json(
                messages=[{"role": "user", "content": "Test"}],
                response_schema={"type": "object"},
            )

        self.assertIs(caught.exception.__cause__, timeout)
        fake_post.assert_called_once()

    @patch("llm.openrouter.requests.post")
    def test_empty_choices_becomes_openrouter_error(self, fake_post):
        response = fake_post.return_value
        response.status_code = 200
        response.json.return_value = {"choices": []}

        with self.assertRaises(OpenRouterError) as caught:
            call_openrouter_json(
                messages=[{"role": "user", "content": "Test"}],
                response_schema={"type": "object"},
            )

        self.assertIsInstance(caught.exception.__cause__, IndexError)
        fake_post.assert_called_once()
        response.json.assert_called_once()


if __name__ == "__main__":
    unittest.main()

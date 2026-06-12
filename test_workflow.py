import unittest
from unittest.mock import patch

import json

from planning import build_proposal, calculate_budget_ceiling
from routing import route_request
from pydantic import ValidationError
from datetime import date

from providers.flights import search_flights
from providers.hotels import search_hotels

from scoring import score_proposal
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
    FlightSearchQuery,
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

class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.openrouter_patcher = patch("agents.travel_window.call_openrouter_json")
        self.fake_openrouter_call = self.openrouter_patcher.start()
        self.addCleanup(self.openrouter_patcher.stop)

        self.fake_openrouter_call.return_value = fake_travel_window_response()


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

        result = build_proposal(request)
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

        result = build_proposal(request)

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

        result = build_proposal(request)
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

        result = build_proposal(request)

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
            departure_date = date(2026, 9, 15),
            return_date = date(2026, 9, 20),
        )

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
            departure_date=date(2026, 9, 15),
            return_date=date(2026, 9, 20),
        )

        flights = search_flights(query)

        self.assertEqual(flights, [])

    
    def test_search_hotels_filters_by_max_total_price(self):
        query = HotelSearchQuery(
            destination = "Lisbon",
            checkin_date = date(2026, 9, 15),
            checkout_date = date(2026, 9, 20),
            max_total_price_eur = 500,
        )

        hotels = search_hotels(query)

        for hotel in hotels:
            self.assertEqual(hotel.destination, query.destination)
            self.assertLessEqual(hotel.total_price_eur, 500)


    def test_search_hotels_filters_by_destination(self):
        query = HotelSearchQuery(
            destination="Madrid",
            checkin_date=date(2026, 9, 15),
            checkout_date=date(2026, 9, 20),
        )

        hotels = search_hotels(query)

        self.assertEqual(hotels, [])


    def test_fixed_destination_becomes_proposal_destination(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
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

        result = build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertGreater(len(result.proposals), 0)
        for proposal in result.proposals:
            self.assertIsNotNone(proposal.destination.name)


    def test_trace_contains_resolved_destinations(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                event.event_type == EventType.DESTINATIONS_RESOLVED
                for event in result.trace.events
            )
        )

    
    def test_calculates_budget_ceiling(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=1000,
            budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        self.assertEqual(calculate_budget_ceiling(request), 1200)

    
    def test_accepts_proposal_within_budget_flexibility(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=400,
            budget_flexibility_pct=20,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = build_proposal(request)

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

        result = build_proposal(request)

        self.assertIsInstance(result, PlanningError)
        self.assertTrue(
            any(
                event.reason_code == ReasonCode.OVER_MAX_BUDGET
                for event in result.trace.events
            )
        )


    def test_proposals_are_sorted_by_score(self):
        request = TripRequest(
            origin="Rome",
            budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon"),
            time_pref=TimePreference(mode="flexible_window", min_days=4, max_days=6),
        )

        result = build_proposal(request)

        self.assertNotIsInstance(result, PlanningError)

        scores = [
            score_proposal(proposal)
            for proposal in result.proposals
        ]

        self.assertEqual(scores, sorted(scores, reverse=True))
    

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

        result = build_proposal(request)

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

        result = build_proposal(request)

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


if __name__ == "__main__":
    unittest.main()

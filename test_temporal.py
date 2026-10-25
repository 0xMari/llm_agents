from datetime import date, timedelta
import json
import unittest
from unittest.mock import Mock, patch

from agents.destination_agent import suggest_destinations
from agents.travel_window import (
    InvalidTravelDates, TravelWindowConstraintError, build_system_prompt,
    build_time_context, evaluate_travel_window, remove_past_alternatives,
    resolve_season_window, validate_request_dates, validate_travel_window_constraints,
)
from execution import SearchLimits
from models import (
    DestinationPreference, EventType, PlanningError, ReasonCode,
    SuggestedDestination, TimePreference, TravelWindowAdvice, TripRequest, TripSearchResult,
)
from planning import build_proposal
from test_flights import fake_city, raw_leg
from test_hotels import raw_hotel
from test_proposal_summary import fake_summary_response


def request_for(time_pref, *, mode="fixed"):
    return TripRequest(
        origin="Rome", budget_eur=900,
        destination_pref=DestinationPreference(
            mode=mode, destination="Lisbon" if mode == "fixed" else None,
        ),
        time_pref=time_pref,
    )


def advice_for(start, end):
    return TravelWindowAdvice.model_validate({
        "selected_periods": [{"start": start, "end": end, "score": 90, "reason": "Test period."}],
        "warnings": [], "alternatives": [], "rationale": "Test advice.",
    })


class TemporalRulesTest(unittest.TestCase):
    def test_all_seasons_resolve_the_current_remainder_or_next_occurrence(self):
        reference = date(2026, 10, 25)
        for season, start, end in (
            ("spring", date(2027, 3, 1), date(2027, 6, 1)),
            ("summer", date(2027, 6, 1), date(2027, 9, 1)),
            ("autumn", reference, date(2026, 12, 1)),
            ("winter", date(2026, 12, 1), date(2027, 3, 1)),
        ):
            with self.subTest(season=season):
                self.assertEqual(resolve_season_window(season, reference), (start, end))

    def test_season_boundaries_are_start_inclusive_and_end_exclusive(self):
        for season, first, end, next_first, next_end in (
            ("spring", date(2026, 3, 1), date(2026, 6, 1), date(2027, 3, 1), date(2027, 6, 1)),
            ("summer", date(2026, 6, 1), date(2026, 9, 1), date(2027, 6, 1), date(2027, 9, 1)),
            ("autumn", date(2026, 9, 1), date(2026, 12, 1), date(2027, 9, 1), date(2027, 12, 1)),
            ("winter", date(2026, 12, 1), date(2027, 3, 1), date(2027, 12, 1), date(2028, 3, 1)),
        ):
            with self.subTest(season=season):
                self.assertEqual(resolve_season_window(season, first), (first, end))
                last = end - timedelta(days=1)
                self.assertEqual(resolve_season_window(season, last), (last, end))
                self.assertEqual(resolve_season_window(season, end), (next_first, next_end))

    def test_winter_resolves_correctly_in_january_and_on_leap_day(self):
        for reference, end in (
            (date(2027, 1, 15), date(2027, 3, 1)),
            (date(2028, 2, 29), date(2028, 3, 1)),
            (date(2026, 12, 31), date(2027, 3, 1)),
        ):
            with self.subTest(reference=reference):
                self.assertEqual(resolve_season_window("winter", reference), (reference, end))

    def test_unknown_season_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported season"):
            resolve_season_window("monsoon", date(2026, 10, 25))

    def test_context_contains_only_the_applicable_time_constraints(self):
        reference = date(2026, 10, 25)
        for preference, expected in (
            (TimePreference(mode="season", season="winter"), {
                "departure_window": {"start": "2026-12-01", "end_exclusive": "2027-03-01"},
            }),
            (TimePreference(mode="month", month=4), {"resolved_year": 2027}),
            (TimePreference(mode="flexible_window"), {}),
        ):
            with self.subTest(mode=preference.mode):
                request = request_for(preference)
                before = request.model_dump_json()
                self.assertEqual(build_time_context(request, reference), {
                    "reference_date": "2026-10-25", **expected,
                })
                self.assertEqual(request.model_dump_json(), before)

    def test_selected_departures_yesterday_today_and_tomorrow_in_every_mode(self):
        reference = date(2026, 10, 25)
        for offset in (-1, 0, 1):
            start = reference + timedelta(days=offset)
            end = start + timedelta(days=5)
            for preference in (
                TimePreference(mode="exact_dates", start_date=start, end_date=end),
                TimePreference(mode="month", month=10),
                TimePreference(mode="season", season="autumn"),
                TimePreference(mode="flexible_window"),
            ):
                with self.subTest(offset=offset, mode=preference.mode):
                    request = request_for(preference)
                    advice = advice_for(start, end)
                    if offset < 0:
                        with self.assertRaisesRegex(TravelWindowConstraintError, "past"):
                            validate_travel_window_constraints(request, advice, reference)
                    else:
                        validate_travel_window_constraints(request, advice, reference)

    def test_invalid_season_departures_are_not_shifted_or_filtered(self):
        request = request_for(TimePreference(mode="season", season="autumn"))
        for start in (date(2026, 8, 20), date(2026, 12, 1), date(2027, 10, 25)):
            with self.subTest(start=start):
                advice = advice_for(start, start + timedelta(days=5))
                before = advice.model_dump_json()
                with self.assertRaises(TravelWindowConstraintError):
                    validate_travel_window_constraints(request, advice, date(2026, 10, 25))
                self.assertEqual(advice.model_dump_json(), before)

    def test_returns_may_cross_month_season_and_year_boundaries(self):
        for preference, start, end, reference in (
            (TimePreference(mode="month", month=12), "2026-12-31", "2027-01-05", date(2026, 10, 25)),
            (TimePreference(mode="month", month=8), "2027-08-30", "2027-09-04", date(2026, 10, 25)),
            (TimePreference(mode="season", season="summer"), "2027-08-31", "2027-09-05", date(2026, 10, 25)),
            (TimePreference(mode="season", season="winter"), "2028-02-29", "2028-03-05", date(2028, 2, 29)),
        ):
            with self.subTest(start=start):
                preference.min_days = preference.max_days = 6
                validate_travel_window_constraints(request_for(preference), advice_for(start, end), reference)

    def test_month_still_rejects_wrong_departure_month_and_year(self):
        request = request_for(TimePreference(mode="month", month=12))
        for start in (date(2026, 11, 30), date(2027, 12, 31)):
            with self.subTest(start=start), self.assertRaises(TravelWindowConstraintError):
                validate_travel_window_constraints(request, advice_for(start, start + timedelta(days=5)), date(2026, 10, 25))

    def test_season_still_enforces_inclusive_minimum_and_maximum_days(self):
        request = request_for(TimePreference(mode="season", season="winter", min_days=6, max_days=6))
        for end in ("2027-03-03", "2027-03-05"):
            with self.subTest(end=end), self.assertRaises(TravelWindowConstraintError):
                validate_travel_window_constraints(request, advice_for("2027-02-27", end), date(2027, 1, 15))

    def test_one_invalid_selected_period_rejects_the_entire_advice(self):
        advice = advice_for("2026-11-05", "2026-11-10")
        advice.selected_periods.extend(advice_for("2026-12-05", "2026-12-10").selected_periods)
        with self.assertRaises(TravelWindowConstraintError):
            validate_travel_window_constraints(
                request_for(TimePreference(mode="season", season="autumn")), advice, date(2026, 10, 25),
            )
        self.assertEqual(len(advice.selected_periods), 2)

    def test_future_alternatives_may_fall_outside_the_requested_season(self):
        advice = advice_for("2026-11-05", "2026-11-10")
        advice.alternatives = advice_for("2027-04-05", "2027-04-10").selected_periods
        validate_travel_window_constraints(
            request_for(TimePreference(mode="season", season="autumn")), advice, date(2026, 10, 25),
        )
        self.assertIs(remove_past_alternatives(advice, date(2026, 10, 25)), advice)

    def test_historical_requests_can_be_deserialized_without_reading_the_clock(self):
        request = request_for(TimePreference(mode="exact_dates", start_date="2020-05-10", end_date="2020-05-15"))
        self.assertEqual(TripRequest.model_validate_json(request.model_dump_json()), request)
        validate_request_dates(request, date(2020, 5, 10))
        with self.assertRaises(InvalidTravelDates):
            validate_request_dates(request, date(2026, 10, 25))


class TemporalAgentsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.reference = date(2026, 10, 25)
        self.destination = SuggestedDestination(name="Lisbon", country="Portugal", reason="Test.", match_score=90)
        self.window_call = self.enterContext(patch("agents.travel_window.call_openrouter_json"))
        self.destination_call = self.enterContext(patch("agents.destination_agent.call_openrouter_json"))
        self.window_call.return_value = advice_for("2026-12-05", "2026-12-10").model_dump_json()
        self.destination_call.return_value = json.dumps({"destinations": [{
            **self.destination.model_dump(mode="json"), "caution": None,
        }]})

    def test_both_agents_reject_past_input_before_an_llm_call(self):
        request = request_for(TimePreference(mode="exact_dates", start_date="2026-10-24", end_date="2026-10-29"), mode="open")
        with self.assertRaises(InvalidTravelDates):
            suggest_destinations(request, reference_date=self.reference)
        with self.assertRaises(InvalidTravelDates):
            evaluate_travel_window(request, self.destination, reference_date=self.reference)
        self.window_call.assert_not_called()
        self.destination_call.assert_not_called()

    def test_both_agents_use_identical_season_context_without_reading_the_clock(self):
        request = request_for(TimePreference(mode="season", season="winter"), mode="open")
        with patch("agents.travel_window.date", wraps=date) as window_clock, patch(
            "agents.destination_agent.date", wraps=date,
        ) as destination_clock:
            suggest_destinations(request, reference_date=self.reference)
            evaluate_travel_window(request, self.destination, reference_date=self.reference)
            window_clock.today.assert_not_called()
            destination_clock.today.assert_not_called()
        contexts = [json.loads(call.call_args.kwargs["messages"][1]["content"])["time_context"]
                    for call in (self.window_call, self.destination_call)]
        self.assertEqual(contexts, [build_time_context(request, self.reference)] * 2)

    def test_calendar_season_does_not_follow_destination_hemisphere(self):
        request = request_for(TimePreference(mode="season", season="summer"))
        destination = self.destination.model_copy(update={"name": "Sydney", "country": "Australia"})
        self.window_call.return_value = advice_for("2027-06-05", "2027-06-10").model_dump_json()
        evaluate_travel_window(request, destination, reference_date=self.reference)
        payload = json.loads(self.window_call.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["time_context"]["departure_window"], {
            "start": "2027-06-01", "end_exclusive": "2027-09-01",
        })

    def test_incorrect_llm_season_is_rejected_without_retry(self):
        request = request_for(TimePreference(mode="season", season="spring"))
        with self.assertRaises(TravelWindowConstraintError):
            evaluate_travel_window(request, self.destination, reference_date=self.reference)
        self.window_call.assert_called_once()

    def test_standalone_agents_read_the_default_clock_once(self):
        request = request_for(TimePreference(mode="season", season="winter"), mode="open")
        with patch("agents.travel_window.date", wraps=date) as clock:
            clock.today.return_value = self.reference
            evaluate_travel_window(request, self.destination)
            clock.today.assert_called_once()
        with patch("agents.destination_agent.date", wraps=date) as clock:
            clock.today.return_value = self.reference
            suggest_destinations(request)
            clock.today.assert_called_once()

    def test_prompts_explain_departure_only_calendar_constraints(self):
        request = request_for(TimePreference(mode="season", season="winter"), mode="open")
        suggest_destinations(request, reference_date=self.reference)
        prompts = [build_system_prompt(), self.destination_call.call_args.kwargs["messages"][0]["content"]]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn("Northern Hemisphere", prompt)
                self.assertIn("end_exclusive", prompt)
                self.assertIn("returns may", prompt.lower())


class TemporalPlannerTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict("os.environ", {
            "SERPAPI_API_KEY": "test-key", "OPENROUTER_API_KEY": "test-key", "OPENROUTER_MODEL": "test-model",
        }))
        self.get = self.enterContext(patch("providers.serpapi_client.requests.get", side_effect=self.search_response))
        self.post = self.enterContext(patch("llm.openrouter.requests.post", side_effect=self.llm_response))
        self.sleep = self.enterContext(patch("llm.openrouter.time.sleep"))
        self.reference = date(2026, 10, 25)
        self.advice = advice_for("2026-12-05", "2026-12-10")
        self.limits = SearchLimits(max_llm_calls=3)

    def response(self, payload):
        result = Mock(status_code=200)
        result.json.return_value = payload
        return result

    def llm_response(self, url, *, json, **kwargs):
        schema = json["response_format"]["json_schema"]["name"]
        if schema == "destination_advice":
            content = '{"destinations":[{"name":"Lisbon","country":"Portugal","reason":"Test.","match_score":90,"themes_matched":[],"caution":null}]}'
        elif schema == "travel_window_advice":
            content = self.advice.model_dump_json()
        else:
            self.assertEqual(schema, "proposal_summaries")
            content = fake_summary_response(json["messages"])
        return self.response({"choices": [{"message": {"content": content}}]})

    def search_response(self, url, *, params, **kwargs):
        if params["engine"] == "google_flights_autocomplete":
            city = "Rome, Italy" if params["q"] == "Rome" else "Lisbon, Portugal"
            payload = {"suggestions": [fake_city(city)]}
        elif params["engine"] == "google_flights":
            if "departure_token" in params:
                payload = {"other_flights": [raw_leg("LIS", "FCO", date.fromisoformat(params["return_date"]), price=180)]}
            else:
                payload = {"best_flights": [raw_leg(day=date.fromisoformat(params["outbound_date"]), token="test-token")]}
        else:
            self.assertEqual(params["engine"], "google_hotels")
            payload = {"properties": [raw_hotel()]}
        return self.response({"search_metadata": {"status": "Success"}, **payload})

    def test_past_exact_input_stops_before_all_calls_even_with_zero_limits(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                request = request_for(TimePreference(mode="exact_dates", start_date="2026-10-24", end_date="2026-10-29"), mode=mode)
                before = request.model_dump_json()
                result = build_proposal(request, limits=SearchLimits(max_serpapi_calls=0, max_llm_calls=0), reference_date=self.reference)
                self.assertIsInstance(result, PlanningError)
                self.assertEqual(result.code, ReasonCode.INVALID_TRAVEL_DATES)
                self.assertEqual([e.event_type for e in result.trace.events], [EventType.REQUEST_ROUTED, EventType.SEARCH_FAILED])
                details = result.trace.events[-1].details
                self.assertEqual(details["stage"], "input_validation")
                self.assertEqual(details["reference_date"], "2026-10-25")
                self.assertEqual((details["serpapi_calls"], details["llm_calls"], details["llm_retries"]), (0, 0, 0))
                self.assertEqual(request.model_dump_json(), before)
                self.assertEqual(PlanningError.model_validate_json(result.model_dump_json()), result)
        self.get.assert_not_called()
        self.post.assert_not_called()
        self.sleep.assert_not_called()

    def test_past_or_out_of_season_llm_output_is_not_an_input_error(self):
        for preference, start, end in (
            (TimePreference(mode="flexible_window"), "2026-10-24", "2026-10-29"),
            (TimePreference(mode="month", month=10), "2026-10-24", "2026-10-29"),
            (TimePreference(mode="season", season="autumn"), "2026-12-05", "2026-12-10"),
        ):
            for mode in ("fixed", "open"):
                with self.subTest(preference=preference, mode=mode):
                    self.post.reset_mock()
                    self.advice = advice_for(start, end)
                    result = build_proposal(request_for(preference, mode=mode), limits=self.limits, reference_date=self.reference)
                    self.assertEqual(result.code, ReasonCode.LLM_CONSTRAINT_VIOLATION)
                    self.assertEqual(result.trace.events[-1].event_type, EventType.TRAVEL_WINDOW_FAILED)
                    self.assertEqual(result.trace.events[-1].details["llm_calls"], 2 if mode == "open" else 1)
                    self.assertEqual(result.trace.events[-1].details["llm_retries"], 0)
        self.get.assert_not_called()
        self.sleep.assert_not_called()

    def test_valid_season_and_cross_year_month_complete_the_real_offline_pipeline(self):
        for preference, start, end in (
            (TimePreference(mode="season", season="winter"), "2026-12-31", "2027-01-05"),
            (TimePreference(mode="month", month=12), "2026-12-31", "2027-01-05"),
            (TimePreference(mode="season", season="autumn"), "2026-11-30", "2026-12-05"),
            (TimePreference(mode="season", season="winter"), "2027-02-28", "2027-03-05"),
        ):
            with self.subTest(preference=preference, start=start):
                self.get.reset_mock()
                self.post.reset_mock()
                self.advice = advice_for(start, end)
                request = request_for(preference, mode="open")
                result = build_proposal(request, limits=self.limits, reference_date=self.reference)
                self.assertIsInstance(result, TripSearchResult)
                self.assertEqual(result.summary_status, "generated")
                self.assertEqual(result.proposals[0].hotel.nights, 5)
                self.assertEqual(result.trace.events[-1].details["serpapi_calls"], 5)
                self.assertEqual(result.trace.events[-1].details["llm_calls"], 3)
                self.assertEqual(self.get.call_count, 5)
                prompts = [json.loads(call.kwargs["json"]["messages"][1]["content"])
                           for call in self.post.call_args_list[:2]]
                self.assertEqual([p["time_context"] for p in prompts], [build_time_context(request, self.reference)] * 2)
                self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_same_day_exact_input_is_accepted(self):
        self.advice = advice_for(self.reference, self.reference + timedelta(days=5))
        request = request_for(TimePreference(
            mode="exact_dates", start_date=self.reference, end_date=self.reference + timedelta(days=5),
        ))
        self.assertIsInstance(build_proposal(request, limits=self.limits, reference_date=self.reference), TripSearchResult)

    def test_default_clock_is_read_once_even_if_midnight_passes_between_agents(self):
        request = request_for(TimePreference(mode="season", season="winter"), mode="open")
        with patch("planning.date", wraps=date) as planner_clock, patch(
            "agents.travel_window.date", wraps=date,
        ) as window_clock, patch("agents.destination_agent.date", wraps=date) as destination_clock:
            planner_clock.today.side_effect = [self.reference, date(2030, 1, 1)]
            result = build_proposal(request, limits=self.limits)
            self.assertIsInstance(result, TripSearchResult)
            planner_clock.today.assert_called_once()
            window_clock.today.assert_not_called()
            destination_clock.today.assert_not_called()

    def test_explicit_reference_does_not_read_any_clock_or_depend_on_current_year(self):
        request = request_for(TimePreference(mode="season", season="winter"), mode="open")
        with patch("planning.date", wraps=date) as planner_clock, patch(
            "agents.travel_window.date", wraps=date,
        ) as window_clock, patch("agents.destination_agent.date", wraps=date) as destination_clock:
            result = build_proposal(request, limits=self.limits, reference_date=self.reference)
            self.assertIsInstance(result, TripSearchResult)
            for clock in (planner_clock, window_clock, destination_clock):
                clock.today.assert_not_called()


if __name__ == "__main__":
    unittest.main()

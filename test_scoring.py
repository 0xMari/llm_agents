from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import permutations
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from models import (
    DecisionTrace,
    DestinationPreference,
    TimePreference,
    TripProposal,
    TripRequest,
    TripSearchResult,
)
from scoring import flight_comfort, select_final_proposals


def paris_proposal(*, return_hour=8, flight_price="143", hotel_name="Nation / Voltaire", rating=4.7168694):
    # Reproduce the prices and return-flight variants from the recorded Paris run.
    return_time = datetime(2026, 12, 10, return_hour, 0 if return_hour == 8 else 5)
    flight_number = {8: "TO 3912", 12: "TO 3914", 17: "TO 3918"}.get(return_hour, "TO TEST")
    if return_hour == 17:
        return_time = return_time.replace(minute=45)
    return TripProposal.model_validate({
        "destination": {"name": "Paris", "country": "France", "reason": "Museums and food", "match_score": 90},
        "period": {"start": date(2026, 12, 5), "end": date(2026, 12, 10), "reason": "Exact dates", "score": 65},
        "flight": {
            "origin": "Rome", "destination": "Paris", "airline": "Transavia",
            "departure_date": date(2026, 12, 5), "return_date": date(2026, 12, 10),
            "price_eur": flight_price, "stops": 0,
            "outbound": {"duration_minutes": 135, "segments": [{
                "departure_airport": "FCO", "arrival_airport": "ORY",
                "departure_local": datetime(2026, 12, 5, 18, 10),
                "arrival_local": datetime(2026, 12, 5, 20, 25),
                "airline": "Transavia", "flight_number": "TO 3917",
            }]},
            "inbound": {"duration_minutes": 125, "segments": [{
                "departure_airport": "ORY", "arrival_airport": "FCO",
                "departure_local": return_time,
                "arrival_local": return_time + timedelta(minutes=125),
                "airline": "Transavia", "flight_number": flight_number,
            }]},
        },
        "hotel": {
            "destination": "Paris", "name": hotel_name, "nightly_price_eur": "111",
            "nights": 5, "total_price_eur": "556", "rating": rating,
        },
        "total_price_eur": Decimal(flight_price) + Decimal("556"),
        "recommendations": ["Check museum opening times"],
        "comments": [{"reason_code": "WITHIN_BUDGET", "comment": "Within budget"}],
    })


def tradeoff_proposal(flight_price, *, stops=0, hotel_name="Nation / Voltaire"):
    payload = paris_proposal(return_hour=12, flight_price=flight_price, hotel_name=hotel_name).model_dump()
    airports = ["FCO", *("MAD", "BCN")[:stops], "ORY"]
    segments = []
    for index, (origin, destination) in enumerate(zip(airports, airports[1:])):
        start = datetime(2026, 12, 5, 9) + timedelta(hours=3 * index)
        segments.append({
            "departure_airport": origin, "arrival_airport": destination,
            "departure_local": start, "arrival_local": start + timedelta(hours=2),
            "airline": "Test Air", "flight_number": f"TEST {index + 1}",
        })
    payload["flight"]["outbound"] = {
        "segments": segments, "duration_minutes": 120 * len(segments) + 60 * stops,
    }
    payload["flight"]["stops"] = stops
    return TripProposal.model_validate(payload)


class FinalProposalSelectionTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.request = TripRequest(
            origin="Rome", budget_eur=900, budget_flexibility_pct=0,
            destination_pref=DestinationPreference(mode="fixed", destination="Paris"),
            time_pref=TimePreference(mode="exact_dates", start_date=date(2026, 12, 5), end_date=date(2026, 12, 10)),
        )
        self.paris = [
            paris_proposal(return_hour=8), paris_proposal(return_hour=12),
            paris_proposal(return_hour=17, flight_price="151"),
        ]
        self.tradeoffs = [
            tradeoff_proposal("100", stops=2),
            tradeoff_proposal("150", stops=1),
            tradeoff_proposal("200"),
        ]

    def assert_selected_quote(self, selected, original):
        self.assertEqual(
            selected.model_dump(exclude={"selection_profiles"}),
            original.model_dump(exclude={"selection_profiles"}),
        )

    def test_paris_keeps_noon_return_once_with_both_profiles(self):
        selected = select_final_proposals(self.paris, self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], self.paris[1])
        self.assertEqual(selected[0].selection_profiles, ["cheapest", "comfortable"])

    def test_three_distinct_tradeoffs_can_share_the_same_hotel(self):
        selected = select_final_proposals(self.tradeoffs, self.request)
        self.assertEqual([p.selection_profiles for p in selected], [["cheapest"], ["balanced"], ["comfortable"]])
        self.assertEqual([p.total_price_eur for p in selected], [Decimal("656"), Decimal("706"), Decimal("756")])
        for proposal, original in zip(selected, self.tradeoffs):
            self.assert_selected_quote(proposal, original)

    def test_all_candidates_are_considered_before_selecting_three(self):
        candidates = [tradeoff_proposal(str(price), stops=2, hotel_name=f"Stay {price}") for price in range(101, 110)]
        selected = select_final_proposals([*candidates, *self.tradeoffs], self.request)
        self.assertEqual(len(selected), 3)
        self.assertEqual([p.total_price_eur for p in selected], [Decimal("656"), Decimal("706"), Decimal("756")])

    def test_missing_middle_returns_only_two_proposals(self):
        selected = select_final_proposals([self.tradeoffs[0], self.tradeoffs[2]], self.request)
        self.assertEqual([p.selection_profiles for p in selected], [["cheapest"], ["comfortable"]])

    def test_middle_must_improve_comfort_and_be_strictly_between_prices(self):
        cheapest, _, comfortable = self.tradeoffs
        for candidate in (
            tradeoff_proposal("150", stops=2),
            tradeoff_proposal("250", stops=1),
            tradeoff_proposal("200", stops=1),
        ):
            with self.subTest(price=candidate.total_price_eur, stops=candidate.flight.stops):
                selected = select_final_proposals([cheapest, candidate, comfortable], self.request)
                self.assertEqual([p.selection_profiles for p in selected], [["cheapest"], ["comfortable"]])

    def test_cheapest_qualifying_middle_wins_not_the_most_expensive_one(self):
        extra = tradeoff_proposal("160", stops=1)
        extra.flight.outbound.duration_minutes -= 10
        selected = select_final_proposals([extra, *self.tradeoffs], self.request)
        self.assert_selected_quote(selected[1], self.tradeoffs[1])

    def test_duplicate_inputs_do_not_create_duplicate_choices(self):
        candidates = [p.model_copy(deep=True) for p in self.tradeoffs] + self.tradeoffs * 2
        selected = select_final_proposals(candidates, self.request)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({p.model_dump_json(exclude={"selection_profiles"}) for p in selected}), 3)

    def test_all_identical_quotes_return_one_proposal(self):
        proposal = self.paris[0]
        selected = select_final_proposals([proposal, proposal.model_copy(deep=True), proposal], self.request)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].selection_profiles, ["cheapest", "comfortable"])

    def test_redundant_same_price_and_time_flights_do_not_fill_slots(self):
        other = self.paris[0].model_copy(deep=True)
        other.flight.inbound.segments[0].flight_number = "TO 9999"
        selected = select_final_proposals([self.paris[0], other], self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], self.paris[0])

    def test_cheapest_is_based_on_complete_trip_not_flight_price(self):
        cheap_flight = paris_proposal(flight_price="100")
        cheaper_trip = paris_proposal(flight_price="200", hotel_name="Other stay")
        cheaper_trip.hotel.total_price_eur = Decimal("400")
        cheaper_trip.total_price_eur = Decimal("600")
        selected = select_final_proposals([cheap_flight, cheaper_trip], self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], cheaper_trip)

    def test_cheaper_direct_flight_gets_both_profiles(self):
        direct = tradeoff_proposal("90")
        selected = select_final_proposals([self.tradeoffs[0], direct], self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], direct)
        self.assertEqual(selected[0].selection_profiles, ["cheapest", "comfortable"])

    def test_stops_take_priority_over_early_departures_and_duration(self):
        direct = paris_proposal(return_hour=8, flight_price="200")
        direct.flight.outbound.duration_minutes = 1000
        connection = tradeoff_proposal("100", stops=1)
        self.assertLess(flight_comfort(direct, 9), flight_comfort(connection, 9))
        selected = select_final_proposals([connection, direct], self.request)
        self.assert_selected_quote(selected[-1], direct)

    def test_early_departures_take_priority_over_duration(self):
        noon = self.paris[1].model_copy(deep=True)
        noon.flight.outbound.duration_minutes = 1000
        selected = select_final_proposals([self.paris[0], noon], self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], noon)

    def test_comfort_uses_provider_duration_not_local_clock_subtraction(self):
        longer = self.paris[1].model_copy(deep=True)
        longer.flight.outbound.duration_minutes = 500
        shorter = self.paris[1].model_copy(deep=True)
        shorter.flight.outbound.segments[-1].arrival_local += timedelta(hours=6)
        selected = select_final_proposals([longer, shorter], self.request)
        self.assert_selected_quote(selected[0], shorter)
        self.assertEqual(flight_comfort(shorter, 9), (0, 0, 260))

    def test_early_hour_is_configurable_and_boundary_is_not_early(self):
        for hour, expected in ((0, self.paris[0]), (8, self.paris[0]), (9, self.paris[1]), (13, self.paris[0])):
            with self.subTest(hour=hour):
                selected = select_final_proposals(self.paris[:2], self.request, early_hour=hour)
                self.assertEqual(len(selected), 1)
                self.assert_selected_quote(selected[0], expected)

    def test_early_departures_count_each_leg_but_not_connection_departures(self):
        proposal = tradeoff_proposal("100", stops=1)
        self.assertEqual(flight_comfort(proposal, 13)[1], 2)

    def test_destination_then_period_then_rating_break_equal_price_comfort_ties(self):
        for field in ("destination", "period", "hotel"):
            with self.subTest(field=field):
                preferred = self.paris[1].model_copy(deep=True)
                if field == "destination":
                    preferred.destination.match_score = 100
                elif field == "period":
                    preferred.period.score = 100
                else:
                    preferred.hotel.rating = 5
                selected = select_final_proposals([self.paris[1], preferred], self.request)
                self.assert_selected_quote(selected[0], preferred)

    def test_price_and_comfort_take_priority_over_old_score(self):
        expensive = self.paris[1].model_copy(deep=True)
        expensive.destination.match_score = 100
        expensive.hotel.rating = 5
        expensive.flight.price_eur += Decimal("50")
        expensive.total_price_eur += Decimal("50")
        selected = select_final_proposals([expensive, self.paris[1]], self.request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], self.paris[1])

    def test_preferred_budget_excludes_more_comfortable_flexible_offers(self):
        request = self.request.model_copy(update={"budget_eur": 656, "budget_flexibility_pct": 20})
        selected = select_final_proposals(self.tradeoffs, request)
        self.assertEqual(len(selected), 1)
        self.assert_selected_quote(selected[0], self.tradeoffs[0])
        self.assertEqual(selected[0].selection_profiles, ["cheapest", "comfortable"])

    def test_flexible_offers_are_used_when_no_offer_meets_preferred_budget(self):
        request = self.request.model_copy(update={"budget_eur": 650, "budget_flexibility_pct": 20})
        selected = select_final_proposals(self.tradeoffs, request)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(p.total_price_eur > request.budget_eur for p in selected))

    def test_absent_budget_does_not_disable_price_comparison(self):
        request = self.request.model_copy(update={"budget_eur": None})
        selected = select_final_proposals(self.tradeoffs, request)
        self.assertEqual([p.total_price_eur for p in selected], [Decimal("656"), Decimal("706"), Decimal("756")])

    def test_complete_quotes_and_stay_dates_are_preserved(self):
        original = self.paris[1].model_dump()
        original["hotel"].update(total_price_eur=Decimal("570"), nightly_price_eur=Decimal("114"))
        original["hotel"]["redirect"] = {
            "url": "https://example.com/property", "kind": "property_website", "label": "Property website",
        }
        original["flight"]["redirect"] = {
            "url": "https://example.com/flights", "kind": "search", "label": "Flight search",
        }
        original["total_price_eur"] = Decimal("713")
        original["comments"][0]["comment"] = "Quote-specific comment"
        original["recommendations"] = ["Quote-specific advice"]
        original = TripProposal.model_validate(original)
        selected = select_final_proposals([original], self.request)
        self.assert_selected_quote(selected[0], original)

    def test_distinct_tradeoffs_do_not_mix_destinations_or_hotels(self):
        other = self.tradeoffs[2].model_copy(deep=True)
        other.destination.name = "Lyon"
        other.hotel.destination = "Lyon"
        other.hotel.name = "Other stay"
        other.flight.destination = "Lyon"
        other.flight.outbound.segments[-1].arrival_airport = "LYS"
        other.flight.inbound.segments[0].departure_airport = "LYS"
        request = self.request.model_copy(update={"destination_pref": DestinationPreference(mode="open")})
        selected = select_final_proposals([self.tradeoffs[0], other], request)
        self.assertEqual(len(selected), 2)
        self.assert_selected_quote(selected[0], self.tradeoffs[0])
        self.assert_selected_quote(selected[1], other)

    def test_order_of_unequal_candidates_does_not_change_selection(self):
        expected = select_final_proposals(self.tradeoffs, self.request)
        for candidates in permutations(self.tradeoffs):
            self.assertEqual(select_final_proposals(list(candidates), self.request), expected)

    def test_inputs_are_not_mutated(self):
        proposals = [self.paris[2], self.paris[0], self.paris[1]]
        before = [p.model_dump_json() for p in proposals]
        request_before = self.request.model_dump_json()
        selected = select_final_proposals(proposals, self.request)
        self.assertEqual([p.model_dump_json() for p in proposals], before)
        self.assertEqual(self.request.model_dump_json(), request_before)
        selected[0].selection_profiles.append("balanced")
        self.assertEqual([p.model_dump_json() for p in proposals], before)

    def test_empty_input_returns_no_proposals(self):
        self.assertEqual(select_final_proposals([], self.request), [])

    def test_invalid_early_hours_are_rejected_even_with_empty_input(self):
        for early_hour in (-1, 24, True, False, "9", 9.5, None):
            with self.subTest(early_hour=early_hour):
                with self.assertRaises(ValueError):
                    select_final_proposals([], self.request, early_hour=early_hour)

    def test_result_serializes_flat_proposals_with_profiles_and_round_trips(self):
        result = TripSearchResult(proposals=select_final_proposals(self.tradeoffs, self.request), trace=DecisionTrace())
        payload = json.loads(result.model_dump_json())
        self.assertNotIn("groups", payload)
        self.assertEqual(len(payload["proposals"]), 3)
        self.assertEqual(payload["proposals"][1]["total_price_eur"], "706")
        self.assertEqual(payload["proposals"][1]["selection_profiles"], ["balanced"])
        self.assertTrue(all("alternatives" not in p for p in payload["proposals"]))
        self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_result_requires_one_to_three_proposals_and_rejects_groups_only_payload(self):
        for count in (0, 4):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                TripSearchResult(proposals=[self.paris[0]] * count, trace=DecisionTrace())
        with self.assertRaises(ValidationError):
            TripSearchResult.model_validate({"groups": [{"recommended": self.paris[0].model_dump()}], "trace": {"events": []}})

    def test_unselected_candidates_have_independent_empty_profile_lists(self):
        first, second = self.paris[:2]
        self.assertEqual(first.selection_profiles, [])
        first.selection_profiles.append("cheapest")
        self.assertEqual(second.selection_profiles, [])

    def test_profiles_reject_unknown_values(self):
        payload = self.paris[0].model_dump()
        payload["selection_profiles"] = ["invented"]
        with self.assertRaises(ValidationError):
            TripProposal.model_validate(payload)

    def test_reselection_replaces_old_profile_labels_without_mutating_input(self):
        proposal = self.paris[0].model_copy(update={"selection_profiles": ["balanced"]})
        selected = select_final_proposals([proposal], self.request)
        self.assertEqual(selected[0].selection_profiles, ["cheapest", "comfortable"])
        self.assertEqual(proposal.selection_profiles, ["balanced"])


if __name__ == "__main__":
    unittest.main()

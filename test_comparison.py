from datetime import timedelta
from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from comparison import add_comparisons
from models import DecisionTrace, ProposalComparison, TripProposal, TripSearchResult
from test_scoring import paris_proposal, tradeoff_proposal


class ProposalComparisonTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.proposals = [
            tradeoff_proposal("100", stops=2),
            tradeoff_proposal("150", stops=1),
            tradeoff_proposal("200"),
        ]

    def test_single_proposal_references_itself_with_zero_deltas(self):
        selected = add_comparisons([self.proposals[0]])
        self.assertEqual(selected[0].comparison.model_dump(), {
            "proposal_id": "p1", "reference_id": "p1",
            "extra_price_eur": Decimal("0"), "stops_delta": 0,
            "duration_delta_minutes": 0, "early_departures_delta": 0,
        })

    def test_three_proposals_have_unique_ids_and_share_cheapest_reference(self):
        selected = add_comparisons(self.proposals)
        self.assertEqual(len(selected), 3)
        self.assertEqual([p.comparison.proposal_id for p in selected], ["p1", "p2", "p3"])
        self.assertEqual([p.comparison.reference_id for p in selected], ["p1"] * 3)
        self.assertEqual([p.comparison.extra_price_eur for p in selected], [Decimal("0"), Decimal("50"), Decimal("100")])
        self.assertEqual([p.comparison.stops_delta for p in selected], [0, -1, -2])
        self.assertEqual([p.comparison.duration_delta_minutes for p in selected], [0, -180, -360])

    def test_cheapest_is_found_without_reordering_input(self):
        proposals = [self.proposals[2], self.proposals[0], self.proposals[1]]
        selected = add_comparisons(proposals)
        self.assertEqual([p.total_price_eur for p in selected], [p.total_price_eur for p in proposals])
        self.assertEqual([p.comparison.reference_id for p in selected], ["p2"] * 3)
        self.assertEqual(selected[1].comparison.extra_price_eur, Decimal("0"))

    def test_price_ties_use_first_cheapest_proposal_as_reference(self):
        proposals = [paris_proposal(return_hour=8), paris_proposal(return_hour=12)]
        selected = add_comparisons(proposals)
        self.assertEqual([p.comparison.reference_id for p in selected], ["p1", "p1"])
        self.assertEqual([p.comparison.extra_price_eur for p in selected], [Decimal("0"), Decimal("0")])
        self.assertEqual(selected[1].comparison.early_departures_delta, -1)

    def test_eighty_euros_extra_and_one_fewer_early_departure(self):
        proposals = [paris_proposal(return_hour=8), paris_proposal(return_hour=12, flight_price="223")]
        comparison = add_comparisons(proposals)[1].comparison
        self.assertEqual(comparison.extra_price_eur, Decimal("80"))
        self.assertEqual(comparison.early_departures_delta, -1)
        self.assertEqual(comparison.stops_delta, 0)
        self.assertEqual(comparison.duration_delta_minutes, 0)

    def test_price_differences_preserve_decimal_cents(self):
        proposals = [paris_proposal(flight_price="143.10"), paris_proposal(flight_price="223.30")]
        comparison = add_comparisons(proposals)[1].comparison
        self.assertIsInstance(comparison.extra_price_eur, Decimal)
        self.assertEqual(comparison.extra_price_eur, Decimal("80.20"))

    def test_duration_uses_both_provider_legs_not_local_time_subtraction(self):
        reference = self.proposals[0]
        other = self.proposals[2].model_copy(deep=True)
        other.flight.inbound.duration_minutes += 35
        other.flight.outbound.segments[-1].arrival_local += timedelta(hours=6)
        comparison = add_comparisons([reference, other])[1].comparison
        self.assertEqual(comparison.duration_delta_minutes, -325)

    def test_higher_price_does_not_imply_better_flight_metrics(self):
        proposals = [tradeoff_proposal("100"), tradeoff_proposal("200", stops=2)]
        comparison = add_comparisons(proposals)[1].comparison
        self.assertEqual(comparison.extra_price_eur, Decimal("100"))
        self.assertEqual(comparison.stops_delta, 2)
        self.assertEqual(comparison.duration_delta_minutes, 360)

    def test_early_departures_use_the_supplied_threshold_on_both_legs(self):
        reference = paris_proposal(return_hour=8)
        reference.flight.outbound.segments[0].departure_local = reference.flight.outbound.segments[0].departure_local.replace(hour=8)
        proposals = [reference, paris_proposal(return_hour=12, flight_price="223")]
        self.assertEqual(add_comparisons(proposals, early_hour=9)[1].comparison.early_departures_delta, -2)
        self.assertEqual(add_comparisons(proposals, early_hour=8)[1].comparison.early_departures_delta, 0)

    def test_difference_is_for_complete_trip_even_when_hotels_change(self):
        reference = paris_proposal()
        other = paris_proposal(hotel_name="Other stay")
        other.hotel.total_price_eur += Decimal("80")
        other.total_price_eur += Decimal("80")
        selected = add_comparisons([reference, other])
        self.assertEqual(selected[1].comparison.extra_price_eur, Decimal("80"))
        self.assertEqual(selected[1].flight.price_eur, reference.flight.price_eur)
        self.assertEqual(selected[1].hotel, other.hotel)
        self.assertEqual(selected[1].comparison.stops_delta, 0)

    def test_only_comparison_is_added_without_mutating_inputs(self):
        original = self.proposals[1].model_dump()
        original["selection_profiles"] = ["balanced"]
        original["hotel"]["redirect"] = {
            "url": "https://example.com/hotel", "kind": "property_website", "label": "Hotel",
        }
        original["flight"]["redirect"] = {
            "url": "https://example.com/flights", "kind": "search", "label": "Flights",
        }
        self.proposals[1] = TripProposal.model_validate(original)
        before = [p.model_dump_json() for p in self.proposals]
        selected = add_comparisons(self.proposals)
        self.assertEqual([p.model_dump_json() for p in self.proposals], before)
        for proposal, original in zip(selected, self.proposals):
            self.assertIsNot(proposal, original)
            self.assertEqual(proposal.model_dump(exclude={"comparison"}), original.model_dump(exclude={"comparison"}))
        selected[1].comparison.extra_price_eur = Decimal("999")
        self.assertEqual([p.model_dump_json() for p in self.proposals], before)

    def test_recomparison_replaces_old_ids_and_reference_without_mutation(self):
        old_result = add_comparisons(self.proposals)
        before = [p.model_dump_json() for p in old_result]
        selected = add_comparisons([old_result[2], old_result[1]])
        self.assertEqual([p.comparison.proposal_id for p in selected], ["p1", "p2"])
        self.assertEqual([p.comparison.reference_id for p in selected], ["p2", "p2"])
        self.assertEqual([p.comparison.extra_price_eur for p in selected], [Decimal("50"), Decimal("0")])
        self.assertEqual([p.model_dump_json() for p in old_result], before)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(add_comparisons([]), [])

    def test_invalid_threshold_is_rejected_even_with_empty_input(self):
        for value in (-1, 24, True, False, "9", 9.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                add_comparisons([], early_hour=value)

    def test_comparison_is_optional_for_unselected_or_previous_proposals(self):
        original = self.proposals[0].model_dump(exclude={"comparison"})
        self.assertIsNone(TripProposal.model_validate(original).comparison)

    def test_result_serializes_comparisons_and_round_trips(self):
        result = TripSearchResult(proposals=add_comparisons(self.proposals), trace=DecisionTrace())
        payload = json.loads(result.model_dump_json())
        self.assertEqual(payload["proposals"][2]["comparison"], {
            "proposal_id": "p3", "reference_id": "p1", "extra_price_eur": "100",
            "stops_delta": -2, "duration_delta_minutes": -360, "early_departures_delta": 0,
        })
        self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)

    def test_comparison_rejects_negative_extra_price_but_accepts_negative_deltas(self):
        payload = add_comparisons(self.proposals)[2].comparison.model_dump()
        self.assertEqual(ProposalComparison.model_validate(payload).stops_delta, -2)
        payload["extra_price_eur"] = Decimal("-0.01")
        with self.assertRaises(ValidationError):
            ProposalComparison.model_validate(payload)


if __name__ == "__main__":
    unittest.main()

from decimal import Decimal
from itertools import permutations
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from comparison import add_comparisons
from models import (
    DecisionTrace, DestinationPreference, ProposalTravelContext, TimePreference,
    TripProposal, TripRequest, TripSearchResult, TravelVibe,
)
from scoring import select_final_proposals
from test_scoring import tradeoff_proposal


def city_proposal(name, country, city_id, price, *, score=90, stops=0):
    proposal = tradeoff_proposal(price, stops=stops)
    proposal.destination.name = name
    proposal.destination.country = country
    proposal.destination.match_score = score
    proposal.destination.reason = f"Museums in {name}"
    proposal.destination.themes_matched = ["museums"]
    proposal.destination_city_id = city_id
    proposal.flight.destination = name
    airport = {"Paris": "ORY", "Lisbon": "LIS", "Vienna": "VIE", "Lyon": "LYS"}.get(name, "TST")
    proposal.flight.outbound.segments[-1].arrival_airport = airport
    proposal.flight.inbound.segments[0].departure_airport = airport
    proposal.hotel.destination = name
    proposal.hotel.name = f"Stay in {name}"
    proposal.travel_context = ProposalTravelContext(warnings=[f"Advice for {name}"])
    return TripProposal.model_validate(proposal.model_dump())


class OpenDestinationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.request = TripRequest(
            origin="Rome", budget_eur=900, budget_flexibility_pct=0,
            destination_pref=DestinationPreference(mode="open"),
            time_pref=TimePreference(mode="exact_dates", start_date="2026-12-05", end_date="2026-12-10"),
            vibe=TravelVibe(themes=["museums"]),
        )
        self.paris = city_proposal("Paris", "France", "/m/test_paris", "100", score=60)
        self.lisbon = city_proposal("Lisbon", "Portugal", "/m/test_lisbon", "150", score=100, stops=1)
        self.vienna = city_proposal("Vienna", "Austria", "/m/test_vienna", "200", score=90, stops=2)
        self.candidates = [self.paris, self.lisbon, self.vienna]

    def select(self, candidates=None, request=None, **kwargs):
        return select_final_proposals(
            self.candidates if candidates is None else candidates,
            self.request if request is None else request, **kwargs,
        )

    def test_cheapest_and_most_comfortable_city_does_not_remove_other_cities(self):
        result = self.select()
        self.assertEqual([p.destination.name for p in result], ["Lisbon", "Vienna", "Paris"])
        self.assertEqual([p.total_price_eur for p in result], [Decimal("706"), Decimal("756"), Decimal("656")])
        self.assertEqual(len({p.destination_city_id for p in result}), 3)

    def test_many_offers_for_one_city_produce_only_one_result(self):
        candidates = [city_proposal("Paris", "France", "/m/test_paris", str(price), stops=stops)
                      for price, stops in ((100, 2), (150, 1), (200, 0))]
        result = self.select(candidates * 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].total_price_eur, Decimal("656"))
        self.assertEqual(result[0].flight.stops, 2)

    def test_fixed_destination_still_returns_flight_tradeoffs(self):
        candidates = [city_proposal("Paris", "France", "/m/test_paris", str(price), stops=stops)
                      for price, stops in ((100, 2), (150, 1), (200, 0))]
        fixed = self.request.model_copy(update={"destination_pref": DestinationPreference(mode="fixed", destination="Paris")})
        result = self.select(candidates, request=fixed)
        self.assertEqual([p.selection_profiles for p in result], [["cheapest"], ["balanced"], ["comfortable"]])
        self.assertEqual(len(self.select(candidates)), 1)

    def test_representative_uses_total_cost_not_flight_cost(self):
        other = city_proposal("Paris", "France", "/m/test_paris", "200")
        other.hotel.total_price_eur = Decimal("400")
        other.total_price_eur = Decimal("600")
        result = self.select([self.paris, other])
        self.assertEqual(result[0].total_price_eur, Decimal("600"))
        self.assertEqual(result[0].flight.price_eur, Decimal("200"))

    def test_representative_ties_prefer_flight_comfort_then_period_then_rating(self):
        for field in ("comfort", "period", "rating"):
            with self.subTest(field=field):
                original = self.paris.model_copy(deep=True)
                preferred = original.model_copy(deep=True)
                if field == "comfort":
                    original.flight.inbound.segments[0].departure_local = original.flight.inbound.segments[0].departure_local.replace(hour=8)
                    preferred.hotel.rating = 0
                elif field == "period":
                    preferred.period.score = 100
                    preferred.hotel.rating = 0
                else:
                    preferred.hotel.rating = 5
                result = self.select([original, preferred])
                self.assertEqual(result[0].model_dump(exclude={"selection_profiles"}), preferred.model_dump(exclude={"selection_profiles"}))

    def test_representative_ties_prefer_known_zero_rating_over_missing(self):
        missing = self.paris.model_copy(deep=True)
        missing.hotel.rating = None
        rated = self.paris.model_copy(deep=True)
        rated.hotel.rating = 0
        self.assertEqual(self.select([missing, rated])[0].hotel.rating, 0)

    def test_early_hour_is_respected_when_selecting_city_representative(self):
        early = self.paris.model_copy(deep=True)
        early.flight.inbound.segments[0].departure_local = early.flight.inbound.segments[0].departure_local.replace(hour=8)
        for threshold, expected in ((9, self.paris), (8, early)):
            with self.subTest(threshold=threshold):
                selected = self.select([early, self.paris], early_hour=threshold)
                self.assertEqual(selected[0].flight, expected.flight)

    def test_city_ranking_uses_match_before_price_and_price_before_comfort(self):
        self.lisbon.destination.match_score = 60
        self.vienna.destination.match_score = 60
        self.assertEqual([p.destination.name for p in self.select()], ["Paris", "Lisbon", "Vienna"])
        self.lisbon.flight.price_eur -= Decimal("50")
        self.lisbon.total_price_eur -= Decimal("50")
        self.assertEqual([p.destination.name for p in self.select()], ["Paris", "Lisbon", "Vienna"])

    def test_aliases_with_same_provider_id_share_one_slot(self):
        alias = self.lisbon.model_copy(deep=True)
        alias.destination.name = "Lisboa"
        alias.flight.destination = alias.hotel.destination = "Lisboa"
        self.assertEqual(len(self.select([self.lisbon, alias])), 1)

    def test_distinct_ids_preserve_homonymous_cities_even_in_same_country(self):
        other = self.paris.model_copy(deep=True)
        other.destination_city_id = "/m/another_paris"
        result = self.select([self.paris, other])
        self.assertEqual({p.destination_city_id for p in result}, {"/m/test_paris", "/m/another_paris"})

    def test_legacy_identity_normalizes_name_and_country(self):
        original = self.paris.model_copy(update={"destination_city_id": None})
        alias = original.model_copy(deep=True)
        alias.destination.name = "  PARIS  "
        alias.destination.country = "  FRANCE "
        self.assertEqual(len(self.select([original, alias])), 1)

    def test_legacy_homonyms_in_different_countries_remain_distinct(self):
        original = self.paris.model_copy(update={"destination_city_id": None})
        other = original.model_copy(deep=True)
        other.destination.country = "United States"
        result = self.select([original, other])
        self.assertEqual({p.destination.country for p in result}, {"France", "United States"})

    def test_preferred_budget_filter_is_global_not_per_city(self):
        request = self.request.model_copy(update={"budget_eur": 656, "budget_flexibility_pct": 20})
        result = self.select(request=request)
        self.assertEqual([p.destination.name for p in result], ["Paris"])

    def test_flexible_budget_and_no_budget_keep_distinct_cities(self):
        for budget in (650, None):
            with self.subTest(budget=budget):
                request = self.request.model_copy(update={"budget_eur": budget, "budget_flexibility_pct": 20})
                self.assertEqual([p.destination.name for p in self.select(request=request)], ["Lisbon", "Vienna", "Paris"])

    def test_only_available_cities_are_returned_without_filling_slots(self):
        self.assertEqual(len(self.select(self.candidates[:1])), 1)
        self.assertEqual(len(self.select(self.candidates[:2])), 2)
        self.assertEqual(self.select([]), [])

    def test_at_most_three_cities_and_profiles_are_based_on_final_set(self):
        lyon = city_proposal("Lyon", "France", "/m/test_lyon", "110", score=99, stops=1)
        result = self.select([*self.candidates, lyon])
        self.assertEqual([p.destination.name for p in result], ["Lisbon", "Lyon", "Vienna"])
        self.assertEqual([p.selection_profiles for p in result], [["comfortable"], ["cheapest", "comfortable"], []])

    def test_profiles_are_global_and_not_assigned_by_rank(self):
        result = self.select()
        self.assertEqual([p.selection_profiles for p in result], [[], [], ["cheapest", "comfortable"]])

    def test_tied_global_profiles_can_apply_to_multiple_cities(self):
        lyon = city_proposal("Lyon", "France", "/m/test_lyon", "100", score=99)
        result = self.select([self.paris, lyon])
        self.assertEqual([p.selection_profiles for p in result], [["cheapest", "comfortable"]] * 2)

    def test_comparisons_reference_cheapest_even_when_it_is_not_first(self):
        selected = self.select()
        result = add_comparisons(selected)
        self.assertEqual([p.destination.name for p in result], ["Lisbon", "Vienna", "Paris"])
        self.assertEqual([p.comparison.reference_id for p in result], ["p3"] * 3)
        self.assertEqual([p.comparison.extra_price_eur for p in result], [Decimal("50"), Decimal("100"), Decimal("0")])

    def test_order_is_stable_for_permuted_inputs_including_equal_city_scores(self):
        self.paris.destination.match_score = self.lisbon.destination.match_score = self.vienna.destination.match_score = 90
        expected = self.select()
        for candidates in permutations(self.candidates):
            self.assertEqual(self.select(list(candidates)), expected)

    def test_inputs_and_quote_facts_remain_intact_and_stale_profiles_are_replaced(self):
        for proposal in self.candidates:
            proposal.selection_profiles = ["balanced"]
        before = [p.model_dump_json() for p in self.candidates]
        request_before = self.request.model_dump_json()
        result = self.select()
        for selected in result:
            original = next(p for p in self.candidates if p.destination_city_id == selected.destination_city_id)
            self.assertIsNot(selected, original)
            self.assertEqual(selected.model_dump(exclude={"selection_profiles"}), original.model_dump(exclude={"selection_profiles"}))
            self.assertNotIn("balanced", selected.selection_profiles)
        self.assertEqual([p.model_dump_json() for p in self.candidates], before)
        self.assertEqual(self.request.model_dump_json(), request_before)

    def test_city_id_model_validation_legacy_default_and_round_trip(self):
        original = self.paris.model_dump()
        for city_id in ("", "FCO", "Paris", "/m/", 123):
            with self.subTest(city_id=city_id), self.assertRaises(ValidationError):
                TripProposal.model_validate({**original, "destination_city_id": city_id})
        del original["destination_city_id"]
        self.assertIsNone(TripProposal.model_validate(original).destination_city_id)
        result = TripSearchResult(proposals=self.select(), trace=DecisionTrace())
        self.assertEqual(TripSearchResult.model_validate_json(result.model_dump_json()), result)


if __name__ == "__main__":
    unittest.main()

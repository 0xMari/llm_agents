from decimal import Decimal

from models import TripProposal, TripRequest


def score_proposal(
    proposal: TripProposal, request: TripRequest
) -> tuple[int, int, int, float, Decimal]:
    within_budget = (
        request.budget_eur is None
        or proposal.total_price_eur <= request.budget_eur
    )
    return (
        int(within_budget),
        proposal.destination.match_score,
        proposal.period.score,
        proposal.hotel.rating if proposal.hotel.rating is not None else -1,
        -proposal.total_price_eur,
    )


def flight_comfort(proposal: TripProposal, early_hour: int) -> tuple[int, int, int]:
    """Lower is better: stops, early leg departures, provider duration."""
    legs = (proposal.flight.outbound, proposal.flight.inbound)
    early_departures = sum(
        leg.segments[0].departure_local.hour < early_hour for leg in legs
    )
    return (
        proposal.flight.stops,
        early_departures,
        sum(leg.duration_minutes for leg in legs),
    )


def select_final_proposals(
    proposals: list[TripProposal],
    request: TripRequest,
    *,
    early_hour: int = 9,
) -> list[TripProposal]:
    """Select at most three tradeoffs from candidates already filtered by the planner."""
    if type(early_hour) is not int or not 0 <= early_hour <= 23:
        raise ValueError("early_hour deve essere un intero tra 0 e 23.")
    if not proposals:
        return []

    # Flexible-budget offers are considered only when none meet the preferred budget.
    preferred = [
        proposal for proposal in proposals
        if request.budget_eur is None or proposal.total_price_eur <= request.budget_eur
    ]
    candidates = sorted(
        preferred or proposals,
        key=lambda proposal: score_proposal(proposal, request),
        reverse=True,
    )

    def comfort(proposal: TripProposal) -> tuple[int, int, int]:
        return flight_comfort(proposal, early_hour)

    cheapest = min(candidates, key=lambda proposal: (proposal.total_price_eur, comfort(proposal)))
    comfortable = min(candidates, key=lambda proposal: (comfort(proposal), proposal.total_price_eur))
    if cheapest is comfortable:
        return [cheapest.model_copy(
            update={"selection_profiles": ["cheapest", "comfortable"]}
        )]

    # Strict inequalities prevent duplicate profiles and an artificial third option.
    middle = [
        proposal for proposal in candidates
        if cheapest.total_price_eur < proposal.total_price_eur < comfortable.total_price_eur
        and comfort(comfortable) < comfort(proposal) < comfort(cheapest)
    ]
    choices = [(cheapest, "cheapest")]
    if middle:
        balanced = min(middle, key=lambda proposal: (proposal.total_price_eur, comfort(proposal)))
        choices.append((balanced, "balanced"))
    choices.append((comfortable, "comfortable"))
    return [
        proposal.model_copy(update={"selection_profiles": [profile]})
        for proposal, profile in choices
    ]

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


def _destination_key(proposal: TripProposal) -> tuple[str, ...]:
    if proposal.destination_city_id is not None:
        return ("city_id", proposal.destination_city_id)
    # Older saved proposals lack the provider ID; keep homonymous countries apart.
    return (
        "name",
        " ".join(proposal.destination.name.casefold().split()),
        " ".join((proposal.destination.country or "").casefold().split()),
    )


def _select_distinct_destinations(
    candidates: list[TripProposal], *, early_hour: int,
) -> list[TripProposal]:
    def representative_key(proposal: TripProposal) -> tuple:
        rating = proposal.hotel.rating if proposal.hotel.rating is not None else -1
        return (
            proposal.total_price_eur,
            flight_comfort(proposal, early_hour),
            -proposal.period.score,
            -rating,
        )

    representatives = {}
    for proposal in candidates:
        key = _destination_key(proposal)
        current = representatives.get(key)
        if current is None or representative_key(proposal) < representative_key(current):
            representatives[key] = proposal

    selected = sorted(
        representatives.values(),
        key=lambda proposal: (
            -proposal.destination.match_score,
            *representative_key(proposal),
            _destination_key(proposal),
        ),
    )[:3]
    cheapest_price = min(p.total_price_eur for p in selected)
    best_comfort = min(flight_comfort(p, early_hour) for p in selected)
    result = []
    for proposal in selected:
        # Labels compare only the final set, not each city's discarded variants.
        profiles = []
        if proposal.total_price_eur == cheapest_price:
            profiles.append("cheapest")
        if flight_comfort(proposal, early_hour) == best_comfort:
            profiles.append("comfortable")
        result.append(proposal.model_copy(update={"selection_profiles": profiles}))
    return result


def select_final_proposals(
    proposals: list[TripProposal],
    request: TripRequest,
    *,
    early_hour: int = 9,
) -> list[TripProposal]:
    """Select city alternatives or flight tradeoffs from planner-filtered quotes."""
    if type(early_hour) is not int or not 0 <= early_hour <= 23:
        raise ValueError("early_hour must be an integer between 0 and 23.")
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

    if request.destination_pref.mode == "open":
        return _select_distinct_destinations(candidates, early_hour=early_hour)

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

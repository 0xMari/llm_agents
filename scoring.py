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


def select_top_proposals(
    proposals: list[TripProposal],
    request: TripRequest,
    limit: int = 3,
) -> list[TripProposal]:
    sorted_proposals = sorted(
        proposals,
        key=lambda proposal: score_proposal(proposal, request),
        reverse=True,
    )

    return sorted_proposals[:limit]

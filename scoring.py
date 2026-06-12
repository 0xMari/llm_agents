from models import TripProposal


def score_proposal(proposal: TripProposal) -> tuple[int, int, float, int]:
    return (
        proposal.destination.match_score,
        proposal.period.score,
        proposal.hotel.rating,
        -proposal.total_price_eur,
    )


def select_top_proposals(
    proposals: list[TripProposal],
    limit: int = 3,
) -> list[TripProposal]:
    sorted_proposals = sorted(
        proposals,
        key=score_proposal,
        reverse=True,
    )

    return sorted_proposals[:limit]
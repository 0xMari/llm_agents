from models import ProposalComparison, TripProposal
from scoring import flight_comfort


def add_comparisons(
    proposals: list[TripProposal], *, early_hour: int = 9
) -> list[TripProposal]:
    """Compare selected trips with the cheapest; IDs are local to this result."""
    if type(early_hour) is not int or not 0 <= early_hour <= 23:
        raise ValueError("early_hour must be an integer between 0 and 23.")
    if not proposals:
        return []

    reference_index = min(
        range(len(proposals)), key=lambda index: proposals[index].total_price_eur
    )
    reference = proposals[reference_index]
    ref_stops, ref_early, ref_duration = flight_comfort(reference, early_hour)
    result = []
    for index, proposal in enumerate(proposals):
        stops, early, duration = flight_comfort(proposal, early_hour)
        comparison = ProposalComparison(
            proposal_id=f"p{index + 1}",
            reference_id=f"p{reference_index + 1}",
            extra_price_eur=proposal.total_price_eur - reference.total_price_eur,
            stops_delta=stops - ref_stops,
            duration_delta_minutes=duration - ref_duration,
            early_departures_delta=early - ref_early,
        )
        result.append(proposal.model_copy(update={"comparison": comparison}))
    return result

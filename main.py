from models import (
    DestinationPreference,
    TimePreference,
    TripRequest,
    TravelVibe,
)

from planning import build_proposal



if __name__ == "__main__":
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
            themes = ["food", "museums"],
        ),
    )

    try:
        result = build_proposal(request)
        print(result.model_dump_json(indent=2))
    except ValueError as error:
        print(f"Impossibile creare una proposta: {error}")

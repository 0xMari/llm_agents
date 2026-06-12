from models import TripRequest, PlanningScenario


def route_request(request: TripRequest) -> PlanningScenario:
    if request.destination_pref.mode == "fixed" and request.time_pref.mode == "exact_dates":
        return PlanningScenario.FIXED_DESTINATION_EXACT_DATES

    if request.destination_pref.mode == "fixed":
        return PlanningScenario.FIXED_DESTINATION_FLEXIBLE_DATES

    if request.time_pref.mode == "exact_dates":
        return PlanningScenario.OPEN_DESTINATION_EXACT_DATES

    return PlanningScenario.OPEN_DESTINATION_FLEXIBLE_DATES
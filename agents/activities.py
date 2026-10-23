# Simulated agents.

from datetime import date

from models import SuggestedTimePeriod, TripRequest


def suggest_periods(request: TripRequest) -> list[SuggestedTimePeriod]:
    return [
        SuggestedTimePeriod(
            start = date(2026, 9, 15),
            end = date(2026, 9, 20),
            reason = "Mild temperatures and generally affordable prices.",
            score = 90,
        ),
        SuggestedTimePeriod(
            start = date(2026, 10, 6),
            end = date(2026, 10, 11),
            reason = "A quieter period suited to museums and local food.",
            score = 82,
        ),
    ]


def suggest_activities(request: TripRequest) -> list[str]:
    recommendations = ["Book a rate with free cancellation."]
    themes = request.vibe.themes

    if "food" in themes:
        recommendations.append("Explore local markets and restaurants.")

    if "museums" in themes or "culture" in themes:
        recommendations.append("Check museum opening days and reservation requirements.")

    return recommendations

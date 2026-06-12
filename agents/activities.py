# agenti simulati

from datetime import date

from models import SuggestedTimePeriod, TripRequest


def suggest_periods(request: TripRequest) -> list[SuggestedTimePeriod]:
    return [
        SuggestedTimePeriod(
            start = date(2026, 9, 15),
            end = date(2026, 9, 20),
            reason = "Temperature miti e prezzi generalmente convenienti.",
            score = 90,
        ),
        SuggestedTimePeriod(
            start = date(2026, 10, 6),
            end = date(2026, 10, 11),
            reason = "Periodo tranquillo, adatto a musei e gastronomia.",
            score = 82,
        ),
    ]


def suggest_activities(request: TripRequest) -> list[str]:
    recommendations = ["Prenota una tariffa con cancellazione gratuita."]
    themes = request.vibe.themes

    if "food" in themes:
        recommendations.append("Esplora i mercati e i ristoranti locali.")

    if "museums" in themes or "culture" in themes:
        recommendations.append("Controlla giorni di apertura e prenotazioni dei musei.")

    return recommendations

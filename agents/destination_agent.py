from models import SuggestedDestination, TripRequest

# solo per i test prima di llm

def suggest_destinations(request: TripRequest) -> list[SuggestedDestination]:
    themes = request.vibe.themes
    free_text = request.vibe.free_text or ""

    if "nature" in themes or "natura" in free_text.lower():
        return [
            SuggestedDestination(
                name = "Lisbon",
                country = "Portugal",
                reason = "Buona opzione dimostrativa per una vacanza rilassata con costa e città.",
                match_score = 70,
                themes_matched = ["nature"],
            )
        ]

    return [
        SuggestedDestination(
            name = "Lisbon",
            country = "Portugal",
            reason = "Destinazione dimostrativa disponibile nei dati mock.",
            match_score = 60,
            themes_matched = [],
        )
    ]
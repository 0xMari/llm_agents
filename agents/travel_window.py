import json
from datetime import date

from models import SuggestedDestination, TravelWindowAdvice, TripRequest

from llm.openrouter import call_openrouter_json

TRAVEL_WINDOW_ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_periods": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"},
                    "reason": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["start", "end", "reason", "score"],
                "additionalProperties": False,
            },
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "format": "date"},
                    "end": {"type": "string", "format": "date"},
                    "reason": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["start", "end", "reason", "score"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["selected_periods", "warnings", "alternatives", "rationale"],
    "additionalProperties": False,
}


def build_system_prompt() -> str:
    return """
Sei un travel window advisor per un sistema AI di pianificazione viaggi.

Il tuo compito è valutare se il periodo richiesto dall'utente è adatto alla destinazione,
considerando clima, stagionalità, eventi, affollamento, vibe della vacanza e vincoli utente.

Devi restituire solo JSON valido conforme allo schema.
Non restituire markdown.
Non inventare voli, hotel o prezzi.

Regole:
- Se l'utente ha date esatte, selected_periods deve usare quelle date.
- Se le date esatte non sono ideali, aggiungi warnings e alternatives.
- Se l'utente indica mese, stagione o finestra flessibile, proponi periodi coerenti.
- Se time_preference.mode è "month", usa time_context.resolved_year come anno obbligatorio per selected_periods.
- Non inventare anni diversi da quelli forniti nel time_context.
- Rispetta min_days e max_days quando presenti.
- min_days e max_days indicano giorni inclusivi: conta sia il giorno di partenza sia quello di ritorno.
- Calcola la durata come (end - start).days + 1.
- Esempio: dal 15 al 20 settembre sono 6 giorni.
- Usa score alto per periodi molto adatti, basso per periodi problematici.
"""


def resolve_month_year(month: int, reference_date: date) -> int:
    if month >= reference_date.month:
        return reference_date.year

    return reference_date.year + 1


def build_time_context(request: TripRequest, reference_date: date) -> dict:
    time_context = {
        "reference_date": reference_date.isoformat(),
    }

    if request.time_pref.mode == "month" and request.time_pref.month is not None:
        time_context["resolved_year"] = resolve_month_year(
            month=request.time_pref.month,
            reference_date=reference_date,
        )

    return time_context


def build_user_prompt(
    request: TripRequest,
    destination: SuggestedDestination,
    reference_date: date,
) -> str:
    payload = {
        "origin": request.origin,
        "budget_eur": request.budget_eur,
        "budget_flexibility_pct": request.budget_flexibility_pct,
        "destination": destination.model_dump(mode="json"),
        "time_preference": request.time_pref.model_dump(mode="json"),
        "time_context": build_time_context(request, reference_date),
        "vibe": request.vibe.model_dump(mode="json"),
    }

    return json.dumps(payload, ensure_ascii=False)


class TravelWindowConstraintError(ValueError):
    pass

def validate_travel_window_constraints(
    request: TripRequest,
    advice: TravelWindowAdvice,
    reference_date: date,
) -> None:
    if request.time_pref.mode == "exact_dates":
        expected_start = request.time_pref.start_date
        expected_end = request.time_pref.end_date

        for period in advice.selected_periods:
            if period.start != expected_start or period.end != expected_end:
                raise TravelWindowConstraintError(
                    "L'LLM ha violato il vincolo di date esatte indicato dall'utente."
                )

    if request.time_pref.mode == "month" and request.time_pref.month is not None:
        expected_year = resolve_month_year(
            month=request.time_pref.month,
            reference_date=reference_date,
        )

        for period in advice.selected_periods:
            if period.start.year != expected_year or period.end.year != expected_year:
                raise TravelWindowConstraintError(
                    "L'LLM ha usato un anno diverso da quello risolto dal sistema."
                )

            if period.start.month != request.time_pref.month:
                raise TravelWindowConstraintError(
                    "L'LLM ha usato un mese diverso da quello indicato dall'utente."
                )

    min_days = request.time_pref.min_days
    max_days = request.time_pref.max_days

    for period in advice.selected_periods:
        duration_days = (period.end - period.start).days + 1

        if min_days is not None and duration_days < min_days:
            raise TravelWindowConstraintError(
                "L'LLM ha proposto un periodo più corto della durata minima richiesta."
            )

        if max_days is not None and duration_days > max_days:
            raise TravelWindowConstraintError(
                "L'LLM ha proposto un periodo più lungo della durata massima richiesta."
            )


def evaluate_travel_window(
    request: TripRequest,
    destination: SuggestedDestination,
    reference_date: date | None = None,
) -> TravelWindowAdvice:
    resolved_reference_date = reference_date or date.today()

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                request,
                destination,
                resolved_reference_date,
            )
        },
    ]

    raw_json = call_openrouter_json(
        messages=messages,
        response_schema=TRAVEL_WINDOW_ADVICE_SCHEMA,
    )

    advice = TravelWindowAdvice.model_validate_json(raw_json)
    validate_travel_window_constraints(request, advice, resolved_reference_date)

    return advice

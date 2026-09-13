from models import SuggestedDestination, TripRequest, DestinationAdvice
import json
from datetime import date
from agents.travel_window import build_time_context
from llm.openrouter import call_openrouter_json


SYSTEM_PROMPT = (
    "Proponi da 1 a 3 citta distinte considerando origine, periodo, durata "
    "in giorni inclusivi, vibe, budget e flessibilita. "
    "Il budget riguarda il totale volo e hotel. "
    "Se budget_eur e null, non e indicato un limite economico: ignora la "
    "flessibilita e non dedurre una preferenza per il lusso. "
    "Se il budget e presente, flessibilita zero indica un limite rigido; "
    "una percentuale positiva indica un budget preferito con limite massimo "
    "budget_eur * (100 + budget_flexibility_pct) // 100 in euro interi. "
    "Se manca la vibe, proponi opzioni diversificate spiegando il criterio. "
    "Usa nomi di citta e paesi in inglese, motivazioni in italiano. "
    "match_score indica compatibilita stimata da 0 a 100; "
    "themes_matched contiene solo temi pertinenti alla richiesta. "
    "Non inventare prezzi, offerte o disponibilita; non garantire il budget. "
    "Il planner verifica i vincoli economici sui prezzi dei provider. "
    "Restituisci solo il JSON previsto dallo schema."
)

def suggest_destinations(request: TripRequest) -> list[SuggestedDestination]:
    payload = request.model_dump(mode="json")
    payload["time_context"] = build_time_context(request, date.today())
    raw_json = call_openrouter_json(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_schema=DestinationAdvice.model_json_schema(),
        schema_name="destination_advice",
    )
    advice = DestinationAdvice.model_validate_json(raw_json)
    return [SuggestedDestination(**d.model_dump()) for d in advice.destinations]

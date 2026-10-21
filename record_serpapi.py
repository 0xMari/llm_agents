import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from models import HotelSearchQuery
from providers.flights import FlightLocationError
from providers.serpapi_client import SerpAPIError, search as live_search
from try_serpapi import main as run_demo


FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures" / "serpapi"
DEFAULT_OUTPUT = FIXTURE_DIRECTORY / "hotels_real.json"
MAX_CALLS = 5


class FixtureRecordingError(RuntimeError):
    pass


def build_hotel_fixture(params: dict, response: dict) -> dict:
    if params.get("engine") != "google_hotels" or params.get("currency") != "EUR":
        raise FixtureRecordingError("Il registratore supporta soltanto hotel in EUR.")
    location = params.get("q")
    if not isinstance(location, str) or not location.startswith("Hotels in "):
        raise FixtureRecordingError("Query hotel inattesa: nessuna fixture scritta.")
    items = response.get("properties")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise FixtureRecordingError("Risposta non in formato elenco hotel: da analizzare.")
    if not items:
        raise FixtureRecordingError("Nessun hotel: non e una fixture per il percorso positivo.")

    query = HotelSearchQuery(
        destination=location.removeprefix("Hotels in "),
        location_name=location.removeprefix("Hotels in "),
        checkin_date=params["check_in_date"],
        checkout_date=params["check_out_date"],
        adults=params["adults"],
    )
    properties = []
    for item in items:
        # Allowlist also inside prices: URLs, tokens and metadata never reach disk.
        clean = {key: item[key] for key in ("name", "overall_rating") if key in item}
        for key in ("rate_per_night", "total_rate"):
            if key not in item:
                continue
            rate = item[key]
            if rate is not None and not isinstance(rate, dict):
                raise FixtureRecordingError("Formato del prezzo inatteso: nessuna fixture scritta.")
            clean[key] = None if rate is None else {
                field: rate[field]
                for field in ("extracted_lowest", "extracted_before_taxes_fees")
                if field in rate
            }
        values = [clean[key] for key in ("name", "overall_rating") if key in clean]
        for key in ("rate_per_night", "total_rate"):
            if isinstance(clean.get(key), dict):
                values.extend(clean[key].values())
        if any(value is not None and not isinstance(value, (str, int, float)) for value in values):
            raise FixtureRecordingError("Campi non scalari inattesi: nessuna fixture scritta.")
        properties.append(clean)

    return {
        "source": {
            "provider": "serpapi",
            "engine": "google_hotels",
            "currency": "EUR",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "query": query.model_dump(mode="json"),
        "response": {"properties": properties},
    }


def record_hotels(output: Path, demo_args: list[str]) -> int:
    if output.exists():
        raise FixtureRecordingError("Il file esiste gia: scegli un altro nome con --output.")
    calls = 0
    captured = False

    def recording_search(params):
        nonlocal calls, captured
        if calls >= MAX_CALLS:
            raise FixtureRecordingError("Limite di cinque chiamate raggiunto: ricerca interrotta.")
        calls += 1
        response = live_search(params)
        if params.get("engine") == "google_hotels":
            fixture = build_hotel_fixture(params, response)
            try:
                payload = json.dumps(fixture, ensure_ascii=False, indent=2, allow_nan=False)
            except ValueError:
                raise FixtureRecordingError("Valori JSON non validi: nessuna fixture scritta.") from None
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as file:
                file.write(payload + "\n")
            captured = True
            print(f"Fixture hotel registrata: {output}")
        return response

    # Patch only the manual run; the actual HTTP client and parsers still execute.
    with patch("providers.flights.search_serpapi", side_effect=recording_search), patch(
        "providers.hotels.search_serpapi", side_effect=recording_search
    ):
        run_demo([*demo_args, "--max-outbound", "1", "--hotels"])
    if not captured:
        raise FixtureRecordingError("Nessuna fixture registrata: non e stata raggiunta la ricerca hotel.")
    print(f"Chiamate SerpAPI eseguite: {calls}/{MAX_CALLS}. Controlla la fixture prima del commit.")
    return calls


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Registra una fixture hotel reale: fino a cinque chiamate SerpAPI, nessuna chiamata LLM."
    )
    parser.add_argument("--origin", default="Rome, Italy")
    parser.add_argument("--destination", default="Lisbon, Portugal")
    parser.add_argument("--origin-city-id", help="Seleziona una citta tra i risultati autocomplete dell'origine.")
    parser.add_argument("--destination-city-id", help="Seleziona una citta tra i risultati autocomplete della destinazione.")
    parser.add_argument("--departure-date", type=date.fromisoformat)
    parser.add_argument("--return-date", type=date.fromisoformat)
    parser.add_argument("--adults", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    demo_args = ["--origin", args.origin, "--destination", args.destination, "--adults", str(args.adults)]
    if args.origin_city_id is not None:
        demo_args.extend(["--origin-city-id", args.origin_city_id])
    if args.destination_city_id is not None:
        demo_args.extend(["--destination-city-id", args.destination_city_id])
    if args.departure_date is not None:
        demo_args.extend(["--departure-date", args.departure_date.isoformat()])
    if args.return_date is not None:
        demo_args.extend(["--return-date", args.return_date.isoformat()])
    record_hotels(args.output.expanduser().resolve(), demo_args)


if __name__ == "__main__":
    try:
        main()
    except FlightLocationError as error:
        print("Localita da chiarire:", error)
        for option in error.options:
            print(option)
        raise SystemExit(1) from None
    except (FixtureRecordingError, SerpAPIError, ValidationError, OSError) as error:
        raise SystemExit(str(error)) from None

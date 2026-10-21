from decimal import Decimal, InvalidOperation
import re

from models import (
    AIRPORT_CODE_PATTERN,
    CITY_ID_PATTERN,
    FlightOffer,
    FlightSearchQuery,
    FlightLeg,
    FlightSegment,
)
from providers.serpapi_client import SerpAPIResponseError, search as search_serpapi
from providers.links import parse_redirect


class FlightLocationError(ValueError):
    def __init__(self, message: str, options: list[str] | None = None):
        super().__init__(message)
        self.options = options or []


def _result_items(payload: dict, *keys: str) -> list[dict]:
    if not any(key in payload for key in keys):
        metadata = payload.get("search_metadata", {})
        if isinstance(metadata, dict) and metadata.get("status") == "Success" and payload.get("error"):
            return []
        raise SerpAPIResponseError("La risposta non contiene i risultati attesi.")
    items = []
    for key in keys:
        group = payload.get(key, [])
        if not isinstance(group, list) or any(not isinstance(item, dict) for item in group):
            raise SerpAPIResponseError("Formato dei risultati SerpAPI inatteso.")
        items.extend(group)
    return items


def _validate_city(city: dict) -> None:
    if (
        city.get("type") != "city"
        or not isinstance(city.get("name"), str)
        or not city["name"].strip()
        or not isinstance(city.get("id"), str)
        or not re.fullmatch(CITY_ID_PATTERN, city["id"])
    ):
        raise SerpAPIResponseError("Identificativo o nome della citta non valido.")
    airports = city.get("airports", [])
    if not isinstance(airports, list) or any(
        not isinstance(airport, dict)
        or not isinstance(airport.get("id"), str)
        or not re.fullmatch(AIRPORT_CODE_PATTERN, airport["id"])
        for airport in airports
    ):
        raise SerpAPIResponseError("Elenco degli aeroporti non valido.")


def find_city_options(text: str) -> list[dict]:
    if not text.strip():
        raise FlightLocationError("Specifica una citta da cercare.")
    response = search_serpapi({
        "engine": "google_flights_autocomplete",
        "q": text,
        "exclude_regions": True,
        "hl": "en",
        "gl": "it",
    })
    cities = {}
    for item in _result_items(response, "suggestions"):
        # Autocomplete can include untyped airports even with exclude_regions.
        if item.get("type") != "city":
            continue
        _validate_city(item)
        cities[item["id"]] = item
    return list(cities.values())


def resolve_city(text: str, *, city_id: str | None = None) -> dict:
    cities = find_city_options(text)
    normalized = " ".join(text.casefold().split())
    if city_id is not None:
        matches = [city for city in cities if city["id"] == city_id]
    else:
        matches = []
        for city in cities:
            name = " ".join(city["name"].casefold().split())
            if name == normalized or ("," not in normalized and name.split(",")[0] == normalized):
                matches.append(city)
    if len(matches) != 1:
        raise FlightLocationError(
            f"Localita non risolta: {text!r}. Specifica la citta e il paese oppure scegli un city_id verificato.",
            options=[f"{city['name']} ({city['id']})" for city in cities],
        )
    return matches[0]


def search_id_for(city: dict, airport_code: str | None = None) -> str:
    _validate_city(city)
    if airport_code is None:
        return city["id"]
    code = airport_code.strip().upper()
    if code not in {airport["id"] for airport in city.get("airports", [])}:
        raise FlightLocationError("Aeroporto richiesto non verificato per questa citta.")
    return code


def parse_flight_leg(raw_leg: dict) -> FlightLeg:
    try:
        segments = [
            FlightSegment(
                departure_airport=item["departure_airport"]["id"],
                arrival_airport=item["arrival_airport"]["id"],
                departure_local=item["departure_airport"]["time"],
                arrival_local=item["arrival_airport"]["time"],
                airline=item["airline"],
                flight_number=item["flight_number"],
            )
            for item in raw_leg["flights"]
        ]
        return FlightLeg(segments=segments, duration_minutes=raw_leg["total_duration"])
    except (KeyError, TypeError, ValueError):
        raise SerpAPIResponseError("Segmenti o durata del volo non validi.") from None


def _validate_leg(leg: FlightLeg, query: FlightSearchQuery, *, returning: bool) -> None:
    expected_date = query.return_date if returning else query.departure_date
    origin_id = query.destination_id if returning else query.origin_id
    destination_id = query.origin_id if returning else query.destination_id
    if leg.segments[0].departure_local.date() != expected_date:
        raise SerpAPIResponseError("Il volo non rispetta la data richiesta.")
    for location_id, airport in (
        (origin_id, leg.segments[0].departure_airport),
        (destination_id, leg.segments[-1].arrival_airport),
    ):
        if re.fullmatch(AIRPORT_CODE_PATTERN, location_id) and location_id != airport:
            raise SerpAPIResponseError("Il volo non rispetta il vincolo aeroportuale.")


def _select_outbounds(items: list[dict], query: FlightSearchQuery, limit: int) -> list[tuple[str, FlightLeg]]:
    first_routes, remaining = [], []
    routes, tokens = set(), set()
    for item in items:
        token = item.get("departure_token")
        if not isinstance(token, str) or not token:
            raise SerpAPIResponseError("Risultato senza departure_token utilizzabile.")
        if token in tokens:
            continue
        tokens.add(token)
        leg = parse_flight_leg(item)
        _validate_leg(leg, query, returning=False)
        route = (leg.segments[0].departure_airport, leg.segments[-1].arrival_airport)
        # Prefer distinct airport pairs before exploring more flights on the same route.
        (remaining if route in routes else first_routes).append((token, leg))
        routes.add(route)
    return (first_routes + remaining)[:limit]


def search_flights(query: FlightSearchQuery, *, max_outbound: int = 3) -> list[FlightOffer]:
    if type(max_outbound) is not int or max_outbound < 1:
        raise ValueError("max_outbound deve essere un intero positivo.")
    params = {
        "engine": "google_flights", "type": 1, "currency": "EUR",
        "adults": query.adults, "hl": "it", "gl": "it",
        "departure_id": query.origin_id, "arrival_id": query.destination_id,
        "outbound_date": query.departure_date.isoformat(),
        "return_date": query.return_date.isoformat(),
    }
    response = search_serpapi(params)
    metadata = response.get("search_metadata")
    redirect = parse_redirect(
        metadata.get("google_flights_url") if isinstance(metadata, dict) else None,
        kind="search",
    )
    items = _result_items(response, "best_flights", "other_flights")
    offers = []
    for token, outbound in _select_outbounds(items, query, max_outbound):
        returns = search_serpapi({**params, "departure_token": token})
        for raw_inbound in _result_items(returns, "best_flights", "other_flights"):
            inbound = parse_flight_leg(raw_inbound)
            _validate_leg(inbound, query, returning=True)
            segments = outbound.segments + inbound.segments
            try:
                offer = FlightOffer(
                    origin=query.origin, destination=query.destination,
                    outbound=outbound, inbound=inbound,
                    departure_date=outbound.segments[0].departure_local.date(),
                    return_date=inbound.segments[0].departure_local.date(),
                    airline=", ".join(dict.fromkeys(segment.airline for segment in segments)),
                    price_eur=Decimal(str(raw_inbound["price"])),
                    stops=outbound.stops + inbound.stops,
                    redirect=redirect,
                )
            except (KeyError, ValueError, InvalidOperation):
                raise SerpAPIResponseError("Prezzo o dati dell'offerta A/R non validi.") from None
            if query.max_price_eur is None or offer.price_eur <= query.max_price_eur:
                offers.append(offer)
    return offers

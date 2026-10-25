from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from execution import RunBudget

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
from providers.location_verification import CityCountryVerifier, CountryMatch


# The autocomplete's "airports" list also contains transport IDs such as 4GS.
AUTOCOMPLETE_LOCATION_PATTERN = r"^[A-Z0-9]{3}$"


class FlightLocationError(ValueError):
    def __init__(
        self, message: str, options: list[str] | None = None,
        *, country_status: CountryMatch | None = None,
    ):
        super().__init__(message)
        self.options = options or []
        self.country_status = country_status


def _result_items(payload: dict, *keys: str) -> list[dict]:
    if not any(key in payload for key in keys):
        metadata = payload.get("search_metadata", {})
        if isinstance(metadata, dict) and metadata.get("status") == "Success" and payload.get("error"):
            return []
        raise SerpAPIResponseError("The response does not contain the expected results.", kind="missing_results")
    items = []
    for key in keys:
        group = payload.get(key, [])
        if not isinstance(group, list) or any(not isinstance(item, dict) for item in group):
            raise SerpAPIResponseError("Unexpected SerpAPI result format.", kind="invalid_results")
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
        raise SerpAPIResponseError("Invalid city identifier or name.", kind="invalid_city")
    airports = city.get("airports", [])
    if not isinstance(airports, list) or any(
        not isinstance(airport, dict)
        or not isinstance(airport.get("id"), str)
        or not re.fullmatch(AUTOCOMPLETE_LOCATION_PATTERN, airport["id"])
        for airport in airports
    ):
        raise SerpAPIResponseError("Invalid list of transport locations.", kind="invalid_city_locations")


def find_city_options(text: str, *, budget: RunBudget | None = None) -> list[dict]:
    if not text.strip():
        raise FlightLocationError("Specify a city to search for.")
    response = search_serpapi({
        "engine": "google_flights_autocomplete",
        "q": text,
        "exclude_regions": True,
        "hl": "en",
        "gl": "it",
    }, budget=budget)
    cities = {}
    for item in _result_items(response, "suggestions"):
        # Autocomplete can include untyped airports even with exclude_regions.
        if item.get("type") != "city":
            continue
        _validate_city(item)
        cities[item["id"]] = item
    return list(cities.values())


def _normalized_location_name(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).split())


def _country_status(city: dict, country: str) -> CountryMatch:
    expected = _normalized_location_name(country)
    label = _normalized_location_name(city["name"])
    if "," in label:
        suffix = label.rsplit(",", 1)[1].strip()
        if not suffix:
            return CountryMatch.UNKNOWN
        return CountryMatch.MATCH if suffix == expected else CountryMatch.MISMATCH

    description = city.get("description")
    if not isinstance(description, str):
        return CountryMatch.UNKNOWN
    # Autocomplete uses hl=en; only explicit country descriptions are accepted.
    description = _normalized_location_name(description).rstrip(".")
    for prefix in ("city in ", "town in ", "capital of ", "municipality in "):
        if description.startswith(prefix):
            place = description[len(prefix):].strip()
            if place:
                # A different or more specific place is not permission to override it.
                return CountryMatch.MATCH if place == expected else CountryMatch.MISMATCH
    return CountryMatch.UNKNOWN


def _unresolved_city(
    cities: list[dict], name: str, country: str | None,
    *, country_status: CountryMatch | None = None,
) -> FlightLocationError:
    text = ", ".join(filter(None, [name, country]))
    return FlightLocationError(
        f"Location unresolved: {text!r}. Specify the city and country or select a verified city_id.",
        options=[f"{city['name']} ({city['id']})" for city in cities],
        country_status=country_status,
    )


def select_city_candidate(
    cities: list[dict], name: str, *,
    country: str | None = None, city_id: str | None = None,
) -> dict:
    wanted = _normalized_location_name(name)
    matches = []
    rejected_country = False
    for city in cities:
        label = _normalized_location_name(city["name"])
        if city_id is not None:
            compatible = city["id"] == city_id
        elif country is not None:
            compatible = label.split(",")[0].strip() == wanted
        else:
            compatible = label == wanted or (
                "," not in wanted and label.split(",")[0].strip() == wanted
            )
        if compatible and country is not None and _country_status(city, country) is CountryMatch.MISMATCH:
            rejected_country = True
            compatible = False
        if compatible:
            matches.append(city)
    if len(matches) != 1:
        raise _unresolved_city(
            cities, name, country,
            country_status=CountryMatch.MISMATCH if not matches and rejected_country else None,
        )
    if country is not None and city_id is None and _country_status(matches[0], country) is CountryMatch.UNKNOWN:
        raise _unresolved_city(cities, name, country, country_status=CountryMatch.UNKNOWN)
    return matches[0]


def resolve_city(
    text: str, *, country: str | None = None,
    city_id: str | None = None, budget: RunBudget | None = None,
    verifier: CityCountryVerifier | None = None,
) -> dict:
    name = text.strip()
    if country is not None:
        country = country.strip()
        if not country:
            raise FlightLocationError("The country cannot be empty.")
        prefix, separator, suffix = name.rpartition(",")
        if separator and _normalized_location_name(suffix) == _normalized_location_name(country):
            name = prefix.strip()

    query = f"{name}, {country}" if country is not None else name
    cities = find_city_options(query, budget=budget)
    try:
        return select_city_candidate(cities, name, country=country, city_id=city_id)
    except FlightLocationError as error:
        if (
            verifier is None or country is None or city_id is not None
            or error.country_status is not CountryMatch.UNKNOWN
        ):
            raise
        # The pure selector reports UNKNOWN only for a single eligible candidate.
        candidate = next(
            city for city in cities
            if _normalized_location_name(city["name"]).split(",")[0].strip() == _normalized_location_name(name)
            and _country_status(city, country) is CountryMatch.UNKNOWN
        )
        status = verifier.verify(deepcopy(candidate), country, budget=budget)
        if not isinstance(status, CountryMatch):
            raise TypeError("The country verifier must return CountryMatch.")
        if status is CountryMatch.MATCH:
            return candidate
        raise _unresolved_city(cities, name, country, country_status=status) from None


def search_id_for(city: dict, airport_code: str | None = None) -> str:
    _validate_city(city)
    if airport_code is None:
        return city["id"]
    code = airport_code.strip().upper()
    if not re.fullmatch(AIRPORT_CODE_PATTERN, code):
        raise FlightLocationError("Specify a three-letter airport code.")
    if code not in {airport["id"] for airport in city.get("airports", [])}:
        raise FlightLocationError("The requested airport has not been verified for this city.")
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
        raise SerpAPIResponseError("Invalid flight segments or duration.", kind="invalid_flight_leg") from None


def _validate_leg(leg: FlightLeg, query: FlightSearchQuery, *, returning: bool) -> None:
    expected_date = query.return_date if returning else query.departure_date
    origin_id = query.destination_id if returning else query.origin_id
    destination_id = query.origin_id if returning else query.destination_id
    if leg.segments[0].departure_local.date() != expected_date:
        raise SerpAPIResponseError("The flight does not match the requested date.", kind="flight_date_mismatch")
    for location_id, airport in (
        (origin_id, leg.segments[0].departure_airport),
        (destination_id, leg.segments[-1].arrival_airport),
    ):
        if re.fullmatch(AIRPORT_CODE_PATTERN, location_id) and location_id != airport:
            raise SerpAPIResponseError("The flight does not match the airport constraint.", kind="flight_airport_mismatch")


def _select_outbounds(items: list[dict], query: FlightSearchQuery, limit: int) -> list[tuple[str, FlightLeg]]:
    first_routes, remaining = [], []
    routes, tokens = set(), set()
    for item in items:
        token = item.get("departure_token")
        if not isinstance(token, str) or not token:
            raise SerpAPIResponseError("The result has no usable departure_token.", kind="missing_departure_token")
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


def search_flights(
    query: FlightSearchQuery, *, max_outbound: int = 3, budget: RunBudget | None = None,
) -> list[FlightOffer]:
    if type(max_outbound) is not int or max_outbound < 1:
        raise ValueError("max_outbound must be a positive integer.")
    params = {
        "engine": "google_flights", "type": 1, "currency": "EUR",
        "adults": query.adults, "hl": "en", "gl": "it",
        "departure_id": query.origin_id, "arrival_id": query.destination_id,
        "outbound_date": query.departure_date.isoformat(),
        "return_date": query.return_date.isoformat(),
    }
    response = search_serpapi(params, budget=budget)
    metadata = response.get("search_metadata")
    redirect = parse_redirect(
        metadata.get("google_flights_url") if isinstance(metadata, dict) else None,
        kind="search",
    )
    items = _result_items(response, "best_flights", "other_flights")
    offers = []
    for token, outbound in _select_outbounds(items, query, max_outbound):
        returns = search_serpapi({**params, "departure_token": token}, budget=budget)
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
                raise SerpAPIResponseError("Invalid round-trip offer price or data.", kind="invalid_flight_offer") from None
            if query.max_price_eur is None or offer.price_eur <= query.max_price_eur:
                offers.append(offer)
    return offers

from decimal import Decimal, InvalidOperation

from execution import RunBudget

from models import HotelOffer, HotelSearchQuery
from providers.links import parse_redirect
from providers.serpapi_client import SerpAPIResponseError, search as search_serpapi


def parse_hotel_offer(raw_hotel: dict, query: HotelSearchQuery) -> HotelOffer:
    if not isinstance(raw_hotel, dict):
        raise SerpAPIResponseError("Formato dell'hotel non valido.")
    total_rate = raw_hotel.get("total_rate")
    if not isinstance(total_rate, dict) or total_rate.get("extracted_lowest") is None:
        raise SerpAPIResponseError("L'hotel non contiene un prezzo totale utilizzabile.")
    nightly_rate = raw_hotel.get("rate_per_night")
    if nightly_rate is not None and not isinstance(nightly_rate, dict):
        raise SerpAPIResponseError("Formato del prezzo per notte non valido.")
    nightly_price = nightly_rate.get("extracted_lowest") if nightly_rate else None
    try:
        return HotelOffer(
            destination=query.destination,
            name=raw_hotel["name"].strip(),
            nightly_price_eur=Decimal(str(nightly_price)) if nightly_price is not None else None,
            rating=raw_hotel.get("overall_rating"),
            nights=(query.checkout_date - query.checkin_date).days,
            total_price_eur=Decimal(str(total_rate["extracted_lowest"])),
            redirect=parse_redirect(raw_hotel.get("link"), kind="property_website"),
        )
    except (KeyError, AttributeError, TypeError, ValueError, InvalidOperation):
        raise SerpAPIResponseError("Prezzi o dati dell'offerta hotel non validi.") from None


def search_hotels(query: HotelSearchQuery, *, budget: RunBudget | None = None) -> list[HotelOffer]:
    query.validate_stay_dates()
    response = search_serpapi({
        "engine": "google_hotels",
        "q": f"Hotels in {query.location_name or query.destination}",
        "check_in_date": query.checkin_date.isoformat(),
        "check_out_date": query.checkout_date.isoformat(),
        "adults": query.adults,
        "currency": "EUR", "hl": "it", "gl": "it",
    }, budget=budget)
    offers = [parse_hotel_offer(item, query) for item in _hotel_results(response)]
    if query.max_total_price_eur is not None:
        offers = [offer for offer in offers if offer.total_price_eur <= query.max_total_price_eur]
    return offers


def _hotel_results(response: dict) -> list[dict]:
    if "properties" in response:
        items = response["properties"]
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise SerpAPIResponseError("Formato dei risultati hotel inatteso.")
        return items
    information = response.get("search_information")
    if isinstance(information, dict) and information.get("hotels_results_state") == "Showing results for property details":
        return [response]
    metadata = response.get("search_metadata", {})
    if isinstance(metadata, dict) and metadata.get("status") == "Success" and response.get("error"):
        return []
    raise SerpAPIResponseError("La risposta non contiene i risultati hotel attesi.")

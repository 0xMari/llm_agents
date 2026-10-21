from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from execution import RunBudget

from models import HotelCoordinates, HotelDetails, HotelImage, HotelOffer, HotelSearchQuery
from providers.links import parse_public_url, parse_redirect
from providers.serpapi_client import SerpAPIResponseError, search as search_serpapi


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _optional_text_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [text for item in value if (text := _optional_text(item)) is not None]
    return items if items or not value else None


def _optional_int(value: object, *, minimum: int, maximum: int | None = None) -> int | None:
    if type(value) is not int or value < minimum:
        return None
    return value if maximum is None or value <= maximum else None


def _parse_images(value: object) -> list[HotelImage] | None:
    if not isinstance(value, list):
        return None
    images = []
    for item in value:
        if not isinstance(item, dict):
            continue
        thumbnail = parse_public_url(item.get("thumbnail"))
        original = parse_public_url(item.get("original_image"))
        if thumbnail is not None or original is not None:
            images.append(HotelImage(thumbnail=thumbnail, original_image=original))
    return images if images or not value else None


def _parse_details(raw_hotel: dict) -> HotelDetails:
    # Optional enrichment must not discard an otherwise valid priced offer.
    coordinates = None
    raw_coordinates = raw_hotel.get("gps_coordinates")
    if isinstance(raw_coordinates, dict):
        try:
            coordinates = HotelCoordinates.model_validate(raw_coordinates)
        except ValidationError:
            pass
    return HotelDetails(
        property_type=_optional_text(raw_hotel.get("type")),
        description=_optional_text(raw_hotel.get("description")),
        address=_optional_text(raw_hotel.get("address")),
        hotel_class=_optional_text(raw_hotel.get("hotel_class")),
        stars=_optional_int(raw_hotel.get("extracted_hotel_class"), minimum=1, maximum=5),
        reviews=_optional_int(raw_hotel.get("reviews"), minimum=0),
        gps_coordinates=coordinates,
        check_in_time=_optional_text(raw_hotel.get("check_in_time")),
        check_out_time=_optional_text(raw_hotel.get("check_out_time")),
        amenities=_optional_text_list(raw_hotel.get("amenities")),
        excluded_amenities=_optional_text_list(raw_hotel.get("excluded_amenities")),
        essential_info=_optional_text_list(raw_hotel.get("essential_info")),
        images=_parse_images(raw_hotel.get("images")),
    )


def parse_hotel_offer(
    raw_hotel: dict, query: HotelSearchQuery, *, search_url: object = None
) -> HotelOffer:
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
            search_redirect=parse_redirect(search_url, kind="hotel_search"),
            property_token=_optional_text(raw_hotel.get("property_token")),
            details=_parse_details(raw_hotel),
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
    metadata = response.get("search_metadata")
    search_url = metadata.get("google_hotels_url") if isinstance(metadata, dict) else None
    offers = [parse_hotel_offer(item, query, search_url=search_url) for item in _hotel_results(response)]
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

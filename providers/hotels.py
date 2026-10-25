from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from execution import RunBudget

from models import (
    HotelCoordinates, HotelDetails, HotelImage, HotelIssueCode, HotelIssueField,
    HotelOffer, HotelOfferIssue, HotelSearchQuery, HotelSearchResult,
)
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


class HotelOfferParseError(ValueError):
    def __init__(self, *, field: HotelIssueField, code: HotelIssueCode):
        super().__init__(f"Hotel offer cannot be used: {field} ({code}).")
        self.field = field
        self.code = code


def _amount(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Invalid amount type.")
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid amount value.")
    return amount


def parse_hotel_offer(
    raw_hotel: object, query: HotelSearchQuery, *, index: int = 0, search_url: object = None,
) -> tuple[HotelOffer, list[HotelOfferIssue]]:
    query.validate_stay_dates()
    if not isinstance(raw_hotel, dict):
        raise HotelOfferParseError(field="offer", code="invalid_format")
    raw_name = raw_hotel.get("name")
    name = _optional_text(raw_name)
    if name is None:
        raise HotelOfferParseError(
            field="name", code="missing_required" if raw_name is None or isinstance(raw_name, str) else "invalid_value",
        )
    total_rate = raw_hotel.get("total_rate")
    if total_rate is None:
        raise HotelOfferParseError(field="total_rate", code="missing_required")
    if not isinstance(total_rate, dict):
        raise HotelOfferParseError(field="total_rate", code="invalid_format")
    if total_rate.get("extracted_lowest") is None:
        raise HotelOfferParseError(field="total_rate", code="missing_required")
    try:
        total_price = _amount(total_rate["extracted_lowest"])
    except (ValueError, InvalidOperation):
        raise HotelOfferParseError(field="total_rate", code="invalid_value") from None

    issues = []

    def ignored(field: HotelIssueField, code: HotelIssueCode = "invalid_value") -> None:
        issues.append(HotelOfferIssue(index=index, field=field, code=code, action="field_ignored"))

    nightly_price = None
    nightly_rate = raw_hotel.get("rate_per_night")
    if nightly_rate is not None and not isinstance(nightly_rate, dict):
        ignored("rate_per_night", "invalid_format")
    elif nightly_rate and nightly_rate.get("extracted_lowest") is not None:
        try:
            nightly_price = _amount(nightly_rate["extracted_lowest"])
        except (ValueError, InvalidOperation):
            ignored("rate_per_night")

    rating = None
    if raw_hotel.get("overall_rating") is not None:
        try:
            amount = _amount(raw_hotel["overall_rating"])
            if amount > 5:
                raise ValueError("Rating out of range.")
            rating = float(amount)
        except (ValueError, InvalidOperation):
            ignored("overall_rating")

    return HotelOffer(
        destination=query.destination, name=name,
        nightly_price_eur=nightly_price, rating=rating,
        nights=(query.checkout_date - query.checkin_date).days,
        total_price_eur=total_price,
        redirect=parse_redirect(raw_hotel.get("link"), kind="property_website"),
        search_redirect=parse_redirect(search_url, kind="hotel_search"),
        property_token=_optional_text(raw_hotel.get("property_token")),
        details=_parse_details(raw_hotel),
    ), issues


def search_hotels(query: HotelSearchQuery, *, budget: RunBudget | None = None) -> HotelSearchResult:
    query.validate_stay_dates()
    response = search_serpapi({
        "engine": "google_hotels",
        "q": f"Hotels in {query.location_name or query.destination}",
        "check_in_date": query.checkin_date.isoformat(),
        "check_out_date": query.checkout_date.isoformat(),
        "adults": query.adults,
        "currency": "EUR", "hl": "en", "gl": "it",
    }, budget=budget)
    metadata = response.get("search_metadata")
    search_url = metadata.get("google_hotels_url") if isinstance(metadata, dict) else None
    items = _hotel_results(response)
    offers, issues = [], []
    for index, item in enumerate(items):
        try:
            offer, field_issues = parse_hotel_offer(item, query, index=index, search_url=search_url)
        except HotelOfferParseError as error:
            issues.append(HotelOfferIssue(
                index=index, field=error.field, code=error.code, action="offer_discarded",
            ))
            continue
        offers.append(offer)
        issues.extend(field_issues)
    parsed_count = len(offers)
    if query.max_total_price_eur is not None:
        offers = [offer for offer in offers if offer.total_price_eur <= query.max_total_price_eur]
    return HotelSearchResult(offers=offers, issues=issues, received_count=len(items), parsed_count=parsed_count)


def _hotel_results(response: dict) -> list[object]:
    if "properties" in response:
        items = response["properties"]
        if not isinstance(items, list):
            raise SerpAPIResponseError("Unexpected hotel result format.", kind="invalid_hotel_results")
        return items
    information = response.get("search_information")
    if isinstance(information, dict) and information.get("hotels_results_state") == "Showing results for property details":
        return [response]
    metadata = response.get("search_metadata", {})
    if isinstance(metadata, dict) and metadata.get("status") == "Success" and response.get("error"):
        return []
    raise SerpAPIResponseError("The response does not contain the expected hotel results.", kind="missing_hotel_results")

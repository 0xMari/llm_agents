from typing import Self, Literal, Annotated
from pydantic import BaseModel, Field, model_validator, ConfigDict, HttpUrl
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


CITY_ID_PATTERN = r"^/[mg]/[A-Za-z0-9_-]+$"
AIRPORT_CODE_PATTERN = r"^[A-Z]{3}$"
FLIGHT_LOCATION_PATTERN = r"^(?:[A-Z]{3}|/[mg]/[A-Za-z0-9_-]+)$"


class PlanningScenario(StrEnum):
    FIXED_DESTINATION_EXACT_DATES = "FIXED_DESTINATION_EXACT_DATES"
    FIXED_DESTINATION_FLEXIBLE_DATES = "FIXED_DESTINATION_FLEXIBLE_DATES"
    OPEN_DESTINATION_EXACT_DATES = "OPEN_DESTINATION_EXACT_DATES"
    OPEN_DESTINATION_FLEXIBLE_DATES = "OPEN_DESTINATION_FLEXIBLE_DATES"

class TimePreference(BaseModel):
    mode: Literal["exact_dates", "month", "season", "flexible_window"]
    start_date: date | None = None
    end_date: date | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    season: Literal["spring", "summer", "autumn", "winter"] | None = None
    min_days: int | None = Field(default=None, ge=1)
    max_days: int | None = Field(default=None, ge=1)

    
    @model_validator(mode="after")
    def validate_time_preference(self) -> Self:
        if self.min_days is not None and self.max_days is not None:
            if self.min_days > self.max_days:
                raise ValueError("min_days non puo superare max_days.")

        if self.mode == "exact_dates":
            if self.start_date is None or self.end_date is None:
                raise ValueError("Con mode='exact_dates' servono start_date e end_date.")
            if self.end_date <= self.start_date:
                raise ValueError("end_date deve essere successiva a start_date.")

        if self.mode == "month" and self.month is None:
            raise ValueError("Con mode='month' devi specificare month.")

        if self.mode == "season" and self.season is None:
            raise ValueError("Con mode='season' devi specificare season.")

        return self

class DestinationPreference(BaseModel):
    mode: Literal["fixed", "open"]
    destination: str | None = None
    city_id: str | None = Field(default=None, pattern=CITY_ID_PATTERN)
    airport_code: str | None = Field(default=None, pattern=AIRPORT_CODE_PATTERN)

    @model_validator(mode="after")
    def validate_destination(self) -> Self:
        if self.mode == "fixed" and not self.destination:
            raise ValueError("Con mode='fixed' devi specificare destination.")
        if self.mode == "open" and (self.city_id is not None or self.airport_code is not None):
            raise ValueError("Gli identificativi di arrivo richiedono una destinazione fissa.")
        return self


class SuggestedDestination(BaseModel):
    name: str
    country: str | None = None
    reason: str
    match_score: int = Field(ge=0, le=100)
    themes_matched: list[str] = Field(default_factory=list)
    caution: str | None = None


class DestinationCandidate(SuggestedDestination):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    country: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    themes_matched: list[str]
    caution: str | None

class DestinationAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    destinations: list[DestinationCandidate] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_destinations(self) -> Self:
        keys = [(d.name.casefold(), d.country.casefold()) for d in self.destinations]
        if len(keys) != len(set(keys)):
            raise ValueError("Destinazioni duplicate.")
        return self

class TravelVibe(BaseModel):
    free_text: str | None = None
    themes: list[str] = Field(default_factory=list)
    pace: Literal["slow", "balanced", "intense"] | None = None
    environments: list[str] = Field(default_factory=list)

class TripRequest(BaseModel):
    origin: str
    adults: int = Field(default=1, ge=1)
    origin_city_id: str | None = Field(default=None, pattern=CITY_ID_PATTERN)
    origin_airport_code: str | None = Field(default=None, pattern=AIRPORT_CODE_PATTERN)
    budget_eur: int | None = Field(default=None, gt=0)
    budget_flexibility_pct: int = Field(default=20, ge=0, le=30)
    destination_pref: DestinationPreference
    time_pref: TimePreference
    vibe: TravelVibe = Field(default_factory=TravelVibe)


class SuggestedTimePeriod(BaseModel):
    start: date
    end: date
    reason: str
    score: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.end <= self.start:
            raise ValueError("La data finale deve essere successiva a quella iniziale.")
        return self


class TravelWindowAdvice(BaseModel):
    selected_periods: list[SuggestedTimePeriod] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    alternatives: list[SuggestedTimePeriod] = Field(default_factory=list)
    rationale: str | None = None


class ProposalTravelContext(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    rationale: str | None = None
    alternatives: list[SuggestedTimePeriod] = Field(default_factory=list)
    # These are LLM suggestions, not additional flight + hotel quotes.
    alternatives_status: Literal["suggestions_only"] = "suggestions_only"


class SuggestedStayLength(BaseModel):
    nights: int = Field(ge=1)
    reason: str
    score: int = Field(ge=0, le=100)


class FlightSearchQuery(BaseModel):
    origin: str
    destination: str
    origin_id: str = Field(pattern=FLIGHT_LOCATION_PATTERN)
    destination_id: str = Field(pattern=FLIGHT_LOCATION_PATTERN)
    departure_date: date
    return_date: date
    adults: int = Field(default=1, ge=1)
    max_price_eur: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if self.return_date <= self.departure_date:
            raise ValueError("Il ritorno deve essere successivo alla partenza.")
        return self

class FlightSegment(BaseModel):
    departure_airport: str = Field(pattern=AIRPORT_CODE_PATTERN)
    arrival_airport: str = Field(pattern=AIRPORT_CODE_PATTERN)
    departure_local: datetime
    arrival_local: datetime
    airline: str = Field(min_length=1)
    flight_number: str = Field(min_length=1)

class FlightLeg(BaseModel):
    segments: list[FlightSegment] = Field(min_length=1)
    duration_minutes: int = Field(gt=0)

    @property
    def stops(self) -> int:
        return len(self.segments) - 1

class OfferRedirect(BaseModel):
    url: HttpUrl
    kind: Literal["search", "property_website", "hotel_search"]
    label: str


class FlightOffer(BaseModel):
    origin: str
    destination: str
    outbound: FlightLeg
    inbound: FlightLeg
    airline: str
    departure_date: date
    return_date: date
    price_eur: Decimal = Field(ge=0)
    stops: int = Field(ge=0)
    redirect: OfferRedirect | None = None

    @model_validator(mode="after")
    def validate_leg_summaries(self) -> Self:
        if self.departure_date != self.outbound.segments[0].departure_local.date():
            raise ValueError("La data di partenza non corrisponde all'andata.")
        if self.return_date != self.inbound.segments[0].departure_local.date():
            raise ValueError("La data di ritorno non corrisponde alla tratta di ritorno.")
        if self.stops != self.outbound.stops + self.inbound.stops:
            raise ValueError("Il numero di scali non corrisponde alle tratte.")
        return self


class HotelSearchQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    destination: str = Field(min_length=1)
    location_name: str | None = Field(default=None, min_length=1)
    checkin_date: date
    checkout_date: date
    adults: int = Field(default=1, ge=1)
    max_total_price_eur: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_stay_dates(self) -> Self:
        if self.checkout_date <= self.checkin_date:
            raise ValueError("Il checkout deve essere successivo al checkin.")
        return self


class Hotel(BaseModel):
    destination: str
    name: str = Field(min_length=1)
    nightly_price_eur: Decimal | None = Field(default=None, ge=0)
    rating: float | None = Field(default=None, ge=0, le=5)


class HotelCoordinates(BaseModel):
    latitude: float = Field(ge=-90, le=90, strict=True, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, strict=True, allow_inf_nan=False)


class HotelImage(BaseModel):
    thumbnail: HttpUrl | None = None
    original_image: HttpUrl | None = None


class HotelDetails(BaseModel):
    property_type: str | None = None
    description: str | None = None
    address: str | None = None
    hotel_class: str | None = None
    stars: int | None = Field(default=None, ge=1, le=5, strict=True)
    reviews: int | None = Field(default=None, ge=0, strict=True)
    gps_coordinates: HotelCoordinates | None = None
    check_in_time: str | None = None
    check_out_time: str | None = None
    amenities: list[str] | None = None
    excluded_amenities: list[str] | None = None
    essential_info: list[str] | None = None
    images: list[HotelImage] | None = None


class HotelOffer(Hotel):
    nights: int = Field(ge=1)
    total_price_eur: Decimal = Field(ge=0)
    redirect: OfferRedirect | None = None
    search_redirect: OfferRedirect | None = None
    property_token: str | None = Field(default=None, min_length=1)
    details: HotelDetails = Field(default_factory=HotelDetails)
    # Provider descriptions and coordinates do not verify the user's requirements.
    vibe_match_status: Literal["not_verified"] = "not_verified"


class EventType(StrEnum):
    SEARCH_SCOPE_REDUCED = "SEARCH_SCOPE_REDUCED"
    DESTINATIONS_RESOLVED = "DESTINATIONS_RESOLVED"
    DESTINATION_RESOLUTION_FAILED = "DESTINATION_RESOLUTION_FAILED"
    REQUEST_ROUTED = "REQUEST_ROUTED"
    PERIOD_ANALYZED = "PERIOD_ANALYZED"
    TRAVEL_WINDOW_EVALUATED = "TRAVEL_WINDOW_EVALUATED"
    COMBINATION_ACCEPTED = "COMBINATION_ACCEPTED"
    COMBINATION_EXCLUDED = "COMBINATION_EXCLUDED"
    PROPOSALS_SELECTED = "PROPOSALS_SELECTED"
    PROPOSAL_SUMMARY_GENERATED = "PROPOSAL_SUMMARY_GENERATED"
    PROPOSAL_SUMMARY_FAILED = "PROPOSAL_SUMMARY_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    DESTINATION_SEARCH_FAILED = "DESTINATION_SEARCH_FAILED"
    TRAVEL_WINDOW_FAILED = "TRAVEL_WINDOW_FAILED"


class ReasonCode(StrEnum):
    SEARCH_SCOPE_REDUCED = "SEARCH_SCOPE_REDUCED"
    SEARCH_LIMIT_REACHED = "SEARCH_LIMIT_REACHED"
    FIXED_DESTINATION_USED = "FIXED_DESTINATION_USED"
    OPEN_DESTINATION_SUGGESTED = "OPEN_DESTINATION_SUGGESTED"
    SCENARIO_DETECTED = "SCENARIO_DETECTED"
    PERIOD_CANDIDATE = "PERIOD_CANDIDATE"
    TRAVEL_WINDOW_SELECTED = "TRAVEL_WINDOW_SELECTED"
    NO_FLIGHTS_FOUND = "NO_FLIGHTS_FOUND"
    FLIGHT_PROVIDER_ERROR = "FLIGHT_PROVIDER_ERROR"
    FLIGHT_OUTPUT_INVALID = "FLIGHT_OUTPUT_INVALID"
    FLIGHT_LOCATION_UNRESOLVED = "FLIGHT_LOCATION_UNRESOLVED"
    NO_HOTELS_FOUND = "NO_HOTELS_FOUND"
    HOTEL_PROVIDER_ERROR = "HOTEL_PROVIDER_ERROR"
    HOTEL_OUTPUT_INVALID = "HOTEL_OUTPUT_INVALID"
    NO_BUDGET_PROVIDED = "NO_BUDGET_PROVIDED"
    WITHIN_BUDGET = "WITHIN_BUDGET"
    OVER_BUDGET = "OVER_BUDGET"
    OVER_PREFERRED_BUDGET = "OVER_PREFERRED_BUDGET"
    OVER_MAX_BUDGET = "OVER_MAX_BUDGET"
    DIRECT_FLIGHT = "DIRECT_FLIGHT"
    GOOD_HOTEL_RATING = "GOOD_HOTEL_RATING"
    SELECTED_BY_TRADEOFF = "SELECTED_BY_TRADEOFF"
    SELECTED_BY_DESTINATION_MATCH = "SELECTED_BY_DESTINATION_MATCH"
    PROPOSAL_SUMMARY_GENERATED = "PROPOSAL_SUMMARY_GENERATED"
    NO_AFFORDABLE_PROPOSAL = "NO_AFFORDABLE_PROPOSAL"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_OUTPUT_INVALID = "LLM_OUTPUT_INVALID"
    LLM_CONSTRAINT_VIOLATION = "LLM_CONSTRAINT_VIOLATION"


# A strict Decimal branch prevents JSON lists being parsed as Decimal tuples.
DecisionDetailValue = list[str] | Annotated[Decimal, Field(strict=True)] | str | int | float | bool | None


class DecisionEvent(BaseModel):
    event_type: EventType
    reason_code: ReasonCode
    details: dict[str, DecisionDetailValue] = Field(default_factory=dict)
    comment: str | None = None


class ProposalComment(BaseModel):
    reason_code: ReasonCode
    details: dict[str, DecisionDetailValue] = Field(default_factory=dict)
    comment: str


class DecisionTrace(BaseModel):
    events: list[DecisionEvent] = Field(default_factory=list)


class ProposalComparison(BaseModel):
    proposal_id: str
    reference_id: str
    extra_price_eur: Decimal = Field(ge=0)
    stops_delta: int
    duration_delta_minutes: int
    early_departures_delta: int


class ProposalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    proposal_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=80)
    advantage: str = Field(min_length=1, max_length=250)
    tradeoff: str | None = Field(min_length=1, max_length=250)


class ProposalSummaries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ProposalSummary] = Field(min_length=1, max_length=3)


class TripProposal(BaseModel):
    destination: SuggestedDestination
    destination_city_id: str | None = Field(default=None, pattern=CITY_ID_PATTERN)
    period: SuggestedTimePeriod
    flight: FlightOffer
    hotel: HotelOffer
    total_price_eur: Decimal = Field(ge=0)
    recommendations: list[str]
    comments: list[ProposalComment] = Field(default_factory=list)
    selection_profiles: list[
        Literal["cheapest", "comfortable", "balanced"]
    ] = Field(default_factory=list)
    comparison: ProposalComparison | None = None
    summary: ProposalSummary | None = None
    travel_context: ProposalTravelContext | None = None


class DestinationSearchIssue(BaseModel):
    destination: str
    country: str | None = None
    stage: Literal["location_resolution", "flight_search", "hotel_search"]
    code: ReasonCode
    message: str


class TripSearchResult(BaseModel):
    proposals: list[TripProposal] = Field(min_length=1, max_length=3)
    trace: DecisionTrace
    summary_status: Literal["not_requested", "generated", "unavailable"] = "not_requested"
    search_status: Literal["complete", "partial"] = "complete"
    search_issues: list[DestinationSearchIssue] = Field(default_factory=list)


class PlanningError(BaseModel):
    code: str
    message: str
    trace: DecisionTrace = Field(default_factory=DecisionTrace)

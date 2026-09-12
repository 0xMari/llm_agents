from typing import Self, Literal
from pydantic import BaseModel, Field, model_validator
from datetime import date
from enum import StrEnum


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

    @model_validator(mode="after")
    def validate_destination(self) -> Self:
        if self.mode == "fixed" and not self.destination:
            raise ValueError("Con mode='fixed' devi specificare destination.")
        return self


class SuggestedDestination(BaseModel):
    name: str
    country: str | None = None
    reason: str
    match_score: int = Field(ge=0, le=100)
    themes_matched: list[str] = Field(default_factory=list)
    caution: str | None = None


class TravelVibe(BaseModel):
    free_text: str | None = None
    themes: list[str] = Field(default_factory=list)
    pace: Literal["slow", "balanced", "intense"] | None = None
    environments: list[str] = Field(default_factory=list)

class TripRequest(BaseModel):
    origin: str
    budget_eur: int = Field(gt=0)
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


class SuggestedStayLength(BaseModel):
    nights: int = Field(ge=1)
    reason: str
    score: int = Field(ge=0, le=100)


class FlightSearchQuery(BaseModel):
    origin: str
    destination: str
    departure_date: date
    return_date: date
    adults: int = Field(default=1, ge=1)
    max_price_eur: int | None = Field(default=None, gt=0)


class FlightOffer(BaseModel):
    origin: str
    destination: str
    airline: str
    departure_date: date
    return_date: date
    price_eur: int
    stops: int = Field(ge=0)


class HotelSearchQuery(BaseModel):
    destination: str
    checkin_date: date
    checkout_date: date
    adults: int = Field(default=1, ge=1)
    max_total_price_eur: int | None = Field(default=None, gt=0)


class Hotel(BaseModel):
    destination: str
    name: str
    nightly_price_eur: int
    rating: float = Field(ge=0, le=5)

class HotelOffer(Hotel):
    nights: int = Field(ge=1)

    @property
    def total_price_eur(self) -> int:
        return self.nightly_price_eur * self.nights


class EventType(StrEnum):
    DESTINATIONS_RESOLVED = "DESTINATIONS_RESOLVED"
    REQUEST_ROUTED = "REQUEST_ROUTED"
    PERIOD_ANALYZED = "PERIOD_ANALYZED"
    TRAVEL_WINDOW_EVALUATED = "TRAVEL_WINDOW_EVALUATED"
    COMBINATION_ACCEPTED = "COMBINATION_ACCEPTED"
    COMBINATION_EXCLUDED = "COMBINATION_EXCLUDED"
    PROPOSALS_SELECTED = "PROPOSALS_SELECTED"
    SEARCH_FAILED = "SEARCH_FAILED"
    TRAVEL_WINDOW_FAILED = "TRAVEL_WINDOW_FAILED"


class ReasonCode(StrEnum):
    FIXED_DESTINATION_USED = "FIXED_DESTINATION_USED"
    OPEN_DESTINATION_SUGGESTED = "OPEN_DESTINATION_SUGGESTED"
    SCENARIO_DETECTED = "SCENARIO_DETECTED"
    PERIOD_CANDIDATE = "PERIOD_CANDIDATE"
    TRAVEL_WINDOW_SELECTED = "TRAVEL_WINDOW_SELECTED"
    NO_FLIGHTS_FOUND = "NO_FLIGHTS_FOUND"
    WITHIN_BUDGET = "WITHIN_BUDGET"
    OVER_BUDGET = "OVER_BUDGET"
    OVER_PREFERRED_BUDGET = "OVER_PREFERRED_BUDGET"
    OVER_MAX_BUDGET = "OVER_MAX_BUDGET"
    DIRECT_FLIGHT = "DIRECT_FLIGHT"
    GOOD_HOTEL_RATING = "GOOD_HOTEL_RATING"
    RANKED_BY_SCORE_RATING_PRICE = "RANKED_BY_SCORE_RATING_PRICE"
    NO_AFFORDABLE_PROPOSAL = "NO_AFFORDABLE_PROPOSAL"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_OUTPUT_INVALID = "LLM_OUTPUT_INVALID"
    LLM_CONSTRAINT_VIOLATION = "LLM_CONSTRAINT_VIOLATION"


class DecisionEvent(BaseModel):
    event_type: EventType
    reason_code: ReasonCode
    details: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    comment: str | None = None


class ProposalComment(BaseModel):
    reason_code: ReasonCode
    details: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    comment: str


class DecisionTrace(BaseModel):
    events: list[DecisionEvent] = Field(default_factory=list)


class TripProposal(BaseModel):
    destination: SuggestedDestination
    period: SuggestedTimePeriod
    flight: FlightOffer
    hotel: HotelOffer
    total_price_eur: int
    recommendations: list[str]
    comments: list[ProposalComment] = Field(default_factory=list)

#numero di risultati
class TripSearchResult(BaseModel):
    proposals: list[TripProposal] = Field(min_length=1, max_length=3)
    trace: DecisionTrace


class PlanningError(BaseModel):
    code: str
    message: str
    trace: DecisionTrace = Field(default_factory=DecisionTrace)

import os
from typing import Annotated, Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from execution import RunBudget, SearchLimitExceeded, SearchLimits
from models import (
    CITY_ID_PATTERN, DestinationPreference, PlanningError, ReasonCode,
    TimePreference, TravelVibe, TripRequest, TripSearchResult,
)
from planning import build_proposal
from providers.flights import FlightLocationError, find_city_options
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError


LocationText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
PreferenceText = Annotated[str, StringConstraints(min_length=1, max_length=80)]


class ApiTimePreference(TimePreference):
    model_config = ConfigDict(extra="forbid")


class ApiDestinationPreference(DestinationPreference):
    model_config = ConfigDict(extra="forbid")
    destination: LocationText | None = None


class ApiTravelVibe(TravelVibe):
    model_config = ConfigDict(extra="forbid")
    free_text: str | None = Field(default=None, max_length=2000)
    themes: list[PreferenceText] = Field(default_factory=list, max_length=20)
    environments: list[PreferenceText] = Field(default_factory=list, max_length=20)


class ProposalRequest(TripRequest):
    model_config = ConfigDict(extra="forbid")
    origin: LocationText
    destination_pref: ApiDestinationPreference
    time_pref: ApiTimePreference
    vibe: ApiTravelVibe = Field(default_factory=ApiTravelVibe)


class TransportOption(BaseModel):
    id: str = Field(pattern=r"^[A-Z0-9]{3}$")
    name: str | None = None


class LocationOption(BaseModel):
    name: str
    city_id: str = Field(pattern=CITY_ID_PATTERN)
    description: str | None = None
    transport_locations: list[TransportOption] = Field(default_factory=list)


class LocationSearchResponse(BaseModel):
    locations: list[LocationOption]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ValidationIssue(BaseModel):
    location: list[str | int]
    type: str
    message: str


class ApiError(BaseModel):
    code: str
    message: str
    issues: list[ValidationIssue] = Field(default_factory=list)
    location_options: list[str] = Field(default_factory=list)


_PLANNING_ERRORS = {
    ReasonCode.INVALID_TRAVEL_DATES: (422, "The departure date cannot be in the past."),
    ReasonCode.FLIGHT_LOCATION_UNRESOLVED: (422, "Select an unambiguous city from the location search."),
    ReasonCode.NO_FLIGHTS_FOUND: (200, "No flights were found for this search."),
    ReasonCode.NO_HOTELS_FOUND: (200, "No matching accommodation was found for this search."),
    ReasonCode.NO_AFFORDABLE_PROPOSAL: (200, "No combinations fit the allowed budget."),
    ReasonCode.FLIGHT_OUTPUT_INVALID: (502, "The flight search returned unusable data."),
    ReasonCode.HOTEL_OUTPUT_INVALID: (502, "The hotel search returned unusable data."),
    ReasonCode.LLM_OUTPUT_INVALID: (502, "The travel assistant returned unusable data."),
    ReasonCode.LLM_CONSTRAINT_VIOLATION: (502, "The travel assistant could not respect the requested constraints."),
    ReasonCode.FLIGHT_PROVIDER_ERROR: (503, "The flight search service is temporarily unavailable."),
    ReasonCode.HOTEL_PROVIDER_ERROR: (503, "The hotel search service is temporarily unavailable."),
    ReasonCode.LLM_PROVIDER_ERROR: (503, "The travel assistant is temporarily unavailable."),
    ReasonCode.SEARCH_LIMIT_REACHED: (503, "The search reached its configured call limit. Try a narrower search."),
}
_ERROR_RESPONSES = {status: {"model": ApiError} for status in (422, 500, 502, 503, 504)}


def _error(status: int, code: str, message: str, **kwargs) -> JSONResponse:
    error = ApiError(code=code, message=message, **kwargs)
    return JSONResponse(status_code=status, content=error.model_dump(mode="json"))


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _limits_for(request: TripRequest) -> SearchLimits:
    # Leave room for the single shared LLM retry; no limits come from HTTP input.
    if request.destination_pref.mode == "open":
        return SearchLimits(max_destinations=3, max_serpapi_calls=13, max_llm_calls=6)
    return SearchLimits(max_serpapi_calls=5, max_llm_calls=3)


def _planning_error_response(error: PlanningError) -> JSONResponse:
    status, message = _PLANNING_ERRORS.get(error.code, (500, "An unexpected planning error occurred."))
    if status == 500:
        return _error(500, "INTERNAL_ERROR", message)
    options = []
    for event in error.trace.events:
        if event.reason_code == ReasonCode.FLIGHT_LOCATION_UNRESOLVED:
            values = event.details.get("location_options", [])
            if isinstance(values, list):
                options.extend(value for value in values if isinstance(value, str))
        if status == 503 and (
            event.details.get("provider_error") == "TIMEOUT"
            or event.details.get("llm_error_kind") == "timeout"
        ):
            status = 504
            message = "A travel provider did not respond in time."
    return _error(status, error.code, message, location_options=list(dict.fromkeys(options)))


def create_app(*, allowed_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="AI Travel Planner", version="0.1.0")
    if allowed_origins is None:
        allowed_origins = [value.strip() for value in os.environ.get("FRONTEND_ORIGINS", "").split(",") if value.strip()]
    if any(origin == "*" or origin == "null" for origin in allowed_origins):
        raise ValueError("Configure explicit frontend origins, not wildcards or null.")
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False,
            allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Do not echo submitted values, request bodies, or exception contexts.
        issues = [ValidationIssue(
            location=list(item["loc"]), type=item["type"], message=item["msg"],
        ) for item in error.errors()]
        return _error(422, "INVALID_REQUEST", "The request contains invalid or missing fields.", issues=issues)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        return _error(500, "INTERNAL_ERROR", "An unexpected server error occurred.")

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse()

    @app.get("/locations", response_model=LocationSearchResponse, responses=_ERROR_RESPONSES)
    def locations(q: Annotated[str, Query(min_length=2, max_length=120)]):
        query = q.strip()
        if len(query) < 2:
            return _error(422, "INVALID_REQUEST", "Enter at least two non-whitespace characters.")
        budget = RunBudget(SearchLimits(max_serpapi_calls=1, max_llm_calls=0))
        try:
            cities = find_city_options(query, budget=budget)
        except FlightLocationError:
            return _error(422, "INVALID_REQUEST", "Enter a city to search for.")
        except SerpAPIResponseError:
            return _error(502, "LOCATION_OUTPUT_INVALID", "The location search returned unusable data.")
        except SerpAPIError as error:
            if error.code == "TIMEOUT":
                return _error(504, "LOCATION_PROVIDER_TIMEOUT", "The location search timed out.")
            return _error(503, "LOCATION_PROVIDER_ERROR", "The location search service is temporarily unavailable.")
        except SearchLimitExceeded:
            return _error(503, "SEARCH_LIMIT_REACHED", "The location search reached its configured call limit.")
        return LocationSearchResponse(locations=[LocationOption(
            name=city["name"], city_id=city["id"], description=_optional_text(city.get("description")),
            transport_locations=[TransportOption(id=item["id"], name=_optional_text(item.get("name")))
                                 for item in city.get("airports", [])],
        ) for city in cities])

    @app.post("/proposals", response_model=TripSearchResult | ApiError, responses=_ERROR_RESPONSES)
    def proposals(body: ProposalRequest):
        request = TripRequest.model_validate(body.model_dump())
        result = build_proposal(request, limits=_limits_for(request))
        if isinstance(result, PlanningError):
            return _planning_error_response(result)
        return result

    return app


app = create_app()

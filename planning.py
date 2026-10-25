from decimal import Decimal
from typing import Literal

from comparison import add_comparisons
from execution import RunBudget, SearchLimitExceeded, SearchLimits

from agents.activities import suggest_activities
from agents.destination_agent import suggest_destinations
from agents.proposal_summary import InvalidProposalSummary, summarize_proposals

from providers.flights import FlightLocationError, resolve_city, search_flights, search_id_for
from providers.hotels import search_hotels
from providers.location_verification import CityCountryVerifier
from providers.serpapi_client import SerpAPIError, SerpAPIResponseError

from scoring import select_final_proposals
from agents.travel_window import evaluate_travel_window

from models import (
    DecisionEvent,
    DecisionTrace,
    DestinationSearchIssue,
    EventType,
    FlightSearchQuery,
    HotelSearchQuery,
    HotelSearchResult,
    PlanningError,
    ProposalComment,
    ProposalTravelContext,
    ReasonCode,
    SuggestedDestination,
    TripProposal,
    TripRequest,
    TripSearchResult,
)
from routing import route_request
from pydantic import ValidationError
from llm.openrouter import OpenRouterError
from agents.travel_window import TravelWindowConstraintError


def resolve_destinations(request: TripRequest, *, budget: RunBudget) -> list[SuggestedDestination]:
    if request.destination_pref.mode == "fixed":
        return [
            SuggestedDestination(
                name=request.destination_pref.destination,
                reason="Destination specified by the user.",
                match_score=100,
            )
        ]

    return suggest_destinations(request, budget=budget)


def calculate_budget_ceiling(request: TripRequest) -> Decimal | None:
    if request.budget_eur is None:
        return None
    return Decimal(request.budget_eur) * (100 + request.budget_flexibility_pct) / 100


def flight_search_failure(
    events: list[DecisionEvent], destination: str, error: SerpAPIError | FlightLocationError,
    *, stage: Literal["location_resolution", "flight_search"] = "flight_search",
    location_role: str | None = None,
) -> PlanningError:
    details: dict[str, str | int | list[str] | None] = {"destination": destination, "stage": stage}
    if location_role is not None:
        details["location_role"] = location_role
    if isinstance(error, SerpAPIError):
        details.update(provider_error=error.code, http_status=error.status_code)
    if isinstance(error, FlightLocationError):
        reason = ReasonCode.FLIGHT_LOCATION_UNRESOLVED
        message = str(error)
        details["location_options"] = error.options
        if error.country_status is not None:
            details["country_status"] = error.country_status.value
    elif isinstance(error, SerpAPIResponseError):
        reason = ReasonCode.FLIGHT_OUTPUT_INVALID
        details["response_error_kind"] = error.kind
        message = (
            "The location search returned unusable data."
            if stage == "location_resolution" else
            "The flight search returned unusable data."
        )
    else:
        reason = ReasonCode.FLIGHT_PROVIDER_ERROR
        message = "The flight search service is unavailable. Check configuration and quota."
    events.append(DecisionEvent(
        event_type=EventType.SEARCH_FAILED,
        reason_code=reason,
        details=details,
        comment=message,
    ))
    return PlanningError(code=reason.value, message=message, trace=DecisionTrace(events=events))


def hotel_search_failure(events: list[DecisionEvent], query: HotelSearchQuery, error: SerpAPIError) -> PlanningError:
    if isinstance(error, SerpAPIResponseError):
        reason = ReasonCode.HOTEL_OUTPUT_INVALID
        message = "The hotel search returned unusable data."
    else:
        reason = ReasonCode.HOTEL_PROVIDER_ERROR
        message = "The hotel search service is unavailable. Check configuration and quota."
    events.append(DecisionEvent(
        event_type=EventType.SEARCH_FAILED,
        reason_code=reason,
        details={
            "destination": query.destination,
            "location_name": query.location_name,
            "checkin_date": query.checkin_date.isoformat(),
            "checkout_date": query.checkout_date.isoformat(),
            "adults": query.adults,
            "provider_error": error.code,
            "http_status": error.status_code,
            "stage": "hotel_search",
            "response_error_kind": error.kind if isinstance(error, SerpAPIResponseError) else None,
        },
        comment=message,
    ))
    return PlanningError(code=reason.value, message=message, trace=DecisionTrace(events=events))


def record_destination_issue(
    events: list[DecisionEvent], issues: list[DestinationSearchIssue],
    destination: SuggestedDestination, failure: PlanningError,
    stage: Literal["location_resolution", "flight_search", "hotel_search"],
) -> None:
    events[-1].event_type = EventType.DESTINATION_SEARCH_FAILED
    events[-1].details["scope"] = "destination"
    issues.append(DestinationSearchIssue(
        destination=destination.name, country=destination.country,
        stage=stage, code=failure.code, message=failure.message,
    ))


def record_hotel_issues(
    events: list[DecisionEvent], issues: list[DestinationSearchIssue],
    destination: SuggestedDestination, query: HotelSearchQuery, result: HotelSearchResult,
) -> None:
    for issue in result.issues:
        discarded = issue.action == "offer_discarded"
        events.append(DecisionEvent(
            event_type=EventType.HOTEL_OFFER_ISSUE,
            reason_code=ReasonCode.HOTEL_OFFER_DISCARDED if discarded else ReasonCode.HOTEL_FIELD_IGNORED,
            details={
                "destination": destination.name, "location_name": query.location_name,
                "checkin_date": query.checkin_date.isoformat(), "checkout_date": query.checkout_date.isoformat(),
                "scope": "offer", "offer_index": issue.index, "field": issue.field,
                "issue_code": issue.code, "action": issue.action,
            },
            comment="An unusable hotel offer was discarded." if discarded else "An invalid optional hotel field was omitted.",
        ))
    discarded_count = sum(issue.action == "offer_discarded" for issue in result.issues)
    if discarded_count and result.parsed_count:
        message = "Some hotel offers were discarded; valid offers remain usable."
        issues.append(DestinationSearchIssue(
            destination=destination.name, country=destination.country, stage="hotel_search",
            code=ReasonCode.HOTEL_OFFERS_DISCARDED, message=message, scope="offer",
        ))
        events.append(DecisionEvent(
            event_type=EventType.HOTEL_SEARCH_PARTIAL, reason_code=ReasonCode.HOTEL_OFFERS_DISCARDED,
            details={
                "destination": destination.name, "scope": "offer", "stage": "hotel_search",
                "received_count": result.received_count, "parsed_count": result.parsed_count,
                "returned_count": len(result.offers), "discarded_count": discarded_count,
            },
            comment=message,
        ))


def build_proposal(
    request: TripRequest, *, limits: SearchLimits,
    country_verifier: CityCountryVerifier | None = None,
) -> TripSearchResult | PlanningError:
    budget = RunBudget(limits)
    events = []
    try:
        result = _build_proposal(request, budget=budget, events=events, country_verifier=country_verifier)
    except SearchLimitExceeded as error:
        message = "Search stopped: the configured call limit was reached."
        events.append(DecisionEvent(
            event_type=EventType.SEARCH_FAILED,
            reason_code=ReasonCode.SEARCH_LIMIT_REACHED,
            details={"service": error.service, "used": error.used, "limit": error.limit},
            comment=message,
        ))
        result = PlanningError(
            code=ReasonCode.SEARCH_LIMIT_REACHED.value,
            message=message,
            trace=DecisionTrace(events=events),
        )

    # Keep the terminal decision last, with a summary of attempted HTTP calls.
    result.trace.events[-1].details.update(
        serpapi_calls=budget.used["serpapi"],
        llm_calls=budget.used["llm"],
        llm_retries=budget.llm_retries,
        **budget.llm_retry_details,
        **limits.model_dump(),
    )
    return result


def record_scope_reduction(
    events: list[DecisionEvent], *, scope: str, available: int,
    unique: int, selected: int, destination: str | None = None,
) -> None:
    if selected == available:
        return
    events.append(DecisionEvent(
        event_type=EventType.SEARCH_SCOPE_REDUCED,
        reason_code=ReasonCode.SEARCH_SCOPE_REDUCED,
        details={
            "scope": scope, "available_count": available, "unique_count": unique,
            "selected_count": selected, "destination": destination,
        },
        comment="Search scope reduced after ranking, deduplication, and applying limits.",
    ))


def _build_proposal(
    request: TripRequest, *, budget: RunBudget, events: list[DecisionEvent],
    country_verifier: CityCountryVerifier | None = None,
) -> TripSearchResult | PlanningError:
    limits = budget.limits
    scenario = route_request(request)
    proposals = []
    found_flights = False
    found_combinations = False
    budget_ceiling = calculate_budget_ceiling(request)
    city_cache = {}
    hotel_cache = {}
    failed_hotel_stays = set()
    search_issues = []
    verification_options = {"verifier": country_verifier} if country_verifier is not None else {}

    events.append(
        DecisionEvent(
            event_type = EventType.REQUEST_ROUTED,
            reason_code = ReasonCode.SCENARIO_DETECTED,
            details = {"scenario": scenario.value},
            comment = "Planning scenario identified.",
        )
    )

    try:
        destinations = resolve_destinations(request, budget=budget)
    except (OpenRouterError, ValidationError) as exc:
        details: dict[str, str | int | None] = {"agent": "destination"}
        if isinstance(exc, OpenRouterError):
            reason = ReasonCode.LLM_PROVIDER_ERROR
            details.update(exc.trace_details())
        else:
            reason = ReasonCode.LLM_OUTPUT_INVALID

        events.append(
            DecisionEvent(
                event_type=EventType.DESTINATION_RESOLUTION_FAILED,
                reason_code=reason,
                details=details,
                comment="Destination selection could not be completed.",
            )
        )

        return PlanningError(
            code=reason.value,
            message="Unable to suggest destinations.",
            trace=DecisionTrace(events=events),
        )

    candidates = destinations
    destinations = sorted(candidates, key=lambda item: item.match_score, reverse=True)[:limits.max_destinations]
    record_scope_reduction(
        events, scope="destinations", available=len(candidates),
        unique=len(candidates), selected=len(destinations),
    )

    events.append(
        DecisionEvent(
            event_type=EventType.DESTINATIONS_RESOLVED,
            reason_code=(
                ReasonCode.FIXED_DESTINATION_USED
                if request.destination_pref.mode == "fixed"
                else ReasonCode.OPEN_DESTINATION_SUGGESTED
            ),
            details={
                "destinations": [destination.name for destination in destinations],
            },
            comment="Candidate destinations resolved.",
        )
    )

    
    for destination in destinations:
        try:
            window_advice = evaluate_travel_window(request, destination, budget=budget)
        except (OpenRouterError, ValidationError, TravelWindowConstraintError) as exc:
            details = {"destination": destination.name}
            if isinstance(exc, OpenRouterError):
                reason = ReasonCode.LLM_PROVIDER_ERROR
                details.update(exc.trace_details())
            elif isinstance(exc, ValidationError):
                reason = ReasonCode.LLM_OUTPUT_INVALID
            else:
                reason = ReasonCode.LLM_CONSTRAINT_VIOLATION

            events.append(DecisionEvent(
                event_type=EventType.TRAVEL_WINDOW_FAILED,
                reason_code=reason,
                details=details,
                comment="Travel window evaluation could not be completed.",
            ))
            return PlanningError(
                code=reason.value,
                message="Unable to evaluate the travel window.",
                trace=DecisionTrace(events=events),
            )

        unique_periods = {}
        for period in sorted(window_advice.selected_periods, key=lambda item: item.score, reverse=True):
            unique_periods.setdefault((period.start, period.end), period)
        periods = list(unique_periods.values())[:limits.max_periods_per_destination]
        record_scope_reduction(
            events, scope="periods", available=len(window_advice.selected_periods),
            unique=len(unique_periods), selected=len(periods), destination=destination.name,
        )

        events.append(
            DecisionEvent(
                event_type=EventType.TRAVEL_WINDOW_EVALUATED,
                reason_code=ReasonCode.TRAVEL_WINDOW_SELECTED,
                details={
                    "destination": destination.name,
                    "selected_periods": [
                        f"{period.start.isoformat()}->{period.end.isoformat()}"
                        for period in periods
                    ],
                    "warnings": window_advice.warnings,
                },
                comment=window_advice.rationale,
            )
        )

        destination_text = ", ".join(filter(None, [destination.name, destination.country]))
        locations = [
            (request.origin, request.origin_city_id, request.origin_airport_code),
            (destination_text, request.destination_pref.city_id, request.destination_pref.airport_code),
        ]
        search_ids = []
        try:
            for location_index, (text, city_id, airport_code) in enumerate(locations):
                key = (text, city_id)
                if key not in city_cache:
                    city_cache[key] = resolve_city(
                        text,
                        country=destination.country if location_index == 1 else None,
                        city_id=city_id,
                        budget=budget,
                        **verification_options,
                    )
                search_ids.append(search_id_for(city_cache[key], airport_code))
        except (SerpAPIError, FlightLocationError) as error:
            failure = flight_search_failure(
                events, destination.name, error, stage="location_resolution",
                location_role="origin" if location_index == 0 else "destination",
            )
            if (request.destination_pref.mode == "open" and location_index == 1
                    and isinstance(error, (SerpAPIResponseError, FlightLocationError))):
                record_destination_issue(events, search_issues, destination, failure, "location_resolution")
                continue
            return failure

        destination_city = city_cache[(destination_text, request.destination_pref.city_id)]

        for period in periods:
            events.append(
                DecisionEvent(
                    event_type = EventType.PERIOD_ANALYZED,
                    reason_code = ReasonCode.PERIOD_CANDIDATE,
                    details = {
                        "start": period.start.isoformat(),
                        "end": period.end.isoformat(),
                        "score": period.score,
                    },
                    comment = "Candidate travel period analyzed.",
                )
            )

            flight_query = FlightSearchQuery(
                origin = request.origin,
                destination = destination.name,
                origin_id = search_ids[0],
                destination_id = search_ids[1],
                departure_date = period.start,
                return_date = period.end,
                adults = request.adults,
            )

            try:
                flights = search_flights(flight_query, max_outbound=limits.max_outbound, budget=budget)
            except SerpAPIError as error:
                failure = flight_search_failure(events, destination.name, error)
                if request.destination_pref.mode == "open" and isinstance(error, SerpAPIResponseError):
                    record_destination_issue(events, search_issues, destination, failure, "flight_search")
                    continue
                return failure

            if not flights:
                events.append(
                    DecisionEvent(
                        event_type = EventType.SEARCH_FAILED,
                        reason_code = ReasonCode.NO_FLIGHTS_FOUND,
                        details = {
                            "destination": destination.name,
                            "start": period.start.isoformat(),
                            "end": period.end.isoformat(),
                        },
                        comment = "No flights found for this travel period.",
                    )
                )
                continue

            found_flights = True

            for flight in flights:
                checkin = flight.outbound.segments[-1].arrival_local.date()
                checkout = flight.inbound.segments[0].departure_local.date()
                if checkout <= checkin:
                    events.append(DecisionEvent(
                        event_type=EventType.SEARCH_FAILED,
                        reason_code=ReasonCode.NO_HOTELS_FOUND,
                        details={"destination": destination.name},
                        comment="The selected flights do not allow an overnight stay.",
                    ))
                    continue
                hotel_query = HotelSearchQuery(
                    destination = destination.name,
                    location_name = destination_city["name"],
                    checkin_date = checkin,
                    checkout_date = checkout,
                    adults = flight_query.adults,
                )
                stay_key = (hotel_query.location_name, checkin, checkout, hotel_query.adults)
                if stay_key in failed_hotel_stays:
                    continue
                if stay_key not in hotel_cache:
                    try:
                        hotel_result = search_hotels(hotel_query, budget=budget)
                        record_hotel_issues(events, search_issues, destination, hotel_query, hotel_result)
                        if hotel_result.received_count and not hotel_result.parsed_count:
                            raise SerpAPIResponseError(
                                "None of the received hotel offers could be used.", kind="all_hotel_offers_invalid",
                            )
                        hotel_cache[stay_key] = hotel_result
                    except SerpAPIError as error:
                        failure = hotel_search_failure(events, hotel_query, error)
                        if request.destination_pref.mode == "open" and isinstance(error, SerpAPIResponseError):
                            failed_hotel_stays.add(stay_key)
                            record_destination_issue(events, search_issues, destination, failure, "hotel_search")
                            continue
                        return failure
                hotels = hotel_cache[stay_key].offers

                if not hotels:
                    events.append(
                        DecisionEvent(
                            event_type=EventType.SEARCH_FAILED,
                            reason_code=ReasonCode.NO_HOTELS_FOUND,
                            details={
                                "destination": destination.name,
                                "start": checkin.isoformat(),
                                "end": checkout.isoformat(),
                            },
                            comment="No hotels found for this stay.",
                        )
                    )
                    continue

                for hotel in hotels:
                    found_combinations = True
                    total_price = flight.price_eur + hotel.total_price_eur

                    if budget_ceiling is not None and total_price > budget_ceiling:
                        events.append(
                            DecisionEvent(
                                event_type = EventType.COMBINATION_EXCLUDED,
                                reason_code = ReasonCode.OVER_MAX_BUDGET,
                                details = {
                                    "airline": flight.airline,
                                    "hotel": hotel.name,
                                    "total_price_eur": total_price,
                                    "budget_eur": request.budget_eur,
                                    "budget_ceiling_eur": budget_ceiling,
                                },
                                comment = "Combination excluded: it exceeds the maximum allowed budget.",
                            )
                        )
                        continue
                    if request.budget_eur is None:
                        budget_reason = ReasonCode.NO_BUDGET_PROVIDED
                        budget_comment = "No budget limit provided."
                    elif total_price <= request.budget_eur:
                        budget_reason = ReasonCode.WITHIN_BUDGET
                        budget_comment = "Combination within budget."
                    else:
                        budget_reason = ReasonCode.OVER_PREFERRED_BUDGET
                        budget_comment = "Combination above the preferred budget, but within the allowed flexibility."

                    comments = [
                        ProposalComment(
                            reason_code = budget_reason,
                            details = {
                                "total_price_eur": total_price,
                                "budget_eur": request.budget_eur,
                                "budget_ceiling_eur": budget_ceiling,
                            },
                            comment = budget_comment,
                        )
                    ]

                    if flight.stops == 0:
                        comments.append(
                            ProposalComment(
                                reason_code = ReasonCode.DIRECT_FLIGHT,
                                comment = "Direct flight.",
                            )
                        )

                    if hotel.rating is not None and hotel.rating >= 4:
                        comments.append(
                            ProposalComment(
                                reason_code = ReasonCode.GOOD_HOTEL_RATING,
                                details = {"hotel_rating": hotel.rating},
                                comment = "Well-rated hotel.",
                            )
                        )
                    
                    events.append(
                        DecisionEvent(
                            event_type = EventType.COMBINATION_ACCEPTED,
                            reason_code = budget_reason,
                            details = {
                                "airline": flight.airline,
                                "hotel": hotel.name,
                                "total_price_eur": total_price,
                                "budget_eur": request.budget_eur,
                                "budget_ceiling_eur": budget_ceiling,
                            },
                            comment = "Combination accepted.",
                        )
                    )

                    proposals.append(
                        TripProposal(
                            destination=destination,
                            destination_city_id=destination_city["id"],
                            period = period,
                            flight = flight,
                            hotel = hotel,
                            total_price_eur = total_price,
                            recommendations = suggest_activities(request),
                            comments = comments,
                            travel_context = ProposalTravelContext(
                                warnings=window_advice.warnings,
                                rationale=window_advice.rationale,
                                alternatives=window_advice.alternatives,
                            ),
                        )
                    )

    if not proposals:
        failed_destinations = [issue for issue in search_issues if issue.scope == "destination"]
        if failed_destinations:
            reason = failed_destinations[0].code
            message = "No usable proposals: some destination searches could not be completed."
            events.append(DecisionEvent(
                event_type=EventType.SEARCH_FAILED, reason_code=reason,
                details={"affected_destinations": list(dict.fromkeys(issue.destination for issue in failed_destinations))},
                comment=message,
            ))
            return PlanningError(code=reason.value, message=message, trace=DecisionTrace(events=events))
        if not found_flights:
            reason = ReasonCode.NO_FLIGHTS_FOUND
            message = "No flights found for the searched destinations and travel periods."
        elif not found_combinations:
            reason = ReasonCode.NO_HOTELS_FOUND
            message = "No hotels found for stays matching the available flights."
        else:
            reason = ReasonCode.NO_AFFORDABLE_PROPOSAL
            message = "The available combinations exceed the maximum allowed budget."

        return PlanningError(
            code = reason.value,
            message = message,
            trace = DecisionTrace(events = events),
        )


    early_hour = 9
    selected = select_final_proposals(proposals, request, early_hour=early_hour)
    selected = add_comparisons(selected, early_hour=early_hour)

    summary_status = "unavailable"
    try:
        selected = summarize_proposals(request, selected, budget=budget, early_hour=early_hour)
    except (OpenRouterError, ValidationError, InvalidProposalSummary, SearchLimitExceeded) as exc:
        details = {"agent": "proposal_summary"}
        if isinstance(exc, OpenRouterError):
            reason = ReasonCode.LLM_PROVIDER_ERROR
            details.update(exc.trace_details())
        elif isinstance(exc, SearchLimitExceeded):
            reason = ReasonCode.SEARCH_LIMIT_REACHED
            details.update(service=exc.service, used=exc.used, limit=exc.limit)
        elif isinstance(exc, InvalidProposalSummary):
            reason = ReasonCode.LLM_CONSTRAINT_VIOLATION
            details["summary_quality_issue"] = exc.kind
            if exc.field is not None:
                details["summary_field"] = exc.field
        else:
            reason = ReasonCode.LLM_OUTPUT_INVALID
        events.append(DecisionEvent(
            event_type=EventType.PROPOSAL_SUMMARY_FAILED,
            reason_code=reason,
            details=details,
            comment="Summary unavailable: offers and comparisons remain usable.",
        ))
    else:
        summary_status = "generated"
        events.append(DecisionEvent(
            event_type=EventType.PROPOSAL_SUMMARY_GENERATED,
            reason_code=ReasonCode.PROPOSAL_SUMMARY_GENERATED,
            details={"summarized_count": len(selected)},
            comment="Summary added without changing offers or comparisons.",
        ))

    events.append(
        DecisionEvent(
            event_type = EventType.PROPOSALS_SELECTED,
            reason_code = (
                ReasonCode.SELECTED_BY_DESTINATION_MATCH
                if request.destination_pref.mode == "open"
                else ReasonCode.SELECTED_BY_TRADEOFF
            ),
            details = {
                "selected_count": len(selected),
                "summary_status": summary_status,
                "destination_mode": request.destination_pref.mode,
                "search_status": "partial" if search_issues else "complete",
                "affected_destinations": list(dict.fromkeys(issue.destination for issue in search_issues)),
                "destination_city_ids": list(dict.fromkeys(p.destination_city_id for p in selected)),
            },
            comment = (
                "Selected one combination per city, up to three, ranked by estimated relevance to the request and within the allowed budget."
                if request.destination_pref.mode == "open" else
                "Selected up to three combinations based on price and flight comfort, within the allowed budget."
            ),
        )
    )

    return TripSearchResult(
        proposals=selected,
        summary_status=summary_status,
        search_status="partial" if search_issues else "complete",
        search_issues=search_issues,
        trace = DecisionTrace(events=events),
    )

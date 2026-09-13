from agents.activities import suggest_activities
from agents.destination_agent import suggest_destinations

from providers.flights import search_flights
from providers.hotels import search_hotels

from scoring import select_top_proposals
from agents.travel_window import evaluate_travel_window

from models import (
    DecisionEvent,
    DecisionTrace,
    EventType,
    FlightSearchQuery,
    HotelSearchQuery,
    PlanningError,
    ProposalComment,
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


def resolve_destinations(request: TripRequest) -> list[SuggestedDestination]:
    if request.destination_pref.mode == "fixed":
        return [
            SuggestedDestination(
                name=request.destination_pref.destination,
                reason="Destinazione indicata dall'utente.",
                match_score=100,
            )
        ]

    return suggest_destinations(request)


def calculate_budget_ceiling(request: TripRequest) -> int | None:
    if request.budget_eur is None:
        return None
    return request.budget_eur * (100 + request.budget_flexibility_pct) // 100


def build_proposal(request: TripRequest) -> TripSearchResult | PlanningError:
    scenario = route_request(request)
    proposals = []
    events = []
    found_flights = False
    found_combinations = False
    budget_ceiling = calculate_budget_ceiling(request)

    events.append(
        DecisionEvent(
            event_type = EventType.REQUEST_ROUTED,
            reason_code = ReasonCode.SCENARIO_DETECTED,
            details = {"scenario": scenario.value},
            comment = "Scenario di pianificazione identificato.",
        )
    )

    try:
        destinations = resolve_destinations(request)
    except (OpenRouterError, ValidationError) as exc:
        if isinstance(exc, OpenRouterError):
            reason = ReasonCode.LLM_PROVIDER_ERROR
        else:
            reason = ReasonCode.LLM_OUTPUT_INVALID

        events.append(
            DecisionEvent(
                event_type=EventType.DESTINATION_RESOLUTION_FAILED,
                reason_code=reason,
                details={"agent": "destination"},
                comment="Selezione delle destinazioni non completata.",
            )
        )

        return PlanningError(
            code=reason.value,
            message="Non e stato possibile proporre destinazioni.",
            trace=DecisionTrace(events=events),
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
            comment="Destinazioni candidate risolte.",
        )
    )

    
    for destination in destinations:
        try:
            window_advice = evaluate_travel_window(request, destination)
        except (OpenRouterError, ValidationError, TravelWindowConstraintError) as exc:
            if isinstance(exc, OpenRouterError):
                reason = ReasonCode.LLM_PROVIDER_ERROR
            elif isinstance(exc, ValidationError):
                reason = ReasonCode.LLM_OUTPUT_INVALID
            else:
                reason = ReasonCode.LLM_CONSTRAINT_VIOLATION

            events.append(DecisionEvent(
                event_type=EventType.TRAVEL_WINDOW_FAILED,
                reason_code=reason,
                details={"destination": destination.name},
                comment="Valutazione del periodo non completata.",
            ))
            return PlanningError(
                code=reason.value,
                message="Non e stato possibile valutare il periodo di viaggio.",
                trace=DecisionTrace(events=events),
            )

        events.append(
            DecisionEvent(
                event_type=EventType.TRAVEL_WINDOW_EVALUATED,
                reason_code=ReasonCode.TRAVEL_WINDOW_SELECTED,
                details={
                    "destination": destination.name,
                    "selected_periods": [
                        f"{period.start.isoformat()}->{period.end.isoformat()}"
                        for period in window_advice.selected_periods
                    ],
                    "warnings": window_advice.warnings,
                },
                comment=window_advice.rationale,
            )
        )

        periods = window_advice.selected_periods

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
                    comment = "Periodo candidato analizzato",
                )
            )

            flight_query = FlightSearchQuery(
                origin = request.origin,
                destination = destination.name,
                departure_date = period.start,
                return_date = period.end,
            )

            flights = search_flights(flight_query)

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
                        comment = "Nessun volo trovato per il periodo.",
                    )
                )
                continue

            found_flights = True

            for flight in flights:

                hotel_query = HotelSearchQuery(
                    destination = destination.name,
                    checkin_date = period.start,
                    checkout_date = period.end,
                )
                hotels = search_hotels(hotel_query)

                if not hotels:
                    events.append(
                        DecisionEvent(
                            event_type=EventType.SEARCH_FAILED,
                            reason_code=ReasonCode.NO_HOTELS_FOUND,
                            details={
                                "destination": destination.name,
                                "start": period.start.isoformat(),
                                "end": period.end.isoformat(),
                            },
                            comment="Nessun hotel trovato per il soggiorno.",
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
                                comment = "Combinazione esclusa: supera anche il budget massimo tollerato.",
                            )
                        )
                        continue
                    if request.budget_eur is None:
                        budget_reason = ReasonCode.NO_BUDGET_PROVIDED
                        budget_comment = "Nessun limite di budget indicato."
                    elif total_price <= request.budget_eur:
                        budget_reason = ReasonCode.WITHIN_BUDGET
                        budget_comment = "Combinazione entro budget."
                    else:
                        budget_reason = ReasonCode.OVER_PREFERRED_BUDGET
                        budget_comment = "Combinazione leggermente sopra il budget preferito, ma entro la flessibilità impostata."

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
                                comment = "Volo diretto.",
                            )
                        )

                    if hotel.rating >= 4:
                        comments.append(
                            ProposalComment(
                                reason_code = ReasonCode.GOOD_HOTEL_RATING,
                                details = {"hotel_rating": hotel.rating},
                                comment = "Hotel con buon rating.",
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
                            comment = "Combinazione accettata.",
                        )
                    )

                    proposals.append(
                        TripProposal(
                            destination=destination,
                            period = period,
                            flight = flight,
                            hotel = hotel,
                            total_price_eur = total_price,
                            recommendations = suggest_activities(request),
                            comments = comments,
                        )
                    )

    if not proposals:
        if not found_flights:
            reason = ReasonCode.NO_FLIGHTS_FOUND
            message = "Nessun volo trovato per le destinazioni e i periodi cercati."
        elif not found_combinations:
            reason = ReasonCode.NO_HOTELS_FOUND
            message = "Nessun hotel trovato per i soggiorni associati ai voli disponibili."
        else:
            reason = ReasonCode.NO_AFFORDABLE_PROPOSAL
            message = "Le combinazioni trovate superano il budget massimo consentito."

        return PlanningError(
            code = reason.value,
            message = message,
            trace = DecisionTrace(events = events),
        )


    selected = select_top_proposals(proposals, request)

    events.append(
        DecisionEvent(
            event_type = EventType.PROPOSALS_SELECTED,
            reason_code = ReasonCode.RANKED_BY_SCORE_RATING_PRICE,
            details = {"selected_count": len(selected)},
            comment = "Proposte ordinate per rispetto del budget, pertinenza, rating e prezzo.",
        )
    )

    return TripSearchResult(
        proposals=selected,
        trace = DecisionTrace(events=events),
    )

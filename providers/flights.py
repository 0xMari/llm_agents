from data import FLIGHTS
from models import FlightOffer, FlightSearchQuery

def search_flights(query: FlightSearchQuery) -> list[FlightOffer]:
    flights = [
        flight
        for flight in FLIGHTS
        if flight.origin == query.origin
        and flight.destination == query.destination
        and flight.departure_date == query.departure_date
        and flight.return_date == query.return_date
    ]

    if query.max_price_eur is not None:
        flights = [
            flight
            for flight in flights
            if flight.price_eur <= query.max_price_eur
        ]
    
    return flights

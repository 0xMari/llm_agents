# #dati finti

from datetime import date

from models import FlightOffer, HotelOffer


FLIGHTS = [
    FlightOffer(
        origin = "Rome",
        destination = "Lisbon",
        airline = "Demo Air",
        departure_date = date(2026, 9, 15),
        return_date = date(2026, 9, 20),
        price_eur = 180,
        stops = 0,
    ),
    FlightOffer(
        origin = "Rome",
        destination = "Lisbon",
        airline = "Budget Wings",
        departure_date = date(2026, 9, 15),
        return_date = date(2026, 9, 20),
        price_eur = 135,
        stops = 1,
    ),
    FlightOffer(
        origin = "Rome",
        destination = "Lisbon",
        airline = "Demo Air",
        departure_date = date(2026, 10, 6),
        return_date = date(2026, 10, 11),
        price_eur = 155,
        stops = 0,
    ),
    FlightOffer(
        origin = "Rome",
        destination = "Lisbon",
        airline = "Budget Wings",
        departure_date = date(2026, 10, 6),
        return_date = date(2026, 10, 11),
        price_eur = 115,
        stops = 1,
    ),
]

HOTELS = [
    HotelOffer(
        destination = "Lisbon",
        name = "Central Stay",
        nightly_price_eur = 95,
        rating = 4.4,
        nights = 5,
    ),
    HotelOffer(
        destination = "Lisbon",
        name = "Simple Rooms",
        nightly_price_eur = 62,
        rating = 3.9,
        nights = 5,
    ),
]

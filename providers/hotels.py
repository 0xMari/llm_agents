from data import HOTELS

from models import HotelOffer, HotelSearchQuery

def search_hotels(query: HotelSearchQuery) -> list[HotelOffer]:
    hotels = [
        hotel
        for hotel in HOTELS
        if hotel.destination == query.destination
    ]

    if query.max_total_price_eur is not None:
        hotels = [
            hotel
            for hotel in hotels
            if hotel.total_price_eur <= query.max_total_price_eur
        ]
    
    return hotels

from data import HOTELS

from models import HotelOffer, HotelSearchQuery

def search_hotels(query: HotelSearchQuery) -> list[HotelOffer]:
    nights = (query.checkout_date - query.checkin_date).days
    if nights < 1:
        raise ValueError("Il checkout deve essere successivo al checkin.")

    offers = [
        HotelOffer(**hotel.model_dump(), nights=nights)
        for hotel in HOTELS
        if hotel.destination == query.destination
    ]

    if query.max_total_price_eur is not None:
        offers = [
            offer for offer in offers
            if offer.total_price_eur <= query.max_total_price_eur
        ]
    
    return offers

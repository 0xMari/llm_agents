import argparse
from datetime import date, timedelta

from pydantic import ValidationError

from models import FlightSearchQuery, HotelSearchQuery
from providers.flights import FlightLocationError, resolve_city, search_flights, search_id_for
from providers.hotels import search_hotels
from providers.serpapi_client import SerpAPIError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prova manuale SerpAPI: consuma ricerche reali.")
    parser.add_argument("--origin", default="Rome, Italy")
    parser.add_argument("--destination", default="Lisbon, Portugal")
    parser.add_argument("--origin-city-id")
    parser.add_argument("--destination-city-id")
    parser.add_argument("--origin-airport")
    parser.add_argument("--destination-airport")
    parser.add_argument("--departure-date", type=date.fromisoformat, default=date.today() + timedelta(days=45))
    parser.add_argument("--return-date", type=date.fromisoformat)
    parser.add_argument("--adults", type=int, default=1)
    parser.add_argument("--max-outbound", type=int, default=1)
    parser.add_argument("--hotels", action="store_true", help="Aggiunge una ricerca hotel per la prima offerta.")
    args = parser.parse_args(argv)
    end = args.return_date or args.departure_date + timedelta(days=5)
    if end <= args.departure_date or args.adults < 1 or args.max_outbound < 1:
        parser.error("Servono date ordinate, almeno un adulto e max-outbound positivo.")

    origin_city = resolve_city(args.origin, city_id=args.origin_city_id)
    destination_city = resolve_city(args.destination, city_id=args.destination_city_id)
    query = FlightSearchQuery(
        origin=origin_city["name"], destination=destination_city["name"],
        origin_id=search_id_for(origin_city, args.origin_airport),
        destination_id=search_id_for(destination_city, args.destination_airport),
        departure_date=args.departure_date, return_date=end, adults=args.adults,
    )
    offers = search_flights(query, max_outbound=args.max_outbound)
    print("Offerte A/R trovate:", len(offers))
    for offer in offers:
        print(offer.model_dump_json(indent=2))

    if not args.hotels or not offers:
        return
    checkin = offers[0].outbound.segments[-1].arrival_local.date()
    checkout = offers[0].inbound.segments[0].departure_local.date()
    if checkout <= checkin:
        print("La prima offerta non lascia una notte per il soggiorno.")
        return
    hotels = search_hotels(HotelSearchQuery(
        destination=query.destination,
        location_name=destination_city["name"],
        checkin_date=checkin, checkout_date=checkout, adults=query.adults,
    ))
    print("Offerte hotel trovate:", len(hotels))
    for hotel in hotels[:3]:
        print(hotel.model_dump_json(indent=2))
        print("Totale A/R + soggiorno (EUR):", offers[0].price_eur + hotel.total_price_eur)
    print("I link non garantiscono la tariffa: verifica prezzo e disponibilita sul sito di destinazione.")


if __name__ == "__main__":
    try:
        main()
    except FlightLocationError as error:
        print("Localita da chiarire:", error)
        for option in error.options:
            print(option)
        raise SystemExit(1)
    except (SerpAPIError, ValidationError) as error:
        raise SystemExit(str(error)) from None

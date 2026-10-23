import argparse
from datetime import date, timedelta

from pydantic import ValidationError

from models import FlightSearchQuery, HotelSearchQuery
from providers.flights import FlightLocationError, resolve_city, search_flights, search_id_for
from providers.hotels import search_hotels
from providers.serpapi_client import SerpAPIError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual SerpAPI test: uses real searches.")
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
    parser.add_argument("--hotels", action="store_true", help="Search hotels for the first flight offer.")
    args = parser.parse_args(argv)
    end = args.return_date or args.departure_date + timedelta(days=5)
    if end <= args.departure_date or args.adults < 1 or args.max_outbound < 1:
        parser.error("Dates must be ordered, adults must be at least one, and max-outbound must be positive.")

    origin_city = resolve_city(args.origin, city_id=args.origin_city_id)
    destination_city = resolve_city(args.destination, city_id=args.destination_city_id)
    query = FlightSearchQuery(
        origin=origin_city["name"], destination=destination_city["name"],
        origin_id=search_id_for(origin_city, args.origin_airport),
        destination_id=search_id_for(destination_city, args.destination_airport),
        departure_date=args.departure_date, return_date=end, adults=args.adults,
    )
    offers = search_flights(query, max_outbound=args.max_outbound)
    print("Round-trip offers found:", len(offers))
    for offer in offers:
        print(offer.model_dump_json(indent=2))

    if not args.hotels or not offers:
        return
    checkin = offers[0].outbound.segments[-1].arrival_local.date()
    checkout = offers[0].inbound.segments[0].departure_local.date()
    if checkout <= checkin:
        print("The first offer does not allow an overnight stay.")
        return
    hotels = search_hotels(HotelSearchQuery(
        destination=query.destination,
        location_name=destination_city["name"],
        checkin_date=checkin, checkout_date=checkout, adults=query.adults,
    ))
    print("Hotel offers found:", len(hotels))
    for hotel in hotels[:3]:
        print(hotel.model_dump_json(indent=2))
        print("Round-trip + stay total (EUR):", offers[0].price_eur + hotel.total_price_eur)
    print("Links do not guarantee the fare: check prices and availability on the destination website.")


if __name__ == "__main__":
    try:
        main()
    except FlightLocationError as error:
        print("Location needs clarification:", error)
        for option in error.options:
            print(option)
        raise SystemExit(1)
    except (SerpAPIError, ValidationError) as error:
        raise SystemExit(str(error)) from None

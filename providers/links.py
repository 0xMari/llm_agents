import re
from typing import Literal
from urllib.parse import parse_qsl

from pydantic import HttpUrl, ValidationError

from models import OfferRedirect


def parse_redirect(raw_url: object, *, kind: Literal["search", "property_website"]) -> OfferRedirect | None:
    if not isinstance(raw_url, str):
        return None
    try:
        url = HttpUrl(raw_url)
    except ValidationError:
        return None
    host = (url.host or "").lower()
    path = url.path or ""
    if (
        url.username or url.password
        or host == "serpapi.com" or host.endswith(".serpapi.com")
        or path.startswith("/_/")
        or any(key.casefold() == "api_key" for key, _ in parse_qsl(url.query or ""))
    ):
        return None
    if kind == "search" and (
        not re.fullmatch(r"(?:www\.)?google\.(?:[a-z]{2,3}|(?:com|co)\.[a-z]{2})", host)
        or not (path == "/travel/flights" or path.startswith("/travel/flights/") or path == "/flights")
    ):
        return None
    return OfferRedirect(
        url=url,
        kind=kind,
        label="Apri la ricerca voli" if kind == "search" else "Visita il sito della struttura",
    )

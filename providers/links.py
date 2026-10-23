import re
from typing import Literal
from urllib.parse import parse_qsl

from pydantic import HttpUrl, ValidationError

from models import OfferRedirect


def parse_public_url(raw_url: object) -> HttpUrl | None:
    if not isinstance(raw_url, str):
        return None
    try:
        url = HttpUrl(raw_url)
    except ValidationError:
        return None
    host = (url.host or "").lower().rstrip(".")
    path = url.path or ""
    if (
        url.username or url.password
        or host == "serpapi.com" or host.endswith(".serpapi.com")
        or path.startswith("/_/")
        or any(key.casefold() == "api_key" for key, _ in parse_qsl(url.query or ""))
    ):
        return None
    return url


def parse_redirect(
    raw_url: object, *, kind: Literal["search", "property_website", "hotel_search"]
) -> OfferRedirect | None:
    url = parse_public_url(raw_url)
    if url is None:
        return None
    if kind in ("search", "hotel_search"):
        host = (url.host or "").lower().rstrip(".")
        path = url.path or ""
        paths = ("/travel/flights",) if kind == "search" else ("/travel/hotels", "/travel/search")
        allowed_path = any(path == prefix or path.startswith(prefix + "/") for prefix in paths)
        if kind == "search" and path == "/flights":
            allowed_path = True
        if (
            not re.fullmatch(r"(?:www\.)?google\.(?:[a-z]{2,3}|(?:com|co)\.[a-z]{2})", host)
            or not allowed_path
        ):
            return None
    labels = {
        "search": "Open flight search",
        "property_website": "Open property page",
        "hotel_search": "Open hotel search",
    }
    return OfferRedirect(
        url=url,
        kind=kind,
        label=labels[kind],
    )

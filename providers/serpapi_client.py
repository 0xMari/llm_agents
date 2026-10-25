import os
import requests

from execution import RunBudget


class SerpAPIError(RuntimeError):
    code = "PROVIDER_ERROR"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.code = code or self.code
        self.status_code = status_code


class SerpAPIResponseError(SerpAPIError):
    code = "INVALID_RESPONSE"

    def __init__(self, message: str, *, kind: str = "invalid_response", **kwargs):
        super().__init__(message, **kwargs)
        allowed = {
            "invalid_response", "invalid_json", "invalid_payload", "invalid_metadata",
            "invalid_status", "missing_results", "invalid_results", "invalid_city",
            "invalid_city_locations", "invalid_flight_leg", "flight_date_mismatch",
            "flight_airport_mismatch", "missing_departure_token", "invalid_flight_offer",
            "invalid_hotel_results", "missing_hotel_results", "all_hotel_offers_invalid",
        }
        self.kind = kind if isinstance(kind, str) and kind in allowed else "invalid_response"


def search(params, *, budget: RunBudget | None = None):
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise SerpAPIError("Configure SERPAPI_API_KEY before searching.", code="CONFIGURATION")
    if budget is not None:
        budget.claim("serpapi")
    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={**params, "api_key": api_key},
            timeout=(10, 60),
        )
        if response.status_code != 200:
            code = {401: "AUTHENTICATION", 403: "AUTHENTICATION", 429: "QUOTA_OR_RATE_LIMIT"}.get(
                response.status_code, "HTTP_ERROR"
            )
            raise SerpAPIError(f"SerpAPI HTTP {response.status_code}", code=code, status_code=response.status_code)
    except requests.Timeout:
        raise SerpAPIError("The SerpAPI search timed out.", code="TIMEOUT") from None
    except requests.RequestException:
        raise SerpAPIError("Network error during the SerpAPI search.", code="NETWORK_ERROR") from None
    try:
        data = response.json()
    except ValueError:
        raise SerpAPIResponseError("Invalid JSON response.", kind="invalid_json") from None
    if not isinstance(data, dict):
        raise SerpAPIResponseError("Unexpected response format.", kind="invalid_payload")
    metadata = data.get("search_metadata")
    if not isinstance(metadata, dict):
        raise SerpAPIResponseError("Missing or invalid search metadata.", kind="invalid_metadata")
    status = metadata.get("status")
    if status == "Error":
        raise SerpAPIError("The SerpAPI search failed.", code="SEARCH_ERROR")
    if status in ("Processing", "Queued"):
        raise SerpAPIError("The SerpAPI search has not completed yet.", code="SEARCH_PENDING")
    if status != "Success":
        raise SerpAPIResponseError("Missing or invalid search status.", kind="invalid_status")
    return data

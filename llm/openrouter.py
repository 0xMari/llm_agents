# llm/openrouter.py

import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time
import requests

from execution import RunBudget


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 5.0

_ERROR_MESSAGES = {
    "configuration_error": "Incomplete OpenRouter configuration.",
    "timeout": "The OpenRouter request timed out.",
    "network_error": "Unable to connect to OpenRouter.",
    "authentication_error": "OpenRouter authentication failed.",
    "payment_required": "Check OpenRouter credit and spending limits.",
    "permission_denied": "The OpenRouter request is not permitted.",
    "rate_limit": "The LLM provider's request limit was reached.",
    "invalid_request": "OpenRouter rejected the request parameters.",
    "not_found": "OpenRouter resource or endpoint not found.",
    "model_incompatible": "No selected endpoint supports the requested parameters.",
    "provider_unavailable": "The LLM provider is unavailable for this request.",
    "provider_error": "The LLM provider returned an error.",
    "invalid_response": "OpenRouter returned an unusable response.",
}

_HTTP_ERROR_KINDS = {
    400: "invalid_request",
    401: "authentication_error",
    402: "payment_required",
    403: "permission_denied",
    404: "not_found",
    408: "timeout",
    413: "invalid_request",
    422: "invalid_request",
    429: "rate_limit",
    502: "provider_unavailable",
    503: "provider_unavailable",
    504: "timeout",
}

_TYPED_ERROR_KINDS = {
    "authentication": "authentication_error",
    "permission_denied": "permission_denied",
    "content_policy_violation": "permission_denied",
    "refusal": "permission_denied",
    "payment_required": "payment_required",
    "rate_limit_exceeded": "rate_limit",
    "provider_overloaded": "provider_unavailable",
    "provider_unavailable": "provider_unavailable",
    "timeout": "timeout",
    "not_found": "not_found",
    "invalid_request": "invalid_request",
    "invalid_prompt": "invalid_request",
    "context_length_exceeded": "invalid_request",
    "max_tokens_exceeded": "invalid_request",
    "token_limit_exceeded": "invalid_request",
    "string_too_long": "invalid_request",
    "payload_too_large": "invalid_request",
    "unprocessable": "invalid_request",
    "server": "provider_error",
}

_UNSUPPORTED_PARAMETER_PREFIXES = (
    "no endpoints found that support the requested parameters",
    "this model does not support json_schema",
    "this model does not support json schema",
    "this model does not support structured outputs",
    "this model does not support response_format",
    "response_format of type json_schema is not supported",
    "unsupported parameter: response_format",
)


def _status_code(value: object, *, minimum: int = 100) -> int | None:
    # Do not copy arbitrary provider strings (or booleans) into the trace.
    return value if type(value) is int and minimum <= value <= 599 else None


class OpenRouterError(Exception):
    def __init__(
        self,
        message: str | None = None,
        *,
        kind: str = "provider_error",
        http_status: int | None = None,
        api_code: int | None = None,
        retry_after: str | None = None,
    ):
        self.kind = kind if isinstance(kind, str) and kind in _ERROR_MESSAGES else "provider_error"
        self.http_status = _status_code(http_status)
        self.api_code = _status_code(api_code, minimum=400)
        self.retry_after = retry_after if isinstance(retry_after, str) else None
        super().__init__(message if message is not None else _ERROR_MESSAGES[self.kind])

    def trace_details(self) -> dict[str, str | int | None]:
        return {
            "llm_error_kind": self.kind,
            "llm_http_status": self.http_status,
            "llm_api_code": self.api_code,
        }


def _provider_error(http_status: int, payload: object = None, *, retry_after: str | None = None) -> OpenRouterError:
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        kind = _HTTP_ERROR_KINDS.get(http_status, "provider_error") if http_status >= 400 else "invalid_response"
        return OpenRouterError(kind=kind, http_status=http_status, retry_after=retry_after)

    api_code = _status_code(error.get("code"), minimum=400)
    code = api_code if api_code is not None else http_status
    kind = _HTTP_ERROR_KINDS.get(code, "provider_error")
    metadata = error.get("metadata")
    error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
    typed_kind = _TYPED_ERROR_KINDS.get(error_type) if isinstance(error_type, str) else None

    # Recognize only explicit diagnostics, never echoed prompts in metadata.raw.
    message = error.get("message")
    if code in (400, 404, 422, 503) and isinstance(message, str):
        normalized = " ".join(message[:1024].lower().split())
        if normalized.startswith(_UNSUPPORTED_PARAMETER_PREFIXES):
            kind = "model_incompatible"

    # A recognized structured provider category is more precise than HTTP alone.
    if typed_kind is not None:
        if kind != "model_incompatible" or typed_kind not in ("invalid_request", "not_found", "provider_unavailable"):
            kind = typed_kind
    return OpenRouterError(kind=kind, http_status=http_status, api_code=api_code, retry_after=retry_after)


def _retry_delay(error: OpenRouterError) -> float | None:
    code = error.api_code or error.http_status
    if error.kind == "provider_error":
        if code != 500:
            return None
    elif error.kind not in {"timeout", "network_error", "provider_unavailable"}:
        return None
    elif code not in {None, 200, 408, 500, 502, 503, 504}:
        return None

    if error.retry_after is None:
        return _RETRY_DELAY_SECONDS
    try:
        delay = float(error.retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(error.retry_after)
            if retry_at.tzinfo is None:
                return None
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    # Do not retry early when the provider asks for a longer or invalid wait.
    if not math.isfinite(delay) or not 0 <= delay <= _MAX_RETRY_DELAY_SECONDS:
        return None
    return max(_RETRY_DELAY_SECONDS, delay)


def call_openrouter_json(
    messages: list[dict[str, str]],
    response_schema: dict,
    schema_name: str = "travel_window_advice",
    *,
    budget: RunBudget | None = None,
) -> str:
    """Automatic retries require the RunBudget shared by the whole planning run."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key or not api_key.strip():
        raise OpenRouterError("OPENROUTER_API_KEY is not configured.", kind="configuration_error")

    if not model or not model.strip():
        raise OpenRouterError("OPENROUTER_MODEL is not configured.", kind="configuration_error")

    if budget is not None:
        budget.claim("llm")
    try:
        return _request_openrouter_json(messages, response_schema, schema_name, api_key=api_key, model=model)
    except OpenRouterError as error:
        delay = _retry_delay(error)
        if budget is None or delay is None or not budget.claim_llm_retry():
            raise
        budget.llm_retry_details = {
            key.replace("llm_", "llm_retry_", 1): value
            for key, value in error.trace_details().items()
        }
    time.sleep(delay)
    return _request_openrouter_json(messages, response_schema, schema_name, api_key=api_key, model=model)


def _request_openrouter_json(
    messages: list[dict[str, str]], response_schema: dict, schema_name: str,
    *, api_key: str, model: str,
) -> str:
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-OpenRouter-Title": "AI Travel Planner",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "provider": {
                    "require_parameters": True,
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
                    },
                },
            },
            timeout=60,
        )
    except requests.Timeout as exc:
        raise OpenRouterError(kind="timeout") from exc
    except requests.RequestException as exc:
        raise OpenRouterError(kind="network_error") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        if response.status_code >= 400:
            raise _provider_error(response.status_code, retry_after=response.headers.get("Retry-After")) from exc
        raise OpenRouterError(kind="invalid_response", http_status=response.status_code) from exc

    if response.status_code >= 400:
        raise _provider_error(response.status_code, payload, retry_after=response.headers.get("Retry-After"))

    if not isinstance(payload, dict):
        raise OpenRouterError(kind="invalid_response", http_status=response.status_code)

    # Generation can fail after HTTP 200 has already been sent.
    if "error" in payload and payload["error"] is not None:
        raise _provider_error(response.status_code, payload, retry_after=response.headers.get("Retry-After"))

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(kind="invalid_response", http_status=response.status_code) from exc

    if not isinstance(content, str) or not content.strip():
        raise OpenRouterError(kind="invalid_response", http_status=response.status_code)

    return content

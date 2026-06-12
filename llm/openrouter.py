# llm/openrouter.py

import os
import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(Exception):
    pass


def call_openrouter_json(
    messages: list[dict[str, str]],
    response_schema: dict,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")

    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY non configurata.")

    if not model:
        raise OpenRouterError("OPENROUTER_MODEL non configurato.")

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
                    "name": "travel_window_advice",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        },
        timeout=60,
    )

    if response.status_code >= 400:
        raise OpenRouterError(response.text)

    data = response.json()

    return data["choices"][0]["message"]["content"]
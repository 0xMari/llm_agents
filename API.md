# Local API

This HTTP layer calls the existing planner. There is no database, booking,
authentication, or persistent per-user quota in this version. Run it locally;
do not publish it as an unrestricted paid-provider proxy.

## Install And Start

From the repository directory:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.
The machine-readable contract is at <http://127.0.0.1:8000/openapi.json>.
Use Ctrl+C to stop the server. For development, optionally add `--reload`.

`SERPAPI_API_KEY`, `OPENROUTER_API_KEY`, and `OPENROUTER_MODEL` must be set
in the server's terminal for real searches. Existing environment variables are
used; `.env` files are not automatically loaded. Never put these keys in the
frontend or submit them in the request body.

Health checks, documentation, and the offline test suite do not require keys.
Swagger's "Execute" button on search endpoints does make real provider calls
when the server has working credentials.

## Endpoints

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

Returns `200` with `{"status":"ok"}`. This confirms the API process is alive,
not that provider credentials or quotas are working. It makes no provider calls.

### GET /locations?q=Rome

```bash
curl --get http://127.0.0.1:8000/locations --data-urlencode 'q=Rome'
```

The query must contain 2-120 characters, with at least two non-whitespace
characters after trimming. Each valid request permits at most one SerpAPI call
and zero LLM calls. There is no cross-request autocomplete cache yet.

Example response shape (the actual candidates come from SerpAPI):

```json
{
  "locations": [
    {
      "name": "Rome",
      "city_id": "/m/06c62",
      "description": "Capital of Italy",
      "transport_locations": [
        {"id": "FCO", "name": "Leonardo da Vinci International Airport"}
      ]
    }
  ]
}
```

`transport_locations` can include railway stations as well as airports, just
like the provider's autocomplete data. Do not automatically treat every ID as
an airport code. This API does not search or price train tickets.

The frontend should display the candidates and retain the selected `name`
and `city_id`. Use these for `origin`/`origin_city_id` or the fixed
`destination_pref.destination`/`destination_pref.city_id`. A selected ID is
checked against fresh provider candidates during proposal generation; it is
not a way to submit an arbitrary location identifier.

Empty results are a successful `200` response with `{"locations":[]}`.
Provider-only fields, including metadata and API URLs, are not forwarded.

### POST /proposals

Send a structured trip request. This example uses month mode so it does not
expire on a hardcoded exact date; the backend resolves the departure year.

```bash
curl http://127.0.0.1:8000/proposals \
  -H 'Content-Type: application/json' \
  -d '{
    "origin": "Rome",
    "origin_city_id": "/m/06c62",
    "adults": 1,
    "budget_eur": 900,
    "budget_flexibility_pct": 0,
    "destination_pref": {
      "mode": "fixed",
      "destination": "Lisbon",
      "city_id": "/m/04llb"
    },
    "time_pref": {
      "mode": "month",
      "month": 5,
      "min_days": 5,
      "max_days": 7
    },
    "vibe": {
      "free_text": "Museums, local food and relaxed walks.",
      "themes": ["museums", "food"],
      "pace": "slow"
    }
  }'
```

For open destination searches use `"destination_pref":{"mode":"open"}`.
Do not include arrival IDs in that mode. The LLM suggests up to three cities;
fewer than three usable proposals is a valid outcome.

Supported time modes are:

- `exact_dates`: `start_date` and `end_date`, as `YYYY-MM-DD`.
- `month`: a single `month` from 1 to 12, optionally with `min_days`/`max_days`.
- `season`: `spring`, `summer`, `autumn`, or `winter`, optionally with duration.
- `flexible_window`: duration preferences without a fixed month or season.

Durations count inclusive days, including departure and return. Month and season
constrain departure, not return. Seasons use the agreed Northern Hemisphere
calendar windows regardless of destination. The backend owns the reference date;
past exact-date input is rejected before calling providers.

`budget_eur` and `vibe` may be omitted. The budget covers flights plus
accommodation for the specified adults, not every holiday expense. Positive
`budget_flexibility_pct` allows the existing flexible-budget behavior; the default
is 20, so send `0` when the user wants a hard limit. The maximum flexibility is 30.

The API rejects unknown fields at the top level and inside destination, time,
and vibe objects. In particular, clients cannot submit keys, model settings,
`limits`, `country_verifier`, or `reference_date`. Location names are limited
to 160 characters, free-text vibe to 2000, and theme/environment lists to 20
items of at most 80 characters each.

## Results And Errors

Successful proposals return the existing `TripSearchResult` JSON with at most
three combinations in total. `search_status` can be `complete` or `partial`;
`search_issues` explains partial searches. `summary_status: unavailable` does
not discard valid offers. These cases still return `200`.

Amounts backed by Decimal are JSON strings, for example `"430.24"`.
Dates are ISO strings. The successful result includes the planner's diagnostic
trace; the frontend can leave it out of the main presentation. Hotel vibe
suitability and LLM travel advice are not verified facts. Redirects may be absent.

Handled errors have a stable public shape, without raw exceptions or submitted
values:

```json
{
  "code": "INVALID_REQUEST",
  "message": "The request contains invalid or missing fields.",
  "issues": [
    {
      "location": ["body", "origin"],
      "type": "missing",
      "message": "Field required"
    }
  ],
  "location_options": []
}
```

| HTTP | Meaning |
| --- | --- |
| 200 | Proposals, or a completed search with no flights/hotels/affordable combinations |
| 422 | Invalid request, past input dates, or a location requiring clarification |
| 502 | Unusable provider/model output or an LLM constraint violation |
| 503 | Provider/configuration/quota problem or the configured search-call limit reached |
| 504 | Provider timeout |
| 500 | Unexpected internal error; exception text is not returned |

For no-result searches, the `200` body has `code`, `message`, `issues`, and
`location_options`, with code `NO_FLIGHTS_FOUND`, `NO_HOTELS_FOUND`, or
`NO_AFFORDABLE_PROPOSAL`, instead of `proposals`. Check the response shape as
well as the HTTP status. Provider quota/authentication failures describe the
backend's provider account, not a failure of the frontend user's credentials.

## Server-Owned Search Profiles

| Mode | Destinations | Periods per city | Outbound candidates | SerpAPI attempts | LLM attempts |
| --- | --- | --- | --- | --- | --- |
| Fixed | 1 | 1 | 1 | 5 | 3 |
| Open | 3 | 1 | 1 | 13 | 6 |

The LLM ceiling includes room for one shared retry. Attempts count even when
they fail. Limits are per execution, not persistent per-user quotas.
Autocomplete requests made before submitting a proposal are separate searches
and are not included in these proposal counters. Avoid automatic frontend
retries on paid search endpoints; debounce autocomplete in the future UI.

## Connecting A Separate Frontend Repository

The frontend only needs this HTTP contract and a configurable API base URL,
for example `http://127.0.0.1:8000` during development. No Python files or
provider credentials need to be copied to the frontend repository.

CORS is disabled by default. If the frontend runs on another origin, configure
its exact origin before starting the API:

```bash
export FRONTEND_ORIGINS='http://localhost:5173'
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

For multiple explicit origins, separate them with commas. `localhost` and
`127.0.0.1` are different origins. Wildcard and `null` origins are rejected;
cross-origin cookies are not enabled. CORS is a browser access policy, not
authentication or protection against automated spending.

Keep a loading state while a proposal request runs: the planner makes several
sequential provider calls. Long-running jobs, cancellation, persistent quotas,
authentication, concurrency controls, request-size limits, and production
deployment/timeouts are outside this local version and need review before
public release.

## Offline Tests

```bash
.venv/bin/python -B -m unittest test_api -v
.venv/bin/python -B -m unittest discover -q
.venv/bin/python -m pip check
```

Tests use FastAPI's TestClient with HTTPX2, the supported client for the pinned
Starlette version. Provider HTTP responses are simulated; no paid provider calls
are made. Tests cover both individual endpoints and the real planner behind
the HTTP boundary, including city-ID validation and past input dates.

## Version Control

This repository currently has existing `.venv` files tracked by Git. Installing
dependencies can therefore show tracked environment changes even though the new
`.gitignore` excludes newly installed packages. Do not use `git add .` for this
change. Stage the API source, tests, documentation, requirements and `.gitignore`
explicitly; removing the already tracked environment from the index is a separate
cleanup and has not been performed here.

References: [FastAPI](https://fastapi.tiangolo.com/features/),
[CORS](https://fastapi.tiangolo.com/tutorial/cors/),
[Starlette TestClient](https://starlette.dev/testclient/).

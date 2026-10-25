from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
import json
import unittest
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, RequestException, Timeout

from execution import RunBudget, SearchLimitExceeded, SearchLimits
from llm.openrouter import OpenRouterError, call_openrouter_json
from models import DestinationPreference, EventType, PlanningError, TimePreference, TripRequest, TravelVibe
from planning import build_proposal


SECRET_KEY = "test-secret-api-key"
SECRET_PROMPT = "test-private-travel-preferences"


def http_response(payload, status=200):
    response = Mock(status_code=status, text=f"{SECRET_KEY} {SECRET_PROMPT}")
    response.json.return_value = payload
    return response


def completion(content):
    return http_response({"choices": [{"message": {"content": content}}]})


class OpenRouterDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict("os.environ", {
            "OPENROUTER_API_KEY": SECRET_KEY, "OPENROUTER_MODEL": "test-model",
        }))
        self.post = self.enterContext(patch("llm.openrouter.requests.post"))
        self.sleep = self.enterContext(patch("llm.openrouter.time.sleep"))
        self.messages = [{"role": "user", "content": SECRET_PROMPT}]

    def assert_response_error(self, response, kind, api_code=None):
        self.post.reset_mock(side_effect=True)
        self.post.return_value = response
        budget = RunBudget(SearchLimits(max_llm_calls=1))
        with self.assertRaises(OpenRouterError) as caught:
            call_openrouter_json(self.messages, {}, budget=budget)
        error = caught.exception
        self.assertEqual(error.kind, kind)
        self.assertEqual(error.http_status, response.status_code)
        self.assertEqual(error.api_code, api_code)
        self.assertEqual(budget.used, {"llm": 1, "serpapi": 0})
        self.post.assert_called_once()
        response.json.assert_called_once()
        for secret in (SECRET_KEY, SECRET_PROMPT):
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, json.dumps(error.trace_details()))
        with self.assertRaises(SearchLimitExceeded):
            call_openrouter_json(self.messages, {}, budget=budget)
        self.post.assert_called_once()
        return error

    def test_success_preserves_request_contract_and_content(self):
        content = '{"selected_periods": []}'
        self.post.return_value = completion(content)
        budget = RunBudget(SearchLimits())
        schema = {"type": "object"}
        result = call_openrouter_json(self.messages, schema, budget=budget)
        self.assertEqual(result, content)
        self.assertEqual(budget.used["llm"], 1)
        self.post.assert_called_once()
        self.post.return_value.json.assert_called_once()
        kwargs = self.post.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["json"]["provider"], {"require_parameters": True})
        self.assertEqual(kwargs["json"]["response_format"]["json_schema"], {
            "name": "travel_window_advice", "strict": True, "schema": schema,
        })

    def test_assistant_json_validation_remains_the_agents_responsibility(self):
        self.post.return_value = completion("not JSON")
        budget = RunBudget(SearchLimits(max_llm_calls=10))
        self.assertEqual(call_openrouter_json(self.messages, {}, budget=budget), "not JSON")
        self.assertEqual(budget.llm_retries, 0)
        self.post.assert_called_once()

    def test_legacy_exception_message_stays_compatible_but_is_not_traced(self):
        error = OpenRouterError(SECRET_PROMPT)
        self.assertEqual(str(error), SECRET_PROMPT)
        self.assertEqual(error.trace_details(), {
            "llm_error_kind": "provider_error", "llm_http_status": None, "llm_api_code": None,
        })

    def test_diagnostic_fields_are_allowlisted(self):
        for value in (SECRET_KEY, {}, [], True, 0, 999, 429.0):
            with self.subTest(value=value):
                error = OpenRouterError(kind=value, http_status=value, api_code=value)
                self.assertEqual(error.trace_details(), {
                    "llm_error_kind": "provider_error", "llm_http_status": None, "llm_api_code": None,
                })

    def test_missing_or_blank_configuration_does_not_consume_budget(self):
        for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
            for value in ("", "   "):
                with self.subTest(name=name, value=value), patch.dict("os.environ", {name: value}):
                    budget = RunBudget(SearchLimits(max_llm_calls=0))
                    with self.assertRaises(OpenRouterError) as caught:
                        call_openrouter_json(self.messages, {}, budget=budget)
                    self.assertEqual(caught.exception.kind, "configuration_error")
                    self.assertIsNone(caught.exception.http_status)
                    self.assertEqual(budget.used["llm"], 0)
        self.post.assert_not_called()

    def test_transport_errors_are_classified_without_retry_or_refund(self):
        for exception, kind in (
            (Timeout, "timeout"), (ConnectTimeout, "timeout"), (ReadTimeout, "timeout"),
            (ConnectionError, "network_error"), (RequestException, "network_error"),
        ):
            with self.subTest(exception=exception.__name__):
                self.post.reset_mock()
                cause = exception(f"{SECRET_KEY} {SECRET_PROMPT}")
                self.post.side_effect = cause
                budget = RunBudget(SearchLimits(max_llm_calls=1))
                with self.assertRaises(OpenRouterError) as caught:
                    call_openrouter_json(self.messages, {}, budget=budget)
                self.assertEqual(caught.exception.kind, kind)
                self.assertIs(caught.exception.__cause__, cause)
                self.assertIsNone(caught.exception.http_status)
                self.assertEqual(budget.used["llm"], 1)
                for secret in (SECRET_KEY, SECRET_PROMPT):
                    self.assertNotIn(secret, str(caught.exception))
                    self.assertNotIn(secret, json.dumps(caught.exception.trace_details()))
                with self.assertRaises(SearchLimitExceeded):
                    call_openrouter_json(self.messages, {}, budget=budget)
                self.post.assert_called_once()

    def test_status_and_body_error_codes_are_classified(self):
        cases = (
            (400, "invalid_request"), (401, "authentication_error"),
            (402, "payment_required"), (403, "permission_denied"), (404, "not_found"),
            (408, "timeout"), (413, "invalid_request"), (422, "invalid_request"),
            (429, "rate_limit"), (500, "provider_error"), (502, "provider_unavailable"),
            (503, "provider_unavailable"), (504, "timeout"), (599, "provider_error"),
        )
        for code, kind in cases:
            for status in (200, code):
                with self.subTest(code=code, status=status):
                    response = http_response({"error": {
                        "code": code, "message": f"{SECRET_KEY} {SECRET_PROMPT}",
                    }}, status)
                    self.assert_response_error(response, kind, api_code=code)

    def test_structured_categories_take_precedence_over_numeric_codes(self):
        cases = (
            ("authentication", "authentication_error"), ("permission_denied", "permission_denied"),
            ("content_policy_violation", "permission_denied"), ("refusal", "permission_denied"),
            ("payment_required", "payment_required"), ("rate_limit_exceeded", "rate_limit"),
            ("provider_overloaded", "provider_unavailable"), ("provider_unavailable", "provider_unavailable"),
            ("timeout", "timeout"), ("not_found", "not_found"), ("invalid_request", "invalid_request"),
            ("context_length_exceeded", "invalid_request"), ("server", "provider_error"),
        )
        for error_type, kind in cases:
            with self.subTest(error_type=error_type):
                response = http_response({"error": {"code": 400, "metadata": {"error_type": error_type}}})
                self.assert_response_error(response, kind, api_code=400)

    def test_untrusted_metadata_and_codes_are_not_exposed(self):
        for code in (SECRET_KEY, "429", True, {}, [], 200, 700, 429.0):
            for metadata in (None, [], SECRET_PROMPT, {"error_type": []}, {
                "error_type": SECRET_KEY, "provider_code": SECRET_KEY, "raw": SECRET_PROMPT,
            }):
                with self.subTest(code=code, metadata=metadata):
                    response = http_response({"error": {
                        "code": code, "message": SECRET_PROMPT, "metadata": metadata,
                    }}, status=429)
                    self.assert_response_error(response, "rate_limit")

    def test_explicit_parameter_incompatibility_is_recognized(self):
        for message in (
            "No endpoints found that support the requested parameters. See the docs.",
            "This model does not support structured outputs.",
            "This model does not support json_schema.",
            "This model does not support JSON schema.",
            "This model does not support response_format.",
            "response_format of type json_schema is not supported.",
            "Unsupported parameter: response_format",
        ):
            with self.subTest(message=message):
                response = http_response({"error": {"code": 404, "message": message}}, status=404)
                self.assert_response_error(response, "model_incompatible", api_code=404)

    def test_generic_failures_or_echoed_prompts_do_not_imply_incompatibility(self):
        for code, kind in ((400, "invalid_request"), (404, "not_found"), (503, "provider_unavailable")):
            for message in ("Provider returned error", "No endpoints found matching your data policy", {
                "text": "This model does not support structured outputs",
            }, "Invalid prompt: This model does not support structured outputs"):
                with self.subTest(code=code, message=message):
                    response = http_response({"error": {
                        "code": code, "message": message,
                        "metadata": {"raw": "This model does not support structured outputs"},
                    }}, status=code)
                    self.assert_response_error(response, kind, api_code=code)

    def test_explicit_auth_or_rate_limit_is_not_misclassified_as_incompatible(self):
        for code, metadata, kind in (
            (401, {}, "authentication_error"),
            (404, {"error_type": "rate_limit_exceeded"}, "rate_limit"),
        ):
            with self.subTest(code=code):
                response = http_response({"error": {
                    "code": code, "metadata": metadata,
                    "message": "No endpoints found that support the requested parameters.",
                }}, status=code)
                self.assert_response_error(response, kind, api_code=code)

    def test_http_error_keeps_status_with_non_json_or_malformed_body(self):
        for payload in (None, [], "not an object", {"error": []}, {"error": None}):
            with self.subTest(payload=payload):
                self.assert_response_error(http_response(payload, status=503), "provider_unavailable")
        response = http_response(None, status=503)
        cause = ValueError(SECRET_PROMPT)
        response.json.side_effect = cause
        error = self.assert_response_error(response, "provider_unavailable")
        self.assertIs(error.__cause__, cause)

    def test_http_200_non_json_response_is_invalid_response(self):
        response = http_response(None)
        cause = ValueError(SECRET_PROMPT)
        response.json.side_effect = cause
        error = self.assert_response_error(response, "invalid_response")
        self.assertIs(error.__cause__, cause)

    def test_invalid_envelopes_and_empty_content_are_rejected(self):
        payloads = [None, [], "not an object", {}, {"error": SECRET_PROMPT}, {"error": []}]
        payloads.extend({"choices": choices} for choices in (None, [], {}, [None], [{}], [{"message": None}]))
        payloads.extend({"choices": [{"message": {"content": content}}]} for content in (None, "", "  ", [], 1))
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assert_response_error(http_response(payload), "invalid_response")

    def test_error_in_body_takes_precedence_over_partial_content(self):
        response = http_response({
            "error": {"code": 429}, "choices": [{"message": {"content": "partial output"}}],
        })
        self.assert_response_error(response, "rate_limit", api_code=429)

    def test_transient_http_errors_retry_once_with_identical_request_and_shared_budget(self):
        for status, code in ((408, 408), (500, 500), (502, 502), (503, 503), (504, 504), (200, 502)):
            with self.subTest(status=status, code=code):
                self.post.reset_mock()
                self.sleep.reset_mock()
                self.post.side_effect = [http_response({"error": {"code": code}}, status), completion("{}")]
                budget = RunBudget(SearchLimits(max_llm_calls=2))
                self.assertEqual(call_openrouter_json(self.messages, {}, budget=budget), "{}")
                self.assertEqual(budget.used, {"llm": 2, "serpapi": 0})
                self.assertEqual(budget.llm_retries, 1)
                self.assertEqual(budget.llm_retry_details["llm_retry_api_code"], code)
                self.assertEqual(self.post.call_count, 2)
                self.assertEqual(self.post.call_args_list[0], self.post.call_args_list[1])
                self.sleep.assert_called_once_with(1.0)

    def test_transient_transport_failure_can_recover(self):
        for exception in (Timeout, ConnectTimeout, ReadTimeout, ConnectionError, RequestException):
            with self.subTest(exception=exception.__name__):
                self.post.reset_mock()
                self.sleep.reset_mock()
                self.post.side_effect = [exception(SECRET_PROMPT), completion("{}")]
                budget = RunBudget(SearchLimits(max_llm_calls=2))
                self.assertEqual(call_openrouter_json(self.messages, {}, budget=budget), "{}")
                self.assertEqual(budget.used["llm"], 2)
                self.assertEqual(budget.llm_retries, 1)
                self.assertNotIn(SECRET_PROMPT, json.dumps(budget.llm_retry_details))
                self.sleep.assert_called_once()

    def test_502_with_non_json_gateway_response_can_recover(self):
        response = http_response(None, 502)
        response.json.side_effect = ValueError("PRIVATE gateway response")
        self.post.side_effect = [response, completion("{}")]
        budget = RunBudget(SearchLimits())
        self.assertEqual(call_openrouter_json(self.messages, {}, budget=budget), "{}")
        self.assertEqual(budget.llm_retries, 1)
        self.assertNotIn("PRIVATE", json.dumps(budget.llm_retry_details))

    def test_second_failure_keeps_latest_diagnostics_without_third_attempt(self):
        self.post.side_effect = [
            http_response({"error": {"code": 502}}, 502),
            http_response({"error": {"code": 504}}, 504),
        ]
        budget = RunBudget(SearchLimits(max_llm_calls=10))
        with self.assertRaises(OpenRouterError) as caught:
            call_openrouter_json(self.messages, {}, budget=budget)
        self.assertEqual(caught.exception.kind, "timeout")
        self.assertEqual(caught.exception.http_status, 504)
        self.assertEqual(budget.llm_retry_details["llm_retry_http_status"], 502)
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(budget.used["llm"], 2)
        self.sleep.assert_called_once()

    def test_retry_is_shared_across_schema_calls_and_resets_for_a_new_run(self):
        self.post.side_effect = [
            http_response({"error": {"code": 502}}, 502), completion("{}"),
            http_response({"error": {"code": 503}}, 503),
            http_response({"error": {"code": 502}}, 502), completion("{}"),
        ]
        budget = RunBudget(SearchLimits(max_llm_calls=10))
        call_openrouter_json(self.messages, {}, schema_name="destination_advice", budget=budget)
        with self.assertRaises(OpenRouterError):
            call_openrouter_json(self.messages, {}, schema_name="travel_window_advice", budget=budget)
        self.assertEqual(budget.used["llm"], 3)
        self.assertEqual(budget.llm_retries, 1)
        next_budget = RunBudget(SearchLimits(max_llm_calls=2))
        call_openrouter_json(self.messages, {}, budget=next_budget)
        self.assertEqual(next_budget.llm_retries, 1)
        self.assertEqual(next_budget.used["llm"], 2)
        self.assertEqual(self.post.call_count, 5)
        self.assertEqual(self.sleep.call_count, 2)

    def test_no_spare_call_preserves_original_error_without_reserving_retry_or_sleeping(self):
        self.post.return_value = http_response({"error": {"code": 502}}, 502)
        budget = RunBudget(SearchLimits(max_llm_calls=1))
        with self.assertRaises(OpenRouterError) as caught:
            call_openrouter_json(self.messages, {}, budget=budget)
        self.assertEqual(caught.exception.kind, "provider_unavailable")
        self.assertEqual(budget.llm_retries, 0)
        self.assertEqual(budget.llm_retry_details, {})
        self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_no_shared_budget_does_not_enable_untracked_retries(self):
        self.post.return_value = http_response({"error": {"code": 502}}, 502)
        with self.assertRaises(OpenRouterError):
            call_openrouter_json(self.messages, {})
        self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_permanent_and_invalid_response_failures_do_not_use_the_retry(self):
        responses = [http_response({"error": {"code": code}}, code) for code in (400, 401, 402, 403, 404, 413, 422, 429)]
        invalid_json = http_response(None)
        invalid_json.json.side_effect = ValueError(SECRET_PROMPT)
        responses.extend([
            http_response({"error": {"code": 503, "message": "This model does not support structured outputs."}}, 503),
            http_response({"error": {"code": 502, "metadata": {"error_type": "authentication"}}}, 502),
            http_response({"error": {"code": 429, "metadata": {"error_type": "provider_overloaded"}}}, 429),
            http_response({"choices": []}),
            invalid_json,
        ])
        for response in responses:
            with self.subTest(status=response.status_code):
                self.post.reset_mock()
                self.post.return_value = response
                budget = RunBudget(SearchLimits(max_llm_calls=10))
                with self.assertRaises(OpenRouterError):
                    call_openrouter_json(self.messages, {}, budget=budget)
                self.assertEqual(budget.llm_retries, 0)
                self.assertEqual(budget.used["llm"], 1)
                self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_retry_after_is_respected_within_a_bounded_wait(self):
        for header, expected in (("0", 1.0), ("2", 2.0), ("5", 5.0)):
            with self.subTest(header=header):
                self.sleep.reset_mock()
                response = http_response({"error": {"code": 503}}, 503)
                response.headers = {"Retry-After": header}
                self.post.side_effect = [response, completion("{}")]
                call_openrouter_json(self.messages, {}, budget=RunBudget(SearchLimits()))
                self.sleep.assert_called_once_with(expected)

    def test_retry_after_http_date_is_supported(self):
        now = datetime(2026, 10, 23, 10, tzinfo=timezone.utc)
        response = http_response({"error": {"code": 503}}, 503)
        response.headers = {"Retry-After": format_datetime(now + timedelta(seconds=3), usegmt=True)}
        self.post.side_effect = [response, completion("{}")]
        with patch("llm.openrouter.datetime") as clock:
            clock.now.return_value = now
            call_openrouter_json(self.messages, {}, budget=RunBudget(SearchLimits()))
        self.sleep.assert_called_once_with(3.0)

    def test_long_or_invalid_retry_after_does_not_retry_early_or_consume_retry(self):
        for header in ("60", "-1", "NaN", "Infinity", "invalid", "Wed, 01 Jan 2031 00:00:00 GMT"):
            with self.subTest(header=header):
                self.post.reset_mock()
                response = http_response({"error": {"code": 503}}, 503)
                response.headers = {"Retry-After": header}
                self.post.return_value = response
                budget = RunBudget(SearchLimits())
                with self.assertRaises(OpenRouterError):
                    call_openrouter_json(self.messages, {}, budget=budget)
                self.assertEqual(budget.llm_retries, 0)
                self.assertEqual(budget.used["llm"], 1)
                self.post.assert_called_once()
        self.sleep.assert_not_called()

    def test_null_error_field_does_not_reject_success(self):
        self.post.return_value = http_response({"error": None, "choices": [{"message": {"content": "{}"}}]})
        self.assertEqual(call_openrouter_json(self.messages, {}), "{}")


class PlannerLLMDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        clock = self.enterContext(patch("planning.date", wraps=date))
        clock.today.return_value = date(2026, 10, 25)
        self.enterContext(patch("requests.sessions.Session.request", side_effect=AssertionError("Live HTTP disabled")))
        self.enterContext(patch.dict("os.environ", {
            "OPENROUTER_API_KEY": SECRET_KEY, "OPENROUTER_MODEL": "test-model",
        }))
        self.post = self.enterContext(patch("llm.openrouter.requests.post"))
        self.sleep = self.enterContext(patch("llm.openrouter.time.sleep"))
        self.get = self.enterContext(patch("providers.serpapi_client.requests.get", side_effect=AssertionError("Unexpected SerpAPI call")))
        self.request = TripRequest(
            origin="Rome", origin_city_id="/m/06c62", budget_eur=900,
            destination_pref=DestinationPreference(mode="fixed", destination="Lisbon", city_id="/m/04llb"),
            time_pref=TimePreference(mode="exact_dates", start_date=date(2027, 5, 10), end_date=date(2027, 5, 15)),
            vibe=TravelVibe(free_text=SECRET_PROMPT),
        )

    def request_for(self, mode):
        if mode == "open":
            return self.request.model_copy(update={"destination_pref": DestinationPreference(mode="open")})
        return self.request

    def assert_failure(self, result, mode, kind, *, status=None, api_code=None, calls=1):
        self.assertIsInstance(result, PlanningError)
        self.assertEqual(result.code, "LLM_PROVIDER_ERROR")
        event = result.trace.events[-1]
        expected_event = EventType.DESTINATION_RESOLUTION_FAILED if mode == "open" else EventType.TRAVEL_WINDOW_FAILED
        self.assertEqual(event.event_type, expected_event)
        details = event.details
        self.assertEqual(details["llm_error_kind"], kind)
        self.assertEqual(details["llm_http_status"], status)
        self.assertEqual(details["llm_api_code"], api_code)
        self.assertEqual(details["llm_calls"], calls)
        self.assertEqual(details["serpapi_calls"], 0)
        self.assertEqual(self.post.call_count, calls)
        self.get.assert_not_called()
        for secret in (SECRET_KEY, SECRET_PROMPT):
            self.assertNotIn(secret, result.model_dump_json())
        self.assertEqual(PlanningError.model_validate_json(result.model_dump_json()), result)

    def test_both_agents_propagate_sanitized_remote_error_and_counts(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                self.post.reset_mock()
                self.post.return_value = http_response({"error": {
                    "code": 429, "message": f"{SECRET_KEY} {SECRET_PROMPT}",
                    "metadata": {"error_type": "rate_limit_exceeded", "raw": SECRET_PROMPT},
                }})
                result = build_proposal(self.request_for(mode), limits=SearchLimits())
                self.assert_failure(result, mode, "rate_limit", status=200, api_code=429)

    def test_both_agents_propagate_timeout_and_network_errors_without_raw_cause(self):
        for mode in ("fixed", "open"):
            for exception, kind in ((Timeout, "timeout"), (ConnectionError, "network_error")):
                with self.subTest(mode=mode, kind=kind):
                    self.post.reset_mock()
                    self.post.side_effect = exception(f"{SECRET_KEY} {SECRET_PROMPT}")
                    result = build_proposal(self.request_for(mode), limits=SearchLimits())
                    self.assert_failure(result, mode, kind, calls=2)
                    self.assertEqual(result.trace.events[-1].details["llm_retries"], 1)

    def test_both_agents_stop_before_http_if_configuration_is_missing(self):
        for mode in ("fixed", "open"):
            for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
                with self.subTest(mode=mode, key=key), patch.dict("os.environ", {key: ""}):
                    result = build_proposal(self.request_for(mode), limits=SearchLimits(max_llm_calls=0))
                    self.assert_failure(result, mode, "configuration_error", calls=0)

    def test_invalid_transport_response_is_not_invalid_assistant_json(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                self.post.reset_mock()
                self.post.return_value = http_response({"choices": []})
                result = build_proposal(self.request_for(mode), limits=SearchLimits())
                self.assert_failure(result, mode, "invalid_response", status=200)

    def test_invalid_assistant_json_keeps_existing_business_error_code(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                self.post.reset_mock()
                self.post.return_value = completion(SECRET_PROMPT)
                result = build_proposal(self.request_for(mode), limits=SearchLimits())
                self.assertEqual(result.code, "LLM_OUTPUT_INVALID")
                self.assertNotIn("llm_error_kind", result.trace.events[-1].details)
                self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
                self.assertNotIn(SECRET_PROMPT, result.model_dump_json())
                self.post.assert_called_once()
                self.get.assert_not_called()

    def test_date_violation_keeps_existing_business_error_code(self):
        self.post.return_value = completion(json.dumps({
            "selected_periods": [{"start": "2027-05-11", "end": "2027-05-16", "reason": "Test", "score": 80}],
            "warnings": [], "alternatives": [], "rationale": "Test",
        }))
        result = build_proposal(self.request, limits=SearchLimits())
        self.assertEqual(result.code, "LLM_CONSTRAINT_VIOLATION")
        self.assertNotIn("llm_error_kind", result.trace.events[-1].details)
        self.assertEqual(result.trace.events[-1].details["llm_calls"], 1)
        self.post.assert_called_once()
        self.get.assert_not_called()

    def test_window_failure_after_destination_success_preserves_shared_budget(self):
        self.post.side_effect = [
            completion(json.dumps({"destinations": [{
                "name": "Lisbon", "country": "Portugal", "reason": "Test", "match_score": 80,
                "themes_matched": [], "caution": None,
            }]})),
            http_response({"error": {"code": 503}}, status=503),
        ]
        result = build_proposal(self.request_for("open"), limits=SearchLimits(max_llm_calls=2))
        self.assert_failure(result, "fixed", "provider_unavailable", status=503, api_code=503, calls=2)
        self.assertIn(EventType.DESTINATIONS_RESOLVED, [event.event_type for event in result.trace.events])

    def test_local_search_limit_stays_distinct_from_provider_rate_limit(self):
        for mode in ("fixed", "open"):
            with self.subTest(mode=mode):
                result = build_proposal(self.request_for(mode), limits=SearchLimits(max_llm_calls=0))
                self.assertEqual(result.code, "SEARCH_LIMIT_REACHED")
                self.assertNotIn("llm_error_kind", result.trace.events[-1].details)
                self.assertEqual(result.trace.events[-1].details["llm_calls"], 0)
                self.post.assert_not_called()
                self.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()

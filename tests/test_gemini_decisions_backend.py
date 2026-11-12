# Copyright 2026 Jigyasu Juneja
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit and guard-branch mutation tests for GeminiDecisionsBackend and GA Vertex presets."""

from __future__ import annotations

from collections.abc import Callable
import http.server
import json
import math
import os
import threading
import time
from typing import Any
import unittest
from unittest import mock

from multimodal_decision_models import CascadeDecisionOrchestrator, DecisionQuestion, DecisionTerminalState, FailureCategory, GA_VERTEX_GEMINI_MODEL_PRESETS, GA_VERTEX_GEMINI_MODELS, GeminiDecisionsBackend, LogprobAlignmentError, MockCallableBackend, SchemaContractViolationError, UpstreamTransientError, VertexGeminiBackend


class _LocalDecisionsServer:
    """Context manager running an ephemeral 127.0.0.1 HTTP server for /v1beta/decisions tests."""

    def __init__(self, handler_fn: Callable[[dict[str, Any], dict[str, str], str], tuple[int, str | dict[str, Any]]]) -> None:
        self._handler_fn = handler_fn
        self.requests: list[dict[str, Any]] = []
        self.headers_seen: list[dict[str, str]] = []
        self.paths_seen: list[str] = []
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str = ""

    def __enter__(self) -> _LocalDecisionsServer:
        outer = self

        class _RequestHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                parsed_body = json.loads(raw_body) if raw_body else {}
                captured_headers = {key.lower(): val for key, val in self.headers.items()}
                outer.requests.append(parsed_body)
                outer.headers_seen.append(captured_headers)
                outer.paths_seen.append(self.path)
                status_code, payload_or_raw = outer._handler_fn(
                    parsed_body, captured_headers, self.path
                )
                encoded = (
                    payload_or_raw.encode("utf-8")
                    if isinstance(payload_or_raw, str)
                    else json.dumps(payload_or_raw).encode("utf-8")
                )
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format_str: str, *args: object) -> None:
                del format_str, args

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RequestHandler)
        port = server.server_address[1]
        self._server = server
        self.base_url = f"http://127.0.0.1:{port}/v1beta/decisions"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        del exc_type, exc_val, exc_tb
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class GeminiDecisionsBackendTests(unittest.TestCase):
    """Contract, routing, log-space normalization, and fail-closed tests for GeminiDecisionsBackend."""

    def test_ga_vertex_gemini_presets_cover_all_public_models(self) -> None:
        """Verify GA_VERTEX_GEMINI_MODELS and VertexGeminiBackend.from_ga_preset."""
        self.assertEqual(
            GA_VERTEX_GEMINI_MODELS,
            (
                "gemini-3.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.8-flash",
                "gemini-3.5-pro",
            ),
        )
        for model_id in GA_VERTEX_GEMINI_MODELS:
            self.assertIn(model_id, GA_VERTEX_GEMINI_MODEL_PRESETS)
            vertex_client = VertexGeminiBackend.from_ga_preset(
                model_id,
                project="demo-project-01",
                location="us-central1",
                access_token_fn=lambda: "unit-test-token",
            )
            self.assertEqual(vertex_client.model, model_id)
            self.assertIn(model_id, vertex_client.endpoint_url())

        with self.assertRaises(SchemaContractViolationError):
            VertexGeminiBackend.from_ga_preset("gemini-unknown-preview", project="demo-project-01")

    def test_https_enforcement_and_query_credential_leak_rejection(self) -> None:
        """Reject non-loopback HTTP URLs, embedded URL API keys, and invalid parameters."""
        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(
                api_key="test-key",
                base_url="http://generativelanguage.googleapis.com/v1beta/decisions",
            )

        for leaked_param in ("key", "api_key", "apikey", "access_token", "token", "secret"):
            with self.assertRaises(SchemaContractViolationError):
                GeminiDecisionsBackend(
                    api_key="test-key",
                    base_url=f"https://generativelanguage.googleapis.com/v1beta/decisions?{leaked_param}=abc",
                )

        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(api_key="test-key", temperature=0.0)
        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(api_key="test-key", temperature=-1.0)
        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(api_key="test-key", timeout_seconds=0.0)
        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(api_key="test-key", execution_mode="INVALID_MODE")
        with self.assertRaises(SchemaContractViolationError):
            GeminiDecisionsBackend(api_key="bad\r\nheader-injection")

    def test_noul_choice_and_score_round_trip_with_extreme_logspace_stability(self) -> None:
        """Verify binary->noul, categorical->choice, and rubric->score over 127.0.0.1 HTTP."""

        def respond(
            request_body: dict[str, Any],
            request_headers: dict[str, str],
            request_path: str,
        ) -> tuple[int, dict[str, Any]]:
            self.assertEqual(request_headers.get("x-goog-api-key"), "env-secret-key")
            self.assertNotIn("env-secret-key", request_path)
            self.assertEqual(request_body["model"], "models/gemini-decision-1")
            wire_questions = request_body["questions"]
            self.assertEqual(len(wire_questions), 3)
            self.assertIn("noul", wire_questions[0])
            self.assertIn("choice", wire_questions[1])
            self.assertIn("score", wire_questions[2])
            return 200, {
                "model_version": "gemini-decision-1-2026-10",
                "usage_metadata": {"total_input_tokens": 412},
                "decisions": [
                    {
                        "id": "policy_compliant",
                        "noul": {
                            "no_log_likelihood": -1050.0,
                            "yes_log_likelihood": -1048.0,
                        },
                    },
                    {
                        "id": "route_queue",
                        "choice": {
                            "log_likelihoods": [-0.3, -1.7, -3.1],
                        },
                    },
                    {
                        "id": "damage_severity",
                        "score": {
                            "log_likelihoods": [-2.5, -0.4, -1.6],
                            "expected_score": 2.15,
                            "variance": 0.34,
                        },
                    },
                ],
            }

        questions = {
            "policy_compliant": DecisionQuestion(
                type="choice",
                instructions="Does the receipt satisfy reimbursement policy?",
                criteria={"Yes": "Compliant", "No": "Non-compliant"},
                order=["Yes", "No"],
            ),
            "route_queue": DecisionQuestion(
                type="choice",
                instructions="Select the handling queue.",
                criteria={
                    "auto_approve": "Automated approval",
                    "manual_audit": "Specialist review",
                    "fraud_hold": "Risk hold",
                },
            ),
            "damage_severity": DecisionQuestion(
                type="choice",
                instructions="Rate visible packaging damage from 1 to 3.",
                criteria={"1": "Intact", "2": "Minor dent", "3": "Crushed"},
                order=["1", "2", "3"],
            ),
        }

        with _LocalDecisionsServer(respond) as server:
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-secret-key"}, clear=False):
                backend = GeminiDecisionsBackend(
                    base_url=server.base_url,
                    model="gemini-decision-1",
                    max_retries=0,
                )
                decision_result = backend.decide(
                    state={"claim_id": "CLM-904", "note": "box corner slightly dented"},
                    questions=questions,
                    images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="],
                )

        self.assertEqual(decision_result.terminal_state, DecisionTerminalState.STAGE1_RESOLVED)
        self.assertEqual(
            decision_result.raw_metadata["model_version"], "gemini-decision-1-2026-10"
        )
        self.assertEqual(
            decision_result.raw_metadata["usage_metadata"]["total_input_tokens"], 412
        )

        noul_probs = decision_result.slots["policy_compliant"].probabilities
        expected_yes = 1.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(noul_probs["Yes"], expected_yes, places=6)
        self.assertAlmostEqual(noul_probs["No"], 1.0 - expected_yes, places=6)

        choice_slot = decision_result.slots["route_queue"]
        self.assertEqual(choice_slot.predicted, "auto_approve")
        self.assertAlmostEqual(sum(choice_slot.probabilities.values()), 1.0, places=7)

        score_meta = decision_result.raw_metadata["slots"]["damage_severity"]
        self.assertEqual(score_meta["question_kind"], "score")
        self.assertAlmostEqual(score_meta["expected_score"], 2.15, places=6)
        self.assertAlmostEqual(score_meta["variance"], 0.34, places=6)

    def test_wide_catalog_300_options_bridges_automatically_without_400(self) -> None:
        """Bridge K=300 (>255) and K=255 with include_none_of_above=True via shortlist_wide_question."""
        wide_criteria = {f"sku_{idx:04d}": f"Beverage item {idx}" for idx in range(300)}
        wide_priors = {f"sku_{idx:04d}": (-0.01 * float(idx)) for idx in range(300)}
        wide_priors["sku_0288"] = 6.5
        wide_question = DecisionQuestion(
            type="choice",
            instructions="Identify the exact shelf SKU.",
            criteria=wide_criteria,
            prior_logits=wide_priors,
        )

        def respond_standard(
            request_body: dict[str, Any],
            _headers: dict[str, str],
            _path: str,
        ) -> tuple[int, dict[str, Any]]:
            wire_options = request_body["questions"][0]["choice"]["options"]
            self.assertLessEqual(len(wire_options), 255)
            option_keys = [entry["key"] for entry in wire_options]
            self.assertIn("sku_0288", option_keys)
            self.assertIn("none_of_the_above", option_keys)
            lls = [0.5 if key == "sku_0288" else -6.0 for key in option_keys]
            return 200, {
                "model_version": "gemini-decision-1-001",
                "usage_metadata": {"total_input_tokens": 980},
                "decisions": [{"id": "sku_slot", "choice": {"log_likelihoods": lls}}],
            }

        with _LocalDecisionsServer(respond_standard) as server:
            backend = GeminiDecisionsBackend(
                api_key="local-key",
                base_url=server.base_url,
                shortlist_top_k=120,
                include_none_of_above=False,
            )
            orchestrator = CascadeDecisionOrchestrator(
                stage1_backend=backend,
                shortlist_top_k=120,
                hesitation_threshold=0.85,
            )
            cascade_result = orchestrator.evaluate(
                state="shelf crop showing beverage 288",
                questions={"sku_slot": wide_question},
            )

        self.assertEqual(cascade_result.slots["sku_slot"].predicted, "sku_0288")
        self.assertEqual(
            cascade_result.raw_metadata["usage_metadata"]["total_input_tokens"], 980
        )

        criteria_255 = {f"cat_{idx:03d}": f"Category {idx}" for idx in range(255)}
        question_255 = DecisionQuestion(
            type="choice",
            instructions="Classify into one of 255 categories or abstain.",
            criteria=criteria_255,
        )

        def respond_with_abstention(
            request_body: dict[str, Any],
            _headers: dict[str, str],
            _path: str,
        ) -> tuple[int, dict[str, Any]]:
            choice_wire = request_body["questions"][0]["choice"]
            self.assertTrue(choice_wire["include_none_of_above"])
            self.assertLessEqual(len(choice_wire["options"]), 254)
            lls = [-4.0 for _ in choice_wire["options"]]
            return 200, {
                "model_version": "gemini-decision-1-001",
                "usage_metadata": {"total_input_tokens": 640},
                "decisions": [
                    {
                        "id": "cat_slot",
                        "choice": {
                            "log_likelihoods": lls,
                            "abstention_log_likelihood": 1.2,
                        },
                    }
                ],
            }

        with _LocalDecisionsServer(respond_with_abstention) as server_abstain:
            backend_abstain = GeminiDecisionsBackend(
                api_key="local-key",
                base_url=server_abstain.base_url,
                include_none_of_above=True,
                shortlist_top_k=254,
            )
            abstain_result = backend_abstain.decide(
                state="unlisted foreign product",
                questions={"cat_slot": question_255},
            )

        self.assertEqual(abstain_result.terminal_state, DecisionTerminalState.ABSTAIN_OOS)
        self.assertEqual(abstain_result.failure_category, FailureCategory.ABSTAINED_OOS)
        self.assertEqual(abstain_result.slots["cat_slot"].predicted, "none_of_the_above")
        self.assertAlmostEqual(
            abstain_result.raw_metadata["abstention_log_likelihoods"]["cat_slot"], 1.2, places=6
        )

    def test_transient_503_and_timeout_yield_retry_pending_without_uniform_fallback(self) -> None:
        """Map HTTP 503 and socket timeouts to UpstreamTransientError / RETRY_PENDING."""
        question = DecisionQuestion(
            type="choice",
            instructions="Approve payout?",
            criteria={"approve": "Pay", "hold": "Review"},
        )

        def respond_503(
            _body: dict[str, Any],
            _headers: dict[str, str],
            _path: str,
        ) -> tuple[int, dict[str, Any]]:
            return 503, {"error": {"code": 503, "message": "Service temporarily overloaded"}}

        with _LocalDecisionsServer(respond_503) as server_503:
            backend = GeminiDecisionsBackend(
                api_key="local-key",
                base_url=server_503.base_url,
                max_retries=1,
                base_backoff_sec=0.005,
            )
            with self.assertRaises(UpstreamTransientError):
                backend.score_questions("invoice #44", {"payout": question})

            orchestrator = CascadeDecisionOrchestrator(
                stage1_backend=backend,
                stage2_backend=MockCallableBackend(
                    "unused_stage2",
                    lambda _s, _q, _i: {"payout": {"approve": 0.9, "hold": 0.1}},
                ),
                total_contract_mode=True,
            )
            quarantined = orchestrator.evaluate("invoice #44", {"payout": question})

        self.assertEqual(len(server_503.requests), 4)
        self.assertEqual(quarantined.terminal_state, DecisionTerminalState.RETRY_PENDING)
        self.assertEqual(
            quarantined.failure_category, FailureCategory.UPSTREAM_TIMEOUT_OR_THROTTLE
        )
        self.assertEqual(quarantined.slots["payout"].probabilities, {})

        def slow_respond(
            _body: dict[str, Any],
            _headers: dict[str, str],
            _path: str,
        ) -> tuple[int, dict[str, Any]]:
            time.sleep(0.25)
            return 200, {"decisions": []}

        with _LocalDecisionsServer(slow_respond) as slow_server:
            timeout_backend = GeminiDecisionsBackend(
                api_key="local-key",
                base_url=slow_server.base_url,
                timeout_seconds=0.05,
                max_retries=0,
            )
            with self.assertRaises(UpstreamTransientError):
                timeout_backend.score_questions("invoice #45", {"payout": question})

    def test_malformed_and_misaligned_logprobs_fail_closed(self) -> None:
        """Reject missing, length-mismatched, NaN, or non-JSON payloads without uniform 1/K."""
        question = DecisionQuestion(
            type="choice",
            instructions="Select route.",
            criteria={"alpha": "Route A", "beta": "Route B", "gamma": "Route C"},
        )

        bad_payloads: list[tuple[int, str | dict[str, Any], type[Exception]]] = [
            (
                200,
                {"decisions": [{"id": "route", "choice": {"log_likelihoods": [-0.2, -1.1]}}]},
                LogprobAlignmentError,
            ),
            (
                200,
                {"decisions": [{"id": "route", "choice": {}}]},
                LogprobAlignmentError,
            ),
            (
                200,
                '{"decisions": [{"id": "route", "choice": {"log_likelihoods": [NaN, -1.0, -2.0]}}]}',
                SchemaContractViolationError,
            ),
            (
                400,
                {"error": {"code": 400, "message": "Invalid schema"}},
                SchemaContractViolationError,
            ),
        ]

        for status_code, body, expected_exc in bad_payloads:
            def make_handler(
                code: int = status_code,
                response_payload: str | dict[str, Any] = body,
            ) -> Callable[[dict[str, Any], dict[str, str], str], tuple[int, str | dict[str, Any]]]:
                def _handle(
                    _req: dict[str, Any],
                    _hdrs: dict[str, str],
                    _path: str,
                ) -> tuple[int, str | dict[str, Any]]:
                    return code, response_payload

                return _handle

            with _LocalDecisionsServer(make_handler()) as bad_server:
                backend = GeminiDecisionsBackend(
                    api_key="local-key",
                    base_url=bad_server.base_url,
                    max_retries=0,
                )
                with self.assertRaises(expected_exc):
                    backend.score_questions("state", {"route": question})


if __name__ == "__main__":
    unittest.main()

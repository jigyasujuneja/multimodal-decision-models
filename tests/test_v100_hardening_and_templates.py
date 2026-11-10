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

"""Verification suite for v1.0.0 stateless HMAC conformal parity, HTTP hardening, routing ladder, and templates."""

from __future__ import annotations

import concurrent.futures
import http.server
import json
import pathlib
import re
import socketserver
import threading
from typing import Any
import unittest

from multimodal_decision_models.backends import (
    DgemSystemOneBackend,
    MockCallableBackend,
    is_raw_predict_endpoint,
    parse_url_origin,
)
from multimodal_decision_models.conformal import (
    calibrate_conformal_quantile,
    conformal_prediction_set,
    hmac_uint64,
    hmac_uniform_float64,
)
from multimodal_decision_models.contracts import SchemaContractViolationError
from multimodal_decision_models.hierarchy import recommend_wide_routing_strategy
from multimodal_decision_models.types import DecisionQuestion

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
GO_TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"\{\{\s*default\s+(\"(?:\\.|[^\"\\])*\"|-?\d+(?:\.\d+)?|true|false)\s+\.([A-Za-z0-9_]+)\s*\|\s*toJson\s*\}\}"
)


def render_go_json_template(raw_template: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Render ``{{ default <lit> .<var> | toJson }}`` placeholders into parsed JSON."""

    def replace_match(match: re.Match[str]) -> str:
        default_literal = match.group(1)
        variable_name = match.group(2)
        if variable_name in variables:
            return json.dumps(variables[variable_name])
        default_value = json.loads(default_literal)
        return json.dumps(default_value)

    rendered_text = GO_TEMPLATE_PLACEHOLDER_RE.sub(replace_match, raw_template)
    parsed_payload = json.loads(rendered_text)
    if not isinstance(parsed_payload, dict):
        raise AssertionError("Rendered policy template root must be a JSON object")
    return parsed_payload


class _ThreadedLoopbackTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StatelessConformalAndHttpHardeningTest(unittest.TestCase):
    """Verify HMAC conformal parity, cross-origin credential stripping, :rawPredict, and templates."""

    def test_hmac_uint64_matches_go_and_ts_canonical_vector(self) -> None:
        canonical_digest = hmac_uint64("dgem-conformal-v1", "req-42:intent", "u")
        self.assertEqual(canonical_digest, 14094744155160802019)
        self.assertEqual(hmac_uint64("", "req-42:intent", "u"), 14094744155160802019)
        uniform_draw = hmac_uniform_float64("dgem-conformal-v1", "req-42:intent", "u")
        self.assertAlmostEqual(uniform_draw, 0.7640776116826353, places=12)

    def test_stateless_hmac_conformal_is_permutation_and_thread_invariant(self) -> None:
        calib_probs = [
            {"allow": 0.45, "ask": 0.45, "deny": 0.10},
            {"allow": 0.20, "ask": 0.40, "deny": 0.40},
            {"allow": 0.70, "ask": 0.15, "deny": 0.15},
            {"allow": 0.33, "ask": 0.33, "deny": 0.34},
        ]
        calib_labels = ["ask", "deny", "allow", "ask"]
        item_keys = ["row-0", "row-1", "row-2", "row-3"]

        forward_cal = calibrate_conformal_quantile(
            calib_probs,
            calib_labels,
            alpha=0.25,
            randomized=True,
            seed_key="parity-key-2026",
            item_keys=item_keys,
        )
        reversed_cal = calibrate_conformal_quantile(
            list(reversed(calib_probs)),
            list(reversed(calib_labels)),
            alpha=0.25,
            randomized=True,
            seed_key="parity-key-2026",
            item_keys=list(reversed(item_keys)),
        )
        self.assertAlmostEqual(forward_cal.q_hat, reversed_cal.q_hat, places=14)

        tie_probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        reference_set = conformal_prediction_set(
            tie_probs,
            0.55,
            randomized=True,
            seed_key="parity-key-2026",
            item_key="req-991",
        )

        def evaluate_worker(worker_index: int) -> list[str]:
            conformal_prediction_set(
                tie_probs,
                0.55,
                randomized=True,
                seed_key="parity-key-2026",
                item_key=f"distractor-{worker_index}",
            )
            evaluated = conformal_prediction_set(
                tie_probs,
                0.55,
                randomized=True,
                seed_key="parity-key-2026",
                item_key="req-991",
            )
            return evaluated.prediction_set

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(evaluate_worker, range(24)))
        for worker_prediction_set in results:
            self.assertEqual(worker_prediction_set, reference_set.prediction_set)

    def test_cross_origin_redirect_permanently_strips_credentials_even_on_bounce_back(self) -> None:
        recorded_hops: list[dict[str, Any]] = []
        state_holder: dict[str, str] = {"origin_a": "", "origin_b": ""}

        class OriginAHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body_len = int(self.headers.get("Content-Length", "0"))
                body_text = self.rfile.read(body_len).decode("utf-8")
                recorded_hops.append(
                    {
                        "server": "A",
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "dgem_key": self.headers.get("X-DGem-Key"),
                        "body": json.loads(body_text),
                    }
                )
                if self.path == "/v1/systemone":
                    self.send_response(307)
                    self.send_header("Location", f"{state_holder['origin_b']}/hop")
                    self.end_headers()
                    return
                response_bytes = json.dumps(
                    {
                        "answers": {
                            "gate": {
                                "choice": "allow",
                                "probabilities": {"allow": 0.92, "deny": 0.08},
                            }
                        }
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)

            def log_message(self, format_str: str, *args: Any) -> None:
                del format_str, args

        class OriginBHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body_len = int(self.headers.get("Content-Length", "0"))
                body_text = self.rfile.read(body_len).decode("utf-8")
                recorded_hops.append(
                    {
                        "server": "B",
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "dgem_key": self.headers.get("X-DGem-Key"),
                        "body": json.loads(body_text),
                    }
                )
                self.send_response(307)
                self.send_header("Location", f"{state_holder['origin_a']}/final")
                self.end_headers()

            def log_message(self, format_str: str, *args: Any) -> None:
                del format_str, args

        server_a = _ThreadedLoopbackTCPServer(("127.0.0.1", 0), OriginAHandler)
        server_b = _ThreadedLoopbackTCPServer(("127.0.0.1", 0), OriginBHandler)
        state_holder["origin_a"] = f"http://127.0.0.1:{server_a.server_address[1]}"
        state_holder["origin_b"] = f"http://127.0.0.1:{server_b.server_address[1]}"

        thread_a = threading.Thread(target=server_a.serve_forever, daemon=True)
        thread_b = threading.Thread(target=server_b.serve_forever, daemon=True)
        thread_a.start()
        thread_b.start()
        try:
            backend = DgemSystemOneBackend(
                endpoint_url=state_holder["origin_a"],
                api_key="bearer-secret-token",
                server_key=" , primary-srv-key , secondary-srv-key ",
            )
            questions = {
                "gate": DecisionQuestion(
                    type="choice",
                    instructions="Allow or deny?",
                    criteria={"allow": "Allow", "deny": "Deny"},
                )
            }
            distributions = backend.score_questions({"req": 1}, questions)
            self.assertAlmostEqual(distributions["gate"]["allow"], 0.92, places=6)
            self.assertEqual(len(recorded_hops), 3)
            self.assertEqual(recorded_hops[0]["auth"], "Bearer bearer-secret-token")
            self.assertEqual(recorded_hops[0]["dgem_key"], "primary-srv-key")
            self.assertIsNone(recorded_hops[1]["auth"])
            self.assertIsNone(recorded_hops[1]["dgem_key"])
            self.assertIsNone(recorded_hops[2]["auth"])
            self.assertIsNone(recorded_hops[2]["dgem_key"])
            self.assertEqual(recorded_hops[2]["body"]["state"], {"req": 1})
        finally:
            server_a.shutdown()
            server_b.shutdown()
            server_a.server_close()
            server_b.server_close()

    def test_raw_predict_single_route_health_and_header_injection_guards(self) -> None:
        observed_requests: list[tuple[str, str, dict[str, Any]]] = []

        class RawPredictHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body_len = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(body_len).decode("utf-8"))
                observed_requests.append((self.command, self.path, payload))
                if payload.get("health") is True:
                    reply = {"status": "ok", "route": "rawPredict"}
                else:
                    reply = {
                        "answers": {
                            "slot_a": {
                                "choice": "yes",
                                "probabilities": {"yes": 0.85, "no": 0.15},
                            }
                        }
                    }
                encoded = json.dumps(reply).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format_str: str, *args: Any) -> None:
                del format_str, args

        raw_server = _ThreadedLoopbackTCPServer(("127.0.0.1", 0), RawPredictHandler)
        raw_thread = threading.Thread(target=raw_server.serve_forever, daemon=True)
        raw_thread.start()
        try:
            raw_url = (
                f"http://127.0.0.1:{raw_server.server_address[1]}"
                "/v1/projects/demo/locations/us-central1/endpoints/123:rawPredict"
            )
            self.assertTrue(is_raw_predict_endpoint(raw_url))
            backend = DgemSystemOneBackend(endpoint_url=raw_url, api_key="tok-1")
            self.assertEqual(backend.resolve_predict_url(), raw_url)
            health_payload = backend.health()
            self.assertEqual(health_payload["status"], "ok")
            scores = backend.score_questions(
                {"item": "x"},
                {
                    "slot_a": DecisionQuestion(
                        type="choice",
                        instructions="Confirm?",
                        criteria={"yes": "Yes", "no": "No"},
                    )
                },
            )
            self.assertAlmostEqual(scores["slot_a"]["yes"], 0.85, places=6)
            self.assertEqual(len(observed_requests), 2)
            self.assertTrue(observed_requests[0][1].endswith(":rawPredict"))
            self.assertTrue(observed_requests[1][1].endswith(":rawPredict"))
        finally:
            raw_server.shutdown()
            raw_server.server_close()

        with self.assertRaises(SchemaContractViolationError):
            parse_url_origin("https://api.example.com/v1\r\nX-Injected: 1")
        with self.assertRaises(SchemaContractViolationError):
            DgemSystemOneBackend("https://api.example.com", api_key="bad\r\ntoken")

        async def async_stub(
            state: Any,
            questions: Any,
            images: list[str] | None = None,
        ) -> dict[str, dict[str, float]]:
            del state, questions, images
            return {}

        with self.assertRaises(SchemaContractViolationError):
            MockCallableBackend("bad_async", async_stub)  # type: ignore[arg-type]

    def test_wide_option_routing_ladder_recommendations(self) -> None:
        rec_small = recommend_wide_routing_strategy(12)
        self.assertEqual(rec_small.strategy, "direct_az")
        self.assertEqual(rec_small.label_scheme, "az")

        rec_medium = recommend_wide_routing_strategy(44, supports_az_aa_52=True)
        self.assertEqual(rec_medium.strategy, "direct_az_aa")
        self.assertEqual(rec_medium.label_scheme, "az_aa")

        rec_bpe = recommend_wide_routing_strategy(180, supports_extended_255=True)
        self.assertEqual(rec_bpe.strategy, "direct_universal_255")
        self.assertEqual(rec_bpe.label_scheme, "universal_256")

        rec_emb = recommend_wide_routing_strategy(1200, has_embeddings=True)
        self.assertEqual(rec_emb.strategy, "csls_shortlist_1pass")

        rec_part = recommend_wide_routing_strategy(140, supports_az_aa_52=False, has_embeddings=False)
        self.assertEqual(rec_part.strategy, "mtp_partitioned_1pass")
        self.assertTrue(rec_part.avoids_unpartitioned_trie_aliasing)

        rec_hybrid = recommend_wide_routing_strategy(400, has_embeddings=False)
        self.assertEqual(rec_hybrid.strategy, "hesitation_hybrid")

    def test_all_policy_templates_have_valid_examples_sidecars_and_render_cleanly(self) -> None:
        template_files = sorted(TEMPLATES_DIR.rglob("*.json.tmpl"))
        self.assertGreaterEqual(len(template_files), 8)

        for tmpl_path in template_files:
            sidecar_path = tmpl_path.with_name(
                tmpl_path.name.removesuffix(".json.tmpl") + ".examples.json"
            )
            self.assertTrue(
                sidecar_path.exists(),
                f"Missing .examples.json sidecar for {tmpl_path.relative_to(REPO_ROOT)}",
            )
            raw_template = tmpl_path.read_text(encoding="utf-8")
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            examples = sidecar_data.get("examples", [])
            self.assertGreaterEqual(len(examples), 3)
            example_names = {entry["name"] for entry in examples}
            self.assertTrue({"typical", "edge", "minimal"}.issubset(example_names))

            for example in examples:
                rendered = render_go_json_template(raw_template, dict(example.get("variables", {})))
                self.assertIn("schema", rendered)
                self.assertIn("state", rendered)
                questions = rendered["schema"].get("questions", [])
                self.assertGreaterEqual(len(questions), 1)
                self.assertLessEqual(len(questions), 8)
                question_map: dict[str, dict[str, Any]] = {}
                for question in questions:
                    question_id = question["id"]
                    question_map[question_id] = question
                    q_type = question["type"]
                    self.assertIn(q_type, {"choice", "boolean", "score"})
                    if q_type == "choice":
                        options = question.get("options", [])
                        self.assertGreaterEqual(len(options), 2)
                        self.assertLessEqual(len(options), 52)
                    elif q_type == "score":
                        levels = question.get("levels", [])
                        self.assertGreaterEqual(len(levels), 2)

                for ans_slot, expected_val in example.get("expected_answers", {}).items():
                    self.assertIn(ans_slot, question_map)
                    target_question = question_map[ans_slot]
                    if target_question["type"] == "choice":
                        valid_names = {opt["name"] for opt in target_question["options"]}
                        self.assertIn(expected_val, valid_names)
                    elif target_question["type"] == "boolean":
                        self.assertIsInstance(expected_val, bool)


if __name__ == "__main__":
    unittest.main()

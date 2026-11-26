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

"""Unit tests for GevStudioSystemOneBackend and formal cascade/calibration propositions."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import run_formal_propositions_v090_bench as prop_bench  # noqa: E402
import multimodal_decision_models as mdm  # noqa: E402


class GevStudioBackendAndPropositionsTest(unittest.TestCase):
    """Verify GevStudioSystemOneBackend wire protocol and Lemma 1 / Prop 1 / Prop 2 invariants."""

    def test_gevstudio_build_url_encodes_target_query_param(self) -> None:
        backend = mdm.GevStudioSystemOneBackend(
            endpoint_url="https://gevstudio.example.com",
            target="decision:/models/gemini-decision-1?hc=true;temp=1.0",
        )
        url = backend.build_url()
        self.assertEqual(
            url,
            "https://gevstudio.example.com/api/gev/systemone"
            "?target=decision%3A%2Fmodels%2Fgemini-decision-1%3Fhc%3Dtrue%3Btemp%3D1.0",
        )

    def test_gevstudio_build_payload_formats_multimodal_media_and_question_types(self) -> None:
        backend = mdm.GevStudioSystemOneBackend(
            target="gemini-decision-1",
            model="gemini-decision-1",
            temperature=0.0,
        )
        questions = {
            "q_choice": mdm.DecisionQuestion(
                type="choice",
                instructions="Choose next action.",
                criteria={"stop": "Full stop", "slow": "Slow down"},
            ),
            "q_noul": mdm.DecisionQuestion(
                type="noul",
                instructions="Is a pedestrian visible?",
                criteria={"no": "No pedestrian", "yes": "Pedestrian visible"},
            ),
            "q_score": mdm.DecisionQuestion(
                type="score",
                instructions="Rate urgency.",
                levels=["low", "medium", "high"],
            ),
        }
        state = {
            "zone": "aisle_4",
            "__media__": [{"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"}],
        }
        payload = backend.build_payload(
            state=state,
            questions=questions,
            images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==", "/9j/4AAQSkZJRg=="],
        )
        self.assertEqual(payload["state"], {"zone": "aisle_4"})
        self.assertEqual(payload["model"], "gemini-decision-1")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(len(payload["media"]), 3)
        self.assertEqual(payload["media"][0], {"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"})
        self.assertEqual(payload["media"][1], {"mime_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg=="})
        self.assertEqual(payload["media"][2], {"mime_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="})
        self.assertEqual(payload["questions"]["q_choice"]["type"], "choice")
        self.assertEqual(
            payload["questions"]["q_noul"],
            {
                "type": "noul",
                "instructions": "Is a pedestrian visible?",
            },
        )
        self.assertEqual(
            payload["questions"]["q_score"],
            {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["low", "medium", "high"],
            },
        )

    def test_gevstudio_parse_response_choice_noul_and_score_probabilities(self) -> None:
        backend = mdm.GevStudioSystemOneBackend(target="gemini-decision-1")
        questions = {
            "action": mdm.DecisionQuestion(
                type="choice",
                instructions="Pick action.",
                criteria={"stop": "Stop", "slow": "Slow", "proceed": "Proceed"},
            ),
            "pedestrian": mdm.DecisionQuestion(
                type="noul",
                instructions="Pedestrian in path?",
                criteria={"no": "No", "yes": "Yes"},
            ),
            "risk": mdm.DecisionQuestion(
                type="score",
                instructions="Risk level.",
                levels=["low", "med", "high"],
            ),
        }
        wire_response = {
            "answers": {
                "action": {
                    "choice": "stop",
                    "probabilities": {"stop": 0.90, "slow": 0.08, "proceed": 0.02},
                },
                "pedestrian": {
                    "noul": True,
                    "noul_prob": 0.96,
                    "probabilities": {"0": 0.04, "1": 0.96},
                },
                "risk": {
                    "score": 2,
                    "score_value": 1.88,
                    "probabilities": {"0": 0.02, "1": 0.08, "2": 0.90},
                },
            }
        }
        dists, evidence_map, trace = backend.parse_response(wire_response, questions)
        self.assertAlmostEqual(dists["action"]["stop"], 0.90, places=5)
        self.assertAlmostEqual(dists["pedestrian"]["yes"], 0.96, places=5)
        self.assertAlmostEqual(dists["pedestrian"]["no"], 0.04, places=5)
        self.assertAlmostEqual(dists["risk"]["high"], 0.90, places=5)
        self.assertEqual(trace["discrete_fallback_slots"], [])
        self.assertEqual(evidence_map, {})

    def test_gevstudio_parse_response_discrete_fallback_and_strict_mode(self) -> None:
        questions = {
            "action": mdm.DecisionQuestion(
                type="choice",
                instructions="Pick action.",
                criteria={"stop": "Stop", "proceed": "Proceed"},
            )
        }
        wire_discrete = {"answers": {"action": {"choice": "stop"}}}

        lenient = mdm.GevStudioSystemOneBackend(
            target="gemini-3.6-flash",
            allow_discrete_fallback=True,
            label_smoothing_eps=1e-3,
        )
        dists, _, trace = lenient.parse_response(wire_discrete, questions)
        self.assertGreater(dists["action"]["stop"], 0.998)
        self.assertEqual(trace["discrete_fallback_slots"], ["action"])

        strict = mdm.GevStudioSystemOneBackend(target="gemini-3.6-flash")
        self.assertFalse(strict.allow_discrete_fallback)
        with self.assertRaisesRegex(mdm.LogprobAlignmentError, r"empty probabilities|lacks continuous probabilities"):
            strict.parse_response(wire_discrete, questions)

        # Injected transport callback works without external shell tools; use_sso_client=True fails closed.
        called_urls: list[str] = []

        def fake_transport(url: str, payload: dict[str, object]) -> dict[str, object]:
            called_urls.append(url)
            self.assertIn("questions", payload)
            return {
                "answers": {
                    "action": {"probabilities": {"stop": 0.80, "proceed": 0.20}},
                }
            }

        custom_transport_backend = mdm.GevStudioSystemOneBackend(
            endpoint_url="https://gevstudio.example.com/api/gev/systemone",
            target="gemini-decision-1",
            transport=fake_transport,
        )
        t_dists = custom_transport_backend.score_questions(state={"x": 1}, questions=questions)
        self.assertAlmostEqual(t_dists["action"]["stop"], 0.80, places=6)
        self.assertEqual(len(called_urls), 1)

        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"use_sso_client=True"):
            mdm.GevStudioSystemOneBackend(use_sso_client=True)

    def test_gevstudio_cascade_orchestrator_end_to_end(self) -> None:
        backend = mdm.GevStudioSystemOneBackend(
            endpoint_url="https://gevstudio.example.com/api/gev/systemone",
            target="gemini-decision-1",
        )
        questions = {
            "sku": mdm.DecisionQuestion(
                type="choice",
                instructions="Select matching SKU.",
                criteria={"sku_a": "SKU A", "sku_b": "SKU B"},
                prior_logits={"sku_a": 1.2, "sku_b": -0.4},
            )
        }
        mock_wire = {
            "answers": {
                "sku": {
                    "choice": "sku_a",
                    "probabilities": {"sku_a": 0.88, "sku_b": 0.12},
                }
            }
        }
        with patch("multimodal_decision_models.backends._post_json_with_retry", return_value=mock_wire):
            orchestrator = mdm.CascadeDecisionOrchestrator(
                stage1_backend=backend,
                prior_weight=0.25,
                hesitation_threshold=0.30,
            )
            result = orchestrator.evaluate(state={"shelf": "bay_1"}, questions=questions)
        self.assertEqual(result.terminal_state, mdm.DecisionTerminalState.STAGE1_RESOLVED)
        self.assertEqual(result.slots["sku"].predicted, "sku_a")
        self.assertGreater(result.slots["sku"].probabilities["sku_a"], 0.88)

    def test_gevstudio_validation_errors(self) -> None:
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"http:// or https://"):
            mdm.GevStudioSystemOneBackend(endpoint_url="file:///etc/passwd")
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"label_smoothing_eps"):
            mdm.GevStudioSystemOneBackend(label_smoothing_eps=0.75)

    def test_gevstudio_v1_systemone_populates_images_alongside_media(self) -> None:
        """Verify /v1/systemone endpoints receive payload['images'] data URIs alongside media."""
        questions = {
            "sku": mdm.DecisionQuestion(
                type="choice",
                instructions="Identify matching shelf SKU.",
                criteria={"sku_a": "SKU A", "sku_b": "SKU B"},
            )
        }
        state_with_audio = {
            "bay": "aisle_2",
            "__media__": [{"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"}],
        }
        sample_images = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
            "/9j/4AAQSkZJRg==",
        ]

        dgem_backend = mdm.GevStudioSystemOneBackend(
            endpoint_url="https://dgem-gateway.example.com/v1/systemone?raw=1",
            target="gemini-decision-1",
        )
        dgem_payload = dgem_backend.build_payload(
            state=state_with_audio,
            questions=questions,
            images=sample_images,
        )
        self.assertEqual(len(dgem_payload["media"]), 3)
        self.assertEqual(
            dgem_payload["images"],
            [
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
                "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
            ],
        )

        studio_backend = mdm.GevStudioSystemOneBackend(
            endpoint_url="https://gevstudio.example.com/api/gev/systemone",
            target="gemini-decision-1",
        )
        studio_payload = studio_backend.build_payload(
            state=state_with_audio,
            questions=questions,
            images=sample_images,
        )
        self.assertEqual(len(studio_payload["media"]), 3)
        self.assertNotIn("images", studio_payload)

    def test_vertex_gemini_raw_base64_and_invalid_image_rejection(self) -> None:
        """Verify VertexGeminiBackend encodes raw base64 images and fails closed on URLs/malformed inputs."""
        vertex_backend = mdm.VertexGeminiBackend(
            project="demo-project-01",
            model="gemini-2.5-flash",
            location="us-central1",
        )
        questions = {
            "sku": mdm.DecisionQuestion(
                type="choice",
                instructions="Identify matching shelf SKU.",
                criteria={"sku_a": "SKU A", "sku_b": "SKU B"},
            )
        }
        payload = vertex_backend.build_generate_content_payload(
            state={"bay": "aisle_1"},
            questions=questions,
            images=[
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
                "/9j/4AAQSkZJRg==",
                "UklGRgAAAABXRUJQ",
            ],
        )
        user_parts = payload["contents"][0]["parts"]
        self.assertEqual(len(user_parts), 4)
        self.assertEqual(
            user_parts[0],
            {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg=="}},
        )
        self.assertEqual(
            user_parts[1],
            {"inlineData": {"mimeType": "image/jpeg", "data": "/9j/4AAQSkZJRg=="}},
        )
        self.assertEqual(
            user_parts[2],
            {"inlineData": {"mimeType": "image/webp", "data": "UklGRgAAAABXRUJQ"}},
        )
        self.assertIn("text", user_parts[3])

        invalid_image_inputs: tuple[object, ...] = (
            ["http://example.com/shelf.jpg"],
            ["https://example.com/shelf.png"],
            ["gs://bucket-name/shelf.png"],
            [""],
            ["   "],
            ["data:image/png,raw-not-base64"],
            ["data:image/png;base64,"],
            ["!!!not_valid_base64!!!"],
            "iVBORw0KGgoAAAANSUhEUg==",
        )
        for bad_images in invalid_image_inputs:
            with self.assertRaisesRegex(
                mdm.SchemaContractViolationError,
                r"must not be empty|images|base64|non-empty|URL|URI",
            ):
                vertex_backend.build_generate_content_payload(
                    state={"bay": "aisle_1"},
                    questions=questions,
                    images=bad_images,  # type: ignore[arg-type]
                )

        token_mints = {"count": 0}

        def _mint_token() -> str:
            token_mints["count"] += 1
            return "test-oauth-token"

        guarded_vertex = mdm.VertexGeminiBackend(
            project="demo-project-01",
            model="gemini-2.5-flash",
            location="us-central1",
            access_token_fn=_mint_token,
        )
        orchestrator = mdm.CascadeDecisionOrchestrator(
            stage1_backend=guarded_vertex,
            total_contract_mode=True,
        )
        quarantined = orchestrator.evaluate(
            state={"bay": "aisle_1"},
            questions=questions,
            images=["http://example.com/shelf.jpg"],
        )
        self.assertEqual(
            quarantined.terminal_state,
            mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION,
        )
        self.assertEqual(token_mints["count"], 0)

    def test_lemma1_top_m_entropy_invariance_and_small_shortlist_fast_exit(self) -> None:
        # 50/50 sibling confusion over K=100 classes dilutes global entropy to ln(2)/ln(100) ~ 0.1505
        sibling_100 = {f"c_{idx}": (0.5 if idx < 2 else 0.0) for idx in range(100)}
        h_global = mdm.normalized_entropy(sibling_100)
        h_top5 = mdm.effective_top_m_entropy(sibling_100, top_m=5)
        self.assertAlmostEqual(h_global, math.log(2.0) / math.log(100.0), places=9)
        self.assertAlmostEqual(h_top5, math.log(2.0) / math.log(5.0), places=9)
        self.assertGreater(h_top5, 0.43)

        # Fail-closed rejection of all-zero raw_probs in pre_gated_bayesian_poe_fusion
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"positive|mass|zero|sum"):
            mdm.pre_gated_bayesian_poe_fusion({"a": 0.0, "b": 0.0})

        # Stage-2A fast-exit gate on M=2 and M=3 shortlists
        ok2, win2 = mdm.stage2a_fast_exit_gate({"sku_1": 0.85, "sku_2": 0.15})
        self.assertTrue(ok2)
        self.assertEqual(win2, "sku_1")
        ok3, win3 = mdm.stage2a_fast_exit_gate({"sku_1": 0.80, "sku_2": 0.12, "sku_3": 0.08})
        self.assertTrue(ok3)
        self.assertEqual(win3, "sku_1")

    def test_proposition1_cascade_risk_and_proposition2_cut_polytope_certificate(self) -> None:
        bound = mdm.evaluate_two_stage_cascade_risk_bound(
            escalation_rate=0.20,
            fast_exit_selective_risk=0.02,
            escalated_stage1_error_rate=0.45,
            stage2_false_flip_rate=0.04,
            stage2_unrecovered_error_rate=0.15,
            alpha_stage1_fast_exit=0.05,
            alpha_stage2_conformal=0.10,
        )
        self.assertTrue(bound.strictly_improves_over_stage1)
        self.assertLess(bound.cascade_marginal_risk, bound.stage1_marginal_risk)
        self.assertAlmostEqual(bound.hybrid_conformal_miscoverage_ub, 0.06, places=9)

        cert_valid = mdm.cut_polytope_triangle_consistency_score(0.92, 0.90, 0.88)
        self.assertTrue(cert_valid.am_gm_boundary_certified)
        self.assertEqual(cert_valid.dominant_valid_vertex, (1, 1, 1))
        self.assertGreater(cert_valid.consistency_score, 0.70)

        cert_boundary = mdm.cut_polytope_triangle_consistency_score(2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0)
        self.assertFalse(cert_boundary.am_gm_boundary_certified)
        self.assertAlmostEqual(cert_boundary.consistency_score, 13.0 / 27.0, places=9)

        # Split-conformal PRM residual quantile calibration and tree search
        q_hat = mdm.calibrate_prm_conformal_residual_quantile(
            [0.85, 0.80, 0.75, 0.20, 0.15],
            [True, True, True, False, False],
            alpha=0.20,
        )
        self.assertGreaterEqual(q_hat, 0.15)
        self.assertLessEqual(q_hat, 0.25)
        tree_res = mdm.run_calibrated_prm_puct_search(
            [
                mdm.PRMBranchNode(
                    node_id="r1",
                    parent_id=None,
                    depth=1,
                    prior_prob=0.70,
                    rubric_probs={"0": 0.02, "1": 0.03, "2": 0.15, "3": 0.80},
                    is_terminal=True,
                    ground_truth_correct=True,
                ),
                mdm.PRMBranchNode(
                    node_id="r2",
                    parent_id=None,
                    depth=1,
                    prior_prob=0.30,
                    rubric_probs={"0": 0.70, "1": 0.20, "2": 0.05, "3": 0.05},
                    is_terminal=True,
                    ground_truth_correct=False,
                ),
            ],
            conformal_q_hat=q_hat,
        )
        self.assertEqual(tree_res.selected_leaf_id, "r1")
        self.assertTrue(tree_res.leaf_correct)

        # Proposition 3: Sinkhorn-Knopp de-aliased trie coupling and partitioned 1-pass zero-aliasing router
        wide_criteria = {f"dom{d}_intent{i}": f"Domain {d} action {i}" for d in range(4) for i in range(8)}
        wide_criteria["oos"] = "Out of scope"
        trie = mdm.build_prefix_taxonomy_trie(wide_criteria, catch_all_key="oos")
        coarse_p = {k: (0.70 if k == "dom0" else 0.10) for k in trie.coarse_criteria}
        fine_p = {k: (0.60 if k == "intent0" else 0.05) for k in trie.fine_criteria}
        sink_probs, sink_win = mdm.sinkhorn_dealiased_trie_decode(coarse_p, fine_p, trie, num_iters=8)
        self.assertEqual(sink_win, "dom0_intent0")
        self.assertAlmostEqual(sum(sink_probs.values()), 1.0, places=9)

        def mock_band_scorer(state, questions, images=None):
            out = {}
            for q_id, q in questions.items():
                keys = q.option_keys()
                if q_id.endswith("_coarse"):
                    out[q_id] = {k: (0.82 if k == "dom0" else 0.18 / max(1, len(keys) - 1)) for k in keys}
                elif "dom0_intent0" in keys:
                    out[q_id] = {k: (0.88 if k == "dom0_intent0" else 0.12 / max(1, len(keys) - 1)) for k in keys}
                else:
                    out[q_id] = {k: 1.0 / len(keys) for k in keys}
            return out

        part_router = mdm.HierarchicalDecisionRouter(
            backend=mdm.MockCallableBackend(name="mock_band", fn=mock_band_scorer),
            mode="mtp_partitioned_1pass",
        )
        part_res = part_router.route(
            state={"query": "run dom0 intent0"},
            slot_id="intent",
            question=mdm.DecisionQuestion(
                type="choice",
                instructions="Classify user intent.",
                criteria=wide_criteria,
            ),
            trie=trie,
        )
        self.assertEqual(part_res.predicted, "dom0_intent0")
        self.assertEqual(part_res.passes_used, 1)
        self.assertTrue(part_res.guard_applied.startswith("partitioned_1pass_"))

        # Proposition 4: Combined label-shift and covariate-shift weighted conformal prediction set
        calib_probs = [
            {"head": 0.90, "tail": 0.10},
            {"head": 0.85, "tail": 0.15},
            {"head": 0.80, "tail": 0.20},
            {"head": 0.65, "tail": 0.35},
            {"head": 0.55, "tail": 0.45},
        ]
        calib_labels = ["head", "head", "head", "tail", "tail"]
        ls_set = mdm.label_shift_weighted_conformal_prediction_set(
            calib_probs=calib_probs,
            calib_labels=calib_labels,
            class_likelihood_ratios={"head": 0.25, "tail": 4.0},
            test_probs={"head": 0.58, "tail": 0.42},
            alpha=0.10,
        )
        self.assertIn("tail", ls_set.prediction_set)

        # Verify exact coarse domain marginal preservation in mtp_partitioned_1pass
        dom0_sum = sum(v for k, v in part_res.probabilities.items() if k.startswith("dom0_"))
        self.assertAlmostEqual(dom0_sum, 0.82, places=6)

    def test_formal_propositions_v090_benchmark_pillars(self) -> None:
        typed_rows = prop_bench.load_typed_receipts(prop_bench.TYPED_RECEIPTS_PATH)
        mdm_rows = prop_bench.load_typed_receipts(prop_bench.MDM_BENCH_RECEIPTS_PATH)

        pillar1 = prop_bench.run_pillar1_top_m_entropy_experiment(typed_rows)
        self.assertEqual(pillar1["sibling_tie_recall_top_m"], 1.0)
        self.assertGreater(pillar1["top_m_entropy_accuracy"], pillar1["global_entropy_accuracy"])

        pillar2 = prop_bench.run_pillar2_cascade_risk_bound_experiment(typed_rows)
        self.assertTrue(pillar2["strictly_improves"])
        self.assertTrue(pillar2["coverage_guarantee_satisfied"])

        pillar3 = prop_bench.run_pillar3_polytope_and_sinkhorn_experiment(typed_rows)
        self.assertTrue(pillar3["facet_bound_verified_all_100"])
        self.assertEqual(pillar3["sinkhorn_dealiased_trie_accuracy"], 1.0)

        pillar4 = prop_bench.run_pillar4_combined_shift_conformal_experiment(mdm_rows)
        self.assertGreaterEqual(pillar4["combined_shift_tail_coverage"], pillar4["target_coverage"])
        self.assertGreater(pillar4["combined_shift_tail_coverage"], pillar4["unweighted_tail_coverage"])


if __name__ == "__main__":
    unittest.main()

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

"""Unit and integration tests for backend-agnostic Multimodal Decision Models."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from multimodal_decision_models import (
    CandidateMark,
    CascadeDecisionOrchestrator,
    DecisionQuestion,
    DgemSystemOneBackend,
    MockCallableBackend,
    OpenAILogprobBackend,
    PairwiseEdge,
    SwarmAgentProposal,
    VertexGeminiBackend,
    apply_layer5_deterministic_guards,
    arbitrate_swarm_locks,
    assign_box_to_grid_cell,
    build_set_of_marks_questions,
    check_transitive_triangles,
    csls_hubness_scores,
    filter_marks_by_grid_cell,
    normalized_entropy,
    parse_pack_multiplier,
    pre_gated_bayesian_poe_fusion,
    select_containing_or_nearest_mark,
    shortlist_wide_question,
    stage2a_fast_exit_gate,
    validate_pack_quantity_consistency,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class MultimodalDecisionModelsTest(unittest.TestCase):
    def test_pre_gated_prior_fusion_preserves_raw_hesitation_and_gate_modes(self) -> None:
        raw = {"sku_a": 0.36, "sku_b": 0.34, "sku_c": 0.30}
        strong_prior = {"sku_a": -2.0, "sku_b": 6.0, "sku_c": -2.0}

        fused, h_raw, should_escalate = pre_gated_bayesian_poe_fusion(
            raw_probs=raw,
            prior_logits=strong_prior,
            prior_weight=0.5,
            hesitation_threshold=0.16,
            gate_mode="always",
        )
        self.assertGreater(fused["sku_b"], 0.95)
        self.assertGreater(h_raw, 0.98)
        self.assertTrue(should_escalate)
        self.assertLess(normalized_entropy(fused), 0.20)

        # Confident gate skips fusion on high-entropy inputs
        conf_dist, h_conf, _ = pre_gated_bayesian_poe_fusion(
            raw_probs=raw,
            prior_logits=strong_prior,
            prior_weight=0.5,
            hesitation_threshold=0.16,
            gate_mode="confident",
        )
        self.assertAlmostEqual(conf_dist["sku_a"], 0.36, places=6)
        self.assertAlmostEqual(h_conf, h_raw, places=9)

        # Hesitant gate fuses on high-entropy inputs to break ties while leaving confident inputs untouched
        hes_dist, h_hes, _ = pre_gated_bayesian_poe_fusion(
            raw_probs=raw,
            prior_logits=strong_prior,
            prior_weight=0.5,
            hesitation_threshold=0.16,
            gate_mode="hesitant",
        )
        self.assertGreater(hes_dist["sku_b"], 0.95)
        self.assertAlmostEqual(h_hes, h_raw, places=9)

        confident_raw = {"sku_a": 0.97, "sku_b": 0.02, "sku_c": 0.01}
        hes_skip, h_low, esc_low = pre_gated_bayesian_poe_fusion(
            raw_probs=confident_raw,
            prior_logits=strong_prior,
            prior_weight=0.5,
            hesitation_threshold=0.16,
            gate_mode="hesitant",
        )
        self.assertFalse(esc_low)
        self.assertLess(h_low, 0.16)
        self.assertAlmostEqual(hes_skip["sku_a"], 0.97, places=6)

    def test_stage2a_fast_exit_and_layer5_guards(self) -> None:
        priors = {
            "sku_1": 0.65,
            "sku_2": 0.12,
            "sku_3": 0.09,
            "sku_4": 0.08,
            "sku_5": 0.06,
        }
        ok, winner = stage2a_fast_exit_gate(priors, sister_pack_collision=False)
        self.assertTrue(ok)
        self.assertEqual(winner, "sku_1")

        ok_coll, _ = stage2a_fast_exit_gate(priors, sister_pack_collision=True)
        self.assertFalse(ok_coll)

        chosen, guard = apply_layer5_deterministic_guards(
            chosen_id="sku_1",
            raw_probs={"sku_1": 0.55, "sku_2": 0.40, "none_of_the_above": 0.05},
            candidate_pack_sizes={"sku_1": 1, "sku_2": 6},
            decoded_pack_size=6,
        )
        self.assertEqual(chosen, "sku_2")
        self.assertEqual(guard, "layer5_pack_size_sister_switch")

        chosen_v, guard_v = apply_layer5_deterministic_guards(
            chosen_id="sku_2",
            raw_probs={"sku_1": 0.44, "sku_2": 0.46, "sku_3": 0.10},
            visual_energies={"sku_1": 0.88, "sku_2": 0.79},
            rank1_id="sku_1",
        )
        self.assertEqual(chosen_v, "sku_1")
        self.assertEqual(guard_v, "layer5_visual_energy_veto")

        chosen_c, guard_c = apply_layer5_deterministic_guards(
            chosen_id="none_of_the_above",
            raw_probs={"sku_1": 0.30, "none_of_the_above": 0.70},
            rank1_id="sku_1",
            rank1_visual_similarity=0.78,
        )
        self.assertEqual(chosen_c, "sku_1")
        self.assertEqual(guard_c, "layer5_competitor_similarity_guard")

    def test_wide_option_shortlist_and_csls(self) -> None:
        criteria = {f"intent_{i:03d}": f"Description for intent {i}" for i in range(60)}
        criteria["none_of_the_above"] = "Out of scope / none of the above"
        wide_q = DecisionQuestion(
            type="choice",
            instructions="Classify the customer utterance",
            criteria=criteria,
            order=list(criteria.keys()),
        )
        scores = {f"intent_{i:03d}": float(i) for i in range(60)}
        narrow_q = shortlist_wide_question(wide_q, scores, top_k=6)

        keys = narrow_q.option_keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], "intent_059")
        self.assertIn("none_of_the_above", keys)
        self.assertIn("distinguish from intent_058", narrow_q.criteria["intent_059"])

        q_vec = [1.0, 0.0]
        protos = {"hub_sku": [0.9, 0.43589], "tail_sku": [0.88, 0.47497]}
        r_t = {"hub_sku": 0.30, "tail_sku": 0.05}
        csls = csls_hubness_scores(q_vec, protos, r_t)
        self.assertGreater(csls["tail_sku"], csls["hub_sku"])

    def test_som_navigator_nested_containment_and_question_builder(self) -> None:
        marks = [
            CandidateMark(id="M1", label="Top navigation bar", box=(0.0, 0.0, 15.0, 100.0)),
            CandidateMark(id="M2", label="Menu icon", box=(3.854, 90.37, 10.417, 100.0)),
            CandidateMark(id="M3", label="Primary checkout CTA", box=(80.0, 20.0, 90.0, 80.0)),
        ]
        self.assertEqual(assign_box_to_grid_cell(marks[1].box), "top_right")
        sel, dist = select_containing_or_nearest_mark(7.0, 95.0, marks)
        self.assertIsNotNone(sel)
        assert sel is not None
        self.assertEqual(sel.id, "M2")
        self.assertEqual(dist, 0.0)

        top_right_marks = filter_marks_by_grid_cell("top_right", marks)
        self.assertEqual([m.id for m in top_right_marks], ["M2"])

        qs = build_set_of_marks_questions("Open the top-right menu", marks)
        self.assertEqual(
            set(qs.keys()),
            {"next_target_mark", "element_state", "loop_or_error_detected", "goal_progress_score"},
        )
        self.assertEqual(qs["next_target_mark"].option_keys(), ["M1", "M2", "M3", "none_visible"])

    def test_triangle_consistency_pack_verifier_and_swarm_arbiter(self) -> None:
        edges = [
            PairwiseEdge("sku_a", "sku_b", 0.93),
            PairwiseEdge("sku_b", "sku_c", 0.89),
            PairwiseEdge("sku_a", "sku_c", 0.08),
        ]
        contradictions = check_transitive_triangles(edges, high_thresh=0.80, low_thresh=0.25)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].weak_edge, ("sku_a", "sku_c"))
        self.assertTrue(math.isclose(contradictions[0].severity, 0.89 - 0.08, abs_tol=1e-6))

        self.assertEqual(parse_pack_multiplier("San Pellegrino 6x330ml Cans"), 6)
        self.assertEqual(parse_pack_multiplier("Coca-Cola Pack of 12"), 12)
        self.assertFalse(
            validate_pack_quantity_consistency(
                "San Pellegrino 6x330ml Cans",
                "San Pellegrino Single 330ml Can",
                "exact_duplicate",
                "identical_pack_and_volume",
            )
        )

        proposals = [
            SwarmAgentProposal("agent_1", ("pkg/engine.go",), confidence=0.95, hesitation=0.04),
            SwarmAgentProposal("agent_2", ("pkg/engine.go",), confidence=0.82, hesitation=0.08),
            SwarmAgentProposal("agent_3", ("docs/README.md",), confidence=0.90, hesitation=0.03),
        ]
        dec = arbitrate_swarm_locks(proposals, max_hesitation=0.16)
        self.assertTrue(dec.write_collision)
        self.assertEqual(dec.winning_agent, "agent_1")
        self.assertEqual(dec.agent_actions["agent_1"], "grant_lock")
        self.assertEqual(dec.agent_actions["agent_2"], "serialize_after_peer")
        self.assertEqual(dec.agent_actions["agent_3"], "grant_lock")

    def test_multi_backend_adapters_and_cascade_orchestrator(self) -> None:
        q = DecisionQuestion(
            type="choice",
            instructions="Select product match",
            criteria={
                "sku_1": "330ml Single",
                "sku_2": "6x330ml Multipack",
                "none_of_the_above": "Neither",
            },
            order=["sku_1", "sku_2", "none_of_the_above"],
        )
        _, letter_map = OpenAILogprobBackend.build_letter_prompt({"query": "6 pack"}, q)
        self.assertEqual(letter_map, {"A": "sku_1", "B": "sku_2", "C": "none_of_the_above"})
        dist = OpenAILogprobBackend.extract_letter_distribution(
            {"B": -0.05, "A": -3.2, "C": -4.5},
            letter_map,
        )
        self.assertGreater(dist["sku_2"], 0.94)

        schema = VertexGeminiBackend.build_evidence_first_response_schema({"match": q})
        self.assertEqual(schema["propertyOrdering"][0], "evidence_read")
        self.assertEqual(schema["propertyOrdering"][1], "match")

        dgem = DgemSystemOneBackend("http://localhost:8095")
        payload = dgem.build_payload({"query": "6 pack"}, {"match": q}, images=["base64_img"])
        self.assertIn("match", payload["questions"])
        self.assertEqual(payload["images"], ["base64_img"])

        stage1 = MockCallableBackend(
            "dgem_stage1",
            lambda state, qs, imgs: {
                qid: {"sku_1": 0.35, "sku_2": 0.35, "none_of_the_above": 0.30} for qid in qs
            },
        )
        stage2 = MockCallableBackend(
            "gemini_3_8_flash_stage2",
            lambda state, qs, imgs: {
                qid: {"sku_1": 0.02, "sku_2": 0.96, "none_of_the_above": 0.02} for qid in qs
            },
        )
        orchestrator = CascadeDecisionOrchestrator(
            stage1_backend=stage1,
            stage2_backend=stage2,
            hesitation_threshold=0.16,
            prior_weight=0.5,
            evidence_prompt="Identify the decisive visual feature before choosing.",
        )
        result = orchestrator.evaluate(
            state={"query": "6 pack"},
            questions={"match": q},
            prior_logits={"match": {"sku_1": 5.0, "sku_2": 0.0, "none_of_the_above": -2.0}},
        )
        self.assertEqual(result.evidence_slot_id, "evidence_feature")
        self.assertTrue(result.escalated)
        self.assertEqual(result.slots["match"].predicted, "sku_2")
        self.assertEqual(result.slots["match"].backend_name, "gemini_3_8_flash_stage2")

    def test_templates_and_architecture_docs_integrity(self) -> None:
        arch_doc = REPO_ROOT / "docs" / "architecture.md"
        self.assertTrue(arch_doc.exists(), f"missing {arch_doc}")

        template_files = [
            "templates/multimodal/contact_sheet_verify.json.tmpl",
            "templates/multimodal/ui_pathfinder_som.json.tmpl",
            "templates/multimodal/entity_resolution_dedup.json.tmpl",
            "templates/live_checklist_verify.json.tmpl",
            "templates/swarm_lock_arbiter.json.tmpl",
        ]
        for tf in template_files:
            p = REPO_ROOT / tf
            self.assertTrue(p.exists(), f"missing template {p}")
            self.assertIn("questions", p.read_text())


if __name__ == "__main__":
    unittest.main()

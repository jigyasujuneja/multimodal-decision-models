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
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from multimodal_decision_models import (
    CandidateMark,
    CascadeDecisionOrchestrator,
    DecisionQuestion,
    DgemSystemOneBackend,
    HierarchicalDecisionRouter,
    MockCallableBackend,
    OpenAILogprobBackend,
    PairwiseEdge,
    SwarmAgentProposal,
    TaxonomyLeaf,
    TaxonomyTrie,
    VertexGeminiBackend,
    apply_layer5_deterministic_guards,
    arbitrate_swarm_locks,
    assign_box_to_grid_cell,
    build_contrastive_criteria,
    build_prefix_taxonomy_trie,
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
    trie_constrained_joint_decode,
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

        conf_dist, h_conf, _ = pre_gated_bayesian_poe_fusion(
            raw_probs=raw,
            prior_logits=strong_prior,
            prior_weight=0.5,
            hesitation_threshold=0.16,
            gate_mode="confident",
        )
        self.assertAlmostEqual(conf_dist["sku_a"], 0.36, places=6)
        self.assertAlmostEqual(h_conf, h_raw, places=9)

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

    def test_wide_option_shortlist_sibling_unique_cues_and_csls(self) -> None:
        criteria = {
            f"intent_{i:03d}": f"Description for intent {i} unique_kw_{i}" for i in range(60)
        }
        criteria["none_of_the_above"] = "Out of scope / none of the above"
        wide_q = DecisionQuestion(
            type="choice",
            instructions="Classify the customer utterance",
            criteria=criteria,
            order=list(criteria.keys()),
        )
        scores = {f"intent_{i:03d}": float(i) for i in range(60)}

        # Default top_k=20 reserves 1 slot for catch-all -> 21 total options <= 26
        default_q = shortlist_wide_question(wide_q, scores)
        self.assertEqual(len(default_q.option_keys()), 21)
        self.assertEqual(default_q.option_keys()[0], "intent_059")
        self.assertEqual(default_q.option_keys()[-1], "none_of_the_above")
        self.assertIn("[distinguishing:", default_q.criteria["intent_059"])
        self.assertNotIn("[distinguishing:", default_q.criteria["none_of_the_above"])

        cues = build_contrastive_criteria(
            {
                "card_arrival": "Track physical card delivery",
                "card_linking": "Link existing card to app",
                "none_of_the_above": "Neither",
            },
            ["card_arrival", "card_linking", "none_of_the_above"],
        )
        self.assertIn("[distinguishing: arrival, track]", cues["card_arrival"])
        self.assertIn("[distinguishing: linking, link]", cues["card_linking"])

        q_vec = [1.0, 0.0]
        protos = {"hub_sku": [0.9, 0.43589], "tail_sku": [0.88, 0.47497]}
        r_t = {"hub_sku": 0.30, "tail_sku": 0.05}
        csls = csls_hubness_scores(q_vec, protos, r_t)
        self.assertGreater(csls["tail_sku"], csls["hub_sku"])

    def test_hierarchical_taxonomy_trie_and_4_arm_router(self) -> None:
        # 6 domains x 8 intents + oos = 49 options (> 26)
        domains = ["banking", "travel", "iot", "calendar", "music", "weather"]
        roles = ["check", "create", "update", "cancel", "list", "share", "alert", "help"]
        crit: dict[str, str] = {}
        for d in domains:
            for r in roles:
                crit[f"{d}_{r}"] = f"{r.capitalize()} item in {d} service"
        crit["oos"] = "Out-of-scope request outside all supported domains"

        wide_q = DecisionQuestion(
            type="choice",
            instructions="Route user utterance to intent",
            criteria=crit,
            order=list(crit.keys()),
        )
        trie = build_prefix_taxonomy_trie(crit)
        self.assertLessEqual(len(trie.coarse_criteria), 26)
        self.assertLessEqual(len(trie.fine_criteria), 26)
        self.assertEqual(trie.catch_all_key, "oos")

        # Sparse trie where (dom_a, role_2) is NOT a valid leaf:
        sparse_trie = TaxonomyTrie(
            coarse_criteria={"dom_a": "Domain A", "dom_b": "Domain B", "oos": "OOS"},
            fine_criteria={"role_1": "Role 1", "role_2": "Role 2", "oos": "OOS"},
            leaves={
                "dom_a_role_1": TaxonomyLeaf("dom_a_role_1", "dom_a", "role_1", "A1"),
                "dom_b_role_2": TaxonomyLeaf("dom_b_role_2", "dom_b", "role_2", "B2"),
                "oos": TaxonomyLeaf("oos", "oos", "oos", "OOS"),
            },
            catch_all_key="oos",
        )
        sparse_q = DecisionQuestion(
            type="choice",
            instructions="Classify",
            criteria={f"extra_{i}": f"Extra {i}" for i in range(28)}
            | {"dom_a_role_1": "A1", "dom_b_role_2": "B2", "oos": "OOS"},
        )

        # Marginal argmax picks (dom_a=0.52, role_2=0.75) -> invalid tuple in mtp_naive,
        # whereas mtp_trie picks valid dom_b_role_2 (0.43 * 0.75 = 0.3225 > 0.52 * 0.20 = 0.104).
        def mock_fn(state, qs, imgs):
            out = {}
            for qid, q_obj in qs.items():
                if qid.endswith("_coarse"):
                    if state.get("confident"):
                        out[qid] = {"dom_a": 0.96, "dom_b": 0.03, "oos": 0.01}
                    else:
                        out[qid] = {"dom_a": 0.52, "dom_b": 0.43, "oos": 0.05}
                elif qid.endswith("_fine"):
                    if state.get("confident"):
                        out[qid] = {"role_1": 0.94, "role_2": 0.05, "oos": 0.01}
                    else:
                        out[qid] = {"role_1": 0.20, "role_2": 0.75, "oos": 0.05}
                else:
                    # Pass-2 leaf read over top-2 coarse domains
                    keys = q_obj.option_keys()
                    out[qid] = {k: (0.88 if k == "dom_b_role_2" else 0.12 / max(1, len(keys) - 1)) for k in keys}
            return out

        backend = MockCallableBackend("mock_s1", mock_fn)

        r_naive = HierarchicalDecisionRouter(backend, mode="mtp_naive").route(
            {"confident": False}, "intent", sparse_q, trie=sparse_trie
        )
        self.assertTrue(r_naive.predicted.startswith("__invalid_tuple__:"))
        self.assertEqual(r_naive.passes_used, 1)

        r_trie = HierarchicalDecisionRouter(backend, mode="mtp_trie").route(
            {"confident": False}, "intent", sparse_q, trie=sparse_trie
        )
        self.assertEqual(r_trie.predicted, "dom_b_role_2")
        self.assertEqual(r_trie.passes_used, 1)

        # Hesitation hybrid: 1 pass when confident, 2 passes when hesitant
        hybrid = HierarchicalDecisionRouter(backend, mode="hesitation_hybrid", hesitation_threshold=0.25)
        r_fast = hybrid.route({"confident": True}, "intent", sparse_q, trie=sparse_trie)
        self.assertEqual(r_fast.predicted, "dom_a_role_1")
        self.assertEqual(r_fast.passes_used, 1)
        self.assertFalse(r_fast.escalated)

        r_hes = hybrid.route({"confident": False}, "intent", sparse_q, trie=sparse_trie)
        self.assertEqual(r_hes.predicted, "dom_b_role_2")
        self.assertEqual(r_hes.passes_used, 2)
        self.assertTrue(r_hes.escalated)

        # Full 49-option prefix trie joint decode check
        p_joint, win_joint = trie_constrained_joint_decode(
            {"music": 0.85, "iot": 0.10, "oos": 0.05},
            {k: (0.80 if "member_01" in k or k == "check" else 0.02) for k in trie.fine_criteria},
            trie,
        )
        self.assertEqual(win_joint, "music_check")
        self.assertGreater(p_joint["music_check"], 0.50)
        self.assertEqual(len(wide_q.option_keys()), 49)

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

    def test_real_systemone_http_contract_vertex_gemini_and_inferred_escalation(self) -> None:
        # Reject file:// URLs to prevent SSRF / local file reads
        with self.assertRaises(ValueError):
            DgemSystemOneBackend("file:///etc/passwd")
        with self.assertRaises(ValueError):
            VertexGeminiBackend(project="../bad_project")

        q = DecisionQuestion(
            type="choice",
            instructions="Select product match",
            criteria={
                "sku_1": "330ml Single",
                "sku_2": "6x330ml Multipack",
                "none_of_the_above": "Neither",
            },
            order=["sku_1", "sku_2", "none_of_the_above"],
            prior_logits={"sku_1": 0.5, "sku_2": 1.5, "none_of_the_above": -1.0},
            evidence_prompt="Is the multipack wrapper directly visible or inferred?",
            evidence_criteria={"printed": "Directly visible", "inferred": "Guessed or occluded"},
        )
        score_q = DecisionQuestion(
            type="score",
            instructions="Rate match quality",
            levels=["poor", "fair", "strong"],
        )

        captured_requests: list[dict] = []

        class MockSystemOneHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                req_json = json.loads(self.rfile.read(length).decode("utf-8"))
                captured_requests.append(req_json)
                resp = {
                    "answers": {
                        "match": {
                            "type": "choice",
                            "choice": "sku_1",
                            # Confident raw probabilities (h_raw < 0.16), YET evidence="inferred"!
                            "probabilities": {"sku_1": 0.97, "sku_2": 0.02, "none_of_the_above": 0.01},
                            "evidence": "inferred",
                        },
                        "quality": {
                            "type": "score",
                            "score": 1.8,
                            "legend": {"0": "poor", "1": "fair", "2": "strong"},
                            "probabilities": {"0": 0.05, "1": 0.10, "2": 0.85},
                        },
                    },
                    "evaluation_trace": {"forward_passes": 1},
                }
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        server = HTTPServer(("127.0.0.1", 0), MockSystemOneHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            dgem = DgemSystemOneBackend(f"http://127.0.0.1:{port}")
            dists = dgem.score_questions(
                {"query": "6 pack"},
                {"match": q, "quality": score_q},
                images=["data:image/png;base64,AAAA"],
            )
            self.assertEqual(len(captured_requests), 1)
            sent = captured_requests[0]
            self.assertEqual(sent["state"], {"query": "6 pack"})
            self.assertNotIn("data", sent)
            self.assertIn("prior_logits", sent["questions"]["match"])
            self.assertEqual(
                sent["questions"]["match"]["evidence_prompt"],
                "Is the multipack wrapper directly visible or inferred?",
            )
            self.assertAlmostEqual(dists["quality"]["strong"], 0.85, places=6)
            self.assertEqual(dgem.last_evidence["match"], "inferred")
            self.assertEqual(dgem.last_trace["forward_passes"], 1)

            # Cascade with escalate_on_inferred_evidence=True escalates "match" despite low h_raw!
            stage2 = MockCallableBackend(
                "gemini_3_8_flash_stage2",
                lambda state, qs, imgs: {
                    qid: {"sku_1": 0.02, "sku_2": 0.96, "none_of_the_above": 0.02} for qid in qs
                },
            )
            orchestrator = CascadeDecisionOrchestrator(
                stage1_backend=dgem,
                stage2_backend=stage2,
                hesitation_threshold=0.16,
                prior_weight=0.35,
                escalate_on_inferred_evidence=True,
            )
            res = orchestrator.evaluate({"query": "6 pack"}, {"match": q})
            self.assertTrue(res.escalated)
            self.assertIn("match:evidence=inferred", res.escalation_reasons)
            self.assertEqual(res.slots["match"].predicted, "sku_2")
            self.assertEqual(res.slots["match"].backend_name, "gemini_3_8_flash_stage2")
        finally:
            server.shutdown()
            server.server_close()

        # Verify VertexGeminiBackend structured logprob response parser
        gemini_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps({"evidence_read": "printed 6x330ml badge", "match": "sku_2"})}
                        ]
                    },
                    "logprobsResult": {
                        "chosenCandidates": [
                            {"token": "{", "logProbability": -0.001},
                            {"token": "sku_2", "logProbability": -0.04},
                        ],
                        "topCandidates": [
                            {"candidates": [{"token": "{", "logProbability": -0.001}]},
                            {
                                "candidates": [
                                    {"token": "sku_2", "logProbability": -0.04},
                                    {"token": "sku_1", "logProbability": -3.40},
                                    {"token": "none_of_the_above", "logProbability": -5.10},
                                ]
                            },
                        ],
                    },
                }
            ]
        }
        v_dists, ev_read = VertexGeminiBackend.parse_generate_content_response(gemini_resp, {"match": q})
        self.assertEqual(ev_read, "printed 6x330ml badge")
        self.assertGreater(v_dists["match"]["sku_2"], 0.95)

        _, letter_map = OpenAILogprobBackend.build_letter_prompt({"query": "6 pack"}, q)
        dist = OpenAILogprobBackend.extract_letter_distribution(
            {"B": -0.05, "A": -3.2, "C": -4.5},
            letter_map,
        )
        self.assertGreater(dist["sku_2"], 0.94)

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

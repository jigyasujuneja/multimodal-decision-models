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

"""Unit and Integration Tests for v0.5.0 L8 GDM Research Upgrades and MDM-Bench."""

from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path

import multimodal_decision_models as mdm
from benchmarks import run_mdm_bench
from examples import how_i_ai_workflows

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestV050GdmUpgradesAndMdmBench(unittest.TestCase):
    """Comprehensive verification of Mondrian/Weighted/Joint Conformal, PriDe, LTT, Submodular MCKP, Polytope Projection, and MDM-Bench."""

    def test_mondrian_and_weighted_conformal_restore_rare_tail_coverage(self) -> None:
        calib_probs = []
        calib_labels = []
        calib_groups = []
        calib_weights = []
        for i in range(200):
            if i < 185:
                calib_probs.append({"head_opt": 0.90, "tail_opt": 0.07, "none_of_the_above": 0.03})
                calib_labels.append("head_opt")
                calib_groups.append("head")
                calib_weights.append(0.20)
            else:
                # Model under-predicts rare tail class (7.5% base rate < alpha=0.10)
                calib_probs.append({"head_opt": 0.64, "tail_opt": 0.30, "none_of_the_above": 0.06})
                calib_labels.append("tail_opt")
                calib_groups.append("tail")
                calib_weights.append(6.50)

        marg_cal = mdm.calibrate_conformal_quantile(calib_probs, calib_labels, alpha=0.10, method="raps")
        mond_cal = mdm.calibrate_mondrian_conformal_quantiles(
            calib_probs, calib_labels, calib_groups=calib_groups, alpha=0.10, method="raps"
        )
        self.assertGreater(mond_cal.group_q_hats["tail"], mond_cal.group_q_hats["head"])

        test_tail_prob = {"head_opt": 0.62, "tail_opt": 0.32, "none_of_the_above": 0.06}
        marg_set = mdm.conformal_prediction_set(test_tail_prob, marg_cal.q_hat, method="raps")
        mond_set = mdm.mondrian_conformal_prediction_set(test_tail_prob, mond_cal, group="tail")
        class_mond = mdm.calibrate_mondrian_conformal_quantiles(
            calib_probs, calib_labels, calib_groups=None, alpha=0.10, method="raps"
        )
        class_set = mdm.mondrian_conformal_prediction_set(test_tail_prob, class_mond, group=None)
        weighted_set = mdm.weighted_conformal_prediction_set(
            calib_probs,
            calib_labels,
            calib_weights,
            test_tail_prob,
            test_weight=5.50,
            alpha=0.10,
            method="raps",
        )

        self.assertNotIn("tail_opt", marg_set.prediction_set)
        self.assertIn("tail_opt", mond_set.prediction_set)
        self.assertIn("tail_opt", class_set.prediction_set)
        self.assertIn("tail_opt", weighted_set.prediction_set)

        # Contract validation checks
        with self.assertRaises(ValueError):
            mdm.calibrate_mondrian_conformal_quantiles(
                calib_probs, calib_labels, calib_groups=["only_one"]
            )
        with self.assertRaises(ValueError):
            mdm.weighted_conformal_prediction_set(
                calib_probs, calib_labels, calib_weights, test_tail_prob, test_weight=-1.0
            )

    def test_multislot_joint_conformal_controls_simultaneous_fwer(self) -> None:
        rows = run_mdm_bench.load_mdm_bench_rows(
            REPO_ROOT / "benchmarks" / "data" / "mdm_bench_8track_receipts.jsonl"
        )
        u2 = run_mdm_bench.run_upgrade2_multislot_joint_conformal(rows, family_alpha=0.10, num_folds=5)
        self.assertLess(u2["independent_uncorrected_raps"]["simultaneous_coverage"], 0.75)
        self.assertGreaterEqual(u2["joint_max_score_conformal"]["simultaneous_coverage"], 0.89)
        self.assertGreaterEqual(u2["joint_bonferroni_conformal"]["simultaneous_coverage"], 0.89)

        with self.assertRaises(ValueError):
            mdm.calibrate_multislot_joint_conformal([], [], alpha=0.10)

    def test_pride_cyclic_permutation_debiasing_and_stability_metrics(self) -> None:
        options = ["opt_a", "opt_b", "opt_c"]
        # True semantic preference: opt_b > opt_a > opt_c, confounded by strong slot-0 bias
        shifts = [
            {"opt_a": 0.54, "opt_b": 0.38, "opt_c": 0.08},
            {"opt_a": 0.16, "opt_b": 0.76, "opt_c": 0.08},
            {"opt_a": 0.26, "opt_b": 0.48, "opt_c": 0.26},
        ]
        debiased, pos_prior = mdm.debiased_cyclic_permutation_probs(shifts, options)
        self.assertAlmostEqual(sum(debiased.values()), 1.0, places=9)
        self.assertAlmostEqual(sum(pos_prior.values()), 1.0, places=9)
        self.assertEqual(max(debiased.items(), key=lambda kv: kv[1])[0], "opt_b")
        self.assertGreater(pos_prior[0], pos_prior[1])

        stab = mdm.option_permutation_stability([shifts[0]], [shifts[1]])
        self.assertEqual(stab.n_items, 1)
        self.assertEqual(stab.top1_flip_rate, 1.0)
        self.assertGreater(stab.mean_js_divergence, 0.0)
        self.assertGreater(stab.mean_tv_distance, 0.0)

        id_stab = mdm.option_permutation_stability([debiased], [debiased])
        self.assertEqual(id_stab.top1_flip_rate, 0.0)
        self.assertAlmostEqual(id_stab.mean_js_divergence, 0.0, places=12)
        self.assertAlmostEqual(id_stab.mean_tv_distance, 0.0, places=12)

    def test_ltt_joint_funnel_gate_calibration_satisfies_ucb_bound(self) -> None:
        rows = run_mdm_bench.load_mdm_bench_rows(
            REPO_ROOT / "benchmarks" / "data" / "mdm_bench_8track_receipts.jsonl"
        )
        cal_raw = [r["models"]["DecisionGemma-2B"]["probabilities"] for r in rows[:250]]
        cal_priors = [r["prior_logits"] for r in rows[:250]]
        cal_golds = [str(r["gold"]) for r in rows[:250]]

        ltt = mdm.calibrate_ltt_funnel_gates(
            cal_raw,
            cal_priors,
            cal_golds,
            target_risk_bound=0.15,
            delta=0.05,
        )
        self.assertTrue(ltt.satisfied_bound)
        self.assertLessEqual(ltt.ucb_risk, 0.15 + 1e-12)
        self.assertGreater(ltt.non_escalated_coverage, 0.50)
        self.assertGreater(ltt.num_valid_candidates, 0)

    def test_submodular_context_pruning_suppresses_duplicate_stack_traces(self) -> None:
        blocks = [
            mdm.ContextBlock("pinned_sys", 100, 50, "system rules", "sys stub", pinned=True),
            mdm.ContextBlock(
                "dup_1",
                100,
                60,
                "Traceback File worker.py line 52 RuntimeError CUDA out of memory in attention block",
                "oom 1",
            ),
            mdm.ContextBlock(
                "dup_2",
                100,
                60,
                "Traceback File worker.py line 52 RuntimeError CUDA out of memory in attention block retry",
                "oom 2",
            ),
            mdm.ContextBlock(
                "root_cause_cfg",
                100,
                60,
                "config batch_size increased from 16 to 256 in commit 7c12e9a",
                "batch_size=256",
            ),
        ]
        probs = {
            "pinned_sys": {"full": 0.10, "stub": 0.85, "drop": 0.05},
            "dup_1": {"full": 0.92, "stub": 0.04, "drop": 0.04},
            "dup_2": {"full": 0.90, "stub": 0.05, "drop": 0.05},
            "root_cause_cfg": {"full": 0.80, "stub": 0.02, "drop": 0.18},
        }
        # Budget 250 allows pinned_sys (50 stub) + two 100-token full blocks:
        # Additive MCKP picks dup_1=full and dup_2=full, leaving 0 tokens for root_cause_cfg!
        add_res = mdm.prune_agent_context_blocks(blocks, probs, 250, redundancy_penalty_weight=0.0)
        sub_res = mdm.prune_agent_context_blocks(blocks, probs, 250, redundancy_penalty_weight=0.75)

        self.assertEqual(add_res.decisions["dup_1"], "full")
        self.assertEqual(add_res.decisions["dup_2"], "full")
        self.assertEqual(add_res.decisions["root_cause_cfg"], "drop")
        self.assertEqual(sub_res.decisions["root_cause_cfg"], "full")
        self.assertLessEqual(sub_res.tokens_after, 250)

        with self.assertRaises(ValueError):
            mdm.prune_agent_context_blocks(blocks, probs, 250, redundancy_penalty_weight=1.5)

    def test_transitive_polytope_projection_eliminates_all_3cycle_violations(self) -> None:
        edges = [
            mdm.PairwiseEdge("n1", "n2", 0.93),
            mdm.PairwiseEdge("n2", "n3", 0.91),
            mdm.PairwiseEdge("n1", "n3", 0.18),  # violated triangle!
            mdm.PairwiseEdge("n4", "n5", 0.88),
            mdm.PairwiseEdge("n1", "n4", 0.12),
        ]
        proj = mdm.project_triangles_to_transitive_consistency(edges, threshold=0.50)
        self.assertEqual(proj.num_violated_triangles_before, 1)
        self.assertEqual(proj.num_violated_triangles_after, 0)
        self.assertEqual(mdm.check_transitive_triangles(proj.projected_edges), [])
        self.assertIn(("n1", "n2", "n3"), proj.clusters)
        self.assertIn(("n4", "n5"), proj.clusters)

    def test_mdm_bench_deterministic_reexecution_and_no_synthetic_rng_in_runner(self) -> None:
        runner_src = (REPO_ROOT / "benchmarks" / "run_mdm_bench.py").read_text(encoding="utf-8")
        tree = ast.parse(runner_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name, "random")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "random")

        committed_json = (REPO_ROOT / "benchmarks" / "results" / "mdm_bench_results.json").read_text(
            encoding="utf-8"
        )
        committed_md = (REPO_ROOT / "benchmarks" / "results" / "mdm_bench_report.md").read_text(
            encoding="utf-8"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "mdm_bench_results.json"
            tmp_md = Path(tmpdir) / "mdm_bench_report.md"
            rc = run_mdm_bench.main(
                [
                    "--receipts",
                    str(REPO_ROOT / "benchmarks" / "data" / "mdm_bench_8track_receipts.jsonl"),
                    "--json-out",
                    str(tmp_json),
                    "--md-out",
                    str(tmp_md),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(tmp_json.read_text(encoding="utf-8"), committed_json)
            self.assertEqual(tmp_md.read_text(encoding="utf-8"), committed_md)

        parsed = json.loads(committed_json)
        self.assertEqual(parsed["n_total_items"], 400)
        self.assertEqual(parsed["n_tracks"], 8)
        for row_line in (
            (REPO_ROOT / "benchmarks" / "data" / "mdm_bench_8track_receipts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            row = json.loads(row_line)
            for m_data in row["models"].values():
                p_sum = sum(m_data["probabilities"].values())
                self.assertTrue(math.isclose(p_sum, 1.0, abs_tol=1e-5))

    def test_how_i_ai_workflows_cookbook_runs_cleanly(self) -> None:
        receipts = how_i_ai_workflows.run_all_workflows()
        self.assertEqual(len(receipts), 8)
        self.assertEqual(receipts[0]["routed_tool"], "db.query")
        self.assertTrue(receipts[1]["injection_neutralized"])
        self.assertTrue(receipts[1]["boundary_notice_applied"])
        self.assertEqual(receipts[2]["snapped_mark_id"], "btn_confirm_left")
        self.assertEqual(receipts[3]["final_sku"], "sku_006")
        self.assertEqual(receipts[4]["decisions"]["git_diff_root_cause"], "full")
        self.assertEqual(receipts[5]["violated_triangles_after"], 0)
        self.assertEqual(receipts[6]["debiased_winner"], "candidate_beta")
        self.assertEqual(receipts[7]["joint_sets"]["user_intent_safety"], ["benign"])


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for v0.6.0 Real-Data Benchmarks, Universal 256-Code Bijection, Thinking-Tag Guard, and Eq. (12) Distillation."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks import run_real_public_benchmarks  # noqa: E402
from multimodal_decision_models import (  # noqa: E402
    HARNESS_DECISION_POINTS,
    INTEGRATION_MODES,
    MAX_EXTENDED_OPTIONS_PER_SLOT,
    UNIVERSAL_256_DECISION_CODES,
    DecisionQuestion,
    LogprobAlignmentError,
    OpenAILogprobBackend,
    SchemaContractViolationError,
    acceptable_set_accuracy,
    build_extended_255_code_bijection,
    evaluate_trigger_guard_utility,
    fit_trajectory_delegation_threshold,
    holm_bonferroni_adjust,
    reasoning_to_readout_distillation_loss,
    selective_teacher_brier_sft_loss,
    tango_paired_non_inferiority_test,
)


class TestV060RealPublicBenchmarksAndArxiv2610(unittest.TestCase):
    """Verify v0.6.0 real-data benchmark reproducibility, extended 255-code bijection, and Eq. (12) distillation."""

    def test_universal_256_codes_and_extended_bijection(self) -> None:
        self.assertEqual(len(UNIVERSAL_256_DECISION_CODES), 256)
        self.assertEqual(len(set(UNIVERSAL_256_DECISION_CODES)), 256)
        self.assertEqual(MAX_EXTENDED_OPTIONS_PER_SLOT, 256)

        # Ensure known BPE-split pairs are absent from UNIVERSAL_256_DECISION_CODES
        for split_pair in ("BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "FJ", "FQ", "FV", "FZ", "GJ", "GQ", "GX", "GZ"):
            self.assertNotIn(split_pair, UNIVERSAL_256_DECISION_CODES)

        opts_255 = [f"sku_{i:03d}" for i in range(255)]
        bij = build_extended_255_code_bijection(opts_255)
        self.assertEqual(len(bij), 255)
        self.assertEqual(bij["A"], "sku_000")
        self.assertEqual(bij["Z"], "sku_025")
        self.assertEqual(bij["AA"], "sku_026")

        # Verify OpenAILogprobBackend.build_letter_prompt supports allow_extended_255=True up to K=255
        q_wide = DecisionQuestion(
            type="choice",
            instructions="Select the matching entity.",
            criteria={k: f"Description for {k}" for k in opts_255},
        )
        prompt, code_map = OpenAILogprobBackend.build_letter_prompt(
            {"query": "test"}, q_wide, allow_extended_255=True
        )
        self.assertEqual(len(code_map), 255)
        self.assertIn("Reply with the option code only.", prompt)

        with self.assertRaises(SchemaContractViolationError):
            build_extended_255_code_bijection([f"x_{i}" for i in range(257)])

    def test_unstripped_think_tag_leakage_raises_logprob_alignment_error(self) -> None:
        # When enable_thinking=False is omitted on a hybrid model, <think> dominates top_logprobs
        leaked = {"<think>": -0.05, "A": -3.50, "B": -4.20}
        with self.assertRaises(LogprobAlignmentError):
            OpenAILogprobBackend.extract_letter_distribution(leaked, {"A": "no", "B": "yes"})

        # Normal clean logprobs succeed
        clean = {"A": -0.15, "B": -1.95}
        dist = OpenAILogprobBackend.extract_letter_distribution(clean, {"A": "no", "B": "yes"})
        self.assertGreater(dist["no"], 0.80)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)

    def test_reasoning_to_readout_distillation_loss_eq12_and_ordinal_penalty(self) -> None:
        teacher = {"0": 0.05, "1": 0.15, "2": 0.60, "3": 0.20}
        student = {"0": 0.40, "1": 0.30, "2": 0.20, "3": 0.10}
        anchor = {"0": 0.10, "1": 0.20, "2": 0.50, "3": 0.20}

        res_score = reasoning_to_readout_distillation_loss(
            teacher_probs=teacher,
            student_probs=student,
            base_anchor_probs=anchor,
            primitive_type="score",
            task_weight=1.0,
            ordinal_lambda=0.50,
            kl_anchor_kappa=0.25,
            teacher_temperature=1.2,
            anchor_temperature=0.8,
        )
        self.assertGreater(res_score.cross_entropy_loss, 0.0)
        self.assertGreater(res_score.ordinal_score_loss, 0.0)
        self.assertGreater(res_score.kl_anchor_loss, 0.0)
        self.assertAlmostEqual(sum(res_score.distilled_posterior.values()), 1.0, places=6)
        self.assertEqual(
            max(res_score.distilled_posterior, key=lambda k: res_score.distilled_posterior[k]),
            "2",
        )

        # Choice primitive has zero ordinal_score_loss term in total_loss
        res_choice = reasoning_to_readout_distillation_loss(
            teacher_probs=teacher,
            student_probs=student,
            base_anchor_probs=anchor,
            primitive_type="choice",
            task_weight=1.0,
            ordinal_lambda=0.50,
            kl_anchor_kappa=0.25,
        )
        self.assertAlmostEqual(
            res_choice.total_loss,
            res_choice.cross_entropy_loss + 0.25 * res_choice.kl_anchor_loss,
            places=8,
        )

    def test_real_public_benchmark_deterministic_reexecution_and_no_rng(self) -> None:
        runner_path = REPO_ROOT / "benchmarks" / "run_real_public_benchmarks.py"
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, ("random", "numpy"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], ("random", "numpy"))

        json_path = REPO_ROOT / "benchmarks" / "results" / "real_public_bench_results.json"
        committed = json.loads(json_path.read_text(encoding="utf-8"))

        typed_rows = run_real_public_benchmarks.load_jsonl(
            REPO_ROOT / "benchmarks" / "data" / "real_public_decisionbench_typed_receipts.jsonl"
        )
        distill_rows = run_real_public_benchmarks.load_jsonl(
            REPO_ROOT / "benchmarks" / "data" / "real_arxiv2610_distillation_receipts.jsonl"
        )
        adb_rows = run_real_public_benchmarks.load_jsonl(
            REPO_ROOT / "benchmarks" / "data" / "real_agentdecisionbench_receipts.jsonl"
        )
        self.assertEqual(len(typed_rows), 500)
        self.assertEqual(len(distill_rows), 300)
        self.assertEqual(len(adb_rows), 600)

        recomputed_s1 = run_real_public_benchmarks.run_study1_real_7model_benchmark(typed_rows)
        recomputed_exp_a = run_real_public_benchmarks.run_experiment_a_extended_255_bijection_and_think_guard()
        recomputed_exp_b = run_real_public_benchmarks.run_experiment_b_reasoning_to_readout_distillation(distill_rows)
        recomputed_exp_c = run_real_public_benchmarks.run_experiment_c_jev_agent_harness_and_agentdecisionbench(adb_rows)

        self.assertEqual(recomputed_s1, committed["study1_real_7model_benchmark"])
        self.assertEqual(recomputed_exp_a, committed["experiment_a_extended_255_bijection"])
        self.assertEqual(recomputed_exp_b, committed["experiment_b_reasoning_to_readout_distillation"])
        self.assertEqual(recomputed_exp_c, committed["experiment_c_jev_agent_harness_and_agentdecisionbench"])

        # Verify key empirical findings on real data
        ms = recomputed_s1["multislot_joint_conformal_5slot"]
        self.assertLess(ms["uncorrected_independent_coverage"], 0.70)
        self.assertGreaterEqual(ms["joint_max_score_coverage"], 0.90)

        arms = recomputed_exp_b["arms"]
        self.assertGreater(
            arms["4_eq12_distilled_posterior"]["accuracy"],
            arms["3_qwen27b_clean_nothink_teacher"]["accuracy"],
        )
        self.assertLess(
            arms["4_eq12_distilled_posterior"]["brier"],
            arms["3_qwen27b_clean_nothink_teacher"]["brier"],
        )

        # Verify arXiv:2610.21412v1 findings on real AgentDecisionBench
        self.assertLess(
            recomputed_exp_c["deferral_shift_eval"]["e14a"]["action_stakes_shift_accepted_irrev_errors"],
            recomputed_exp_c["deferral_shift_eval"]["e14a"]["prop1_shift_accepted_irrev_errors"],
        )
        self.assertEqual(
            recomputed_exp_c["selective_teacher_brier_sft"]["selective_eq1_corrupted_targets"],
            0,
        )

    def test_arxiv_2610_21412_delegation_guard_selective_sft_and_tango(self) -> None:
        self.assertEqual(INTEGRATION_MODES, ("replace", "cascade", "assist", "guard", "trigger"))
        self.assertEqual(
            HARNESS_DECISION_POINTS,
            ("tool_select", "tool_args", "action", "memory_write", "memory_read", "reflection", "stop", "routing"),
        )

        # 1. Proposition 1 + Action-Stakes Trajectory Delegation
        cal_conf = [0.99, 0.97, 0.96, 0.92, 0.85, 0.70]
        cal_corr = [True, True, True, False, False, False]
        del_res = fit_trajectory_delegation_threshold(
            cal_conf,
            cal_corr,
            delta_cost_usd=0.049,
            error_stakes_v=0.15,
            irreversible_stakes_v=5.0,
            reversible_stakes_v=0.15,
            min_bin_size=2,
            num_bins=3,
        )
        self.assertAlmostEqual(del_res.tau_prop1_marginal, 1.0 - 0.049 / 0.15, places=6)
        self.assertGreaterEqual(del_res.tau_irreversible_action, del_res.tau_reversible_action)

        # 2. Proposition 2 Trigger/Guard Resolution Utility (Eq. 5)
        y_true = [True, True, False, False, False, False, False, False, False, False]
        always_no = [False] * 10
        skilled = [True, True, False, False, False, False, False, False, False, True]
        u_no = evaluate_trigger_guard_utility(
            y_true, always_no, benefit_when_needed=0.80, harm_when_unneeded=0.05, module_cost=0.001, query_cost_kappa=0.0001
        )
        u_sk = evaluate_trigger_guard_utility(
            y_true, skilled, benefit_when_needed=0.80, harm_when_unneeded=0.05, module_cost=0.001, query_cost_kappa=0.0001
        )
        self.assertAlmostEqual(u_no.youden_j, 0.0, places=6)
        self.assertLessEqual(u_no.delta_utility_per_opportunity, 0.0)
        self.assertGreater(u_sk.youden_j, 0.75)
        self.assertGreater(u_sk.delta_utility_per_opportunity, 0.0)

        # 3. Section 5 Eq. (1) Selective Teacher + Brier SFT
        student = {"ask_clarification": 0.40, "cancel_order": 0.40, "lookup_order": 0.20}
        wrong_teacher = {"ask_clarification": 0.05, "cancel_order": 0.90, "lookup_order": 0.05}
        sft_res = selective_teacher_brier_sft_loss(
            teacher_probs=wrong_teacher,
            student_probs=student,
            gold_key="ask_clarification",
            acceptable_keys=["ask_clarification"],
            gold_weight_alpha=0.30,
            brier_weight_mu=0.50,
        )
        self.assertFalse(sft_res.teacher_accepted)
        self.assertEqual(sft_res.target_distribution["ask_clarification"], 1.0)
        self.assertEqual(acceptable_set_accuracy(["a", "b"], [["a", "c"], ["x"]]), 0.5)

        # 4. Appendix E Tango Paired Non-Inferiority Test + Holm-Bonferroni
        ni_pass = tango_paired_non_inferiority_test(
            wins=3, losses=0, n_total=500, margin=0.02, alpha=0.025
        )
        ni_fail = tango_paired_non_inferiority_test(
            wins=0, losses=12, n_total=200, margin=0.02, alpha=0.025
        )
        self.assertTrue(ni_pass.non_inferior)
        self.assertFalse(ni_fail.non_inferior)
        adj_ps, rej_flags = holm_bonferroni_adjust([ni_pass.p_berger_boos, ni_fail.p_berger_boos], alpha=0.025)
        self.assertEqual(len(adj_ps), 2)
        self.assertTrue(rej_flags[0])
        self.assertFalse(rej_flags[1])


if __name__ == "__main__":
    unittest.main()

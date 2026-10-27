"""Unit and deterministic verification tests for the v0.7.0 Frontier GenAI Suite."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from multimodal_decision_models import (
    BanditRoutingDecision,
    BanditStreamSummary,
    CorrelatedEnsembleCalibrationResult,
    ModelArmSpec,
    OnlineBCACIResult,
    OnlineBCACITracker,
    PRMBranchNode,
    PRMTreeSearchResult,
    PrimalDualParetoBanditRouter,
    calibrate_correlated_ensemble_weights,
    calibrate_prm_horizon_discount,
    covariance_discounted_logit_pool,
    evaluate_online_bc_aci_stream,
    evaluate_pareto_bandit_stream,
    run_calibrated_prm_puct_search,
    score_prm_rubric_distribution,
)


class TestV070FrontierGenAISuite(unittest.TestCase):
    """Tests all 4 pillars of the v0.7.0 Frontier GenAI Suite."""

    def test_pillar1_online_bc_aci_stream_and_tracker(self) -> None:
        warmup_probs = [
            {"A": 0.76, "B": 0.08, "C": 0.08, "D": 0.08} if i % 2 == 0 else {"A": 0.08, "B": 0.76, "C": 0.08, "D": 0.08}
            for i in range(40)
        ]
        warmup_labels = ["A" if i % 2 == 0 else "B" for i in range(40)]

        stream_probs = []
        stream_labels = []
        for t in range(80):
            if t < 40:
                lbl = "A" if t % 2 == 0 else "B"
                dist = {"A": 0.72, "B": 0.10, "C": 0.09, "D": 0.09} if lbl == "A" else {"A": 0.10, "B": 0.72, "C": 0.09, "D": 0.09}
            else:
                lbl = "D"
                dist = {"A": 0.36, "B": 0.24, "C": 0.18, "D": 0.22}
            stream_probs.append(dist)
            stream_labels.append(lbl)

        res = evaluate_online_bc_aci_stream(
            stream_probs,
            stream_labels,
            alpha_target=0.10,
            warmup_probs=warmup_probs,
            warmup_labels=warmup_labels,
            gamma=0.06,
            beta=0.04,
            eta_d=0.25,
            window_size=35,
            rolling_eval_window=25,
        )
        self.assertIsInstance(res, OnlineBCACIResult)
        self.assertGreaterEqual(res.empirical_coverage, 0.80)
        self.assertEqual(len(res.alpha_trajectory), 80)

        tracker = OnlineBCACITracker(alpha_target=0.10, gamma=0.05, beta=0.025)
        cset, covered, next_alpha = tracker.observe_and_update(
            {"A": 0.55, "B": 0.30, "C": 0.15}, "B"
        )
        self.assertIsInstance(covered, bool)
        self.assertGreater(len(cset.prediction_set), 0)
        self.assertGreater(next_alpha, 0.0)

    def test_pillar2_pareto_bandit_router_respects_budget(self) -> None:
        arms = [
            ModelArmSpec(arm_id="fast", cost_usd=0.0002, latency_ms=25.0),
            ModelArmSpec(arm_id="mid", cost_usd=0.0010, latency_ms=70.0),
            ModelArmSpec(arm_id="pro", cost_usd=0.0060, latency_ms=300.0),
        ]
        router = PrimalDualParetoBanditRouter(
            arms=arms,
            context_dim=3,
            budget_per_query_usd=0.0015,
            latency_sla_ms=120.0,
        )
        dec = router.select_arm([1.0, 0.15, 0.20])
        self.assertIsInstance(dec, BanditRoutingDecision)
        self.assertIn(dec.chosen_arm, {"fast", "mid", "pro"})

        contexts = [[1.0, 0.15 if i % 3 != 0 else 0.75, 0.20] for i in range(60)]
        arm_rewards = [
            {
                "fast": 0.88 if i % 3 != 0 else 0.35,
                "mid": 0.90 if i % 3 != 0 else 0.78,
                "pro": 0.93,
            }
            for i in range(60)
        ]
        dueling_pairs: list[tuple[str, str] | None] = [("mid", "fast") for _ in range(60)]

        unpaced = evaluate_pareto_bandit_stream(
            arms,
            contexts,
            arm_rewards,
            budget_per_query_usd=0.0015,
            eta_dual_cost=0.0,
        )
        paced = evaluate_pareto_bandit_stream(
            arms,
            contexts,
            arm_rewards,
            budget_per_query_usd=0.0015,
            latency_sla_ms=120.0,
            eta_dual_cost=15.0,
            dueling_pairs=dueling_pairs,
        )
        self.assertIsInstance(paced, BanditStreamSummary)
        self.assertLessEqual(paced.mean_cost_usd, 0.0015 + 1e-6)
        self.assertGreaterEqual(paced.mean_quality, unpaced.mean_quality)
        self.assertEqual(paced.budget_violation_ratio, 0.0)

    def test_pillar3_calibrated_prm_puct_search_and_rubric_scoring(self) -> None:
        mean_v, std_v = score_prm_rubric_distribution(
            {"0": 0.05, "1": 0.10, "2": 0.25, "3": 0.60}, horizon_discount=0.95
        )
        self.assertGreater(mean_v, 0.5)
        self.assertGreater(std_v, 0.0)

        cal_depths = [1, 1, 2, 2, 3, 3]
        cal_rubrics = [
            {"0": 0.05, "1": 0.05, "2": 0.20, "3": 0.70},
            {"0": 0.70, "1": 0.20, "2": 0.05, "3": 0.05},
            {"0": 0.05, "1": 0.10, "2": 0.25, "3": 0.60},
            {"0": 0.65, "1": 0.20, "2": 0.10, "3": 0.05},
            {"0": 0.05, "1": 0.05, "2": 0.15, "3": 0.75},
            {"0": 0.75, "1": 0.15, "2": 0.05, "3": 0.05},
        ]
        cal_outcomes = [True, False, True, False, True, False]
        discounts = calibrate_prm_horizon_discount(
            cal_depths, cal_rubrics, cal_outcomes, max_depth=3
        )
        self.assertEqual(set(discounts.keys()), {1, 2, 3})

        nodes = [
            PRMBranchNode("r1", None, 1, 0.60, {"0": 0.02, "1": 0.03, "2": 0.15, "3": 0.80}),
            PRMBranchNode("r2", None, 1, 0.40, {"0": 0.75, "1": 0.15, "2": 0.05, "3": 0.05}),
            PRMBranchNode("r1_l1", "r1", 2, 0.80, {"0": 0.01, "1": 0.04, "2": 0.10, "3": 0.85}, is_terminal=True, ground_truth_correct=True),
            PRMBranchNode("r2_l1", "r2", 2, 0.80, {"0": 0.80, "1": 0.10, "2": 0.05, "3": 0.05}, is_terminal=True, ground_truth_correct=False),
        ]
        res = run_calibrated_prm_puct_search(
            nodes,
            conformal_z=1.28155,
            c_puct=0.75,
            horizon_discounts=discounts,
            enable_conformal_pruning=True,
        )
        self.assertIsInstance(res, PRMTreeSearchResult)
        self.assertTrue(res.leaf_correct)
        self.assertLessEqual(res.evaluated_nodes, res.total_nodes)

    def test_pillar4_covariance_discounted_logit_pool_discounts_clones(self) -> None:
        calib_labels = ["A" if i % 2 == 0 else "B" for i in range(40)]
        indep_probs = []
        clone_probs = []
        for i, y in enumerate(calib_labels):
            indep_probs.append({"A": 0.82, "B": 0.18} if y == "A" else {"A": 0.18, "B": 0.82})
            clone_wrong = (i % 3) != 0
            if (y == "A") != clone_wrong:
                clone_probs.append({"A": 0.78, "B": 0.22})
            else:
                clone_probs.append({"A": 0.22, "B": 0.78})

        calib_agent_probs = {
            "indep_expert": indep_probs,
            "clone_1": clone_probs,
            "clone_2": clone_probs,
            "clone_3": clone_probs,
        }
        cal = calibrate_correlated_ensemble_weights(
            calib_agent_probs, calib_labels, shrinkage_delta=0.12
        )
        self.assertIsInstance(cal, CorrelatedEnsembleCalibrationResult)
        self.assertGreater(cal.weights["indep_expert"], cal.weights["clone_1"])
        self.assertLess(cal.effective_sample_size, 4.0)

        pooled = covariance_discounted_logit_pool(
            {aid: calib_agent_probs[aid][0] for aid in cal.agent_ids},
            calibration=cal,
        )
        self.assertAlmostEqual(sum(pooled.values()), 1.0, places=6)

    def test_benchmark_runner_has_zero_rng_imports(self) -> None:
        bench_path = (
            Path(__file__).resolve().parent.parent
            / "benchmarks"
            / "run_frontier_genai_v070_bench.py"
        )
        tree = ast.parse(bench_path.read_text(encoding="utf-8"))
        forbidden = {"random", "numpy.random"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, forbidden)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module, forbidden)


if __name__ == "__main__":
    unittest.main()

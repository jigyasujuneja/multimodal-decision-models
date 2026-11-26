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

"""Unit and guard-branch mutation tests for v1.2.0 research-science primitives."""

from __future__ import annotations

import math
import random
from typing import Mapping
import unittest

import multimodal_decision_models as mdm
from multimodal_decision_models.calibration import adapt_sld_label_shift_posteriors, debias_null_prior_probabilities
from multimodal_decision_models.conformal import BayesValueOfInformationResult, ConformalRiskCalibrationResult, ConformalRiskDecisionResult, OmniscienceActionDecision, asymmetric_false_allow_loss, calibrate_conformal_risk_control, compute_bayes_value_of_information, evaluate_conformal_risk_decision, omniscience_abstention_loss, select_omniscience_action
from multimodal_decision_models.eval_suite import brier_score_multiclass, expected_calibration_error


def _gaussian_source_posterior(
    feature_x: float,
    source_positive_prior: float,
    mean_separation: float = 1.2,
) -> dict[str, float]:
    """Compute exact Bayes source posterior for N(+mu, 1) vs N(-mu, 1)."""
    pos_log = math.log(source_positive_prior) + (mean_separation * feature_x)
    neg_log = math.log(1.0 - source_positive_prior) - (mean_separation * feature_x)
    max_log = max(pos_log, neg_log)
    pos_exp = math.exp(pos_log - max_log)
    neg_exp = math.exp(neg_log - max_log)
    denom = pos_exp + neg_exp
    return {
        "positive": pos_exp / denom,
        "negative": neg_exp / denom,
    }


class TestV120ResearchPrimitives(unittest.TestCase):
    """Comprehensive verification for SLD EM, PriDe null debiasing, CRC, and VoI."""

    def test_public_exports_and_sorted_dunder_all(self) -> None:
        """Verify all v1.2.0 functions and dataclasses are exported and sorted in __all__."""
        expected_symbols = (
            "BayesValueOfInformationResult",
            "ConformalRiskCalibrationResult",
            "ConformalRiskDecisionResult",
            "OmniscienceActionDecision",
            "adapt_sld_label_shift_posteriors",
            "asymmetric_false_allow_loss",
            "calibrate_conformal_risk_control",
            "compute_bayes_value_of_information",
            "debias_null_prior_probabilities",
            "evaluate_conformal_risk_decision",
            "omniscience_abstention_loss",
            "select_omniscience_action",
        )
        for symbol_name in expected_symbols:
            self.assertIn(symbol_name, mdm.__all__)
            self.assertTrue(hasattr(mdm, symbol_name))
        self.assertEqual(mdm.__all__, sorted(mdm.__all__))

    def test_sld_em_recovery_from_30_percent_to_80_and_10_percent(self) -> None:
        """Verify SLD EM recovers shifted priors (80% and 10%) and lowers Brier and ECE."""
        source_priors = {"positive": 0.30, "negative": 0.70}

        for target_positive_rate, seed_value in ((0.80, 202601), (0.10, 202602)):
            generator = random.Random(seed_value)
            unlabeled_batch: list[dict[str, float]] = []
            gold_labels: list[str] = []
            for _ in range(800):
                is_positive = generator.random() < target_positive_rate
                gold_key = "positive" if is_positive else "negative"
                mean_val = 1.2 if is_positive else -1.2
                feature_x = generator.gauss(mean_val, 1.0)
                unlabeled_batch.append(_gaussian_source_posterior(feature_x, 0.30, 1.2))
                gold_labels.append(gold_key)

            estimated_priors, adapted_posteriors, iterations_used = (
                adapt_sld_label_shift_posteriors(
                    source_priors=source_priors,
                    unlabeled_posteriors=unlabeled_batch,
                    max_iters=100,
                    tol=1e-7,
                    min_prior_floor=1e-6,
                    smoothing_alpha=1e-4,
                )
            )

            self.assertGreaterEqual(iterations_used, 2)
            self.assertLessEqual(iterations_used, 100)
            self.assertEqual(len(adapted_posteriors), len(unlabeled_batch))
            self.assertAlmostEqual(sum(estimated_priors.values()), 1.0, places=9)
            self.assertAlmostEqual(
                estimated_priors["positive"], target_positive_rate, delta=0.035
            )

            unadapted_brier = brier_score_multiclass(unlabeled_batch, gold_labels)
            adapted_brier = brier_score_multiclass(adapted_posteriors, gold_labels)
            unadapted_ece = expected_calibration_error(unlabeled_batch, gold_labels)
            adapted_ece = expected_calibration_error(adapted_posteriors, gold_labels)

            self.assertLess(adapted_brier, unadapted_brier)
            self.assertLess(adapted_ece, unadapted_ece)

        # Also verify 3-class SLD EM convergence on exact Gaussian source posteriors.
        multiclass_source = {"cat_a": 0.50, "cat_b": 0.30, "cat_c": 0.20}
        mc_rng = random.Random(202603)
        mc_batch: list[dict[str, float]] = []
        mu_sep = 2.0
        target_labels = (["cat_c"] * 650) + (["cat_b"] * 200) + (["cat_a"] * 150)
        for true_cls in target_labels:
            obs_vec = {
                cls_key: (mu_sep if cls_key == true_cls else 0.0) + mc_rng.gauss(0.0, 1.0)
                for cls_key in multiclass_source
            }
            logits = {
                cls_key: math.log(multiclass_source[cls_key]) + (mu_sep * obs_vec[cls_key])
                for cls_key in multiclass_source
            }
            max_logit = max(logits.values())
            exps = {cls_key: math.exp(val - max_logit) for cls_key, val in logits.items()}
            total_exp = sum(exps.values())
            mc_batch.append({cls_key: val / total_exp for cls_key, val in exps.items()})

        mc_priors, mc_adapted, mc_iters = adapt_sld_label_shift_posteriors(
            multiclass_source, mc_batch
        )
        self.assertGreater(mc_iters, 1)
        self.assertAlmostEqual(mc_priors["cat_c"], 0.65, delta=0.03)
        self.assertAlmostEqual(sum(mc_adapted[0].values()), 1.0, places=9)

    def test_pride_null_prior_debiasing_removes_slot_a_bias(self) -> None:
        """Verify PriDe log-space debiasing removes [0.88, 0.12] slot-A bias cleanly."""
        null_prior = {"slot_a": 0.88, "slot_b": 0.12}

        # Content-free raw distribution matching the biased null prior becomes exact uniform.
        neutral_debiased = debias_null_prior_probabilities(
            raw_probs={"slot_a": 0.88, "slot_b": 0.12},
            null_prior_probs=null_prior,
            alpha=1.0,
        )
        self.assertAlmostEqual(neutral_debiased["slot_a"], 0.50, places=9)
        self.assertAlmostEqual(neutral_debiased["slot_b"], 0.50, places=9)

        # True semantic preference [0.35, 0.65] corrupted by [0.88, 0.12] bias flips raw argmax
        # to slot_a (~0.7979), and debiasing restores exact [0.35, 0.65] and winner slot_b.
        true_semantic = {"slot_a": 0.35, "slot_b": 0.65}
        corrupted_unnorm = {
            key: true_semantic[key] * null_prior[key] for key in true_semantic
        }
        corrupted_sum = sum(corrupted_unnorm.values())
        corrupted_raw = {
            key: val / corrupted_sum for key, val in corrupted_unnorm.items()
        }
        self.assertGreater(corrupted_raw["slot_a"], 0.79)

        recovered = debias_null_prior_probabilities(corrupted_raw, null_prior, alpha=1.0)
        self.assertAlmostEqual(recovered["slot_a"], 0.35, places=9)
        self.assertAlmostEqual(recovered["slot_b"], 0.65, places=9)

        # alpha=0.0 preserves normalized raw distribution; zero probability entry stays finite.
        passthrough = debias_null_prior_probabilities(corrupted_raw, null_prior, alpha=0.0)
        self.assertAlmostEqual(passthrough["slot_a"], corrupted_raw["slot_a"], places=9)
        zero_entry = debias_null_prior_probabilities(
            {"slot_a": 1.0, "slot_b": 0.0}, null_prior, alpha=1.0
        )
        self.assertAlmostEqual(sum(zero_entry.values()), 1.0, places=9)
        self.assertGreater(zero_entry["slot_a"], 0.999)

    def test_crc_risk_bound_across_20_splits_fail_closed_and_omniscience(self) -> None:
        """Verify CRC finite-sample bound over 20 splits, fail-closed guards, and p*=0.50."""
        target_alpha = 0.10
        finite_sample_tolerance = 0.045
        selective_test_risks: list[float] = []
        safety_test_risks: list[float] = []

        for split_seed in range(20):
            rng = random.Random(9000 + split_seed)
            all_selective_probs: list[dict[str, float]] = []
            all_selective_labels: list[str] = []
            for _ in range(1000):
                true_idx = rng.randrange(3)
                true_key = f"opt_{true_idx}"
                signal_strength = rng.uniform(0.2, 2.5)
                logits = {
                    f"opt_{idx}": (signal_strength if idx == true_idx else 0.0)
                    + rng.gauss(0.0, 0.85)
                    for idx in range(3)
                }
                max_logit = max(logits.values())
                exps = {k: math.exp(v - max_logit) for k, v in logits.items()}
                denom = sum(exps.values())
                all_selective_probs.append({k: v / denom for k, v in exps.items()})
                all_selective_labels.append(true_key)

            calib_probs = all_selective_probs[:500]
            calib_labels = all_selective_labels[:500]
            test_probs = all_selective_probs[500:]
            test_labels = all_selective_labels[500:]

            crc_cal = calibrate_conformal_risk_control(
                calib_probs=calib_probs,
                calib_labels=calib_labels,
                alpha=target_alpha,
                loss_bound_b=1.0,
                loss_mode="omniscience_abstention",
            )
            self.assertIsInstance(crc_cal, ConformalRiskCalibrationResult)
            self.assertTrue(crc_cal.satisfied_bound)
            self.assertLessEqual(crc_cal.crc_upper_bound, target_alpha + 1e-12)

            split_loss_sum = 0.0
            for item_probs, gold_label in zip(test_probs, test_labels):
                decision = evaluate_conformal_risk_decision(item_probs, crc_cal)
                self.assertIsInstance(decision, ConformalRiskDecisionResult)
                realized_loss = omniscience_abstention_loss(
                    item_probs, gold_label, crc_cal.lambda_hat, hallucination_penalty=1.0
                )
                if decision.abstained:
                    self.assertEqual(realized_loss, 0.0)
                split_loss_sum += realized_loss
            empirical_test_risk = split_loss_sum / float(len(test_probs))
            selective_test_risks.append(empirical_test_risk)
            self.assertLessEqual(
                empirical_test_risk, target_alpha + finite_sample_tolerance
            )

            # Asymmetric safety gate with B=5.0 and target expected risk alpha_safety=0.50
            safety_probs: list[dict[str, float]] = []
            safety_labels: list[str] = []
            for _ in range(1000):
                is_safe = rng.random() < 0.60
                gold_safety = "safe" if is_safe else "unsafe"
                safe_logit = rng.gauss(1.3 if is_safe else -1.1, 1.0)
                safe_prob = 1.0 / (1.0 + math.exp(-safe_logit))
                safety_probs.append({"safe": safe_prob, "unsafe": 1.0 - safe_prob})
                safety_labels.append(gold_safety)

            safety_cal = calibrate_conformal_risk_control(
                calib_probs=safety_probs[:500],
                calib_labels=safety_labels[:500],
                alpha=0.50,
                loss_bound_b=5.0,
                loss_mode="asymmetric_false_allow",
                allow_key="safe",
            )
            self.assertTrue(safety_cal.satisfied_bound)
            safety_loss_sum = sum(
                asymmetric_false_allow_loss(
                    prob_map,
                    gold_key,
                    safety_cal.lambda_hat,
                    false_allow_cost=5.0,
                    allow_key="safe",
                )
                for prob_map, gold_key in zip(safety_probs[500:], safety_labels[500:])
            )
            safety_emp_risk = safety_loss_sum / 500.0
            safety_test_risks.append(safety_emp_risk)
            self.assertLessEqual(safety_emp_risk, 0.50 + 5.0 * finite_sample_tolerance)

        self.assertLessEqual(
            sum(selective_test_risks) / float(len(selective_test_risks)), target_alpha
        )
        self.assertLessEqual(
            sum(safety_test_risks) / float(len(safety_test_risks)), 0.50
        )

        # Exact boundary and fail-closed verification when (B / (n + 1)) > alpha:
        # For B=1.0, alpha=0.05: n=18 -> 1/19 = 0.05263 > 0.05 (fails closed to lambda=1.0);
        # n=19 -> 1/20 = 0.05 <= 0.05 (succeeds on clean calibration data).
        clean_18_probs = [{"ans_a": 0.99, "ans_b": 0.01} for _ in range(18)]
        clean_18_labels = ["ans_a"] * 18
        fail_closed_cal = calibrate_conformal_risk_control(
            clean_18_probs, clean_18_labels, alpha=0.05, loss_bound_b=1.0
        )
        self.assertFalse(fail_closed_cal.satisfied_bound)
        self.assertEqual(fail_closed_cal.lambda_hat, 1.0)
        # Even probability 1.0 test input must abstain when lambda_hat == 1.0!
        fail_closed_decision = evaluate_conformal_risk_decision(
            {"ans_a": 1.0, "ans_b": 0.0}, fail_closed_cal
        )
        self.assertTrue(fail_closed_decision.abstained)
        self.assertIsNone(fail_closed_decision.selected_option)

        clean_19_probs = [{"ans_a": 0.99, "ans_b": 0.01} for _ in range(19)]
        clean_19_labels = ["ans_a"] * 19
        boundary_pass_cal = calibrate_conformal_risk_control(
            clean_19_probs, clean_19_labels, alpha=0.05, loss_bound_b=1.0
        )
        self.assertTrue(boundary_pass_cal.satisfied_bound)
        self.assertEqual(boundary_pass_cal.lambda_hat, 0.0)

        # Custom monotone loss function where floor loss 0.25 > alpha=0.10 fails closed to 1.0.
        def _infeasible_monotone_loss(
            _probs: Mapping[str, float], _gold: str, lam_val: float
        ) -> float:
            return 0.25 + 0.50 * (1.0 - lam_val)

        infeasible_cal = calibrate_conformal_risk_control(
            clean_19_probs,
            clean_19_labels,
            alpha=0.10,
            loss_bound_b=1.0,
            loss_fn=_infeasible_monotone_loss,
        )
        self.assertFalse(infeasible_cal.satisfied_bound)
        self.assertEqual(infeasible_cal.lambda_hat, 1.0)

        # Verify select_omniscience_action threshold p* = 0.50 maximizes [-1, +1] score.
        below_half = select_omniscience_action(
            {"ans_a": 0.499, "ans_b": 0.501 - 0.002, "ans_c": 0.002},
            penalty_wrong=1.0,
            reward_correct=1.0,
        )
        self.assertIsInstance(below_half, OmniscienceActionDecision)
        self.assertAlmostEqual(below_half.optimal_threshold, 0.50, places=9)
        self.assertTrue(below_half.abstained)
        self.assertIsNone(below_half.selected_option)
        self.assertEqual(below_half.action, "abstain")

        exact_half = select_omniscience_action(
            {"ans_a": 0.50, "ans_b": 0.30, "ans_c": 0.20},
            penalty_wrong=1.0,
            reward_correct=1.0,
        )
        self.assertFalse(exact_half.abstained)
        self.assertEqual(exact_half.selected_option, "ans_a")
        self.assertAlmostEqual(exact_half.expected_optimal_score, 0.0, places=9)

        asymmetric_omni = select_omniscience_action(
            {"ans_a": 0.70, "ans_b": 0.30},
            penalty_wrong=3.0,
            reward_correct=1.0,
        )
        self.assertAlmostEqual(asymmetric_omni.optimal_threshold, 0.75, places=9)
        self.assertTrue(asymmetric_omni.abstained)

        # Empirical + analytical optimality check across candidate thresholds tau != 0.50.
        omni_rng = random.Random(424242)
        omni_items: list[tuple[dict[str, float], str]] = []
        for idx in range(8000):
            top_conf = 0.20 + 0.75 * (float(idx) / 7999.0)
            rem = (1.0 - top_conf) / 2.0
            item_p = {"opt_0": top_conf, "opt_1": rem, "opt_2": rem}
            draw_u = omni_rng.random()
            if draw_u < top_conf:
                gold_opt = "opt_0"
            elif draw_u < top_conf + rem:
                gold_opt = "opt_1"
            else:
                gold_opt = "opt_2"
            omni_items.append((item_p, gold_opt))

        bayes_expected_sum = sum(
            select_omniscience_action(p_map, 1.0, 1.0).expected_optimal_score
            for p_map, _ in omni_items
        )
        bayes_realized_sum = 0.0
        for p_map, gold_opt in omni_items:
            act = select_omniscience_action(p_map, 1.0, 1.0)
            if not act.abstained:
                bayes_realized_sum += 1.0 if act.selected_option == gold_opt else -1.0

        for subopt_tau in (0.0, 0.25, 0.38, 0.62, 0.75, 0.90, 1.0):
            subopt_expected_sum = 0.0
            subopt_realized_sum = 0.0
            for p_map, gold_opt in omni_items:
                top_p = p_map["opt_0"]
                if subopt_tau < 1.0 and top_p >= subopt_tau:
                    subopt_expected_sum += (2.0 * top_p) - 1.0
                    subopt_realized_sum += 1.0 if gold_opt == "opt_0" else -1.0
            self.assertGreater(bayes_expected_sum, subopt_expected_sum + 5.0)
            self.assertGreater(bayes_realized_sum, subopt_realized_sum)

    def test_voi_monotonicity_and_cost_matrix_consistency(self) -> None:
        """Verify VoI monotonicity in error_cost, Stage-1 confidence, and RPC/latency cost."""
        base_probs = {"route_a": 0.72, "route_b": 0.28}

        # 1. Strictly increasing in error_cost when expected_stage2_accuracy (0.94) > 0.72
        voi_by_error_cost = [
            compute_bayes_value_of_information(
                stage1_probs=base_probs,
                error_cost=cost_val,
                expected_stage2_accuracy=0.94,
                stage2_rpc_cost_usd=0.02,
                latency_cost_usd=0.005,
            ).expected_net_voi
            for cost_val in (0.05, 0.20, 1.0, 5.0, 25.0)
        ]
        for idx in range(len(voi_by_error_cost) - 1):
            self.assertLess(voi_by_error_cost[idx], voi_by_error_cost[idx + 1])

        # 2. Strictly decreasing in Stage-1 confidence
        voi_by_confidence = [
            compute_bayes_value_of_information(
                stage1_probs={"route_a": conf_val, "route_b": 1.0 - conf_val},
                error_cost=2.0,
                expected_stage2_accuracy=0.92,
                stage2_rpc_cost_usd=0.01,
                latency_cost_usd=0.002,
            ).expected_net_voi
            for conf_val in (0.55, 0.65, 0.75, 0.85, 0.95, 0.99)
        ]
        for idx in range(len(voi_by_confidence) - 1):
            self.assertGreater(voi_by_confidence[idx], voi_by_confidence[idx + 1])

        # 3. Strictly decreasing in stage2_rpc_cost_usd and latency_cost_usd
        voi_by_rpc_cost = [
            compute_bayes_value_of_information(
                stage1_probs=base_probs,
                error_cost=1.0,
                expected_stage2_accuracy=0.95,
                stage2_rpc_cost_usd=rpc_val,
                latency_cost_usd=0.001,
            ).expected_net_voi
            for rpc_val in (0.0, 0.005, 0.02, 0.10, 0.50)
        ]
        for idx in range(len(voi_by_rpc_cost) - 1):
            self.assertGreater(voi_by_rpc_cost[idx], voi_by_rpc_cost[idx + 1])

        # 4. Exact consistency across scalar, per-class mapping, and 2D loss matrix
        scalar_voi = compute_bayes_value_of_information(
            base_probs, error_cost=3.5, expected_stage2_accuracy=0.90, stage2_rpc_cost_usd=0.04
        )
        self.assertIsInstance(scalar_voi, BayesValueOfInformationResult)
        class_map_voi = compute_bayes_value_of_information(
            base_probs,
            error_cost={"route_a": 3.5, "route_b": 3.5},
            expected_stage2_accuracy=0.90,
            stage2_rpc_cost_usd=0.04,
        )
        matrix_voi = compute_bayes_value_of_information(
            base_probs,
            loss_matrix={
                "route_a": {"route_a": 0.0, "route_b": 3.5},
                "route_b": {"route_a": 3.5, "route_b": 0.0},
            },
            expected_stage2_accuracy=0.90,
            stage2_rpc_cost_usd=0.04,
        )
        self.assertAlmostEqual(scalar_voi.expected_net_voi, class_map_voi.expected_net_voi, places=12)
        self.assertAlmostEqual(scalar_voi.expected_net_voi, matrix_voi.expected_net_voi, places=12)
        self.assertEqual(scalar_voi.should_escalate, scalar_voi.expected_net_voi > 0.0)

        # Asymmetric safety cost shifts Stage-1 Bayes optimal action toward lower-risk option.
        asym_voi = compute_bayes_value_of_information(
            {"safe": 0.85, "unsafe": 0.15},
            class_error_costs={"safe": 1.0, "unsafe": 20.0},
            expected_stage2_accuracy=0.96,
            stage2_rpc_cost_usd=0.02,
        )
        self.assertEqual(asym_voi.stage1_best_action, "unsafe")
        self.assertAlmostEqual(asym_voi.risk_stage1, 0.85, places=9)
        self.assertTrue(asym_voi.should_escalate)

    def test_guard_branches_nan_inf_negative_empty_bool_and_mismatched_keys(self) -> None:
        """Mutate every validation guard across SLD, PriDe, CRC, Omniscience, and VoI."""
        valid_pair = {"a": 0.6, "b": 0.4}

        # 1. SLD guard branches
        for bad_priors in (
            {},
            {"a": -0.1, "b": 1.1},
            {"a": float("nan"), "b": 0.5},
            {"a": float("inf"), "b": 0.5},
            {"a": 0.0, "b": 0.0},
            {"a": True, "b": 0.5},  # type: ignore[dict-item]
            {"": 0.5, "b": 0.5},
        ):
            with self.assertRaisesRegex(ValueError, r"source_priors keys must be non-empty strings|source_priors must be a non-empty mapping|source_priors sum must be > 0|source_priors\['a'\] must be a finite non-boolean|source_priors\['a'\] must be finite and >= 0"):
                adapt_sld_label_shift_posteriors(bad_priors, [valid_pair])

        for bad_batch in (
            [],
            "ab",  # type: ignore[arg-type]
            [{}],
            [{"a": 0.5, "c": 0.5}],
            [{"a": -0.2, "b": 1.2}],
            [{"a": float("nan"), "b": 0.5}],
            [{"a": float("-inf"), "b": 0.5}],
            [{"a": 0.0, "b": 0.0}],
        ):
            with self.assertRaisesRegex(ValueError, r"unlabeled_posteriors must be a non-empty sequence|unlabeled_posteriors\[0\] keys \['a', 'c'\] do not|unlabeled_posteriors\[0\] must be a non-empty mapping|unlabeled_posteriors\[0\] sum must be > 0|unlabeled_posteriors\[0\]\['a'\] must be finite and >= 0"):
                adapt_sld_label_shift_posteriors(valid_pair, bad_batch)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, r"max_iters must be an integer >= 1"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], max_iters=0)
        with self.assertRaisesRegex(ValueError, r"max_iters must be an integer >= 1"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], max_iters=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"tol must be finite and >= 0"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], tol=-1e-5)
        with self.assertRaisesRegex(ValueError, r"tol must be finite and >= 0"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], tol=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"min_prior_floor must be finite and >= 0"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], min_prior_floor=-0.01)
        with self.assertRaisesRegex(ValueError, r"min_prior_floor \* K must be < 1\.0 \(got floor=0\.5"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], min_prior_floor=0.5)
        with self.assertRaisesRegex(ValueError, r"min_prior_floor must be <= 1\.0"):
            adapt_sld_label_shift_posteriors({"only": 1.0}, [{"only": 1.0}], min_prior_floor=1.2)
        with self.assertRaisesRegex(ValueError, r"smoothing_alpha must be finite and >= 0"):
            adapt_sld_label_shift_posteriors(valid_pair, [valid_pair], smoothing_alpha=-0.1)
        with self.assertRaisesRegex(ValueError, r"source_priors entries must be strictly > 0 when"):
            adapt_sld_label_shift_posteriors({"a": 0.0, "b": 1.0}, [valid_pair], min_prior_floor=0.0)

        # 2. PriDe null-prior debiasing guard branches
        for bad_raw, bad_null, bad_alpha in (
            ({}, valid_pair, 1.0),
            (valid_pair, {}, 1.0),
            (valid_pair, {"a": 0.5, "c": 0.5}, 1.0),
            ({"a": -0.1, "b": 1.1}, valid_pair, 1.0),
            ({"a": float("nan"), "b": 0.5}, valid_pair, 1.0),
            (valid_pair, {"a": float("inf"), "b": 0.5}, 1.0),
            (valid_pair, {"a": 0.0, "b": 0.0}, 1.0),
            (valid_pair, valid_pair, -0.5),
            (valid_pair, valid_pair, float("nan")),
            (valid_pair, valid_pair, float("inf")),
            (valid_pair, valid_pair, True),  # type: ignore[list-item]
        ):
            with self.assertRaisesRegex(ValueError, r"alpha must be finite and >=|null_prior_probs must be a n|null_prior_probs sum must be|null_prior_probs\['a'\] must b|raw_probs keys \['a', 'b'\] do|raw_probs must be a non-empt|raw_probs\['a'\] must be finit"):
                debias_null_prior_probabilities(bad_raw, bad_null, alpha=bad_alpha)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, r"eps must be finite and strictly in \(0, 1\)"):
            debias_null_prior_probabilities(valid_pair, valid_pair, alpha=1.0, eps=0.0)
        with self.assertRaisesRegex(ValueError, r"eps must be finite and strictly in \(0, 1\)"):
            debias_null_prior_probabilities(valid_pair, valid_pair, alpha=1.0, eps=1.0)

        # 3. Omniscience & CRC guard branches
        with self.assertRaisesRegex(ValueError, r"probability mapping must be non-empty"):
            select_omniscience_action({}, penalty_wrong=1.0, reward_correct=1.0)
        with self.assertRaisesRegex(ValueError, r"option keys must be non-empty strings"):
            select_omniscience_action({"": 0.5, "b": 0.5}, penalty_wrong=1.0, reward_correct=1.0)
        with self.assertRaisesRegex(ValueError, r"abstain_action must be a non-empty string"):
            select_omniscience_action(valid_pair, abstain_action="")
        with self.assertRaisesRegex(ValueError, r"penalty_wrong must be finite and >= 0"):
            select_omniscience_action(valid_pair, penalty_wrong=-1.0, reward_correct=1.0)
        with self.assertRaisesRegex(ValueError, r"reward_correct must be finite and > 0"):
            select_omniscience_action(valid_pair, penalty_wrong=1.0, reward_correct=0.0)
        with self.assertRaisesRegex(ValueError, r"penalty_wrong must be finite and >= 0"):
            select_omniscience_action(valid_pair, penalty_wrong=float("nan"), reward_correct=1.0)

        with self.assertRaisesRegex(ValueError, r"gold_label 'missing_key' not present in option"):
            omniscience_abstention_loss(valid_pair, "missing_key", 0.5)
        with self.assertRaisesRegex(ValueError, r"threshold_lambda must lie in \[0, 1\]"):
            omniscience_abstention_loss(valid_pair, "a", 1.2)
        with self.assertRaisesRegex(ValueError, r"hallucination_penalty must be finite and > 0"):
            omniscience_abstention_loss(valid_pair, "a", 0.5, hallucination_penalty=0.0)
        with self.assertRaisesRegex(ValueError, r"allow_key 'missing_safe' not present in option"):
            asymmetric_false_allow_loss(valid_pair, "a", 0.5, allow_key="missing_safe")
        with self.assertRaisesRegex(ValueError, r"gold_label 'missing_gold' not present in option"):
            asymmetric_false_allow_loss(valid_pair, "missing_gold", 0.5, allow_key="a")

        with self.assertRaisesRegex(ValueError, r"calib_probs and calib_labels must be non-empty"):
            calibrate_conformal_risk_control([], [], alpha=0.1)
        with self.assertRaisesRegex(ValueError, r"calib_probs and calib_labels must be non-empty"):
            calibrate_conformal_risk_control([valid_pair], ["a", "b"], alpha=0.1)
        with self.assertRaisesRegex(ValueError, r"calibration label 'unknown' at index 0 not present"):
            calibrate_conformal_risk_control([valid_pair], ["unknown"], alpha=0.1)
        with self.assertRaisesRegex(ValueError, r"alpha must be finite and > 0"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.0)
        with self.assertRaisesRegex(ValueError, r"alpha must be finite and > 0"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=-0.1)
        with self.assertRaisesRegex(ValueError, r"loss_bound_b must be finite and > 0"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, loss_bound_b=0.0)
        with self.assertRaisesRegex(ValueError, r"allow_key must be a non-empty string"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, allow_key="")
        with self.assertRaisesRegex(ValueError, r"lambda_grid must be non-empty when provided"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, lambda_grid=[])
        with self.assertRaisesRegex(ValueError, r"lambda_grid must lie in \[0, 1\]"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, lambda_grid=[1.5])
        with self.assertRaisesRegex(ValueError, r"num_grid_points must be an integer >= 2"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, num_grid_points=1)
        with self.assertRaisesRegex(ValueError, r"loss_fn must be callable when provided"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, loss_fn="bad")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"unsupported loss_mode 'invalid_mode'; expected one"):
            calibrate_conformal_risk_control([valid_pair], ["a"], alpha=0.1, loss_mode="invalid_mode")
        with self.assertRaisesRegex(ValueError, r"allow_key 'safe' missing from calibration"):
            calibrate_conformal_risk_control(
                [valid_pair], ["a"], alpha=0.1, loss_mode="asymmetric_false_allow", allow_key="safe"
            )

        # Non-monotone, NaN, and out-of-bounds loss_fn must be rejected!
        def _non_monotone_loss(_p: Mapping[str, float], _g: str, lam: float) -> float:
            return lam

        def _out_of_bounds_loss(_p: Mapping[str, float], _g: str, _lam: float) -> float:
            return 2.5

        def _nan_loss(_p: Mapping[str, float], _g: str, _lam: float) -> float:
            return float("nan")

        with self.assertRaisesRegex(ValueError, r"loss_fn is not monotone non-increasing at item 0"):
            calibrate_conformal_risk_control(
                [valid_pair], ["a"], alpha=0.2, loss_bound_b=1.0, loss_fn=_non_monotone_loss
            )
        with self.assertRaisesRegex(ValueError, r"loss_fn value 2\.5 outside \[0, 1\.0\] at item 0"):
            calibrate_conformal_risk_control(
                [valid_pair], ["a"], alpha=0.2, loss_bound_b=1.0, loss_fn=_out_of_bounds_loss
            )
        with self.assertRaisesRegex(ValueError, r"loss_fn returned non-finite value nan at item 0"):
            calibrate_conformal_risk_control(
                [valid_pair], ["a"], alpha=0.2, loss_bound_b=1.0, loss_fn=_nan_loss
            )

        with self.assertRaisesRegex(ValueError, r"calibration must lie in \[0, 1\]"):
            evaluate_conformal_risk_decision(valid_pair, -0.1)
        with self.assertRaisesRegex(ValueError, r"abstain_action must be a non-empty string"):
            evaluate_conformal_risk_decision(valid_pair, 0.5, abstain_action="")
        with self.assertRaisesRegex(ValueError, r"loss_mode must be a string"):
            evaluate_conformal_risk_decision(valid_pair, 0.5, loss_mode=123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"unsupported loss_mode 'bad_mode'"):
            evaluate_conformal_risk_decision(valid_pair, 0.5, loss_mode="bad_mode")
        with self.assertRaisesRegex(ValueError, r"allow_key 'missing' not present in option"):
            evaluate_conformal_risk_decision(
                valid_pair, 0.5, loss_mode="asymmetric_false_allow", allow_key="missing"
            )

        # 4. VoI guard branches
        with self.assertRaisesRegex(ValueError, r"probability mapping must be non-empty"):
            compute_bayes_value_of_information({})
        with self.assertRaisesRegex(ValueError, r"probability for 'a' must be finite and >= 0"):
            compute_bayes_value_of_information({"a": float("nan"), "b": 0.5})
        with self.assertRaisesRegex(ValueError, r"error_cost must be finite and >= 0"):
            compute_bayes_value_of_information(valid_pair, error_cost=-1.0)
        with self.assertRaisesRegex(ValueError, r"error_cost mapping must be non-empty"):
            compute_bayes_value_of_information(valid_pair, error_cost={})
        with self.assertRaisesRegex(ValueError, r"class_error_costs keys must match stage1_probs"):
            compute_bayes_value_of_information(valid_pair, error_cost={"a": 1.0, "c": 1.0})
        with self.assertRaisesRegex(ValueError, r"loss_matrix outer keys must match stage1_probs"):
            compute_bayes_value_of_information(
                valid_pair,
                loss_matrix={"a": {"a": 0.0, "b": 1.0}},
            )
        with self.assertRaisesRegex(ValueError, r"loss_matrix\['b'\] keys must match stage1_probs"):
            compute_bayes_value_of_information(
                valid_pair,
                loss_matrix={"a": {"a": 0.0, "b": 1.0}, "b": {"a": 1.0}},
            )
        with self.assertRaisesRegex(ValueError, r"expected_stage2_accuracy must lie in \[0, 1\]"):
            compute_bayes_value_of_information(valid_pair, expected_stage2_accuracy=1.05)
        with self.assertRaisesRegex(ValueError, r"stage2_rpc_cost_usd must be finite and >= 0"):
            compute_bayes_value_of_information(valid_pair, stage2_rpc_cost_usd=-0.01)
        with self.assertRaisesRegex(ValueError, r"latency_cost_usd must be finite and >= 0"):
            compute_bayes_value_of_information(valid_pair, latency_cost_usd=float("inf"))
        with self.assertRaisesRegex(ValueError, r"cannot specify mapping error_cost alongside"):
            compute_bayes_value_of_information(
                valid_pair,
                error_cost={"a": 1.0, "b": 1.0},
                class_error_costs={"a": 1.0, "b": 1.0},
            )
        with self.assertRaisesRegex(ValueError, r"cannot specify both class_error_costs and"):
            compute_bayes_value_of_information(
                valid_pair,
                error_cost=1.0,
                class_error_costs={"a": 1.0, "b": 1.0},
                loss_matrix={"a": {"a": 0.0, "b": 1.0}, "b": {"a": 1.0, "b": 0.0}},
            )


if __name__ == "__main__":
    unittest.main()

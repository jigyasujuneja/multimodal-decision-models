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

"""Unit and mutation tests for DomainCompetenceProfile and Dawid-Skene weights."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import concurrent.futures as cf
import math
import threading
import time
import unittest

import multimodal_decision_models as mdm
from multimodal_decision_models import cascade as mdm_cascade

CANONICAL_PARITY_TOLERANCE = 1e-12


class V130CompetenceAndFleetRouterTests(unittest.TestCase):
    """Verify competence circuit-breaker branches, parity vectors, and router."""

    def test_canonical_cross_sdk_parity_vectors(self) -> None:
        """Assert 12-decimal parity on the 4 canonical Python/Go/TS vectors."""
        # Vector 1: CLINC150 (K=31)
        v1_w1, v1_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.98,
            stage2_accuracy=0.94,
            num_options=31,
            floor_accuracy=1e-4,
        )
        self.assertLess(abs(v1_w1 - 7.2930176797727810), CANONICAL_PARITY_TOLERANCE)
        self.assertLess(abs(v1_w2 - 6.1527326947041034), CANONICAL_PARITY_TOLERANCE)

        clinc_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.98,
            stage2_accuracy=0.94,
            num_options=31,
            paired_escalation_delta=-1,
            sample_count=150,
            stage1_brier=0.0637,
            stage2_brier=0.9591,
        )
        self.assertTrue(clinc_profile.is_negative_transfer())

        # Vector 2: TypedDecisions (K=2)
        v2_w1, v2_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.7125,
            stage2_accuracy=0.625,
            num_options=2,
            floor_accuracy=1e-4,
        )
        self.assertLess(abs(v2_w1 - 0.9075570519054006), CANONICAL_PARITY_TOLERANCE)
        self.assertLess(abs(v2_w2 - 0.5108256237659907), CANONICAL_PARITY_TOLERANCE)

        typed_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.7125,
            stage2_accuracy=0.625,
            num_options=2,
            paired_escalation_delta=-3,
            sample_count=64,
        )
        self.assertTrue(typed_profile.is_negative_transfer())

        # Vector 3: Safety v2 (K=2)
        v3_w1, v3_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.88,
            stage2_accuracy=0.84,
            num_options=2,
            floor_accuracy=1e-4,
        )
        self.assertLess(abs(v3_w1 - 1.9924301646902063), CANONICAL_PARITY_TOLERANCE)
        self.assertLess(abs(v3_w2 - 1.6582280766035322), CANONICAL_PARITY_TOLERANCE)

        safety_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.88,
            stage2_accuracy=0.84,
            num_options=2,
            paired_escalation_delta=1,
            sample_count=40,
        )
        self.assertFalse(safety_profile.is_negative_transfer())

        # Vector 4: Chance-level Stage 2 (K=4)
        v4_w1, v4_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.85,
            stage2_accuracy=0.20,
            num_options=4,
            floor_accuracy=1e-4,
        )
        self.assertLess(abs(v4_w1 - 2.8332133440562162), CANONICAL_PARITY_TOLERANCE)
        self.assertEqual(v4_w2, 0.0)

    def test_negative_transfer_three_branch_contract_and_boundaries(self) -> None:
        """Verify the exact 3-branch negative-transfer contract and boundaries."""
        # Singleton delta = -1 on N=40 (36/40 = 0.90 vs 35/40 = 0.875) must NOT false-trip
        singleton_noise_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.90,
            stage2_accuracy=0.875,
            num_options=2,
            paired_escalation_delta=-1,
            sample_count=40,
        )
        self.assertFalse(singleton_noise_profile.is_negative_transfer())

        # Branch 1 boundary: paired_escalation_delta <= min_delta_hurt even if stage2 > stage1
        hurt_at_boundary = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.80,
            stage2_accuracy=0.82,
            num_options=2,
            paired_escalation_delta=-2,
        )
        self.assertTrue(hurt_at_boundary.is_negative_transfer(min_delta_hurt=-2))
        self.assertFalse(hurt_at_boundary.is_negative_transfer(min_delta_hurt=-3))

        # Branch 2 Brier degradation: triggers when stage2_accuracy < stage1_accuracy and stage2_brier > ratio * max(0.02, stage1_brier)
        brier_trip = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.96,
            stage2_accuracy=0.94,
            num_options=10,
            paired_escalation_delta=0,
            stage1_brier=0.06,
            stage2_brier=0.1501,
        )
        self.assertTrue(brier_trip.is_negative_transfer(brier_degradation_ratio=2.5))

        # Exact Brier equality (0.15 == 2.5 * 0.06) does NOT trip '>'
        brier_exact_tie = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.96,
            stage2_accuracy=0.94,
            num_options=10,
            paired_escalation_delta=0,
            stage1_brier=0.06,
            stage2_brier=0.15,
        )
        self.assertFalse(brier_exact_tie.is_negative_transfer(brier_degradation_ratio=2.5))

        # Partial None Brier scores skip the Brier guard cleanly
        partial_brier_s1_only = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.96,
            stage2_accuracy=0.94,
            num_options=10,
            paired_escalation_delta=0,
            stage1_brier=0.06,
            stage2_brier=None,
        )
        self.assertFalse(partial_brier_s1_only.is_negative_transfer())
        partial_brier_s2_only = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.96,
            stage2_accuracy=0.94,
            num_options=10,
            paired_escalation_delta=0,
            stage1_brier=None,
            stage2_brier=0.50,
        )
        self.assertFalse(partial_brier_s2_only.is_negative_transfer())

        # Strict stage2_accuracy < stage1_accuracy in Brier guard: tied 1.00 == 1.00 and 0.95 == 0.95 do NOT trip
        brier_perfect_tie = mdm.DomainCompetenceProfile(
            stage1_accuracy=1.0,
            stage2_accuracy=1.0,
            num_options=5,
            paired_escalation_delta=0,
            stage1_brier=0.02,
            stage2_brier=0.12,
        )
        self.assertFalse(brier_perfect_tie.is_negative_transfer(brier_degradation_ratio=2.5))

        brier_acc_tie = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.95,
            stage2_accuracy=0.95,
            num_options=5,
            paired_escalation_delta=0,
            stage1_brier=0.04,
            stage2_brier=0.11,
        )
        self.assertFalse(brier_acc_tie.is_negative_transfer(brier_degradation_ratio=2.5))

        brier_acc_strictly_lower = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.95,
            stage2_accuracy=0.9499,
            num_options=5,
            paired_escalation_delta=0,
            stage1_brier=0.04,
            stage2_brier=0.1001,
        )
        self.assertTrue(brier_acc_strictly_lower.is_negative_transfer(brier_degradation_ratio=2.5))

        # max(0.02, stage1_brier) floor prevents tiny near-zero Stage-1 Brier from false-tripping
        brier_floor_mutant_killer = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.99,
            stage2_accuracy=0.98,
            num_options=4,
            paired_escalation_delta=0,
            stage1_brier=0.005,
            stage2_brier=0.03,
        )
        self.assertFalse(brier_floor_mutant_killer.is_negative_transfer(brier_degradation_ratio=2.5))

        brier_floor_below = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.99,
            stage2_accuracy=0.98,
            num_options=4,
            paired_escalation_delta=0,
            stage1_brier=0.004,
            stage2_brier=0.05,
        )
        self.assertFalse(brier_floor_below.is_negative_transfer(brier_degradation_ratio=2.5))

        brier_floor_above = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.99,
            stage2_accuracy=0.98,
            num_options=4,
            paired_escalation_delta=0,
            stage1_brier=0.004,
            stage2_brier=0.0501,
        )
        self.assertTrue(brier_floor_above.is_negative_transfer(brier_degradation_ratio=2.5))

        # Brier blowup precedes paired_escalation_delta > 0 exemption
        brier_beats_positive_delta = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.95,
            stage2_accuracy=0.94,
            num_options=4,
            paired_escalation_delta=1,
            stage1_brier=0.05,
            stage2_brier=0.20,
        )
        self.assertTrue(brier_beats_positive_delta.is_negative_transfer(brier_degradation_ratio=2.5))

        # Branch 4 vs Branch 5: positive delta protects trailing overall accuracy, delta == 0 does not
        positive_delta_trailing = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.88,
            stage2_accuracy=0.70,
            num_options=2,
            paired_escalation_delta=1,
        )
        self.assertFalse(positive_delta_trailing.is_negative_transfer(margin=0.05))

        zero_delta_trailing = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.88,
            stage2_accuracy=0.70,
            num_options=2,
            paired_escalation_delta=0,
        )
        self.assertTrue(zero_delta_trailing.is_negative_transfer(margin=0.05))

        # Branch 5 exact margin boundary (0.75 - 0.25 == 0.50 exact in IEEE-754)
        margin_tie_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.75,
            stage2_accuracy=0.50,
            num_options=2,
            paired_escalation_delta=0,
        )
        self.assertFalse(margin_tie_profile.is_negative_transfer(margin=0.25))
        self.assertTrue(margin_tie_profile.is_negative_transfer(margin=0.0))

        margin_below_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.75,
            stage2_accuracy=0.4999,
            num_options=2,
            paired_escalation_delta=0,
        )
        self.assertTrue(margin_below_profile.is_negative_transfer(margin=0.25))

    def test_domain_competence_profile_validation_and_immutability(self) -> None:
        """Verify strict type, finiteness, range, and frozen-dataclass contracts."""
        valid_boundary = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.0,
            stage2_accuracy=1.0,
            num_options=2,
            paired_escalation_delta=0,
            sample_count=0,
            stage1_brier=0.0,
            stage2_brier=2.0,
        )
        self.assertFalse(valid_boundary.is_negative_transfer(margin=0.0, brier_degradation_ratio=1.0001))
        with self.assertRaisesRegex(FrozenInstanceError, r"cannot assign to field 'stage1_accuracy'"):
            valid_boundary.stage1_accuracy = 0.5  # type: ignore[misc]

        for bad_acc in (True, False, -0.001, 1.001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, r"stage1_accuracy must be a finite non-boolean number"):
                mdm.DomainCompetenceProfile(stage1_accuracy=bad_acc, stage2_accuracy=0.8)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, r"stage2_accuracy must be a finite non-boolean number"):
                mdm.DomainCompetenceProfile(stage1_accuracy=0.8, stage2_accuracy=bad_acc)  # type: ignore[arg-type]

        for bad_k in (True, False, 1, 0, -2, 2.5):
            with self.assertRaisesRegex(ValueError, r"num_options must be an integer >= 2"):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    num_options=bad_k,  # type: ignore[arg-type]
                )

        for bad_delta in (True, False, -1.5):
            with self.assertRaisesRegex(ValueError, r"paired_escalation_delta must be a non-boolean"):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    paired_escalation_delta=bad_delta,  # type: ignore[arg-type]
                )

        for bad_count in (True, False, -1, 3.0):
            with self.assertRaisesRegex(ValueError, r"sample_count must be an integer >= 0"):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    sample_count=bad_count,  # type: ignore[arg-type]
                )

        for bad_brier in (True, False, -0.0001, 2.0001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, r"stage1_brier must be None or a finite non-boolean"):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    stage1_brier=bad_brier,  # type: ignore[arg-type]
                )
            with self.assertRaisesRegex(ValueError, r"stage2_brier must be None or a finite non-boolean"):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    stage2_brier=bad_brier,  # type: ignore[arg-type]
                )

        base_profile = mdm.DomainCompetenceProfile(stage1_accuracy=0.8, stage2_accuracy=0.8)
        for bad_margin in (True, False, -1e-6, float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, r"margin must be a finite non-boolean float >= 0"):
                base_profile.is_negative_transfer(margin=bad_margin)  # type: ignore[arg-type]
        for bad_min_hurt in (True, False, -2.5, 1, 2, 5):
            with self.assertRaisesRegex(ValueError, r"min_delta_hurt must be <= 0|min_delta_hurt must be a non-boolean integer"):
                base_profile.is_negative_transfer(min_delta_hurt=bad_min_hurt)  # type: ignore[arg-type]
        high_delta_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.80,
            stage2_accuracy=0.95,
            num_options=2,
            paired_escalation_delta=5,
        )
        with self.assertRaisesRegex(ValueError, "min_delta_hurt must be <= 0"):
            high_delta_profile.is_negative_transfer(min_delta_hurt=1)
        self.assertFalse(high_delta_profile.is_negative_transfer(min_delta_hurt=0))
        zero_delta_higher_s2 = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.80,
            stage2_accuracy=0.95,
            num_options=2,
            paired_escalation_delta=0,
        )
        self.assertTrue(zero_delta_higher_s2.is_negative_transfer(min_delta_hurt=0))
        for bad_ratio in (True, False, 1.0, 0.99, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, r"brier_degradation_ratio must be a finite"):
                base_profile.is_negative_transfer(brier_degradation_ratio=bad_ratio)  # type: ignore[arg-type]

    def test_dawid_skene_weights_boundaries_and_validation(self) -> None:
        """Verify Dawid-Skene clamps at 0, 1/K, and 1 plus invalid input rejection."""
        # Stage 1 at 0.0 clamps to w1 = 0.01; Stage 2 at exact 1/K chance (0.25 for K=4) yields w2 = 0.0
        min_w1, chance_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.0,
            stage2_accuracy=0.25,
            num_options=4,
            floor_accuracy=1e-4,
        )
        self.assertEqual(min_w1, 0.01)
        self.assertLess(abs(chance_w2), 1e-15)

        # Perfect 1.0 accuracy uses 1 - effective_floor clamp without division by zero
        ceil_w1, ceil_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=1.0,
            stage2_accuracy=1.0,
            num_options=3,
            floor_accuracy=1e-3,
        )
        expected_ceil = math.log(0.999 / 0.001) + math.log(2.0)
        self.assertLess(abs(ceil_w1 - expected_ceil), CANONICAL_PARITY_TOLERANCE)
        self.assertLess(abs(ceil_w2 - expected_ceil), CANONICAL_PARITY_TOLERANCE)

        # Wide-catalog K=20,000 (1/K = 5e-5 < floor_accuracy = 1e-4): effective_floor = 0.5 / K keeps w2 == 0.0
        wide_w1, wide_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.85,
            stage2_accuracy=0.0,
            num_options=20_000,
            floor_accuracy=1e-4,
        )
        self.assertGreater(wide_w1, 11.0)
        self.assertEqual(wide_w2, 0.0)

        high_floor_w1, high_floor_w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.85,
            stage2_accuracy=0.0,
            num_options=5,
            floor_accuracy=0.24,
        )
        self.assertGreater(high_floor_w1, 3.0)
        self.assertEqual(high_floor_w2, 0.0)

        for bad_val in (True, False, -0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, r"stage1_accuracy must be a finite non-boolean number"):
                mdm.compute_dawid_skene_poe_weights(bad_val, 0.8)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, r"stage2_accuracy must be a finite non-boolean number"):
                mdm.compute_dawid_skene_poe_weights(0.8, bad_val)  # type: ignore[arg-type]

        for bad_k in (True, False, 1, 0, -3, 2.5):
            with self.assertRaisesRegex(ValueError, r"num_options must be an integer >= 2"):
                mdm.compute_dawid_skene_poe_weights(0.8, 0.8, num_options=bad_k)  # type: ignore[arg-type]

        for bad_floor in (True, False, 0.0, -1e-4, 0.25, 0.5, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, r"floor_accuracy must be a finite non-boolean float"):
                mdm.compute_dawid_skene_poe_weights(0.8, 0.8, floor_accuracy=bad_floor)  # type: ignore[arg-type]

    def test_pareto_bandit_router_selects_within_budget(self) -> None:
        """Verify primal-dual Pareto bandit raises dual price on budget overrun and converges to cheap arm."""
        arms = (
            mdm.ModelArmSpec(
                arm_id="decisions_fast",
                cost_usd=0.000012,
                latency_ms=120.0,
            ),
            mdm.ModelArmSpec(
                arm_id="flash_verifier",
                cost_usd=0.0022,
                latency_ms=950.0,
            ),
        )
        router = mdm.PrimalDualParetoBanditRouter(
            arms=arms,
            context_dim=3,
            budget_per_query_usd=0.0010,
            latency_sla_ms=600.0,
            eta_dual_cost=50_000.0,
            eta_dual_latency=0.5,
        )
        context = [1.0, 0.25, 0.0]
        # Observing over-budget costly runs must strictly increase the dual cost/latency prices.
        initial_dual_lambda = router.dual_lambda
        initial_dual_mu = router.dual_mu
        for _ in range(8):
            router.update(
                "flash_verifier",
                context,
                reward=0.92,
                incurred_cost_usd=0.0022,
                incurred_latency_ms=950.0,
            )
        self.assertGreater(router.dual_lambda, initial_dual_lambda)
        self.assertGreater(router.dual_mu, initial_dual_mu)

        for _ in range(4):
            router.update(
                "decisions_fast",
                context,
                reward=0.88,
                incurred_cost_usd=0.000012,
                incurred_latency_ms=120.0,
            )

        # Under primal-dual pacing, realized average cost must stay within the per-query budget.
        chosen_arms: list[str] = []
        realized_costs: list[float] = []
        for _ in range(20):
            decision = router.select_arm(context)
            self.assertIn(decision.chosen_arm, {"decisions_fast", "flash_verifier"})
            chosen_arms.append(decision.chosen_arm)
            arm_spec = arms[0] if decision.chosen_arm == "decisions_fast" else arms[1]
            realized_costs.append(arm_spec.cost_usd)
            router.update(
                decision.chosen_arm,
                context,
                reward=0.88 if decision.chosen_arm == "decisions_fast" else 0.92,
                incurred_cost_usd=arm_spec.cost_usd,
                incurred_latency_ms=arm_spec.latency_ms,
            )
        mean_cost = sum(realized_costs) / float(len(realized_costs))
        self.assertLessEqual(mean_cost, router.budget_per_query_usd)
        self.assertGreater(chosen_arms.count("decisions_fast"), chosen_arms.count("flash_verifier"))

    def test_v131_exports_version_and_orchestrator_competence_wiring(self) -> None:
        """Verify v1.3.1 exports, version, circuit-breaker short-circuit, and PoE fusion."""
        self.assertEqual(mdm.__version__, "1.3.1")
        self.assertIn("DomainCompetenceProfile", mdm.__all__)
        self.assertIn("compute_dawid_skene_poe_weights", mdm.__all__)
        self.assertEqual(mdm.__all__, sorted(mdm.__all__))

        questions = {
            "route": mdm.DecisionQuestion(
                type="choice",
                instructions="Select route",
                criteria={"opt_a": "Route A", "opt_b": "Route B"},
            )
        }
        stage2_calls: list[int] = []

        def _stage1_fn(_state: object, _qs: object, _imgs: object) -> dict[str, dict[str, float]]:
            return {"route": {"opt_a": 0.62, "opt_b": 0.38}}

        def _stage2_fn(_state: object, _qs: object, _imgs: object) -> dict[str, dict[str, float]]:
            stage2_calls.append(1)
            return {"route": {"opt_a": 0.44, "opt_b": 0.56}}

        stage1_backend = mdm.MockCallableBackend("mock_stage1", _stage1_fn)
        stage2_backend = mdm.MockCallableBackend("mock_stage2", _stage2_fn)

        # 1. Negative-transfer profile short-circuits Stage 2 to STAGE1_RESOLVED (0 Stage-2 calls)
        neg_transfer_orchestrator = mdm.CascadeDecisionOrchestrator(
            stage1_backend=stage1_backend,
            stage2_backend=stage2_backend,
            hesitation_threshold=0.20,
            domain_competence=mdm.DomainCompetenceProfile(
                stage1_accuracy=0.7125,
                stage2_accuracy=0.625,
                num_options=2,
                paired_escalation_delta=-3,
            ),
            use_dawid_skene_poe_weights=True,
        )
        short_circuit_result = neg_transfer_orchestrator.evaluate({"id": 1}, questions)
        self.assertEqual(len(stage2_calls), 0)
        self.assertFalse(short_circuit_result.escalated)
        self.assertEqual(
            short_circuit_result.terminal_state,
            mdm.DecisionTerminalState.STAGE1_RESOLVED,
        )
        self.assertEqual(short_circuit_result.slots["route"].predicted, "opt_a")
        self.assertEqual(
            short_circuit_result.slots["route"].terminal_state,
            mdm.DecisionTerminalState.STAGE1_RESOLVED,
        )

        # 2. Positive-transfer profile with use_dawid_skene_poe_weights=True fuses via convex (w1, w2)
        pos_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.92,
            stage2_accuracy=0.65,
            num_options=2,
            paired_escalation_delta=1,
        )
        poe_orchestrator = mdm.CascadeDecisionOrchestrator(
            stage1_backend=stage1_backend,
            stage2_backend=stage2_backend,
            hesitation_threshold=0.20,
            domain_competence=pos_profile,
            use_dawid_skene_poe_weights=True,
        )
        poe_result = poe_orchestrator.evaluate({"id": 2}, questions)
        self.assertEqual(len(stage2_calls), 1)
        self.assertTrue(poe_result.escalated)
        self.assertEqual(poe_result.terminal_state, mdm.DecisionTerminalState.STAGE2_VERIFIED)
        # Raw Stage 2 favored opt_b (0.56), whereas convex Dawid-Skene PoE retains opt_a
        self.assertEqual(poe_result.slots["route"].predicted, "opt_a")
        ds_w1, ds_w2 = mdm.compute_dawid_skene_poe_weights(0.92, 0.65, num_options=2)
        alpha_1 = ds_w1 / (ds_w1 + ds_w2)
        alpha_2 = ds_w2 / (ds_w1 + ds_w2)
        log_a = alpha_1 * math.log(0.62) + alpha_2 * math.log(0.44)
        log_b = alpha_1 * math.log(0.38) + alpha_2 * math.log(0.56)
        expected_prob_a = 1.0 / (1.0 + math.exp(log_b - log_a))
        self.assertLess(
            abs(poe_result.slots["route"].probabilities["opt_a"] - expected_prob_a),
            CANONICAL_PARITY_TOLERANCE,
        )
        self.assertEqual(poe_result.slots["route"].raw_probabilities, {"opt_a": 0.44, "opt_b": 0.56})

        # Fail-closed validation when use_dawid_skene_poe_weights=True lacks domain_competence
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"use_dawid_skene_poe_weights=True requires"):
            mdm.CascadeDecisionOrchestrator(
                stage1_backend=stage1_backend,
                stage2_backend=stage2_backend,
                use_dawid_skene_poe_weights=True,
            )
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"domain_competence must be a DomainCompetenceProfile"):
            mdm.CascadeDecisionOrchestrator(
                stage1_backend=stage1_backend,
                domain_competence="invalid",  # type: ignore[arg-type]
            )

    def test_mcnemar_large_discordant_and_holm_bonferroni_guards(self) -> None:
        """Verify paired_mcnemar_test overflow/underflow stability and Holm validation."""
        # C2: n01 = 600, n10 = 600 (disc = 1200 >= 1025) returns p == 1.0 without OverflowError
        tied_arm = [True] * 600 + [False] * 600
        tied_base = [False] * 600 + [True] * 600
        tied_mcnemar = mdm.paired_mcnemar_test(tied_arm, tied_base)
        self.assertEqual(tied_mcnemar.n01_arm_wins, 600)
        self.assertEqual(tied_mcnemar.n10_baseline_wins, 600)
        self.assertEqual(tied_mcnemar.p_two_sided, 1.0)
        self.assertEqual(tied_mcnemar.chi2_cc, 0.0)

        # Near-tie at disc = 1201 >= 1025 also evaluates cleanly via lgamma without OverflowError
        near_tied_mcnemar = mdm.paired_mcnemar_test(
            [True] * 600 + [False] * 601,
            [False] * 600 + [True] * 601,
        )
        self.assertGreater(near_tied_mcnemar.p_two_sided, 0.95)
        self.assertLessEqual(near_tied_mcnemar.p_two_sided, 1.0)

        # Extreme discordant skew n01 = 5, n10 = 1080 (disc = 1085 > 1075) stays strictly > 0
        skew_mcnemar = mdm.paired_mcnemar_test(
            [True] * 5 + [False] * 1080,
            [False] * 5 + [True] * 1080,
        )
        self.assertGreater(skew_mcnemar.p_two_sided, 0.0)
        self.assertLess(skew_mcnemar.p_two_sided, 1e-300)

        # N2: holm_bonferroni_adjust rejects booleans and NaN BEFORE sorting
        for invalid_p_seq in (
            [0.01, False, 0.20],
            [True, 0.02],
            [0.01, float("nan"), 0.20],
            [0.01, float("inf")],
            [-0.01, 0.05],
            [0.05, 1.01],
        ):
            with self.assertRaisesRegex(ValueError, r"index"):
                mdm.holm_bonferroni_adjust(invalid_p_seq)  # type: ignore[arg-type]

        # Non-monotonic raw product step-down: (0.01*4=0.04, 0.014*3=0.042, 0.02*2=0.040 -> 0.042, 0.50*1=0.50)
        permuted_ps = (0.02, 0.50, 0.01, 0.014)
        adj_p, rejected = mdm.holm_bonferroni_adjust(permuted_ps, alpha=0.05)
        expected_adj = (0.042, 0.50, 0.04, 0.042)
        for got_p, exp_p in zip(adj_p, expected_adj):
            self.assertLess(abs(got_p - exp_p), CANONICAL_PARITY_TOLERANCE)
        self.assertEqual(rejected, [True, False, True, True])

    def test_prune_choice_top_k_retains_option_descriptions_without_aliasing(self) -> None:
        """Verify prune_choice_top_k and shortlist_wide_question retain option_descriptions."""
        original_descriptions = {
            "opt_a": "Alpha description",
            "opt_b": "Bravo description",
            "opt_c": "Charlie description",
            "opt_d": "Delta description",
        }
        question = mdm.DecisionQuestion(
            type="choice",
            instructions="Select route.",
            criteria={"opt_a": "A", "opt_b": "B", "opt_c": "C", "opt_d": "D"},
            option_descriptions=dict(original_descriptions),
            prior_logits={"opt_a": -1.0, "opt_b": 0.8, "opt_c": 0.4, "opt_d": -2.0},
            domain_competence=mdm.DomainCompetenceProfile(
                stage1_accuracy=0.90,
                stage2_accuracy=0.92,
                num_options=4,
                paired_escalation_delta=2,
            ),
        )
        pruned = mdm_cascade.prune_choice_top_k(
            question,
            {"opt_a": 0.10, "opt_b": 0.52, "opt_c": 0.31, "opt_d": 0.07},
            top_k=2,
        )
        self.assertEqual(pruned.option_keys(), ["opt_b", "opt_c"])
        self.assertEqual(
            pruned.option_descriptions,
            {"opt_b": "Bravo description", "opt_c": "Charlie description"},
        )
        self.assertNotIn("opt_a", pruned.option_descriptions)
        self.assertNotIn("opt_d", pruned.option_descriptions)
        pruned.option_descriptions["opt_b"] = "Mutated"
        self.assertEqual(question.option_descriptions["opt_b"], "Bravo description")
        self.assertEqual(pruned.domain_competence, question.domain_competence)

        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"top_k must be an integer >= 2"):
            mdm.prune_choice_top_k(question, {"opt_a": 0.5, "opt_b": 0.5}, top_k=1)

        seen_stage2_descriptions: dict[str, str] = {}

        def _s1_wide(_state: object, _qs: object, _imgs: object) -> dict[str, dict[str, float]]:
            return {"slot": {"opt_a": 0.12, "opt_b": 0.46, "opt_c": 0.34, "opt_d": 0.08}}

        def _s2_pruned(
            _state: object,
            pruned_questions: dict[str, mdm.DecisionQuestion],
            _imgs: object,
        ) -> dict[str, dict[str, float]]:
            seen_stage2_descriptions.update(pruned_questions["slot"].option_descriptions)
            return {"slot": {"opt_b": 0.20, "opt_c": 0.80}}

        prune_orchestrator = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("s1_wide", _s1_wide),
            stage2_backend=mdm.MockCallableBackend("s2_pruned", _s2_pruned),
            hesitation_threshold=0.20,
            stage2_prune_top_k=2,
        )
        pruned_res = prune_orchestrator.evaluate({"req": 1}, {"slot": question})
        self.assertEqual(
            seen_stage2_descriptions,
            {"opt_b": "Bravo description", "opt_c": "Charlie description"},
        )
        self.assertEqual(pruned_res.slots["slot"].predicted, "opt_c")
        self.assertEqual(pruned_res.slots["slot"].probabilities["opt_a"], 0.0)
        self.assertAlmostEqual(pruned_res.slots["slot"].probabilities["opt_c"], 0.80, places=9)

    def test_speculative_t0_circuit_breaker_and_uniform_rescue_and_preservation(self) -> None:
        """Verify speculative t=0 short-circuit, uniform Stage-1 rescue, and Stage-2 guard."""
        neg_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.75,
            stage2_accuracy=0.62,
            num_options=2,
            paired_escalation_delta=-3,
        )
        pos_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.90,
            stage2_accuracy=0.88,
            num_options=2,
            paired_escalation_delta=2,
        )
        choice_q = mdm.DecisionQuestion(
            type="choice",
            instructions="Choose route.",
            criteria={"opt_a": "Route A", "opt_b": "Route B"},
        )

        # 1. All choice questions trip is_negative_transfer -> 0 speculative Stage-2 calls at t=0
        spec_calls: list[int] = []
        spec_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1",
                lambda _s, _q, _i: {"q1": {"opt_a": 0.60, "opt_b": 0.40}},
            ),
            stage2_backend=mdm.MockCallableBackend(
                "s2",
                lambda _s, _q, _i: (spec_calls.append(1) or {"q1": {"opt_a": 0.1, "opt_b": 0.9}}),
            ),
            hesitation_threshold=0.16,
            domain_competence=neg_profile,
            speculative_stage2=True,
        )
        blocked_res = spec_orch.evaluate({"case": "all_blocked"}, {"q1": choice_q})
        self.assertEqual(len(spec_calls), 0)
        self.assertFalse(blocked_res.escalated)
        self.assertEqual(blocked_res.slots["q1"].predicted, "opt_a")

        # 2. Mixed request (1 negative-transfer choice + 1 score question) -> speculative launches on score slot only
        mixed_s2_qids: list[list[str]] = []
        mixed_questions = {
            "blocked_choice": mdm.DecisionQuestion(
                type="choice",
                instructions="Blocked choice.",
                criteria={"opt_a": "A", "opt_b": "B"},
                domain_competence=neg_profile,
            ),
            "rubric_score": mdm.DecisionQuestion(
                type="score",
                instructions="Score 1..2.",
                levels=["1", "2"],
                domain_competence=neg_profile,
            ),
        }

        def _mixed_s2(
            _s: object,
            qs: dict[str, mdm.DecisionQuestion],
            _i: object,
        ) -> dict[str, dict[str, float]]:
            mixed_s2_qids.append(sorted(qs.keys()))
            return {qid: {"1": 0.15, "2": 0.85} for qid in qs}

        mixed_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_mixed",
                lambda _s, _q, _i: {
                    "blocked_choice": {"opt_a": 0.60, "opt_b": 0.40},
                    "rubric_score": {"1": 0.55, "2": 0.45},
                },
            ),
            stage2_backend=mdm.MockCallableBackend("s2_mixed", _mixed_s2),
            hesitation_threshold=0.16,
            speculative_stage2=True,
        )
        mixed_res = mixed_orch.evaluate({"case": "mixed"}, mixed_questions)
        self.assertEqual(mixed_s2_qids, [["rubric_score"]])
        self.assertFalse(mixed_res.slots["blocked_choice"].escalated)
        self.assertEqual(mixed_res.slots["blocked_choice"].predicted, "opt_a")
        self.assertTrue(mixed_res.slots["rubric_score"].escalated)
        self.assertEqual(mixed_res.slots["rubric_score"].predicted, "2")

        # 3. Uniform Stage-1 abstention (|p_i - 1/K| <= 1e-6) rescues via two-phase (0.0, 1.0)
        # even when is_negative_transfer is tripped and use_dawid_skene_poe_weights=True!
        rescue_calls: list[int] = []
        rescue_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_uniform",
                lambda _s, _q, _i: {"q1": {"opt_a": 0.5000004, "opt_b": 0.4999996}},
            ),
            stage2_backend=mdm.MockCallableBackend(
                "s2_rescue",
                lambda _s, _q, _i: (
                    rescue_calls.append(1) or {"q1": {"opt_a": 0.15, "opt_b": 0.85}}
                ),
            ),
            hesitation_threshold=0.16,
            domain_competence=neg_profile,
            use_dawid_skene_poe_weights=True,
            speculative_stage2=True,
        )
        rescue_res = rescue_orch.evaluate({"case": "uniform_s1_rescue"}, {"q1": choice_q})
        self.assertEqual(len(rescue_calls), 1)
        self.assertTrue(rescue_res.escalated)
        self.assertEqual(rescue_res.slots["q1"].predicted, "opt_b")
        self.assertLess(
            abs(rescue_res.slots["q1"].probabilities["opt_b"] - 0.85),
            CANONICAL_PARITY_TOLERANCE,
        )

        # Boundary just outside 1e-6 (|p_i - 0.5| = 2e-6 > 1e-6) stays blocked on neg_profile
        non_uniform_calls: list[int] = []
        near_uniform_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_near_uniform",
                lambda _s, _q, _i: {"q1": {"opt_a": 0.500002, "opt_b": 0.499998}},
            ),
            stage2_backend=mdm.MockCallableBackend(
                "s2_unused",
                lambda _s, _q, _i: (
                    non_uniform_calls.append(1) or {"q1": {"opt_a": 0.1, "opt_b": 0.9}}
                ),
            ),
            hesitation_threshold=0.16,
            domain_competence=neg_profile,
            speculative_stage2=True,
        )
        near_res = near_uniform_orch.evaluate({"case": "near_uniform"}, {"q1": choice_q})
        self.assertEqual(len(non_uniform_calls), 0)
        self.assertFalse(near_res.escalated)

        # 4. Uniform Stage-2 fallback while Stage 1 is non-uniform preserves Stage 1 (1.0, 0.0)
        for use_poe in (False, True):
            preserve_orch = mdm.CascadeDecisionOrchestrator(
                stage1_backend=mdm.MockCallableBackend(
                    "s1_calibrated",
                    lambda _s, _q, _i: {"q1": {"opt_a": 0.62, "opt_b": 0.38}},
                ),
                stage2_backend=mdm.MockCallableBackend(
                    "s2_uniform_fallback",
                    lambda _s, _q, _i: {"q1": {"opt_a": 0.50, "opt_b": 0.50}},
                ),
                hesitation_threshold=0.16,
                domain_competence=pos_profile,
                use_dawid_skene_poe_weights=use_poe,
            )
            preserve_res = preserve_orch.evaluate({"case": f"preserve_{use_poe}"}, {"q1": choice_q})
            self.assertTrue(preserve_res.escalated)
            self.assertEqual(preserve_res.slots["q1"].predicted, "opt_a")
            self.assertLess(
                abs(preserve_res.slots["q1"].probabilities["opt_a"] - 0.62),
                CANONICAL_PARITY_TOLERANCE,
            )
            self.assertLess(
                abs(preserve_res.slots["q1"].probabilities["opt_b"] - 0.38),
                CANONICAL_PARITY_TOLERANCE,
            )

    def test_cancel_event_propagation_fast_exit_non_mutation_and_voi_bounds(self) -> None:
        """Verify cancel_event propagation without caller mutation and VoI [0, 1] bounds."""
        choice_q = mdm.DecisionQuestion(
            type="choice",
            instructions="Select action.",
            criteria={"allow": "Allow", "deny": "Deny"},
        )
        stage2_saw_local_abort = threading.Event()
        stage2_started = threading.Event()

        def _slow_spec_s2(
            _s: object,
            _q: object,
            _i: object,
            *,
            cancel_event: object = None,
        ) -> dict[str, dict[str, float]]:
            stage2_started.set()
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if cancel_event is not None and getattr(cancel_event, "is_set")():
                    stage2_saw_local_abort.set()
                    return {"act": {"allow": 0.2, "deny": 0.8}}
                time.sleep(0.005)
            return {"act": {"allow": 0.2, "deny": 0.8}}

        def _confident_s1(_s: object, _q: object, _i: object) -> dict[str, dict[str, float]]:
            stage2_started.wait(timeout=0.5)
            return {"act": {"allow": 0.99, "deny": 0.01}}

        fast_exit_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("s1_fast", _confident_s1),
            stage2_backend=mdm.MockCallableBackend("s2_slow", _slow_spec_s2),
            hesitation_threshold=0.25,
            speculative_stage2=True,
        )
        caller_event = threading.Event()
        fast_res = fast_exit_orch.evaluate(
            {"req": "fast_exit"},
            {"act": choice_q},
            cancel_event=caller_event,
        )
        self.assertFalse(fast_res.escalated)
        self.assertEqual(fast_res.slots["act"].predicted, "allow")
        # Caller's event MUST NOT be mutated on Stage-1 fast-exit!
        self.assertFalse(caller_event.is_set())
        self.assertTrue(stage2_saw_local_abort.wait(timeout=0.5))

        # Pre-set cancel_event raises CancelledError before calling Stage 1
        s1_invocations: list[int] = []
        cancel_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_never",
                lambda _s, _q, _i: (s1_invocations.append(1) or {"act": {"allow": 0.6, "deny": 0.4}}),
            ),
            stage2_backend=mdm.MockCallableBackend(
                "s2_never",
                lambda _s, _q, _i: {"act": {"allow": 0.2, "deny": 0.8}},
            ),
        )
        preset_event = threading.Event()
        preset_event.set()
        with self.assertRaisesRegex(cf.CancelledError, r"Decision evaluation cancelled before Stage 1"):
            cancel_orch.evaluate({"req": "pre_cancelled"}, {"act": choice_q}, cancel_event=preset_event)
        self.assertEqual(len(s1_invocations), 0)

        # Mid-flight cancellation while speculative Stage 2 is running raises CancelledError promptly
        mid_event = threading.Event()

        def _hesitant_s1(_s: object, _q: object, _i: object) -> dict[str, dict[str, float]]:
            return {"act": {"allow": 0.58, "deny": 0.42}}

        def _blocking_s2(_s: object, _q: object, _i: object) -> dict[str, dict[str, float]]:
            time.sleep(0.5)
            return {"act": {"allow": 0.1, "deny": 0.9}}

        mid_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("s1_hes", _hesitant_s1),
            stage2_backend=mdm.MockCallableBackend("s2_block", _blocking_s2),
            hesitation_threshold=0.16,
            speculative_stage2=True,
        )
        timer = threading.Timer(0.03, mid_event.set)
        timer.start()
        try:
            with self.assertRaisesRegex(cf.CancelledError, r"Decision evaluation cancelled during speculative"):
                mid_orch.evaluate({"req": "mid_cancel"}, {"act": choice_q}, cancel_event=mid_event)
        finally:
            timer.cancel()

        # voi_stage2_expected_accuracy relaxed to full [0.0, 1.0] (including < 0.50)
        for valid_voi_acc in (0.0, 0.25, 0.49, 0.50, 0.95, 1.0):
            voi_orch = mdm.CascadeDecisionOrchestrator(
                stage1_backend=mdm.MockCallableBackend(
                    "s1_voi",
                    lambda _s, _q, _i: {"act": {"allow": 0.85, "deny": 0.15}},
                ),
                voi_error_cost=1.0,
                voi_stage2_expected_accuracy=valid_voi_acc,
            )
            self.assertEqual(voi_orch.voi_stage2_expected_accuracy, valid_voi_acc)

        for invalid_voi_acc in (True, False, -0.01, 1.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"voi_stage2_expected_accuracy must be a finite|voi_stage2_expected_accuracy must be finite|voi_stage2_expected_accuracy must lie in \[0\.0, 1\.0\]"):
                mdm.CascadeDecisionOrchestrator(
                    stage1_backend=mdm.MockCallableBackend(
                        "s1_voi",
                        lambda _s, _q, _i: {"act": {"allow": 0.85, "deny": 0.15}},
                    ),
                    voi_stage2_expected_accuracy=invalid_voi_acc,  # type: ignore[arg-type]
                )

    def test_temperature_bounds_and_pre_division_logit_shift_stability(self) -> None:
        """Verify [1e-3, 100.0] temperature bounds and log-space pre-division shift."""
        self.assertEqual(mdm.MIN_TEMPERATURE, 1e-3)
        self.assertEqual(mdm.MAX_TEMPERATURE, 100.0)

        raw_three = {"opt_a": 0.35, "opt_b": 0.33, "opt_c": 0.32}
        expected_unscaled_h = mdm.normalized_entropy(raw_three)

        fused_sharp, h_sharp, esc_sharp = mdm.pre_gated_bayesian_poe_fusion(
            raw_three,
            temperature=1e-3,
            hesitation_threshold=0.50,
        )
        self.assertTrue(all(math.isfinite(prob) for prob in fused_sharp.values()))
        self.assertAlmostEqual(sum(fused_sharp.values()), 1.0, places=12)
        self.assertAlmostEqual(fused_sharp["opt_a"], 1.0, places=12)
        self.assertAlmostEqual(h_sharp, expected_unscaled_h, places=12)
        self.assertTrue(esc_sharp)

        fused_flat, h_flat, _ = mdm.pre_gated_bayesian_poe_fusion(
            raw_three,
            prior_logits={"opt_a": 0.5, "opt_b": -0.2, "opt_c": -0.3},
            prior_weight=0.4,
            temperature=100.0,
        )
        self.assertTrue(all(math.isfinite(prob) for prob in fused_flat.values()))
        self.assertAlmostEqual(sum(fused_flat.values()), 1.0, places=12)
        self.assertAlmostEqual(h_flat, expected_unscaled_h, places=12)

        choice_q = mdm.DecisionQuestion(
            type="choice",
            instructions="Choose option.",
            criteria={"opt_a": "A", "opt_b": "B", "opt_c": "C"},
        )
        min_temp_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_three",
                lambda _s, _q, _i: {"slot": dict(raw_three)},
            ),
            temperature=1e-3,
        )
        min_temp_res = min_temp_orch.evaluate({"id": 1}, {"slot": choice_q})
        self.assertEqual(min_temp_res.slots["slot"].predicted, "opt_a")
        self.assertAlmostEqual(min_temp_res.slots["slot"].probabilities["opt_a"], 1.0, places=12)
        self.assertAlmostEqual(
            min_temp_res.slots["slot"].confidence,
            1.0 - expected_unscaled_h,
            places=12,
        )

        for invalid_temp in (True, False, 0.0, -0.5, 9.99e-4, 100.01, float("nan"), float("inf")):
            with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"temperature must lie in \[0\.001, 100\.0\]"):
                mdm.pre_gated_bayesian_poe_fusion(
                    raw_three,
                    temperature=invalid_temp,  # type: ignore[arg-type]
                )
            with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"temperature must be > 0|temperature must be a finite non-boolean number|temperature must be finite|temperature must lie in \[0\.001, 100\.0\]"):
                mdm.CascadeDecisionOrchestrator(
                    stage1_backend=mdm.MockCallableBackend(
                        "s1_bad_temp",
                        lambda _s, _q, _i: {"slot": dict(raw_three)},
                    ),
                    temperature=invalid_temp,  # type: ignore[arg-type]
                )

        mutated_orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend(
                "s1_mut_temp",
                lambda _s, _q, _i: {"slot": dict(raw_three)},
            ),
            total_contract_mode=False,
        )
        mutated_orch.temperature = 5e-4
        with self.assertRaisesRegex(mdm.SchemaContractViolationError, r"temperature must lie in \[0\.001, 100\.0\]"):
            mutated_orch.evaluate({"id": 2}, {"slot": choice_q})

    def test_directional_paired_mcnemar_and_holm_superiority_guard(self) -> None:
        """Verify arm_wins < baseline_wins never sets directional win or Holm rejection."""
        arm_inferior = [False] * 10
        base_superior = [True] * 10
        loss_res = mdm.paired_mcnemar_test(arm_inferior, base_superior, alpha=0.05)
        self.assertEqual(loss_res.n01_arm_wins, 0)
        self.assertEqual(loss_res.n10_baseline_wins, 10)
        self.assertLess(loss_res.p_two_sided, 0.01)
        self.assertEqual(loss_res.p_one_sided_greater, 1.0)
        self.assertFalse(loss_res.positive_direction)
        self.assertFalse(loss_res.significant_improvement)
        self.assertFalse(loss_res.reject_h0)

        arm_superior = [True] * 10
        base_inferior = [False] * 10
        win_res = mdm.paired_mcnemar_test(arm_superior, base_inferior, alpha=0.05)
        self.assertEqual(win_res.n01_arm_wins, 10)
        self.assertEqual(win_res.n10_baseline_wins, 0)
        self.assertLess(win_res.p_two_sided, 0.01)
        self.assertLess(win_res.p_one_sided_greater, 0.001)
        self.assertTrue(win_res.positive_direction)
        self.assertTrue(win_res.significant_improvement)
        self.assertTrue(win_res.reject_h0)

        adj_bool, rej_bool = mdm.holm_bonferroni_adjust(
            [loss_res.p_two_sided, win_res.p_two_sided],
            alpha=0.05,
            effect_signs=[loss_res.positive_direction, win_res.positive_direction],
        )
        self.assertLess(adj_bool[0], 0.01)
        self.assertLess(adj_bool[1], 0.01)
        self.assertEqual(rej_bool, [False, True])

        _, rej_numeric = mdm.holm_bonferroni_adjust(
            [loss_res.p_two_sided, loss_res.p_two_sided, win_res.p_two_sided],
            alpha=0.05,
            effect_signs=[-10.0, 0.0, 10.0],
        )
        self.assertEqual(rej_numeric, [False, False, True])

        for bad_alpha in (True, False, 0.0, 1.0, -0.1, float("nan")):
            with self.assertRaisesRegex(ValueError, r"alpha must lie in \(0, 1\)"):
                mdm.paired_mcnemar_test(
                    arm_superior,
                    base_inferior,
                    alpha=bad_alpha,  # type: ignore[arg-type]
                )
        with self.assertRaisesRegex(ValueError, r"effect_signs length \(1\) must match p_values length"):
            mdm.holm_bonferroni_adjust([0.01, 0.02], effect_signs=[True])
        with self.assertRaisesRegex(ValueError, r"invalid effect_signs entry at index 1: nan"):
            mdm.holm_bonferroni_adjust([0.01, 0.02], effect_signs=[True, float("nan")])


if __name__ == "__main__":
    unittest.main()

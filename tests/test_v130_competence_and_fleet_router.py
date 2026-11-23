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
import math
import unittest

import multimodal_decision_models as mdm

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

        # Vector 3: Aegis v2 (K=2)
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
        with self.assertRaises(FrozenInstanceError):
            valid_boundary.stage1_accuracy = 0.5  # type: ignore[misc]

        for bad_acc in (True, False, -0.001, 1.001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(stage1_accuracy=bad_acc, stage2_accuracy=0.8)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(stage1_accuracy=0.8, stage2_accuracy=bad_acc)  # type: ignore[arg-type]

        for bad_k in (True, False, 1, 0, -2, 2.5):
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    num_options=bad_k,  # type: ignore[arg-type]
                )

        for bad_delta in (True, False, -1.5):
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    paired_escalation_delta=bad_delta,  # type: ignore[arg-type]
                )

        for bad_count in (True, False, -1, 3.0):
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    sample_count=bad_count,  # type: ignore[arg-type]
                )

        for bad_brier in (True, False, -0.0001, 2.0001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    stage1_brier=bad_brier,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                mdm.DomainCompetenceProfile(
                    stage1_accuracy=0.8,
                    stage2_accuracy=0.8,
                    stage2_brier=bad_brier,  # type: ignore[arg-type]
                )

        base_profile = mdm.DomainCompetenceProfile(stage1_accuracy=0.8, stage2_accuracy=0.8)
        for bad_margin in (True, False, -1e-6, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                base_profile.is_negative_transfer(margin=bad_margin)  # type: ignore[arg-type]
        for bad_min_hurt in (True, False, -2.5):
            with self.assertRaises(ValueError):
                base_profile.is_negative_transfer(min_delta_hurt=bad_min_hurt)  # type: ignore[arg-type]
        for bad_ratio in (True, False, 1.0, 0.99, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError):
                mdm.compute_dawid_skene_poe_weights(bad_val, 0.8)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                mdm.compute_dawid_skene_poe_weights(0.8, bad_val)  # type: ignore[arg-type]

        for bad_k in (True, False, 1, 0, -3, 2.5):
            with self.assertRaises(ValueError):
                mdm.compute_dawid_skene_poe_weights(0.8, 0.8, num_options=bad_k)  # type: ignore[arg-type]

        for bad_floor in (True, False, 0.0, -1e-4, 0.25, 0.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mdm.compute_dawid_skene_poe_weights(0.8, 0.8, floor_accuracy=bad_floor)  # type: ignore[arg-type]

    def test_pareto_bandit_router_selects_within_budget(self) -> None:
        """Verify primal-dual Pareto bandit selects a valid registered arm."""
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
        )
        decision = router.select_arm([1.0, 0.25, 0.0])
        self.assertIn(decision.chosen_arm, {"decisions_fast", "flash_verifier"})

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
        with self.assertRaises(mdm.SchemaContractViolationError):
            mdm.CascadeDecisionOrchestrator(
                stage1_backend=stage1_backend,
                stage2_backend=stage2_backend,
                use_dawid_skene_poe_weights=True,
            )
        with self.assertRaises(mdm.SchemaContractViolationError):
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
            with self.assertRaises(ValueError):
                mdm.holm_bonferroni_adjust(invalid_p_seq)  # type: ignore[arg-type]

        # Non-monotonic raw product step-down: (0.01*4=0.04, 0.014*3=0.042, 0.02*2=0.040 -> 0.042, 0.50*1=0.50)
        permuted_ps = (0.02, 0.50, 0.01, 0.014)
        adj_p, rejected = mdm.holm_bonferroni_adjust(permuted_ps, alpha=0.05)
        expected_adj = (0.042, 0.50, 0.04, 0.042)
        for got_p, exp_p in zip(adj_p, expected_adj):
            self.assertLess(abs(got_p - exp_p), CANONICAL_PARITY_TOLERANCE)
        self.assertEqual(rejected, [True, False, True, True])


if __name__ == "__main__":
    unittest.main()

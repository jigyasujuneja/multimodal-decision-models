# Copyright 2026 Jigyasu Juneja
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for v1.3 DomainCompetenceProfile and Dawid-Skene PoE weights."""

from __future__ import annotations

import unittest

import multimodal_decision_models as mdm


class V130CompetenceAndFleetRouterTests(unittest.TestCase):
    """Verify conditional circuit-breaker and Dawid-Skene multi-class weights."""

    def test_conditional_negative_transfer_distinguishes_aegis_from_typed(self) -> None:
        # When Stage 2 hurts on the escalated subset and trails overall -> trip!
        typed_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.7125,
            stage2_accuracy=0.6625,
            num_options=2,
            paired_escalation_delta=-3,
            sample_count=64,
        )
        self.assertTrue(typed_profile.is_negative_transfer())

        # When Stage 2 trails marginally overall (0.84 vs 0.88) on easy items
        # but HELPS on escalated hard items (paired_escalation_delta = +1) -> allow!
        safety_profile = mdm.DomainCompetenceProfile(
            stage1_accuracy=0.88,
            stage2_accuracy=0.84,
            num_options=2,
            paired_escalation_delta=1,
            sample_count=40,
        )
        self.assertFalse(safety_profile.is_negative_transfer())

    def test_dawid_skene_weights_zero_out_chance_level_verifier(self) -> None:
        w1, w2 = mdm.compute_dawid_skene_poe_weights(
            stage1_accuracy=0.85,
            stage2_accuracy=0.20,
            num_options=4,
        )
        self.assertGreater(w1, 2.5)
        self.assertEqual(w2, 0.0)

    def test_pareto_bandit_router_selects_within_budget(self) -> None:
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


if __name__ == "__main__":
    unittest.main()

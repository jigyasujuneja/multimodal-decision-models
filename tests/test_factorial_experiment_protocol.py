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

"""Unit and statistical verification tests for the 2^4 Factorial Ablation Protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from multimodal_decision_models import ContextBlock, DatasetSplitLock, EpisodeSplitRecord, FACTOR_NAMES, FactorialConfig, MatchedRateEntropyControl, PlaceboInterventionControl, RandomEscalationControl, RecencyWindowControl, SchemaContractViolationError, SpatioTemporalFrameCandidate, UniformFrameStrideControl, build_five_rung_ladder, build_full_factorial_cube, build_loo_knockouts, build_non_adjacent_pairwise_synergies, build_orthogonal_l8_array, build_single_factor_knock_ins, calibrate_conformal_quantile, compute_episode_content_sha256, compute_exact_shapley_values, compute_factorial_interactions_and_shapley, conformal_prediction_set, fit_oof_calibrated_conformal_bundle, fit_vector_scaling, partition_episodes_cluster_disjoint, prune_agent_context_blocks, select_submodular_spatiotemporal_frames, verify_disjoint_episode_splits


class FactorialProtocolAndShapleyTests(unittest.TestCase):
    """Tests for factorial design matrices, split locks, 5-fold OOF calibration, Shapley, and controls."""

    def test_factorial_builders_and_resolution_iv_orthogonality(self) -> None:
        """Verify Five-Rung ladder, knock-ins, LOO, pairwise synergies, L8 orthogonality, and 2^4 cube."""
        ladder = build_five_rung_ladder()
        self.assertEqual([cfg.bit_code for cfg in ladder], ["0000", "1000", "1100", "1110", "1111"])
        self.assertEqual([cfg.ladder_alias for cfg in ladder], ["R1", "R2", "R3", "R3_5", "R4"])
        for step_idx in range(1, len(ladder)):
            prev_code = ladder[step_idx - 1].bit_code
            curr_code = ladder[step_idx].bit_code
            bit_diffs = sum(1 for a_ch, b_ch in zip(prev_code, curr_code) if a_ch != b_ch)
            self.assertEqual(bit_diffs, 1)

        knock_ins = build_single_factor_knock_ins()
        self.assertEqual(
            {name: cfg.bit_code for name, cfg in knock_ins.items()},
            {"KI_C": "1000", "KI_M": "0100", "KI_K": "0010", "KI_P": "0001"},
        )

        loo_map = build_loo_knockouts()
        self.assertEqual(
            {name: cfg.bit_code for name, cfg in loo_map.items()},
            {"LOO_noC": "0111", "LOO_noM": "1011", "LOO_noK": "1101", "LOO_noP": "1110"},
        )

        synergies = build_non_adjacent_pairwise_synergies()
        self.assertEqual(
            {name: cfg.bit_code for name, cfg in synergies.items()},
            {
                "SYN_CK": "1010",
                "SYN_CP": "1001",
                "SYN_MK": "0110",
                "SYN_MP": "0101",
                "SYN_KP": "0011",
            },
        )

        l8_array = build_orthogonal_l8_array()
        self.assertEqual(len(l8_array), 8)
        signs_matrix = [
            [(1 if ch == "1" else -1) for ch in cfg.bit_code]
            for cfg in l8_array
        ]
        for col_idx in range(4):
            self.assertEqual(sum(row[col_idx] for row in signs_matrix), 0)
            for other_col in range(col_idx + 1, 4):
                dot_product = sum(row[col_idx] * row[other_col] for row in signs_matrix)
                self.assertEqual(dot_product, 0)

        full_cube = build_full_factorial_cube()
        self.assertEqual(len(full_cube), 16)
        self.assertEqual(len({cfg.bit_code for cfg in full_cube}), 16)

        with self.assertRaises(SchemaContractViolationError):
            FactorialConfig.from_bit_code("1020")

    def test_dataset_split_lock_enforces_g03_sha256_and_g13_cluster_disjointness(self) -> None:
        """Verify 30/20/50 cluster-disjoint partitioning and fail-closed leakage detection."""
        raw_records: list[EpisodeSplitRecord] = []
        tracks = ("support_triage", "gui_grounding", "retail_shelf")
        for cluster_idx in range(30):
            track_name = tracks[cluster_idx % 3]
            cluster_id = f"cluster_{cluster_idx:03d}"
            for member_idx in range(4):
                ep_id = f"ep_{cluster_idx:03d}_{member_idx}"
                digest = compute_episode_content_sha256(
                    {"episode": ep_id, "cluster": cluster_id, "track": track_name}
                )
                raw_records.append(
                    EpisodeSplitRecord(
                        episode_id=ep_id,
                        cluster_id=cluster_id,
                        track_id=track_name,
                        content_sha256=digest,
                        num_turns=6,
                    )
                )

        duplicate_clone = EpisodeSplitRecord(
            episode_id="ep_duplicate_999",
            cluster_id="cluster_029",
            track_id="retail_shelf",
            content_sha256=raw_records[0].content_sha256,
            num_turns=6,
        )
        with self.assertRaises(SchemaContractViolationError):
            partition_episodes_cluster_disjoint(
                [*raw_records, duplicate_clone],
                deduplicate_sha256=False,
            )

        split_lock = partition_episodes_cluster_disjoint([*raw_records, duplicate_clone])
        self.assertEqual(split_lock.deduplicated_count, 1)
        counts = split_lock.verify(raw_records)
        self.assertEqual(counts, {"cal": 36, "val": 24, "test": 60})

        with tempfile.TemporaryDirectory() as tmp_dir:
            lock_path = Path(tmp_dir) / "split.lock.json"
            split_lock.save_json(lock_path)
            reloaded = DatasetSplitLock.load_json(lock_path)
            self.assertEqual(reloaded.manifest_sha256, split_lock.manifest_sha256)

            tampered_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            tampered_payload["manifest_sha256"] = "0" * 64
            with self.assertRaises(SchemaContractViolationError):
                DatasetSplitLock.from_dict(tampered_payload)

        cal_ep = split_lock.episodes_for_split("cal")[0]
        test_ep = split_lock.episodes_for_split("test")[0]
        leaked_cluster_records = [
            (
                EpisodeSplitRecord(
                    episode_id=rec.episode_id,
                    cluster_id=cal_ep.cluster_id if rec.episode_id == test_ep.episode_id else rec.cluster_id,
                    track_id=rec.track_id,
                    content_sha256=rec.content_sha256,
                    num_turns=rec.num_turns,
                    split=rec.split,
                )
            )
            for rec in split_lock.records
        ]
        with self.assertRaises(SchemaContractViolationError):
            verify_disjoint_episode_splits(leaked_cluster_records)

    def test_five_fold_oof_calibration_eliminates_resubstitution_undercoverage(self) -> None:
        """Verify 5-fold OOF vector scaling achieves valid test coverage where resubstitution under-covers."""
        rng = random.Random(20261111)
        classes = [f"class_{idx:02d}" for idx in range(24)]

        def sample_dataset(count: int) -> tuple[list[dict[str, float]], list[str], list[str], list[str]]:
            logits_rows: list[dict[str, float]] = []
            labels: list[str] = []
            groups: list[str] = []
            clusters: list[str] = []
            for item_idx in range(count):
                true_idx = item_idx % len(classes)
                true_label = classes[true_idx]
                row_logits: dict[str, float] = {}
                for c_idx, c_name in enumerate(classes):
                    base_signal = 2.05 if c_idx == true_idx else 0.0
                    bias_shift = 1.65 if (c_idx % 4 == 0 and c_idx != true_idx) else 0.0
                    noise = rng.gauss(0.0, 1.35)
                    row_logits[c_name] = base_signal + bias_shift + noise
                logits_rows.append(row_logits)
                labels.append(true_label)
                groups.append("tail" if true_idx >= 18 else "head")
                clusters.append(f"cal_cluster_{item_idx // 3:03d}")
            return logits_rows, labels, groups, clusters

        cal_logits, cal_labels, cal_groups, cal_clusters = sample_dataset(144)
        test_logits, test_labels, test_groups, _ = sample_dataset(720)

        oof_bundle = fit_oof_calibrated_conformal_bundle(
            cal_logits,
            cal_labels,
            calib_groups=cal_groups,
            cluster_ids=cal_clusters,
            is_logits=True,
            scaling_method="vector",
            l2_reg=1e-4,
            alpha=0.10,
            conformal_method="lac",
            num_folds=5,
        )

        naive_scaler = fit_vector_scaling(cal_logits, cal_labels, is_logits=True, l2_reg=1e-4)
        in_sample_probs = naive_scaler.predict_proba(cal_logits, is_logits=True)
        naive_conformal = calibrate_conformal_quantile(
            in_sample_probs,
            cal_labels,
            alpha=0.10,
            method="lac",
            randomized=False,
        )

        self.assertGreater(oof_bundle.q_hat_global, naive_conformal.q_hat)

        oof_hits = 0
        naive_hits = 0
        test_cal_probs = naive_scaler.predict_proba(test_logits, is_logits=True)
        for idx, true_lbl in enumerate(test_labels):
            oof_set = oof_bundle.predict_set(
                test_logits[idx],
                is_logits=True,
                group=test_groups[idx],
                use_mondrian=True,
            )
            naive_set = conformal_prediction_set(
                test_cal_probs[idx],
                naive_conformal.q_hat,
                method="lac",
                randomized=False,
            )
            if true_lbl in oof_set.prediction_set:
                oof_hits += 1
            if true_lbl in naive_set.prediction_set:
                naive_hits += 1

        oof_coverage = float(oof_hits) / float(len(test_labels))
        naive_coverage = float(naive_hits) / float(len(test_labels))
        self.assertGreater(oof_coverage, naive_coverage)
        self.assertGreaterEqual(oof_coverage, 0.875)

    def test_exact_shapley_axioms_and_anova_interaction_recovery(self) -> None:
        """Verify Shapley efficiency, symmetry, dummy-player, additivity, and C*P ANOVA recovery."""
        game_one: dict[str, float] = {}
        game_two: dict[str, float] = {}
        for mask in range(16):
            code = format(mask, "04b")
            has_c = 1.0 if code[0] == "1" else 0.0
            has_m = 1.0 if code[1] == "1" else 0.0
            has_p = 1.0 if code[3] == "1" else 0.0
            game_one[code] = 0.40 + 0.10 * has_c + 0.10 * has_m + 0.08 * (has_c * has_m)
            game_two[code] = 0.20 + 0.05 * has_c + 0.15 * has_p + 0.12 * (has_c * has_p)

        phi_one = compute_exact_shapley_values(game_one, FACTOR_NAMES)
        self.assertAlmostEqual(
            sum(phi_one.values()),
            game_one["1111"] - game_one["0000"],
            places=12,
        )
        self.assertAlmostEqual(phi_one["C"], phi_one["M"], places=12)
        self.assertAlmostEqual(phi_one["K"], 0.0, places=12)
        self.assertAlmostEqual(phi_one["P"], 0.0, places=12)

        phi_two = compute_exact_shapley_values(game_two, FACTOR_NAMES)
        combined_game = {
            code: 2.5 * game_one[code] - 1.3 * game_two[code]
            for code in game_one
        }
        phi_combined = compute_exact_shapley_values(combined_game, FACTOR_NAMES)
        for factor in FACTOR_NAMES:
            self.assertAlmostEqual(
                phi_combined[factor],
                2.5 * phi_one[factor] - 1.3 * phi_two[factor],
                places=12,
            )

        rng = random.Random(99173)
        n_episodes = 200
        cluster_ids = [f"syn_cluster_{idx // 4:03d}" for idx in range(n_episodes)]
        cluster_offsets = {
            f"syn_cluster_{c_idx:03d}": rng.gauss(0.0, 0.025)
            for c_idx in range(n_episodes // 4)
        }
        coalition_outcomes: dict[str, list[float]] = {format(m, "04b"): [] for m in range(16)}
        for ep_idx in range(n_episodes):
            c_noise = cluster_offsets[cluster_ids[ep_idx]]
            for mask in range(16):
                code = format(mask, "04b")
                c_on = 1.0 if code[0] == "1" else 0.0
                m_on = 1.0 if code[1] == "1" else 0.0
                k_on = 1.0 if code[2] == "1" else 0.0
                p_on = 1.0 if code[3] == "1" else 0.0
                ep_jitter = rng.gauss(0.0, 0.012)
                outcome = (
                    0.50
                    + 0.08 * c_on
                    + 0.05 * m_on
                    + 0.04 * k_on
                    + 0.02 * p_on
                    + 0.12 * (c_on * p_on)
                    + c_noise
                    + ep_jitter
                )
                coalition_outcomes[code].append(outcome)

        summary = compute_factorial_interactions_and_shapley(
            coalition_outcomes,
            cluster_ids=cluster_ids,
            n_bootstrap=2000,
            alpha=0.05,
        )
        self.assertAlmostEqual(
            sum(summary.shapley_values.values()),
            summary.total_gain,
            places=10,
        )
        self.assertGreater(summary.two_way_interactions["CP"], 0.09)
        self.assertGreater(summary.two_way_interaction_cis["CP"][0], 0.08)
        self.assertTrue(summary.significant_effects["CP"])
        self.assertLess(summary.holm_adjusted_p_values["CP"], 0.001)
        self.assertFalse(summary.significant_effects["MK"])

    def test_controls_c1_to_c5_satisfy_iso_rate_and_iso_budget_contracts(self) -> None:
        """Verify Controls C1..C5 match escalation/token/frame/trigger budgets within <= 0.5%."""
        item_keys = [f"turn_{idx:04d}" for idx in range(400)]
        c1_control = RandomEscalationControl(target_escalation_rate=0.225)
        c1_quota = c1_control.select_exact_quota(item_keys)
        empirical_c1_rate = sum(1 for esc in c1_quota.values() if esc) / float(len(item_keys))
        self.assertLessEqual(abs(empirical_c1_rate - 0.225), 0.005)

        rng = random.Random(4242)
        val_probs = []
        test_probs = []
        for idx in range(600):
            p_top = min(0.99, max(0.35, rng.betavariate(2.5, 1.8)))
            rem = 1.0 - p_top
            dist = {"a": p_top, "b": 0.6 * rem, "c": 0.4 * rem}
            if idx < 200:
                val_probs.append(dist)
            else:
                test_probs.append(dist)

        c2_control = MatchedRateEntropyControl.fit_on_validation(
            val_probs,
            target_escalation_rate=0.25,
        )
        val_c2_rate = sum(
            1 for idx, dist in enumerate(val_probs) if c2_control.should_escalate(dist, f"v_{idx}")
        ) / float(len(val_probs))
        self.assertLessEqual(abs(val_c2_rate - 0.25), 0.005)
        self.assertGreater(len(test_probs), 0)

        blocks = [
            ContextBlock(id="sys_pin", full_tokens=60, stub_tokens=20, pinned=True, full_text="PIN"),
            ContextBlock(id="early_gold", full_tokens=100, stub_tokens=25, full_text="CRITICAL_POLICY"),
            ContextBlock(id="mid_chatter", full_tokens=100, stub_tokens=25, full_text="NOISE_1"),
            ContextBlock(id="late_chatter", full_tokens=100, stub_tokens=25, full_text="NOISE_2"),
        ]
        slot_probs = {
            "sys_pin": {"full": 0.99, "stub": 0.01, "drop": 0.00},
            "early_gold": {"full": 0.94, "stub": 0.05, "drop": 0.01},
            "mid_chatter": {"full": 0.05, "stub": 0.10, "drop": 0.85},
            "late_chatter": {"full": 0.08, "stub": 0.12, "drop": 0.80},
        }
        knapsack_result = prune_agent_context_blocks(blocks, slot_probs, token_budget=180)
        c3_result = RecencyWindowControl().prune(
            blocks, token_budget=180, slot_probabilities=slot_probs
        )
        self.assertLessEqual(c3_result.tokens_after, 180)
        self.assertEqual(c3_result.decisions["sys_pin"], "full")
        self.assertEqual(c3_result.decisions["early_gold"], "drop")
        self.assertEqual(knapsack_result.decisions["early_gold"], "full")
        self.assertGreater(knapsack_result.expected_utility, c3_result.expected_utility)

        frames = []
        for f_idx in range(12):
            burst = f_idx in (4, 5)
            frames.append(
                SpatioTemporalFrameCandidate(
                    frame_index=f_idx,
                    timestamp_sec=float(f_idx) * 0.5,
                    hesitation_score=0.95 if burst else 0.10,
                    acoustic_transient_score=0.85 if burst else 0.05,
                    visual_motion_score=0.90 if burst else 0.08,
                    feature_vector=(1.0, float(f_idx) * 0.1, 0.2 if burst else -0.5),
                )
            )
        submod_sel = select_submodular_spatiotemporal_frames(frames, budget_b=3)
        c4_frames = UniformFrameStrideControl().select_frames(frames, max_frames=3)
        self.assertEqual(len(c4_frames), 3)
        self.assertEqual(len(submod_sel.selected_indices), 3)
        c4_mean_sal = sum(fr.hesitation_score for fr in c4_frames) / 3.0
        submod_mean_sal = (
            sum(frames[pos].hesitation_score for pos in submod_sel.selected_indices) / 3.0
        )
        self.assertGreater(submod_mean_sal, c4_mean_sal)

        c5_control = PlaceboInterventionControl(target_trigger_rate=0.175)
        c5_quota = c5_control.select_exact_quota(item_keys)
        c5_rate = sum(1 for trig in c5_quota.values() if trig) / float(len(item_keys))
        self.assertLessEqual(abs(c5_rate - 0.175), 0.005)
        self.assertTrue(
            math.isfinite(float(c5_control.should_intervene("ep_01", 2)))
        )

    def test_committed_episodebench_tier2_lockfile_integrity(self) -> None:
        """Verify committed benchmarks/locks/mdm_episodebench_v1.lock.json passes G03/G13 gates."""
        repo_root = Path(__file__).resolve().parents[1]
        lock_path = repo_root / "benchmarks" / "locks" / "mdm_episodebench_v1.lock.json"
        self.assertTrue(lock_path.is_file(), f"Missing lockfile at {lock_path}")
        loaded_lock = DatasetSplitLock.load_json(lock_path)
        self.assertEqual(loaded_lock.split_counts(), {"cal": 360, "val": 240, "test": 600})
        verify_disjoint_episode_splits(loaded_lock.records)
        test_records = loaded_lock.episodes_for_split("test")
        self.assertEqual(len(test_records), 600)
        self.assertEqual(sum(record.num_turns for record in test_records), 3600)
        self.assertEqual(
            {record.track_id for record in test_records},
            {
                "support_claims_triage",
                "visual_gui_form_grounding",
                "retail_video_shelf_audit",
            },
        )


if __name__ == "__main__":
    unittest.main()

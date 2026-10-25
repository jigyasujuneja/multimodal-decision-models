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

"""Unit and integration tests for v0.4.0 conformal, calibration, visual compiler, index, pruner, and MCP."""

from __future__ import annotations

import base64
import http.server
import io
import itertools
import json
import math
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import zlib
from typing import Any

import multimodal_decision_models as mdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _png_chunk_raw(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def _encode_png_with_explicit_filters(
    width: int,
    height: int,
    color_type: int,
    raw_rows: list[bytes],
    filter_types: list[int],
) -> bytes:
    bpp = {0: 1, 2: 3, 6: 4}[color_type]
    row_bytes = width * bpp
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    prev = bytearray(row_bytes)
    scan = bytearray()

    for y in range(height):
        cur = raw_rows[y]
        ftype = filter_types[y]
        scan.append(ftype)
        for i in range(row_bytes):
            x_val = cur[i]
            a_val = cur[i - bpp] if i >= bpp else 0
            b_val = prev[i]
            c_val = prev[i - bpp] if i >= bpp else 0
            if ftype == 0:
                filt = x_val
            elif ftype == 1:
                filt = (x_val - a_val) & 0xFF
            elif ftype == 2:
                filt = (x_val - b_val) & 0xFF
            elif ftype == 3:
                filt = (x_val - ((a_val + b_val) >> 1)) & 0xFF
            elif ftype == 4:
                p = a_val + b_val - c_val
                pa = abs(p - a_val)
                pb = abs(p - b_val)
                pc = abs(p - c_val)
                pred = a_val if (pa <= pb and pa <= pc) else (b_val if pb <= pc else c_val)
                filt = (x_val - pred) & 0xFF
            else:
                raise ValueError(f"bad filter {ftype}")
            scan.append(filt)
        prev = bytearray(cur)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk_raw(b"IHDR", ihdr)
        + _png_chunk_raw(b"IDAT", zlib.compress(bytes(scan)))
        + _png_chunk_raw(b"IEND", b"")
    )


def _make_solid_buffer(width: int, height: int, rgb_triplet: tuple[int, int, int]) -> mdm.ImageBuffer:
    pixel = bytes(rgb_triplet)
    return mdm.ImageBuffer(width, height, pixel * (width * height))


class V040ConformalVisualAndMcpTest(unittest.TestCase):
    def test_v040_version_and_public_contract_exports(self) -> None:
        self.assertEqual(mdm.__version__, "0.4.0")
        self.assertEqual(mdm.MCP_SERVER_NAME, "dgem-catalog")
        expected_exports = [
            "BlackBoxSampledBackend",
            "ConformalCalibrationResult",
            "ConformalSetResult",
            "calibrate_conformal_quantile",
            "conformal_prediction_set",
            "TemperatureScalingModel",
            "VectorScalingModel",
            "fit_temperature_scaling",
            "fit_vector_scaling",
            "MurphyDecomposition",
            "brier_murphy_decomposition",
            "adaptive_calibration_error",
            "macro_f1_score",
            "ndcg_at_k",
            "ImageBuffer",
            "CompiledContactSheet",
            "encode_png",
            "decode_png",
            "encode_ppm",
            "decode_ppm",
            "shades_of_gray_white_balance",
            "inner_crop_cie_lab_similarity",
            "apply_residual_mlp_adapter",
            "extract_inter_candidate_saliency_band",
            "compile_dual_buffer_contact_sheet",
            "VertexMultimodalEmbeddingAdapter",
            "MultimodalEmbeddingIndex",
            "ContextBlock",
            "ContextPruneResult",
            "prune_agent_context_blocks",
            "build_context_pruning_questions",
            "serve_stdio",
            "handle_jsonrpc_message",
        ]
        for sym in expected_exports:
            self.assertTrue(hasattr(mdm, sym), f"missing export {sym}")
            self.assertIn(sym, mdm.__all__)

    def test_blackbox_sampled_backend_dirichlet_epistemic_and_http(self) -> None:
        question = mdm.DecisionQuestion(
            type="choice",
            instructions="Select the matching SKU option.",
            criteria={
                "sku_100": "Brand Alpha 250ml",
                "sku_200": "Brand Alpha 500ml",
                "sku_300": "Brand Beta 250ml",
                "none_of_the_above": "Unlisted item",
            },
            order=["sku_100", "sku_200", "sku_300", "none_of_the_above"],
        )
        questions = {"slot_sku": question}

        unanimous_5 = mdm.BlackBoxSampledBackend(
            "sampled_m5",
            lambda state, qs, imgs, idx: "```json\n{\"evidence_read\": \"250ml printed\", \"slot_sku\": \"A\"}\n```",
            num_samples=5,
            prior_alpha=0.5,
        )
        dists_5 = unanimous_5.score_questions({"crop": "c1"}, questions)
        probs_5 = dists_5["slot_sku"]
        # With M=5 unanimous votes on K=4 options and alpha_0=0.5: (5 + 0.5) / (5 + 2.0) = 5.5 / 7.0
        self.assertAlmostEqual(probs_5["sku_100"], 5.5 / 7.0, places=9)
        self.assertAlmostEqual(probs_5["sku_200"], 0.5 / 7.0, places=9)
        self.assertGreater(mdm.normalized_entropy(probs_5), 0.05)
        self.assertTrue(math.isfinite(mdm.negative_log_likelihood([probs_5], ["sku_200"])))
        self.assertEqual(unanimous_5.last_evidence["slot_sku"], "250ml printed")
        var_m5 = unanimous_5.last_epistemic_variance["slot_sku"]
        self.assertGreater(var_m5, 0.0)

        unanimous_1 = mdm.BlackBoxSampledBackend(
            "sampled_m1",
            lambda state, qs: {"slot_sku": "sku_100"},
            num_samples=1,
            prior_alpha=0.5,
        )
        unanimous_20 = mdm.BlackBoxSampledBackend(
            "sampled_m20",
            lambda state, qs: {"slot_sku": "sku_100"},
            num_samples=20,
            prior_alpha=0.5,
        )
        unanimous_1.score_questions({}, questions)
        unanimous_20.score_questions({}, questions)
        self.assertGreater(unanimous_1.last_epistemic_variance["slot_sku"], var_m5)
        self.assertGreater(var_m5, unanimous_20.last_epistemic_variance["slot_sku"])

        # Verbalized vote counts + explicit abstain alias routing; unrecognized tokens must fail closed
        verbalized = mdm.BlackBoxSampledBackend(
            "verbalized_votes",
            lambda state, qs: {"votes": {"sku_200": 4, "oos": 1}, "evidence": "500ml bottle"},
            num_samples=5,
            prior_alpha=0.5,
        )
        v_probs = verbalized.score_questions({}, questions)["slot_sku"]
        self.assertAlmostEqual(v_probs["sku_200"], 4.5 / 7.0, places=9)
        self.assertAlmostEqual(v_probs["none_of_the_above"], 1.5 / 7.0, places=9)

        hallucinated = mdm.BlackBoxSampledBackend(
            "verbalized_hallucinated",
            lambda state, qs: {"votes": {"sku_200": 4, "unknown_hallucination": 1}, "evidence": "500ml bottle"},
            num_samples=5,
            prior_alpha=0.5,
        )
        with self.assertRaises(mdm.SchemaContractViolationError):
            hallucinated.score_questions({}, questions)

        # Live local ThreadingHTTPServer /v1/chat/completions without top_logprobs
        draw_seq = ["B", "B", "A", "B"]
        lock = threading.Lock()
        counter = {"idx": 0}

        class _NoLogprobChatHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                with lock:
                    pick = draw_seq[counter["idx"] % len(draw_seq)]
                    counter["idx"] += 1
                resp = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"evidence_read": "tall bottle", "slot_sku": pick}),
                            }
                        }
                    ]
                }
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoLogprobChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            http_backend = mdm.BlackBoxSampledBackend(
                "http_sampled",
                endpoint_url=f"http://{host}:{port}",
                num_samples=4,
                prior_alpha=0.5,
                max_concurrency=2,
            )
            http_probs = http_backend.score_questions({"scene": "aisle_2"}, questions)["slot_sku"]
            self.assertAlmostEqual(http_probs["sku_200"], 3.5 / 6.0, places=9)
            self.assertAlmostEqual(http_probs["sku_100"], 1.5 / 6.0, places=9)
            self.assertEqual(http_backend.last_evidence["slot_sku"], "tall bottle")
        finally:
            server.shutdown()
            server.server_close()

    def test_split_conformal_and_raps_exchangeable_coverage_and_edge_cases(self) -> None:
        rng = random.Random(20260417)
        classes = [f"cls_{i:02d}" for i in range(16)]
        n_calib = 400
        n_test = 1200

        all_probs: list[dict[str, float]] = []
        all_labels: list[str] = []
        for _ in range(n_calib + n_test):
            logits = [rng.gauss(0.0, 1.35) for _ in classes]
            max_l = max(logits)
            exps = [math.exp(z - max_l) for z in logits]
            tot = sum(exps)
            true_p = [e / tot for e in exps]
            # Sample true label from true_p, then perturb model probs slightly (imperfect model)
            u_draw = rng.random()
            cum = 0.0
            gold = classes[-1]
            for c_name, p_val in zip(classes, true_p):
                cum += p_val
                if u_draw <= cum:
                    gold = c_name
                    break
            noisy_logits = [z * 1.3 + rng.gauss(0.0, 0.25) for z in logits]
            m_nl = max(noisy_logits)
            n_exps = [math.exp(z - m_nl) for z in noisy_logits]
            n_tot = sum(n_exps)
            all_probs.append({c_name: e / n_tot for c_name, e in zip(classes, n_exps)})
            all_labels.append(gold)

        calib_probs = all_probs[:n_calib]
        calib_labels = all_labels[:n_calib]
        test_probs = all_probs[n_calib:]
        test_labels = all_labels[n_calib:]

        mean_sizes: dict[str, float] = {}
        for alpha in (0.05, 0.10):
            for method in ("lac", "aps", "raps"):
                for randomized in (False, True):
                    cal = mdm.calibrate_conformal_quantile(
                        calib_probs,
                        calib_labels,
                        alpha=alpha,
                        method=method,
                        lam_reg=0.02,
                        k_reg=2,
                        randomized=randomized,
                        seed=101,
                    )
                    hits = 0
                    size_sum = 0
                    for idx, (tp, tg) in enumerate(zip(test_probs, test_labels)):
                        cset = mdm.conformal_prediction_set(
                            tp,
                            cal.q_hat,
                            method=method,
                            lam_reg=0.02,
                            k_reg=2,
                            randomized=randomized,
                            seed=10000 + idx,
                        )
                        size_sum += len(cset.prediction_set)
                        if tg in cset.prediction_set:
                            hits += 1
                    emp_cov = hits / float(n_test)
                    self.assertGreaterEqual(
                        emp_cov,
                        1.0 - alpha - 0.025,
                        f"coverage {emp_cov:.4f} below target for alpha={alpha}, method={method}, rand={randomized}",
                    )
                    if alpha == 0.10 and not randomized:
                        mean_sizes[method] = size_sum / float(n_test)

        # RAPS regularization shrinks tail prediction set sizes on wide shortlists with diffuse OOD/ambiguous tails
        tail_probs: list[dict[str, float]] = []
        tail_labels: list[str] = []
        for idx_t in range(600):
            if idx_t % 12 == 0:
                rot = idx_t % len(classes)
                raw_w = [1.0 + 0.02 * float(len(classes) - 1 - j) for j in range(len(classes))]
                w_sum = sum(raw_w)
                tail_probs.append(
                    {classes[(rot + j) % len(classes)]: raw_w[j] / w_sum for j in range(len(classes))}
                )
                tail_labels.append(classes[(rot + (idx_t // 12)) % len(classes)])
            else:
                w0 = classes[idx_t % len(classes)]
                w1 = classes[(idx_t + 1) % len(classes)]
                rem_p = 0.08 / float(len(classes) - 2)
                dist_t = {c_name: rem_p for c_name in classes}
                dist_t[w0] = 0.72
                dist_t[w1] = 0.20
                tail_probs.append(dist_t)
                tail_labels.append(w0 if (idx_t % 5 != 0) else w1)

        cal_aps = mdm.calibrate_conformal_quantile(tail_probs[:250], tail_labels[:250], alpha=0.10, method="aps")
        cal_raps = mdm.calibrate_conformal_quantile(
            tail_probs[:250], tail_labels[:250], alpha=0.10, method="raps", lam_reg=0.05, k_reg=2
        )
        aps_tail_sizes = [
            len(mdm.conformal_prediction_set(tp, cal_aps.q_hat, method="aps").prediction_set)
            for tp in tail_probs[250:]
        ]
        raps_tail_sizes = [
            len(
                mdm.conformal_prediction_set(
                    tp, cal_raps.q_hat, method="raps", lam_reg=0.05, k_reg=2
                ).prediction_set
            )
            for tp in tail_probs[250:]
        ]
        self.assertLess(sum(raps_tail_sizes), sum(aps_tail_sizes))

        # Finite-sample overflow: n=4, alpha=0.05 -> ceil(5 * 0.95) = 5 > 4 -> q_hat = +inf
        tiny_cal = mdm.calibrate_conformal_quantile(
            calib_probs[:4],
            calib_labels[:4],
            alpha=0.05,
            method="raps",
        )
        self.assertTrue(math.isinf(tiny_cal.q_hat))
        full_set = mdm.conformal_prediction_set({"a": 0.7, "b": 0.2, "none_of_the_above": 0.1}, tiny_cal.q_hat)
        self.assertEqual(len(full_set.prediction_set), 3)
        self.assertTrue(full_set.contains_abstain)
        self.assertTrue(full_set.should_escalate)

        # Singleton abstain option also triggers should_escalate
        abstain_singleton = mdm.conformal_prediction_set(
            {"none_of_the_above": 0.96, "sku_a": 0.04},
            q_hat=0.10,
            method="lac",
        )
        self.assertTrue(abstain_singleton.singleton)
        self.assertTrue(abstain_singleton.contains_abstain)
        self.assertTrue(abstain_singleton.should_escalate)

    def test_pre_fusion_conformal_cascade_gate_and_cache_roundtrip(self) -> None:
        q = mdm.DecisionQuestion(
            type="choice",
            instructions="Pick exact catalog variant.",
            criteria={"sku_1": "Variant 1", "sku_2": "Variant 2"},
            order=["sku_1", "sku_2"],
        )
        questions = {"variant": q}

        stage1 = mdm.MockCallableBackend(
            "fast_stage1",
            lambda st, qs, imgs: {"variant": {"sku_1": 0.52, "sku_2": 0.48}},
        )
        stage2 = mdm.BlackBoxSampledBackend(
            "frontier_stage2",
            lambda st, qs: {"variant": "sku_2", "evidence_read": "verified variant 2"},
            num_samples=6,
            prior_alpha=0.5,
        )

        # Baseline v0.3.0 cache key parity when conformal_q_hat is None
        orch_no_conf = mdm.CascadeDecisionOrchestrator(stage1_backend=stage1, stage2_backend=stage2)
        res_no_conf = orch_no_conf.evaluate({"id": 1}, questions)
        expected_v030_key = mdm.compute_decision_cache_key(
            "fast_stage1->frontier_stage2",
            {"id": 1},
            questions,
            None,
            extra_config={
                "hesitation_threshold": mdm.DEFAULT_HESITATION_THRESHOLD,
                "prior_weight": 0.0,
                "prior_gate_mode": "hesitant",
                "temperature": 1.0,
                "shortlist_top_k": 0,
                "shortlist_pairwise_diff": False,
                "evidence_prompt": "",
                "enable_stage2a_fast_exit": False,
                "sister_pack_collision": False,
                "decoded_pack_size": None,
            },
        )
        self.assertEqual(res_no_conf.cache_key, expected_v030_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "decisions.jsonl"
            cache = mdm.ContentAddressedDecisionCache(persist_path=cache_path)

            # Set hesitation_threshold=0.999 so entropy alone would NOT escalate, and pass an extreme
            # prior for sku_1 so post-fusion distribution would look 99.98% confident. Pre-fusion
            # conformal gate MUST still see raw_dist = {0.52, 0.48}, produce |C_alpha|=2, and escalate!
            orch = mdm.CascadeDecisionOrchestrator(
                stage1_backend=stage1,
                stage2_backend=stage2,
                hesitation_threshold=0.999,
                prior_weight=1.0,
                prior_gate_mode="always",
                conformal_q_hat=0.55,
                conformal_method="aps",
                response_cache=cache,
            )
            res = orch.evaluate(
                {"id": 7},
                questions,
                prior_logits={"variant": {"sku_1": 9.0, "sku_2": 0.0}},
            )
            self.assertTrue(res.escalated)
            self.assertTrue(any("conformal_set_size=2" in r for r in res.escalation_reasons))
            slot = res.slots["variant"]
            self.assertEqual(slot.predicted, "sku_2")
            self.assertEqual(slot.conformal_set, ["sku_1", "sku_2"])
            self.assertGreater(slot.epistemic_variance, 0.0)

            reloaded_cache = mdm.ContentAddressedDecisionCache(persist_path=cache_path)
            cached_res = reloaded_cache.get(res.cache_key)
            self.assertIsNotNone(cached_res)
            assert cached_res is not None
            self.assertTrue(cached_res.cache_hit)
            self.assertEqual(cached_res.slots["variant"].conformal_set, ["sku_1", "sku_2"])
            self.assertAlmostEqual(
                cached_res.slots["variant"].epistemic_variance,
                slot.epistemic_variance,
                places=9,
            )

    def test_temperature_and_vector_scaling_reduce_heldout_nll_and_ece(self) -> None:
        rng = random.Random(9001)
        classes = ["c0", "c1", "c2", "c3", "c4"]
        skew_gain = {"c0": 3.4, "c1": 2.9, "c2": 3.8, "c3": 2.5, "c4": 3.2}
        skew_bias = {"c0": 0.75, "c1": -0.45, "c2": 0.55, "c3": -0.35, "c4": -0.50}

        def _sample_split(n_rows: int) -> tuple[list[dict[str, float]], list[dict[str, float]], list[str]]:
            logits_out: list[dict[str, float]] = []
            probs_out: list[dict[str, float]] = []
            labels_out: list[str] = []
            for _ in range(n_rows):
                true_z = {c: rng.gauss(0.0, 1.1) for c in classes}
                m_true = max(true_z.values())
                exps_true = {c: math.exp(v - m_true) for c, v in true_z.items()}
                tot_true = sum(exps_true.values())
                u = rng.random()
                cum = 0.0
                gold = classes[-1]
                for c in classes:
                    cum += exps_true[c] / tot_true
                    if u <= cum:
                        gold = c
                        break
                raw_z = {c: skew_gain[c] * true_z[c] + skew_bias[c] for c in classes}
                m_raw = max(raw_z.values())
                exps_raw = {c: math.exp(v - m_raw) for c, v in raw_z.items()}
                tot_raw = sum(exps_raw.values())
                logits_out.append(raw_z)
                probs_out.append({c: exps_raw[c] / tot_raw for c in classes})
                labels_out.append(gold)
            return logits_out, probs_out, labels_out

        fit_logits, fit_probs, fit_labels = _sample_split(320)
        eval_logits, eval_probs, eval_labels = _sample_split(450)

        raw_nll = mdm.negative_log_likelihood(eval_probs, eval_labels)
        raw_ece = mdm.expected_calibration_error(eval_probs, eval_labels)

        temp_model = mdm.fit_temperature_scaling(fit_logits, fit_labels, is_logits=True, l2_reg=1e-3)
        self.assertGreater(temp_model.temperature, 1.5)
        temp_probs = temp_model.predict_proba(eval_logits, is_logits=True)
        temp_nll = mdm.negative_log_likelihood(temp_probs, eval_labels)
        temp_ece = mdm.expected_calibration_error(temp_probs, eval_labels)
        self.assertLess(temp_nll, raw_nll)
        self.assertLess(temp_ece, raw_ece)

        vec_model = mdm.fit_vector_scaling(fit_logits, fit_labels, is_logits=True, l2_reg=0.05)
        vec_probs = vec_model.predict_proba(eval_logits, is_logits=True)
        vec_nll = mdm.negative_log_likelihood(vec_probs, eval_labels)
        vec_ece = mdm.expected_calibration_error(vec_probs, eval_labels)
        self.assertLess(vec_nll, raw_nll)
        self.assertLess(vec_ece, raw_ece)
        self.assertLess(vec_nll, temp_nll)

        # Probability-input mode (`is_logits=False`)
        prob_temp_model = mdm.fit_temperature_scaling(fit_probs, fit_labels, is_logits=False)
        prob_cal = prob_temp_model.predict_proba(eval_probs, is_logits=False)
        self.assertLess(mdm.negative_log_likelihood(prob_cal, eval_labels), raw_nll)

    def test_murphy_brier_decomposition_ace_macro_f1_and_ndcg(self) -> None:
        # On quantized forecasts whose values lie at bin centers, Brier == REL - RES + UNC identically
        quantized_preds = [
            {"a": 0.1, "b": 0.9},
            {"a": 0.1, "b": 0.9},
            {"a": 0.8, "b": 0.2},
            {"a": 0.8, "b": 0.2},
            {"a": 0.5, "b": 0.5},
            {"a": 0.5, "b": 0.5},
        ]
        quantized_golds = ["b", "b", "a", "b", "a", "b"]
        decomp = mdm.brier_murphy_decomposition(quantized_preds, quantized_golds, num_bins=10)
        self.assertGreaterEqual(decomp.reliability, 0.0)
        self.assertGreaterEqual(decomp.resolution, 0.0)
        self.assertGreaterEqual(decomp.uncertainty, 0.0)
        self.assertAlmostEqual(
            decomp.reliability - decomp.resolution + decomp.uncertainty,
            decomp.brier,
            places=9,
        )

        # Calibrated vs overconfident ACE
        calibrated_preds = [{"x": 0.75, "y": 0.25}] * 40 + [{"x": 0.25, "y": 0.75}] * 40
        calibrated_golds = (["x"] * 30 + ["y"] * 10) + (["x"] * 10 + ["y"] * 30)
        overconf_preds = [{"x": 0.99, "y": 0.01}] * 40 + [{"x": 0.01, "y": 0.99}] * 40
        ace_cal = mdm.adaptive_calibration_error(calibrated_preds, calibrated_golds, num_bins=2)
        ace_over = mdm.adaptive_calibration_error(overconf_preds, calibrated_golds, num_bins=2)
        self.assertAlmostEqual(ace_cal, 0.0, places=9)
        self.assertGreater(ace_over, 0.20)

        # Macro-F1
        f1 = mdm.macro_f1_score(
            ["cat", "cat", "dog", "bird", "cat"],
            ["cat", "dog", "dog", "bird", "bird"],
        )
        # cat: TP=1, FP=2, FN=0 -> 2/4=0.5; dog: TP=1, FP=0, FN=1 -> 2/3; bird: TP=1, FP=0, FN=1 -> 2/3
        self.assertAlmostEqual(f1, (0.5 + (2.0 / 3.0) + (2.0 / 3.0)) / 3.0, places=9)

        # nDCG@k + duplicate ID guard
        relevance = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
        self.assertAlmostEqual(mdm.ndcg_at_k(["d1", "d2", "d3"], relevance, k=3), 1.0, places=9)
        dup_score = mdm.ndcg_at_k(["d1", "d1", "d1"], relevance, k=3)
        self.assertLess(dup_score, 1.0)
        self.assertGreater(dup_score, 0.4)

    def test_visual_compiler_png_filters_crc_ppm_dual_buffer_and_saliency_iou(self) -> None:
        # 1. Explicit PNG filter types 0..4 on RGB (color_type=2)
        w, h = 6, 5
        rgb_rows = [
            bytes(((x * 37 + y * 53 + c * 19) & 0xFF) for x in range(w) for c in range(3))
            for y in range(h)
        ]
        png_all_filters = _encode_png_with_explicit_filters(w, h, 2, rgb_rows, [0, 1, 2, 3, 4])
        decoded_rgb = mdm.decode_png(png_all_filters)
        self.assertEqual((decoded_rgb.width, decoded_rgb.height), (w, h))
        self.assertEqual(decoded_rgb.rgb, b"".join(rgb_rows))

        # Round-trip encode_png -> decode_png
        rt_png = mdm.encode_png(decoded_rgb)
        self.assertTrue(rt_png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", rt_png)
        self.assertIn(b"IDAT", rt_png)
        self.assertIn(b"IEND", rt_png)
        self.assertEqual(mdm.decode_png(rt_png).rgb, decoded_rgb.rgb)

        # Corrupt CRC32 must fail closed with ValueError
        corrupted = bytearray(rt_png)
        corrupted[-6] ^= 0x01
        with self.assertRaises(ValueError):
            mdm.decode_png(bytes(corrupted))

        # RGBA (color_type=6) and Grayscale (color_type=0)
        rgba_rows = [bytes([200, 100, 50, 255, 0, 0, 0, 0]) for _ in range(2)]
        rgba_buf = mdm.decode_png(_encode_png_with_explicit_filters(2, 2, 6, rgba_rows, [0, 4]))
        self.assertEqual(rgba_buf.pixel(0, 0), (200, 100, 50))
        self.assertEqual(rgba_buf.pixel(1, 0), (255, 255, 255))

        gray_rows = [bytes([42, 180]) for _ in range(2)]
        gray_buf = mdm.decode_png(_encode_png_with_explicit_filters(2, 2, 0, gray_rows, [0, 2]))
        self.assertEqual(gray_buf.pixel(0, 1), (42, 42, 42))
        self.assertEqual(gray_buf.pixel(1, 1), (180, 180, 180))

        # 2. PPM P6 round-trip and P3 ASCII with comments
        ppm_bytes = mdm.encode_ppm(decoded_rgb)
        self.assertEqual(mdm.decode_ppm(ppm_bytes).rgb, decoded_rgb.rgb)
        p3_ascii = b"P3\n# synthetic comment\n2 1\n255\n10 20 30  200 210 220\n"
        p3_buf = mdm.decode_ppm(p3_ascii)
        self.assertEqual(p3_buf.pixel(0, 0), (10, 20, 30))
        self.assertEqual(p3_buf.pixel(1, 0), (200, 210, 220))

        # 3. Shades-of-Gray white balance + all-black image safety + CIE Lab inner-80% crop
        black_buf = _make_solid_buffer(8, 8, (0, 0, 0))
        self.assertEqual(mdm.shades_of_gray_white_balance(black_buf).rgb, black_buf.rgb)

        warm_cast = _make_solid_buffer(12, 12, (220, 160, 110))
        wb_neutral = mdm.shades_of_gray_white_balance(warm_cast, p=6.0)
        r_px, g_px, b_px = wb_neutral.pixel(6, 6)
        self.assertLessEqual(max(r_px, g_px, b_px) - min(r_px, g_px, b_px), 3)

        # Inner-80% CIE Lab ignores outer 10% border clutter
        clean_blue = _make_solid_buffer(20, 20, (30, 80, 210))
        noisy_border_rgb = bytearray(clean_blue.rgb)
        for y in range(20):
            for x in range(20):
                if x < 2 or x >= 18 or y < 2 or y >= 18:
                    off = (y * 20 + x) * 3
                    noisy_border_rgb[off : off + 3] = bytes((250, 240, 10))
        noisy_border_blue = mdm.ImageBuffer(20, 20, bytes(noisy_border_rgb))
        sim_same, de_same = mdm.inner_crop_cie_lab_similarity(clean_blue, noisy_border_blue, crop_fraction=0.80)
        self.assertAlmostEqual(de_same, 0.0, places=9)
        self.assertAlmostEqual(sim_same, 1.0, places=9)

        red_pack = _make_solid_buffer(20, 20, (220, 25, 25))
        sim_diff, de_diff = mdm.inner_crop_cie_lab_similarity(clean_blue, red_pack, crop_fraction=0.80)
        self.assertGreater(de_diff, 50.0)
        self.assertLess(sim_diff, 0.20)

        # Residual MLP adapter
        adapted = mdm.apply_residual_mlp_adapter(
            [1.0, 0.0, 0.0, 0.0],
            w1=[[0.5, -0.2, 0.1, 0.0], [0.0, 0.8, -0.4, 0.2]],
            w2=[[0.3, 0.1], [0.4, -0.2], [-0.1, 0.5], [0.2, 0.2]],
            alpha=0.35,
        )
        self.assertEqual(len(adapted), 4)
        self.assertAlmostEqual(sum(v * v for v in adapted), 1.0, places=9)

        # 4. Saliency band extraction IoU >= 0.80 across synthetic horizontal variant bands
        band_specs = [(10, 24), (22, 38), (36, 52), (4, 18)]
        img_w, img_h = 32, 60
        for gt_y0, gt_y1 in band_specs:
            cands: list[mdm.ImageBuffer] = []
            for k_idx, badge_rgb in enumerate([(235, 30, 30), (30, 220, 60), (30, 60, 240)]):
                buf_bytes = bytearray(img_w * img_h * 3)
                for y in range(img_h):
                    for x in range(img_w):
                        off = (y * img_w + x) * 3
                        if gt_y0 <= y < gt_y1:
                            # Variant-specific high-contrast stripe pattern inside ground-truth band
                            stripe = badge_rgb if ((x + k_idx * 3) % 6 < 3) else (245, 245, 245)
                            buf_bytes[off : off + 3] = bytes(stripe)
                        else:
                            # Shared packaging body across siblings
                            base_c = (120 + (x % 5), 130 + (y % 4), 140)
                            buf_bytes[off : off + 3] = bytes(base_c)
                cands.append(mdm.ImageBuffer(img_w, img_h, bytes(buf_bytes)))

            pred_y0, pred_y1, variances = mdm.extract_inter_candidate_saliency_band(cands)
            self.assertEqual(len(variances), img_h)
            inter = max(0, min(gt_y1, pred_y1) - max(gt_y0, pred_y0))
            union = max(gt_y1, pred_y1) - min(gt_y0, pred_y0)
            iou_1d = float(inter) / float(max(1, union))
            self.assertGreaterEqual(
                iou_1d,
                0.80,
                f"saliency band IoU {iou_1d:.3f} < 0.80 for gt=({gt_y0}, {gt_y1}), pred=({pred_y0}, {pred_y1})",
            )

        # 5. Dual-buffer contact sheet compilation + Stage-2A unsharpened invariance + curved unwarp
        query_crop = cands[1]
        sheet_flat = mdm.compile_dual_buffer_contact_sheet(
            query_crop,
            cands,
            candidate_metadata=[
                {"title": "Alpha Shampoo 180ml Bottle"},
                {"title": "Alpha Shampoo 340ml Bottle"},
                {"title": "Alpha Shampoo 650ml Pump"},
            ],
            curved_container=False,
        )
        sheet_curved = mdm.compile_dual_buffer_contact_sheet(
            query_crop,
            cands,
            curved_container=True,
        )
        decoded_flat = mdm.decode_png(sheet_flat.png_bytes)
        decoded_curved = mdm.decode_png(sheet_curved.png_bytes)
        self.assertGreater(decoded_curved.width, decoded_flat.width)
        self.assertEqual(sheet_flat.panel_labels, ["0", "1", "2", "3"])
        self.assertEqual(len(sheet_flat.diff_hints), 3)
        self.assertIn("340ml", sheet_flat.diff_hints[1])

        # Stage-2A buffer must equal unsharpened Shades-of-Gray white balance of query
        expected_s2a = mdm.shades_of_gray_white_balance(query_crop, p=6.0)
        self.assertEqual(sheet_flat.stage2a_query_buffer.rgb, expected_s2a.rgb)

    def test_multimodal_embedding_index_vertex_adapter_and_context_pruner(self) -> None:
        # 1. VertexMultimodalEmbeddingAdapter wire payload + mock HTTP embed()
        sample_png = mdm.encode_png(_make_solid_buffer(4, 4, (50, 120, 200)))

        class _EmbedHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                n_bytes = int(self.headers.get("Content-Length", "0"))
                req_json = json.loads(self.rfile.read(n_bytes).decode("utf-8"))
                assert req_json["parameters"]["outputDimensionality"] == 4
                assert "image" in req_json["instances"][0]
                body = json.dumps(
                    {"predictions": [{"multimodalEmbedding": [3.0, 4.0, 0.0, 0.0]}]}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            adapter = mdm.VertexMultimodalEmbeddingAdapter(
                project="public-demo-proj",
                model="gemini-embedding-2",
                location="us-central1",
                output_dimensionality=4,
                access_token_fn=lambda: "demo-token",
                endpoint_override=f"http://{host}:{port}/predict",
            )
            emb_vec = adapter.embed(text="hydrating cleanser", image_bytes=sample_png)
            self.assertAlmostEqual(emb_vec[0], 0.6, places=9)
            self.assertAlmostEqual(emb_vec[1], 0.8, places=9)
        finally:
            server.shutdown()
            server.server_close()

        # 2. MultimodalEmbeddingIndex fast-exit vs sister pack-size collision + JSONL persistence
        idx = mdm.MultimodalEmbeddingIndex(inv_temperature=24.0)
        idx.add(
            "sku_blue_180",
            [0.98, 0.20, 0.0, 0.0],
            title="Hydra Wash 180ml",
            lab_color=(45.0, 5.0, -48.0),
            aspect_bin="tall_cylinder",
            pack_size=180,
        )
        with self.assertRaises(ValueError):
            idx.add("sku_blue_180", [0.98, 0.20, 0.0, 0.0])

        idx.upsert(
            "sku_blue_340",
            [0.97, 0.24, 0.0, 0.0],
            title="Hydra Wash 340ml",
            lab_color=(45.0, 5.0, -48.0),
            aspect_bin="tall_cylinder",
            pack_size=340,
        )
        idx.add(
            "sku_red_500",
            [0.10, 0.99, 0.0, 0.0],
            title="Berry Tub 500g",
            lab_color=(52.0, 62.0, 38.0),
            aspect_bin="wide_tub",
            pack_size=500,
        )
        idx.add(
            "sku_green_250",
            [0.0, 0.15, 0.98, 0.0],
            title="Herbal Mist 250ml",
            lab_color=(60.0, -45.0, 30.0),
            aspect_bin="standard",
            pack_size=250,
        )

        # Sister collision between 180ml and 340ml blocks Stage-2A fast exit
        q_colliding = idx.query(
            [0.98, 0.20, 0.0, 0.0],
            query_lab_color=(45.0, 5.0, -48.0),
            query_aspect_bin="tall_cylinder",
            top_k=4,
            use_csls=True,
        )
        self.assertEqual(q_colliding.shortlist_ids[0], "sku_blue_180")
        self.assertFalse(q_colliding.fast_exit)

        # Clear non-colliding query for Berry Tub 500g triggers fast_exit
        q_clear = idx.query(
            [0.08, 0.995, 0.0, 0.0],
            query_lab_color=(52.0, 62.0, 38.0),
            query_aspect_bin="wide_tub",
            top_k=4,
        )
        self.assertTrue(q_clear.fast_exit)
        self.assertEqual(q_clear.fast_exit_id, "sku_red_500")

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = pathlib.Path(tmpdir) / "catalog_index.jsonl"
            idx.save_jsonl(idx_path)
            loaded_idx = mdm.MultimodalEmbeddingIndex.load_jsonl(idx_path)
            self.assertEqual(len(loaded_idx), 4)
            q_loaded = loaded_idx.query(
                [0.08, 0.995, 0.0, 0.0],
                query_lab_color=(52.0, 62.0, 38.0),
                query_aspect_bin="wide_tub",
                top_k=4,
            )
            self.assertEqual(q_loaded.shortlist_ids, q_clear.shortlist_ids)

        # 3. 8-slot Context Pruner exact MCKP optimality vs brute-force 3^8 search
        blocks = [
            mdm.ContextBlock("sys_rules", 180, 70, "full sys", "stub sys", pinned=True),
            mdm.ContextBlock("task_spec", 220, 90, "full spec", "stub spec", pinned=True),
            mdm.ContextBlock("diff_hunk_1", 310, 85, "full d1", "stub d1", pinned=False),
            mdm.ContextBlock("old_test_log", 420, 60, "full log", "stub log", pinned=False),
            mdm.ContextBlock("grep_trace", 260, 55, "full grep", "stub grep", pinned=False),
            mdm.ContextBlock("type_defs", 190, 65, "full types", "stub types", pinned=False),
            mdm.ContextBlock("git_status", 140, 40, "full git", "stub git", pinned=False),
            mdm.ContextBlock("active_err", 280, 95, "full err", "stub err", pinned=False),
        ]
        batches = mdm.build_context_pruning_questions(blocks)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 8)

        slot_probs = {
            "sys_rules": {"full": 0.75, "stub": 0.20, "drop": 0.05},
            "task_spec": {"full": 0.85, "stub": 0.12, "drop": 0.03},
            "diff_hunk_1": {"full": 0.70, "stub": 0.20, "drop": 0.10},
            "old_test_log": {"full": 0.05, "stub": 0.25, "drop": 0.70},
            "grep_trace": {"full": 0.10, "stub": 0.55, "drop": 0.35},
            "type_defs": {"full": 0.45, "stub": 0.45, "drop": 0.10},
            "git_status": {"full": 0.08, "stub": 0.32, "drop": 0.60},
            "active_err": {"full": 0.90, "stub": 0.08, "drop": 0.02},
        }
        budget = 950
        pruned = mdm.prune_agent_context_blocks(blocks, slot_probs, token_budget=budget)
        self.assertLessEqual(pruned.tokens_after, budget)
        self.assertGreater(pruned.token_reduction_ratio, 0.50)
        self.assertNotEqual(pruned.decisions["sys_rules"], "drop")
        self.assertNotEqual(pruned.decisions["task_spec"], "drop")

        # Brute-force verify exact global optimum over all feasible assignments
        best_bf_util = -1.0
        for combo in itertools.product(("full", "stub", "drop"), repeat=len(blocks)):
            tok_sum = 0
            u_sum = 0.0
            feasible = True
            for blk, act in zip(blocks, combo):
                if blk.pinned and act == "drop":
                    feasible = False
                    break
                sp = slot_probs[blk.id]
                pf, ps = sp["full"], sp["stub"]
                if act == "full":
                    tok_sum += blk.full_tokens
                    u_sum += 1.0 * pf + 0.5 * ps
                elif act == "stub":
                    tok_sum += blk.stub_tokens
                    u_sum += 0.6 * pf + 1.0 * ps
            if feasible and tok_sum <= budget and u_sum > best_bf_util:
                best_bf_util = u_sum
        self.assertAlmostEqual(pruned.expected_utility, best_bf_util, places=9)

        # Budget below sum of pinned stub tokens (70 + 90 = 160) must raise ValueError
        with self.assertRaises(ValueError):
            mdm.prune_agent_context_blocks(blocks, slot_probs, token_budget=150)

    def test_mcp_stdio_server_end_to_end_subprocess_and_content_length_framing(self) -> None:
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)

        q_png_b64 = base64.b64encode(mdm.encode_png(_make_solid_buffer(8, 12, (40, 90, 190)))).decode("ascii")
        c1_png_b64 = base64.b64encode(mdm.encode_png(_make_solid_buffer(8, 12, (40, 90, 190)))).decode("ascii")
        c2_png_b64 = base64.b64encode(mdm.encode_png(_make_solid_buffer(8, 12, (200, 50, 50)))).decode("ascii")

        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "unit-test"}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "shortlist_candidates",
                    "arguments": {
                        "items": [
                            {"id": "item_a", "embedding": [1.0, 0.0, 0.0], "title": "Item A"},
                            {"id": "item_b", "embedding": [0.0, 1.0, 0.0], "title": "Item B"},
                        ],
                        "query_embedding": [0.98, 0.15, 0.0],
                        "top_k": 2,
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "evaluate_conformal_decision",
                    "arguments": {
                        "probabilities": {"item_a": 0.51, "item_b": 0.49},
                        "q_hat": 0.60,
                        "method": "aps",
                        "prior_logits": {"item_a": 4.0, "item_b": 0.0},
                        "prior_weight": 0.5,
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "prune_context_blocks",
                    "arguments": {
                        "blocks": [
                            {"id": "b1", "full_tokens": 100, "stub_tokens": 30, "pinned": True, "stub_text": "s1"},
                            {"id": "b2", "full_tokens": 200, "stub_tokens": 50, "pinned": False, "full_text": "f2"},
                        ],
                        "slot_probabilities": {
                            "b1": {"full": 0.2, "stub": 0.7, "drop": 0.1},
                            "b2": {"full": 0.1, "stub": 0.1, "drop": 0.8},
                        },
                        "token_budget": 60,
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "compile_contact_sheet",
                    "arguments": {
                        "query_png_base64": q_png_b64,
                        "candidates": [{"png_base64": c1_png_b64}, {"png_base64": c2_png_b64}],
                        "curved_container": True,
                    },
                },
            },
        ]

        stdin_payload = "".join(json.dumps(r) + "\n" for r in requests).encode("utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "multimodal_decision_models.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT),
            env=env,
        )
        stdout_bytes, stderr_bytes = proc.communicate(input=stdin_payload, timeout=15)
        self.assertEqual(proc.returncode, 0, msg=stderr_bytes.decode("utf-8", errors="replace"))

        lines = [ln for ln in stdout_bytes.decode("utf-8").splitlines() if ln.strip()]
        # 7 messages sent, 1 notification without id -> 6 responses
        self.assertEqual(len(lines), 6)
        by_id = {resp["id"]: resp for resp in (json.loads(ln) for ln in lines)}

        self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "dgem-catalog")
        tool_names = [t["name"] for t in by_id[2]["result"]["tools"]]
        self.assertEqual(
            tool_names,
            [
                "shortlist_candidates",
                "evaluate_conformal_decision",
                "prune_context_blocks",
                "compile_contact_sheet",
            ],
        )

        shortlist_out = json.loads(by_id[3]["result"]["content"][0]["text"])
        self.assertEqual(shortlist_out["shortlist_ids"][0], "item_a")

        conf_out = json.loads(by_id[4]["result"]["content"][0]["text"])
        self.assertEqual(conf_out["prediction_set"], ["item_a", "item_b"])
        self.assertTrue(conf_out["conformal_escalate"])
        self.assertTrue(conf_out["should_escalate"])

        prune_out = json.loads(by_id[5]["result"]["content"][0]["text"])
        self.assertEqual(prune_out["decisions"], {"b1": "stub", "b2": "drop"})

        sheet_out = json.loads(by_id[6]["result"]["content"][0]["text"])
        sheet_img = mdm.decode_png(base64.b64decode(sheet_out["png_base64"]))
        self.assertGreater(sheet_img.width, 16)

        # Also verify LSP Content-Length framing and JSON-RPC error responses in-process
        framed_req = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}).encode("utf-8")
        raw_in = f"Content-Length: {len(framed_req)}\r\n\r\n".encode("ascii") + framed_req
        in_buf = io.BytesIO(raw_in)
        out_buf = io.BytesIO()
        rc = mdm.serve_stdio(in_buf, out_buf)
        self.assertEqual(rc, 0)
        out_raw = out_buf.getvalue()
        self.assertTrue(out_raw.startswith(b"Content-Length:"))
        _, body_part = out_raw.split(b"\r\n\r\n", 1)
        self.assertEqual(json.loads(body_part.decode("utf-8"))["id"], 99)


class TestIter2AdversarialContractsAndRegressions(unittest.TestCase):
    """Regression suite covering Iteration 2 conformal, MCP, cascade, Murphy/ACE, calibration, and PNG contracts."""

    def test_conformal_exact_tie_symmetry_and_stateful_randomized_coverage(self) -> None:
        unif_a = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        unif_z = {"z_3": 0.25, "z_1": 0.25, "z_2": 0.25, "z_0": 0.25}
        for method in ("aps", "raps"):
            set_a = mdm.conformal_prediction_set(unif_a, q_hat=0.0, method=method, lam_reg=0.05, k_reg=1)
            set_z = mdm.conformal_prediction_set(unif_z, q_hat=0.0, method=method, lam_reg=0.05, k_reg=1)
            self.assertEqual(len(set_a.prediction_set), 4)
            self.assertEqual(len(set_z.prediction_set), 4)
            self.assertTrue(set_a.should_escalate)
            self.assertFalse(set_a.singleton)

            pt_set = mdm.conformal_prediction_set(
                {"A": 1.0, "B": 0.0, "C": 0.0}, q_hat=0.0, method=method, lam_reg=0.05, k_reg=1
            )
            self.assertEqual(pt_set.prediction_set, ["A"])
            self.assertTrue(pt_set.singleton)
            self.assertFalse(pt_set.should_escalate)

        # Repeated calls with fixed seed=2 advance stateful stream across exact ties
        calib_tie = [{"A": 0.5, "B": 0.5} for _ in range(100)]
        labels_tie = ["A" if i % 2 == 0 else "B" for i in range(100)]
        cal_res = mdm.calibrate_conformal_quantile(
            calib_tie, labels_tie, alpha=0.10, method="aps", randomized=True, seed=2
        )
        hits_seed = 0
        for idx in range(200):
            gold = "A" if idx % 2 == 0 else "B"
            cset = mdm.conformal_prediction_set(
                {"A": 0.5, "B": 0.5}, cal_res.q_hat, method="aps", randomized=True, seed=2
            )
            if gold in cset.prediction_set:
                hits_seed += 1
        self.assertGreaterEqual(hits_seed / 200.0, 0.85)

        # seed=None randomized stream smooths discrete point-mass coverage toward 1 - alpha = 0.75
        calib_pt = [{"A": 1.0, "B": 0.0} for _ in range(600)]
        labels_pt = ["A"] * 600
        cal_pt = mdm.calibrate_conformal_quantile(
            calib_pt, labels_pt, alpha=0.25, method="aps", randomized=True
        )
        hits_pt = sum(
            1
            for _ in range(2000)
            if "A"
            in mdm.conformal_prediction_set(
                {"A": 1.0, "B": 0.0}, cal_pt.q_hat, method="aps", randomized=True
            ).prediction_set
        )
        cov_pt = hits_pt / 2000.0
        self.assertGreaterEqual(cov_pt, 0.65)
        self.assertLessEqual(cov_pt, 0.85)

    def test_mcp_stdio_framing_bounds_and_strict_jsonrpc_param_validation(self) -> None:
        chunks: list[bytes] = [
            b"Content-Length: -1\r\n\r\n",
            b"Content-Length: 999999999\r\nX-Extra: 1\r\n\r\n",
        ]
        bad_msgs = [
            {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": []},
            {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "shortlist_candidates", "arguments": None}},
            {"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {"name": "shortlist_candidates", "arguments": {"query_embedding": [1.0, 0.0], "candidates": [None]}}},
            {"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {"name": "evaluate_conformal_decision", "arguments": {"probabilities": {"a": 0.6, "b": 0.4}, "q_hat": True}}},
            {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "evaluate_conformal_decision", "arguments": {"probabilities": {"a": 0.6, "b": 0.4}, "calib_probs": [None], "calib_labels": ["a"]}}},
            {"jsonrpc": "2.0", "id": 15, "method": "tools/call", "params": {"name": "prune_context_blocks", "arguments": {"blocks": [None], "slot_probs": {}, "token_budget": 100}}},
            {"jsonrpc": "2.0", "id": 16, "method": "tools/call", "params": {"name": "compile_contact_sheet", "arguments": {"query_png_base64": "AAAA", "candidates": [123]}}},
            {"jsonrpc": "2.0", "id": 77, "method": "ping"},
        ]
        for msg in bad_msgs:
            raw_b = json.dumps(msg).encode("utf-8")
            chunks.append(f"Content-Length: {len(raw_b)}\r\n\r\n".encode("ascii") + raw_b)

        in_buf = io.BytesIO(b"".join(chunks))
        out_buf = io.BytesIO()
        rc = mdm.serve_stdio(in_buf, out_buf)
        self.assertEqual(rc, 0)

        # Parse all framed JSON-RPC responses and verify live session survived to id=77
        raw_out = out_buf.getvalue()
        pos = 0
        responses: list[dict[str, Any]] = []
        while pos < len(raw_out):
            hdr_end = raw_out.index(b"\r\n\r\n", pos)
            hdr_text = raw_out[pos:hdr_end].decode("ascii")
            clen = int(hdr_text.split(":", 1)[1].strip())
            body_start = hdr_end + 4
            body_bytes = raw_out[body_start : body_start + clen]
            responses.append(json.loads(body_bytes.decode("utf-8")))
            pos = body_start + clen

        self.assertEqual(responses[0]["error"]["code"], -32600)
        self.assertEqual(responses[1]["error"]["code"], -32600)
        by_id = {r["id"]: r for r in responses if r.get("id") is not None}
        for bad_id in (10, 11, 12, 13, 14, 15, 16):
            self.assertEqual(by_id[bad_id]["error"]["code"], -32602, msg=f"id={bad_id}: {by_id[bad_id]}")
        self.assertEqual(by_id[77]["result"], {})

    def test_stage2_escalation_reapplies_layer5_guards_and_accurate_slot_telemetry(self) -> None:
        q_pack = mdm.DecisionQuestion(
            type="choice",
            instructions="Select pack variant.",
            criteria={"6pk": "6 pack", "12pk": "12 pack"},
            order=["6pk", "12pk"],
        )
        q_clean = mdm.DecisionQuestion(
            type="choice",
            instructions="Select brand.",
            criteria={"b_6": "Brand 6", "b_12": "Brand 12"},
            order=["b_6", "b_12"],
        )
        questions = {"pack": q_pack, "brand": q_clean}

        # Stage 1 predicts wrong 12pk on both slots with hesitation triggering escalation;
        # Stage 2 still predicts wrong 12pk on "pack" (must be overridden to 6pk by Layer-5),
        # while Stage 2 already predicts valid b_6 on "brand" (guard_applied must be "none", not leaked).
        s1 = mdm.MockCallableBackend(
            "s1_fast",
            lambda st, qs, imgs: {
                "pack": {"12pk": 0.55, "6pk": 0.45},
                "brand": {"b_12": 0.55, "b_6": 0.45},
            },
        )
        s2 = mdm.MockCallableBackend(
            "s2_vlm",
            lambda st, qs, imgs: {
                "pack": {"12pk": 0.88, "6pk": 0.12},
                "brand": {"b_6": 0.91, "b_12": 0.09},
            },
        )
        orch = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1,
            stage2_backend=s2,
            hesitation_threshold=0.20,
            layer5_slot_context={
                "pack": {"candidate_pack_sizes": {"6pk": 6, "12pk": 12}, "decoded_pack_size": 6},
                "brand": {"candidate_pack_sizes": {"b_6": 6, "b_12": 12}, "decoded_pack_size": 6},
            },
        )
        res = orch.evaluate({"item": "case_layer5"}, questions)
        self.assertTrue(res.escalated)
        self.assertEqual(res.terminal_state, mdm.DecisionTerminalState.GUARD_OVERRIDDEN)

        pack_slot = res.slots["pack"]
        self.assertEqual(pack_slot.predicted, "6pk")
        self.assertEqual(pack_slot.guard_applied, "layer5_pack_size_sister_switch")
        self.assertEqual(pack_slot.terminal_state, mdm.DecisionTerminalState.GUARD_OVERRIDDEN)

        brand_slot = res.slots["brand"]
        self.assertEqual(brand_slot.predicted, "b_6")
        self.assertEqual(brand_slot.guard_applied, "none")
        self.assertEqual(brand_slot.terminal_state, mdm.DecisionTerminalState.STAGE2_VERIFIED)

    def test_murphy_exact_continuous_identity_and_heterogeneous_active_support_ace(self) -> None:
        rng = random.Random(20261025)
        probs_list: list[dict[str, float]] = []
        golds: list[str] = []
        for _ in range(150):
            raw = [rng.random() + 0.05 for _ in range(3)]
            s = sum(raw)
            dist = {"c0": raw[0] / s, "c1": raw[1] / s, "c2": raw[2] / s}
            probs_list.append(dist)
            u = rng.random()
            if u < dist["c0"]:
                golds.append("c0")
            elif u < dist["c0"] + dist["c1"]:
                golds.append("c1")
            else:
                golds.append("c2")

        murphy = mdm.brier_murphy_decomposition(probs_list, golds, num_bins=10)
        self.assertGreaterEqual(murphy.reliability, 0.0)
        self.assertGreaterEqual(murphy.resolution, 0.0)
        self.assertAlmostEqual(
            murphy.brier,
            murphy.reliability - murphy.resolution + murphy.uncertainty,
            places=12,
        )

        # 50 completely wrong binary predictions with disjoint option schemas -> ACE == 1.0
        disjoint_probs = [{f"opt_{i}_a": 1.0, f"opt_{i}_b": 0.0} for i in range(50)]
        disjoint_golds = [f"opt_{i}_b" for i in range(50)]
        ace_val = mdm.adaptive_calibration_error(disjoint_probs, disjoint_golds, num_bins=10)
        self.assertAlmostEqual(ace_val, 1.0, places=9)

    def test_vector_scaling_shift_invariance_underconfident_gain_and_unobserved_class(self) -> None:
        rng = random.Random(77)
        classes = ["A", "B", "C"]
        true_bias = {"A": 0.85, "B": -0.35, "C": -0.50}
        train_logits: list[dict[str, float]] = []
        train_shifted: list[dict[str, float]] = []
        train_probs: list[dict[str, float]] = []
        train_golds: list[str] = []
        val_logits: list[dict[str, float]] = []
        val_probs: list[dict[str, float]] = []
        val_golds: list[str] = []

        for idx in range(320):
            true_l = {c: rng.gauss(0.0, 1.4) + true_bias[c] for c in classes}
            max_tl = max(true_l.values())
            true_exps = {c: math.exp(true_l[c] - max_tl) for c in classes}
            tsum = sum(true_exps.values())
            true_p = {c: true_exps[c] / tsum for c in classes}
            u = rng.random()
            cum = 0.0
            gold = "C"
            for c in classes:
                cum += true_p[c]
                if u <= cum:
                    gold = c
                    break

            # Underconfident T=3.0 model with class bias distortion
            raw_m = {
                "A": true_l["A"] / 3.0 - 0.65,
                "B": true_l["B"] / 3.0 + 0.45,
                "C": true_l["C"] / 3.0 + 0.20,
            }
            max_ml = max(raw_m.values())
            m_exps = {c: math.exp(raw_m[c] - max_ml) for c in classes}
            msum = sum(m_exps.values())
            mp = {c: m_exps[c] / msum for c in classes}

            if idx < 200:
                train_logits.append(raw_m)
                train_shifted.append({c: v + 17.5 for c, v in raw_m.items()})
                train_probs.append(mp)
                train_golds.append(gold)
            else:
                val_logits.append(raw_m)
                val_probs.append(mp)
                val_golds.append(gold)

        vs_base = mdm.fit_vector_scaling(train_logits, train_golds, is_logits=True, l2_reg=1e-2)
        vs_shift = mdm.fit_vector_scaling(train_shifted, train_golds, is_logits=True, l2_reg=1e-2)
        preds_b = vs_base.predict_proba(train_logits[:20], is_logits=True)
        preds_s = vs_shift.predict_proba(train_shifted[:20], is_logits=True)
        for p_b, p_s in zip(preds_b, preds_s):
            for c in classes:
                self.assertAlmostEqual(p_b[c], p_s[c], places=10)

        cal_val_from_l = vs_base.predict_proba(val_logits, is_logits=True)
        vs_prob = mdm.fit_vector_scaling(train_probs, train_golds, is_logits=False)
        cal_val_from_p = vs_prob.predict_proba(val_probs, is_logits=False)
        raw_nll = mdm.negative_log_likelihood(val_probs, val_golds)
        raw_ece = mdm.expected_calibration_error(val_probs, val_golds, num_bins=10)
        self.assertLess(mdm.negative_log_likelihood(cal_val_from_l, val_golds), raw_nll)
        self.assertLess(mdm.negative_log_likelihood(cal_val_from_p, val_golds), raw_nll)
        self.assertLess(mdm.expected_calibration_error(cal_val_from_p, val_golds, num_bins=10), raw_ece)

        # Unobserved class "C" with l2_reg=0.0 must not inflate rare class probability
        unobs_rows = [{"A": 0.45, "B": 0.40, "C": 0.15} for _ in range(30)]
        unobs_golds = ["A" if i % 2 == 0 else "B" for i in range(30)]
        vs_unobs = mdm.fit_vector_scaling(unobs_rows, unobs_golds, is_logits=False, l2_reg=0.0)
        pred_unobs = vs_unobs.predict_proba([{"A": 0.40, "B": 0.40, "C": 0.20}], is_logits=False)[0]
        self.assertLess(pred_unobs["C"], 0.40)

    def test_png_chunk_invariants_and_tall_vertical_gradient_saliency_band(self) -> None:
        base_img = _make_solid_buffer(4, 4, (20, 40, 60))
        valid_png = mdm.encode_png(base_img)

        # Inject duplicate IHDR before IDAT
        ihdr_end = 8 + 8 + 13 + 4
        ihdr_chunk = valid_png[8:ihdr_end]
        dup_ihdr_png = valid_png[:ihdr_end] + ihdr_chunk + valid_png[ihdr_end:]
        with self.assertRaisesRegex(ValueError, "duplicate IHDR"):
            mdm.decode_png(dup_ihdr_png)

        # Replace IEND with non-empty IEND chunk
        iend_offset = valid_png.rindex(b"IEND") - 4
        bad_iend_png = valid_png[:iend_offset] + _png_chunk_raw(b"IEND", b"\x00")
        with self.assertRaisesRegex(ValueError, "IEND chunk must have 0 payload bytes"):
            mdm.decode_png(bad_iend_png)

        # 80%-height vertical linear gradient discriminative band (rows 6..54 on H=60)
        h, w = 60, 24
        px1 = bytearray(w * h * 3)
        px2 = bytearray(w * h * 3)
        for y in range(h):
            for x in range(w):
                off = (y * w + x) * 3
                if 6 <= y < 54:
                    alpha_g = (y - 6) / 47.0
                    delta = int(round(20.0 + 180.0 * alpha_g))
                    px1[off : off + 3] = bytes((min(255, 30 + delta), 40, 40))
                    px2[off : off + 3] = bytes((30, 40, min(255, 40 + delta)))
                else:
                    px1[off : off + 3] = bytes((90, 90, 90))
                    px2[off : off + 3] = bytes((90, 90, 90))
        c1 = mdm.ImageBuffer(w, h, bytes(px1))
        c2 = mdm.ImageBuffer(w, h, bytes(px2))
        y0, y1, _ = mdm.extract_inter_candidate_saliency_band([c1, c2])
        inter = max(0, min(y1, 54) - max(y0, 6))
        union = max(y1, 54) - min(y0, 6)
        iou = inter / float(union)
        self.assertGreaterEqual(iou, 0.80, f"expected IoU >= 0.80 on 80% gradient band, got {iou:.3f} ({y0}..{y1})")

    def test_cache_key_bijectivity_across_per_call_and_orchestrator_parameters(self) -> None:
        q3 = mdm.DecisionQuestion(
            type="choice",
            instructions="Select matching SKU.",
            criteria={"sku_a": "SKU Alpha", "sku_b": "SKU Beta", "none_of_the_above": "None visible"},
            order=["sku_a", "sku_b", "none_of_the_above"],
        )
        cache = mdm.ContentAddressedDecisionCache()

        # 1. Per-call prior_logits must isolate cache keys and predictions
        s1_hesitant = mdm.MockCallableBackend(
            "s1_hesitant",
            lambda _st, _qs, _im=None: {"q": {"sku_a": 0.45, "sku_b": 0.43, "none_of_the_above": 0.12}},
        )
        orch_prior = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_hesitant,
            prior_weight=1.0,
            prior_gate_mode="always",
            response_cache=cache,
        )
        r_pa = orch_prior.evaluate(
            {"id": 10}, {"q": q3}, prior_logits={"q": {"sku_a": 8.0, "sku_b": 0.0, "none_of_the_above": -4.0}}
        )
        r_pb = orch_prior.evaluate(
            {"id": 10}, {"q": q3}, prior_logits={"q": {"sku_a": 0.0, "sku_b": 8.0, "none_of_the_above": -4.0}}
        )
        self.assertFalse(r_pa.cache_hit)
        self.assertFalse(r_pb.cache_hit)
        self.assertNotEqual(r_pa.cache_key, r_pb.cache_key)
        self.assertEqual(r_pa.slots["q"].predicted, "sku_a")
        self.assertEqual(r_pb.slots["q"].predicted, "sku_b")

        # 2. Per-call shortlist_scores on wide K=30 (>26) catalog must isolate cache keys
        wide_criteria = {f"sku_{i:02d}": f"Catalog Option {i:02d}" for i in range(30)}
        q_wide = mdm.DecisionQuestion(
            type="choice",
            instructions="Select wide catalog SKU.",
            criteria=wide_criteria,
            order=list(wide_criteria.keys()),
        )

        def _score_first_shortlisted(
            _st: Any,
            qs: dict[str, mdm.DecisionQuestion],
            _im: list[str] | None = None,
        ) -> dict[str, dict[str, float]]:
            opts = list(qs["w"].criteria.keys())
            return {"w": {k: (0.85 if idx == 0 else 0.15 / max(1, len(opts) - 1)) for idx, k in enumerate(opts)}}

        orch_wide = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("s1_wide", _score_first_shortlisted),
            shortlist_top_k=4,
            response_cache=cache,
        )
        scores_hi_0 = {f"sku_{i:02d}": (10.0 - i if i < 4 else -5.0) for i in range(30)}
        scores_hi_20 = {f"sku_{i:02d}": (10.0 - (i - 20) if 20 <= i < 24 else -5.0) for i in range(30)}
        r_w0 = orch_wide.evaluate({"id": 11}, {"w": q_wide}, shortlist_scores={"w": scores_hi_0})
        r_w20 = orch_wide.evaluate({"id": 11}, {"w": q_wide}, shortlist_scores={"w": scores_hi_20})
        self.assertFalse(r_w0.cache_hit)
        self.assertFalse(r_w20.cache_hit)
        self.assertNotEqual(r_w0.cache_key, r_w20.cache_key)
        self.assertEqual(r_w0.slots["w"].predicted, "sku_00")
        self.assertEqual(r_w20.slots["w"].predicted, "sku_20")

        # 3. Per-call candidate_pack_sizes, visual_energies, and rank1_visual_similarity must isolate Layer-5 guards
        s1_confident_a = mdm.MockCallableBackend(
            "s1_conf_a",
            lambda _st, _qs, _im=None: {"q": {"sku_a": 0.92, "sku_b": 0.06, "none_of_the_above": 0.02}},
        )
        orch_l5 = mdm.CascadeDecisionOrchestrator(stage1_backend=s1_confident_a, response_cache=cache)
        r_pack1 = orch_l5.evaluate(
            {"id": 12},
            {"q": q3},
            sister_pack_collision=True,
            decoded_pack_size=6,
            candidate_pack_sizes={"sku_a": 6, "sku_b": 12},
        )
        r_pack2 = orch_l5.evaluate(
            {"id": 12},
            {"q": q3},
            sister_pack_collision=True,
            decoded_pack_size=6,
            candidate_pack_sizes={"sku_a": 12, "sku_b": 6},
        )
        self.assertNotEqual(r_pack1.cache_key, r_pack2.cache_key)
        self.assertEqual(r_pack1.slots["q"].predicted, "sku_a")
        self.assertEqual(r_pack2.slots["q"].predicted, "sku_b")

        s1_low_conf_a = mdm.MockCallableBackend(
            "s1_low_conf_a",
            lambda _st, _qs, _im=None: {"q": {"sku_a": 0.58, "sku_b": 0.30, "none_of_the_above": 0.12}},
        )
        orch_ve = mdm.CascadeDecisionOrchestrator(stage1_backend=s1_low_conf_a, response_cache=cache)
        r_ve_ok = orch_ve.evaluate(
            {"id": 13},
            {"q": q3},
            rank1_id="sku_b",
            visual_energies={"sku_b": 0.22, "sku_a": 0.20},
        )
        r_ve_veto = orch_ve.evaluate(
            {"id": 13},
            {"q": q3},
            rank1_id="sku_b",
            visual_energies={"sku_b": 0.30, "sku_a": 0.05},
        )
        self.assertFalse(r_ve_veto.cache_hit)
        self.assertNotEqual(r_ve_ok.cache_key, r_ve_veto.cache_key)
        self.assertEqual(r_ve_ok.terminal_state, mdm.DecisionTerminalState.STAGE1_RESOLVED)
        self.assertEqual(r_ve_veto.terminal_state, mdm.DecisionTerminalState.GUARD_OVERRIDDEN)
        self.assertEqual(r_ve_veto.slots["q"].predicted, "sku_b")

        s1_oos_top = mdm.MockCallableBackend(
            "s1_oos_top",
            lambda _st, _qs, _im=None: {"q": {"none_of_the_above": 0.90, "sku_b": 0.08, "sku_a": 0.02}},
        )
        orch_sim = mdm.CascadeDecisionOrchestrator(stage1_backend=s1_oos_top, response_cache=cache)
        r_sim_lo = orch_sim.evaluate({"id": 14}, {"q": q3}, rank1_id="sku_b", rank1_visual_similarity=0.50)
        r_sim_hi = orch_sim.evaluate({"id": 14}, {"q": q3}, rank1_id="sku_b", rank1_visual_similarity=0.85)
        self.assertNotEqual(r_sim_lo.cache_key, r_sim_hi.cache_key)
        self.assertEqual(r_sim_lo.terminal_state, mdm.DecisionTerminalState.ABSTAIN_OOS)
        self.assertEqual(r_sim_hi.terminal_state, mdm.DecisionTerminalState.GUARD_OVERRIDDEN)
        self.assertEqual(r_sim_hi.slots["q"].predicted, "sku_b")

        # 4. Custom oos_keys, init vs per-call layer5_slot_context, and conformal_q_hat=+inf support
        q_custom_oos = mdm.DecisionQuestion(
            type="choice",
            instructions="Select SKU.",
            criteria={"sku_a": "Alpha", "custom_oos": "Unlisted"},
            order=["sku_a", "custom_oos"],
        )
        s1_cust = mdm.MockCallableBackend(
            "s1_cust",
            lambda _st, _qs, _im=None: {"q": {"custom_oos": 0.93, "sku_a": 0.07}},
        )
        orch_def_oos = mdm.CascadeDecisionOrchestrator(stage1_backend=s1_cust, response_cache=cache)
        orch_alt_oos = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_cust,
            oos_keys={"none_of_the_above", "custom_oos"},
            response_cache=cache,
        )
        r_do = orch_def_oos.evaluate({"id": 15}, {"q": q_custom_oos})
        r_ao = orch_alt_oos.evaluate({"id": 15}, {"q": q_custom_oos})
        self.assertFalse(r_ao.cache_hit)
        self.assertNotEqual(r_do.cache_key, r_ao.cache_key)
        self.assertEqual(r_do.terminal_state, mdm.DecisionTerminalState.STAGE1_RESOLVED)
        self.assertEqual(r_ao.terminal_state, mdm.DecisionTerminalState.ABSTAIN_OOS)

        s1_two = mdm.MockCallableBackend(
            "s1_two",
            lambda _st, _qs, _im=None: {
                "q1": {"none_of_the_above": 0.90, "sku_b": 0.08, "sku_a": 0.02},
                "q2": {"none_of_the_above": 0.90, "sku_b": 0.08, "sku_a": 0.02},
            },
        )
        orch_ctx_a = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_two,
            layer5_slot_context={"q2": {"rank1_id": "sku_b", "rank1_visual_similarity": 0.50}},
            response_cache=cache,
        )
        orch_ctx_b = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_two,
            layer5_slot_context={"q2": {"rank1_id": "sku_b", "rank1_visual_similarity": 0.85}},
            response_cache=cache,
        )
        r_ctx_a = orch_ctx_a.evaluate(
            {"id": 17},
            {"q1": q3, "q2": q3},
            layer5_slot_context={"q1": {"rank1_id": "sku_b", "rank1_visual_similarity": 0.85}},
        )
        r_ctx_b = orch_ctx_b.evaluate(
            {"id": 17},
            {"q1": q3, "q2": q3},
            layer5_slot_context={"q1": {"rank1_id": "sku_b", "rank1_visual_similarity": 0.85}},
        )
        self.assertFalse(r_ctx_b.cache_hit)
        self.assertNotEqual(r_ctx_a.cache_key, r_ctx_b.cache_key)
        self.assertEqual(r_ctx_a.slots["q2"].predicted, "none_of_the_above")
        self.assertEqual(r_ctx_b.slots["q2"].predicted, "sku_b")

        orch_inf_q = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_confident_a,
            conformal_q_hat=float("inf"),
            response_cache=cache,
        )
        r_inf = orch_inf_q.evaluate({"id": 16}, {"q": q3})
        self.assertEqual(len(r_inf.cache_key), 64)

    def test_mixed_key_state_canonicalization_randomized_conformal_bypass_and_quarantine(self) -> None:
        q = mdm.DecisionQuestion(
            type="choice",
            instructions="Classify SKU.",
            criteria={"sku_a": "Alpha", "sku_b": "Beta"},
            order=["sku_a", "sku_b"],
        )
        k_mixed_1 = mdm.compute_decision_cache_key(
            "m1",
            {1: "alpha", "b": (2, 3), "tags": {"x", "y"}},
            {"q": q},
        )
        k_mixed_2 = mdm.compute_decision_cache_key(
            "m1",
            {"tags": {"y", "x"}, "b": (2, 3), 1: "alpha"},
            {"q": q},
        )
        self.assertEqual(k_mixed_1, k_mixed_2)

        keys_scalar = {
            mdm.compute_decision_cache_key("m1", {1: "v"}, {"q": q}),
            mdm.compute_decision_cache_key("m1", {"1": "v"}, {"q": q}),
            mdm.compute_decision_cache_key("m1", {True: "v"}, {"q": q}),
        }
        self.assertEqual(len(keys_scalar), 3)

        keys_seq = {
            mdm.compute_decision_cache_key("m1", {"seq": (1, 2)}, {"q": q}),
            mdm.compute_decision_cache_key("m1", {"seq": [1, 2]}, {"q": q}),
            mdm.compute_decision_cache_key("m1", {"seq": {"__mdm_type__": "tuple", "items": [1, 2]}}, {"q": q}),
        }
        self.assertEqual(len(keys_seq), 3)

        wrapped_collision = mdm.wrap_untrusted_state({1: "int_val", "1": "str_val"})
        self.assertIn("int_val", wrapped_collision)
        self.assertIn("str_val", wrapped_collision)

        # Randomized conformal mode must bypass response_cache instead of freezing a single U draw
        cache = mdm.ContentAddressedDecisionCache()
        s1_sharp = mdm.MockCallableBackend(
            "s1_sharp",
            lambda _st, _qs, _im=None: {"q": {"sku_a": 0.80, "sku_b": 0.20}},
        )
        orch_rand = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1_sharp,
            conformal_q_hat=0.82,
            conformal_method="aps",
            conformal_randomized=True,
            conformal_seed=19,
            response_cache=cache,
        )
        set_sizes: set[int] = set()
        for _ in range(30):
            res_r = orch_rand.evaluate({1: "mixed", "k": 2}, {"q": q})
            self.assertFalse(res_r.cache_hit)
            self.assertIsNone(cache.get(res_r.cache_key))
            set_sizes.add(len(res_r.slots["q"].conformal_set))
        self.assertEqual(set_sizes, {1, 2})

        # Unserializable, non-finite, cyclic, and malformed questions must quarantine cleanly
        cyc: list[Any] = []
        cyc.append(cyc)
        orch_tc = mdm.CascadeDecisionOrchestrator(stage1_backend=s1_sharp, total_contract_mode=True)
        for bad_state in (object(), {"x": float("nan")}, {"x": float("inf")}, {"c": cyc}):
            q_res = orch_tc.evaluate(bad_state, {"q": q})
            self.assertEqual(q_res.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION)
            self.assertEqual(q_res.failure_category, mdm.FailureCategory.SCHEMA_CONTRACT_VIOLATION)

        q_none_res = orch_tc.evaluate({"ok": 1}, None)  # type: ignore[arg-type]
        self.assertEqual(q_none_res.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION)
        self.assertEqual(q_none_res.slots, {})

    def test_blackbox_and_upstream_parsers_quarantine_malformed_types_under_total_contracts(self) -> None:
        q = mdm.DecisionQuestion(
            type="choice",
            instructions="Select option.",
            criteria={"sku_a": "Alpha", "sku_b": "Beta", "oos": "Out of scope"},
            order=["sku_a", "sku_b", "oos"],
        )
        bad_draws: list[Any] = [
            '{"votes": {"A": true, "B": false}}',
            '{"votes": {"A": "three", "B": 1}}',
            '{"votes": {"A": null, "B": 1}}',
            '{"votes": {"A": [3], "B": 1}}',
            '{"votes": {"A": -2, "B": 4}}',
            {"votes": {"A": float("nan"), "B": 1.0}},
            {"votes": {"A": float("inf"), "B": 1.0}},
            {"votes": "not_a_dict"},
            {"slot": {"votes": {"sku_a": True}}},
            {"slot": {"choice": True}},
            {"slot": [1, 2]},
        ]
        for idx, bad_draw in enumerate(bad_draws):
            bb = mdm.BlackBoxSampledBackend(
                sampler=lambda _st, _qs, _im=None, _d=bad_draw: _d,
                num_samples=1,
                prior_alpha=0.5,
                name=f"bb_bad_{idx}",
            )
            with self.assertRaises(mdm.SchemaContractViolationError) as err_ctx:
                bb.score_questions({"case": idx}, {"slot": q})
            self.assertIsInstance(err_ctx.exception, ValueError)

            orch = mdm.CascadeDecisionOrchestrator(stage1_backend=bb, total_contract_mode=True)
            res = orch.evaluate({"case": idx}, {"slot": q})
            self.assertEqual(res.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION)
            self.assertEqual(res.failure_category, mdm.FailureCategory.SCHEMA_CONTRACT_VIOLATION)

        # Single-slot qid == "A" with flat votes must still route as flat option votes
        bb_qid_a = mdm.BlackBoxSampledBackend(
            name="bb_qid_a",
            sampler=lambda _st, _qs, _im=None: '{"votes": {"A": 5, "B": 1}}',
            num_samples=1,
            prior_alpha=0.5,
        )
        dist_a = bb_qid_a.score_questions({"case": "qid_A"}, {"A": q})
        self.assertGreater(dist_a["A"]["sku_a"], dist_a["A"]["sku_b"])

        # DgemSystemOneBackend, OpenAILogprobBackend, VertexGeminiBackend, and MockCallableBackend guards
        with self.assertRaises(mdm.SchemaContractViolationError):
            mdm.DgemSystemOneBackend.parse_response(
                {"results": {"slot": {"probabilities": {"sku_a": True, "sku_b": False, "oos": 0.0}}}},
                {"slot": q},
            )
        with self.assertRaises(mdm.SchemaContractViolationError):
            mdm.OpenAILogprobBackend.extract_letter_distribution(
                {"A": True, "B": -1.2},  # type: ignore[dict-item]
                {"A": "sku_a", "B": "sku_b"},
            )
        with self.assertRaises(mdm.SchemaContractViolationError):
            mdm.VertexGeminiBackend.parse_generate_content_response(
                {"candidates": ["not_an_object"]},
                {"slot": q},
            )
        bad_mock = mdm.MockCallableBackend(
            "bad_mock",
            lambda _st, _qs, _im=None: {"slot": {"sku_a": True, "sku_b": False, "oos": 0.0}},  # type: ignore[dict-item]
        )
        res_bad_mock = mdm.CascadeDecisionOrchestrator(stage1_backend=bad_mock, total_contract_mode=True).evaluate(
            {"id": 99},
            {"slot": q},
        )
        self.assertEqual(res_bad_mock.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION)


class TestIter3EmbeddingAndMcpMimeContracts(unittest.TestCase):
    """Iteration 3 regressions for extreme float64 L2 norms, strict embedding contracts, unordered MCP MIME headers, and 9-key abstain."""

    def test_extreme_float64_l2_normalization_and_sequence_contract_enforcement(self) -> None:
        for extreme_vec in (
            [1e200, 1e200],
            [1e308, 1e308, 1e308, 1e308],
            [1e-200, 1e-200],
            [5e-324, 5e-324],
            [-3e250, 4e250],
        ):
            parsed = mdm.VertexMultimodalEmbeddingAdapter.parse_predict_response(
                {"predictions": [{"multimodalEmbedding": extreme_vec}]}
            )
            self.assertAlmostEqual(math.hypot(*parsed), 1.0, places=12)

            idx = mdm.MultimodalEmbeddingIndex()
            idx.upsert("sku_pos", extreme_vec)
            idx.upsert("sku_neg", [-extreme_vec[0]] + list(extreme_vec[1:]))
            stored = idx.get("sku_pos")
            assert stored is not None
            self.assertAlmostEqual(math.hypot(*stored.embedding), 1.0, places=12)
            q_out = idx.query(extreme_vec, top_k=2, use_csls=True)
            self.assertEqual(q_out.shortlist_ids[0], "sku_pos")
            self.assertAlmostEqual(mdm.cosine_similarity(extreme_vec, extreme_vec), 1.0, places=12)

        adapted_hi = mdm.apply_residual_mlp_adapter(
            [1e200, 1e200],
            [[0.5, -0.2], [0.1, 0.3]],
            [[0.3, 0.1], [0.4, -0.2]],
            alpha=0.35,
        )
        self.assertAlmostEqual(math.hypot(*adapted_hi), 1.0, places=12)

        for bad_resp in (
            {"predictions": [{"embedding": "123"}]},
            {"predictions": [{"multimodalEmbedding": b"123"}]},
            {"predictions": [{"textEmbedding": [True, False]}]},
            {"predictions": [{"embedding": ["0.6", "0.8"]}]},
            {"predictions": [{"embedding": [None, 1.0]}]},
            {"predictions": [{"embedding": [0.0, 0.0]}]},
        ):
            with self.assertRaises(mdm.SchemaContractViolationError) as ctx:
                mdm.VertexMultimodalEmbeddingAdapter.parse_predict_response(bad_resp)
            self.assertIsInstance(ctx.exception, ValueError)

        idx_chk = mdm.MultimodalEmbeddingIndex()
        for bad_vec in ("123", b"123", [True, False], [None, 1.0], [0.0, 0.0], [float("nan"), 1.0]):
            with self.assertRaises(mdm.SchemaContractViolationError):
                idx_chk.upsert("bad", bad_vec)  # type: ignore[arg-type]

        idx_chk.add("valid_sku", [0.6, 0.8])
        with self.assertRaises(mdm.SchemaContractViolationError):
            idx_chk.add("bad_lab", [0.6, 0.8], lab_color="abc")  # type: ignore[arg-type]
        with self.assertRaises(mdm.SchemaContractViolationError):
            idx_chk.query([0.6, 0.8], query_lab_color="abc")  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_jsonl = pathlib.Path(tmpdir) / "bad_str_emb.jsonl"
            bad_jsonl.write_text(json.dumps({"id": "s1", "embedding": "123"}) + "\n", encoding="utf-8")
            with self.assertRaises(mdm.SchemaContractViolationError):
                mdm.MultimodalEmbeddingIndex.load_jsonl(bad_jsonl)

    def test_mcp_unordered_mime_headers_and_framing_error_recovery(self) -> None:
        ping_77 = json.dumps({"jsonrpc": "2.0", "id": 77, "method": "ping", "params": {}}).encode("utf-8")
        ping_78 = json.dumps({"jsonrpc": "2.0", "id": 78, "method": "ping"}).encode("utf-8")
        over_hdrs = b"".join(f"X-Hdr-{i}: {i}\r\n".encode("ascii") for i in range(70)) + b"Content-Length: 0\r\n\r\n"

        wire = b"".join(
            [
                b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
                + f"Content-Length: {len(ping_77)}\r\n\r\n".encode("ascii")
                + ping_77,
                b"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n",
                b"Content-Length: 10\r\nContent-Length: 20\r\n\r\n",
                b"Content-Length: not_an_int\r\n\r\n",
                over_hdrs,
                b"X-Trace: t1\ncontent-type: application/json\ncontent-length: "
                + str(len(ping_78)).encode("ascii")
                + b"\nX-Post: 1\n\n"
                + ping_78,
            ]
        )

        out_buf = io.BytesIO()
        rc = mdm.serve_stdio(io.BytesIO(wire), out_buf)
        self.assertEqual(rc, 0)

        raw_out = out_buf.getvalue()
        pos = 0
        frames: list[dict[str, Any]] = []
        while pos < len(raw_out):
            hdr_end = raw_out.index(b"\r\n\r\n", pos)
            clen = int(raw_out[pos:hdr_end].decode("ascii").split(":", 1)[1].strip())
            body_start = hdr_end + 4
            frames.append(json.loads(raw_out[body_start : body_start + clen].decode("utf-8")))
            pos = body_start + clen

        self.assertEqual(len(frames), 6)
        self.assertEqual(frames[0], {"jsonrpc": "2.0", "id": 77, "result": {}})
        for err_idx in (1, 2, 3, 4):
            self.assertEqual(frames[err_idx]["error"]["code"], -32600)
        self.assertEqual(frames[5], {"jsonrpc": "2.0", "id": 78, "result": {}})

        # Malformed NDJSON line must still return unframed -32700 parse error
        nd_out = io.BytesIO()
        self.assertEqual(mdm.serve_stdio(io.BytesIO(b"{not valid json\n"), nd_out), 0)
        nd_resp = json.loads(nd_out.getvalue().decode("utf-8").strip())
        self.assertEqual(nd_resp["error"]["code"], -32700)

    def test_nine_key_case_insensitive_conformal_abstain_union(self) -> None:
        expected_9 = {
            "0",
            "abstain",
            "insufficient_evidence",
            "none",
            "none_of_the_above",
            "none_visible",
            "oos",
            "uncertain",
            "unknown",
        }
        self.assertEqual(set(mdm.DEFAULT_CONFORMAL_ABSTAIN_KEYS), expected_9)
        for key in [*sorted(expected_9), " OOS ", "ABSTAIN", " Uncertain "]:
            res_def = mdm.conformal_prediction_set({key: 0.96, "sku_1": 0.04}, 0.20, method="lac")
            self.assertTrue(res_def.singleton)
            self.assertTrue(res_def.contains_abstain, msg=f"key={key!r}")
            self.assertTrue(res_def.should_escalate, msg=f"key={key!r}")

            res_none = mdm.conformal_prediction_set(
                {key: 0.96, "sku_1": 0.04},
                0.20,
                method="lac",
                abstain_keys=None,
            )
            self.assertTrue(res_none.contains_abstain)
            self.assertTrue(res_none.should_escalate)

            res_opt_out = mdm.conformal_prediction_set(
                {key: 0.96, "sku_1": 0.04},
                0.20,
                method="lac",
                abstain_keys=(),
            )
            self.assertFalse(res_opt_out.contains_abstain)
            self.assertFalse(res_opt_out.should_escalate)

    def test_iter4_positive_allowlist_draw_grammar_and_constructor_contracts(self) -> None:
        q_open = {
            "sku": mdm.DecisionQuestion(
                type="choice",
                instructions="Select SKU.",
                criteria={"sku_a": "Alpha", "sku_b": "Beta", "oos": "Out of scope"},
                order=["sku_a", "sku_b", "oos"],
            )
        }
        q_closed = {
            "sku": mdm.DecisionQuestion(
                type="choice",
                instructions="Select SKU.",
                criteria={"sku_a": "Alpha", "sku_b": "Beta"},
                order=["sku_a", "sku_b"],
            )
        }
        q_multi = {
            "s1": q_open["sku"],
            "s2": q_closed["sku"],
        }

        single_malformed: list[Any] = [
            {"A": -0.5},
            {"A": True},
            {"A": float("nan")},
            {"A": 0.0},
            {"A": {"nested": 1}},
            "bad",
            "-0.5",
            "0.0",
            "",
            "not a valid token",
            "A bad string",
            {},
            {"evidence": "label only"},
            {"votes": {}},
            {"votes": {"A": 0.0, "B": 0.0}},
            {"votes": {"A": {"nested": 1}, "B": 2}},
            {"votes": {"INVALID": 0.0, "A": 1.0}},
            '{"votes": {"A": NaN, "B": 1}}',
            {"sku": -0.5},
            {"sku": 0.0},
            {"sku": {}},
            {"choice": "bad"},
        ]
        for q_schema in (q_open, q_closed):
            for idx, payload in enumerate(single_malformed):
                bb = mdm.BlackBoxSampledBackend(
                    f"iter4_bad_{idx}",
                    sampler=lambda _s, _qs, _im=None, _p=payload: _p,
                    num_samples=1,
                )
                with self.assertRaises(mdm.SchemaContractViolationError, msg=f"payload={payload!r}"):
                    bb.score_questions({"x": idx}, q_schema)
                env = mdm.CascadeDecisionOrchestrator(
                    stage1_backend=bb, total_contract_mode=True
                ).evaluate({"x": idx}, q_schema)
                self.assertEqual(
                    env.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION
                )

        for mixed_seq in (["A", {"A": float("nan")}], ["A", {"A": True}], ["A", "bad"], ["A", {}]):
            state_idx = {"i": 0}

            def _mixed_sampler(
                _s: Any, _qs: Any, _im: list[str] | None = None, _k: int = 0, _seq: list[Any] = mixed_seq
            ) -> Any:
                val = _seq[state_idx["i"] % len(_seq)]
                state_idx["i"] += 1
                return val

            bb_mix = mdm.BlackBoxSampledBackend("iter4_mix", sampler=_mixed_sampler, num_samples=2)
            env_mix = mdm.CascadeDecisionOrchestrator(
                stage1_backend=bb_mix, total_contract_mode=True
            ).evaluate({"x": 1}, q_open)
            self.assertEqual(
                env_mix.terminal_state, mdm.DecisionTerminalState.REJECTED_CONTRACT_VIOLATION
            )

        for multi_bad in (
            {"votes": {"A": "not_a_number"}},
            {"votes": {"s1": {"A": 2}}},
            {"s1": "A"},
            {"s1": "A", "s2": "bad"},
        ):
            bb_m = mdm.BlackBoxSampledBackend(
                "iter4_multi_bad",
                sampler=lambda _s, _qs, _im=None, _mb=multi_bad: _mb,
                num_samples=1,
            )
            with self.assertRaises(mdm.SchemaContractViolationError):
                bb_m.score_questions({"x": 1}, q_multi)

        bb_flat_ok = mdm.BlackBoxSampledBackend(
            "iter4_flat_ok",
            sampler=lambda _s, _qs: {"A": 3, "B": 1},
            num_samples=4,
            prior_alpha=0.5,
        )
        p_flat = bb_flat_ok.score_questions({}, q_closed)["sku"]
        self.assertAlmostEqual(p_flat["sku_a"], 3.5 / 5.0, places=9)

        s1_valid = mdm.MockCallableBackend(
            "s1_valid", lambda _s, _qs, _im=None: {"sku": {"sku_a": 0.8, "sku_b": 0.2}}
        )
        bad_ctors = [
            lambda: mdm.BlackBoxSampledBackend("bb", lambda _s, _qs: "A", timeout_sec=True),
            lambda: mdm.BlackBoxSampledBackend("bb", lambda _s, _qs: "A", max_retries=False),
            lambda: mdm.BlackBoxSampledBackend("bb", lambda _s, _qs: "A", base_backoff_sec=True),
            lambda: mdm.BlackBoxSampledBackend("bb", lambda _s, _qs: "A", max_concurrency=False),
            lambda: mdm.DgemSystemOneBackend("https://example.com", timeout_sec=True),
            lambda: mdm.DgemSystemOneBackend("https://example.com", max_retries=False),
            lambda: mdm.DgemSystemOneBackend("https://example.com", base_backoff_sec=True),
            lambda: mdm.OpenAILogprobBackend("https://example.com", "gpt-4o", timeout_sec=True),
            lambda: mdm.OpenAILogprobBackend("https://example.com", "gpt-4o", max_retries=False),
            lambda: mdm.OpenAILogprobBackend("https://example.com", "gpt-4o", base_backoff_sec=True),
            lambda: mdm.OpenAILogprobBackend("https://example.com", "gpt-4o", max_concurrency=False),
            lambda: mdm.OpenAILogprobBackend("https://example.com", "gpt-4o", min_captured_mass=True),
            lambda: mdm.VertexGeminiBackend("proj-1", timeout_sec=True),
            lambda: mdm.VertexGeminiBackend("proj-1", max_retries=False),
            lambda: mdm.VertexGeminiBackend("proj-1", base_backoff_sec=True),
            lambda: mdm.CascadeDecisionOrchestrator(s1_valid, hesitation_threshold=True),
            lambda: mdm.CascadeDecisionOrchestrator(s1_valid, temperature=False),
            lambda: mdm.CascadeDecisionOrchestrator(s1_valid, sla_timeout_sec=True),
            lambda: mdm.CascadeDecisionOrchestrator(s1_valid, conformal_lam_reg=True),
            lambda: mdm.CascadeDecisionOrchestrator(s1_valid, conformal_k_reg=False),
        ]
        for ctor in bad_ctors:
            with self.assertRaises(mdm.SchemaContractViolationError):
                ctor()

    def test_iter4_prior_gate_modes_fast_exit_closure_and_cache_bijection(self) -> None:
        q = mdm.DecisionQuestion(
            type="choice",
            instructions="Select SKU",
            criteria={"sku_a": "A", "sku_b": "B"},
        )
        qs = {"slot": q}
        s1_fn = lambda _s, _q, _i=None: {"slot": {"sku_a": 0.52, "sku_b": 0.48}}
        s2_fn = lambda _s, _q, _i=None: {"slot": {"sku_a": 0.05, "sku_b": 0.95}}
        s1 = mdm.MockCallableBackend("s1", s1_fn)

        # 1. All 6 prior_gate_mode values across calibration, orchestrator, and MCP
        self.assertEqual(
            mdm.VALID_PRIOR_GATE_MODES,
            ("always", "confident", "hesitant", "proportional", "never", "off"),
        )
        for mode in mdm.VALID_PRIOR_GATE_MODES:
            orch = mdm.CascadeDecisionOrchestrator(
                stage1_backend=s1,
                prior_weight=1.0,
                prior_gate_mode=mode,
                total_contract_mode=False,
            )
            res = orch.evaluate({"id": mode}, qs, prior_logits={"slot": {"sku_a": 0.0, "sku_b": 4.0}})
            self.assertIn(res.slots["slot"].predicted, ("sku_a", "sku_b"))
            if mode in ("never", "off"):
                self.assertEqual(res.slots["slot"].predicted, "sku_a")
                self.assertAlmostEqual(res.slots["slot"].probabilities["sku_a"], 0.52, places=9)

            rpc_resp = mdm.handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "evaluate_conformal_decision",
                        "arguments": {
                            "probabilities": {"sku_a": 0.52, "sku_b": 0.48},
                            "q_hat": 0.55,
                            "prior_logits": {"sku_a": 0.0, "sku_b": 4.0},
                            "prior_weight": 1.0,
                            "prior_gate_mode": mode,
                        },
                    },
                }
            )
            assert rpc_resp is not None
            self.assertNotIn("error", rpc_resp)

        bad_rpc = mdm.handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "evaluate_conformal_decision",
                    "arguments": {
                        "probabilities": {"sku_a": 0.52, "sku_b": 0.48},
                        "q_hat": 0.55,
                        "prior_gate_mode": "invalid_gate",
                    },
                },
            }
        )
        assert bad_rpc is not None
        self.assertEqual(bad_rpc["error"]["code"], -32602)

        # 2. Stage-2A fast-exit and Layer-5 domain closure when unlisted SKU is present
        orch_fast = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1,
            enable_stage2a_fast_exit=True,
            total_contract_mode=False,
        )
        res_unlisted = orch_fast.evaluate(
            {"case": "unlisted"},
            qs,
            prior_logits={
                "slot": {"unlisted_sku": 10.0, "sku_a": 0.0, "sku_b": 0.0, "c3": 0.0, "c4": 0.0}
            },
            candidate_pack_sizes={"sku_a": 6, "sku_b": 6, "unlisted_pack12": 12},
            decoded_pack_size=12,
            visual_energies={"sku_a": -0.9, "sku_b": -0.1, "unlisted_r1": -0.95},
            rank1_id="unlisted_r1",
            rank1_visual_similarity=0.95,
        )
        self.assertTrue(mdm.verify_slot_domain_closure(res_unlisted.slots["slot"], q))
        self.assertEqual(res_unlisted.slots["slot"].terminal_state, mdm.DecisionTerminalState.STAGE1_RESOLVED)

        res_listed = orch_fast.evaluate(
            {"case": "listed"},
            qs,
            prior_logits={
                "slot": {"sku_a": 10.0, "sku_b": 0.0, "unlisted_1": 0.0, "unlisted_2": 0.0}
            },
        )
        self.assertTrue(mdm.verify_slot_domain_closure(res_listed.slots["slot"], q))
        self.assertEqual(res_listed.slots["slot"].terminal_state, mdm.DecisionTerminalState.STAGE2A_FAST_EXIT)
        self.assertEqual(set(res_listed.slots["slot"].probabilities.keys()), {"sku_a", "sku_b"})
        self.assertAlmostEqual(sum(res_listed.slots["slot"].probabilities.values()), 1.0, places=9)

        # 3. Cache-key disambiguation: stage2_backend=None vs Backend("none") and "->" delimiter
        cache = mdm.ContentAddressedDecisionCache()
        s2_named_none = mdm.MockCallableBackend("none", s2_fn)
        o_no_s2 = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1, stage2_backend=None, hesitation_threshold=0.10, response_cache=cache
        )
        o_named_none = mdm.CascadeDecisionOrchestrator(
            stage1_backend=s1,
            stage2_backend=s2_named_none,
            hesitation_threshold=0.10,
            response_cache=cache,
        )
        r_no_s2 = o_no_s2.evaluate({"id": 99}, qs)
        r_named_none = o_named_none.evaluate({"id": 99}, qs)
        self.assertNotEqual(r_no_s2.cache_key, r_named_none.cache_key)
        self.assertEqual(r_no_s2.slots["slot"].predicted, "sku_a")
        self.assertEqual(r_named_none.slots["slot"].predicted, "sku_b")

        o_delim_1 = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("a->b", s1_fn),
            stage2_backend=mdm.MockCallableBackend("c", s2_fn),
        )
        o_delim_2 = mdm.CascadeDecisionOrchestrator(
            stage1_backend=mdm.MockCallableBackend("a", s1_fn),
            stage2_backend=mdm.MockCallableBackend("b->c", s2_fn),
        )
        self.assertNotEqual(
            o_delim_1.evaluate({"id": 1}, qs).cache_key,
            o_delim_2.evaluate({"id": 1}, qs).cache_key,
        )

        # 4. compute_decision_cache_key with "__mdm_type__" + conformal_q_hat=inf
        key_inf = mdm.compute_decision_cache_key(
            "s1->none",
            {"x": 1},
            qs,
            extra_config={"__mdm_type__": "user_tag", "conformal_q_hat": float("inf")},
        )
        self.assertEqual(len(key_inf), 64)
        with self.assertRaises(mdm.SchemaContractViolationError):
            mdm.compute_decision_cache_key(
                "s1->none",
                {"x": 1},
                qs,
                extra_config={"__mdm_type__": "user_tag", "other": float("inf")},
            )

    def test_mcp_evaluate_conformal_decision_small_calib_qhat_inf_rfc8259_and_roundtrip(self) -> None:
        from multimodal_decision_models.mcp_server import handle_jsonrpc_message, serve_stdio

        def _reject_const(tok: str) -> None:
            raise ValueError(f"Non-RFC-8259 JSON token: {tok}")

        for n_cal, alpha_val in ((2, 0.05), (5, 0.10)):
            req = {
                "jsonrpc": "2.0",
                "id": n_cal,
                "method": "tools/call",
                "params": {
                    "name": "evaluate_conformal_decision",
                    "arguments": {
                        "probabilities": {"a": 0.9, "b": 0.1},
                        "calib_probs": [{"a": 0.8, "b": 0.2} for _ in range(n_cal)],
                        "calib_labels": ["a" for _ in range(n_cal)],
                        "alpha": alpha_val,
                        "method": "lac",
                    },
                },
            }
            in_buf = io.BytesIO((json.dumps(req, allow_nan=False) + "\n").encode("utf-8"))
            out_buf = io.BytesIO()
            self.assertEqual(serve_stdio(in_buf, out_buf), 0)
            wire_text = out_buf.getvalue().decode("utf-8")
            self.assertNotIn("Infinity", wire_text)
            outer = json.loads(wire_text, parse_constant=_reject_const)
            inner = json.loads(outer["result"]["content"][0]["text"], parse_constant=_reject_const)
            self.assertIsNone(inner["q_hat"])
            self.assertIs(inner["q_hat_infinite"], True)
            self.assertEqual(inner["prediction_set"], ["a", "b"])
            self.assertTrue(inner["should_escalate"])

            rt = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 100 + n_cal,
                    "method": "tools/call",
                    "params": {
                        "name": "evaluate_conformal_decision",
                        "arguments": {
                            "probabilities": {"a": 0.9, "b": 0.1},
                            "q_hat": inner["q_hat"],
                            "q_hat_infinite": inner["q_hat_infinite"],
                            "method": "lac",
                        },
                    },
                }
            )
            assert rt is not None
            rt_inner = json.loads(rt["result"]["content"][0]["text"], parse_constant=_reject_const)
            self.assertEqual(rt_inner, inner)

        for bad_args in (
            {"probabilities": {"a": 0.9, "b": 0.1}, "q_hat": 0.2, "q_hat_infinite": True},
            {"probabilities": {"a": 0.9, "b": 0.1}, "q_hat": None, "q_hat_infinite": False},
            {"probabilities": {"a": 0.9, "b": 0.1}, "q_hat": None},
        ):
            err = handle_jsonrpc_message(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {"name": "evaluate_conformal_decision", "arguments": bad_args},
                }
            )
            assert err is not None
            self.assertEqual(err["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()

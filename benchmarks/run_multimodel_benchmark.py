#!/usr/bin/env python3
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

"""Deterministic Multi-Model Head-to-Head Calibration, Conformal, and Cascade Benchmark.

Evaluates 10 distinct real decision and frontier models plus out-of-fold post-hoc calibration
and pre-fusion split conformal RAPS cascades across two real-probability receipt panels:
  1. Panel 1 (`multimodel_calibration_50`, N = 50 items, 15 tasks, K in {2, 3, 4, 5, 12, 26}):
     DecisionGemma, dgem v0.3.7, Strands Decider 2B v19, Gemini 3.8 Flash (Verbalized Conf),
     Gemini 3.5 Flash Lite (Dirichlet M=3), Gemini 3.6 Flash (Dirichlet M=3),
     Gemini 3.8 Flash (Dirichlet M=3), out-of-fold Temperature Scaling arms, and
     5-fold out-of-fold Split Conformal RAPS cascades.
  2. Panel 2 (`safety_xstest_360`, N = 360 items, 12 XSTest prompt-type clusters, K = 2):
     DecisionGemma, dgem v0.3.1, Strands Decider 2B v19, Classy v0.3.3, Classier v0.2.0,
     out-of-fold Vector Scaling arms, and 5-fold out-of-fold Split Conformal RAPS cascades.

Zero network calls and zero synthetic RNG draws occur in this benchmark runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_decision_models import (  # noqa: E402
    adaptive_calibration_error,
    brier_murphy_decomposition,
    brier_score_multiclass,
    calibrate_conformal_quantile,
    cluster_bootstrap_ci,
    conformal_prediction_set,
    expected_calibration_error,
    fit_temperature_scaling,
    fit_vector_scaling,
    macro_f1_score,
    ndcg_at_k,
    negative_log_likelihood,
    normalized_entropy,
    paired_mcnemar_test,
    selective_risk_coverage_aurc,
)

CLOSED_SET_ABSTAIN_KEYS: frozenset[str] = frozenset(
    {"none", "none_of_the_above", "none_visible", "oos", "abstain"}
)

PANEL1_STANDALONE_ORDER: tuple[str, ...] = (
    "DecisionGemma",
    "dgem v0.3.7",
    "Strands Decider 2B v19",
    "Gemini 3.8 Flash (Verbalized Conf)",
    "Gemini 3.5 Flash Lite (Dirichlet M=3)",
    "Gemini 3.6 Flash (Dirichlet M=3)",
    "Gemini 3.8 Flash (Dirichlet M=3)",
)

PANEL2_STANDALONE_ORDER: tuple[str, ...] = (
    "DecisionGemma",
    "dgem v0.3.1",
    "Strands Decider 2B v19",
    "Classier v0.2.0",
    "Classy v0.3.3",
)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load and deterministically sort JSONL receipt rows by (cluster, id)."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark receipt file: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Receipt file is empty: {path}")
    rows.sort(key=lambda r: (str(r["cluster"]), str(r["id"])))
    return rows


def infer_model_family(arm_name: str) -> str:
    """Classify an evaluation arm into its primary model family."""
    low = arm_name.lower()
    if low.startswith("conformal cascade:"):
        return "conformal_cascade"
    if "decisiongemma" in low:
        return "gemma_systemone"
    if "dgem" in low:
        return "dgem_systemone"
    if "strands" in low:
        return "strands_systemone"
    if "classier" in low:
        return "classier_embedding"
    if "classy" in low:
        return "classy_embedding"
    if "gemini" in low:
        return "gemini"
    return "other"


def top1_label(probs: Mapping[str, float]) -> str:
    """Return deterministic top-1 option key with lexicographic tie-break."""
    return sorted(probs.keys(), key=lambda k: (-float(probs[k]), str(k)))[0]


def build_oof_temperature_scaled_arm(
    rows: Sequence[Mapping[str, Any]],
    base_model: str,
    num_folds: int = 5,
    l2_reg: float = 1e-2,
) -> tuple[list[dict[str, float]], list[float], float]:
    """Fit 5-fold out-of-fold scalar temperature scaling without test-fold leakage."""
    n = len(rows)
    oof_probs: list[dict[str, float]] = [{} for _ in range(n)]
    latencies = [float(r["models"][base_model]["latency_ms"]) for r in rows]
    learned_temps: list[float] = []

    for fold in range(num_folds):
        train_p = [
            rows[i]["models"][base_model]["probabilities"]
            for i in range(n)
            if (i % num_folds) != fold
        ]
        train_y = [str(rows[i]["gold"]) for i in range(n) if (i % num_folds) != fold]
        test_indices = [i for i in range(n) if (i % num_folds) == fold]
        if not test_indices:
            continue
        ts_model = fit_temperature_scaling(train_p, train_y, is_logits=False, l2_reg=l2_reg)
        learned_temps.append(ts_model.temperature)
        test_p = [rows[i]["models"][base_model]["probabilities"] for i in test_indices]
        cal_p = ts_model.predict_proba(test_p, is_logits=False)
        for idx, dist in zip(test_indices, cal_p):
            oof_probs[idx] = dist

    mean_t = sum(learned_temps) / float(len(learned_temps))
    return oof_probs, latencies, mean_t


def build_oof_vector_scaled_arm(
    rows: Sequence[Mapping[str, Any]],
    base_model: str,
    num_folds: int = 5,
    l2_reg: float = 0.15,
) -> tuple[list[dict[str, float]], list[float]]:
    """Fit 5-fold out-of-fold diagonal vector scaling on homogeneous label-space panels."""
    n = len(rows)
    oof_probs: list[dict[str, float]] = [{} for _ in range(n)]
    latencies = [float(r["models"][base_model]["latency_ms"]) for r in rows]

    for fold in range(num_folds):
        train_p = [
            rows[i]["models"][base_model]["probabilities"]
            for i in range(n)
            if (i % num_folds) != fold
        ]
        train_y = [str(rows[i]["gold"]) for i in range(n) if (i % num_folds) != fold]
        test_indices = [i for i in range(n) if (i % num_folds) == fold]
        if not test_indices:
            continue
        vs_model = fit_vector_scaling(train_p, train_y, is_logits=False, l2_reg=l2_reg)
        test_p = [rows[i]["models"][base_model]["probabilities"] for i in test_indices]
        cal_p = vs_model.predict_proba(test_p, is_logits=False)
        for idx, dist in zip(test_indices, cal_p):
            oof_probs[idx] = dist

    return oof_probs, latencies


def build_oof_conformal_cascade_arm(
    rows: Sequence[Mapping[str, Any]],
    stage1_model: str,
    stage2_model: str,
    alpha_gate: float = 0.10,
    num_folds: int = 5,
) -> tuple[list[dict[str, float]], list[float], float]:
    """Route items using a 5-fold out-of-fold pre-fusion Split Conformal RAPS set-size gate.

    Fast-exits with Stage 1 when |C_alpha(x)| == 1 and escalates non-singleton prediction sets
    to Stage 2, incurring Stage-1 + Stage-2 wall latency on escalated items.
    """
    n = len(rows)
    routed_probs: list[dict[str, float]] = [{} for _ in range(n)]
    routed_latencies: list[float] = [0.0 for _ in range(n)]
    escalated_count = 0

    for fold in range(num_folds):
        train_p = [
            rows[i]["models"][stage1_model]["probabilities"]
            for i in range(n)
            if (i % num_folds) != fold
        ]
        train_y = [str(rows[i]["gold"]) for i in range(n) if (i % num_folds) != fold]
        cal = calibrate_conformal_quantile(
            train_p,
            train_y,
            alpha=alpha_gate,
            method="raps",
            lam_reg=0.01,
            k_reg=1,
            randomized=False,
        )
        for i in range(n):
            if (i % num_folds) != fold:
                continue
            s1_obj = rows[i]["models"][stage1_model]
            s2_obj = rows[i]["models"][stage2_model]
            s1_probs = s1_obj["probabilities"]
            c_set = conformal_prediction_set(
                s1_probs,
                cal.q_hat,
                method="raps",
                lam_reg=0.01,
                k_reg=1,
                randomized=False,
                abstain_keys=CLOSED_SET_ABSTAIN_KEYS,
            )
            if c_set.should_escalate:
                escalated_count += 1
                routed_probs[i] = dict(s2_obj["probabilities"])
                routed_latencies[i] = float(s1_obj["latency_ms"]) + float(s2_obj["latency_ms"])
            else:
                routed_probs[i] = dict(s1_probs)
                routed_latencies[i] = float(s1_obj["latency_ms"])

    esc_rate = float(escalated_count) / float(n)
    return routed_probs, routed_latencies, esc_rate


def evaluate_oof_conformal(
    probs_list: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    alpha: float,
    num_folds: int = 5,
) -> dict[str, float]:
    """Compute 5-fold out-of-fold Split Conformal RAPS coverage, mean set size, and singleton rate."""
    n = len(probs_list)
    covered = 0
    total_size = 0
    singletons = 0
    q_hats: list[float] = []

    for fold in range(num_folds):
        train_p = [probs_list[i] for i in range(n) if (i % num_folds) != fold]
        train_y = [golds[i] for i in range(n) if (i % num_folds) != fold]
        cal = calibrate_conformal_quantile(
            train_p,
            train_y,
            alpha=alpha,
            method="raps",
            lam_reg=0.01,
            k_reg=1,
            randomized=False,
        )
        if math.isfinite(cal.q_hat):
            q_hats.append(cal.q_hat)
        for i in range(n):
            if (i % num_folds) != fold:
                continue
            res = conformal_prediction_set(
                probs_list[i],
                cal.q_hat,
                method="raps",
                lam_reg=0.01,
                k_reg=1,
                randomized=False,
                abstain_keys=CLOSED_SET_ABSTAIN_KEYS,
            )
            if golds[i] in res.prediction_set:
                covered += 1
            total_size += len(res.prediction_set)
            if res.singleton:
                singletons += 1

    mean_q = (sum(q_hats) / float(len(q_hats))) if q_hats else 1.0
    return {
        "alpha": alpha,
        "coverage": round(float(covered) / float(n), 6),
        "mean_set_size": round(float(total_size) / float(n), 6),
        "singleton_rate": round(float(singletons) / float(n), 6),
        "mean_q_hat": round(mean_q, 6),
    }


def compute_macro_f1_by_metric(
    preds: Sequence[str],
    golds: Sequence[str],
    metrics: Sequence[str],
) -> float:
    """Compute unweighted Macro-F1 per metric schema and average across metrics."""
    grouped_p: dict[str, list[str]] = {}
    grouped_g: dict[str, list[str]] = {}
    for p, g, m in zip(preds, golds, metrics):
        grouped_p.setdefault(m, []).append(p)
        grouped_g.setdefault(m, []).append(g)
    scores = [macro_f1_score(grouped_p[m], grouped_g[m]) for m in sorted(grouped_p.keys())]
    return sum(scores) / float(len(scores))


def score_arm(
    panel_id: str,
    panel_short: str,
    arm_name: str,
    rows: Sequence[Mapping[str, Any]],
    probs_list: Sequence[Mapping[str, float]],
    latencies_ms: Sequence[float],
    reference_correct: Sequence[bool],
    extra_meta: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[bool]]:
    """Evaluate all proper scoring rules, conformal sets, paired McNemar, and cluster bootstrap CIs."""
    golds = [str(r["gold"]) for r in rows]
    clusters = [str(r["cluster"]) for r in rows]
    metrics = [str(r["metric"]) for r in rows]

    pred_labels = [top1_label(p) for p in probs_list]
    is_correct = [pred == gold for pred, gold in zip(pred_labels, golds)]
    acc = sum(1.0 for ok in is_correct if ok) / float(len(is_correct))

    brier = brier_score_multiclass(probs_list, golds)
    murphy = brier_murphy_decomposition(probs_list, golds, num_bins=10)
    nll = negative_log_likelihood(probs_list, golds)
    ece = expected_calibration_error(probs_list, golds, num_bins=10)
    ace = adaptive_calibration_error(probs_list, golds, num_bins=10)

    hesitations = [normalized_entropy(p) for p in probs_list]
    aurc = selective_risk_coverage_aurc(hesitations, is_correct)
    f1_macro = compute_macro_f1_by_metric(pred_labels, golds, metrics)

    ndcg_vals: list[float] = []
    for p, g in zip(probs_list, golds):
        ranked = sorted(p.keys(), key=lambda k: (-float(p[k]), str(k)))
        ndcg_vals.append(ndcg_at_k(ranked, {g: 1.0}, k=3))
    mean_ndcg3 = sum(ndcg_vals) / float(len(ndcg_vals))

    conf_005 = evaluate_oof_conformal(probs_list, golds, alpha=0.05, num_folds=5)
    conf_010 = evaluate_oof_conformal(probs_list, golds, alpha=0.10, num_folds=5)
    if conf_005["coverage"] + 1e-9 < conf_010["coverage"]:
        raise AssertionError(
            f"Conformal monotonicity violated on {arm_name}: "
            f"cov@0.05={conf_005['coverage']} < cov@0.10={conf_010['coverage']}"
        )

    acc_by_cluster: dict[str, list[float]] = {}
    brier_by_cluster: dict[str, list[float]] = {}
    for cl, ok, p, g in zip(clusters, is_correct, probs_list, golds):
        acc_by_cluster.setdefault(cl, []).append(1.0 if ok else 0.0)
        brier_by_cluster.setdefault(cl, []).append(brier_score_multiclass([p], [g]))

    _, acc_lo, acc_hi = cluster_bootstrap_ci(acc_by_cluster, n_boot=500, seed=42)
    _, brier_lo, brier_hi = cluster_bootstrap_ci(brier_by_cluster, n_boot=500, seed=42)

    mcn = paired_mcnemar_test(is_correct, reference_correct)

    sorted_lat = sorted(float(x) for x in latencies_ms)
    mean_lat = sum(sorted_lat) / float(len(sorted_lat))
    p50_lat = sorted_lat[len(sorted_lat) // 2]
    p95_lat = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * (len(sorted_lat) - 1)))]

    ep_vars: list[float] = []
    if arm_name in rows[0]["models"]:
        for r in rows:
            ev = r["models"][arm_name].get("epistemic_variance")
            if isinstance(ev, (int, float)):
                ep_vars.append(float(ev))

    entry: dict[str, Any] = {
        "model": f"{arm_name} [{panel_short}]",
        "arm_name": arm_name,
        "family": infer_model_family(arm_name),
        "panel": panel_id,
        "n_items": len(rows),
        "n_clusters": len(set(clusters)),
        "accuracy": round(acc, 6),
        "macro_f1": round(f1_macro, 6),
        "ndcg_at_3": round(mean_ndcg3, 6),
        "brier": round(brier, 6),
        "murphy_reliability": round(murphy.reliability, 6),
        "murphy_resolution": round(murphy.resolution, 6),
        "murphy_uncertainty": round(murphy.uncertainty, 6),
        "nll": round(nll, 6),
        "ece": round(ece, 6),
        "ace": round(ace, 6),
        "aurc": round(aurc, 6),
        "mean_normalized_entropy": round(sum(hesitations) / float(len(hesitations)), 6),
        "conformal_coverage_005": conf_005["coverage"],
        "conformal_coverage_010": conf_010["coverage"],
        "conformal_mean_set_size_005": conf_005["mean_set_size"],
        "conformal_mean_set_size_010": conf_010["mean_set_size"],
        "conformal": {
            "0.05": conf_005,
            "0.10": conf_010,
        },
        "bootstrap_ci95": {
            "accuracy": [round(acc_lo, 6), round(acc_hi, 6)],
            "brier": [round(brier_lo, 6), round(brier_hi, 6)],
        },
        "mcnemar_vs_reference": {
            "reference_arm": "DecisionGemma",
            "n_pairs": mcn.n_pairs,
            "n01_arm_wins": mcn.n01_arm_wins,
            "n10_reference_wins": mcn.n10_baseline_wins,
            "chi2_cc": round(mcn.chi2_cc, 6),
            "p_two_sided": float(f"{mcn.p_two_sided:.6g}"),
        },
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "pareto_optimal": False,
    }
    if ep_vars:
        entry["mean_epistemic_variance"] = round(sum(ep_vars) / float(len(ep_vars)), 8)
    if extra_meta:
        entry.update(extra_meta)
    return entry, is_correct


def mark_pareto_frontier(arms: list[dict[str, Any]]) -> list[str]:
    """Identify arms on the (Brier ASC, Mean Latency ASC) Pareto frontier."""
    frontier_names: list[str] = []
    for idx, cand in enumerate(arms):
        dominated = False
        for jdx, other in enumerate(arms):
            if idx == jdx:
                continue
            better_or_eq_brier = other["brier"] <= cand["brier"]
            better_or_eq_lat = other["mean_latency_ms"] <= cand["mean_latency_ms"]
            strictly_better = (
                other["brier"] < cand["brier"] - 1e-9
                or other["mean_latency_ms"] < cand["mean_latency_ms"] - 1e-6
            )
            if better_or_eq_brier and better_or_eq_lat and strictly_better:
                dominated = True
                break
        cand["pareto_optimal"] = not dominated
        if not dominated:
            frontier_names.append(str(cand["arm_name"]))
    return frontier_names


def evaluate_panel1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate all standalone, OOF temperature-scaled, and OOF conformal cascade arms on Panel 1."""
    ref_probs = [r["models"]["DecisionGemma"]["probabilities"] for r in rows]
    ref_correct = [top1_label(p) == str(r["gold"]) for p, r in zip(ref_probs, rows)]

    arms: list[dict[str, Any]] = []
    correct_by_arm: dict[str, list[bool]] = {}

    for model_name in PANEL1_STANDALONE_ORDER:
        p_list = [r["models"][model_name]["probabilities"] for r in rows]
        l_list = [float(r["models"][model_name]["latency_ms"]) for r in rows]
        entry, ok_vec = score_arm(
            "multimodel_calibration_50",
            "Calib-50",
            model_name,
            rows,
            p_list,
            l_list,
            ref_correct,
            extra_meta={"kind": "standalone"},
        )
        arms.append(entry)
        correct_by_arm[model_name] = ok_vec

    for base_m, cal_label in (
        ("Gemini 3.8 Flash (Verbalized Conf)", "Gemini 3.8 Flash (Verbalized + TempScaled OOF)"),
        ("dgem v0.3.7", "dgem v0.3.7 (TempScaled OOF)"),
    ):
        oof_p, oof_l, mean_t = build_oof_temperature_scaled_arm(rows, base_m, num_folds=5)
        entry, ok_vec = score_arm(
            "multimodel_calibration_50",
            "Calib-50",
            cal_label,
            rows,
            oof_p,
            oof_l,
            ref_correct,
            extra_meta={"kind": "posthoc_temperature_oof", "base_model": base_m, "mean_temperature": round(mean_t, 4)},
        )
        arms.append(entry)
        correct_by_arm[cal_label] = ok_vec

    for s1, s2, casc_label in (
        (
            "dgem v0.3.7",
            "Gemini 3.8 Flash (Dirichlet M=3)",
            "Conformal Cascade: dgem v0.3.7 -> Gemini 3.8 Flash (Dirichlet M=3)",
        ),
        (
            "Strands Decider 2B v19",
            "DecisionGemma",
            "Conformal Cascade: Strands Decider 2B v19 -> DecisionGemma",
        ),
    ):
        c_probs, c_lat, esc_rate = build_oof_conformal_cascade_arm(rows, s1, s2, alpha_gate=0.10, num_folds=5)
        mcn_s1 = paired_mcnemar_test(
            [top1_label(p) == str(r["gold"]) for p, r in zip(c_probs, rows)],
            correct_by_arm[s1],
        )
        entry, _ = score_arm(
            "multimodel_calibration_50",
            "Calib-50",
            casc_label,
            rows,
            c_probs,
            c_lat,
            ref_correct,
            extra_meta={
                "kind": "conformal_raps_cascade_oof",
                "stage1_model": s1,
                "stage2_model": s2,
                "escalation_rate": round(esc_rate, 6),
                "mcnemar_vs_stage1": {
                    "stage1_arm": s1,
                    "n01_cascade_wins": mcn_s1.n01_arm_wins,
                    "n10_stage1_wins": mcn_s1.n10_baseline_wins,
                    "chi2_cc": round(mcn_s1.chi2_cc, 6),
                    "p_two_sided": float(f"{mcn_s1.p_two_sided:.6g}"),
                },
            },
        )
        arms.append(entry)

    frontier = mark_pareto_frontier(arms)
    arms.sort(key=lambda a: (float(a["brier"]), -float(a["accuracy"]), str(a["arm_name"])))
    return {
        "panel_id": "multimodel_calibration_50",
        "description": "50-item multi-task calibration suite across 15 public tasks (K in {2, 3, 4, 5, 12, 26})",
        "n_items": len(rows),
        "n_clusters": len({str(r["cluster"]) for r in rows}),
        "pareto_frontier_brier_latency": frontier,
        "arms": arms,
    }


def evaluate_panel2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate standalone, OOF vector-scaled, and OOF conformal cascade arms on Panel 2 (XSTest N=360)."""
    ref_probs = [r["models"]["DecisionGemma"]["probabilities"] for r in rows]
    ref_correct = [top1_label(p) == str(r["gold"]) for p, r in zip(ref_probs, rows)]

    arms: list[dict[str, Any]] = []
    correct_by_arm: dict[str, list[bool]] = {}

    for model_name in PANEL2_STANDALONE_ORDER:
        p_list = [r["models"][model_name]["probabilities"] for r in rows]
        l_list = [float(r["models"][model_name]["latency_ms"]) for r in rows]
        entry, ok_vec = score_arm(
            "safety_xstest_360",
            "XSTest-360",
            model_name,
            rows,
            p_list,
            l_list,
            ref_correct,
            extra_meta={"kind": "standalone"},
        )
        arms.append(entry)
        correct_by_arm[model_name] = ok_vec

    for base_m, cal_label in (
        ("Classy v0.3.3", "Classy v0.3.3 (VectorScaled OOF)"),
        ("Strands Decider 2B v19", "Strands Decider 2B v19 (VectorScaled OOF)"),
    ):
        oof_p, oof_l = build_oof_vector_scaled_arm(rows, base_m, num_folds=5)
        entry, ok_vec = score_arm(
            "safety_xstest_360",
            "XSTest-360",
            cal_label,
            rows,
            oof_p,
            oof_l,
            ref_correct,
            extra_meta={"kind": "posthoc_vector_oof", "base_model": base_m},
        )
        arms.append(entry)
        correct_by_arm[cal_label] = ok_vec

    for s1, s2, casc_label in (
        (
            "Classier v0.2.0",
            "DecisionGemma",
            "Conformal Cascade: Classier v0.2.0 -> DecisionGemma",
        ),
        (
            "Classy v0.3.3",
            "DecisionGemma",
            "Conformal Cascade: Classy v0.3.3 -> DecisionGemma",
        ),
    ):
        c_probs, c_lat, esc_rate = build_oof_conformal_cascade_arm(rows, s1, s2, alpha_gate=0.10, num_folds=5)
        mcn_s1 = paired_mcnemar_test(
            [top1_label(p) == str(r["gold"]) for p, r in zip(c_probs, rows)],
            correct_by_arm[s1],
        )
        entry, _ = score_arm(
            "safety_xstest_360",
            "XSTest-360",
            casc_label,
            rows,
            c_probs,
            c_lat,
            ref_correct,
            extra_meta={
                "kind": "conformal_raps_cascade_oof",
                "stage1_model": s1,
                "stage2_model": s2,
                "escalation_rate": round(esc_rate, 6),
                "mcnemar_vs_stage1": {
                    "stage1_arm": s1,
                    "n01_cascade_wins": mcn_s1.n01_arm_wins,
                    "n10_stage1_wins": mcn_s1.n10_baseline_wins,
                    "chi2_cc": round(mcn_s1.chi2_cc, 6),
                    "p_two_sided": float(f"{mcn_s1.p_two_sided:.6g}"),
                },
            },
        )
        arms.append(entry)

    frontier = mark_pareto_frontier(arms)
    arms.sort(key=lambda a: (float(a["brier"]), -float(a["accuracy"]), str(a["arm_name"])))
    return {
        "panel_id": "safety_xstest_360",
        "description": "360-item XSTest contrast and homonym refusal-boundary benchmark across 12 prompt-type clusters (K = 2)",
        "n_items": len(rows),
        "n_clusters": len({str(r["cluster"]) for r in rows}),
        "pareto_frontier_brier_latency": frontier,
        "arms": arms,
    }


def _fmt_p(p_val: float) -> str:
    """Format exact p-values cleanly across regular and extreme tail regimes."""
    return f"{p_val:.2e}" if p_val < 1e-4 else f"{p_val:.4f}"


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    """Render a clean, anti-slop Markdown report with zero dashes or horizontal dividers."""
    p1 = summary["panels"]["multimodel_calibration_50"]
    p2 = summary["panels"]["safety_xstest_360"]
    m_cnt = summary["distinct_model_count"]

    lines: list[str] = [
        "# Multi-Model Head-to-Head Calibration and Conformal Benchmark",
        "",
        f"This report compares {m_cnt} real decision and frontier models plus 5-fold out-of-fold post-hoc calibration "
        "and pre-fusion Split Conformal RAPS cascades across two evaluation panels:",
        "",
        f"1. **Multi-Task Calibration Panel (`N = {p1['n_items']}` items, `{p1['n_clusters']}` clusters, `K` from 2 to 26)**: "
        "evaluates parallel masked diffusion (`/v1/systemone`), autoregressive verbalized confidence, and 3-draw "
        "symmetric Dirichlet (`alpha_0 = 0.5`) black-box consensus models.",
        f"2. **XSTest Safety Refusal Boundary Panel (`N = {p2['n_items']}` items, `{p2['n_clusters']}` prompt-type clusters, `K = 2`)**: "
        "evaluates embedding probes (`Classy v0.3.3`, `Classier v0.2.0`), compact 2B deciders (`Strands Decider 2B v19`), "
        "and masked diffusion verifiers (`dgem v0.3.1`, `DecisionGemma`).",
        "",
        "## Panel 1: Multi-Task Calibration Leaderboard (`N = 50`)",
        "",
        "| Arm | Acc | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Mean Latency (ms) | Pareto |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for a in p1["arms"]:
        b_lo, b_hi = a["bootstrap_ci95"]["brier"]
        m_p = float(a["mcnemar_vs_reference"]["p_two_sided"])
        pareto_str = "yes" if a["pareto_optimal"] else "no"
        lines.append(
            f"| `{a['arm_name']}` | {a['accuracy']:.3f} | "
            f"{a['brier']:.4f} ([{b_lo:.4f}, {b_hi:.4f}]) | "
            f"{a['murphy_reliability']:.4f} / {a['murphy_resolution']:.4f} | "
            f"{a['nll']:.4f} | {a['ece']:.4f} | {a['ace']:.4f} | {a['aurc']:.4f} | "
            f"{a['conformal_coverage_005']:.3f} / {a['conformal_coverage_010']:.3f} | "
            f"{a['conformal_mean_set_size_010']:.2f} | {_fmt_p(m_p)} | "
            f"{a['mean_latency_ms']:.1f} | {pareto_str} |"
        )

    lines.extend(
        [
            "",
            f"**Panel 1 Brier-Latency Pareto Frontier**: {', '.join(f'`{n}`' for n in p1['pareto_frontier_brier_latency'])}.",
            "",
            "## Panel 2: XSTest Safety Boundary Leaderboard (`N = 360`)",
            "",
            "| Arm | Acc | Macro-F1 | Brier (95% Bootstrap CI) | Murphy REL / RES | NLL | ECE | ACE | AURC | Conformal Cov@0.05 / @0.10 | Mean Set@0.10 | McNemar p (vs Ref) | Latency (ms) | Pareto |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for a in p2["arms"]:
        b_lo, b_hi = a["bootstrap_ci95"]["brier"]
        m_p = float(a["mcnemar_vs_reference"]["p_two_sided"])
        pareto_str = "yes" if a["pareto_optimal"] else "no"
        lines.append(
            f"| `{a['arm_name']}` | {a['accuracy']:.4f} | {a['macro_f1']:.4f} | "
            f"{a['brier']:.4f} ([{b_lo:.4f}, {b_hi:.4f}]) | "
            f"{a['murphy_reliability']:.4f} / {a['murphy_resolution']:.4f} | "
            f"{a['nll']:.4f} | {a['ece']:.4f} | {a['ace']:.4f} | {a['aurc']:.4f} | "
            f"{a['conformal_coverage_005']:.4f} / {a['conformal_coverage_010']:.4f} | "
            f"{a['conformal_mean_set_size_010']:.2f} | {_fmt_p(m_p)} | "
            f"{a['mean_latency_ms']:.1f} | {pareto_str} |"
        )

    lines.extend(
        [
            "",
            f"**Panel 2 Brier-Latency Pareto Frontier**: {', '.join(f'`{n}`' for n in p2['pareto_frontier_brier_latency'])}.",
            "",
            "## Conformal Cascade Escalation Analysis",
            "",
        ]
    )

    for panel_obj in (p1, p2):
        for a in panel_obj["arms"]:
            if a.get("kind") == "conformal_raps_cascade_oof":
                m_s1 = a["mcnemar_vs_stage1"]
                p_s1 = float(m_s1["p_two_sided"])
                lines.append(
                    f"- **`{a['arm_name']}` (`{panel_obj['panel_id']}`)**: escalates "
                    f"`{a['escalation_rate'] * 100.0:.1f}%` of items where Stage-1 RAPS (`alpha = 0.10`) "
                    f"produces a non-singleton set, reaching accuracy `{a['accuracy']:.4f}`, "
                    f"Brier `{a['brier']:.4f}`, AURC `{a['aurc']:.4f}`, and discordant Stage-1 wins/losses "
                    f"`{m_s1['n01_cascade_wins']}/{m_s1['n10_stage1_wins']}` "
                    f"(exact two-sided McNemar `p = {_fmt_p(p_s1)}`) at `{a['mean_latency_ms']:.1f} ms` mean latency."
                )

    lines.extend(
        [
            "",
            "## Reproducibility Command",
            "",
            "```bash",
            "PYTHONPATH=src python3 benchmarks/run_multimodel_benchmark.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def verify_artifact_hygiene_and_antislop(json_text: str, md_text: str) -> None:
    """Self-audit generated JSON and Markdown artifacts for numeric validity and formatting."""
    if "NaN" in json_text or "Infinity" in json_text:
        raise AssertionError("JSON summary contains non-finite float tokens")
    if "\u2014" in md_text or "\u2013" in md_text:
        raise AssertionError("Markdown report contains forbidden em/en dash")
    for line in md_text.splitlines():
        if line.strip() == "---":
            raise AssertionError("Markdown report contains standalone horizontal rule")


def run_benchmark(data_dir: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    """Execute the full deterministic multi-model benchmark and write JSON and Markdown outputs."""
    calib_rows = load_jsonl_rows(data_dir / "multimodel_calibration_receipts.jsonl")
    xstest_rows = load_jsonl_rows(data_dir / "multimodel_safety_xstest_receipts.jsonl")

    p1_summary = evaluate_panel1(calib_rows)
    p2_summary = evaluate_panel2(xstest_rows)

    distinct_standalone_models = sorted(set(PANEL1_STANDALONE_ORDER) | set(PANEL2_STANDALONE_ORDER))
    combined_arms = list(p1_summary["arms"]) + list(p2_summary["arms"])

    results_payload: dict[str, Any] = {
        "schema_version": "multimodal_decision_models.benchmark.v0.4.0",
        "protocol": {
            "conformal_method": "raps",
            "conformal_alphas": [0.05, 0.10],
            "out_of_fold_splits": 5,
            "bootstrap_resamples": 500,
            "bootstrap_seed": 42,
            "murphy_bins": 10,
            "ece_bins": 10,
            "ace_bins": 10,
        },
        "distinct_standalone_models": distinct_standalone_models,
        "distinct_model_count": len(distinct_standalone_models),
        "total_evaluated_arms": len(combined_arms),
        "models": combined_arms,
        "panels": {
            "multimodel_calibration_50": p1_summary,
            "safety_xstest_360": p2_summary,
        },
    }

    json_text = json.dumps(results_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    md_text = render_markdown_report(results_payload)
    verify_artifact_hygiene_and_antislop(json_text, md_text)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json_text, encoding="utf-8")
    output_md.write_text(md_text, encoding="utf-8")
    return results_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic multi-model calibration, conformal, and cascade benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BENCH_DIR / "data",
        help="Directory containing multimodel_*_receipts.jsonl files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BENCH_DIR / "results" / "multimodel_benchmark_results.json",
        help="Destination path for structured JSON benchmark summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=BENCH_DIR / "results" / "multimodel_benchmark_report.md",
        help="Destination path for Markdown benchmark report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args.data_dir, args.output_json, args.output_md)
    print(
        f"Evaluated {summary['distinct_model_count']} distinct models across "
        f"{summary['total_evaluated_arms']} total arms -> {args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

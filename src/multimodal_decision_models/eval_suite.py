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

"""Proper Scoring Rules, Calibration Metrics, and Paired Statistical Inference Suite.

Provides zero-dependency implementations for Tier-2 and Tier-3 LLM/VLM evaluation:
  - Multiclass Brier Score and Negative Log-Likelihood (NLL)
  - Expected Calibration Error (ECE)
  - Binary AUROC (Wilcoxon-Mann-Whitney tie-aware exact concordance)
  - Selective Risk-Coverage Area (AURC) for hesitation-gated cascades
  - Exact two-sided Paired McNemar test on discordant cascade/treatment pairs
  - Rogan-Gladen prevalence calibration estimator under imperfect sensitivity/specificity
  - Unweighted Cohen's Kappa inter-rater / run-to-run agreement
  - Cluster-robust bootstrap confidence intervals
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class McNemarResult:
    """Two-sided exact binomial and continuity-corrected chi-square Paired McNemar test."""

    n_pairs: int
    n01_arm_wins: int
    n10_baseline_wins: int
    chi2_cc: float
    p_two_sided: float


def _normalize_dist(dist: Mapping[str, float]) -> dict[str, float]:
    if not dist:
        raise ValueError("prediction probability distribution must not be empty")
    tot = 0.0
    for k, v in dist.items():
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(f"invalid probability for {k!r}: {v!r}")
        tot += fv
    if tot <= 0.0:
        raise ValueError("prediction probability distribution sums to zero")
    return {k: float(v) / tot for k, v in dist.items()}


def brier_score_multiclass(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
) -> float:
    """Compute multiclass Brier score mean_i sum_k (p_{i,k} - y_{i,k})^2 in [0, 2]."""
    if len(predictions) != len(golds) or not predictions:
        raise ValueError("predictions and golds must be non-empty sequences of equal length")
    total = 0.0
    for dist_raw, gold in zip(predictions, golds):
        dist = _normalize_dist(dist_raw)
        keys = set(dist.keys()) | {gold}
        total += sum((dist.get(k, 0.0) - (1.0 if k == gold else 0.0)) ** 2 for k in keys)
    return total / len(predictions)


def negative_log_likelihood(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    eps: float = 1e-12,
) -> float:
    """Compute mean multiclass negative log-likelihood -mean_i ln(p_{i, y_i})."""
    if len(predictions) != len(golds) or not predictions:
        raise ValueError("predictions and golds must be non-empty sequences of equal length")
    total = 0.0
    for dist_raw, gold in zip(predictions, golds):
        dist = _normalize_dist(dist_raw)
        p_gold = max(eps, dist.get(gold, 0.0))
        total -= math.log(p_gold)
    return total / len(predictions)


def expected_calibration_error(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    num_bins: int = 10,
) -> float:
    """Compute top-label Expected Calibration Error (ECE) across `num_bins` equal-width bins."""
    if len(predictions) != len(golds) or not predictions:
        raise ValueError("predictions and golds must be non-empty sequences of equal length")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    bin_counts = [0] * num_bins
    bin_conf_sums = [0.0] * num_bins
    bin_acc_sums = [0.0] * num_bins

    for dist_raw, gold in zip(predictions, golds):
        dist = _normalize_dist(dist_raw)
        pred_key, conf = max(dist.items(), key=lambda kv: kv[1])
        idx = min(num_bins - 1, max(0, int(conf * num_bins)))
        bin_counts[idx] += 1
        bin_conf_sums[idx] += conf
        bin_acc_sums[idx] += 1.0 if pred_key == gold else 0.0

    n = float(len(predictions))
    ece = 0.0
    for b in range(num_bins):
        if bin_counts[b] > 0:
            avg_c = bin_conf_sums[b] / bin_counts[b]
            avg_a = bin_acc_sums[b] / bin_counts[b]
            ece += (bin_counts[b] / n) * abs(avg_a - avg_c)
    return ece


def binary_auroc(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    """Compute exact tie-aware Wilcoxon-Mann-Whitney AUROC in [0, 1]."""
    if not positive_scores or not negative_scores:
        raise ValueError("positive_scores and negative_scores must both be non-empty")
    concordant = 0.0
    for ps in positive_scores:
        for ns in negative_scores:
            if ps > ns:
                concordant += 1.0
            elif ps == ns:
                concordant += 0.5
    return concordant / (len(positive_scores) * len(negative_scores))


def selective_risk_coverage_aurc(
    uncertainties: Sequence[float],
    is_correct: Sequence[bool],
) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC) sorted from lowest to highest hesitation."""
    if len(uncertainties) != len(is_correct) or not uncertainties:
        raise ValueError("uncertainties and is_correct must be non-empty sequences of equal length")
    ordered = sorted(zip(uncertainties, is_correct), key=lambda pair: float(pair[0]))
    errors_so_far = 0
    risks: list[float] = []
    for k_idx, (_, ok) in enumerate(ordered, start=1):
        if not ok:
            errors_so_far += 1
        risks.append(errors_so_far / float(k_idx))
    return sum(risks) / len(risks)


def paired_mcnemar_test(
    arm_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
) -> McNemarResult:
    """Evaluate paired binary accuracy differences strictly via discordant pairs (n01 vs n10)."""
    if len(arm_correct) != len(baseline_correct) or not arm_correct:
        raise ValueError("arm_correct and baseline_correct must be non-empty sequences of equal length")
    n01 = 0  # baseline wrong, arm right
    n10 = 0  # baseline right, arm wrong
    for a_ok, b_ok in zip(arm_correct, baseline_correct):
        if bool(a_ok) and not bool(b_ok):
            n01 += 1
        elif bool(b_ok) and not bool(a_ok):
            n10 += 1

    disc = n01 + n10
    if disc == 0:
        return McNemarResult(
            n_pairs=len(arm_correct),
            n01_arm_wins=0,
            n10_baseline_wins=0,
            chi2_cc=0.0,
            p_two_sided=1.0,
        )
    chi2_cc = ((abs(n01 - n10) - 1.0) ** 2) / float(disc) if abs(n01 - n10) > 1 else 0.0
    k_min = min(n01, n10)
    tail = sum(math.comb(disc, i) for i in range(k_min + 1)) * (0.5**disc)
    p_val = min(1.0, 2.0 * tail)
    return McNemarResult(
        n_pairs=len(arm_correct),
        n01_arm_wins=n01,
        n10_baseline_wins=n10,
        chi2_cc=chi2_cc,
        p_two_sided=p_val,
    )


def rogan_gladen_prevalence(
    observed_positive_rate: float,
    sensitivity_tpr: float,
    specificity_tnr: float,
) -> float:
    """Correct raw classifier positive rate p_hat for imperfect TPR/TNR via Rogan-Gladen (1978).

    pi_hat = clip((p_hat + TNR - 1) / (TPR + TNR - 1), 0, 1).
    Raises ValueError when Youden's J = TPR + TNR - 1 <= 0 (uninformative detector).
    """
    if not (0.0 <= observed_positive_rate <= 1.0):
        raise ValueError(f"observed_positive_rate must be in [0, 1], got {observed_positive_rate}")
    if not (0.0 <= sensitivity_tpr <= 1.0) or not (0.0 <= specificity_tnr <= 1.0):
        raise ValueError("sensitivity_tpr and specificity_tnr must lie in [0, 1]")
    youden_j = sensitivity_tpr + specificity_tnr - 1.0
    if youden_j <= 1e-9:
        raise ValueError(
            f"Rogan-Gladen requires Youden's J = TPR + TNR - 1 > 0, got {youden_j:.6f}"
        )
    est = (observed_positive_rate + specificity_tnr - 1.0) / youden_j
    return min(1.0, max(0.0, est))


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Compute unweighted Cohen's Kappa agreement between two label sequences."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("labels_a and labels_b must be non-empty sequences of equal length")
    n = float(len(labels_a))
    agree = sum(1.0 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}
    for a, b in zip(labels_a, labels_b):
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1
    all_keys = set(counts_a) | set(counts_b)
    p_expected = sum((counts_a.get(k, 0) / n) * (counts_b.get(k, 0) / n) for k in all_keys)
    if math.isclose(p_expected, 1.0):
        return 1.0
    return (agree - p_expected) / (1.0 - p_expected)


def cluster_bootstrap_ci(
    values_by_cluster: Mapping[str, Sequence[float]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute (mean, ci95_low, ci95_high) resampling whole clusters with replacement."""
    non_empty = {k: list(v) for k, v in values_by_cluster.items() if v}
    if not non_empty:
        raise ValueError("values_by_cluster must contain at least one non-empty cluster")
    if n_boot < 20:
        raise ValueError(f"n_boot must be >= 20, got {n_boot}")
    clusters = sorted(non_empty.keys())
    obs = [x for cl in clusters for x in non_empty[cl]]
    obs_mean = sum(obs) / len(obs)

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        picked = [rng.choice(clusters) for _ in range(len(clusters))]
        sample_vals = [x for cl in picked for x in non_empty[cl]]
        boot.append(sum(sample_vals) / len(sample_vals))
    boot.sort()
    lo = boot[int(0.025 * (n_boot - 1))]
    hi = boot[int(0.975 * (n_boot - 1))]
    return obs_mean, lo, hi


@dataclass(frozen=True)
class MurphyDecomposition:
    """Murphy (1973) multiclass Brier score decomposition into Reliability, Resolution, and Uncertainty."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float


def brier_murphy_decomposition(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    num_bins: int = 15,
) -> MurphyDecomposition:
    """Decompose multiclass Brier score into exact non-negative REL, RES, and UNC (Stephenson et al., 2008)."""
    if len(predictions) != len(golds) or not predictions:
        raise ValueError("predictions and golds must be non-empty sequences of equal length")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    norm_preds = [_normalize_dist(d) for d in predictions]
    gold_strs = [str(g) for g in golds]
    all_classes = sorted({k for d in norm_preds for k in d.keys()} | set(gold_strs))
    n = float(len(norm_preds))
    brier = brier_score_multiclass(norm_preds, gold_strs)

    rel_total = 0.0
    res_total = 0.0
    unc_total = 0.0

    for cls_key in all_classes:
        outcomes = [1.0 if g == cls_key else 0.0 for g in gold_strs]
        o_bar = sum(outcomes) / n
        unc_k = o_bar * (1.0 - o_bar)
        unc_total += unc_k

        bins_p: list[list[float]] = [[] for _ in range(num_bins)]
        bins_o: list[list[float]] = [[] for _ in range(num_bins)]

        for dist, out_val in zip(norm_preds, outcomes):
            p_val = dist.get(cls_key, 0.0)
            b_idx = min(num_bins - 1, max(0, int(p_val * num_bins)))
            bins_p[b_idx].append(p_val)
            bins_o[b_idx].append(out_val)

        rel_k = 0.0
        res_k = 0.0
        for b_idx in range(num_bins):
            p_list = bins_p[b_idx]
            cnt = len(p_list)
            if cnt > 0:
                o_list = bins_o[b_idx]
                f_cnt = float(cnt)
                p_b = sum(p_list) / f_cnt
                o_b = sum(o_list) / f_cnt
                weight = f_cnt / n
                wbv_b = sum((pv - p_b) ** 2 for pv in p_list) / n
                wbc_b = 2.0 * sum((pv - p_b) * (ov - o_b) for pv, ov in zip(p_list, o_list)) / n
                rel_k += weight * ((p_b - o_b) ** 2) + wbv_b
                res_k += weight * ((o_b - o_bar) ** 2) + wbc_b

        if res_k < 0.0:
            rel_k -= res_k
            res_k = 0.0
        if rel_k < 0.0:
            res_k -= rel_k
            rel_k = 0.0

        rel_total += rel_k
        res_total += res_k

    residual = brier - (rel_total - res_total + unc_total)
    if rel_total + residual >= 0.0:
        rel_total += residual
    else:
        res_total -= residual

    return MurphyDecomposition(
        brier=brier,
        reliability=rel_total,
        resolution=res_total,
        uncertainty=unc_total,
    )


def adaptive_calibration_error(
    predictions: Sequence[Mapping[str, float]],
    golds: Sequence[str],
    num_bins: int = 15,
) -> float:
    """Compute equal-mass Adaptive Calibration Error (ACE, Nixon et al. 2019) over active candidate options."""
    if len(predictions) != len(golds) or not predictions:
        raise ValueError("predictions and golds must be non-empty sequences of equal length")
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    norm_preds = [_normalize_dist(d) for d in predictions]
    gold_strs = [str(g) for g in golds]
    all_classes = sorted({k for d in norm_preds for k in d.keys()} | set(gold_strs))
    if not all_classes:
        return 0.0

    weighted_err_sum = 0.0
    total_active_pairs = 0

    for cls_key in all_classes:
        active_pairs = sorted(
            (dist.get(cls_key, 0.0), 1.0 if g == cls_key else 0.0)
            for dist, g in zip(norm_preds, gold_strs)
            if (cls_key in dist) or (g == cls_key)
        )
        m = len(active_pairs)
        if m == 0:
            continue
        r_bins = min(num_bins, m)
        cursor = 0
        cls_err = 0.0
        for r_idx in range(r_bins):
            b_size = (m // r_bins) + (1 if r_idx < (m % r_bins) else 0)
            if b_size <= 0:
                continue
            bucket = active_pairs[cursor : cursor + b_size]
            cursor += b_size
            avg_p = sum(p for p, _ in bucket) / float(b_size)
            avg_o = sum(o for _, o in bucket) / float(b_size)
            cls_err += (float(b_size) / float(m)) * abs(avg_o - avg_p)
        weighted_err_sum += float(m) * cls_err
        total_active_pairs += m

    if total_active_pairs == 0:
        return 0.0
    return weighted_err_sum / float(total_active_pairs)


def macro_f1_score(
    predicted_labels: Sequence[str],
    gold_labels: Sequence[str],
) -> float:
    """Compute unweighted multiclass Macro-F1 across all unique labels present in gold or predicted."""
    if len(predicted_labels) != len(gold_labels) or not gold_labels:
        raise ValueError("predicted_labels and gold_labels must be non-empty sequences of equal length")
    preds = [str(p) for p in predicted_labels]
    golds = [str(g) for g in gold_labels]
    classes = sorted(set(golds) | set(preds))

    tp_map: dict[str, int] = {c: 0 for c in classes}
    fp_map: dict[str, int] = {c: 0 for c in classes}
    fn_map: dict[str, int] = {c: 0 for c in classes}

    for pred, gold in zip(preds, golds):
        if pred == gold:
            tp_map[gold] += 1
        else:
            fp_map[pred] += 1
            fn_map[gold] += 1

    f1_sum = 0.0
    for c in classes:
        denom = 2 * tp_map[c] + fp_map[c] + fn_map[c]
        if denom > 0:
            f1_sum += (2.0 * float(tp_map[c])) / float(denom)
    return f1_sum / float(len(classes))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance_by_id: Mapping[str, float],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k) in [0, 1] with duplicate-ID deduplication."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    clean_rel: dict[str, float] = {}
    for raw_id, raw_val in relevance_by_id.items():
        val = float(raw_val)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"relevance for {raw_id!r} must be finite and >= 0, got {raw_val!r}")
        if val > 0.0:
            clean_rel[str(raw_id)] = val

    if not ranked_ids or not clean_rel:
        return 0.0

    ideal_gains = sorted(clean_rel.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(float(idx + 2)) for idx, rel in enumerate(ideal_gains))
    if idcg <= 0.0:
        return 0.0

    dcg = 0.0
    seen: set[str] = set()
    for idx, doc_id_raw in enumerate(ranked_ids[:k]):
        doc_id = str(doc_id_raw)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rel = clean_rel.get(doc_id, 0.0)
        if rel > 0.0:
            dcg += rel / math.log2(float(idx + 2))
    return min(1.0, max(0.0, dcg / idcg))


@dataclass(frozen=True)
class PermutationStabilityResult:
    """Sensitivity of decision distributions across option-order permutations (Zheng et al. ICLR 2024)."""

    n_items: int
    top1_flip_rate: float
    mean_js_divergence: float
    mean_tv_distance: float


def option_permutation_stability(
    canonical_predictions: Sequence[Mapping[str, float]],
    permuted_predictions: Sequence[Mapping[str, float]],
) -> PermutationStabilityResult:
    """Measure top-1 decision flip rate, Jensen-Shannon divergence (bits), and TV distance under option reordering."""
    if (
        not canonical_predictions
        or len(canonical_predictions) != len(permuted_predictions)
    ):
        raise ValueError(
            "canonical_predictions and permuted_predictions must be non-empty sequences of equal length"
        )

    flips = 0
    js_sum = 0.0
    tv_sum = 0.0
    eps = 1e-12

    for p_raw, q_raw in zip(canonical_predictions, permuted_predictions):
        p = _normalize_dist(p_raw)
        q = _normalize_dist(q_raw)
        keys = sorted(set(p.keys()) | set(q.keys()))
        top_p = max(keys, key=lambda k: (p.get(k, 0.0), k))
        top_q = max(keys, key=lambda k: (q.get(k, 0.0), k))
        if top_p != top_q:
            flips += 1

        tv_item = 0.0
        js_item = 0.0
        for k in keys:
            pk = p.get(k, 0.0)
            qk = q.get(k, 0.0)
            mk = 0.5 * (pk + qk)
            tv_item += 0.5 * abs(pk - qk)
            if pk > eps and mk > eps:
                js_item += 0.5 * pk * math.log2(pk / mk)
            if qk > eps and mk > eps:
                js_item += 0.5 * qk * math.log2(qk / mk)

        tv_sum += min(1.0, max(0.0, tv_item))
        js_sum += min(1.0, max(0.0, js_item))

    n = len(canonical_predictions)
    return PermutationStabilityResult(
        n_items=n,
        top1_flip_rate=float(flips) / float(n),
        mean_js_divergence=js_sum / float(n),
        mean_tv_distance=tv_sum / float(n),
    )


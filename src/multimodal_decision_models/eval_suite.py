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
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .conformal import calibrate_conformal_quantile, conformal_prediction_set, hmac_uint64, nonconformity_score


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


@dataclass(frozen=True)
class NonInferiorityTestResult:
    """Tango (1998) one-sided paired score non-inferiority test result (arXiv:2610.21412v1 Appendix E)."""

    wins: int
    losses: int
    n_total: int
    margin: float
    delta_hat: float
    tango_z: float
    p_asymptotic: float
    p_berger_boos: float
    zero_discordance_loss_ub: float
    non_inferior: bool


def tango_paired_non_inferiority_test(
    wins: int,
    losses: int,
    n_total: int,
    *,
    margin: float = 0.03,
    alpha: float = 0.05,
    gamma: float = 1e-4,
) -> NonInferiorityTestResult:
    """Evaluate Tango's (1998) one-sided paired score test of H0: Delta <= -margin with Berger-Boos bound.

    Follows the registered statistical protocol of Zhu et al. (arXiv:2610.21412v1 Appendix E):
    given `wins` (arm succeeds, reference fails), `losses` (arm fails, reference succeeds), and `n_total`
    paired tasks, tests H0: p_arm - p_ref <= -margin against H1: p_arm - p_ref > -margin, bounding
    the nuisance discordance parameter over a (1 - gamma) Clopper-Pearson/Wilson interval.
    """
    if n_total <= 0 or wins < 0 or losses < 0 or (wins + losses) > n_total:
        raise ValueError(f"invalid paired counts: wins={wins}, losses={losses}, n_total={n_total}")
    if not math.isfinite(margin) or margin <= 0.0 or margin >= 1.0:
        raise ValueError(f"margin must lie in (0, 1), got {margin!r}")
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if not math.isfinite(gamma) or gamma <= 0.0 or gamma >= 0.1:
        raise ValueError(f"gamma must lie in (0, 0.1), got {gamma!r}")

    w = int(wins)
    loss_count = int(losses)
    d = w + loss_count
    delta_hat = float(w - loss_count) / float(n_total)
    zero_ub = 1.0 - (alpha ** (1.0 / float(n_total)))

    if d == 0:
        p_null = (1.0 - margin) ** n_total
        p_bb = min(1.0, gamma + p_null)
        return NonInferiorityTestResult(
            wins=w,
            losses=loss_count,
            n_total=n_total,
            margin=margin,
            delta_hat=0.0,
            tango_z=float("inf"),
            p_asymptotic=0.0,
            p_berger_boos=p_bb,
            zero_discordance_loss_ub=zero_ub,
            non_inferior=(p_bb <= alpha),
        )

    best_ll = -1e30
    best_q = max(margin + 1e-6, min(0.999999, float(d) / float(n_total)))
    for i in range(1, 2000):
        q = margin + (1.0 - margin) * (float(i) / 2001.0)
        p10 = 0.5 * (q + margin)
        p01 = 0.5 * (q - margin)
        p_same = 1.0 - q
        if p10 <= 0.0 or p01 <= 0.0 or p_same <= 0.0:
            continue
        ll = (
            float(loss_count) * math.log(p10)
            + float(w) * math.log(p01)
            + float(n_total - d) * math.log(p_same)
        )
        if ll > best_ll:
            best_ll = ll
            best_q = q

    var_h0 = float(n_total) * max(best_q - margin * margin, 1e-12)
    z_stat = (float(w - loss_count) + float(n_total) * margin) / math.sqrt(var_h0)
    p_asym = 0.5 * math.erfc(z_stat / math.sqrt(2.0))

    z_gamma = 3.890592
    phat_d = float(d) / float(n_total)
    rad = z_gamma * math.sqrt(max(phat_d * (1.0 - phat_d) / float(n_total), 1e-9)) + 1.0 / (2.0 * float(n_total))
    q_lo = max(margin, phat_d - rad)
    q_hi = min(1.0, phat_d + rad)

    worst_tail = 0.0
    for idx in range(51):
        q_cand = q_lo + (q_hi - q_lo) * (float(idx) / 50.0)
        var_cand = float(n_total) * max(q_cand - margin * margin, 1e-12)
        z_cand = (float(w - loss_count) + float(n_total) * margin) / math.sqrt(var_cand)
        tail_cand = 0.5 * math.erfc(z_cand / math.sqrt(2.0))
        if tail_cand > worst_tail:
            worst_tail = tail_cand

    p_bb = min(1.0, gamma + worst_tail)
    return NonInferiorityTestResult(
        wins=w,
        losses=loss_count,
        n_total=n_total,
        margin=margin,
        delta_hat=delta_hat,
        tango_z=z_stat,
        p_asymptotic=p_asym,
        p_berger_boos=p_bb,
        zero_discordance_loss_ub=zero_ub,
        non_inferior=(p_bb <= alpha),
    )


def holm_bonferroni_adjust(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Apply Holm-Bonferroni step-down family-wise error rate adjustment (Holm 1979; arXiv:2610.21412v1 Sec. 6).

    Returns `(adjusted_p_values, rejected_flags)` in the original input order.
    """
    m = len(p_values)
    if m == 0:
        raise ValueError("p_values must be non-empty")
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda kv: kv[1])
    adj = [0.0] * m
    rej = [False] * m
    running_max = 0.0
    still_rejecting = True
    for rank, (orig_idx, p_val) in enumerate(indexed):
        if not math.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
            raise ValueError(f"invalid p-value at index {orig_idx}: {p_val!r}")
        mult = float(m - rank)
        running_max = max(running_max, min(1.0, p_val * mult))
        adj[orig_idx] = running_max
        if still_rejecting and running_max <= alpha:
            rej[orig_idx] = True
        else:
            still_rejecting = False
            rej[orig_idx] = False
    return adj, rej


def acceptable_set_accuracy(
    predictions: Sequence[str],
    acceptable_sets: Sequence[Sequence[str]],
) -> float:
    """Compute step-level acceptable-set accuracy P(y_hat in A_acceptable(s)) (arXiv:2610.21412v1 Sec. 3 L2)."""
    n = len(predictions)
    if n == 0 or len(acceptable_sets) != n:
        raise ValueError("predictions and acceptable_sets must be non-empty sequences of equal length")
    hits = 0
    for pred, acc_seq in zip(predictions, acceptable_sets):
        if str(pred) in {str(x) for x in acc_seq}:
            hits += 1
    return float(hits) / float(n)


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate Lentz's continued fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c_val = 1.0
    d_val = 1.0 - qab * x / qap
    if abs(d_val) < 1e-30:
        d_val = 1e-30
    d_val = 1.0 / d_val
    h_val = d_val
    for step_m in range(1, 201):
        m2 = 2 * step_m
        aa_even = (
            float(step_m) * (b - float(step_m)) * x / ((qam + float(m2)) * (a + float(m2)))
        )
        d_val = 1.0 + aa_even * d_val
        if abs(d_val) < 1e-30:
            d_val = 1e-30
        c_val = 1.0 + aa_even / c_val
        if abs(c_val) < 1e-30:
            c_val = 1e-30
        d_val = 1.0 / d_val
        h_val *= d_val * c_val

        aa_odd = (
            -(a + float(step_m))
            * (qab + float(step_m))
            * x
            / ((a + float(m2)) * (qap + float(m2)))
        )
        d_val = 1.0 + aa_odd * d_val
        if abs(d_val) < 1e-30:
            d_val = 1e-30
        c_val = 1.0 + aa_odd / c_val
        if abs(c_val) < 1e-30:
            c_val = 1e-30
        d_val = 1.0 / d_val
        delta = d_val * c_val
        h_val *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h_val


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute regularized incomplete beta function I_x(a, b) in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_front = a * math.log(x) + b * math.log(1.0 - x) - math.log(a) - log_beta
    if log_front < -740.0:
        return 0.0
    return min(1.0, max(0.0, math.exp(log_front) * _betacf(a, b, x)))


def _student_t_two_sided_p(abs_t: float, df: float) -> float:
    """Compute exact two-sided Student-t tail probability P(|T_df| >= |t|)."""
    if not math.isfinite(abs_t) or abs_t < 0.0:
        return 0.0 if math.isinf(abs_t) else 1.0
    if abs_t == 0.0:
        return 1.0
    eff_df = max(1.0, float(df))
    x_ratio = eff_df / (eff_df + abs_t * abs_t)
    return _regularized_incomplete_beta(x_ratio, 0.5 * eff_df, 0.5)


def _normal_two_sided_z(alpha: float) -> float:
    """Compute standard normal two-sided upper critical value z_{1 - alpha/2}."""
    half_alpha = min(0.499999, max(1e-15, 0.5 * float(alpha)))
    t_val = math.sqrt(-2.0 * math.log(half_alpha))
    num = 2.515517 + 0.802853 * t_val + 0.010328 * t_val * t_val
    den = 1.0 + 1.432788 * t_val + 0.189269 * t_val * t_val + 0.001308 * (t_val**3)
    return max(1e-6, t_val - (num / den))


def _student_t_critical(alpha: float, df: float) -> float:
    """Invert exact Student-t two-sided tail probability P(|T_df| >= t_crit) == alpha."""
    target_alpha = min(0.999999, max(1e-15, float(alpha)))
    eff_df = max(1.0, float(df))
    lo = 0.0
    hi = max(2.0, _normal_two_sided_z(target_alpha) * 2.0)
    while _student_t_two_sided_p(hi, eff_df) > target_alpha and hi < 1e9:
        lo = hi
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _student_t_two_sided_p(mid, eff_df) > target_alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PPIConfidenceInterval:
    """Power-Tuned Prediction-Powered Inference (PPI++) point estimate and confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    gold_only_mean: float
    gold_only_ci_low: float
    gold_only_ci_high: float
    gold_only_se: float
    naive_autorater_mean: float
    naive_autorater_ci_low: float
    naive_autorater_ci_high: float
    lambda_star: float
    n_gold: int
    n_unlabeled: int
    effective_sample_size_ratio: float


@dataclass(frozen=True)
class PPIPairedWinRateResult:
    """Debiased PPI++ paired model win-rate difference and hypothesis test."""

    delta_estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    z_statistic: float
    p_two_sided: float
    lambda_star: float
    effective_sample_size_ratio: float


@dataclass(frozen=True)
class RoganGladenVsPPIComparison:
    """Diagnostic selector comparing PPI++ and Rogan-Gladen autorater estimators."""

    ppi_estimate: float
    ppi_standard_error: float
    rogan_gladen_estimate: float
    rogan_gladen_standard_error: float
    prevalence_shift_abs: float
    youden_j: float
    recommended_estimator: str
    recommendation_reason: str


@dataclass(frozen=True)
class ParaphraseStabilityAudit:
    """Prompt-paraphrase conformal stability and worst-case coverage envelope audit."""

    num_paraphrases: int
    n_calib: int
    n_eval: int
    single_template_q_hat: float
    envelope_q_hat: float
    q_hat_spread: float
    single_template_worst_coverage: float
    envelope_worst_coverage: float
    envelope_mean_set_size: float


@dataclass(frozen=True)
class PositionSwappedJudgeAudit:
    """Bilateral (A, B) vs (B, A) autorater positional-bias and debiasing audit."""

    n_pairs: int
    single_order_flip_rate: float
    first_position_bias: float
    post_debias_flip_rate: float
    debiased_win_rate_a: float
    debiased_win_rate_b: float
    debiased_distributions: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class TrajectoryStageRecord:
    """Four-stage agent trajectory verification record (pre-guard and post-guard)."""

    trajectory_id: str
    pre_guard_stages: Mapping[str, bool]
    post_guard_stages: Mapping[str, bool]
    final_output_passed: bool


@dataclass(frozen=True)
class FourStageTrajectoryAudit:
    """Summary of 4-stage Agent-as-a-Judge trajectory verification (arXiv:2410.10934)."""

    n_trajectories: int
    final_output_only_pass_rate: float
    pre_guard_trajectory_pass_rate: float
    post_guard_trajectory_pass_rate: float
    hidden_process_violation_rate: float
    pre_guard_stage_pass_rates: dict[str, float]
    post_guard_stage_pass_rates: dict[str, float]
    first_fault_stage_by_id: dict[str, str]


def _clamp_interval_value(val: float, low: float, high: float) -> float:
    return min(high, max(low, val))


def _compute_ppi_core(
    gold_labels: Sequence[float],
    gold_autorater_scores: Sequence[float],
    unlabeled_autorater_scores: Sequence[float],
    *,
    alpha: float,
    small_sample_shrinkage: bool,
) -> tuple[PPIConfidenceInterval, float]:
    """Compute unclipped PPI++ interval statistics and Welch-Satterthwaite degrees of freedom."""
    n_gold = len(gold_labels)
    n_unlab = len(unlabeled_autorater_scores)
    if n_gold < 4 or len(gold_autorater_scores) != n_gold:
        raise ValueError("gold_labels and gold_autorater_scores must match and have length >= 4")
    if n_unlab < 2:
        raise ValueError("unlabeled_autorater_scores must have length >= 2")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    for seq_name, seq_vals in (
        ("gold_labels", gold_labels),
        ("gold_autorater_scores", gold_autorater_scores),
        ("unlabeled_autorater_scores", unlabeled_autorater_scores),
    ):
        for raw_item in seq_vals:
            if isinstance(raw_item, bool) or not isinstance(raw_item, (int, float)):
                raise ValueError(f"{seq_name} must contain finite non-boolean numbers")

    y_vals = [float(v) for v in gold_labels]
    f_gold = [float(v) for v in gold_autorater_scores]
    f_unlab = [float(v) for v in unlabeled_autorater_scores]
    if not all(math.isfinite(x) for x in y_vals + f_gold + f_unlab):
        raise ValueError("all gold and autorater scores must be finite")

    mean_y = sum(y_vals) / float(n_gold)
    mean_f_unlab = sum(f_unlab) / float(n_unlab)
    var_y = sum((yv - mean_y) ** 2 for yv in y_vals) / float(n_gold - 1)
    var_f_unlab = sum((fv - mean_f_unlab) ** 2 for fv in f_unlab) / float(n_unlab - 1)

    def _fit_lambda(y_sub: Sequence[float], f_sub: Sequence[float]) -> float:
        m_sub = len(y_sub)
        if m_sub < 3:
            return 0.0
        my = sum(y_sub) / float(m_sub)
        mf = sum(f_sub) / float(m_sub)
        vf = sum((fv - mf) ** 2 for fv in f_sub) / float(m_sub - 1)
        cyf = sum((yv - my) * (fv - mf) for yv, fv in zip(y_sub, f_sub)) / float(m_sub - 1)
        pf = (float(m_sub - 1) * vf + float(n_unlab - 1) * var_f_unlab) / float(
            m_sub + n_unlab - 2
        )
        den = (1.0 + float(n_gold) / float(n_unlab)) * pf
        if den <= 1e-12:
            return 0.0
        raw = cyf / den
        sh = max(0.0, float(m_sub - 2) / float(m_sub)) if small_sample_shrinkage else 1.0
        return min(1.0, max(0.0, raw * sh))

    if n_gold >= 8:
        # 2-fold cross-fitted PPI++ (Zrnic & Candes, PNAS 2024): split via Thue-Morse parity
        # and pool centered within-fold residual variances so lam_A != lam_B mean shifts never leak.
        y_a = [yv for idx_i, yv in enumerate(y_vals) if idx_i.bit_count() % 2 == 0]
        f_a = [fv for idx_i, fv in enumerate(f_gold) if idx_i.bit_count() % 2 == 0]
        y_b = [yv for idx_i, yv in enumerate(y_vals) if idx_i.bit_count() % 2 == 1]
        f_b = [fv for idx_i, fv in enumerate(f_gold) if idx_i.bit_count() % 2 == 1]
        lam_for_a = _fit_lambda(y_b, f_b)
        lam_for_b = _fit_lambda(y_a, f_a)
        r_a = [yv - lam_for_a * fv for yv, fv in zip(y_a, f_a)]
        r_b = [yv - lam_for_b * fv for yv, fv in zip(y_b, f_b)]
        mean_r_a = sum(r_a) / float(len(r_a))
        mean_r_b = sum(r_b) / float(len(r_b))
        var_r_a = sum((rv - mean_r_a) ** 2 for rv in r_a) / float(max(1, len(r_a) - 1))
        var_r_b = sum((rv - mean_r_b) ** 2 for rv in r_b) / float(max(1, len(r_b) - 1))
        w_a = float(len(r_a)) / float(n_gold)
        w_b = float(len(r_b)) / float(n_gold)
        lam_star = w_a * lam_for_a + w_b * lam_for_b
        ppi_est = w_a * (lam_for_a * mean_f_unlab + mean_r_a) + w_b * (
            lam_for_b * mean_f_unlab + mean_r_b
        )
        v_gold_term = (float(len(r_a)) * var_r_a + float(len(r_b)) * var_r_b) / float(
            n_gold * n_gold
        )
        resid_df = max(2.0, float(n_gold - 2))
    else:
        lam_star = _fit_lambda(y_vals, f_gold)
        rectified = [yv - lam_star * fv for yv, fv in zip(y_vals, f_gold)]
        mean_rect = sum(rectified) / float(n_gold)
        resid_df = max(1.0, float(n_gold - 2))
        var_rect = sum((rv - mean_rect) ** 2 for rv in rectified) / resid_df
        ppi_est = lam_star * mean_f_unlab + mean_rect
        v_gold_term = var_rect / float(n_gold)

    v_unlab_term = (lam_star * lam_star * var_f_unlab) / float(n_unlab)
    ppi_var = max(1e-15, v_gold_term + v_unlab_term)
    ppi_se = math.sqrt(ppi_var)

    # Welch-Satterthwaite effective degrees of freedom across anchor and unlabeled components.
    ws_num = ppi_var * ppi_var
    ws_den = (v_gold_term * v_gold_term) / resid_df + (
        v_unlab_term * v_unlab_term
    ) / max(1.0, float(n_unlab - 1))
    eff_df = ws_num / ws_den if ws_den > 1e-24 else resid_df
    t_crit = _student_t_critical(alpha, eff_df)
    t_gold = _student_t_critical(alpha, float(n_gold - 1))

    gold_se = math.sqrt(max(1e-15, var_y / float(n_gold)))
    all_autorater = f_gold + f_unlab
    n_all = len(all_autorater)
    naive_mean = sum(all_autorater) / float(n_all)
    naive_var = sum((fv - naive_mean) ** 2 for fv in all_autorater) / float(n_all - 1)
    naive_se = math.sqrt(max(1e-15, naive_var / float(n_all)))
    z_norm = _normal_two_sided_z(alpha)

    ci_lo = ppi_est - t_crit * ppi_se
    ci_hi = ppi_est + t_crit * ppi_se
    gold_lo = mean_y - t_gold * gold_se
    gold_hi = mean_y + t_gold * gold_se
    naive_lo = naive_mean - z_norm * naive_se
    naive_hi = naive_mean + z_norm * naive_se

    n_eff_ratio = max(1.0, (var_y / float(n_gold)) / ppi_var) if var_y > 1e-12 else 1.0
    interval = PPIConfidenceInterval(
        estimate=ppi_est,
        ci_low=ci_lo,
        ci_high=ci_hi,
        standard_error=ppi_se,
        gold_only_mean=mean_y,
        gold_only_ci_low=gold_lo,
        gold_only_ci_high=gold_hi,
        gold_only_se=gold_se,
        naive_autorater_mean=naive_mean,
        naive_autorater_ci_low=naive_lo,
        naive_autorater_ci_high=naive_hi,
        lambda_star=lam_star,
        n_gold=n_gold,
        n_unlabeled=n_unlab,
        effective_sample_size_ratio=n_eff_ratio,
    )
    return interval, eff_df


def power_tuned_ppi_mean_ci(
    gold_labels: Sequence[float],
    gold_autorater_scores: Sequence[float],
    unlabeled_autorater_scores: Sequence[float],
    *,
    alpha: float = 0.05,
    small_sample_shrinkage: bool = True,
    clip_bounds: tuple[float, float] | None = None,
) -> PPIConfidenceInterval:
    """Compute Power-Tuned Prediction-Powered Inference (PPI++) mean and confidence interval.

    Implements PPI++ (Angelopoulos et al., JMLR; Arawjo, arXiv:2609.35815; Chen et al.,
    arXiv:2601.05420) combining `n` anchor gold labels `(Y_i, f_hat_i)` with `N` unlabeled
    autorater scores `f_tilde_j` via shrinkage-stabilized optimal power tuning `lambda* in [0, 1]`.

    Args:
        gold_labels: Ground-truth scalar outcomes `Y_1..Y_n` on the labeled anchor slice.
        gold_autorater_scores: Autorater predictions `f_hat_1..f_hat_n` on the same anchor slice.
        unlabeled_autorater_scores: Autorater predictions `f_tilde_1..f_tilde_N` on unlabeled items.
        alpha: Two-sided miscoverage level in `(0, 1)` (default `0.05` for 95% CI).
        small_sample_shrinkage: Shrink `lambda*` by `(n - 2)/n` to prevent small-n overfitting.
        clip_bounds: Optional `(low, high)` domain bounds applied symmetrically to estimates and CIs.

    Returns:
        `PPIConfidenceInterval` comparing `PPI++`, gold-only, and naive autorater intervals.

    Raises:
        ValueError: If inputs have `< 4` anchor items, mismatched lengths, or invalid bounds.
    """
    if clip_bounds is not None:
        if (
            len(clip_bounds) != 2
            or isinstance(clip_bounds[0], bool)
            or isinstance(clip_bounds[1], bool)
            or not isinstance(clip_bounds[0], (int, float))
            or not isinstance(clip_bounds[1], (int, float))
        ):
            raise ValueError("clip_bounds must be a 2-tuple of finite numbers (low, high)")
        b_lo, b_hi = float(clip_bounds[0]), float(clip_bounds[1])
        if not (math.isfinite(b_lo) and math.isfinite(b_hi)) or b_lo >= b_hi:
            raise ValueError(f"clip_bounds must satisfy finite low < high, got {clip_bounds!r}")
    else:
        b_lo, b_hi = -math.inf, math.inf

    unclipped, _ = _compute_ppi_core(
        gold_labels,
        gold_autorater_scores,
        unlabeled_autorater_scores,
        alpha=alpha,
        small_sample_shrinkage=small_sample_shrinkage,
    )
    if clip_bounds is None:
        return unclipped

    return PPIConfidenceInterval(
        estimate=_clamp_interval_value(unclipped.estimate, b_lo, b_hi),
        ci_low=_clamp_interval_value(unclipped.ci_low, b_lo, b_hi),
        ci_high=_clamp_interval_value(unclipped.ci_high, b_lo, b_hi),
        standard_error=unclipped.standard_error,
        gold_only_mean=_clamp_interval_value(unclipped.gold_only_mean, b_lo, b_hi),
        gold_only_ci_low=_clamp_interval_value(unclipped.gold_only_ci_low, b_lo, b_hi),
        gold_only_ci_high=_clamp_interval_value(unclipped.gold_only_ci_high, b_lo, b_hi),
        gold_only_se=unclipped.gold_only_se,
        naive_autorater_mean=_clamp_interval_value(unclipped.naive_autorater_mean, b_lo, b_hi),
        naive_autorater_ci_low=_clamp_interval_value(unclipped.naive_autorater_ci_low, b_lo, b_hi),
        naive_autorater_ci_high=_clamp_interval_value(unclipped.naive_autorater_ci_high, b_lo, b_hi),
        lambda_star=unclipped.lambda_star,
        n_gold=unclipped.n_gold,
        n_unlabeled=unclipped.n_unlabeled,
        effective_sample_size_ratio=unclipped.effective_sample_size_ratio,
    )


def ppi_paired_win_rate_test(
    gold_diff_outcomes: Sequence[float],
    gold_autorater_diffs: Sequence[float],
    unlabeled_autorater_diffs: Sequence[float],
    *,
    alpha: float = 0.05,
) -> PPIPairedWinRateResult:
    """Test paired win-rate difference `theta_A - theta_B` using PPI++ debiased inference.

    Args:
        gold_diff_outcomes: Anchor paired differences `Y_{A,i} - Y_{B,i}` in `[-1, 1]`.
        gold_autorater_diffs: Autorater score differences `f_{A,i} - f_{B,i}` on anchor pairs.
        unlabeled_autorater_diffs: Autorater score differences on unlabeled pairs.
        alpha: Two-sided significance level in `(0, 1)` used only for confidence interval bounds.

    Returns:
        `PPIPairedWinRateResult` with CI, standard error, t-statistic, and exact Student-t p-value.
    """
    unclipped, eff_df = _compute_ppi_core(
        gold_diff_outcomes,
        gold_autorater_diffs,
        unlabeled_autorater_diffs,
        alpha=alpha,
        small_sample_shrinkage=True,
    )
    se = max(1e-12, unclipped.standard_error)
    t_stat = 0.0 if abs(unclipped.estimate) <= 1e-15 else (unclipped.estimate / se)
    p_val = _student_t_two_sided_p(abs(t_stat), eff_df)
    return PPIPairedWinRateResult(
        delta_estimate=_clamp_interval_value(unclipped.estimate, -1.0, 1.0),
        ci_low=_clamp_interval_value(unclipped.ci_low, -1.0, 1.0),
        ci_high=_clamp_interval_value(unclipped.ci_high, -1.0, 1.0),
        standard_error=unclipped.standard_error,
        z_statistic=t_stat,
        p_two_sided=p_val,
        lambda_star=unclipped.lambda_star,
        effective_sample_size_ratio=unclipped.effective_sample_size_ratio,
    )


def compare_rogan_gladen_vs_ppi(
    gold_binary_labels: Sequence[int | bool],
    gold_autorater_probs: Sequence[float],
    unlabeled_autorater_probs: Sequence[float],
    *,
    decision_threshold: float = 0.50,
    shift_tolerance: float = 0.12,
) -> RoganGladenVsPPIComparison:
    """Compare PPI++ continuous rectification against Rogan-Gladen binary prevalence adjustment.

    Grounded in Chen et al. (arXiv:2601.05420, ICML 2026): PPI++ achieves strictly smaller
    standard error when the anchor and unlabeled slices are exchangeable and continuous judge
    posteriors are available, whereas Rogan-Gladen remains consistent under severe target label
    prevalence shift provided class-conditional `TPR` and `TNR` stay invariant.

    Args:
        gold_binary_labels: Anchor binary labels (`0`/`1` or `bool`).
        gold_autorater_probs: Autorater positive-class probabilities on anchor items.
        unlabeled_autorater_probs: Autorater positive-class probabilities on target unlabeled items.
        decision_threshold: Threshold converting probabilities into binary calls for Rogan-Gladen.
        shift_tolerance: Absolute autorater-rate drift threshold triggering a shift warning.

    Returns:
        `RoganGladenVsPPIComparison` with both estimates, SEs, and regime recommendation.

    Raises:
        ValueError: If inputs are non-binary, non-finite, lack both classes, or Youden's J <= 0.
    """
    if (
        isinstance(decision_threshold, bool)
        or not isinstance(decision_threshold, (int, float))
        or not math.isfinite(float(decision_threshold))
        or float(decision_threshold) <= 0.0
        or float(decision_threshold) >= 1.0
    ):
        raise ValueError(f"decision_threshold must lie strictly in (0, 1), got {decision_threshold!r}")
    if (
        isinstance(shift_tolerance, bool)
        or not isinstance(shift_tolerance, (int, float))
        or not math.isfinite(float(shift_tolerance))
        or float(shift_tolerance) < 0.0
    ):
        raise ValueError(f"shift_tolerance must be finite and >= 0, got {shift_tolerance!r}")

    y_bin: list[float] = []
    for raw_label in gold_binary_labels:
        if isinstance(raw_label, bool):
            y_bin.append(1.0 if raw_label else 0.0)
        elif isinstance(raw_label, (int, float)) and math.isfinite(float(raw_label)) and float(raw_label) in (0.0, 1.0):
            y_bin.append(float(raw_label))
        else:
            raise ValueError(
                f"gold_binary_labels must contain binary values in {{0, 1}} or bool, got {raw_label!r}"
            )

    for seq_name, seq_vals in (
        ("gold_autorater_probs", gold_autorater_probs),
        ("unlabeled_autorater_probs", unlabeled_autorater_probs),
    ):
        for raw_prob in seq_vals:
            if isinstance(raw_prob, bool) or not isinstance(raw_prob, (int, float)):
                raise ValueError(f"{seq_name} must contain finite non-boolean probabilities")
            if not math.isfinite(float(raw_prob)) or float(raw_prob) < 0.0 or float(raw_prob) > 1.0:
                raise ValueError(f"{seq_name} probabilities must lie in [0, 1], got {raw_prob!r}")

    f_gold = [float(v) for v in gold_autorater_probs]
    f_unlab = [float(v) for v in unlabeled_autorater_probs]
    ppi_ci = power_tuned_ppi_mean_ci(y_bin, f_gold, f_unlab, clip_bounds=(0.0, 1.0))

    positives = [fv >= decision_threshold for yv, fv in zip(y_bin, f_gold) if yv > 0.5]
    negatives = [fv < decision_threshold for yv, fv in zip(y_bin, f_gold) if yv < 0.5]
    if not positives or not negatives:
        raise ValueError("gold_binary_labels must contain at least one positive and one negative")

    tpr = sum(1.0 for ok in positives if ok) / float(len(positives))
    tnr = sum(1.0 for ok in negatives if ok) / float(len(negatives))
    youden_j = tpr + tnr - 1.0
    if youden_j <= 1e-6:
        raise ValueError(f"uninformative autorater for Rogan-Gladen: Youden's J = {youden_j:.4f}")

    p_gold_bin = sum(1.0 for fv in f_gold if fv >= decision_threshold) / float(len(f_gold))
    p_unlab_bin = sum(1.0 for fv in f_unlab if fv >= decision_threshold) / float(len(f_unlab))
    rg_est = rogan_gladen_prevalence(p_unlab_bin, tpr, tnr)

    # Delta-method standard error of Rogan-Gladen accounting for finite TPR, TNR, and p_hat_unlab.
    var_p = p_unlab_bin * (1.0 - p_unlab_bin) / float(len(f_unlab))
    var_tpr = tpr * (1.0 - tpr) / float(len(positives))
    var_tnr = tnr * (1.0 - tnr) / float(len(negatives))
    rg_var = (
        var_p + (rg_est * rg_est) * var_tpr + ((1.0 - rg_est) ** 2) * var_tnr
    ) / max(1e-9, youden_j * youden_j)
    rg_se = math.sqrt(max(1e-15, rg_var))

    shift_abs = abs(p_unlab_bin - p_gold_bin)
    if shift_abs >= shift_tolerance:
        rec = "rogan_gladen"
        reason = (
            f"detected large prevalence shift |p_unlab - p_gold| = {shift_abs:.3f} >= "
            f"{shift_tolerance:.3f}; Rogan-Gladen preserves consistency under label shift"
        )
    else:
        rec = "ppi_plusplus"
        reason = (
            f"exchangeable regime (|p_unlab - p_gold| = {shift_abs:.3f} < {shift_tolerance:.3f}); "
            f"PPI++ reduces SE ({ppi_ci.standard_error:.4f} vs {rg_se:.4f})"
        )

    return RoganGladenVsPPIComparison(
        ppi_estimate=ppi_ci.estimate,
        ppi_standard_error=ppi_ci.standard_error,
        rogan_gladen_estimate=rg_est,
        rogan_gladen_standard_error=rg_se,
        prevalence_shift_abs=shift_abs,
        youden_j=youden_j,
        recommended_estimator=rec,
        recommendation_reason=reason,
    )


def paraphrase_conformal_stability_audit(
    calib_probs_by_paraphrase: Sequence[Sequence[Mapping[str, float]]],
    calib_golds: Sequence[str],
    eval_probs_by_paraphrase: Sequence[Sequence[Mapping[str, float]]],
    eval_golds: Sequence[str],
    *,
    alpha: float = 0.10,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> ParaphraseStabilityAudit:
    """Audit conformal threshold sensitivity across prompt paraphrases and fit envelope `q_hat`.

    Implements Paraphrase-Aware Conformal Prediction (arXiv:2610.04239, Oct 2026): calibrates
    `q_hat_env` on the per-example worst-paraphrase nonconformity score `max_m s(p_{i,m}, y_i)` so
    prediction sets maintain `>= 1 - alpha` simultaneous worst-paraphrase coverage.

    Args:
        calib_probs_by_paraphrase: Per-item list of `M >= 2` paraphrase distributions (calib).
        calib_golds: Ground-truth labels for calibration items.
        eval_probs_by_paraphrase: Per-item list of `M >= 2` paraphrase distributions (eval).
        eval_golds: Ground-truth labels for evaluation items.
        alpha: Target miscoverage rate in `(0, 1)`.
        method: Conformal score method (`"lac"`, `"aps"`, or `"raps"`).
        lam_reg: RAPS penalty weight.
        k_reg: RAPS onset rank.

    Returns:
        `ParaphraseStabilityAudit` comparing single-phrasing vs. paraphrase-envelope coverage.

    Raises:
        ValueError: If item counts or paraphrase counts `M >= 2` are inconsistent.
    """
    n_calib = len(calib_probs_by_paraphrase)
    n_eval = len(eval_probs_by_paraphrase)
    if n_calib == 0 or len(calib_golds) != n_calib or n_eval == 0 or len(eval_golds) != n_eval:
        raise ValueError("calibration and evaluation sequences must be non-empty and match gold lengths")

    m_count = len(calib_probs_by_paraphrase[0])
    if m_count < 2:
        raise ValueError("at least M >= 2 paraphrases per item are required")
    for seq in list(calib_probs_by_paraphrase) + list(eval_probs_by_paraphrase):
        if len(seq) != m_count:
            raise ValueError("every item must contain the exact same number of paraphrases M")

    per_template_q_hats: list[float] = []
    for m_idx in range(m_count):
        slice_probs = [item[m_idx] for item in calib_probs_by_paraphrase]
        cal_m = calibrate_conformal_quantile(
            slice_probs,
            calib_golds,
            alpha=alpha,
            method=method,
            lam_reg=lam_reg,
            k_reg=k_reg,
        )
        per_template_q_hats.append(cal_m.q_hat)

    single_q_hat = per_template_q_hats[0]
    finite_qs = [q for q in per_template_q_hats if math.isfinite(q)]
    q_spread = (max(finite_qs) - min(finite_qs)) if len(finite_qs) >= 2 else 0.0

    # Paraphrase-envelope calibration over s_i^{max} = max_{m=1..M} s(p_{i,m}, y_i).
    max_calib_scores: list[float] = []
    for item_paraphrases, gold in zip(calib_probs_by_paraphrase, calib_golds):
        worst_score = max(
            nonconformity_score(
                dist,
                str(gold),
                method=method,
                lam_reg=lam_reg,
                k_reg=k_reg,
                randomized=False,
            )
            for dist in item_paraphrases
        )
        max_calib_scores.append(worst_score)

    sorted_max = sorted(max_calib_scores)
    k_rank = math.ceil((n_calib + 1) * (1.0 - alpha))
    envelope_q_hat = sorted_max[k_rank - 1] if k_rank <= n_calib else float("inf")

    single_all_covered = 0
    env_all_covered = 0
    total_env_size = 0
    for item_paraphrases, gold in zip(eval_probs_by_paraphrase, eval_golds):
        gold_str = str(gold)
        ok_single = True
        ok_env = True
        for dist in item_paraphrases:
            c_single = conformal_prediction_set(
                dist, single_q_hat, method=method, lam_reg=lam_reg, k_reg=k_reg
            )
            c_env = conformal_prediction_set(
                dist, envelope_q_hat, method=method, lam_reg=lam_reg, k_reg=k_reg
            )
            total_env_size += len(c_env.prediction_set)
            if gold_str not in c_single.prediction_set:
                ok_single = False
            if gold_str not in c_env.prediction_set:
                ok_env = False
        if ok_single:
            single_all_covered += 1
        if ok_env:
            env_all_covered += 1

    return ParaphraseStabilityAudit(
        num_paraphrases=m_count,
        n_calib=n_calib,
        n_eval=n_eval,
        single_template_q_hat=single_q_hat,
        envelope_q_hat=envelope_q_hat,
        q_hat_spread=q_spread,
        single_template_worst_coverage=float(single_all_covered) / float(n_eval),
        envelope_worst_coverage=float(env_all_covered) / float(n_eval),
        envelope_mean_set_size=float(total_env_size) / float(n_eval * m_count),
    )


def position_swapped_autorater_audit(
    forward_ab_probs: Sequence[Mapping[str, float]],
    reverse_ba_mapped_probs: Sequence[Mapping[str, float]],
) -> PositionSwappedJudgeAudit:
    """Audit and debias bilateral `(A, B)` vs. `(B, A)` pairwise autorater posteriors.

    Follows Zheng et al. (arXiv:2306.05685): `forward_ab_probs` scores `(A first, B second)` and
    `reverse_ba_mapped_probs` scores `(B first, A second)` with labels already mapped back onto
    canonical keys `("a", "b")` (plus optional `"tie"`). Averaging both directions guarantees
    exact permutation symmetry (`post_debias_flip_rate == 0.0`).

    Args:
        forward_ab_probs: Forward `(A, B)` distributions over `{"a", "b"}` (and optional `"tie"`).
        reverse_ba_mapped_probs: Swapped `(B, A)` distributions mapped back to `{"a", "b", ...}`.

    Returns:
        `PositionSwappedJudgeAudit` with single-order flip rate, first-slot bias, and debiased probs.

    Raises:
        ValueError: If sequences are empty, differ in length, or lack `"a"` / `"b"` keys.
    """
    n_pairs = len(forward_ab_probs)
    if n_pairs == 0 or len(reverse_ba_mapped_probs) != n_pairs:
        raise ValueError("forward_ab_probs and reverse_ba_mapped_probs must match and be non-empty")

    flips = 0
    first_bias_sum = 0.0
    post_flips = 0
    wins_a = 0
    wins_b = 0
    debiased_list: list[dict[str, float]] = []

    for fwd_raw, rev_raw in zip(forward_ab_probs, reverse_ba_mapped_probs):
        fwd = _normalize_dist(fwd_raw)
        rev = _normalize_dist(rev_raw)
        if "a" not in fwd or "b" not in fwd or "a" not in rev or "b" not in rev:
            raise ValueError("pairwise distributions must include canonical keys 'a' and 'b'")

        keys = sorted(set(fwd.keys()) | set(rev.keys()))
        top_fwd = max(keys, key=lambda key: (fwd.get(key, 0.0), key))
        top_rev = max(keys, key=lambda key: (rev.get(key, 0.0), key))
        if top_fwd != top_rev:
            flips += 1

        # First-position bias: excess mass assigned to whichever candidate occupied slot 1.
        first_bias_sum += 0.5 * ((fwd["a"] - rev["a"]) + (rev["b"] - fwd["b"]))

        deb_ab = {key: 0.5 * (fwd.get(key, 0.0) + rev.get(key, 0.0)) for key in keys}
        # Verify exact symmetry if caller had passed (B, A) as the primary pair:
        deb_ba_mapped = {key: 0.5 * (rev.get(key, 0.0) + fwd.get(key, 0.0)) for key in keys}

        def _modal_pairwise_winner(dist: Mapping[str, float]) -> str:
            pa = dist.get("a", 0.0)
            pb = dist.get("b", 0.0)
            ptie = dist.get("tie", 0.0)
            if ptie >= max(pa, pb) - 1e-12 or abs(pa - pb) <= 1e-12:
                return "tie"
            return "a" if pa > pb else "b"

        win_ab = _modal_pairwise_winner(deb_ab)
        win_ba = _modal_pairwise_winner(deb_ba_mapped)
        if win_ab != win_ba:
            post_flips += 1
        if win_ab == "a":
            wins_a += 1
        elif win_ab == "b":
            wins_b += 1
        debiased_list.append(deb_ab)

    return PositionSwappedJudgeAudit(
        n_pairs=n_pairs,
        single_order_flip_rate=float(flips) / float(n_pairs),
        first_position_bias=first_bias_sum / float(n_pairs),
        post_debias_flip_rate=float(post_flips) / float(n_pairs),
        debiased_win_rate_a=float(wins_a) / float(n_pairs),
        debiased_win_rate_b=float(wins_b) / float(n_pairs),
        debiased_distributions=tuple(debiased_list),
    )


def evaluate_4stage_trajectory_judge(
    trajectories: Sequence[TrajectoryStageRecord],
    *,
    stage_order: Sequence[str] = (
        "1_routing",
        "2_retrieval_ipi",
        "3_conformal_stakes_guard",
        "4_domain_closure",
    ),
) -> FourStageTrajectoryAudit:
    """Evaluate 4-stage Agent-as-a-Judge trajectory verification with Pre/Post-Guard reporting.

    Follows Zhuge et al. (arXiv:2410.10934): final-output-only evaluation misses intermediate
    routing, indirect prompt injection, or uncalibrated tool-guard violations that coincidentally
    land on an acceptable final string. Reports both Pre-Guard and Post-Guard stage pass rates
    and localizes the earliest failing stage per trajectory.

    Args:
        trajectories: Non-empty sequence of `TrajectoryStageRecord` instances.
        stage_order: Ordered unique stage names verified for every trajectory.

    Returns:
        `FourStageTrajectoryAudit` with dual Pre/Post-Guard metrics and fault localization.

    Raises:
        ValueError: If `trajectories` or `stage_order` is empty, has duplicates, or has non-bools.
    """
    n_traj = len(trajectories)
    ordered_stages = [str(st) for st in stage_order]
    if n_traj == 0 or not ordered_stages:
        raise ValueError("trajectories and stage_order must be non-empty")
    if any(not st.strip() for st in ordered_stages):
        raise ValueError("stage_order entries must be non-empty strings")
    if len(set(ordered_stages)) != len(ordered_stages):
        raise ValueError("stage_order must not contain duplicate stage names")

    final_only_ok = 0
    pre_all_ok = 0
    post_all_ok = 0
    hidden_violations = 0
    pre_stage_counts = {st: 0 for st in ordered_stages}
    post_stage_counts = {st: 0 for st in ordered_stages}
    first_fault_map: dict[str, str] = {}
    seen_ids: set[str] = set()

    for rec in trajectories:
        if not isinstance(rec.trajectory_id, str) or not rec.trajectory_id.strip():
            raise ValueError("trajectory_id must be a non-empty string")
        if rec.trajectory_id in seen_ids:
            raise ValueError(f"duplicate trajectory_id {rec.trajectory_id!r}")
        seen_ids.add(rec.trajectory_id)
        if not isinstance(rec.final_output_passed, bool):
            raise ValueError(
                f"trajectory {rec.trajectory_id!r} final_output_passed must be a bool"
            )
        for st in ordered_stages:
            if st not in rec.pre_guard_stages or st not in rec.post_guard_stages:
                raise ValueError(
                    f"trajectory {rec.trajectory_id!r} missing required stage {st!r}"
                )
            if not isinstance(rec.pre_guard_stages[st], bool) or not isinstance(
                rec.post_guard_stages[st], bool
            ):
                raise ValueError(
                    f"trajectory {rec.trajectory_id!r} stage {st!r} pass flags must be bool"
                )

        pre_ok = True
        post_ok = True
        first_fault = "none"
        for st in ordered_stages:
            if rec.pre_guard_stages[st]:
                pre_stage_counts[st] += 1
            else:
                if pre_ok:
                    first_fault = st
                pre_ok = False
            if rec.post_guard_stages[st]:
                post_stage_counts[st] += 1
            else:
                post_ok = False

        if rec.final_output_passed:
            final_only_ok += 1
            if not pre_ok:
                hidden_violations += 1
        if pre_ok:
            pre_all_ok += 1
        if post_ok:
            post_all_ok += 1
        first_fault_map[rec.trajectory_id] = first_fault

    return FourStageTrajectoryAudit(
        n_trajectories=n_traj,
        final_output_only_pass_rate=float(final_only_ok) / float(n_traj),
        pre_guard_trajectory_pass_rate=float(pre_all_ok) / float(n_traj),
        post_guard_trajectory_pass_rate=float(post_all_ok) / float(n_traj),
        hidden_process_violation_rate=float(hidden_violations) / float(n_traj),
        pre_guard_stage_pass_rates={
            st: float(pre_stage_counts[st]) / float(n_traj) for st in ordered_stages
        },
        post_guard_stage_pass_rates={
            st: float(post_stage_counts[st]) / float(n_traj) for st in ordered_stages
        },
        first_fault_stage_by_id=first_fault_map,
    )


@dataclass(frozen=True)
class FactorialEffectSummary:
    """Yates/ANOVA main effects, two-way synergies, and exact Shapley values on a 2^k cube."""

    factor_names: tuple[str, ...]
    n_episodes: int
    n_clusters: int
    n_bootstrap: int
    coalition_means: dict[str, float]
    total_gain: float
    main_effects: dict[str, float]
    main_effect_cis: dict[str, tuple[float, float]]
    two_way_interactions: dict[str, float]
    two_way_interaction_cis: dict[str, tuple[float, float]]
    shapley_values: dict[str, float]
    shapley_cis: dict[str, tuple[float, float]]
    raw_p_values: dict[str, float]
    holm_adjusted_p_values: dict[str, float]
    significant_effects: dict[str, bool]
    shapley_holm_p_values: dict[str, float]


def _percentile_from_sorted(sorted_values: Sequence[float], quantile_prob: float) -> float:
    """Compute linear-interpolated empirical quantile from an ascending numeric sequence."""
    count = len(sorted_values)
    if count == 0:
        raise ValueError("sorted_values must be non-empty")
    if count == 1:
        return float(sorted_values[0])
    clamped = min(1.0, max(0.0, float(quantile_prob)))
    position = clamped * float(count - 1)
    lower_idx = int(math.floor(position))
    upper_idx = min(count - 1, lower_idx + 1)
    weight = position - float(lower_idx)
    return float(sorted_values[lower_idx]) * (1.0 - weight) + float(sorted_values[upper_idx]) * weight


def _two_sided_bootstrap_wald_p(estimate: float, bootstrap_draws: Sequence[float]) -> float:
    """Compute two-sided asymptotic cluster-bootstrap Wald p-value."""
    if not bootstrap_draws:
        return 1.0
    std_err = statistics.pstdev(bootstrap_draws)
    if std_err <= 1e-15:
        return 0.0 if abs(estimate) > 1e-15 else 1.0
    z_abs = abs(estimate) / std_err
    return min(1.0, max(0.0, math.erfc(z_abs / math.sqrt(2.0))))


def compute_exact_shapley_values(
    coalition_utilities: Mapping[str, float],
    factor_names: Sequence[str] = ("C", "M", "K", "P"),
) -> dict[str, float]:
    """Compute exact Shapley attributions over a complete 2^k factorial utility map.

    Args:
        coalition_utilities: Mapping from k-bit binary string (for example, '1010') to scalar utility.
        factor_names: Ordered sequence of k unique factor identifiers.

    Returns:
        Dictionary mapping each factor name to its exact Shapley value summing to U(1..1) - U(0..0).

    Raises:
        ValueError: If factor_names is empty, duplicated, or any of the 2^k bit codes is missing.
    """
    if isinstance(factor_names, (str, bytes)) or not isinstance(factor_names, Sequence):
        raise ValueError("factor_names must be a non-empty sequence of unique strings")
    clean_factors = [str(name).strip() for name in factor_names]
    num_factors = len(clean_factors)
    if num_factors == 0 or any(not name for name in clean_factors) or len(set(clean_factors)) != num_factors:
        raise ValueError("factor_names must be non-empty and unique")
    if not isinstance(coalition_utilities, Mapping):
        raise ValueError("coalition_utilities must be a mapping of k-bit codes to finite numbers")

    expected_codes = {format(mask, f"0{num_factors}b") for mask in range(1 << num_factors)}
    missing_codes = expected_codes - set(coalition_utilities.keys())
    if missing_codes:
        raise ValueError(f"missing factorial coalition codes: {sorted(missing_codes)}")

    validated_utilities: dict[str, float] = {}
    for code in expected_codes:
        raw_val = coalition_utilities[code]
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)) or not math.isfinite(float(raw_val)):
            raise ValueError(f"coalition utility for {code!r} must be a finite number, got {raw_val!r}")
        validated_utilities[code] = float(raw_val)

    factorial_n = math.factorial(num_factors)
    shapley: dict[str, float] = {}
    for idx, name in enumerate(clean_factors):
        bit = 1 << (num_factors - 1 - idx)
        accum = 0.0
        for mask in range(1 << num_factors):
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            weight = (
                math.factorial(subset_size)
                * math.factorial(num_factors - subset_size - 1)
                / float(factorial_n)
            )
            code_without = format(mask, f"0{num_factors}b")
            code_with = format(mask | bit, f"0{num_factors}b")
            accum += weight * (validated_utilities[code_with] - validated_utilities[code_without])
        shapley[name] = accum
    return shapley


def _build_factorial_linear_weights(
    factor_names: Sequence[str],
) -> tuple[
    list[str],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """Precompute linear contrast weight vectors over ordered bit masks 0..(2^k - 1)."""
    num_factors = len(factor_names)
    num_cells = 1 << num_factors
    ordered_codes = [format(mask, f"0{num_factors}b") for mask in range(num_cells)]

    main_weights: dict[str, list[float]] = {}
    half_cells = float(1 << (num_factors - 1))
    for idx, fname in enumerate(factor_names):
        bit = 1 << (num_factors - 1 - idx)
        main_weights[fname] = [
            (1.0 / half_cells if (mask & bit) else -1.0 / half_cells)
            for mask in range(num_cells)
        ]

    interaction_weights: dict[str, list[float]] = {}
    if num_factors >= 2:
        bg_cells = float(1 << (num_factors - 2))
        for first_idx in range(num_factors):
            for second_idx in range(first_idx + 1, num_factors):
                pair_label = f"{factor_names[first_idx]}{factor_names[second_idx]}"
                bit_a = 1 << (num_factors - 1 - first_idx)
                bit_b = 1 << (num_factors - 1 - second_idx)
                weights_pair: list[float] = []
                for mask in range(num_cells):
                    has_a = bool(mask & bit_a)
                    has_b = bool(mask & bit_b)
                    sign = 1.0 if (has_a == has_b) else -1.0
                    weights_pair.append(sign / bg_cells)
                interaction_weights[pair_label] = weights_pair

    factorial_n = float(math.factorial(num_factors))
    shapley_weights: dict[str, list[float]] = {}
    for idx, fname in enumerate(factor_names):
        bit = 1 << (num_factors - 1 - idx)
        s_vec = [0.0] * num_cells
        for mask in range(num_cells):
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            coeff = (
                math.factorial(subset_size)
                * math.factorial(num_factors - subset_size - 1)
                / factorial_n
            )
            s_vec[mask | bit] += coeff
            s_vec[mask] -= coeff
        shapley_weights[fname] = s_vec

    return ordered_codes, main_weights, interaction_weights, shapley_weights


def compute_factorial_interactions_and_shapley(
    episode_coalition_outcomes: Mapping[str, Sequence[float]],
    *,
    cluster_ids: Sequence[str] | None = None,
    factor_names: Sequence[str] = ("C", "M", "K", "P"),
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed_key: str = "mdm-factorial-bootstrap-v1",
) -> FactorialEffectSummary:
    """Compute Yates/ANOVA main effects, two-way synergies, exact Shapley values, and bootstrap CIs.

    Args:
        episode_coalition_outcomes: Mapping from k-bit coalition code ('0000'..'1111') to a
            per-episode outcome sequence of identical length N.
        cluster_ids: Optional length-N episode cluster identifiers for cluster-robust resampling.
        factor_names: Ordered sequence of k unique factor identifiers.
        n_bootstrap: Number of deterministic HMAC-seeded cluster bootstrap draws (default 2,000).
        alpha: Family-wise significance level for confidence intervals and Holm-Bonferroni tests.
        seed_key: Deterministic HMAC seed domain string.

    Returns:
        ``FactorialEffectSummary`` with main effects, all two-way synergies, exact Shapley values,
        95% cluster-bootstrap confidence intervals, and Holm-Bonferroni adjusted p-values.

    Raises:
        ValueError: If inputs have missing codes, mismatched episode counts, non-finite values,
            or invalid bootstrap/alpha parameters.
    """
    if isinstance(factor_names, (str, bytes)) or not isinstance(factor_names, Sequence):
        raise ValueError("factor_names must be a non-empty sequence of unique strings")
    clean_factors = tuple(str(name).strip() for name in factor_names)
    num_factors = len(clean_factors)
    if num_factors == 0 or any(not name for name in clean_factors) or len(set(clean_factors)) != num_factors:
        raise ValueError("factor_names must be non-empty and unique")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be an integer >= 100, got {n_bootstrap!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if not isinstance(seed_key, str) or not seed_key.strip():
        raise ValueError("seed_key must be a non-empty string")
    if not isinstance(episode_coalition_outcomes, Mapping):
        raise ValueError("episode_coalition_outcomes must be a mapping")

    ordered_codes, main_weights, interaction_weights, shapley_weights = _build_factorial_linear_weights(
        clean_factors
    )
    missing_codes = set(ordered_codes) - set(episode_coalition_outcomes.keys())
    if missing_codes:
        raise ValueError(f"missing factorial coalition codes: {sorted(missing_codes)}")

    num_cells = len(ordered_codes)
    first_seq = episode_coalition_outcomes[ordered_codes[0]]
    n_episodes = len(first_seq)
    if n_episodes < 2:
        raise ValueError(f"at least 2 episodes are required for factorial inference, got {n_episodes}")

    matrix_by_cell: list[list[float]] = []
    for code in ordered_codes:
        seq = episode_coalition_outcomes[code]
        if len(seq) != n_episodes:
            raise ValueError(
                f"coalition {code!r} has {len(seq)} episodes, expected {n_episodes}"
            )
        clean_col: list[float] = []
        for idx, raw_val in enumerate(seq):
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)) or not math.isfinite(float(raw_val)):
                raise ValueError(f"non-finite outcome at coalition {code!r}, episode {idx}: {raw_val!r}")
            clean_col.append(float(raw_val))
        matrix_by_cell.append(clean_col)

    if cluster_ids is not None:
        if len(cluster_ids) != n_episodes:
            raise ValueError(
                f"cluster_ids length {len(cluster_ids)} must match n_episodes {n_episodes}"
            )
        resolved_clusters = [str(cid).strip() for cid in cluster_ids]
        if any(not cid for cid in resolved_clusters):
            raise ValueError("cluster_ids entries must be non-empty strings")
    else:
        resolved_clusters = [f"ep_{idx:06d}" for idx in range(n_episodes)]

    cluster_order: list[str] = []
    cluster_index_map: dict[str, int] = {}
    cluster_sums: list[list[float]] = []
    cluster_counts: list[int] = []

    for ep_idx, cid in enumerate(resolved_clusters):
        c_pos = cluster_index_map.get(cid)
        if c_pos is None:
            c_pos = len(cluster_order)
            cluster_index_map[cid] = c_pos
            cluster_order.append(cid)
            cluster_sums.append([0.0] * num_cells)
            cluster_counts.append(0)
        row_acc = cluster_sums[c_pos]
        for cell_idx in range(num_cells):
            row_acc[cell_idx] += matrix_by_cell[cell_idx][ep_idx]
        cluster_counts[c_pos] += 1

    n_clusters = len(cluster_order)
    if n_clusters < 2:
        raise ValueError(f"at least 2 distinct clusters are required for bootstrap CIs, got {n_clusters}")

    point_means_vec = [
        sum(matrix_by_cell[cell_idx]) / float(n_episodes) for cell_idx in range(num_cells)
    ]
    coalition_means = {ordered_codes[idx]: point_means_vec[idx] for idx in range(num_cells)}
    total_gain = point_means_vec[-1] - point_means_vec[0]

    main_effects = {
        fname: sum(w * m for w, m in zip(w_vec, point_means_vec))
        for fname, w_vec in main_weights.items()
    }
    two_way_interactions = {
        pair_name: sum(w * m for w, m in zip(w_vec, point_means_vec))
        for pair_name, w_vec in interaction_weights.items()
    }
    shapley_values = compute_exact_shapley_values(coalition_means, clean_factors)

    boot_main: dict[str, list[float]] = {fname: [] for fname in main_weights}
    boot_inter: dict[str, list[float]] = {pname: [] for pname in interaction_weights}
    boot_shapley: dict[str, list[float]] = {fname: [] for fname in shapley_weights}

    for draw_idx in range(n_bootstrap):
        sampled_totals = [0.0] * num_cells
        sampled_episode_count = 0
        for pick_idx in range(n_clusters):
            chosen_cluster = (
                hmac_uint64(seed_key, f"b{draw_idx}:g{pick_idx}", "cluster_bootstrap") % n_clusters
            )
            c_vec = cluster_sums[chosen_cluster]
            sampled_episode_count += cluster_counts[chosen_cluster]
            for cell_idx in range(num_cells):
                sampled_totals[cell_idx] += c_vec[cell_idx]
        inv_count = 1.0 / float(sampled_episode_count)
        draw_means = [val * inv_count for val in sampled_totals]

        for fname, w_vec in main_weights.items():
            boot_main[fname].append(sum(w * m for w, m in zip(w_vec, draw_means)))
        for pname, w_vec in interaction_weights.items():
            boot_inter[pname].append(sum(w * m for w, m in zip(w_vec, draw_means)))
        for fname, w_vec in shapley_weights.items():
            boot_shapley[fname].append(sum(w * m for w, m in zip(w_vec, draw_means)))

    lower_q = 0.5 * float(alpha)
    upper_q = 1.0 - lower_q

    main_effect_cis: dict[str, tuple[float, float]] = {}
    for fname, draws in boot_main.items():
        sorted_draws = sorted(draws)
        main_effect_cis[fname] = (
            _percentile_from_sorted(sorted_draws, lower_q),
            _percentile_from_sorted(sorted_draws, upper_q),
        )

    two_way_interaction_cis: dict[str, tuple[float, float]] = {}
    for pname, draws in boot_inter.items():
        sorted_draws = sorted(draws)
        two_way_interaction_cis[pname] = (
            _percentile_from_sorted(sorted_draws, lower_q),
            _percentile_from_sorted(sorted_draws, upper_q),
        )

    shapley_cis: dict[str, tuple[float, float]] = {}
    shapley_raw_p_list: list[float] = []
    shapley_keys_ordered = list(clean_factors)
    for fname in shapley_keys_ordered:
        draws = boot_shapley[fname]
        sorted_draws = sorted(draws)
        shapley_cis[fname] = (
            _percentile_from_sorted(sorted_draws, lower_q),
            _percentile_from_sorted(sorted_draws, upper_q),
        )
        shapley_raw_p_list.append(_two_sided_bootstrap_wald_p(shapley_values[fname], draws))

    effect_keys_ordered: list[str] = list(main_weights.keys()) + list(interaction_weights.keys())
    raw_p_list: list[float] = []
    raw_p_values: dict[str, float] = {}
    for fname in main_weights:
        p_val = _two_sided_bootstrap_wald_p(main_effects[fname], boot_main[fname])
        raw_p_values[fname] = p_val
        raw_p_list.append(p_val)
    for pname in interaction_weights:
        p_val = _two_sided_bootstrap_wald_p(two_way_interactions[pname], boot_inter[pname])
        raw_p_values[pname] = p_val
        raw_p_list.append(p_val)

    adjusted_p_list, rejected_list = holm_bonferroni_adjust(raw_p_list, alpha=float(alpha))
    holm_adjusted_p_values = {
        key: adjusted_p_list[idx] for idx, key in enumerate(effect_keys_ordered)
    }
    significant_effects = {
        key: rejected_list[idx] for idx, key in enumerate(effect_keys_ordered)
    }

    shapley_adj_p_list, _ = holm_bonferroni_adjust(shapley_raw_p_list, alpha=float(alpha))
    shapley_holm_p_values = {
        key: shapley_adj_p_list[idx] for idx, key in enumerate(shapley_keys_ordered)
    }

    return FactorialEffectSummary(
        factor_names=clean_factors,
        n_episodes=n_episodes,
        n_clusters=n_clusters,
        n_bootstrap=n_bootstrap,
        coalition_means=coalition_means,
        total_gain=total_gain,
        main_effects=main_effects,
        main_effect_cis=main_effect_cis,
        two_way_interactions=two_way_interactions,
        two_way_interaction_cis=two_way_interaction_cis,
        shapley_values=shapley_values,
        shapley_cis=shapley_cis,
        raw_p_values=raw_p_values,
        holm_adjusted_p_values=holm_adjusted_p_values,
        significant_effects=significant_effects,
        shapley_holm_p_values=shapley_holm_p_values,
    )





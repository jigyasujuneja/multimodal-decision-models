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

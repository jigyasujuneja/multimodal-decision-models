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

"""Distribution-Free Split Conformal Prediction and Regularized Adaptive Prediction Sets (LAC, APS, RAPS).

Implements finite-sample marginal coverage guarantees P(Y_{n+1} in C_alpha(X_{n+1})) >= 1 - alpha
under exchangeability via the exact ceil((n + 1) * (1 - alpha)) / n empirical quantile:
  - "lac": Least Ambiguous Set-Valued Classifiers (Sadinle et al., 2019), s(x, y) = 1 - pi_y(x).
  - "aps": Adaptive Prediction Sets (Romano et al., 2020), cumulative probability mass prior to rank(y)
    in deterministic mode (u = 0) or randomized with u ~ Uniform(0, 1).
  - "raps": Regularized Adaptive Prediction Sets (Angelopoulos et al., 2021), adding
    lam_reg * max(0, rank(y) - k_reg) to shrink tail set sizes on wide-option slots.
"""

from __future__ import annotations

import math
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

DEFAULT_CONFORMAL_ABSTAIN_KEYS: frozenset[str] = frozenset(
    {
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
)

_VALID_CONFORMAL_METHODS: frozenset[str] = frozenset({"lac", "aps", "raps"})
_TIE_TOL: float = 1e-12
_MAX_SEEDED_STREAMS: int = 256
_RNG_LOCK = threading.Lock()
_DEFAULT_RNG = random.Random()
_SEEDED_STREAMS: OrderedDict[int, random.Random] = OrderedDict()


def _store_seeded_rng(seed: int, rng_obj: random.Random) -> None:
    with _RNG_LOCK:
        _SEEDED_STREAMS[seed] = rng_obj
        _SEEDED_STREAMS.move_to_end(seed)
        while len(_SEEDED_STREAMS) > _MAX_SEEDED_STREAMS:
            _SEEDED_STREAMS.popitem(last=False)


def _get_or_create_inference_rng(seed: int | None, rng: random.Random | None) -> random.Random:
    if rng is not None:
        return rng
    if seed is None:
        return _DEFAULT_RNG
    key = int(seed)
    with _RNG_LOCK:
        existing = _SEEDED_STREAMS.get(key)
        if existing is not None:
            _SEEDED_STREAMS.move_to_end(key)
            return existing
        created = random.Random(key)
        _SEEDED_STREAMS[key] = created
        while len(_SEEDED_STREAMS) > _MAX_SEEDED_STREAMS:
            _SEEDED_STREAMS.popitem(last=False)
        return created


def _partition_tie_groups(
    ranked_keys: Sequence[str],
    norm_probs: Mapping[str, float],
) -> list[list[str]]:
    groups: list[list[str]] = []
    for key in ranked_keys:
        p_k = norm_probs[key]
        if not groups or abs(p_k - norm_probs[groups[-1][0]]) > _TIE_TOL:
            groups.append([key])
        else:
            groups[-1].append(key)
    return groups


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Calibrated conformal nonconformity threshold and calibration summary."""

    q_hat: float
    alpha: float
    n_calib: int
    method: str
    empirical_calib_coverage: float = 1.0
    lam_reg: float = 0.01
    k_reg: int = 1


@dataclass(frozen=True)
class ConformalSetResult:
    """Conformal prediction set and pre-fusion cascade escalation decision for a single slot."""

    prediction_set: list[str]
    q_hat: float
    singleton: bool
    contains_abstain: bool
    should_escalate: bool
    method: str = "raps"


def _normalize_and_rank_probs(probs: Mapping[str, float]) -> tuple[list[str], dict[str, float], dict[str, int]]:
    if not isinstance(probs, Mapping) or not probs:
        raise ValueError("probability mapping must be non-empty")
    total = 0.0
    clean: dict[str, float] = {}
    for raw_k, raw_v in probs.items():
        key = str(raw_k)
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
            raise ValueError(f"probability for {key!r} must be a non-boolean number, got {raw_v!r}")
        val = float(raw_v)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"probability for {key!r} must be finite and >= 0, got {raw_v!r}")
        clean[key] = val
        total += val
    if total <= 0.0:
        raise ValueError("probability mapping sum must be > 0")
    norm = {k: v / total for k, v in clean.items()}
    ranked_keys = sorted(norm.keys(), key=lambda k: (-norm[k], k))
    ranks = {k: idx + 1 for idx, k in enumerate(ranked_keys)}
    return ranked_keys, norm, ranks


def _validate_conformal_hyperparams(method: str, lam_reg: float, k_reg: int) -> str:
    if not isinstance(method, str):
        raise ValueError(f"conformal method must be a string, got {method!r}")
    clean_method = method.strip().lower()
    if clean_method not in _VALID_CONFORMAL_METHODS:
        raise ValueError(f"unsupported conformal method {method!r}; expected one of {sorted(_VALID_CONFORMAL_METHODS)}")
    if isinstance(lam_reg, bool) or not isinstance(lam_reg, (int, float)) or not math.isfinite(float(lam_reg)) or float(lam_reg) < 0.0:
        raise ValueError(f"lam_reg must be finite and >= 0, got {lam_reg!r}")
    if isinstance(k_reg, bool) or not isinstance(k_reg, int) or k_reg < 0:
        raise ValueError(f"k_reg must be >= 0, got {k_reg!r}")
    return clean_method


def _compute_all_nonconformity_scores(
    ranked_keys: Sequence[str],
    norm_probs: Mapping[str, float],
    method: str,
    lam_reg: float,
    k_reg: int,
    *,
    randomized: bool = False,
    active_rng: random.Random | None = None,
    use_global_lock: bool = False,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    groups = _partition_tie_groups(ranked_keys, norm_probs)

    if not randomized:
        cum_strictly_greater = 0.0
        strict_count = 0
        for grp in groups:
            p_grp = norm_probs[grp[0]]
            strict_rank = strict_count + 1
            if method == "lac":
                grp_score = 1.0 - p_grp
            elif method == "aps":
                grp_score = cum_strictly_greater
            else:
                penalty = lam_reg * float(max(0, strict_rank - k_reg))
                grp_score = cum_strictly_greater + penalty
            for key in grp:
                scores[key] = grp_score
            cum_strictly_greater += p_grp * float(len(grp))
            strict_count += len(grp)
        return scores

    rng_use = active_rng if active_rng is not None else _DEFAULT_RNG
    if use_global_lock:
        with _RNG_LOCK:
            u_draw = rng_use.random()
            ordered_rand: list[str] = []
            for grp in groups:
                if len(grp) > 1:
                    grp_copy = list(grp)
                    rng_use.shuffle(grp_copy)
                    ordered_rand.extend(grp_copy)
                else:
                    ordered_rand.append(grp[0])
    else:
        u_draw = rng_use.random()
        ordered_rand = []
        for grp in groups:
            if len(grp) > 1:
                grp_copy = list(grp)
                rng_use.shuffle(grp_copy)
                ordered_rand.extend(grp_copy)
            else:
                ordered_rand.append(grp[0])

    cum_before = 0.0
    for idx, key in enumerate(ordered_rand):
        rank_1based = idx + 1
        p_k = norm_probs[key]
        if method == "lac":
            scores[key] = 1.0 - p_k
        elif method == "aps":
            scores[key] = cum_before + (u_draw * p_k)
        else:
            penalty = lam_reg * float(max(0, rank_1based - k_reg))
            scores[key] = cum_before + (u_draw * p_k) + penalty
        cum_before += p_k
    return scores


def calibrate_conformal_quantile(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    alpha: float = 0.10,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    randomized: bool = False,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> ConformalCalibrationResult:
    """Compute the finite-sample split conformal quantile q_hat at miscoverage level alpha in (0, 1).

    Uses the exact order statistic index m = ceil((n + 1) * (1 - alpha)) over n exchangeable
    calibration nonconformity scores. When m > n, returns q_hat = +inf so prediction sets cover
    the full label space conservatively.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0 or float(alpha) >= 1.0:
        raise ValueError(f"alpha must be finite and strictly in (0, 1), got {alpha!r}")
    if not calib_probs or len(calib_probs) != len(calib_labels):
        raise ValueError("calib_probs and calib_labels must be non-empty sequences of equal length")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    use_lock = False
    cal_rng: random.Random | None = None
    if randomized:
        if rng is not None:
            cal_rng = rng
        elif seed is not None:
            cal_rng = random.Random(int(seed))
        else:
            cal_rng = _DEFAULT_RNG
            use_lock = True

    calib_scores: list[float] = []
    for idx, (dist, gold_raw) in enumerate(zip(calib_probs, calib_labels)):
        gold = str(gold_raw)
        ranked_keys, norm_probs, _ = _normalize_and_rank_probs(dist)
        if gold not in norm_probs:
            raise ValueError(f"calibration label {gold!r} at index {idx} not present in option probabilities")
        item_scores = _compute_all_nonconformity_scores(
            ranked_keys,
            norm_probs,
            clean_method,
            lam_reg,
            k_reg,
            randomized=randomized,
            active_rng=cal_rng,
            use_global_lock=use_lock,
        )
        calib_scores.append(item_scores[gold])

    if randomized and rng is None and seed is not None and cal_rng is not None:
        _store_seeded_rng(int(seed), cal_rng)

    n = len(calib_scores)
    sorted_scores = sorted(calib_scores)
    order_idx = int(math.ceil(float(n + 1) * (1.0 - alpha) - 1e-12))
    if order_idx > n:
        q_hat = float("inf")
        emp_cov = 1.0
    else:
        pos = max(0, order_idx - 1)
        q_hat = sorted_scores[pos]
        covered = sum(1 for s in calib_scores if s <= q_hat + 1e-12)
        emp_cov = float(covered) / float(n)

    return ConformalCalibrationResult(
        q_hat=q_hat,
        alpha=alpha,
        n_calib=n,
        method=clean_method,
        empirical_calib_coverage=emp_cov,
        lam_reg=lam_reg,
        k_reg=k_reg,
    )


def conformal_prediction_set(
    probs: Mapping[str, float],
    q_hat: float,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    randomized: bool = False,
    seed: int | None = None,
    rng: random.Random | None = None,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct the calibrated prediction set C_alpha(x) = {y : s(x, y) <= q_hat} in probability rank order.

    Flags should_escalate = True whenever |C_alpha(x)| != 1 or any abstain/catch-all option belongs to C_alpha(x).
    """
    if isinstance(q_hat, bool) or not isinstance(q_hat, (int, float)) or math.isnan(float(q_hat)) or float(q_hat) < 0.0:
        raise ValueError(f"q_hat must be >= 0 and not NaN, got {q_hat!r}")
    q_hat_f = float(q_hat)
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)
    ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)

    active_rng = _get_or_create_inference_rng(seed, rng) if randomized else None
    use_lock = bool(randomized and rng is None)
    scores = _compute_all_nonconformity_scores(
        ranked_keys,
        norm_probs,
        clean_method,
        lam_reg,
        k_reg,
        randomized=randomized,
        active_rng=active_rng,
        use_global_lock=use_lock,
    )

    if math.isinf(q_hat_f):
        pred_set = list(ranked_keys)
    else:
        pred_set = [k for k in ranked_keys if scores[k] <= q_hat_f + 1e-12]

    effective_abstain = DEFAULT_CONFORMAL_ABSTAIN_KEYS if abstain_keys is None else abstain_keys
    abstain_lookup = {str(k).strip().lower() for k in effective_abstain if str(k).strip()}
    singleton = len(pred_set) == 1
    contains_abstain = any(str(k).strip().lower() in abstain_lookup for k in pred_set)
    should_escalate = (not singleton) or contains_abstain

    return ConformalSetResult(
        prediction_set=pred_set,
        q_hat=q_hat_f,
        singleton=singleton,
        contains_abstain=contains_abstain,
        should_escalate=should_escalate,
        method=clean_method,
    )


@dataclass(frozen=True)
class MondrianCalibrationResult:
    """Class-conditional or group-conditional (Mondrian) conformal quantile map."""

    group_q_hats: dict[str, float]
    group_counts: dict[str, int]
    global_q_hat: float
    alpha: float
    method: str
    class_conditional: bool = True
    lam_reg: float = 0.01
    k_reg: int = 1


@dataclass(frozen=True)
class MultiSlotJointConformalResult:
    """Simultaneous multi-slot family-wise conformal calibration across S coupled slots."""

    joint_max_q_hat: float
    bonferroni_slot_q_hats: dict[str, float]
    slot_names: tuple[str, ...]
    alpha: float
    n_calib: int
    empirical_joint_coverage: float
    method: str = "raps"
    lam_reg: float = 0.01
    k_reg: int = 1


def calibrate_mondrian_conformal_quantiles(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    calib_groups: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> MondrianCalibrationResult:
    """Calibrate per-class (when calib_groups is None) or per-group Mondrian conformal quantiles.

    Guarantees group-conditional or class-conditional coverage P(Y in C_alpha(X) | G = g) >= 1 - alpha
    via order index ceil((n_g + 1) * (1 - alpha)) within each stratum g, falling back to global_q_hat
    for unseen strata.
    """
    global_cal = calibrate_conformal_quantile(
        calib_probs=calib_probs,
        calib_labels=calib_labels,
        alpha=alpha,
        method=method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        randomized=False,
    )
    if calib_groups is not None and len(calib_groups) != len(calib_labels):
        raise ValueError("calib_groups must have the same length as calib_labels")

    class_cond = calib_groups is None
    grouped_probs: dict[str, list[Mapping[str, float]]] = {}
    grouped_labels: dict[str, list[str]] = {}

    for idx, (dist, lbl_raw) in enumerate(zip(calib_probs, calib_labels)):
        lbl = str(lbl_raw)
        grp = lbl if calib_groups is None else str(calib_groups[idx])
        grouped_probs.setdefault(grp, []).append(dist)
        grouped_labels.setdefault(grp, []).append(lbl)

    group_q_hats: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for grp in sorted(grouped_probs.keys()):
        sub_cal = calibrate_conformal_quantile(
            calib_probs=grouped_probs[grp],
            calib_labels=grouped_labels[grp],
            alpha=alpha,
            method=global_cal.method,
            lam_reg=lam_reg,
            k_reg=k_reg,
            randomized=False,
        )
        group_q_hats[grp] = sub_cal.q_hat
        group_counts[grp] = sub_cal.n_calib

    return MondrianCalibrationResult(
        group_q_hats=group_q_hats,
        group_counts=group_counts,
        global_q_hat=global_cal.q_hat,
        alpha=alpha,
        method=global_cal.method,
        class_conditional=class_cond,
        lam_reg=lam_reg,
        k_reg=k_reg,
    )


def mondrian_conformal_prediction_set(
    probs: Mapping[str, float],
    mondrian_cal: MondrianCalibrationResult,
    *,
    group: str | None = None,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct a Mondrian prediction set using per-class thresholds (when group is None) or a stratum threshold."""
    clean_method = _validate_conformal_hyperparams(
        mondrian_cal.method, mondrian_cal.lam_reg, mondrian_cal.k_reg
    )
    ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)
    scores = _compute_all_nonconformity_scores(
        ranked_keys,
        norm_probs,
        clean_method,
        mondrian_cal.lam_reg,
        mondrian_cal.k_reg,
        randomized=False,
    )

    if group is not None:
        q_use = mondrian_cal.group_q_hats.get(str(group), mondrian_cal.global_q_hat)
        return conformal_prediction_set(
            probs=probs,
            q_hat=q_use,
            method=clean_method,
            lam_reg=mondrian_cal.lam_reg,
            k_reg=mondrian_cal.k_reg,
            randomized=False,
            abstain_keys=abstain_keys,
        )

    pred_set: list[str] = []
    max_q_used = 0.0
    for k in ranked_keys:
        q_k = mondrian_cal.group_q_hats.get(k, mondrian_cal.global_q_hat)
        max_q_used = max(max_q_used, q_k)
        if math.isinf(q_k) or scores[k] <= q_k + 1e-12:
            pred_set.append(k)

    effective_abstain = DEFAULT_CONFORMAL_ABSTAIN_KEYS if abstain_keys is None else abstain_keys
    abstain_lookup = {str(k).strip().lower() for k in effective_abstain if str(k).strip()}
    singleton = len(pred_set) == 1
    contains_abstain = any(str(k).strip().lower() in abstain_lookup for k in pred_set)
    should_escalate = (not singleton) or contains_abstain
    return ConformalSetResult(
        prediction_set=pred_set,
        q_hat=max_q_used,
        singleton=singleton,
        contains_abstain=contains_abstain,
        should_escalate=should_escalate,
        method=clean_method,
    )


def weighted_conformal_prediction_set(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    calib_weights: Sequence[float],
    test_probs: Mapping[str, float],
    test_weight: float = 1.0,
    alpha: float = 0.10,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct a likelihood-ratio weighted split conformal prediction set (Tibshirani et al., 2019).

    Guarantees valid coverage under covariate shift where w(x) = dP_test(x) / dP_calib(x) > 0.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0 or float(alpha) >= 1.0:
        raise ValueError(f"alpha must be finite and strictly in (0, 1), got {alpha!r}")
    if not calib_probs or len(calib_probs) != len(calib_labels) or len(calib_probs) != len(calib_weights):
        raise ValueError("calib_probs, calib_labels, and calib_weights must be non-empty sequences of equal length")
    if isinstance(test_weight, bool) or not isinstance(test_weight, (int, float)) or not math.isfinite(float(test_weight)) or float(test_weight) <= 0.0:
        raise ValueError(f"test_weight must be finite and > 0, got {test_weight!r}")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    pairs: list[tuple[float, float]] = []
    w_sum = float(test_weight)
    for idx, (dist, gold_raw, w_raw) in enumerate(zip(calib_probs, calib_labels, calib_weights)):
        if isinstance(w_raw, bool) or not isinstance(w_raw, (int, float)) or not math.isfinite(float(w_raw)) or float(w_raw) <= 0.0:
            raise ValueError(f"calib_weights[{idx}] must be finite and > 0, got {w_raw!r}")
        w_val = float(w_raw)
        gold = str(gold_raw)
        ranked_keys, norm_probs, _ = _normalize_and_rank_probs(dist)
        if gold not in norm_probs:
            raise ValueError(f"calibration label {gold!r} at index {idx} not present in option probabilities")
        item_scores = _compute_all_nonconformity_scores(
            ranked_keys, norm_probs, clean_method, lam_reg, k_reg, randomized=False
        )
        pairs.append((item_scores[gold], w_val))
        w_sum += w_val

    pairs.sort(key=lambda item: item[0])
    target_mass = 1.0 - float(alpha)
    cum_mass = 0.0
    q_hat_w = float("inf")
    for score_i, w_i in pairs:
        cum_mass += w_i / w_sum
        if cum_mass >= target_mass - 1e-12:
            q_hat_w = score_i
            break

    return conformal_prediction_set(
        probs=test_probs,
        q_hat=q_hat_w,
        method=clean_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        randomized=False,
        abstain_keys=abstain_keys,
    )


def calibrate_multislot_joint_conformal(
    calib_slot_probs: Sequence[Mapping[str, Mapping[str, float]]],
    calib_slot_labels: Sequence[Mapping[str, str]],
    alpha: float = 0.10,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> MultiSlotJointConformalResult:
    """Calibrate simultaneous family-wise conformal prediction across S coupled decision slots.

    Computes both the exact max-score quantile M_i = max_{s in S} s(X_{i,s}, Y_{i,s}) at order
    index ceil((n + 1) * (1 - alpha)) and Bonferroni per-slot quantiles at level alpha / S.
    Guarantees P(forall s in S: Y_{n+1,s} in C_alpha^{(s)}(X_{n+1})) >= 1 - alpha.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0 or float(alpha) >= 1.0:
        raise ValueError(f"alpha must be finite and strictly in (0, 1), got {alpha!r}")
    if not calib_slot_probs or len(calib_slot_probs) != len(calib_slot_labels):
        raise ValueError("calib_slot_probs and calib_slot_labels must be non-empty sequences of equal length")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    first_slots = sorted(str(k) for k in calib_slot_probs[0].keys())
    if not first_slots:
        raise ValueError("calib_slot_probs items must contain at least one slot")
    slot_tuple = tuple(first_slots)
    s_count = len(slot_tuple)

    max_scores: list[float] = []
    per_slot_probs: dict[str, list[Mapping[str, float]]] = {s: [] for s in slot_tuple}
    per_slot_labels: dict[str, list[str]] = {s: [] for s in slot_tuple}

    for idx, (slot_dist_map, slot_lbl_map) in enumerate(zip(calib_slot_probs, calib_slot_labels)):
        row_max = 0.0
        for s_name in slot_tuple:
            if s_name not in slot_dist_map or s_name not in slot_lbl_map:
                raise ValueError(f"missing slot {s_name!r} at calibration index {idx}")
            dist = slot_dist_map[s_name]
            gold = str(slot_lbl_map[s_name])
            ranked_keys, norm_probs, _ = _normalize_and_rank_probs(dist)
            if gold not in norm_probs:
                raise ValueError(f"label {gold!r} missing from slot {s_name!r} at index {idx}")
            s_scores = _compute_all_nonconformity_scores(
                ranked_keys, norm_probs, clean_method, lam_reg, k_reg, randomized=False
            )
            row_max = max(row_max, s_scores[gold])
            per_slot_probs[s_name].append(dist)
            per_slot_labels[s_name].append(gold)
        max_scores.append(row_max)

    n = len(max_scores)
    sorted_max = sorted(max_scores)
    order_idx = int(math.ceil(float(n + 1) * (1.0 - alpha) - 1e-12))
    if order_idx > n:
        joint_q = float("inf")
        emp_joint = 1.0
    else:
        joint_q = sorted_max[max(0, order_idx - 1)]
        emp_joint = float(sum(1 for ms in max_scores if ms <= joint_q + 1e-12)) / float(n)

    bonf_alpha = float(alpha) / float(s_count)
    bonf_q_hats: dict[str, float] = {}
    for s_name in slot_tuple:
        s_cal = calibrate_conformal_quantile(
            calib_probs=per_slot_probs[s_name],
            calib_labels=per_slot_labels[s_name],
            alpha=bonf_alpha,
            method=clean_method,
            lam_reg=lam_reg,
            k_reg=k_reg,
            randomized=False,
        )
        bonf_q_hats[s_name] = s_cal.q_hat

    return MultiSlotJointConformalResult(
        joint_max_q_hat=joint_q,
        bonferroni_slot_q_hats=bonf_q_hats,
        slot_names=slot_tuple,
        alpha=alpha,
        n_calib=n,
        empirical_joint_coverage=emp_joint,
        method=clean_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
    )


def multislot_joint_conformal_sets(
    slot_probs: Mapping[str, Mapping[str, float]],
    joint_cal: MultiSlotJointConformalResult,
    *,
    use_bonferroni: bool = False,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> dict[str, ConformalSetResult]:
    """Construct simultaneous family-wise conformal prediction sets across all S slots."""
    out: dict[str, ConformalSetResult] = {}
    for s_name in joint_cal.slot_names:
        if s_name not in slot_probs:
            raise ValueError(f"missing slot {s_name!r} in slot_probs")
        q_s = (
            joint_cal.bonferroni_slot_q_hats[s_name]
            if use_bonferroni
            else joint_cal.joint_max_q_hat
        )
        out[s_name] = conformal_prediction_set(
            probs=slot_probs[s_name],
            q_hat=q_s,
            method=joint_cal.method,
            lam_reg=joint_cal.lam_reg,
            k_reg=joint_cal.k_reg,
            randomized=False,
            abstain_keys=abstain_keys,
        )
    return out

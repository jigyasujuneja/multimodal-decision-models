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
    {"0", "none", "none_of_the_above", "none_visible", "oos", "abstain"}
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
    if not probs:
        raise ValueError("probability mapping must be non-empty")
    total = 0.0
    clean: dict[str, float] = {}
    for raw_k, raw_v in probs.items():
        key = str(raw_k)
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
    clean_method = method.strip().lower()
    if clean_method not in _VALID_CONFORMAL_METHODS:
        raise ValueError(f"unsupported conformal method {method!r}; expected one of {sorted(_VALID_CONFORMAL_METHODS)}")
    if not math.isfinite(lam_reg) or lam_reg < 0.0:
        raise ValueError(f"lam_reg must be finite and >= 0, got {lam_reg!r}")
    if k_reg < 0:
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
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
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
    abstain_keys: Collection[str] = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct the calibrated prediction set C_alpha(x) = {y : s(x, y) <= q_hat} in probability rank order.

    Flags should_escalate = True whenever |C_alpha(x)| != 1 or any abstain/catch-all option belongs to C_alpha(x).
    """
    if math.isnan(q_hat) or q_hat < 0.0:
        raise ValueError(f"q_hat must be >= 0 and not NaN, got {q_hat!r}")
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

    if math.isinf(q_hat):
        pred_set = list(ranked_keys)
    else:
        pred_set = [k for k in ranked_keys if scores[k] <= q_hat + 1e-12]

    abstain_lookup = {str(k) for k in abstain_keys}
    singleton = len(pred_set) == 1
    contains_abstain = any(k in abstain_lookup for k in pred_set)
    should_escalate = (not singleton) or contains_abstain

    return ConformalSetResult(
        prediction_set=pred_set,
        q_hat=q_hat,
        singleton=singleton,
        contains_abstain=contains_abstain,
        should_escalate=should_escalate,
        method=clean_method,
    )

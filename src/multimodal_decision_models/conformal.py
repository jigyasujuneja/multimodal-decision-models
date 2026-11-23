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

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import math
import random
import struct
import threading
from typing import Callable, Collection, Mapping, Sequence

DEFAULT_CONFORMAL_SEED_KEY: str = "dgem-conformal-v1"

STANDARD_CONFORMAL_ABSTAIN_KEYS: frozenset[str] = frozenset(
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

DEFAULT_CONFORMAL_ABSTAIN_KEYS: frozenset[str] = frozenset()

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


def hmac_uint64(seed_key: str, item_key: str, tag: str) -> int:
    """Compute a stateless big-endian uint64 HMAC-SHA256 digest matching Go and TypeScript SDKs.

    Args:
        seed_key: Secret or deterministic domain seed key (defaults to ``DEFAULT_CONFORMAL_SEED_KEY`` when empty).
        item_key: Per-example identifier ensuring order-independent tie-breaking across threadpools.
        tag: Domain-separation suffix such as ``"u"`` or ``"tie:<label>"``.

    Returns:
        Unsigned 64-bit integer unpacked from the first 8 bytes of ``HMAC-SHA256(seed_key, item_key + "\\x00" + tag)``.
    """
    if not isinstance(seed_key, str) or not isinstance(item_key, str) or not isinstance(tag, str):
        raise ValueError("seed_key, item_key, and tag must be strings")
    effective_key = seed_key if seed_key else DEFAULT_CONFORMAL_SEED_KEY
    digest = hmac.new(effective_key.encode("utf-8"), digestmod=hashlib.sha256)
    digest.update(item_key.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(tag.encode("utf-8"))
    return int(struct.unpack(">Q", digest.digest()[:8])[0])


def hmac_uniform_float64(seed_key: str, item_key: str, tag: str) -> float:
    """Draw a stateless Uniform[0, 1) float64 with 53-bit mantissa precision via HMAC-SHA256.

    Args:
        seed_key: Secret or deterministic domain seed key.
        item_key: Per-example identifier.
        tag: Domain-separation suffix.

    Returns:
        Deterministic float in ``[0.0, 1.0)`` identical across Python, Go, and TypeScript.
    """
    top_53 = hmac_uint64(seed_key, item_key, tag) >> 11
    return float(top_53) * (1.0 / float(1 << 53))


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
    hmac_seed_key: str | None = None,
    hmac_item_key: str = "",
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

    if hmac_seed_key is not None:
        u_draw = hmac_uniform_float64(hmac_seed_key, hmac_item_key, "u")
        ordered_rand: list[str] = []
        for grp in groups:
            if len(grp) > 1:
                shuffled_group = sorted(
                    grp,
                    key=lambda label_key: (
                        hmac_uint64(hmac_seed_key, hmac_item_key, f"tie:{label_key}"),
                        label_key,
                    ),
                )
                ordered_rand.extend(shuffled_group)
            else:
                ordered_rand.append(grp[0])
    else:
        rng_use = active_rng if active_rng is not None else _DEFAULT_RNG
        if use_global_lock:
            with _RNG_LOCK:
                u_draw = rng_use.random()
                ordered_rand = []
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


def nonconformity_score(
    probs: Mapping[str, float],
    label: str,
    *,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    randomized: bool = False,
    seed: int | None = None,
    rng: random.Random | None = None,
    seed_key: str | None = None,
    item_key: str = "",
) -> float:
    """Compute the conformal nonconformity score s(x, y) for a candidate label.

    Args:
        probs: Option probability mapping for a single decision slot.
        label: Candidate or ground-truth option key present in ``probs``.
        method: Score method (``"lac"``, ``"aps"``, or ``"raps"``).
        lam_reg: RAPS regularization weight ``>= 0``.
        k_reg: RAPS unpenalized top-k cutoff ``>= 0``.
        randomized: Whether to enable randomized tie/rank smoothing.
        seed: Optional integer PRNG seed when ``randomized=True`` without HMAC keys.
        rng: Optional explicit ``random.Random`` instance when ``randomized=True``.
        seed_key: Optional HMAC-SHA256 seed key when ``randomized=True``.
        item_key: Per-example HMAC item identifier.

    Returns:
        Nonconformity score in ``[0, +inf)``.

    Raises:
        ValueError: If ``probs``, hyperparameters, or ``label`` are invalid.
    """
    if seed_key is not None and not isinstance(seed_key, str):
        raise ValueError("seed_key must be a string or None")
    if not isinstance(item_key, str):
        raise ValueError("item_key must be a string")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)
    ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)
    target = str(label)
    if target not in norm_probs:
        raise ValueError(f"label {target!r} not present in option probabilities")

    use_hmac = bool(
        randomized
        and (seed_key is not None or (seed is None and rng is None))
    )
    effective_seed = (
        (seed_key if seed_key is not None else DEFAULT_CONFORMAL_SEED_KEY)
        if use_hmac
        else None
    )
    active_rng = (
        (rng if rng is not None else random.Random(int(seed)))
        if (randomized and not use_hmac and seed is not None or rng is not None)
        else None
    )
    scores = _compute_all_nonconformity_scores(
        ranked_keys,
        norm_probs,
        clean_method,
        lam_reg,
        k_reg,
        randomized=randomized,
        active_rng=active_rng if (randomized and not use_hmac) else None,
        use_global_lock=False,
        hmac_seed_key=effective_seed,
        hmac_item_key=item_key,
    )
    return float(scores[target])


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
    seed_key: str | None = None,
    item_keys: Sequence[str] | None = None,
) -> ConformalCalibrationResult:
    """Compute the finite-sample split conformal quantile q_hat at miscoverage level alpha in (0, 1).

    Uses the exact order statistic index m = ceil((n + 1) * (1 - alpha)) over n exchangeable
    calibration nonconformity scores. When m > n, returns q_hat = +inf so prediction sets cover
    the full label space conservatively. Passing ``seed_key`` enables stateless, lock-free
    ``HMAC-SHA256`` tie-breaking with exact cross-language parity across Python, Go, and TypeScript.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0 or float(alpha) >= 1.0:
        raise ValueError(f"alpha must be finite and strictly in (0, 1), got {alpha!r}")
    if not calib_probs or len(calib_probs) != len(calib_labels):
        raise ValueError("calib_probs and calib_labels must be non-empty sequences of equal length")
    if seed_key is not None and not isinstance(seed_key, str):
        raise ValueError("seed_key must be a string or None")
    if item_keys is not None and len(item_keys) != len(calib_probs):
        raise ValueError("item_keys must match calib_probs length when provided")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    use_hmac = bool(randomized and (seed_key is not None or item_keys is not None))
    effective_seed_key = (
        (seed_key if seed_key is not None else DEFAULT_CONFORMAL_SEED_KEY)
        if use_hmac
        else None
    )

    use_lock = False
    cal_rng: random.Random | None = None
    if randomized and not use_hmac:
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
        current_item_key = (
            str(item_keys[idx]) if item_keys is not None else f"cal:{idx}"
        )
        item_scores = _compute_all_nonconformity_scores(
            ranked_keys,
            norm_probs,
            clean_method,
            lam_reg,
            k_reg,
            randomized=randomized,
            active_rng=cal_rng,
            use_global_lock=use_lock,
            hmac_seed_key=effective_seed_key,
            hmac_item_key=current_item_key,
        )
        calib_scores.append(item_scores[gold])

    if randomized and not use_hmac and rng is None and seed is not None and cal_rng is not None:
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
    seed_key: str | None = None,
    item_key: str = "",
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct the calibrated prediction set C_alpha(x) = {y : s(x, y) <= q_hat} in probability rank order.

    Flags should_escalate = True whenever |C_alpha(x)| != 1 or any abstain/catch-all option belongs to C_alpha(x).
    When ``seed_key`` (or ``item_key`` without integer ``seed``/``rng``) is provided under ``randomized=True``,
    uses stateless ``HMAC-SHA256`` tie-breaking for thread-safe, cross-language deterministic sets.
    """
    if isinstance(q_hat, bool) or not isinstance(q_hat, (int, float)) or math.isnan(float(q_hat)) or float(q_hat) < 0.0:
        raise ValueError(f"q_hat must be >= 0 and not NaN, got {q_hat!r}")
    if seed_key is not None and not isinstance(seed_key, str):
        raise ValueError("seed_key must be a string or None")
    if not isinstance(item_key, str):
        raise ValueError("item_key must be a string")
    q_hat_f = float(q_hat)
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)
    ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)

    use_hmac = bool(randomized and (seed_key is not None or (bool(item_key) and seed is None and rng is None)))
    effective_seed_key = (
        (seed_key if seed_key is not None else DEFAULT_CONFORMAL_SEED_KEY)
        if use_hmac
        else None
    )
    active_rng = _get_or_create_inference_rng(seed, rng) if (randomized and not use_hmac) else None
    use_lock = bool(randomized and not use_hmac and rng is None)
    scores = _compute_all_nonconformity_scores(
        ranked_keys,
        norm_probs,
        clean_method,
        lam_reg,
        k_reg,
        randomized=randomized,
        active_rng=active_rng,
        use_global_lock=use_lock,
        hmac_seed_key=effective_seed_key,
        hmac_item_key=item_key,
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


def label_shift_weighted_conformal_prediction_set(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    class_likelihood_ratios: Mapping[str, float],
    test_probs: Mapping[str, float],
    *,
    calib_covariate_weights: Sequence[float] | None = None,
    test_covariate_weight: float = 1.0,
    alpha: float = 0.10,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> ConformalSetResult:
    """Construct a conformal prediction set under combined label and covariate shift (Proposition 4).

    Follows Podkopaev & Ramdas (UAI 2021) and Tibshirani et al. (NeurIPS 2019): each calibration
    example i receives weight W_i = w_X(X_i) * rho(Y_i) where rho(y) = pi_test(y) / pi_cal(y) > 0,
    and candidate label y in Y is included in C_alpha(X_{n+1}) iff s(X_{n+1}, y) does not exceed
    the (1 - alpha)-weighted quantile evaluated with candidate-specific test weight
    W_{n+1}(y) = w_X(X_{n+1}) * rho(y).
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0 or float(alpha) >= 1.0:
        raise ValueError(f"alpha must be finite and strictly in (0, 1), got {alpha!r}")
    if not calib_probs or len(calib_probs) != len(calib_labels):
        raise ValueError("calib_probs and calib_labels must be non-empty sequences of equal length")
    if calib_covariate_weights is not None and len(calib_covariate_weights) != len(calib_probs):
        raise ValueError("calib_covariate_weights must match length of calib_probs")
    if isinstance(test_covariate_weight, bool) or not isinstance(test_covariate_weight, (int, float)) or not math.isfinite(float(test_covariate_weight)) or float(test_covariate_weight) <= 0.0:
        raise ValueError(f"test_covariate_weight must be finite and > 0, got {test_covariate_weight!r}")
    if not class_likelihood_ratios:
        raise ValueError("class_likelihood_ratios must be non-empty")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    rho_map: dict[str, float] = {}
    for k, raw_rho in class_likelihood_ratios.items():
        if isinstance(raw_rho, bool) or not isinstance(raw_rho, (int, float)) or not math.isfinite(float(raw_rho)) or float(raw_rho) <= 0.0:
            raise ValueError(f"class_likelihood_ratios[{k!r}] must be finite and > 0, got {raw_rho!r}")
        rho_map[str(k)] = float(raw_rho)

    pairs: list[tuple[float, float]] = []
    cal_w_sum = 0.0
    for idx, (dist, gold_raw) in enumerate(zip(calib_probs, calib_labels)):
        cov_w = 1.0
        if calib_covariate_weights is not None:
            raw_cw = calib_covariate_weights[idx]
            if isinstance(raw_cw, bool) or not isinstance(raw_cw, (int, float)) or not math.isfinite(float(raw_cw)) or float(raw_cw) <= 0.0:
                raise ValueError(f"calib_covariate_weights[{idx}] must be finite and > 0, got {raw_cw!r}")
            cov_w = float(raw_cw)
        gold = str(gold_raw)
        ranked_keys, norm_probs, _ = _normalize_and_rank_probs(dist)
        if gold not in norm_probs:
            raise ValueError(f"calibration label {gold!r} at index {idx} not present in option probabilities")
        item_scores = _compute_all_nonconformity_scores(
            ranked_keys, norm_probs, clean_method, lam_reg, k_reg, randomized=False
        )
        w_i = cov_w * rho_map.get(gold, 1.0)
        pairs.append((item_scores[gold], w_i))
        cal_w_sum += w_i

    pairs.sort(key=lambda item: item[0])
    target_mass = 1.0 - float(alpha)

    test_ranked, test_norm, _ = _normalize_and_rank_probs(test_probs)
    test_scores = _compute_all_nonconformity_scores(
        test_ranked, test_norm, clean_method, lam_reg, k_reg, randomized=False
    )

    pred_set: list[str] = []
    max_q_used = 0.0
    for cand_k in test_ranked:
        w_test_y = float(test_covariate_weight) * rho_map.get(cand_k, 1.0)
        tot_w = cal_w_sum + w_test_y
        cum_mass = 0.0
        q_y = float("inf")
        for score_i, w_i in pairs:
            cum_mass += w_i / tot_w
            if cum_mass >= target_mass - 1e-12:
                q_y = score_i
                break
        if not math.isinf(q_y):
            max_q_used = max(max_q_used, q_y)
        if math.isinf(q_y) or test_scores[cand_k] <= q_y + 1e-12:
            pred_set.append(cand_k)

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


@dataclass(frozen=True)
class OnlineBCACIResult:
    """Step-level diagnostic and summary for Online Bias-Corrected Adaptive Conformal Inference (BC-ACI).

    Implements Gibbs & Candes (2021) Adaptive Conformal Inference augmented with online
    bias-corrected momentum tracking (BC-ACI, 2025-2026) so marginal and rolling coverage
    converge to 1 - alpha_target under abrupt distribution shifts and adversarial streams
    without requiring exchangeability.
    """

    empirical_coverage: float
    rolling_window_worst_coverage: float
    mean_set_size: float
    singleton_rate: float
    escalation_rate: float
    final_alpha_t: float
    alpha_trajectory: tuple[float, ...]
    q_hat_trajectory: tuple[float, ...]
    covered_flags: tuple[bool, ...]


class OnlineBCACITracker:
    """Stateful online Bias-Corrected Adaptive Conformal Inference (BC-ACI) tracker.

    Updates the effective miscoverage level alpha_t via online subgradient descent with
    exponentially smoothed bias-correction momentum:
        err_t = 1{Y_t not in C_{alpha_t}(X_t)}
        bar_d_t = (1 - eta_d) * bar_d_{t-1} + eta_d * (err_t - alpha_target)
        alpha_{t+1} = clip_{[alpha_min, alpha_max]}(
            alpha_t + gamma * (alpha_target - err_t) - beta * bar_d_t
        )
    """

    def __init__(
        self,
        alpha_target: float = 0.10,
        *,
        gamma: float = 0.05,
        beta: float = 0.025,
        eta_d: float = 0.15,
        window_size: int = 100,
        method: str = "raps",
        lam_reg: float = 0.01,
        k_reg: int = 1,
        alpha_min: float = 0.001,
        alpha_max: float = 0.999,
        initial_scores: Sequence[float] | None = None,
    ) -> None:
        if isinstance(alpha_target, bool) or not isinstance(alpha_target, (int, float)) or not math.isfinite(float(alpha_target)) or float(alpha_target) <= 0.0 or float(alpha_target) >= 1.0:
            raise ValueError(f"alpha_target must lie strictly in (0, 1), got {alpha_target!r}")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
            raise ValueError(f"gamma must be finite and > 0, got {gamma!r}")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)) or float(beta) < 0.0:
            raise ValueError(f"beta must be finite and >= 0, got {beta!r}")
        if isinstance(eta_d, bool) or not isinstance(eta_d, (int, float)) or not math.isfinite(float(eta_d)) or float(eta_d) <= 0.0 or float(eta_d) > 1.0:
            raise ValueError(f"eta_d must lie in (0, 1], got {eta_d!r}")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 5:
            raise ValueError(f"window_size must be an integer >= 5, got {window_size!r}")
        clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

        self.alpha_target: float = float(alpha_target)
        self.gamma: float = float(gamma)
        self.beta: float = float(beta)
        self.eta_d: float = float(eta_d)
        self.window_size: int = int(window_size)
        self.method: str = clean_method
        self.lam_reg: float = float(lam_reg)
        self.k_reg: int = int(k_reg)
        self.alpha_min: float = float(alpha_min)
        self.alpha_max: float = float(alpha_max)

        self.alpha_t: float = float(alpha_target)
        self.bias_momentum: float = 0.0
        self._score_window: list[float] = []
        if initial_scores is not None:
            for s in initial_scores:
                val = float(s)
                if not math.isfinite(val) or val < 0.0:
                    raise ValueError(f"initial_scores must be finite and >= 0, got {s!r}")
                self._score_window.append(val)
            if len(self._score_window) > self.window_size:
                self._score_window = self._score_window[-self.window_size :]

    def current_quantile(self) -> float:
        """Compute current empirical nonconformity quantile q_hat_t from the sliding score window."""
        if not self._score_window:
            return 1.0 - self.alpha_t
        eff_alpha = min(self.alpha_max, max(self.alpha_min, self.alpha_t))
        if eff_alpha <= self.alpha_min + 1e-9:
            return float("inf")
        n = len(self._score_window)
        sorted_scores = sorted(self._score_window)
        order_idx = int(math.ceil(float(n + 1) * (1.0 - eff_alpha) - 1e-12))
        if order_idx > n:
            return float("inf")
        pos = max(0, order_idx - 1)
        q_emp = sorted_scores[pos]
        # Blend with worst-case structural envelope (1 - eff_alpha) when alpha_t < alpha_target
        # so abrupt distribution shifts expand C_{alpha_t} without waiting for the entire window to turn over
        if eff_alpha < self.alpha_target:
            urgency = min(1.0, (self.alpha_target - eff_alpha) / max(1e-6, self.alpha_target))
            q_floor = (1.0 - eff_alpha) * urgency
            return max(q_emp, q_floor)
        return q_emp

    def predict_set(
        self,
        probs: Mapping[str, float],
        *,
        abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    ) -> ConformalSetResult:
        """Construct the adaptive prediction set C_{alpha_t}(X_t) prior to observing Y_t."""
        q_hat_t = self.current_quantile()
        return conformal_prediction_set(
            probs=probs,
            q_hat=q_hat_t,
            method=self.method,
            lam_reg=self.lam_reg,
            k_reg=self.k_reg,
            randomized=False,
            abstain_keys=abstain_keys,
        )

    def observe_and_update(
        self,
        probs: Mapping[str, float],
        gold_label: str,
        *,
        abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    ) -> tuple[ConformalSetResult, bool, float]:
        """Predict C_{alpha_t}(X_t), observe gold_label Y_t, append score s_t, and update alpha_{t+1}."""
        set_res = self.predict_set(probs, abstain_keys=abstain_keys)
        gold = str(gold_label)
        ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)
        if gold not in norm_probs:
            raise ValueError(f"gold_label {gold!r} not present in option probabilities")

        item_scores = _compute_all_nonconformity_scores(
            ranked_keys,
            norm_probs,
            self.method,
            self.lam_reg,
            self.k_reg,
            randomized=False,
        )
        score_t = item_scores[gold]
        covered = gold in set_res.prediction_set
        err_t = 0.0 if covered else 1.0

        # Online bias-corrected subgradient update
        self.bias_momentum = (1.0 - self.eta_d) * self.bias_momentum + self.eta_d * (err_t - self.alpha_target)
        raw_next_alpha = (
            self.alpha_t
            + self.gamma * (self.alpha_target - err_t)
            - self.beta * self.bias_momentum
        )
        # Prevent excessive upward drift of alpha_t during easy stationary runs so abrupt shifts
        # do not have to burn through a large positive alpha_t buffer before expanding sets
        max_calm_alpha = min(self.alpha_max, max(self.alpha_target * 1.5, self.alpha_target + 2.0 * self.gamma))
        self.alpha_t = min(max_calm_alpha, max(self.alpha_min, raw_next_alpha))

        self._score_window.append(score_t)
        if len(self._score_window) > self.window_size:
            self._score_window.pop(0)

        return set_res, covered, self.alpha_t


def evaluate_online_bc_aci_stream(
    stream_probs: Sequence[Mapping[str, float]],
    stream_labels: Sequence[str],
    alpha_target: float = 0.10,
    *,
    warmup_probs: Sequence[Mapping[str, float]] | None = None,
    warmup_labels: Sequence[str] | None = None,
    gamma: float = 0.05,
    beta: float = 0.025,
    eta_d: float = 0.15,
    window_size: int = 100,
    rolling_eval_window: int = 50,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    abstain_keys: Collection[str] | None = DEFAULT_CONFORMAL_ABSTAIN_KEYS,
) -> OnlineBCACIResult:
    """Evaluate Online Bias-Corrected Adaptive Conformal Inference (BC-ACI) over a non-stationary stream."""
    if not stream_probs or len(stream_probs) != len(stream_labels):
        raise ValueError("stream_probs and stream_labels must be non-empty sequences of equal length")
    if isinstance(rolling_eval_window, bool) or not isinstance(rolling_eval_window, int) or rolling_eval_window < 1:
        raise ValueError(f"rolling_eval_window must be >= 1, got {rolling_eval_window!r}")
    clean_method = _validate_conformal_hyperparams(method, lam_reg, k_reg)

    init_scores: list[float] = []
    if warmup_probs is not None or warmup_labels is not None:
        if warmup_probs is None or warmup_labels is None or len(warmup_probs) != len(warmup_labels):
            raise ValueError("warmup_probs and warmup_labels must both be provided with equal length")
        for idx, (dist, lbl_raw) in enumerate(zip(warmup_probs, warmup_labels)):
            lbl = str(lbl_raw)
            ranked_keys, norm_probs, _ = _normalize_and_rank_probs(dist)
            if lbl not in norm_probs:
                raise ValueError(f"warmup label {lbl!r} at index {idx} not present in probabilities")
            s_map = _compute_all_nonconformity_scores(
                ranked_keys, norm_probs, clean_method, lam_reg, k_reg, randomized=False
            )
            init_scores.append(s_map[lbl])

    tracker = OnlineBCACITracker(
        alpha_target=alpha_target,
        gamma=gamma,
        beta=beta,
        eta_d=eta_d,
        window_size=window_size,
        method=clean_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        initial_scores=init_scores,
    )

    alpha_traj: list[float] = []
    q_hat_traj: list[float] = []
    covered_list: list[bool] = []
    total_size = 0
    singleton_count = 0
    escalate_count = 0

    for dist, gold_raw in zip(stream_probs, stream_labels):
        alpha_traj.append(tracker.alpha_t)
        set_res, covered, _ = tracker.observe_and_update(
            dist, str(gold_raw), abstain_keys=abstain_keys
        )
        q_hat_traj.append(set_res.q_hat)
        covered_list.append(covered)
        total_size += len(set_res.prediction_set)
        if set_res.singleton:
            singleton_count += 1
        if set_res.should_escalate:
            escalate_count += 1

    n_total = len(stream_probs)
    emp_cov = float(sum(1 for c in covered_list if c)) / float(n_total)

    if n_total <= rolling_eval_window:
        worst_rolling = emp_cov
    else:
        worst_rolling = 1.0
        w = rolling_eval_window
        running_sum = sum(1 for c in covered_list[:w] if c)
        worst_rolling = min(worst_rolling, float(running_sum) / float(w))
        for idx in range(w, n_total):
            running_sum += (1 if covered_list[idx] else 0) - (1 if covered_list[idx - w] else 0)
            worst_rolling = min(worst_rolling, float(running_sum) / float(w))

    return OnlineBCACIResult(
        empirical_coverage=emp_cov,
        rolling_window_worst_coverage=worst_rolling,
        mean_set_size=float(total_size) / float(n_total),
        singleton_rate=float(singleton_count) / float(n_total),
        escalation_rate=float(escalate_count) / float(n_total),
        final_alpha_t=tracker.alpha_t,
        alpha_trajectory=tuple(alpha_traj),
        q_hat_trajectory=tuple(q_hat_traj),
        covered_flags=tuple(covered_list),
    )


@dataclass(frozen=True)
class TwoStageCascadeRiskBound:
    """Exact selective risk and hybrid conformal miscoverage bound for a hint-conditioned cascade (Proposition 1)."""

    escalation_rate: float
    fast_exit_selective_risk: float
    escalated_stage1_error_rate: float
    stage2_false_flip_rate: float
    stage2_unrecovered_error_rate: float
    stage1_marginal_risk: float
    cascade_marginal_risk: float
    strictly_improves_over_stage1: bool
    hybrid_conformal_miscoverage_ub: float


def evaluate_two_stage_cascade_risk_bound(
    escalation_rate: float,
    fast_exit_selective_risk: float,
    escalated_stage1_error_rate: float,
    stage2_false_flip_rate: float,
    stage2_unrecovered_error_rate: float,
    *,
    alpha_stage1_fast_exit: float = 0.05,
    alpha_stage2_conformal: float = 0.10,
) -> TwoStageCascadeRiskBound:
    """Evaluate exact two-stage hint-conditioned cascade selective risk and miscoverage bound (Proposition 1).

    Args:
      escalation_rate: Probability pi_esc = P(G(X) = 1) in [0, 1].
      fast_exit_selective_risk: Stage-1 error rate eps_fast = P(Y_hat_1 != Y | G(X) = 0) in [0, 1].
      escalated_stage1_error_rate: Stage-1 error rate on escalated slice e_1^{esc} = P(Y_hat_1 != Y | G(X) = 1).
      stage2_false_flip_rate: Stage-2 flip rate eps_{2,+} = P(Y_hat_2 != Y | G(X) = 1, Y_hat_1 = Y).
      stage2_unrecovered_error_rate: Stage-2 unrecovered error eps_{2,-} = P(Y_hat_2 != Y | G(X) = 1, Y_hat_1 != Y).
      alpha_stage1_fast_exit: Target selective risk bound alpha_1 on Stage-1 fast-exits.
      alpha_stage2_conformal: Target conformal miscoverage level alpha_2 on Stage-2 escalated sets.

    Returns:
      TwoStageCascadeRiskBound containing exact marginal risk and hybrid miscoverage upper bound.

    Raises:
      ValueError: If any probability argument lies outside [0, 1].
    """
    for name, val in (
        ("escalation_rate", escalation_rate),
        ("fast_exit_selective_risk", fast_exit_selective_risk),
        ("escalated_stage1_error_rate", escalated_stage1_error_rate),
        ("stage2_false_flip_rate", stage2_false_flip_rate),
        ("stage2_unrecovered_error_rate", stage2_unrecovered_error_rate),
        ("alpha_stage1_fast_exit", alpha_stage1_fast_exit),
        ("alpha_stage2_conformal", alpha_stage2_conformal),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite number in [0, 1], got {val!r}")
        fval = float(val)
        if fval < 0.0 or fval > 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {val!r}")

    pi_esc = float(escalation_rate)
    eps_fast = float(fast_exit_selective_risk)
    e1_esc = float(escalated_stage1_error_rate)
    eps_2_pos = float(stage2_false_flip_rate)
    eps_2_neg = float(stage2_unrecovered_error_rate)
    a1 = float(alpha_stage1_fast_exit)
    a2 = float(alpha_stage2_conformal)

    stage1_risk = (1.0 - pi_esc) * eps_fast + pi_esc * e1_esc
    escalated_stage2_risk = (1.0 - e1_esc) * eps_2_pos + e1_esc * eps_2_neg
    cascade_risk = (1.0 - pi_esc) * eps_fast + pi_esc * escalated_stage2_risk
    strictly_improves = cascade_risk < stage1_risk - 1e-12
    hybrid_ub = (1.0 - pi_esc) * a1 + pi_esc * a2

    return TwoStageCascadeRiskBound(
        escalation_rate=pi_esc,
        fast_exit_selective_risk=eps_fast,
        escalated_stage1_error_rate=e1_esc,
        stage2_false_flip_rate=eps_2_pos,
        stage2_unrecovered_error_rate=eps_2_neg,
        stage1_marginal_risk=stage1_risk,
        cascade_marginal_risk=cascade_risk,
        strictly_improves_over_stage1=strictly_improves,
        hybrid_conformal_miscoverage_ub=hybrid_ub,
    )


_VALID_CRC_LOSS_MODES: frozenset[str] = frozenset(
    {
        "omniscience_abstention",
        "asymmetric_false_allow",
    }
)


@dataclass(frozen=True)
class OmniscienceActionDecision:
    """Exact Bayes-optimal selective abstention decision for Omniscience scoring."""

    selected_option: str | None
    action: str
    abstained: bool
    max_probability: float
    optimal_threshold: float
    expected_commit_score: float
    expected_optimal_score: float


@dataclass(frozen=True)
class ConformalRiskCalibrationResult:
    """Calibrated Angelopoulos et al. (2024) Conformal Risk Control (CRC) result."""

    lambda_hat: float
    alpha: float
    loss_bound_b: float
    n_calib: int
    empirical_calib_risk: float
    crc_upper_bound: float
    calib_abstention_rate: float
    satisfied_bound: bool
    loss_mode: str = "omniscience_abstention"
    allow_key: str = "safe"


@dataclass(frozen=True)
class ConformalRiskDecisionResult:
    """Test-time selective or safety gate decision under Conformal Risk Control."""

    selected_option: str | None
    action: str
    abstained: bool
    should_escalate: bool
    decision_confidence: float
    lambda_hat: float
    loss_mode: str


@dataclass(frozen=True)
class BayesValueOfInformationResult:
    """Closed-form Bayesian Value-of-Information (VoI) cascade stopping result."""

    should_escalate: bool
    expected_net_voi: float
    risk_stage1: float
    risk_stage2: float
    total_stage2_cost: float
    stage2_rpc_cost_usd: float
    latency_cost_usd: float
    stage1_best_action: str
    stage1_max_prob: float
    expected_stage2_accuracy: float


def _validate_strict_probs(
    probs: Mapping[str, float],
) -> tuple[list[str], dict[str, float]]:
    """Validate string-keyed non-negative finite probability mapping and rank keys."""
    if not isinstance(probs, Mapping) or not probs:
        raise ValueError("probability mapping must be non-empty")
    for raw_key in probs:
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"option keys must be non-empty strings, got {raw_key!r}")
    ranked_keys, norm_probs, _ = _normalize_and_rank_probs(probs)
    return ranked_keys, norm_probs


def _validate_unit_threshold(value: float, param_name: str) -> float:
    """Validate finite scalar threshold in [0, 1]."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{param_name} must be a finite non-boolean number, got {value!r}")
    float_val = float(value)
    if float_val < 0.0 or float_val > 1.0:
        raise ValueError(f"{param_name} must lie in [0, 1], got {value!r}")
    return float_val


def _validate_positive_bound(value: float, param_name: str) -> float:
    """Validate finite strictly positive scalar (> 0)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{param_name} must be finite and > 0, got {value!r}")
    return float(value)


def _validate_non_negative_scalar(value: float, param_name: str) -> float:
    """Validate finite non-negative scalar (>= 0)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{param_name} must be finite and >= 0, got {value!r}")
    return float(value)


def select_omniscience_action(
    probs: Mapping[str, float],
    penalty_wrong: float = 1.0,
    reward_correct: float = 1.0,
    *,
    abstain_action: str = "abstain",
) -> OmniscienceActionDecision:
    """Select the Bayes-optimal commit-or-abstain action under Omniscience scoring.

    Maximizes expected reward `p_max * reward_correct - (1 - p_max) * penalty_wrong`
    against `0.0` for abstention, abstaining iff
    `max_k p(k) < penalty_wrong / (reward_correct + penalty_wrong)` (`0.50` for
    symmetric `+1 / 0 / -1` AA-Omniscience scoring).

    Args:
      probs: Non-empty mapping of option keys to calibrated probabilities.
      penalty_wrong: Non-negative penalty magnitude incurred on an incorrect commit.
      reward_correct: Positive reward earned on a correct commit.
      abstain_action: Action token returned when abstaining.

    Returns:
      OmniscienceActionDecision containing the optimal action and expected score.

    Raises:
      ValueError: If `probs` is invalid or reward/penalty weights are invalid.
    """
    ranked_keys, norm_probs = _validate_strict_probs(probs)
    penalty_float = _validate_non_negative_scalar(penalty_wrong, "penalty_wrong")
    reward_float = _validate_positive_bound(reward_correct, "reward_correct")
    if not isinstance(abstain_action, str) or not abstain_action:
        raise ValueError("abstain_action must be a non-empty string")

    top_option = ranked_keys[0]
    max_prob = norm_probs[top_option]
    optimal_threshold = penalty_float / (reward_float + penalty_float)
    expected_commit = (max_prob * reward_float) - ((1.0 - max_prob) * penalty_float)
    abstained = max_prob < optimal_threshold
    return OmniscienceActionDecision(
        selected_option=None if abstained else top_option,
        action=abstain_action if abstained else top_option,
        abstained=abstained,
        max_probability=max_prob,
        optimal_threshold=optimal_threshold,
        expected_commit_score=expected_commit,
        expected_optimal_score=0.0 if abstained else expected_commit,
    )


def omniscience_abstention_loss(
    probs: Mapping[str, float],
    gold_label: str,
    threshold_lambda: float,
    *,
    hallucination_penalty: float = 1.0,
) -> float:
    """Compute monotone non-increasing selective hallucination loss in `[0, B]`.

    Abstains (loss `0.0`) when `max_k p(k) < threshold_lambda` or at full-abstention
    boundary `threshold_lambda == 1.0`. Commits to `argmax_k p(k)` otherwise, incurring
    `0.0` when `argmax_k p(k) == gold_label` and `hallucination_penalty` (`B`) when wrong.

    Args:
      probs: Non-empty mapping of option keys to model probabilities.
      gold_label: Ground-truth option key present in `probs`.
      threshold_lambda: Confidence threshold `lambda` in `[0, 1]`.
      hallucination_penalty: Upper loss bound `B > 0` for an incorrect committed answer.

    Returns:
      Bounded monotone non-increasing loss value in `[0, hallucination_penalty]`.

    Raises:
      ValueError: If `probs`, `gold_label`, `threshold_lambda`, or penalty are invalid.
    """
    ranked_keys, norm_probs = _validate_strict_probs(probs)
    if not isinstance(gold_label, str) or gold_label not in norm_probs:
        raise ValueError(f"gold_label {gold_label!r} not present in option probabilities")
    lam_float = _validate_unit_threshold(threshold_lambda, "threshold_lambda")
    penalty_b = _validate_positive_bound(hallucination_penalty, "hallucination_penalty")

    top_option = ranked_keys[0]
    max_prob = norm_probs[top_option]
    if lam_float >= 1.0 or max_prob < lam_float:
        return 0.0
    return 0.0 if top_option == gold_label else penalty_b


def asymmetric_false_allow_loss(
    probs: Mapping[str, float],
    gold_label: str,
    threshold_lambda: float,
    *,
    false_allow_cost: float = 1.0,
    allow_key: str = "safe",
) -> float:
    """Compute monotone non-increasing safety false-allow loss in `[0, B]`.

    Abstains/escalates (loss `0.0`) when `p(allow_key) < threshold_lambda` or at
    `threshold_lambda == 1.0`. Auto-allows otherwise, incurring `0.0` when
    `gold_label == allow_key` and `false_allow_cost` (`B`) on an unsafe false-allow.

    Args:
      probs: Non-empty mapping of safety/disposition keys to model probabilities.
      gold_label: Ground-truth disposition key present in `probs`.
      threshold_lambda: Minimum `p(allow_key)` threshold `lambda` in `[0, 1]`.
      false_allow_cost: Loss `B > 0` incurred when allowing an unsafe example.
      allow_key: Option key corresponding to safe/benign allow disposition.

    Returns:
      Bounded monotone non-increasing safety loss in `[0, false_allow_cost]`.

    Raises:
      ValueError: If `probs`, `allow_key`, `gold_label`, `threshold_lambda`, or cost are invalid.
    """
    _, norm_probs = _validate_strict_probs(probs)
    if not isinstance(allow_key, str) or allow_key not in norm_probs:
        raise ValueError(f"allow_key {allow_key!r} not present in option probabilities")
    if not isinstance(gold_label, str) or gold_label not in norm_probs:
        raise ValueError(f"gold_label {gold_label!r} not present in option probabilities")
    lam_float = _validate_unit_threshold(threshold_lambda, "threshold_lambda")
    cost_b = _validate_positive_bound(false_allow_cost, "false_allow_cost")

    allow_prob = norm_probs[allow_key]
    if lam_float >= 1.0 or allow_prob < lam_float:
        return 0.0
    return 0.0 if gold_label == allow_key else cost_b


def calibrate_conformal_risk_control(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    alpha: float = 0.10,
    *,
    loss_bound_b: float = 1.0,
    loss_fn: Callable[[Mapping[str, float], str, float], float] | None = None,
    loss_mode: str = "omniscience_abstention",
    allow_key: str = "safe",
    lambda_grid: Sequence[float] | None = None,
    num_grid_points: int = 1001,
) -> ConformalRiskCalibrationResult:
    """Calibrate Angelopoulos et al. (2024) Conformal Risk Control (CRC).

    Selects the smallest threshold `lambda_hat` in `Lambda subseteq [0, 1]` satisfying:
        `(n / (n + 1)) * R_hat_n(lambda) + B / (n + 1) <= alpha`
    for bounded monotone non-increasing losses `L_i(lambda) in [0, B]`, guaranteeing
    `E[L_{n+1}(lambda_hat)] <= alpha` under exchangeability. Fails closed to full
    abstention (`lambda_hat = 1.0`, `satisfied_bound = False`) whenever
    `B / (n + 1) > alpha` or no grid threshold satisfies the risk bound.

    Args:
      calib_probs: Non-empty sequence of calibration probability mappings.
      calib_labels: Ground-truth calibration labels matching `calib_probs` length.
      alpha: Target maximum expected risk `> 0`.
      loss_bound_b: Upper bound `B > 0` on per-example loss `L_i(lambda)`.
      loss_fn: Optional custom monotone non-increasing callable `(probs, label, lam) -> loss`.
      loss_mode: Built-in loss selector (`"omniscience_abstention"` or `"asymmetric_false_allow"`)
        used when `loss_fn is None`.
      allow_key: Safe disposition key used when `loss_mode == "asymmetric_false_allow"`.
      lambda_grid: Optional explicit sequence of candidate thresholds in `[0, 1]`.
      num_grid_points: Uniform grid resolution (`>= 2`) over `[0, 1]` when `lambda_grid is None`.

    Returns:
      ConformalRiskCalibrationResult with `lambda_hat` and finite-sample CRC diagnostics.

    Raises:
      ValueError: If inputs are empty, mismatched, out-of-range, or violate monotonicity.
    """
    alpha_float = _validate_positive_bound(alpha, "alpha")
    b_float = _validate_positive_bound(loss_bound_b, "loss_bound_b")
    if (
        isinstance(calib_probs, (str, bytes))
        or isinstance(calib_labels, (str, bytes))
        or not calib_probs
        or len(calib_probs) != len(calib_labels)
    ):
        raise ValueError("calib_probs and calib_labels must be non-empty sequences of equal length")
    if not isinstance(allow_key, str) or not allow_key:
        raise ValueError("allow_key must be a non-empty string")

    if lambda_grid is not None:
        if isinstance(lambda_grid, (str, bytes)) or not lambda_grid:
            raise ValueError("lambda_grid must be non-empty when provided")
        sorted_grid = sorted(
            {_validate_unit_threshold(cand_lam, "lambda_grid") for cand_lam in lambda_grid}
        )
    else:
        if isinstance(num_grid_points, bool) or not isinstance(num_grid_points, int) or num_grid_points < 2:
            raise ValueError(f"num_grid_points must be an integer >= 2, got {num_grid_points!r}")
        grid_denom = float(num_grid_points - 1)
        sorted_grid = [float(step_idx) / grid_denom for step_idx in range(num_grid_points)]

    n_calib = len(calib_probs)

    if loss_fn is not None:
        if not callable(loss_fn):
            raise ValueError("loss_fn must be callable when provided")
        effective_mode = "custom"
        normalized_rows: list[tuple[dict[str, float], str, float]] = []
        for idx, (dist, raw_gold) in enumerate(zip(calib_probs, calib_labels)):
            ranked_keys, norm_probs = _validate_strict_probs(dist)
            if not isinstance(raw_gold, str) or raw_gold not in norm_probs:
                raise ValueError(
                    f"calibration label {raw_gold!r} at index {idx} not present in option probabilities"
                )
            normalized_rows.append((norm_probs, raw_gold, norm_probs[ranked_keys[0]]))

        grid_losses: list[float] = [0.0] * len(sorted_grid)
        grid_abstains: list[float] = [0.0] * len(sorted_grid)
        for item_idx, (norm_probs, gold_str, max_prob) in enumerate(normalized_rows):
            prev_loss = float("inf")
            for g_idx, lam_val in enumerate(sorted_grid):
                raw_loss = loss_fn(norm_probs, gold_str, lam_val)
                if (
                    isinstance(raw_loss, bool)
                    or not isinstance(raw_loss, (int, float))
                    or not math.isfinite(float(raw_loss))
                ):
                    raise ValueError(
                        f"loss_fn returned non-finite value {raw_loss!r} at item {item_idx}, lambda={lam_val}"
                    )
                loss_val = float(raw_loss)
                if loss_val < -1e-12 or loss_val > b_float + 1e-12:
                    raise ValueError(
                        f"loss_fn value {loss_val} outside [0, {b_float}] at item {item_idx}, lambda={lam_val}"
                    )
                if loss_val > prev_loss + 1e-12:
                    raise ValueError(
                        f"loss_fn is not monotone non-increasing at item {item_idx}: "
                        f"L({lam_val})={loss_val} > previous {prev_loss}"
                    )
                clamped_loss = min(b_float, max(0.0, loss_val))
                prev_loss = clamped_loss
                grid_losses[g_idx] += clamped_loss
                if lam_val >= 1.0 or max_prob < lam_val:
                    grid_abstains[g_idx] += 1.0

        if b_float / float(n_calib + 1) > alpha_float:
            fallback_emp_risk = 0.0
            if sorted_grid[-1] >= 1.0:
                fallback_emp_risk = grid_losses[-1] / float(n_calib)
            else:
                loss_at_one = sum(
                    min(b_float, max(0.0, float(loss_fn(np_map, g_lbl, 1.0))))
                    for np_map, g_lbl, _ in normalized_rows
                )
                fallback_emp_risk = loss_at_one / float(n_calib)
            fallback_crc = (float(n_calib) * fallback_emp_risk + b_float) / float(n_calib + 1)
            return ConformalRiskCalibrationResult(
                lambda_hat=1.0,
                alpha=alpha_float,
                loss_bound_b=b_float,
                n_calib=n_calib,
                empirical_calib_risk=fallback_emp_risk,
                crc_upper_bound=fallback_crc,
                calib_abstention_rate=1.0,
                satisfied_bound=False,
                loss_mode=effective_mode,
                allow_key=allow_key,
            )

        for g_idx, lam_val in enumerate(sorted_grid):
            emp_risk = grid_losses[g_idx] / float(n_calib)
            crc_bound = (grid_losses[g_idx] + b_float) / float(n_calib + 1)
            if crc_bound <= alpha_float + 1e-15:
                return ConformalRiskCalibrationResult(
                    lambda_hat=lam_val,
                    alpha=alpha_float,
                    loss_bound_b=b_float,
                    n_calib=n_calib,
                    empirical_calib_risk=emp_risk,
                    crc_upper_bound=crc_bound,
                    calib_abstention_rate=grid_abstains[g_idx] / float(n_calib),
                    satisfied_bound=True,
                    loss_mode=effective_mode,
                    allow_key=allow_key,
                )

        loss_at_one = sum(
            min(b_float, max(0.0, float(loss_fn(np_map, g_lbl, 1.0))))
            for np_map, g_lbl, _ in normalized_rows
        )
        emp_at_one = loss_at_one / float(n_calib)
        crc_at_one = (loss_at_one + b_float) / float(n_calib + 1)
        return ConformalRiskCalibrationResult(
            lambda_hat=1.0,
            alpha=alpha_float,
            loss_bound_b=b_float,
            n_calib=n_calib,
            empirical_calib_risk=emp_at_one,
            crc_upper_bound=crc_at_one,
            calib_abstention_rate=1.0,
            satisfied_bound=False,
            loss_mode=effective_mode,
            allow_key=allow_key,
        )

    if not isinstance(loss_mode, str) or loss_mode.strip().lower() not in _VALID_CRC_LOSS_MODES:
        raise ValueError(
            f"unsupported loss_mode {loss_mode!r}; expected one of {sorted(_VALID_CRC_LOSS_MODES)}"
        )
    effective_mode = loss_mode.strip().lower()

    confidence_loss_pairs: list[tuple[float, float]] = []
    for idx, (dist, raw_gold) in enumerate(zip(calib_probs, calib_labels)):
        ranked_keys, norm_probs = _validate_strict_probs(dist)
        if not isinstance(raw_gold, str) or raw_gold not in norm_probs:
            raise ValueError(
                f"calibration label {raw_gold!r} at index {idx} not present in option probabilities"
            )
        if effective_mode == "asymmetric_false_allow":
            if allow_key not in norm_probs:
                raise ValueError(
                    f"allow_key {allow_key!r} missing from calibration probabilities at index {idx}"
                )
            decision_conf = norm_probs[allow_key]
            commit_loss = 0.0 if raw_gold == allow_key else b_float
        else:
            top_key = ranked_keys[0]
            decision_conf = norm_probs[top_key]
            commit_loss = 0.0 if top_key == raw_gold else b_float
        confidence_loss_pairs.append((decision_conf, commit_loss))

    if b_float / float(n_calib + 1) > alpha_float:
        return ConformalRiskCalibrationResult(
            lambda_hat=1.0,
            alpha=alpha_float,
            loss_bound_b=b_float,
            n_calib=n_calib,
            empirical_calib_risk=0.0,
            crc_upper_bound=b_float / float(n_calib + 1),
            calib_abstention_rate=1.0,
            satisfied_bound=False,
            loss_mode=effective_mode,
            allow_key=allow_key,
        )

    confidence_loss_pairs.sort(key=lambda pair: pair[0])
    suffix_losses: list[float] = [0.0] * (n_calib + 1)
    for rev_idx in range(n_calib - 1, -1, -1):
        suffix_losses[rev_idx] = suffix_losses[rev_idx + 1] + confidence_loss_pairs[rev_idx][1]

    pointer = 0
    for lam_val in sorted_grid:
        if lam_val >= 1.0:
            total_loss = 0.0
            abstained_count = n_calib
        else:
            while pointer < n_calib and confidence_loss_pairs[pointer][0] < lam_val:
                pointer += 1
            total_loss = suffix_losses[pointer]
            abstained_count = pointer

        emp_risk = total_loss / float(n_calib)
        crc_bound = (total_loss + b_float) / float(n_calib + 1)
        if crc_bound <= alpha_float + 1e-15:
            return ConformalRiskCalibrationResult(
                lambda_hat=lam_val,
                alpha=alpha_float,
                loss_bound_b=b_float,
                n_calib=n_calib,
                empirical_calib_risk=emp_risk,
                crc_upper_bound=crc_bound,
                calib_abstention_rate=float(abstained_count) / float(n_calib),
                satisfied_bound=True,
                loss_mode=effective_mode,
                allow_key=allow_key,
            )

    return ConformalRiskCalibrationResult(
        lambda_hat=1.0,
        alpha=alpha_float,
        loss_bound_b=b_float,
        n_calib=n_calib,
        empirical_calib_risk=0.0,
        crc_upper_bound=b_float / float(n_calib + 1),
        calib_abstention_rate=1.0,
        satisfied_bound=False,
        loss_mode=effective_mode,
        allow_key=allow_key,
    )


def evaluate_conformal_risk_decision(
    probs: Mapping[str, float],
    calibration: ConformalRiskCalibrationResult | float,
    *,
    loss_mode: str | None = None,
    allow_key: str | None = None,
    abstain_action: str = "abstain",
) -> ConformalRiskDecisionResult:
    """Evaluate a calibrated Conformal Risk Control threshold on test probabilities.

    Args:
      probs: Non-empty mapping of option keys to test-time probabilities.
      calibration: Either a `ConformalRiskCalibrationResult` or scalar `lambda_hat` in `[0, 1]`.
      loss_mode: Optional mode override (`"omniscience_abstention"`, `"asymmetric_false_allow"`,
        or `"custom"`).
      allow_key: Optional safe-disposition key override for `"asymmetric_false_allow"`.
      abstain_action: Action label returned when abstaining/escalating.

    Returns:
      ConformalRiskDecisionResult indicating commit vs. abstain/escalate.

    Raises:
      ValueError: If `probs`, `calibration`, `loss_mode`, or `allow_key` are invalid.
    """
    ranked_keys, norm_probs = _validate_strict_probs(probs)
    if not isinstance(abstain_action, str) or not abstain_action:
        raise ValueError("abstain_action must be a non-empty string")

    if isinstance(calibration, ConformalRiskCalibrationResult):
        lam_float = _validate_unit_threshold(calibration.lambda_hat, "calibration.lambda_hat")
        raw_mode = loss_mode if loss_mode is not None else calibration.loss_mode
        effective_allow_key = allow_key if allow_key is not None else calibration.allow_key
    else:
        lam_float = _validate_unit_threshold(calibration, "calibration")
        raw_mode = loss_mode if loss_mode is not None else "omniscience_abstention"
        effective_allow_key = allow_key if allow_key is not None else "safe"

    if not isinstance(raw_mode, str):
        raise ValueError(f"loss_mode must be a string, got {raw_mode!r}")
    clean_mode = raw_mode.strip().lower()
    if clean_mode not in (_VALID_CRC_LOSS_MODES | {"custom"}):
        raise ValueError(f"unsupported loss_mode {raw_mode!r}")

    if clean_mode == "asymmetric_false_allow":
        if not isinstance(effective_allow_key, str) or effective_allow_key not in norm_probs:
            raise ValueError(
                f"allow_key {effective_allow_key!r} not present in option probabilities"
            )
        target_option = effective_allow_key
        decision_conf = norm_probs[effective_allow_key]
    else:
        target_option = ranked_keys[0]
        decision_conf = norm_probs[target_option]

    abstained = lam_float >= 1.0 or decision_conf < lam_float
    return ConformalRiskDecisionResult(
        selected_option=None if abstained else target_option,
        action=abstain_action if abstained else target_option,
        abstained=abstained,
        should_escalate=abstained,
        decision_confidence=decision_conf,
        lambda_hat=lam_float,
        loss_mode=clean_mode,
    )


def compute_bayes_value_of_information(
    stage1_probs: Mapping[str, float],
    error_cost: float | Mapping[str, float] | Mapping[str, Mapping[str, float]] = 1.0,
    expected_stage2_accuracy: float = 0.95,
    stage2_rpc_cost_usd: float = 0.005,
    latency_cost_usd: float = 0.0,
    *,
    class_error_costs: Mapping[str, float] | None = None,
    loss_matrix: Mapping[str, Mapping[str, float]] | None = None,
) -> BayesValueOfInformationResult:
    """Compute the closed-form Bayesian Value-of-Information (VoI) cascade criterion.

    Compares the Bayes expected error risk of immediate Stage-1 fast-exit (`risk_stage1`)
    against the expected residual error risk plus RPC and latency costs of Stage-2
    verifier escalation (`risk_stage2 + stage2_rpc_cost_usd + latency_cost_usd`),
    escalating iff `expected_net_voi > 0.0`.

    Args:
      stage1_probs: Calibrated Stage-1 posterior mapping over candidate options.
      error_cost: Symmetric scalar error cost in USD (`>= 0`), per-class asymmetric
        misclassification cost mapping `{true_class: cost_usd}`, or full pairwise loss
        matrix `{true_class: {pred_class: cost_usd}}`.
      expected_stage2_accuracy: Expected Stage-2 verifier accuracy on the shortlist in `[0, 1]`.
      stage2_rpc_cost_usd: Incremental Stage-2 RPC token cost in USD (`>= 0`).
      latency_cost_usd: Incremental Stage-2 SLA/latency penalty in USD (`>= 0`).
      class_error_costs: Optional keyword alias for per-class asymmetric error costs.
      loss_matrix: Optional keyword alias for a full pairwise loss matrix.

    Returns:
      BayesValueOfInformationResult with `expected_net_voi` and `should_escalate`.

    Raises:
      ValueError: If probabilities, costs, accuracy, or option key sets are invalid.
    """
    ranked_keys, norm_probs = _validate_strict_probs(stage1_probs)
    expected_key_set = set(ranked_keys)
    acc2_float = _validate_unit_threshold(expected_stage2_accuracy, "expected_stage2_accuracy")
    rpc_float = _validate_non_negative_scalar(stage2_rpc_cost_usd, "stage2_rpc_cost_usd")
    latency_float = _validate_non_negative_scalar(latency_cost_usd, "latency_cost_usd")

    active_matrix: Mapping[str, Mapping[str, float]] | None = None
    active_class_costs: Mapping[str, float] | None = None
    scalar_error_cost: float | None = None

    if isinstance(error_cost, Mapping):
        if class_error_costs is not None or loss_matrix is not None:
            raise ValueError(
                "cannot specify mapping error_cost alongside class_error_costs or loss_matrix"
            )
        if not error_cost:
            raise ValueError("error_cost mapping must be non-empty")
        first_value = next(iter(error_cost.values()))
        if isinstance(first_value, Mapping):
            active_matrix = error_cost  # type: ignore[assignment]
        else:
            active_class_costs = error_cost  # type: ignore[assignment]
    else:
        scalar_val = _validate_non_negative_scalar(error_cost, "error_cost")
        if class_error_costs is not None and loss_matrix is not None:
            raise ValueError("cannot specify both class_error_costs and loss_matrix")
        if loss_matrix is not None:
            active_matrix = loss_matrix
        elif class_error_costs is not None:
            active_class_costs = class_error_costs
        else:
            scalar_error_cost = scalar_val

    stage1_max_prob = norm_probs[ranked_keys[0]]

    if active_matrix is not None:
        if not isinstance(active_matrix, Mapping) or set(active_matrix.keys()) != expected_key_set:
            raise ValueError("loss_matrix outer keys must match stage1_probs option keys")
        clean_matrix: dict[str, dict[str, float]] = {}
        for true_key in ranked_keys:
            row_map = active_matrix[true_key]
            if not isinstance(row_map, Mapping) or set(row_map.keys()) != expected_key_set:
                raise ValueError(
                    f"loss_matrix[{true_key!r}] keys must match stage1_probs option keys"
                )
            clean_matrix[true_key] = {
                pred_key: _validate_non_negative_scalar(
                    row_map[pred_key], f"loss_matrix[{true_key!r}][{pred_key!r}]"
                )
                for pred_key in ranked_keys
            }

        action_risks = {
            cand_action: sum(
                norm_probs[true_key] * clean_matrix[true_key][cand_action]
                for true_key in ranked_keys
            )
            for cand_action in ranked_keys
        }
        stage1_best_action = min(
            ranked_keys,
            key=lambda act_key: (action_risks[act_key], -norm_probs[act_key], act_key),
        )
        risk_stage1 = action_risks[stage1_best_action]

        num_classes = len(ranked_keys)
        risk_stage2 = 0.0
        for true_key in ranked_keys:
            diag_cost = clean_matrix[true_key][true_key]
            if num_classes > 1:
                off_diag_mean = sum(
                    clean_matrix[true_key][pred_key]
                    for pred_key in ranked_keys
                    if pred_key != true_key
                ) / float(num_classes - 1)
            else:
                off_diag_mean = diag_cost
            cond_loss = (acc2_float * diag_cost) + ((1.0 - acc2_float) * off_diag_mean)
            risk_stage2 += norm_probs[true_key] * cond_loss

    elif active_class_costs is not None:
        if (
            not isinstance(active_class_costs, Mapping)
            or set(active_class_costs.keys()) != expected_key_set
        ):
            raise ValueError("class_error_costs keys must match stage1_probs option keys")
        clean_costs = {
            key: _validate_non_negative_scalar(active_class_costs[key], f"class_error_costs[{key!r}]")
            for key in ranked_keys
        }
        action_risks = {
            cand_action: sum(
                norm_probs[true_key] * clean_costs[true_key]
                for true_key in ranked_keys
                if true_key != cand_action
            )
            for cand_action in ranked_keys
        }
        stage1_best_action = min(
            ranked_keys,
            key=lambda act_key: (action_risks[act_key], -norm_probs[act_key], act_key),
        )
        risk_stage1 = action_risks[stage1_best_action]
        weighted_cost = sum(norm_probs[key] * clean_costs[key] for key in ranked_keys)
        risk_stage2 = (1.0 - acc2_float) * weighted_cost

    else:
        assert scalar_error_cost is not None
        stage1_best_action = ranked_keys[0]
        risk_stage1 = (1.0 - stage1_max_prob) * scalar_error_cost
        risk_stage2 = (1.0 - acc2_float) * scalar_error_cost

    total_stage2_cost = risk_stage2 + rpc_float + latency_float
    expected_net_voi = risk_stage1 - total_stage2_cost
    return BayesValueOfInformationResult(
        should_escalate=expected_net_voi > 0.0,
        expected_net_voi=expected_net_voi,
        risk_stage1=risk_stage1,
        risk_stage2=risk_stage2,
        total_stage2_cost=total_stage2_cost,
        stage2_rpc_cost_usd=rpc_float,
        latency_cost_usd=latency_float,
        stage1_best_action=stage1_best_action,
        stage1_max_prob=stage1_max_prob,
        expected_stage2_accuracy=acc2_float,
    )




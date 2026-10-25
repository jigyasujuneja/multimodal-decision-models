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

"""Pre-Gated Prior Fusion, Evidence Slot Injection, and Layer-5 Post-Generation Guards.

Implements:
  1. Cardinality-normalized Shannon hesitation H_raw = H(p_raw) / ln(K) and confidence C_2 = 1 - H_raw.
  2. Pre-gated Bayesian Product-of-Experts (PoE) prior fusion: evaluates H_raw strictly BEFORE
     fusing retrieval or base-rate prior logits so strong priors never mask out-of-distribution hesitation.
     Supports gate_mode in {"always", "confident", "hesitant", "proportional"}.
  3. Stage 2A Adaptive Fast-Exit Gate (0-VLM resolution when P_1 - P_2 >= 0.14, Z >= 1.65, and no sister
     pack-size collision).
  4. Co-denoised / Causal Evidence-First slot injection.
  5. Layer-5 deterministic post-generation guards:
     - Precedence 1: Pack-Size Sister Switch
     - Precedence 2: 0.045 Visual Energy Veto
     - Precedence 3: none_of_the_above Competitor Similarity Guard
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    MAX_SLOTS_PER_PASS,
    DecisionQuestion,
)

DEFAULT_EVIDENCE_CRITERIA: dict[str, str] = {
    "neck_band_and_cap": "Distinctive cap color, neck ring, or seal geometry",
    "front_brand_crest": "Primary brand logo, crest, or typography layout",
    "net_weight_and_multipack_badge": "Printed net weight/volume callout or multipack badge",
    "variant_color_strip": "Flavor, scent, or formulation color band",
    "occluded_or_unreadable": "Decisive visual region is blurred, occluded, or absent",
}


def normalized_entropy(probs: Mapping[str, float]) -> float:
    """Compute cardinality-normalized Shannon entropy H(p) / ln(K) in [0, 1]."""
    k = len(probs)
    if k <= 1:
        return 0.0
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 0.0:
        return 1.0
    h = 0.0
    for v in probs.values():
        p = max(0.0, float(v)) / total
        if p > 1e-12:
            h -= p * math.log(p)
    return min(1.0, max(0.0, h / math.log(float(k))))


def pre_gated_bayesian_poe_fusion(
    raw_probs: Mapping[str, float],
    prior_logits: Mapping[str, float] | None = None,
    prior_weight: float = 0.0,
    temperature: float = 1.0,
    hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD,
    gate_mode: str = "always",
) -> tuple[dict[str, float], float, bool]:
    """Fuse model probabilities with prior logits AFTER computing raw hesitation H_raw.

    Args:
      raw_probs: Uncalibrated model probabilities keyed by option name.
      prior_logits: Optional log-prior scores keyed by option name.
      prior_weight: Log-linear weight alpha >= 0 applied to prior_logits.
      temperature: Temperature scaling T > 0 applied to raw log-probabilities.
      hesitation_threshold: Raw normalized entropy threshold tau in (0, 1].
      gate_mode:
        - "always": always fuse prior_logits while preserving pre-fusion h_raw.
        - "confident": fuse only when h_raw < hesitation_threshold.
        - "hesitant": fuse only when h_raw >= hesitation_threshold (tie-breaking on uncertain items).
        - "proportional": scale effective weight by prior_weight * h_raw.

    Returns:
      (fused_probabilities, h_raw, should_escalate)
    """
    if not raw_probs:
        return {}, 1.0, True
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if prior_weight < 0.0:
        raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
    if gate_mode not in ("always", "confident", "hesitant", "proportional"):
        raise ValueError(f"unsupported gate_mode: {gate_mode!r}")

    for k, v in raw_probs.items():
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(f"raw_probs[{k!r}] must be finite and >= 0, got {v!r}")
    if prior_logits:
        for k, v in prior_logits.items():
            if not math.isfinite(float(v)):
                raise ValueError(f"prior_logits[{k!r}] must be finite, got {v!r}")

    # Step 1: Normalize raw probabilities and evaluate H_raw BEFORE touching priors.
    total_raw = sum(max(0.0, float(v)) for v in raw_probs.values())
    if total_raw <= 0.0:
        norm_raw = {k: 1.0 / len(raw_probs) for k in raw_probs}
    else:
        norm_raw = {k: max(0.0, float(v)) / total_raw for k, v in raw_probs.items()}

    h_raw = normalized_entropy(norm_raw)
    should_escalate = h_raw >= hesitation_threshold

    eff_weight = prior_weight
    if gate_mode == "confident" and h_raw >= hesitation_threshold:
        eff_weight = 0.0
    elif gate_mode == "hesitant" and h_raw < hesitation_threshold:
        eff_weight = 0.0
    elif gate_mode == "proportional":
        eff_weight = prior_weight * h_raw

    if (not prior_logits or eff_weight == 0.0) and math.isclose(temperature, 1.0):
        return dict(norm_raw), h_raw, should_escalate

    # Step 2: Log-space Bayesian Product-of-Experts fusion:
    # log P_fused(k) = (1 / T) * log P_raw(k) + eff_weight * prior_logit(k)
    eps = 1e-12
    log_scores: dict[str, float] = {}
    for k, p in norm_raw.items():
        prior_val = float(prior_logits.get(k, 0.0)) if prior_logits else 0.0
        log_scores[k] = (math.log(max(eps, p)) / temperature) + (eff_weight * prior_val)

    max_log = max(log_scores.values())
    exp_scores = {k: math.exp(v - max_log) for k, v in log_scores.items()}
    denom = sum(exp_scores.values())
    fused = {k: v / denom for k, v in exp_scores.items()}
    return fused, h_raw, should_escalate


def stage2a_fast_exit_gate(
    prior_probs: Mapping[str, float],
    min_margin: float = 0.14,
    min_z: float = 1.65,
    sister_pack_collision: bool = False,
) -> tuple[bool, str | None]:
    """Evaluate Stage 2A Adaptive Fast-Exit Gate (0-VLM resolution when margin and Z-score pass).

    Auto-accepts Stage 2A Rank-1 when:
      1. Top-1 minus Top-2 probability margin >= min_margin (default 0.14).
      2. Top-1 Z-score across shortlist candidates >= min_z (default 1.65).
      3. sister_pack_collision is False (top candidates do not disagree solely on pack size).
    """
    if sister_pack_collision or len(prior_probs) < 2:
        return False, None
    ranked = sorted(prior_probs.items(), key=lambda kv: kv[1], reverse=True)
    top1_id, p1 = ranked[0]
    _, p2 = ranked[1]
    if (p1 - p2) < min_margin:
        return False, None

    vals = [float(v) for _, v in ranked]
    mean_v = statistics.fmean(vals)
    std_v = statistics.pstdev(vals)
    if std_v <= 1e-9:
        return False, None
    z_score = (p1 - mean_v) / std_v
    if z_score >= min_z:
        return True, top1_id
    return False, None


def inject_evidence_slot(
    questions: Mapping[str, DecisionQuestion],
    evidence_prompt: str,
    evidence_criteria: Mapping[str, str] | None = None,
    evidence_slot_id: str = "evidence_feature",
) -> dict[str, DecisionQuestion]:
    """Inject an explicit co-denoised / causal evidence-first slot into a question schema."""
    prompt = evidence_prompt.strip()
    if not prompt:
        return dict(questions)
    if evidence_slot_id in questions:
        return dict(questions)
    if len(questions) >= MAX_SLOTS_PER_PASS:
        raise ValueError(
            f"cannot inject {evidence_slot_id!r}: schema already has {len(questions)} slots "
            f"(MAX_SLOTS_PER_PASS={MAX_SLOTS_PER_PASS})"
        )

    crit = dict(evidence_criteria) if evidence_criteria is not None else dict(DEFAULT_EVIDENCE_CRITERIA)
    if len(crit) < 2 or len(crit) > MAX_OPTIONS_PER_SLOT:
        raise ValueError(
            f"evidence_criteria must contain between 2 and {MAX_OPTIONS_PER_SLOT} options, got {len(crit)}"
        )

    # Put evidence_slot_id FIRST so causal autoregressive models decode evidence before choice,
    # while bidirectional diffusion models co-denoise both on the same canvas.
    out: dict[str, DecisionQuestion] = {
        evidence_slot_id: DecisionQuestion(
            type="choice",
            instructions=prompt,
            criteria=crit,
            order=list(crit.keys()),
        )
    }
    out.update(questions)
    return out


def apply_layer5_deterministic_guards(
    chosen_id: str,
    raw_probs: Mapping[str, float],
    candidate_pack_sizes: Mapping[str, int] | None = None,
    decoded_pack_size: int | None = None,
    visual_energies: Mapping[str, float] | None = None,
    rank1_id: str | None = None,
    rank1_visual_similarity: float | None = None,
    catch_all_key: str = "none_of_the_above",
    energy_veto_delta: float = 0.045,
    low_confidence_threshold: float = 0.55,
    competitor_sim_threshold: float = 0.72,
) -> tuple[str, str]:
    """Apply the three Layer-5 deterministic post-generation guards.

    Precedence 1 (Pack-Size Sister Switch):
      If decoded_pack_size contradicts chosen_id's pack size and a sister candidate in the
      shortlist has decoded_pack_size, deterministically switch to that sister candidate.
    Precedence 2 (0.045 Visual Energy Veto):
      If Stage 2B flips Stage 2A rank1_id (chosen_id != rank1_id) with low token confidence
      C_2 = 1 - H_raw < 0.55 and visual energy drop E(rank1_id) - E(chosen_id) > 0.045,
      veto the flip and retain rank1_id.
    Precedence 3 (Catch-All Competitor Guard):
      If chosen_id == catch_all_key, retain rank1_id when rank1_visual_similarity >= 0.72;
      otherwise preserve catch_all_key.

    Returns:
      (final_chosen_id, guard_name_applied)
    """
    c2_confidence = 1.0 - normalized_entropy(raw_probs)

    # Precedence 1: Pack-Size Sister Switch
    if (
        decoded_pack_size is not None
        and decoded_pack_size > 0
        and candidate_pack_sizes
        and chosen_id in candidate_pack_sizes
    ):
        if candidate_pack_sizes[chosen_id] != decoded_pack_size:
            matching_sisters = [
                cid
                for cid, pk in candidate_pack_sizes.items()
                if pk == decoded_pack_size and cid != catch_all_key
            ]
            if matching_sisters:
                matching_sisters.sort(key=lambda cid: raw_probs.get(cid, 0.0), reverse=True)
                return matching_sisters[0], "layer5_pack_size_sister_switch"

    # Precedence 2: 0.045 Visual Energy Veto
    if (
        rank1_id
        and chosen_id != rank1_id
        and chosen_id != catch_all_key
        and visual_energies
        and rank1_id in visual_energies
        and chosen_id in visual_energies
    ):
        delta_e = float(visual_energies[rank1_id]) - float(visual_energies[chosen_id])
        if c2_confidence < low_confidence_threshold and delta_e > energy_veto_delta:
            return rank1_id, "layer5_visual_energy_veto"

    # Precedence 3: Catch-All / Competitor Guard
    if chosen_id == catch_all_key and rank1_id and rank1_visual_similarity is not None:
        if rank1_visual_similarity >= competitor_sim_threshold:
            return rank1_id, "layer5_competitor_similarity_guard"

    return chosen_id, "none"


def _row_to_shifted_logits(
    row: Mapping[str, float],
    *,
    is_logits: bool,
    eps: float = 1e-12,
) -> dict[str, float]:
    if not row:
        raise ValueError("calibration input row must be non-empty")
    clean: dict[str, float] = {}
    for raw_k, raw_v in row.items():
        key = str(raw_k)
        val = float(raw_v)
        if not math.isfinite(val):
            raise ValueError(f"non-finite value for {key!r}: {raw_v!r}")
        if not is_logits and val < 0.0:
            raise ValueError(f"probability for {key!r} must be >= 0, got {raw_v!r}")
        clean[key] = val

    if is_logits:
        max_z = max(clean.values())
        return {k: v - max_z for k, v in clean.items()}

    tot = sum(clean.values())
    if tot <= 0.0:
        raise ValueError("probability row sum must be > 0")
    raw_logs = {k: math.log(max(eps, v / tot)) for k, v in clean.items()}
    max_l = max(raw_logs.values())
    return {k: v - max_l for k, v in raw_logs.items()}


def _row_to_vector_logits(
    row: Mapping[str, float],
    *,
    is_logits: bool,
    eps: float = 1e-12,
) -> dict[str, float]:
    if not row:
        raise ValueError("calibration input row must be non-empty")
    clean: dict[str, float] = {}
    for raw_k, raw_v in row.items():
        key = str(raw_k)
        val = float(raw_v)
        if not math.isfinite(val):
            raise ValueError(f"non-finite value for {key!r}: {raw_v!r}")
        if not is_logits and val < 0.0:
            raise ValueError(f"probability for {key!r} must be >= 0, got {raw_v!r}")
        clean[key] = val

    if is_logits:
        mean_z = sum(clean.values()) / float(len(clean))
        return {k: v - mean_z for k, v in clean.items()}

    tot = sum(clean.values())
    if tot <= 0.0:
        raise ValueError("probability row sum must be > 0")
    raw_logs = {k: math.log(max(eps, v / tot)) for k, v in clean.items()}
    mean_l = sum(raw_logs.values()) / float(len(raw_logs))
    return {k: v - mean_l for k, v in raw_logs.items()}


def _softmax_dict(scores: Mapping[str, float]) -> dict[str, float]:
    max_s = max(scores.values())
    exps = {k: math.exp(v - max_s) for k, v in scores.items()}
    denom = sum(exps.values())
    return {k: v / denom for k, v in exps.items()}


@dataclass(frozen=True)
class TemperatureScalingModel:
    """Post-hoc scalar temperature scaling model P_cal(k | x) = softmax(logits / T)_k."""

    temperature: float
    l2_reg: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {self.temperature!r}")
        if not math.isfinite(self.l2_reg) or self.l2_reg < 0.0:
            raise ValueError(f"l2_reg must be finite and >= 0, got {self.l2_reg!r}")

    def predict_proba(
        self,
        probs_or_logits: Sequence[Mapping[str, float]],
        *,
        is_logits: bool = False,
    ) -> list[dict[str, float]]:
        inv_t = 1.0 / self.temperature
        out: list[dict[str, float]] = []
        for row in probs_or_logits:
            logits = _row_to_shifted_logits(row, is_logits=is_logits)
            scaled = {k: v * inv_t for k, v in logits.items()}
            out.append(_softmax_dict(scaled))
        return out


@dataclass(frozen=True)
class VectorScalingModel:
    """Post-hoc L2-regularized diagonal vector scaling model P_cal(k | x) = softmax(a_k * l_k + b_k)."""

    scales: dict[str, float]
    biases: dict[str, float]
    temperature_init: float
    l2_reg: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature_init) or self.temperature_init <= 0.0:
            raise ValueError(f"temperature_init must be finite and > 0, got {self.temperature_init!r}")
        if not math.isfinite(self.l2_reg) or self.l2_reg < 0.0:
            raise ValueError(f"l2_reg must be finite and >= 0, got {self.l2_reg!r}")

    def predict_proba(
        self,
        probs_or_logits: Sequence[Mapping[str, float]],
        *,
        is_logits: bool = False,
    ) -> list[dict[str, float]]:
        default_scale = 1.0 / self.temperature_init
        out: list[dict[str, float]] = []
        for row in probs_or_logits:
            logits = _row_to_vector_logits(row, is_logits=is_logits)
            scaled = {
                k: float(self.scales.get(k, default_scale)) * val + float(self.biases.get(k, 0.0))
                for k, val in logits.items()
            }
            out.append(_softmax_dict(scaled))
        return out


def _temp_objective_and_derivs(
    tau: float,
    logit_rows: Sequence[dict[str, float]],
    gold_labels: Sequence[str],
    l2_reg: float,
) -> tuple[float, float, float]:
    """Compute (loss, grad_tau, hess_tau) for tau = ln(T)."""
    inv_t = math.exp(-tau)
    n = float(len(logit_rows))
    nll_sum = 0.0
    grad_sum = 0.0
    hess_sum = 0.0

    for logits, gold in zip(logit_rows, gold_labels):
        scaled = {k: v * inv_t for k, v in logits.items()}
        probs = _softmax_dict(scaled)
        p_gold = max(1e-15, probs[gold])
        nll_sum -= math.log(p_gold)

        mean_l = sum(probs[k] * logits[k] for k in logits)
        var_l = sum(probs[k] * ((logits[k] - mean_l) ** 2) for k in logits)
        l_gold = logits[gold]

        # d(-ln p_y)/d(inv_t) = mean_l - l_gold; d(inv_t)/d(tau) = -inv_t
        grad_sum += (l_gold - mean_l) * inv_t
        # Exact second derivative wrt tau: inv_t^2 * var_l - inv_t * (l_gold - mean_l)
        # Gauss-Newton positive curvature inv_t^2 * var_l guarantees convexity in inv_t;
        # use exact Hessian when positive and fall back to Gauss-Newton curvature otherwise.
        exact_h = (inv_t * inv_t * var_l) - ((l_gold - mean_l) * inv_t)
        hess_sum += max(inv_t * inv_t * var_l, exact_h)

    loss = (nll_sum / n) + 0.5 * l2_reg * (tau * tau)
    grad = (grad_sum / n) + l2_reg * tau
    hess = (hess_sum / n) + l2_reg
    return loss, grad, hess


def fit_temperature_scaling(
    probs_or_logits: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    *,
    is_logits: bool = False,
    l2_reg: float = 1e-3,
) -> TemperatureScalingModel:
    """Fit post-hoc scalar temperature scaling via grid-bracketed damped Newton-Raphson on tau = ln(T)."""
    if not probs_or_logits or len(probs_or_logits) != len(labels):
        raise ValueError("probs_or_logits and labels must be non-empty sequences of equal length")
    if not math.isfinite(l2_reg) or l2_reg < 0.0:
        raise ValueError(f"l2_reg must be finite and >= 0, got {l2_reg!r}")

    logit_rows: list[dict[str, float]] = []
    gold_labels: list[str] = []
    for idx, (row, raw_lbl) in enumerate(zip(probs_or_logits, labels)):
        logits = _row_to_shifted_logits(row, is_logits=is_logits)
        lbl = str(raw_lbl)
        if lbl not in logits:
            raise ValueError(f"label {lbl!r} at index {idx} not present in row keys")
        logit_rows.append(logits)
        gold_labels.append(lbl)

    tau_min = -4.0
    tau_max = 4.0
    best_tau = 0.0
    best_loss, _, _ = _temp_objective_and_derivs(0.0, logit_rows, gold_labels, l2_reg)

    for step_i in range(81):
        cand_tau = tau_min + (tau_max - tau_min) * (float(step_i) / 80.0)
        cand_loss, _, _ = _temp_objective_and_derivs(cand_tau, logit_rows, gold_labels, l2_reg)
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_tau = cand_tau

    tau = best_tau
    for _ in range(50):
        loss, grad, hess = _temp_objective_and_derivs(tau, logit_rows, gold_labels, l2_reg)
        if abs(grad) < 1e-10:
            break
        delta = -grad / max(abs(hess), 1e-6)
        step = 1.0
        accepted = False
        while step >= 1e-6:
            trial_tau = min(tau_max, max(tau_min, tau + step * delta))
            trial_loss, _, _ = _temp_objective_and_derivs(trial_tau, logit_rows, gold_labels, l2_reg)
            if trial_loss <= loss - 1e-12:
                tau = trial_tau
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return TemperatureScalingModel(temperature=math.exp(tau), l2_reg=l2_reg)


def _vector_loss_grad_diag(
    scales: Sequence[float],
    biases: Sequence[float],
    indexed_rows: Sequence[list[tuple[int, float]]],
    gold_indices: Sequence[int],
    inv_t0: float,
    l2_reg: float,
) -> tuple[float, list[float], list[float], list[float], list[float]]:
    c_len = len(scales)
    n = float(len(indexed_rows))
    grad_a = [l2_reg * (scales[c] - inv_t0) for c in range(c_len)]
    grad_b = [l2_reg * biases[c] for c in range(c_len)]
    diag_a = [max(l2_reg, 1e-4) for _ in range(c_len)]
    diag_b = [max(l2_reg, 1e-4) for _ in range(c_len)]
    nll_sum = 0.0

    for items, g_idx in zip(indexed_rows, gold_indices):
        z_vals = [scales[c_idx] * l_val + biases[c_idx] for c_idx, l_val in items]
        max_z = max(z_vals)
        exps = [math.exp(z - max_z) for z in z_vals]
        denom = sum(exps)
        inv_n = 1.0 / n
        for (c_idx, l_val), exp_v in zip(items, exps):
            q_val = exp_v / denom
            if c_idx == g_idx:
                nll_sum -= math.log(max(1e-15, q_val))
                diff = q_val - 1.0
            else:
                diff = q_val
            grad_a[c_idx] += diff * l_val * inv_n
            grad_b[c_idx] += diff * inv_n
            curv = q_val * (1.0 - q_val) * inv_n
            diag_a[c_idx] += curv * (l_val * l_val)
            diag_b[c_idx] += curv

    reg_pen = 0.5 * l2_reg * sum(
        ((scales[c] - inv_t0) ** 2) + (biases[c] ** 2) for c in range(c_len)
    )
    return (nll_sum / n) + reg_pen, grad_a, grad_b, diag_a, diag_b


def fit_vector_scaling(
    probs_or_logits: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    *,
    is_logits: bool = False,
    l2_reg: float = 1e-3,
) -> VectorScalingModel:
    """Fit post-hoc diagonal vector scaling shrunk toward scalar temperature scaling (1 / T_0, 0)."""
    if not probs_or_logits or len(probs_or_logits) != len(labels):
        raise ValueError("probs_or_logits and labels must be non-empty sequences of equal length")
    if not math.isfinite(l2_reg) or l2_reg < 0.0:
        raise ValueError(f"l2_reg must be finite and >= 0, got {l2_reg!r}")

    t_model = fit_temperature_scaling(
        probs_or_logits,
        labels,
        is_logits=is_logits,
        l2_reg=min(l2_reg, 1e-2) if l2_reg > 0.0 else 0.0,
    )
    t0 = t_model.temperature
    inv_t0 = min(50.0, max(0.02, 1.0 / t0))

    logit_rows: list[dict[str, float]] = []
    gold_labels: list[str] = []
    key_set: set[str] = set()
    for idx, (row, raw_lbl) in enumerate(zip(probs_or_logits, labels)):
        logits = _row_to_vector_logits(row, is_logits=is_logits)
        lbl = str(raw_lbl)
        if lbl not in logits:
            raise ValueError(f"label {lbl!r} at index {idx} not present in row keys")
        logit_rows.append(logits)
        gold_labels.append(lbl)
        key_set.update(logits.keys())

    classes = sorted(key_set)
    cls_to_idx = {c: idx for idx, c in enumerate(classes)}
    c_len = len(classes)
    gold_set = set(gold_labels)
    observed_mask = [c in gold_set for c in classes]

    indexed_rows: list[list[tuple[int, float]]] = [
        [(cls_to_idx[k], v) for k, v in row.items()] for row in logit_rows
    ]
    gold_indices = [cls_to_idx[lbl] for lbl in gold_labels]

    # Parameter vector x of length 2 * c_len: [scales..., biases...]
    x_vec = [inv_t0] * c_len + [0.0] * c_len
    s_hist: list[list[float]] = []
    y_hist: list[list[float]] = []
    m_hist = 7

    loss, ga, gb, da, db = _vector_loss_grad_diag(
        x_vec[:c_len], x_vec[c_len:], indexed_rows, gold_indices, inv_t0, l2_reg
    )
    for idx_c in range(c_len):
        if not observed_mask[idx_c]:
            ga[idx_c] = 0.0
            gb[idx_c] = 0.0
    grad = ga + gb
    diag = da + db

    for _ in range(200):
        max_g = max(abs(v) for v in grad)
        if max_g < 1e-8:
            break

        # L-BFGS two-loop recursion with diagonal Gauss-Newton preconditioner
        q_vec = list(grad)
        alphas: list[float] = []
        rhos: list[float] = []
        for s_item, y_item in reversed(list(zip(s_hist, y_hist))):
            sy = sum(si * yi for si, yi in zip(s_item, y_item))
            rho = (1.0 / sy) if sy > 1e-14 else 0.0
            rhos.append(rho)
            a_val = rho * sum(si * qi for si, qi in zip(s_item, q_vec))
            alphas.append(a_val)
            q_vec = [qi - a_val * yi for qi, yi in zip(q_vec, y_item)]
        alphas.reverse()
        rhos.reverse()

        r_vec = [qi / max(di, 1e-4) for qi, di in zip(q_vec, diag)]
        for s_item, y_item, a_val, rho in zip(s_hist, y_hist, alphas, rhos):
            if rho == 0.0:
                continue
            b_val = rho * sum(yi * ri for yi, ri in zip(y_item, r_vec))
            r_vec = [ri + (a_val - b_val) * si for ri, si in zip(r_vec, s_item)]

        direction = [-ri for ri in r_vec]
        dir_deriv = sum(di * gi for di, gi in zip(direction, grad))
        if dir_deriv >= -1e-14:
            direction = [-gi / max(di, 1e-4) for gi, di in zip(grad, diag)]
            dir_deriv = sum(di * gi for di, gi in zip(direction, grad))

        step = 1.0
        accepted = False
        new_x = x_vec
        new_loss = loss
        new_grad = grad
        new_diag = diag

        while step >= 1e-6:
            cand_x = [xi + step * di for xi, di in zip(x_vec, direction)]
            for idx_a in range(c_len):
                if observed_mask[idx_a]:
                    cand_x[idx_a] = min(50.0, max(0.02, cand_x[idx_a]))
                    cand_x[c_len + idx_a] = min(20.0, max(-20.0, cand_x[c_len + idx_a]))
                else:
                    cand_x[idx_a] = inv_t0
                    cand_x[c_len + idx_a] = 0.0
            c_loss, c_ga, c_gb, c_da, c_db = _vector_loss_grad_diag(
                cand_x[:c_len], cand_x[c_len:], indexed_rows, gold_indices, inv_t0, l2_reg
            )
            for idx_c in range(c_len):
                if not observed_mask[idx_c]:
                    c_ga[idx_c] = 0.0
                    c_gb[idx_c] = 0.0
            if c_loss <= loss + 1e-4 * step * dir_deriv or c_loss < loss - 1e-12:
                new_x = cand_x
                new_loss = c_loss
                new_grad = c_ga + c_gb
                new_diag = c_da + c_db
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

        s_step = [nx - ox for nx, ox in zip(new_x, x_vec)]
        y_step = [ng - og for ng, og in zip(new_grad, grad)]
        if sum(si * yi for si, yi in zip(s_step, y_step)) > 1e-14:
            s_hist.append(s_step)
            y_hist.append(y_step)
            if len(s_hist) > m_hist:
                s_hist.pop(0)
                y_hist.pop(0)

        x_vec = new_x
        loss = new_loss
        grad = new_grad
        diag = new_diag

    scales_out = {c: x_vec[idx] for idx, c in enumerate(classes)}
    biases_out = {c: x_vec[c_len + idx] for idx, c in enumerate(classes)}
    return VectorScalingModel(
        scales=scales_out,
        biases=biases_out,
        temperature_init=t0,
        l2_reg=l2_reg,
    )

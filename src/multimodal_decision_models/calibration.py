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
from typing import Mapping

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

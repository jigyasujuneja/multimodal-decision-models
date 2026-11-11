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

"""Adversarial Security, Unicode Smuggling Defense, CUA Auditing, and Poisoning-Resilient Conformal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import re
from typing import Mapping, Sequence

from .conformal import DEFAULT_CONFORMAL_SEED_KEY, nonconformity_score
from .contracts import InvisibleSmugglingAudit, SchemaContractViolationError, audit_and_normalize_untrusted_text, normalize_and_strip_invisible_smuggling

_CANARY_TAG_FAMILY_RE = re.compile(
    r"<[ \t\r\n]*(?:/[ \t\r\n]*)?untrusted_[^<>]*>",
    re.IGNORECASE,
)
_INJECTION_DIRECTIVE_ANYWHERE_RE = re.compile(
    r"(?im)(?:^\s*(?:system|developer|assistant)\s*:\s*"
    r"(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output|return"
    r"|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b"
    r"|\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|preceding|system)\s+"
    r"(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b"
    r"|\breply\s+with\s+the\s+single\s+uppercase\s+option\s+letter\b)"
)
_CONSECUTIVE_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060-\u2064\ufeff\ufe00-\ufe0f]{2,}")
_ASCII_EMBEDDED_ZWJ_RE = re.compile(r"[A-Za-z0-9][\u200c\u200d][A-Za-z0-9]")
_SUSPICIOUS_XML_TAG_RE = re.compile(
    r"<[ \t\r\n]*(?:/[ \t\r\n]*)?(?:system|developer|assistant|instructions?|rubric|prompt|untrusted)\b",
    re.IGNORECASE,
)

_CUA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rubric_authority_spoof",
        re.compile(
            r"(?i)\b(?:evaluation\s+rubric\s+override|system\s+rubric\s*:"
            r"|official\s+scoring\s+directive|mandatory\s+judge\s+instruction"
            r"|authoritative\s+reranker\s+directive)\b"
        ),
    ),
    (
        "option_letter_hijack",
        re.compile(
            r"(?im)(?:\b(?:select|choose|pick|output|return|rank\s+first)\s+option\s+[A-Z]\b"
            r"|^\s*(?:final\s+)?answer\s*:\s*[A-Z]\s*$)"
        ),
    ),
    (
        "forced_probability_spoof",
        re.compile(
            r"(?i)\b(?:assign|give|set|output)\s+(?:a\s+)?(?:probability|posterior|score|logprob)"
            r"\s*(?:(?:of|=|:)\s*)?(?:1\.0+|100%)\b"
        ),
    ),
    (
        "peer_candidate_sabotage",
        re.compile(
            r"(?i)(?:\ball\s+other\s+(?:[a-z0-9_-]+\s+){0,2}"
            r"(?:candidates?|options?|passages?|documents?|choices?|items?)\s+"
            r"(?:are|were|have\s+been)\s+(?:[a-z0-9_-]+\s+){0,2}"
            r"(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid"
            r"|outdated|obsolete|decoys?|untrusted|fake|irrelevant)\b"
            r"|\bother\s+(?:[a-z0-9_-]+\s+){0,2}"
            r"(?:candidates?|options?|passages?|documents?|choices?|items?)\s+are\s+"
            r"(?:[a-z0-9_-]+\s+){0,2}"
            r"(?:deprecated|revoked|corrupted|malicious|fabricated|disqualified|unsafe|invalid"
            r"|outdated|obsolete|decoys?|untrusted|fake)\b"
            r"|\b(?:select|choose|pick|rank)\s+this\s+"
            r"(?:document|candidate|passage|option|item)\s+(?:only|exclusively|first)\b)"
        ),
    ),
    (
        "prompt_override_directive",
        re.compile(
            r"(?im)(?:\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?"
            r"(?:previous|prior|above|preceding|system)\s+"
            r"(?:instructions?|prompts?|rules?|rubrics?|criteria|directives?)\b"
            r"|^\s*(?:system|developer|assistant)\s*:\s*"
            r"(?:approve|allow|grant|accept|deny|reject|block|select|choose|pick|output"
            r"|return|ignore|disregard|forget|override|bypass|assign|rank|you\s+are|do\s+not)\b)"
        ),
    ),
    (
        "untrusted_tag_breakout",
        re.compile(
            r"(?i)(?:<[ \t\r\n]*(?:/[ \t\r\n]*)?untrusted_[^<>]*>|\[escaped_untrusted_tag\])"
        ),
    ),
)


def _has_zero_width_smuggling(raw_text: str, zero_width_count: int) -> bool:
    """Distinguish adversarial zero-width smuggling from benign Persian/Indic ZWNJ or emoji ZWJ."""
    if zero_width_count <= 0:
        return False
    explicit_smuggle_count = 0
    for idx, ch in enumerate(raw_text):
        code = ord(ch)
        if 0xE0100 <= code <= 0xE01EF:
            return True
        if code == 0x200B or (0x2060 <= code <= 0x2064) or (code == 0xFEFF and idx > 0):
            explicit_smuggle_count += 1
    if explicit_smuggle_count >= 1 and zero_width_count >= 2:
        return True
    if _CONSECUTIVE_ZERO_WIDTH_RE.search(raw_text) is not None:
        return True
    return _ASCII_EMBEDDED_ZWJ_RE.search(raw_text) is not None


@dataclass(frozen=True)
class CanaryUntrustedEnvelope:
    """Per-request HMAC-salted untrusted XML boundary envelope."""

    canary_hex: str
    open_tag: str
    close_tag: str
    sanitized_body: str
    fenced_text: str
    smuggling_audit: InvisibleSmugglingAudit


@dataclass(frozen=True)
class ComparativeAttackAudit:
    """Result of auditing candidate passages/options for CUA, JMA, and IPI smuggling."""

    quarantined_ids: tuple[str, ...]
    clean_ids: tuple[str, ...]
    reasons_by_id: dict[str, tuple[str, ...]]
    smuggling_counts_by_id: dict[str, int]
    any_attack_detected: bool


@dataclass(frozen=True)
class RobustConformalResult:
    """Corruption-resilient split conformal calibration summary."""

    q_hat: float
    standard_q_hat: float
    alpha: float
    max_corruption_fraction: float
    upper_trim_fraction: float
    n_calib: int
    k_standard: int
    k_robust: int
    method: str


@dataclass(frozen=True)
class PoisoningInfluenceAudit:
    """Boundary-spike and influence audit for conformal calibration nonconformity scores."""

    n_scores: int
    zero_spike_fraction: float
    unit_spike_fraction: float
    suspected_integrity_poisoning: bool
    suspected_availability_dos: bool
    anomaly_scores: tuple[float, ...]


def wrap_canary_untrusted_envelope(
    raw_text: str,
    *,
    request_id: str,
    seed_key: str = DEFAULT_CONFORMAL_SEED_KEY,
    max_chars: int = 16384,
) -> CanaryUntrustedEnvelope:
    """Wrap untrusted text in an unpredictable per-request HMAC-salted XML canary boundary.

    Prevents static `</untrusted_data>` tag-guessing attacks by binding the boundary tag to
    `HMAC-SHA256(seed_key, "canary:" + request_id)[:12]` and verifying zero unescaped closing
    tags remain inside the sanitized body.

    Args:
        raw_text: Untrusted OCR, RAG, DOM, or tool output text.
        request_id: Non-empty per-request identifier bound into the canary digest.
        seed_key: Secret or domain HMAC seed key.
        max_chars: Maximum sanitized character length retained inside the envelope.

    Returns:
        `CanaryUntrustedEnvelope` containing the canary tag pair and fenced payload.

    Raises:
        SchemaContractViolationError: If arguments are invalid or post-condition verification fails.
    """
    if not isinstance(raw_text, str):
        raise SchemaContractViolationError("raw_text must be a string")
    if not isinstance(request_id, str) or not request_id.strip():
        raise SchemaContractViolationError("request_id must be a non-empty string")
    if not isinstance(seed_key, str) or not seed_key.strip():
        raise SchemaContractViolationError("seed_key must be a non-empty string")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise SchemaContractViolationError(f"max_chars must be > 0, got {max_chars!r}")

    canary_hex = hmac.new(
        seed_key.encode("utf-8"),
        f"canary:{request_id.strip()}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:12]
    open_tag = f"<untrusted_data_{canary_hex}>"
    close_tag = f"</untrusted_data_{canary_hex}>"

    normalized, smuggling_audit = audit_and_normalize_untrusted_text(raw_text)
    escaped_tags = _CANARY_TAG_FAMILY_RE.sub("[escaped_untrusted_tag]", normalized)
    escaped_canary = re.sub(re.escape(canary_hex), "[escaped_canary]", escaped_tags, flags=re.IGNORECASE)
    neutralized = _INJECTION_DIRECTIVE_ANYWHERE_RE.sub("[neutralized_directive]", escaped_canary)
    clipped = neutralized[:max_chars] if len(neutralized) > max_chars else neutralized

    if close_tag.lower() in clipped.lower() or open_tag.lower() in clipped.lower():
        raise SchemaContractViolationError("canary tag collision survived sanitization")

    fenced = f"{open_tag}\n{clipped}\n{close_tag}"
    return CanaryUntrustedEnvelope(
        canary_hex=canary_hex,
        open_tag=open_tag,
        close_tag=close_tag,
        sanitized_body=clipped,
        fenced_text=fenced,
        smuggling_audit=smuggling_audit,
    )


def detect_comparative_undermining_attack(
    candidate_texts: Mapping[str, str],
) -> ComparativeAttackAudit:
    """Audit candidate texts for Comparative Undermining Attacks (CUA), JMA, and smuggling.

    Args:
        candidate_texts: Mapping from candidate or document ID to raw text content.

    Returns:
        `ComparativeAttackAudit` listing quarantined IDs, clean IDs, and per-candidate reasons.

    Raises:
        SchemaContractViolationError: If `candidate_texts` is empty or contains invalid entries.
    """
    if not isinstance(candidate_texts, Mapping) or not candidate_texts:
        raise SchemaContractViolationError("candidate_texts must be a non-empty mapping")

    quarantined: list[str] = []
    clean: list[str] = []
    reasons_by_id: dict[str, tuple[str, ...]] = {}
    smuggling_counts: dict[str, int] = {}

    for raw_id, raw_text in candidate_texts.items():
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise SchemaContractViolationError("candidate_texts keys must be non-empty strings")
        if not isinstance(raw_text, str):
            raise SchemaContractViolationError(f"candidate_texts[{raw_id!r}] must be a string")

        normalized, audit = audit_and_normalize_untrusted_text(raw_text)
        smuggling_counts[raw_id] = audit.total_suspicious_chars
        reasons: list[str] = []

        if audit.unicode_tag_chars > 0:
            reasons.append("unicode_tag_smuggling")
        if audit.bidi_override_chars > 0:
            reasons.append("bidi_override_smuggling")
        if _has_zero_width_smuggling(raw_text, audit.zero_width_chars):
            reasons.append("zero_width_smuggling")

        for label, pattern in _CUA_PATTERNS:
            if pattern.search(normalized):
                reasons.append(label)

        # Fullwidth homoglyphs combined with suspicious XML tags or directive keywords indicate evasion.
        if audit.fullwidth_homoglyphs >= 2 and (
            _SUSPICIOUS_XML_TAG_RE.search(normalized) is not None or len(reasons) > 0
        ):
            if "fullwidth_homoglyph_evasion" not in reasons:
                reasons.append("fullwidth_homoglyph_evasion")

        if reasons:
            quarantined.append(raw_id)
            reasons_by_id[raw_id] = tuple(reasons)
        else:
            clean.append(raw_id)
            reasons_by_id[raw_id] = ()

    return ComparativeAttackAudit(
        quarantined_ids=tuple(quarantined),
        clean_ids=tuple(clean),
        reasons_by_id=reasons_by_id,
        smuggling_counts_by_id=smuggling_counts,
        any_attack_detected=bool(quarantined),
    )


def calibrate_trimmed_robust_conformal_quantile(
    calib_probs: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    *,
    alpha: float = 0.10,
    max_corruption_fraction: float = 0.08,
    upper_trim_fraction: float = 0.0,
    method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> RobustConformalResult:
    """Calibrate a corruption-resilient split conformal quantile under adversarial poisoning.

    Defends against two adversarial threats to conformal calibration logs:
      1. Integrity poisoning (adversary injects up to `max_corruption_fraction` score-zero items
         to shrink `q_hat` and induce test miscoverage): shifts the conformal order statistic
         upward by `ceil(max_corruption_fraction * n)` so clean exchangeability coverage holds.
      2. Availability DoS poisoning (optional `upper_trim_fraction > 0` trims extreme upper-tail
         `1.0` label-flip spikes when bounded availability set sizes are prioritized).

    Args:
        calib_probs: Calibration probability distributions per example.
        calib_labels: Calibration ground-truth option keys.
        alpha: Target miscoverage rate in `(0, 1)`.
        max_corruption_fraction: Worst-case adversarial integrity corruption fraction in `[0, 0.45)`.
        upper_trim_fraction: Optional upper-tail Winsorization fraction in `[0, 0.25)`.
        method: Nonconformity score method (`"lac"`, `"aps"`, or `"raps"`).
        lam_reg: RAPS regularization weight `>= 0`.
        k_reg: RAPS unpenalized top-k cutoff `>= 0`.

    Returns:
        `RobustConformalResult` with both `q_hat` (robust) and `standard_q_hat`.

    Raises:
        SchemaContractViolationError: If inputs or corruption bounds are invalid.
    """
    n_calib = len(calib_probs)
    if n_calib == 0 or len(calib_labels) != n_calib:
        raise SchemaContractViolationError(
            "calib_probs and calib_labels must be non-empty sequences of equal length"
        )
    if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise SchemaContractViolationError(f"alpha must lie in (0, 1), got {alpha!r}")
    if (
        isinstance(max_corruption_fraction, bool)
        or not math.isfinite(max_corruption_fraction)
        or max_corruption_fraction < 0.0
        or max_corruption_fraction >= 0.45
    ):
        raise SchemaContractViolationError(
            f"max_corruption_fraction must lie in [0, 0.45), got {max_corruption_fraction!r}"
        )
    if (
        isinstance(upper_trim_fraction, bool)
        or not math.isfinite(upper_trim_fraction)
        or upper_trim_fraction < 0.0
        or upper_trim_fraction >= 0.25
    ):
        raise SchemaContractViolationError(
            f"upper_trim_fraction must lie in [0, 0.25), got {upper_trim_fraction!r}"
        )

    scores: list[float] = [
        nonconformity_score(
            probs,
            str(label),
            method=method,
            lam_reg=lam_reg,
            k_reg=k_reg,
            randomized=False,
        )
        for probs, label in zip(calib_probs, calib_labels)
    ]
    sorted_scores = sorted(scores)

    k_standard = math.ceil((n_calib + 1) * (1.0 - alpha))
    standard_q_hat = sorted_scores[k_standard - 1] if k_standard <= n_calib else float("inf")

    # Apply optional upper-tail Winsorization before computing corruption-guarded quantile.
    if upper_trim_fraction > 0.0:
        trim_cap_idx = max(0, n_calib - 1 - int(math.floor(upper_trim_fraction * n_calib)))
        cap_value = sorted_scores[trim_cap_idx]
        working_scores = [min(val, cap_value) for val in sorted_scores]
    else:
        working_scores = sorted_scores

    corruption_shift = int(math.ceil(max_corruption_fraction * n_calib))
    k_target = k_standard + corruption_shift
    if k_target > n_calib:
        k_robust = n_calib + 1
        robust_q_hat = float("inf")
    else:
        k_robust = k_target
        robust_q_hat = working_scores[k_robust - 1]

    return RobustConformalResult(
        q_hat=robust_q_hat,
        standard_q_hat=standard_q_hat,
        alpha=alpha,
        max_corruption_fraction=max_corruption_fraction,
        upper_trim_fraction=upper_trim_fraction,
        n_calib=n_calib,
        k_standard=k_standard,
        k_robust=k_robust,
        method=method,
    )


def audit_calibration_poisoning_influence(
    nonconformity_scores: Sequence[float],
    *,
    zero_tol: float = 1e-4,
    unit_tol: float = 1e-4,
) -> PoisoningInfluenceAudit:
    """Score per-item poisoning anomaly likelihood across conformal calibration logs.

    Combines boundary point-mass density amplification (detecting coordinated `0.0` integrity
    poisoning or `>= 1.0` availability label-flip spikes) with robust Median/MAD deviation so
    poisoned calibration entries receive significantly higher anomaly scores than clean items.

    Args:
        nonconformity_scores: Sequence of finite non-negative conformal scores.
        zero_tol: Threshold `<= zero_tol` treated as a boundary-zero score.
        unit_tol: Threshold `>= 1.0 - unit_tol` treated as an upper-boundary spike.

    Returns:
        `PoisoningInfluenceAudit` with summary flags and per-index anomaly scores.

    Raises:
        SchemaContractViolationError: If `nonconformity_scores` is empty or contains invalid floats.
    """
    total_n = len(nonconformity_scores)
    if total_n == 0:
        raise SchemaContractViolationError("nonconformity_scores must be non-empty")

    clean_scores: list[float] = []
    for raw_val in nonconformity_scores:
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise SchemaContractViolationError("nonconformity_scores must contain finite numbers")
        f_val = float(raw_val)
        if not math.isfinite(f_val) or f_val < 0.0:
            raise SchemaContractViolationError(
                f"nonconformity score must be finite and >= 0, got {raw_val!r}"
            )
        clean_scores.append(f_val)

    ordered = sorted(clean_scores)
    mid = total_n // 2
    median = (
        ordered[mid]
        if total_n % 2 == 1
        else 0.5 * (ordered[mid - 1] + ordered[mid])
    )
    abs_devs = sorted(abs(val - median) for val in clean_scores)
    mad = (
        abs_devs[mid]
        if total_n % 2 == 1
        else 0.5 * (abs_devs[mid - 1] + abs_devs[mid])
    )
    p15 = ordered[int(0.15 * float(total_n - 1))]
    p85 = ordered[min(total_n - 1, int(0.85 * float(total_n - 1)))]
    spread_scale = max(1.4826 * mad, p85 - p15, 0.18)

    zero_count = sum(1 for val in clean_scores if val <= zero_tol)
    unit_count = sum(1 for val in clean_scores if val >= 1.0 - unit_tol)
    zero_frac = float(zero_count) / float(total_n)
    unit_frac = float(unit_count) / float(total_n)

    # Anchor clean distribution statistics on non-boundary interior scores so increasing
    # corruption fraction (e.g., 16%, 30%, or 60% boundary spikes) remains strictly monotonic.
    interior = [val for val in ordered if zero_tol < val < 1.0 - unit_tol]
    if interior:
        int_mid = len(interior) // 2
        interior_median = (
            interior[int_mid]
            if len(interior) % 2 == 1
            else 0.5 * (interior[int_mid - 1] + interior[int_mid])
        )
    else:
        interior_median = 0.5

    suspected_integrity = zero_frac >= 0.05 and (not interior or interior_median > 0.03)
    suspected_dos = unit_frac >= 0.06 and (not interior or interior_median < 0.85)
    anchor_center = interior_median if (interior and (suspected_integrity or suspected_dos)) else median

    anomaly_list: list[float] = []
    for val in clean_scores:
        robust_z = abs(val - anchor_center) / spread_scale
        boundary_boost = 0.0
        if suspected_integrity and val <= zero_tol:
            boundary_boost = 10.0 + 20.0 * zero_frac
        elif suspected_dos and val >= 1.0 - unit_tol:
            boundary_boost = 10.0 + 20.0 * unit_frac
        anomaly_list.append(robust_z + boundary_boost)

    return PoisoningInfluenceAudit(
        n_scores=total_n,
        zero_spike_fraction=zero_frac,
        unit_spike_fraction=unit_frac,
        suspected_integrity_poisoning=suspected_integrity,
        suspected_availability_dos=suspected_dos,
        anomaly_scores=tuple(anomaly_list),
    )

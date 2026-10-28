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

"""Streaming Audio, Video, and Audio-Visual Decision Models (v0.8.0).

Implements four zero-dependency primitives extending calibrated decision models
to continuous audio, video, and full-duplex multimodal streams:

1. **Full-Duplex Voice Activity Projection (VAP) & Cost-Sensitive Turn-Taking**:
   Combines Stage-1 `/v1/systemone` slot readouts with prosodic pause duration,
   RMS energy slope, F0 pitch contour delta, and semantic completion probability
   under an asymmetric interruption loss matrix and a split-conformal singleton
   barge-in guard (`evaluate_duplex_turn_taking_decision`).
2. **Anytime-Valid Conformal Test Martingales & Streaming Video Early-Exit**:
   Converts frame-level nonconformity scores into smoothed conformal p-values
   and updates a predictable betting martingale bounded by Ville's maximal
   inequality (`ConformalMartingaleTracker`), paired with sequential tempered
   log-odds pooling for certified early exit (`evaluate_streaming_early_exit`).
3. **Dense Audio-Visual Event Localization (AVEL) & Cross-Modal Consistency**:
   Computes normalized Jensen-Shannon divergence (`jensen_shannon_divergence_bits`)
   and 3-slot spatio-temporal triangle violation mass (`score_audiovisual_triangle_consistency`)
   to partition segments into synchronous audio-visual events, audible off-screen
   events, visible silent events, and cross-modal spoofs/contradictions, while
   fencing untrusted ASR transcripts (`evaluate_audiovisual_event_agreement`).
4. **Pure-Stdlib RIFF WAV Acoustic Analyzer & Spatio-Temporal Contact Sheets (`ST-SoM`)**:
   Parses and synthesizes 16-bit PCM RIFF WAV streams using stdlib `wave` and
   `struct`, extracts 4-band Goertzel DFT spectral energies (`extract_wav_acoustic_features`),
   selects $B \\le 8$ temporally diverse keyframes and acoustic bursts via monotone
   submodular maximization (`select_submodular_spatiotemporal_frames`), and renders
   an indexed PNG Spatio-Temporal Contact Sheet (`compile_spatiotemporal_contact_sheet`).
"""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Mapping, Sequence

from .calibration import normalized_entropy, pre_gated_bayesian_poe_fusion
from .conformal import conformal_prediction_set
from .contracts import sanitize_untrusted_text
from .visual_compiler import ImageBuffer, encode_png

TURN_TAKING_ACTIONS: tuple[str, ...] = (
    "hold_floor",
    "yield_turn",
    "backchannel",
    "barge_in_interrupt",
    "wait_user_thinking",
)

AV_REGIME_LABELS: tuple[str, ...] = (
    "synchronous_av_event",
    "audible_offscreen_event",
    "visible_silent_event",
    "cross_modal_spoof_or_conflict",
)

_SILENT_BACKGROUND_KEYS: frozenset[str] = frozenset(
    {"background", "none", "silent", "silence", "ambient", "no_event", "off"}
)

_GOERTZEL_BANDS_HZ: tuple[float, float, float, float] = (250.0, 1000.0, 2500.0, 4000.0)

_DIGIT_GLYPHS_5X7: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
}


def _normalize_distribution(probs: Mapping[str, float]) -> dict[str, float]:
    if not probs:
        raise ValueError("probability mapping must be non-empty")
    cleaned: dict[str, float] = {}
    total = 0.0
    for k, v in probs.items():
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(f"invalid probability {v!r} for key {k!r}")
        cleaned[str(k)] = fv
        total += fv
    if total <= 0.0:
        uniform = 1.0 / float(len(cleaned))
        return {k: uniform for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


# ============================================================================
# Pillar 1: Full-Duplex Voice Activity Projection (VAP) & Turn-Taking Readout
# ============================================================================


@dataclass(frozen=True)
class VoiceActivityProjectionConfig:
    """Configuration for cost-sensitive full-duplex turn-taking decisions."""

    false_barge_in_cost: float = 6.0
    false_yield_on_thinking_cost: float = 2.8
    missed_barge_in_cost: float = 2.2
    turn_latency_cost: float = 1.0
    backchannel_mismatch_cost: float = 1.2
    pause_thinking_window_ms: tuple[float, float] = (180.0, 650.0)
    pause_yield_threshold_ms: float = 700.0
    prior_weight: float = 0.45
    hesitation_threshold: float = 0.28
    conformal_q_hat: float = 0.85

    def __post_init__(self) -> None:
        if self.false_barge_in_cost <= 0.0 or not math.isfinite(self.false_barge_in_cost):
            raise ValueError("false_barge_in_cost must be finite and > 0")
        if self.turn_latency_cost <= 0.0 or not math.isfinite(self.turn_latency_cost):
            raise ValueError("turn_latency_cost must be finite and > 0")
        if not (0.0 < self.conformal_q_hat <= 1.0):
            raise ValueError("conformal_q_hat must lie in (0, 1]")
        low_ms, high_ms = self.pause_thinking_window_ms
        if low_ms < 0.0 or high_ms <= low_ms:
            raise ValueError("pause_thinking_window_ms must satisfy 0 <= low < high")


@dataclass(frozen=True)
class TurnTakingDecisionResult:
    """Calibrated output of a full-duplex Voice Activity Projection turn-taking check."""

    selected_action: str
    raw_argmax_action: str
    calibrated_probabilities: dict[str, float]
    expected_losses: dict[str, float]
    conformal_set: tuple[str, ...]
    normalized_hesitation: float
    barge_in_blocked_by_guard: bool
    escalate_to_deliberative: bool


def compute_vap_prosodic_prior_logits(
    pause_ms: float,
    rms_slope_db: float,
    f0_delta_hz: float,
    semantic_complete_prob: float,
    user_overlap_active: bool = False,
    config: VoiceActivityProjectionConfig | None = None,
) -> dict[str, float]:
    """Computes zero-centered acoustic-prosodic log-prior logits over turn-taking states."""
    cfg = config or VoiceActivityProjectionConfig()
    if not math.isfinite(pause_ms) or pause_ms < 0.0:
        raise ValueError(f"pause_ms must be finite and >= 0, got {pause_ms}")
    if not math.isfinite(rms_slope_db) or not math.isfinite(f0_delta_hz):
        raise ValueError("rms_slope_db and f0_delta_hz must be finite")
    if not math.isfinite(semantic_complete_prob) or not (0.0 <= semantic_complete_prob <= 1.0):
        raise ValueError(f"semantic_complete_prob must be in [0, 1], got {semantic_complete_prob}")

    low_think, high_think = cfg.pause_thinking_window_ms
    in_thinking_pause = low_think <= pause_ms <= high_think
    long_pause = pause_ms >= cfg.pause_yield_threshold_ms
    falling_prosody = (f0_delta_hz <= -12.0) and (rms_slope_db <= -2.0)
    flat_prosody = (abs(f0_delta_hz) <= 10.0) and (abs(rms_slope_db) <= 2.0)

    logits: dict[str, float] = {a: 0.0 for a in TURN_TAKING_ACTIONS}

    # 1. Yield turn: boosted by long silence + syntactic completion + falling terminal contour
    yield_score = 1.8 * (semantic_complete_prob - 0.5)
    if long_pause:
        yield_score += min(1.6, 0.8 + (pause_ms - cfg.pause_yield_threshold_ms) / 600.0)
    if falling_prosody:
        yield_score += 0.9
    if user_overlap_active:
        yield_score -= 1.2
    logits["yield_turn"] = yield_score

    # 2. Wait user thinking: boosted by mid-utterance cognitive pause + flat pitch + incomplete syntax
    think_score = 1.6 * (0.55 - semantic_complete_prob)
    if in_thinking_pause:
        think_score += 1.15
    if flat_prosody:
        think_score += 0.75
    if long_pause and semantic_complete_prob >= 0.80:
        think_score -= 1.1
    logits["wait_user_thinking"] = think_score

    # 3. Hold floor: active agent/user speech without pause or overlap
    hold_score = 0.0
    if pause_ms < low_think and not user_overlap_active:
        hold_score += 1.10 + 0.6 * (1.0 - semantic_complete_prob)
    else:
        hold_score -= 0.75
    logits["hold_floor"] = hold_score

    # 4. Backchannel: short clause-boundary pause with rising/continuing pitch
    bc_score = -0.35
    if 120.0 <= pause_ms <= 450.0 and 0.35 <= semantic_complete_prob <= 0.72 and not user_overlap_active:
        bc_score += 0.95
    logits["backchannel"] = bc_score

    # 5. Barge-in interrupt: requires active user overlap or strong rising acoustic onset
    barge_score = -1.25
    if user_overlap_active:
        barge_score += 1.85 + max(0.0, min(1.2, rms_slope_db / 4.0))
    elif rms_slope_db >= 3.5 and pause_ms <= 100.0:
        barge_score += 0.85
    logits["barge_in_interrupt"] = barge_score

    mean_logit = sum(logits.values()) / float(len(logits))
    return {k: v - mean_logit for k, v in logits.items()}


def evaluate_duplex_turn_taking_decision(
    stage1_probs: Mapping[str, float],
    pause_ms: float,
    rms_slope_db: float,
    f0_delta_hz: float,
    semantic_complete_prob: float,
    user_overlap_active: bool = False,
    config: VoiceActivityProjectionConfig | None = None,
) -> TurnTakingDecisionResult:
    """Evaluates a calibrated full-duplex turn-taking action with asymmetric cost and conformal guards."""
    cfg = config or VoiceActivityProjectionConfig()
    norm_raw = _normalize_distribution(stage1_probs)
    for action in TURN_TAKING_ACTIONS:
        if action not in norm_raw:
            raise ValueError(f"stage1_probs missing required turn-taking action {action!r}")

    raw_argmax = max(TURN_TAKING_ACTIONS, key=lambda a: (norm_raw[a], -TURN_TAKING_ACTIONS.index(a)))
    h_raw = normalized_entropy(norm_raw)

    prior_logits = compute_vap_prosodic_prior_logits(
        pause_ms=pause_ms,
        rms_slope_db=rms_slope_db,
        f0_delta_hz=f0_delta_hz,
        semantic_complete_prob=semantic_complete_prob,
        user_overlap_active=user_overlap_active,
        config=cfg,
    )
    fused_probs, _, _ = pre_gated_bayesian_poe_fusion(
        raw_probs=norm_raw,
        prior_logits=prior_logits,
        prior_weight=cfg.prior_weight,
        hesitation_threshold=cfg.hesitation_threshold,
        gate_mode="proportional",
    )

    conf_res = conformal_prediction_set(
        probs=fused_probs,
        q_hat=cfg.conformal_q_hat,
        method="raps",
        lam_reg=0.02,
        k_reg=1,
    )
    conf_set = tuple(conf_res.prediction_set)
    h_cal = normalized_entropy(fused_probs)

    # Asymmetric loss matrix L(candidate_action, true_state)
    loss_matrix: dict[str, dict[str, float]] = {a: {b: 1.0 for b in TURN_TAKING_ACTIONS} for a in TURN_TAKING_ACTIONS}
    for a in TURN_TAKING_ACTIONS:
        loss_matrix[a][a] = 0.0

    # Interrupting while user holds floor or is thinking is severely penalized
    loss_matrix["barge_in_interrupt"]["hold_floor"] = cfg.false_barge_in_cost
    loss_matrix["barge_in_interrupt"]["wait_user_thinking"] = cfg.false_barge_in_cost
    loss_matrix["barge_in_interrupt"]["yield_turn"] = cfg.false_barge_in_cost * 0.55
    loss_matrix["barge_in_interrupt"]["backchannel"] = cfg.false_barge_in_cost * 0.65

    # Taking the turn (yield_turn -> agent speaks) when user is only pausing to think
    loss_matrix["yield_turn"]["wait_user_thinking"] = cfg.false_yield_on_thinking_cost
    loss_matrix["yield_turn"]["hold_floor"] = cfg.false_yield_on_thinking_cost
    loss_matrix["yield_turn"]["barge_in_interrupt"] = cfg.missed_barge_in_cost

    # Waiting when user yielded turn incurs moderate latency cost
    loss_matrix["wait_user_thinking"]["yield_turn"] = cfg.turn_latency_cost
    loss_matrix["wait_user_thinking"]["barge_in_interrupt"] = cfg.missed_barge_in_cost
    loss_matrix["hold_floor"]["barge_in_interrupt"] = cfg.missed_barge_in_cost
    loss_matrix["hold_floor"]["yield_turn"] = cfg.turn_latency_cost * 1.15

    for b in TURN_TAKING_ACTIONS:
        if b != "backchannel":
            loss_matrix["backchannel"][b] = cfg.backchannel_mismatch_cost

    expected_losses: dict[str, float] = {}
    for candidate in TURN_TAKING_ACTIONS:
        expected_losses[candidate] = sum(
            loss_matrix[candidate][truth] * fused_probs[truth] for truth in TURN_TAKING_ACTIONS
        )

    min_risk_action = min(
        TURN_TAKING_ACTIONS,
        key=lambda a: (expected_losses[a], -fused_probs[a], TURN_TAKING_ACTIONS.index(a)),
    )

    barge_in_blocked = False
    selected = min_risk_action

    # Conformal singleton safety guard for state-mutating barge-in or premature turn takeover
    if selected == "barge_in_interrupt":
        if len(conf_set) > 1 and ("hold_floor" in conf_set or "wait_user_thinking" in conf_set):
            barge_in_blocked = True
            selected = (
                "wait_user_thinking"
                if fused_probs["wait_user_thinking"] >= fused_probs["hold_floor"]
                else "hold_floor"
            )
    elif (
        selected == "yield_turn"
        and cfg.pause_thinking_window_ms[0] <= pause_ms <= cfg.pause_thinking_window_ms[1]
        and "wait_user_thinking" in conf_set
        and fused_probs["wait_user_thinking"] >= 0.25
    ):
        barge_in_blocked = True
        selected = "wait_user_thinking"

    escalate = bool(len(conf_set) > 2 or h_cal >= cfg.hesitation_threshold * 1.45 or h_raw >= 0.65)
    return TurnTakingDecisionResult(
        selected_action=selected,
        raw_argmax_action=raw_argmax,
        calibrated_probabilities=fused_probs,
        expected_losses=expected_losses,
        conformal_set=conf_set,
        normalized_hesitation=h_cal,
        barge_in_blocked_by_guard=barge_in_blocked,
        escalate_to_deliberative=escalate,
    )


# ============================================================================
# Pillar 2: Anytime-Valid Conformal Martingales & Streaming Video Early-Exit
# ============================================================================


@dataclass(frozen=True)
class ConformalMartingaleStepResult:
    """Single-step diagnostic for a sequential Conformal Test Martingale."""

    step_index: int
    nonconformity_score: float
    conformal_p_value: float
    betting_fraction: float
    martingale_value: float
    log_martingale_value: float
    drift_or_anomaly_alarm: bool


class ConformalMartingaleTracker:
    """Anytime-valid Conformal Test Martingale tracker bounded by Ville's maximal inequality.

    Maintains a non-negative test martingale $M_t = \\prod_{i=1}^t (1 + \\kappa_i (0.5 - p_i))$
    over sequential conformal p-values $p_t$. Under exchangeability, $p_t$ is super-uniform on
    $[0, 1]$, so $\\mathbb{E}[M_t \\mid \\mathcal{F}_{t-1}] \\le M_{t-1}$ with $M_0 = 1$, and
    $\\mathbb{P}(\\sup_{t \\ge 1} M_t \\ge 1/\\alpha) \\le \\alpha$ at any adaptive stopping time.
    """

    def __init__(
        self,
        calibration_scores: Sequence[float],
        alpha_alarm: float = 0.05,
        max_betting_fraction: float = 1.65,
        betting_Adaptation_rate: float = 0.45,
    ) -> None:
        if not calibration_scores:
            raise ValueError("calibration_scores must be non-empty")
        cleaned: list[float] = []
        for s in calibration_scores:
            fs = float(s)
            if not math.isfinite(fs) or fs < 0.0:
                raise ValueError(f"invalid calibration nonconformity score {s!r}")
            cleaned.append(fs)
        if not (0.0 < alpha_alarm < 1.0) or not math.isfinite(alpha_alarm):
            raise ValueError(f"alpha_alarm must lie in (0, 1), got {alpha_alarm}")
        if not (0.0 < max_betting_fraction < 2.0) or not math.isfinite(max_betting_fraction):
            raise ValueError(f"max_betting_fraction must lie in (0, 2), got {max_betting_fraction}")
        if not (0.0 < betting_Adaptation_rate <= 1.0) or not math.isfinite(betting_Adaptation_rate):
            raise ValueError(f"betting_Adaptation_rate must lie in (0, 1], got {betting_Adaptation_rate}")

        self._calib_scores: tuple[float, ...] = tuple(sorted(cleaned))
        self.alpha_alarm: float = float(alpha_alarm)
        self.alarm_threshold: float = 1.0 / float(alpha_alarm)
        self.max_betting_fraction: float = float(max_betting_fraction)
        self.betting_adaptation_rate: float = float(betting_Adaptation_rate)

        self._step_index: int = 0
        self._log_martingale: float = 0.0
        self._ema_deficit: float = 0.25
        self._alarm_triggered: bool = False

    @property
    def martingale_value(self) -> float:
        return math.exp(min(80.0, self._log_martingale))

    @property
    def alarm_triggered(self) -> bool:
        return self._alarm_triggered

    def compute_conformal_p_value(self, score: float, tie_break_u: float = 0.5) -> float:
        """Computes smoothed conformal p-value against the reference calibration bag."""
        fs = float(score)
        if not math.isfinite(fs) or fs < 0.0:
            raise ValueError(f"nonconformity score must be finite and >= 0, got {score}")
        if not (0.0 <= tie_break_u <= 1.0) or not math.isfinite(tie_break_u):
            raise ValueError(f"tie_break_u must lie in [0, 1], got {tie_break_u}")

        n = len(self._calib_scores)
        greater = 0
        equal = 0
        for ref in self._calib_scores:
            if ref > fs + 1e-12:
                greater += 1
            elif abs(ref - fs) <= 1e-12:
                equal += 1
        p_val = (float(greater) + float(tie_break_u) * float(equal + 1)) / float(n + 1)
        return max(1.0 / float(2 * (n + 1)), min(1.0, p_val))

    def step(self, nonconformity_score: float, tie_break_u: float = 0.5) -> ConformalMartingaleStepResult:
        """Processes one streaming nonconformity score and updates the test martingale."""
        p_t = self.compute_conformal_p_value(nonconformity_score, tie_break_u=tie_break_u)
        # Predictable betting fraction kappa_t depends strictly on past steps (F_{t-1})
        kappa_t = max(
            0.0,
            min(
                self.max_betting_fraction,
                self.max_betting_fraction * math.tanh(2.4 * max(0.0, self._ema_deficit)),
            ),
        )
        multiplier = max(1e-6, 1.0 + kappa_t * (0.5 - p_t))
        self._log_martingale += math.log(multiplier)
        m_val = self.martingale_value

        # Update predictable deficit tracker for t+1 after using kappa_t on p_t
        deficit = 0.5 - p_t
        self._ema_deficit = (
            (1.0 - self.betting_adaptation_rate) * self._ema_deficit
            + self.betting_adaptation_rate * deficit
        )
        self._step_index += 1
        if m_val >= self.alarm_threshold:
            self._alarm_triggered = True

        return ConformalMartingaleStepResult(
            step_index=self._step_index,
            nonconformity_score=float(nonconformity_score),
            conformal_p_value=p_t,
            betting_fraction=kappa_t,
            martingale_value=m_val,
            log_martingale_value=self._log_martingale,
            drift_or_anomaly_alarm=self._alarm_triggered,
        )


@dataclass(frozen=True)
class StreamingEarlyExitResult:
    """Result of streaming video/audio chunk early-exit evaluation."""

    selected_class: str
    exit_step: int
    total_steps: int
    compute_saved_fraction: float
    certified_early_exit: bool
    anomaly_alarm_triggered: bool
    posterior_probabilities: dict[str, float]
    conformal_set_at_exit: tuple[str, ...]
    martingale_trajectory: tuple[float, ...]
    hesitation_trajectory: tuple[float, ...]


def evaluate_streaming_early_exit(
    chunk_prob_stream: Sequence[Mapping[str, float]],
    calib_nonconformity_scores: Sequence[float],
    conformal_q_hat: float = 0.80,
    hesitation_exit_threshold: float = 0.22,
    min_chunks_before_exit: int = 2,
    stability_streak_required: int = 2,
    martingale_alarm_alpha: float = 0.05,
    temporal_decay: float = 0.92,
) -> StreamingEarlyExitResult:
    """Evaluates sequential video/audio chunks with Conformal Martingale drift protection and early exit."""
    if not chunk_prob_stream:
        raise ValueError("chunk_prob_stream must contain at least 1 chunk")
    if min_chunks_before_exit < 1:
        raise ValueError("min_chunks_before_exit must be >= 1")
    if stability_streak_required < 1:
        raise ValueError("stability_streak_required must be >= 1")
    if not (0.0 < temporal_decay <= 1.0):
        raise ValueError("temporal_decay must lie in (0, 1]")

    first_norm = _normalize_distribution(chunk_prob_stream[0])
    classes = tuple(first_norm.keys())
    if len(classes) < 2:
        raise ValueError("each chunk probability dict must have at least 2 classes")

    tracker = ConformalMartingaleTracker(
        calibration_scores=calib_nonconformity_scores,
        alpha_alarm=martingale_alarm_alpha,
    )

    accum_log_odds: dict[str, float] = {c: 0.0 for c in classes}
    martingale_traj: list[float] = []
    hesitation_traj: list[float] = []

    prev_leader: str | None = None
    leader_streak = 0
    exit_step = len(chunk_prob_stream)
    certified_exit = False
    posterior_at_exit: dict[str, float] = first_norm
    conf_set_at_exit: tuple[str, ...] = classes

    for idx, raw_chunk in enumerate(chunk_prob_stream, start=1):
        norm_chunk = _normalize_distribution(raw_chunk)
        if set(norm_chunk.keys()) != set(classes):
            raise ValueError("all chunks in chunk_prob_stream must share identical class keys")

        # Nonconformity proxy on unlabelled stream chunk: 1 - max_k p_t(k) + 0.35 * H_norm(p_t)
        h_chunk = normalized_entropy(norm_chunk)
        max_p = max(norm_chunk.values())
        stream_nonconf = min(1.0, (1.0 - max_p) + 0.35 * h_chunk)
        m_step = tracker.step(stream_nonconf)
        martingale_traj.append(m_step.martingale_value)

        # Tempered sequential log-odds update weighted by inverse chunk hesitation
        chunk_Reliability = max(0.15, 1.0 - 0.65 * h_chunk)
        for c in classes:
            p_c = max(1e-6, min(1.0 - 1e-6, norm_chunk[c]))
            accum_log_odds[c] = temporal_decay * accum_log_odds[c] + chunk_Reliability * math.log(p_c)

        max_logit = max(accum_log_odds.values())
        exp_scores = {c: math.exp(accum_log_odds[c] - max_logit) for c in classes}
        denom = sum(exp_scores.values())
        posterior = {c: exp_scores[c] / denom for c in classes}
        h_post = normalized_entropy(posterior)
        hesitation_traj.append(h_post)

        conf_res = conformal_prediction_set(
            probs=posterior,
            q_hat=conformal_q_hat,
            method="raps",
            lam_reg=0.02,
            k_reg=1,
        )
        leader = max(classes, key=lambda c: (posterior[c], -classes.index(c)))
        if leader == prev_leader:
            leader_streak += 1
        else:
            prev_leader = leader
            leader_streak = 1

        posterior_at_exit = posterior
        conf_set_at_exit = tuple(conf_res.prediction_set)

        if (
            idx >= min_chunks_before_exit
            and len(conf_set_at_exit) == 1
            and h_post <= hesitation_exit_threshold
            and leader_streak >= stability_streak_required
            and not tracker.alarm_triggered
        ):
            exit_step = idx
            certified_exit = True
            break

    total_steps = len(chunk_prob_stream)
    saved_fraction = float(total_steps - exit_step) / float(total_steps)
    selected_class = max(classes, key=lambda c: (posterior_at_exit[c], -classes.index(c)))

    return StreamingEarlyExitResult(
        selected_class=selected_class,
        exit_step=exit_step,
        total_steps=total_steps,
        compute_saved_fraction=saved_fraction,
        certified_early_exit=certified_exit,
        anomaly_alarm_triggered=tracker.alarm_triggered,
        posterior_probabilities=posterior_at_exit,
        conformal_set_at_exit=conf_set_at_exit,
        martingale_trajectory=tuple(martingale_traj),
        hesitation_trajectory=tuple(hesitation_traj),
    )


# ============================================================================
# Pillar 3: Dense Audio-Visual Event Localization (AVEL) & Cross-Modal Guards
# ============================================================================


@dataclass(frozen=True)
class AudioVisualAgreementResult:
    """Result of cross-modal audio-visual agreement and spoof/off-screen gating."""

    regime: str
    selected_event: str
    fused_probabilities: dict[str, float]
    jensen_shannon_divergence: float
    audio_hesitation: float
    visual_hesitation: float
    triangle_violation_score: float
    escalate_to_stage2: bool
    fenced_transcript: str | None


def jensen_shannon_divergence_bits(
    audio_probs: Mapping[str, float],
    visual_probs: Mapping[str, float],
) -> float:
    """Computes normalized Jensen-Shannon divergence in [0, 1] (log2 basis)."""
    pa = _normalize_distribution(audio_probs)
    pv = _normalize_distribution(visual_probs)
    if set(pa.keys()) != set(pv.keys()):
        raise ValueError("audio_probs and visual_probs must have identical keys")

    jsd_nats = 0.0
    for k in pa:
        a = pa[k]
        v = pv[k]
        m = 0.5 * (a + v)
        if a > 1e-12 and m > 1e-12:
            jsd_nats += 0.5 * a * math.log(a / m)
        if v > 1e-12 and m > 1e-12:
            jsd_nats += 0.5 * v * math.log(v / m)
    return max(0.0, min(1.0, jsd_nats / math.log(2.0)))


def score_audiovisual_triangle_consistency(
    audio_probs: Mapping[str, float],
    visual_probs: Mapping[str, float],
    sync_same_source_prob: float,
) -> float:
    """Computes continuous 3-slot spatio-temporal triangle violation mass in [0, 1].

    Inspects the 3-cycle `(AudioEvent, VisualEvent, CrossModalSync)`:
      - False-Sync Conflict: Audio and Visual confidently predict mutually exclusive
        active events ($c \\ne c'$), yet the sync head claims synchronous alignment.
      - Replay/Dubbing Desync Conflict: Audio and Visual both predict the same
        active foreground event $c$, yet the cross-modal sync verifier detects
        temporal/viseme desynchronization ($1 - p_{\\text{sync}}$).
    """
    pa = _normalize_distribution(audio_probs)
    pv = _normalize_distribution(visual_probs)
    if set(pa.keys()) != set(pv.keys()):
        raise ValueError("audio_probs and visual_probs must have identical keys")
    if not math.isfinite(sync_same_source_prob) or not (0.0 <= sync_same_source_prob <= 1.0):
        raise ValueError(f"sync_same_source_prob must lie in [0, 1], got {sync_same_source_prob}")

    keys = tuple(pa.keys())
    active_keys = [k for k in keys if k.lower() not in _SILENT_BACKGROUND_KEYS]
    if not active_keys:
        active_keys = list(keys)

    false_sync_mass = 0.0
    for ca in active_keys:
        for cv in active_keys:
            if ca != cv:
                mass = pa[ca] * pv[cv] * sync_same_source_prob
                if mass > false_sync_mass:
                    false_sync_mass = mass

    desync_spoof_mass = 0.0
    for c in active_keys:
        mass = pa[c] * pv[c] * (1.0 - sync_same_source_prob)
        if mass > desync_spoof_mass:
            desync_spoof_mass = mass

    return max(0.0, min(1.0, max(false_sync_mass, desync_spoof_mass)))


def fence_untrusted_audio_transcript(transcript: str) -> str:
    """Sanitizes and wraps an untrusted ASR transcript inside XML quarantine tags."""
    cleaned = sanitize_untrusted_text(transcript)
    return f"<untrusted_audio_transcript>\n{cleaned}\n</untrusted_audio_transcript>"


def evaluate_audiovisual_event_agreement(
    audio_probs: Mapping[str, float],
    visual_probs: Mapping[str, float],
    sync_same_source_prob: float = 0.85,
    jsd_conflict_threshold: float = 0.36,
    triangle_violation_threshold: float = 0.30,
    confident_hesitation_ceiling: float = 0.42,
    Uncertain_hesitation_floor: float = 0.52,
    asr_transcript: str | None = None,
) -> AudioVisualAgreementResult:
    """Classifies an audio-visual segment into synchronous, off-screen, silent, or spoof/conflict."""
    pa = _normalize_distribution(audio_probs)
    pv = _normalize_distribution(visual_probs)
    if set(pa.keys()) != set(pv.keys()):
        raise ValueError("audio_probs and visual_probs must have identical keys")

    keys = tuple(pa.keys())
    h_aud = normalized_entropy(pa)
    h_vis = normalized_entropy(pv)
    jsd = jensen_shannon_divergence_bits(pa, pv)
    s_tri = score_audiovisual_triangle_consistency(pa, pv, sync_same_source_prob)

    top_aud = max(keys, key=lambda k: (pa[k], -keys.index(k)))
    top_vis = max(keys, key=lambda k: (pv[k], -keys.index(k)))
    aud_is_bg = top_aud.lower() in _SILENT_BACKGROUND_KEYS
    vis_is_bg = top_vis.lower() in _SILENT_BACKGROUND_KEYS

    fenced = fence_untrusted_audio_transcript(asr_transcript) if asr_transcript is not None else None

    # Case 1: Audible off-screen event (confident active audio + silent/background or highly uncertain video)
    if not aud_is_bg and h_aud <= confident_hesitation_ceiling and (vis_is_bg or h_vis >= Uncertain_hesitation_floor):
        return AudioVisualAgreementResult(
            regime="audible_offscreen_event",
            selected_event=top_aud,
            fused_probabilities=dict(pa),
            jensen_shannon_divergence=jsd,
            audio_hesitation=h_aud,
            visual_hesitation=h_vis,
            triangle_violation_score=s_tri,
            escalate_to_stage2=False,
            fenced_transcript=fenced,
        )

    # Case 2: Visible silent event (confident active video + silent/background or highly uncertain audio)
    if not vis_is_bg and h_vis <= confident_hesitation_ceiling and (aud_is_bg or h_aud >= Uncertain_hesitation_floor):
        return AudioVisualAgreementResult(
            regime="visible_silent_event",
            selected_event=top_vis,
            fused_probabilities=dict(pv),
            jensen_shannon_divergence=jsd,
            audio_hesitation=h_aud,
            visual_hesitation=h_vis,
            triangle_violation_score=s_tri,
            escalate_to_stage2=False,
            fenced_transcript=fenced,
        )

    # Case 3: Cross-modal spoof or confident contradiction between two active foreground classes
    confident_disagreement = (
        top_aud != top_vis
        and not aud_is_bg
        and not vis_is_bg
        and h_aud <= confident_hesitation_ceiling
        and h_vis <= confident_hesitation_ceiling
        and jsd >= jsd_conflict_threshold
    )
    if s_tri >= triangle_violation_threshold or confident_disagreement:
        # Reliability-weighted conservative pool, flagged for mandatory Stage-2 verification
        w_a = max(0.1, 1.0 - h_aud)
        w_v = max(0.1, 1.0 - h_vis)
        w_sum = w_a + w_v
        pooled = {k: (w_a * pa[k] + w_v * pv[k]) / w_sum for k in keys}
        selected = max(keys, key=lambda k: (pooled[k], -keys.index(k)))
        return AudioVisualAgreementResult(
            regime="cross_modal_spoof_or_conflict",
            selected_event=selected,
            fused_probabilities=pooled,
            jensen_shannon_divergence=jsd,
            audio_hesitation=h_aud,
            visual_hesitation=h_vis,
            triangle_violation_score=s_tri,
            escalate_to_stage2=True,
            fenced_transcript=fenced,
        )

    # Case 4: Synchronous audio-visual event -> inverse-hesitation Product-of-Experts log-pool
    w_a = max(0.15, 1.0 - h_aud)
    w_v = max(0.15, 1.0 - h_vis)
    logits: dict[str, float] = {}
    for k in keys:
        la = math.log(max(1e-6, pa[k]))
        lv = math.log(max(1e-6, pv[k]))
        logits[k] = w_a * la + w_v * lv
    max_l = max(logits.values())
    exps = {k: math.exp(logits[k] - max_l) for k in keys}
    denom = sum(exps.values())
    fused = {k: exps[k] / denom for k in keys}
    selected = max(keys, key=lambda k: (fused[k], -keys.index(k)))
    h_fused = normalized_entropy(fused)

    return AudioVisualAgreementResult(
        regime="synchronous_av_event",
        selected_event=selected,
        fused_probabilities=fused,
        jensen_shannon_divergence=jsd,
        audio_hesitation=h_aud,
        visual_hesitation=h_vis,
        triangle_violation_score=s_tri,
        escalate_to_stage2=bool(h_fused >= Uncertain_hesitation_floor),
        fenced_transcript=fenced,
    )


# ============================================================================
# Pillar 4: Pure-Stdlib WAV Analyzer, Submodular Keyframes, & ST-SoM Sheet
# ============================================================================


@dataclass(frozen=True)
class WavAcousticWindowFeatures:
    """Short-time acoustic features extracted from a 16-bit PCM WAV window."""

    window_index: int
    start_sec: float
    end_sec: float
    rms_amplitude: float
    rms_dbfs: float
    zero_crossing_rate: float
    band_energies_norm: tuple[float, float, float, float]
    spectral_centroid_hz: float


@dataclass(frozen=True)
class WavAcousticSummary:
    """Full-stream acoustic summary over a 16-bit PCM RIFF WAV byte buffer."""

    sample_rate_hz: int
    num_channels: int
    num_samples: int
    duration_sec: float
    peak_dbfs: float
    mean_rms_dbfs: float
    windows: tuple[WavAcousticWindowFeatures, ...]


def write_pcm16_wav_bytes(
    samples: Sequence[int | float],
    sample_rate_hz: int = 16000,
) -> bytes:
    """Encodes mono 16-bit PCM samples (int16 or normalized float in [-1, 1]) into a RIFF WAV byte buffer."""
    if sample_rate_hz < 1000:
        raise ValueError(f"sample_rate_hz must be >= 1000, got {sample_rate_hz}")
    if not samples:
        raise ValueError("samples must be non-empty")

    pcm_ints: list[int] = []
    for s in samples:
        if isinstance(s, float) and abs(s) <= 1.05:
            val = int(round(max(-1.0, min(1.0, s)) * 32767.0))
        else:
            val = int(round(float(s)))
        pcm_ints.append(max(-32768, min(32767, val)))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(struct.pack(f"<{len(pcm_ints)}h", *pcm_ints))
    return buf.getvalue()


def _goertzel_band_power(samples: Sequence[float], sample_rate_hz: int, target_hz: float) -> float:
    n = len(samples)
    if n == 0 or target_hz <= 0.0 or target_hz >= sample_rate_hz * 0.5:
        return 0.0
    k = int(round((float(n) * target_hz) / float(sample_rate_hz)))
    omega = (2.0 * math.pi * float(k)) / float(n)
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return max(0.0, power / float(n * n))


def extract_wav_acoustic_features(
    wav_bytes: bytes,
    num_windows: int = 8,
) -> WavAcousticSummary:
    """Parses a 16-bit PCM RIFF WAV buffer and extracts windowed RMS, ZCR, and 4-band Goertzel spectra."""
    if not isinstance(wav_bytes, (bytes, bytearray)) or len(wav_bytes) < 44:
        raise ValueError("wav_bytes must be a valid RIFF WAV byte buffer")
    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")

    with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"only 16-bit PCM WAV (sampwidth=2) is supported, got {sampwidth}")
    if n_frames < 1 or n_channels < 1:
        raise ValueError("WAV stream contains no audio frames")

    total_shorts = len(raw_frames) // 2
    unpacked = struct.unpack(f"<{total_shorts}h", raw_frames[: total_shorts * 2])

    # Downmix multi-channel PCM to normalized mono float in [-1, 1]
    mono: list[float] = []
    if n_channels == 1:
        mono = [float(x) / 32768.0 for x in unpacked]
    else:
        for i in range(0, total_shorts - n_channels + 1, n_channels):
            avg = sum(unpacked[i : i + n_channels]) / float(n_channels)
            mono.append(avg / 32768.0)

    n_mono = len(mono)
    duration_sec = float(n_mono) / float(sample_rate)
    peak_amp = max((abs(x) for x in mono), default=0.0)
    peak_dbfs = 20.0 * math.log10(max(1e-9, peak_amp))

    win_len = max(1, n_mono // num_windows)
    windows: list[WavAcousticWindowFeatures] = []

    for w_idx in range(num_windows):
        start_idx = w_idx * win_len
        end_idx = n_mono if w_idx == num_windows - 1 else min(n_mono, (w_idx + 1) * win_len)
        seg = mono[start_idx:end_idx]
        if not seg:
            seg = [0.0]

        rms = math.sqrt(sum(x * x for x in seg) / float(len(seg)))
        rms_dbfs = 20.0 * math.log10(max(1e-9, rms))

        zc = 0
        for i in range(1, len(seg)):
            if (seg[i - 1] >= 0.0 and seg[i] < 0.0) or (seg[i - 1] < 0.0 and seg[i] >= 0.0):
                zc += 1
        zcr = float(zc) / float(max(1, len(seg) - 1))

        raw_bands = [
            _goertzel_band_power(seg, sample_rate, hz) for hz in _GOERTZEL_BANDS_HZ
        ]
        band_sum = sum(raw_bands)
        if band_sum <= 1e-15:
            norm_bands = (0.25, 0.25, 0.25, 0.25)
        else:
            norm_bands = (
                raw_bands[0] / band_sum,
                raw_bands[1] / band_sum,
                raw_bands[2] / band_sum,
                raw_bands[3] / band_sum,
            )

        centroid_hz = sum(
            norm_bands[b] * _GOERTZEL_BANDS_HZ[b] for b in range(4)
        )

        windows.append(
            WavAcousticWindowFeatures(
                window_index=w_idx,
                start_sec=float(start_idx) / float(sample_rate),
                end_sec=float(end_idx) / float(sample_rate),
                rms_amplitude=rms,
                rms_dbfs=rms_dbfs,
                zero_crossing_rate=zcr,
                band_energies_norm=norm_bands,
                spectral_centroid_hz=centroid_hz,
            )
        )

    mean_rms_dbfs = sum(w.rms_dbfs for w in windows) / float(len(windows))
    return WavAcousticSummary(
        sample_rate_hz=sample_rate,
        num_channels=n_channels,
        num_samples=n_mono,
        duration_sec=duration_sec,
        peak_dbfs=peak_dbfs,
        mean_rms_dbfs=mean_rms_dbfs,
        windows=tuple(windows),
    )


@dataclass(frozen=True)
class SpatioTemporalFrameCandidate:
    """Candidate temporal slice combining visual hesitation, acoustic energy, and embedding."""

    frame_index: int
    timestamp_sec: float
    hesitation_score: float
    acoustic_transient_score: float
    visual_motion_score: float
    feature_vector: tuple[float, ...]


@dataclass(frozen=True)
class SpatioTemporalSelectionResult:
    """Submodular selection of $B \\le 8$ keyframes and acoustic bursts."""

    selected_indices: tuple[int, ...]
    objective_value: float
    marginal_gains: tuple[float, ...]
    temporal_coverage_ratio: float


def _cosine_sim_tuples(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def select_submodular_spatiotemporal_frames(
    candidates: Sequence[SpatioTemporalFrameCandidate],
    budget_b: int = 8,
    facility_weight: float = 0.55,
    redundancy_penalty: float = 0.45,
    temporal_kernel_decay: float = 0.65,
) -> SpatioTemporalSelectionResult:
    """Selects up to `budget_b` keyframes maximizing utility + facility coverage - temporal redundancy."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if budget_b < 1:
        raise ValueError(f"budget_b must be >= 1, got {budget_b}")

    n = len(candidates)
    target_k = min(n, budget_b)
    max_time = max((c.timestamp_sec for c in candidates), default=1.0)
    time_span = max(1e-6, max_time - min((c.timestamp_sec for c in candidates), default=0.0))

    # Intrinsic salience utility u_t
    utilities: list[float] = []
    for c in candidates:
        u = (
            0.45 * max(0.0, min(1.0, float(c.hesitation_score)))
            + 0.30 * max(0.0, min(1.0, float(c.acoustic_transient_score)))
            + 0.25 * max(0.0, min(1.0, float(c.visual_motion_score)))
        )
        utilities.append(u)

    # Precompute spatio-temporal similarity kernel K(t, t') = cos(f_t, f_t') * exp(-gamma * |t - t'|)
    sim_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            feat_sim = _cosine_sim_tuples(candidates[i].feature_vector, candidates[j].feature_vector)
            dt_norm = abs(candidates[i].timestamp_sec - candidates[j].timestamp_sec) / time_span
            time_k = math.exp(-temporal_kernel_decay * dt_norm * float(n))
            val = feat_sim * time_k
            sim_matrix[i][j] = val
            sim_matrix[j][i] = val

    selected_pos: list[int] = []
    current_cov = [0.0] * n
    marginal_gains: list[float] = []
    total_obj = 0.0

    for _ in range(target_k):
        best_idx = -1
        best_gain = -1e18
        for cand_pos in range(n):
            if cand_pos in selected_pos:
                continue
            # Facility-location coverage gain + intrinsic utility - pairwise redundancy with already selected
            cov_gain = 0.0
            for j in range(n):
                if sim_matrix[cand_pos][j] > current_cov[j]:
                    cov_gain += sim_matrix[cand_pos][j] - current_cov[j]
            cov_gain /= float(n)

            red_loss = sum(sim_matrix[cand_pos][s] for s in selected_pos)
            gain = utilities[cand_pos] + facility_weight * cov_gain - redundancy_penalty * red_loss
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_idx = cand_pos

        if best_idx < 0:
            break
        selected_pos.append(best_idx)
        marginal_gains.append(best_gain)
        total_obj += best_gain
        for j in range(n):
            if sim_matrix[best_idx][j] > current_cov[j]:
                current_cov[j] = sim_matrix[best_idx][j]

    # Sort selected indices chronologically while preserving marginal gains order
    paired = sorted(zip(selected_pos, marginal_gains), key=lambda pair: candidates[pair[0]].frame_index)
    chronological_indices = tuple(candidates[p[0]].frame_index for p in paired)
    chronological_gains = tuple(p[1] for p in paired)
    cov_ratio = sum(current_cov) / float(n)

    return SpatioTemporalSelectionResult(
        selected_indices=chronological_indices,
        objective_value=total_obj,
        marginal_gains=chronological_gains,
        temporal_coverage_ratio=cov_ratio,
    )


@dataclass(frozen=True)
class CompiledSpatioTemporalSheet:
    """Compiled Spatio-Temporal Contact Sheet (`ST-SoM`) pairing video keyframes and WAV spectrogram."""

    sheet_buffer: ImageBuffer
    png_bytes: bytes
    selected_indices: tuple[int, ...]
    tile_width: int
    tile_height: int


def _draw_pixel(buf: bytearray, width: int, height: int, x: int, y: int, rgb: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        off = (y * width + x) * 3
        buf[off] = rgb[0]
        buf[off + 1] = rgb[1]
        buf[off + 2] = rgb[2]


def _draw_rect(
    buf: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    rgb: tuple[int, int, int],
) -> None:
    cx0 = max(0, min(width, x0))
    cy0 = max(0, min(height, y0))
    cx1 = max(cx0, min(width, x1))
    cy1 = max(cy0, min(height, y1))
    for y in range(cy0, cy1):
        row_off = y * width * 3
        for x in range(cx0, cx1):
            off = row_off + x * 3
            buf[off] = rgb[0]
            buf[off + 1] = rgb[1]
            buf[off + 2] = rgb[2]


def _draw_glyph_5x7(
    buf: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    char: str,
    rgb: tuple[int, int, int],
) -> None:
    rows = _DIGIT_GLYPHS_5X7.get(char)
    if rows is None:
        return
    for dy, mask in enumerate(rows):
        for dx in range(5):
            if (mask >> (4 - dx)) & 1:
                _draw_pixel(buf, width, height, x0 + dx, y0 + dy, rgb)


def compile_spatiotemporal_contact_sheet(
    frames: Sequence[ImageBuffer],
    selected_indices: Sequence[int],
    wav_summary: WavAcousticSummary,
    tile_width: int = 80,
    tile_height: int = 60,
) -> CompiledSpatioTemporalSheet:
    """Compiles a single PNG Spatio-Temporal Contact Sheet (`ST-SoM`) with keyframes + WAV spectral strip."""
    if not frames:
        raise ValueError("frames must be non-empty")
    if not selected_indices:
        raise ValueError("selected_indices must be non-empty")
    if tile_width < 24 or tile_height < 24:
        raise ValueError("tile_width and tile_height must be >= 24")

    chosen: list[int] = []
    for idx in selected_indices[:8]:
        if idx < 0 or idx >= len(frames):
            raise ValueError(f"selected frame index {idx} out of range [0, {len(frames) - 1}]")
        chosen.append(int(idx))

    num_tiles = len(chosen)
    sheet_w = num_tiles * tile_width
    spectro_h = max(36, tile_height // 2 + 12)
    sheet_h = tile_height + spectro_h

    canvas = bytearray(sheet_w * sheet_h * 3)
    # Dark slate background for spectrogram strip
    _draw_rect(canvas, sheet_w, sheet_h, 0, tile_height, sheet_w, sheet_h, (18, 22, 32))

    # 1. Render top-row selected RGB keyframes with [F1..FB] badges
    for pos, frame_idx in enumerate(chosen):
        resized = frames[frame_idx].resize_bilinear(tile_width, tile_height)
        x_offset = pos * tile_width
        for y in range(tile_height):
            src_off = y * tile_width * 3
            dst_off = (y * sheet_w + x_offset) * 3
            canvas[dst_off : dst_off + tile_width * 3] = resized.rgb[src_off : src_off + tile_width * 3]

        # Draw dark badge box in top-left corner of tile + F<pos+1> label
        _draw_rect(canvas, sheet_w, sheet_h, x_offset + 2, 2, x_offset + 16, 11, (12, 16, 24))
        _draw_glyph_5x7(canvas, sheet_w, sheet_h, x_offset + 3, 3, "F", (255, 220, 80))
        digit_char = str((pos + 1) % 10)
        _draw_glyph_5x7(canvas, sheet_w, sheet_h, x_offset + 10, 3, digit_char, (255, 255, 255))

    # 2. Render bottom-row synchronized WAV RMS envelope + 4-band Goertzel spectral bars
    windows = wav_summary.windows
    if windows:
        col_w = max(1, sheet_w // len(windows))
        band_colors: tuple[tuple[int, int, int], ...] = (
            (66, 133, 244),   # 250 Hz low voicing (blue)
            (52, 168, 83),    # 1000 Hz vowel formant (green)
            (251, 188, 4),    # 2500 Hz fricative burst (amber)
            (234, 67, 53),    # 4000 Hz high transient (red)
        )
        for w_idx, win in enumerate(windows):
            wx0 = w_idx * col_w
            wx1 = sheet_w if w_idx == len(windows) - 1 else min(sheet_w, (w_idx + 1) * col_w)
            sub_w = max(1, (wx1 - wx0) // 4)
            # Map RMS dBFS from [-60, 0] onto [0.15, 1.0]
            rms_norm = max(0.15, min(1.0, (win.rms_dbfs + 60.0) / 60.0))
            for b_idx in range(4):
                bx0 = wx0 + b_idx * sub_w
                bx1 = wx1 if b_idx == 3 else min(wx1, bx0 + sub_w)
                bar_frac = max(0.08, min(1.0, rms_norm * (0.35 + 1.45 * win.band_energies_norm[b_idx])))
                bar_h = int(round(bar_frac * float(spectro_h - 6)))
                by0 = sheet_h - 2 - bar_h
                _draw_rect(canvas, sheet_w, sheet_h, bx0, by0, max(bx0 + 1, bx1 - 1), sheet_h - 2, band_colors[b_idx])

    # 3. Draw vertical cyan synchronization markers showing where each selected keyframe falls in time
    total_frames = max(1, len(frames) - 1)
    for frame_idx in chosen:
        marker_x = min(sheet_w - 1, int(round((float(frame_idx) / float(total_frames)) * float(sheet_w - 1))))
        _draw_rect(canvas, sheet_w, sheet_h, marker_x, tile_height, min(sheet_w, marker_x + 2), sheet_h, (0, 229, 255))

    sheet_buf = ImageBuffer(width=sheet_w, height=sheet_h, rgb=bytes(canvas))
    png_data = encode_png(sheet_buf)
    return CompiledSpatioTemporalSheet(
        sheet_buffer=sheet_buf,
        png_bytes=png_data,
        selected_indices=tuple(chosen),
        tile_width=tile_width,
        tile_height=tile_height,
    )

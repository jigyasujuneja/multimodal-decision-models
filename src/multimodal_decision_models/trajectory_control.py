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

"""Trajectory-Level Polynomial Alpha-Spending and Net Expected Value of Control (NEVC).

Addresses two failure modes of step-wise agent decision controllers:
  1. Ning et al. (EMNLP 2026): per-step split conformal prediction at flat alpha=0.10 suffers up to
     83.5% trajectory-level miscoverage over multi-step horizons due to error compounding and
     Markov transitions; polynomial alpha-spending restores trajectory-level validity.
  2. Vats (arXiv:2610.24649): uncalibrated decision-model loop controllers inflate agent cost (+19%)
     with 0% accuracy gain when base-rate miscalibration triggers wasteful double-check turns;
     NEVC gates interventions strictly on calibrated net expected utility vs. a placebo baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .conformal import DEFAULT_CONFORMAL_SEED_KEY, calibrate_conformal_quantile, conformal_prediction_set, hmac_uniform_float64
from .contracts import SchemaContractViolationError
from .eval_suite import paired_mcnemar_test


def _riemann_zeta_approx(exponent_p: float, terms: int = 512) -> float:
    """Compute convergent Riemann zeta(p) for p > 1 using Euler-Maclaurin tail remainder."""
    partial = sum(float(idx) ** (-exponent_p) for idx in range(1, terms + 1))
    n_float = float(terms)
    tail = (n_float ** (1.0 - exponent_p)) / (exponent_p - 1.0) - 0.5 * (n_float ** (-exponent_p))
    return partial + tail


def polynomial_alpha_at_step(
    step_index: int,
    *,
    alpha_total: float = 0.10,
    exponent_p: float = 1.6,
    horizon_max_steps: int | None = None,
    markov_slack_factor: float = 1.0,
) -> float:
    """Compute step-t miscoverage budget `alpha_t` under a polynomial risk-spending schedule.

    Allocates `sum_{t=1}^{T_max} alpha_t <= alpha_total` (finite horizon) or
    `sum_{t=1}^inf alpha_t = alpha_total` via `zeta(p)` when `horizon_max_steps is None` (`p > 1`).

    Args:
        step_index: 1-based trajectory step index (`>= 1`).
        alpha_total: End-to-end trajectory miscoverage budget in `(0, 1)`.
        exponent_p: Polynomial decay exponent (`> 0` for finite horizon; `> 1` for unbounded).
        horizon_max_steps: Optional finite upper bound `T_max >= 1` on trajectory length.
        markov_slack_factor: Multiplicative tightening factor in `(0, 1]` calibrated on validation
            trajectories to absorb non-i.i.d. Markov state transitions.

    Returns:
        Step-specific miscoverage budget `alpha_t` in `(0, 1)`.

    Raises:
        SchemaContractViolationError: If any schedule argument is out of domain.
    """
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 1:
        raise SchemaContractViolationError(f"step_index must be an integer >= 1, got {step_index!r}")
    if (
        isinstance(alpha_total, bool)
        or not isinstance(alpha_total, (int, float))
        or not math.isfinite(float(alpha_total))
        or float(alpha_total) <= 0.0
        or float(alpha_total) >= 1.0
    ):
        raise SchemaContractViolationError(f"alpha_total must lie in (0, 1), got {alpha_total!r}")
    if (
        isinstance(exponent_p, bool)
        or not isinstance(exponent_p, (int, float))
        or not math.isfinite(float(exponent_p))
        or float(exponent_p) <= 0.0
    ):
        raise SchemaContractViolationError(f"exponent_p must be > 0, got {exponent_p!r}")
    if (
        isinstance(markov_slack_factor, bool)
        or not isinstance(markov_slack_factor, (int, float))
        or not math.isfinite(float(markov_slack_factor))
        or float(markov_slack_factor) <= 0.0
        or float(markov_slack_factor) > 1.0
    ):
        raise SchemaContractViolationError("markov_slack_factor must lie in (0, 1]")

    p_val = float(exponent_p)
    effective_budget = float(alpha_total) * float(markov_slack_factor)
    if horizon_max_steps is not None:
        if (
            isinstance(horizon_max_steps, bool)
            or not isinstance(horizon_max_steps, int)
            or horizon_max_steps < 1
        ):
            raise SchemaContractViolationError("horizon_max_steps must be an integer >= 1 or None")
        if step_index > horizon_max_steps:
            raise SchemaContractViolationError(
                f"step_index ({step_index}) exceeds horizon_max_steps ({horizon_max_steps})"
            )
        norm_const = sum(float(j) ** (-p_val) for j in range(1, horizon_max_steps + 1))
    else:
        if p_val <= 1.0:
            raise SchemaContractViolationError(
                f"unbounded horizon requires exponent_p > 1.0 so zeta(p) converges, got {p_val}"
            )
        norm_const = _riemann_zeta_approx(p_val)

    alpha_step = effective_budget * (float(step_index) ** (-p_val)) / norm_const
    return min(0.999999, max(math.nextafter(0.0, 1.0), alpha_step))


class PolynomialAlphaSpendingTracker:
    """Stateful or step-indexed conformal quantile resolver for multi-step agent trajectories."""

    def __init__(
        self,
        calib_probs: Sequence[Mapping[str, float]],
        calib_labels: Sequence[str],
        *,
        alpha_total: float = 0.10,
        exponent_p: float = 1.6,
        horizon_max_steps: int | None = 12,
        markov_slack_factor: float = 1.0,
        conformal_method: str = "raps",
        lam_reg: float = 0.01,
        k_reg: int = 1,
    ) -> None:
        """Initialize a trajectory conformal risk-spending tracker over calibration receipts.

        Args:
            calib_probs: Calibration probability distributions per step.
            calib_labels: Ground-truth calibration step labels.
            alpha_total: Target trajectory-wide miscoverage bound in `(0, 1)`.
            exponent_p: Polynomial decay exponent (`> 1` when `horizon_max_steps is None`).
            horizon_max_steps: Optional finite trajectory length `T_max`.
            markov_slack_factor: Tightening factor in `(0, 1]` for Markov transition slack.
            conformal_method: Score method (`"lac"`, `"aps"`, or `"raps"`).
            lam_reg: RAPS penalty weight.
            k_reg: RAPS penalty rank onset.
        """
        if not calib_probs or len(calib_probs) != len(calib_labels):
            raise SchemaContractViolationError(
                "calib_probs and calib_labels must be non-empty sequences of equal length"
            )
        # Validate schedule parameters eagerly at construction time.
        _ = polynomial_alpha_at_step(
            1,
            alpha_total=alpha_total,
            exponent_p=exponent_p,
            horizon_max_steps=horizon_max_steps,
            markov_slack_factor=markov_slack_factor,
        )
        self._calib_probs = [dict(dist) for dist in calib_probs]
        self._calib_labels = [str(lbl) for lbl in calib_labels]
        self.alpha_total = float(alpha_total)
        self.exponent_p = float(exponent_p)
        self.horizon_max_steps = horizon_max_steps
        self.markov_slack_factor = float(markov_slack_factor)
        self.conformal_method = conformal_method
        self.lam_reg = float(lam_reg)
        self.k_reg = int(k_reg)
        self._quantile_cache: dict[int, float] = {}

    def alpha_for_step(self, step_index: int) -> float:
        """Return step-specific `alpha_t` allocated to 1-based `step_index`."""
        return polynomial_alpha_at_step(
            step_index,
            alpha_total=self.alpha_total,
            exponent_p=self.exponent_p,
            horizon_max_steps=self.horizon_max_steps,
            markov_slack_factor=self.markov_slack_factor,
        )

    def q_hat_for_step(self, step_index: int) -> float:
        """Return cached split-conformal quantile `q_hat_t` for 1-based `step_index`."""
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 1:
            raise SchemaContractViolationError(
                f"step_index must be an integer >= 1, got {step_index!r}"
            )
        if step_index not in self._quantile_cache:
            alpha_t = self.alpha_for_step(step_index)
            cal_res = calibrate_conformal_quantile(
                self._calib_probs,
                self._calib_labels,
                alpha=alpha_t,
                method=self.conformal_method,
                lam_reg=self.lam_reg,
                k_reg=self.k_reg,
            )
            self._quantile_cache[step_index] = cal_res.q_hat
        return self._quantile_cache[step_index]


def fit_markov_trajectory_slack(
    calib_trajectories: Sequence[Sequence[tuple[Mapping[str, float], str]]],
    step_calib_probs: Sequence[Mapping[str, float]],
    step_calib_labels: Sequence[str],
    *,
    alpha_total: float = 0.10,
    exponent_p: float = 1.6,
    horizon_max_steps: int | None = 12,
    conformal_method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> float:
    """Calibrate `markov_slack_factor` in `(0, 1]` on held-out trajectories for finite-sample validity.

    Searches descending grid `1.00, 0.95, ..., 0.05, 0.01` and selects the largest factor
    (smallest sets) achieving `valid_trajectories >= ceil((m + 1) * (1 - alpha_total))`.

    Args:
        calib_trajectories: Non-empty sequence of validation trajectories of `(probs, gold_label)`.
        step_calib_probs: Step-level calibration distributions.
        step_calib_labels: Step-level calibration labels.
        alpha_total: Target trajectory miscoverage rate in `(0, 1)`.
        exponent_p: Polynomial decay exponent.
        horizon_max_steps: Optional finite trajectory horizon.
        conformal_method: Conformal scoring method.
        lam_reg: RAPS penalty weight.
        k_reg: RAPS penalty onset rank.

    Returns:
        Calibrated `markov_slack_factor` in `(0, 1]`.

    Raises:
        SchemaContractViolationError: If `calib_trajectories` is too small for `(m + 1)(1 - alpha)`
            finite-sample coverage or no positive slack factor satisfies the coverage bound.
    """
    if not calib_trajectories:
        raise SchemaContractViolationError("calib_trajectories must be non-empty")
    if (
        isinstance(alpha_total, bool)
        or not isinstance(alpha_total, (int, float))
        or not math.isfinite(float(alpha_total))
        or float(alpha_total) <= 0.0
        or float(alpha_total) >= 1.0
    ):
        raise SchemaContractViolationError(f"alpha_total must lie in (0, 1), got {alpha_total!r}")
    m_traj = len(calib_trajectories)
    required_valid = math.ceil(float(m_traj + 1) * (1.0 - float(alpha_total)) - 1e-12)
    if required_valid > m_traj:
        raise SchemaContractViolationError(
            f"insufficient calib_trajectories ({m_traj}) for finite-sample trajectory validity "
            f"at alpha_total={alpha_total} (requires ceil((m + 1)*(1 - alpha)) <= m)"
        )

    candidate_grid = [round(1.0 - 0.05 * float(idx), 4) for idx in range(20)] + [0.01]
    for candidate_slack in candidate_grid:
        tracker = PolynomialAlphaSpendingTracker(
            step_calib_probs,
            step_calib_labels,
            alpha_total=alpha_total,
            exponent_p=exponent_p,
            horizon_max_steps=horizon_max_steps,
            markov_slack_factor=candidate_slack,
            conformal_method=conformal_method,
            lam_reg=lam_reg,
            k_reg=k_reg,
        )
        valid_trajectories = 0
        for traj in calib_trajectories:
            if not traj:
                raise SchemaContractViolationError("each calibration trajectory must be non-empty")
            all_steps_covered = True
            for t_idx, (dist, gold_label) in enumerate(traj, start=1):
                q_t = tracker.q_hat_for_step(t_idx)
                cset = conformal_prediction_set(
                    dist,
                    q_t,
                    method=conformal_method,
                    lam_reg=lam_reg,
                    k_reg=k_reg,
                )
                if str(gold_label) not in cset.prediction_set:
                    all_steps_covered = False
                    break
            if all_steps_covered:
                valid_trajectories += 1
        if valid_trajectories >= required_valid:
            return candidate_slack
    raise SchemaContractViolationError(
        f"no positive markov_slack_factor achieved {required_valid}/{m_traj} valid trajectories "
        f"at alpha_total={alpha_total}"
    )


@dataclass(frozen=True)
class NEVCInterventionDecision:
    """Decision emitted by the calibrated Net Expected Value of Control (NEVC) gate."""

    should_intervene: bool
    raw_fail_probability: float
    calibrated_fail_probability: float
    net_expected_value: float
    min_fail_probability_threshold: float
    reason: str


@dataclass(frozen=True)
class PlaceboControllerComparison:
    """Empirical comparison of NEVC vs. uncalibrated controller and frequency-matched placebo."""

    n_items: int
    no_control_accuracy: float
    uncalibrated_accuracy: float
    uncalibrated_intervention_rate: float
    uncalibrated_net_value: float
    nevc_accuracy: float
    nevc_intervention_rate: float
    nevc_net_value: float
    placebo_accuracy: float
    placebo_intervention_rate: float
    placebo_net_value: float
    nevc_vs_placebo_net_gain: float
    mcnemar_nevc_vs_placebo_p: float


def _logit(prob: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, float(prob)))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(logit_val: float) -> float:
    if logit_val >= 0.0:
        exp_neg = math.exp(-min(60.0, logit_val))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(-60.0, logit_val))
    return exp_pos / (1.0 + exp_pos)


class NetExpectedValueController:
    """Platt-calibrated expected-value gate preventing wasteful agent loop interventions."""

    def __init__(
        self,
        *,
        task_value: float = 1.0,
        intervention_cost: float = 0.08,
        fix_probability_if_failing: float = 0.45,
        regression_probability_if_passing: float = 0.04,
        platt_slope: float = 1.0,
        platt_intercept: float = 0.0,
    ) -> None:
        """Initialize an NEVC controller with explicit cost, fix-rate, and regression parameters.

        Args:
            task_value: Utility value `V_task > 0` of a successful task completion.
            intervention_cost: Per-intervention compute/latency cost `C_turn >= 0` in `V_task` units.
            fix_probability_if_failing: Conditional probability `Delta P(fix | fail)` in `(0, 1]`.
            regression_probability_if_passing: Probability `P(regress | already_passing)` in `[0, 1)`.
            platt_slope: Monotonic logistic calibration slope (`> 0`).
            platt_intercept: Logistic calibration intercept shifting base-rate log-odds.
        """
        if (
            isinstance(task_value, bool)
            or not isinstance(task_value, (int, float))
            or not math.isfinite(float(task_value))
            or float(task_value) <= 0.0
        ):
            raise SchemaContractViolationError("task_value must be finite and > 0")
        if (
            isinstance(intervention_cost, bool)
            or not isinstance(intervention_cost, (int, float))
            or not math.isfinite(float(intervention_cost))
            or float(intervention_cost) < 0.0
        ):
            raise SchemaContractViolationError("intervention_cost must be finite and >= 0")
        if (
            isinstance(fix_probability_if_failing, bool)
            or not isinstance(fix_probability_if_failing, (int, float))
            or not math.isfinite(float(fix_probability_if_failing))
            or float(fix_probability_if_failing) <= 0.0
            or float(fix_probability_if_failing) > 1.0
        ):
            raise SchemaContractViolationError("fix_probability_if_failing must lie in (0, 1]")
        if (
            isinstance(regression_probability_if_passing, bool)
            or not isinstance(regression_probability_if_passing, (int, float))
            or not math.isfinite(float(regression_probability_if_passing))
            or float(regression_probability_if_passing) < 0.0
            or float(regression_probability_if_passing) >= 1.0
        ):
            raise SchemaContractViolationError(
                "regression_probability_if_passing must lie in [0, 1)"
            )
        if (
            isinstance(platt_slope, bool)
            or not isinstance(platt_slope, (int, float))
            or not math.isfinite(float(platt_slope))
            or float(platt_slope) <= 0.0
        ):
            raise SchemaContractViolationError("platt_slope must be finite and > 0")
        if (
            isinstance(platt_intercept, bool)
            or not isinstance(platt_intercept, (int, float))
            or not math.isfinite(float(platt_intercept))
        ):
            raise SchemaContractViolationError("platt_intercept must be finite")

        self.task_value = float(task_value)
        self.intervention_cost = float(intervention_cost)
        self.fix_probability_if_failing = float(fix_probability_if_failing)
        self.regression_probability_if_passing = float(regression_probability_if_passing)
        self.platt_slope = float(platt_slope)
        self.platt_intercept = float(platt_intercept)

    @classmethod
    def fit_from_validation_traces(
        cls,
        raw_fail_probabilities: Sequence[float],
        actual_failed_flags: Sequence[bool],
        *,
        task_value: float = 1.0,
        intervention_cost: float = 0.08,
        fix_probability_if_failing: float = 0.45,
        regression_probability_if_passing: float = 0.04,
    ) -> NetExpectedValueController:
        """Fit monotonic Platt calibration `(slope >= 0.05, intercept)` via damped Newton-Raphson.

        Args:
            raw_fail_probabilities: Uncalibrated decision-model failure probabilities in `[0, 1]`.
            actual_failed_flags: Ground-truth boolean failure outcomes (`True` if attempt failed).
            task_value: Utility value of successful completion.
            intervention_cost: Cost of an extra intervention turn.
            fix_probability_if_failing: Probability intervention fixes an actually failing state.
            regression_probability_if_passing: Probability intervention breaks a passing state.

        Returns:
            Calibrated `NetExpectedValueController` instance.
        """
        if (
            not raw_fail_probabilities
            or len(raw_fail_probabilities) != len(actual_failed_flags)
        ):
            raise SchemaContractViolationError(
                "raw_fail_probabilities and actual_failed_flags must be equal-length non-empty sequences"
            )
        logits: list[float] = []
        for idx, prob in enumerate(raw_fail_probabilities):
            if (
                isinstance(prob, bool)
                or not isinstance(prob, (int, float))
                or not math.isfinite(float(prob))
                or float(prob) < 0.0
                or float(prob) > 1.0
            ):
                raise SchemaContractViolationError(
                    f"raw_fail_probabilities[{idx}] must lie in [0, 1], got {prob!r}"
                )
            logits.append(_logit(float(prob)))
        for idx, flag in enumerate(actual_failed_flags):
            if not isinstance(flag, bool):
                raise SchemaContractViolationError(
                    f"actual_failed_flags[{idx}] must be a bool, got {flag!r}"
                )
        # Platt (1999) finite-sample target smoothing prevents infinite intercepts on small slices.
        n_pos = sum(1 for flag in actual_failed_flags if flag)
        n_neg = len(actual_failed_flags) - n_pos
        pos_target = (float(n_pos) + 1.0) / (float(n_pos) + 2.0)
        neg_target = 1.0 / (float(n_neg) + 2.0)
        targets = [pos_target if flag else neg_target for flag in actual_failed_flags]

        slope = 1.0
        intercept = 0.0
        l2_reg = 1e-3
        for _ in range(40):
            grad_s = l2_reg * (slope - 1.0)
            grad_b = l2_reg * intercept
            hess_ss = l2_reg
            hess_sb = 0.0
            hess_bb = l2_reg
            for z_i, y_i in zip(logits, targets):
                pred_i = _sigmoid(slope * z_i + intercept)
                diff_i = pred_i - y_i
                weight_i = max(1e-6, pred_i * (1.0 - pred_i))
                grad_s += diff_i * z_i
                grad_b += diff_i
                hess_ss += weight_i * z_i * z_i
                hess_sb += weight_i * z_i
                hess_bb += weight_i
            det = hess_ss * hess_bb - hess_sb * hess_sb
            if det <= 1e-12:
                break
            step_s = (hess_bb * grad_s - hess_sb * grad_b) / det
            step_b = (-hess_sb * grad_s + hess_ss * grad_b) / det
            slope = max(0.05, min(10.0, slope - 0.5 * step_s))
            intercept = max(-10.0, min(10.0, intercept - 0.5 * step_b))
            if abs(step_s) < 1e-6 and abs(step_b) < 1e-6:
                break

        return cls(
            task_value=task_value,
            intervention_cost=intervention_cost,
            fix_probability_if_failing=fix_probability_if_failing,
            regression_probability_if_passing=regression_probability_if_passing,
            platt_slope=slope,
            platt_intercept=intercept,
        )

    def calibrate_fail_probability(self, raw_fail_probability: float) -> float:
        """Map raw failure probability `1 - p_complete` to calibrated failure probability."""
        if (
            isinstance(raw_fail_probability, bool)
            or not isinstance(raw_fail_probability, (int, float))
            or not math.isfinite(float(raw_fail_probability))
            or float(raw_fail_probability) < 0.0
            or float(raw_fail_probability) > 1.0
        ):
            raise SchemaContractViolationError(
                f"raw_fail_probability must lie in [0, 1], got {raw_fail_probability!r}"
            )
        return _sigmoid(self.platt_slope * _logit(float(raw_fail_probability)) + self.platt_intercept)

    def min_fail_probability_threshold(self) -> float:
        """Return exact calibrated failure probability cutoff above which `NEVC > 0`."""
        denom = self.task_value * (
            self.fix_probability_if_failing + self.regression_probability_if_passing
        )
        numer = (
            self.intervention_cost
            + self.regression_probability_if_passing * self.task_value
        )
        return min(1.0, max(0.0, numer / denom))

    def evaluate(self, raw_fail_probability: float) -> NEVCInterventionDecision:
        """Evaluate whether triggering a corrective agent turn has strictly positive expected value.

        Args:
            raw_fail_probability: Raw decision-model failure probability `1 - p_complete` in `[0, 1]`.

        Returns:
            `NEVCInterventionDecision` with calibrated probability, net value, and boolean verdict.
        """
        cal_fail = self.calibrate_fail_probability(raw_fail_probability)
        expected_gain = cal_fail * self.fix_probability_if_failing * self.task_value
        expected_regression_loss = (
            (1.0 - cal_fail) * self.regression_probability_if_passing * self.task_value
        )
        net_value = expected_gain - self.intervention_cost - expected_regression_loss
        threshold = self.min_fail_probability_threshold()
        intervene = net_value > 0.0
        reason = (
            f"positive_nevc:{net_value:.4f}"
            if intervene
            else f"negative_or_zero_nevc:{net_value:.4f}"
        )
        return NEVCInterventionDecision(
            should_intervene=intervene,
            raw_fail_probability=round(float(raw_fail_probability), 6),
            calibrated_fail_probability=round(cal_fail, 6),
            net_expected_value=round(net_value, 6),
            min_fail_probability_threshold=round(threshold, 6),
            reason=reason,
        )


def evaluate_placebo_controller_baseline(
    raw_fail_probabilities: Sequence[float],
    no_control_passed: Sequence[bool],
    post_intervention_passed: Sequence[bool],
    controller: NetExpectedValueController,
    *,
    uncalibrated_fail_threshold: float = 0.50,
    seed_key: str = DEFAULT_CONFORMAL_SEED_KEY,
) -> PlaceboControllerComparison:
    """Benchmark `NEVC` against an uncalibrated threshold controller and frequency-matched placebo.

    Implements the causal placebo evaluation protocol of Vats (arXiv:2610.24649): a controller
    adds genuine value only if its net utility `V_task * pass - C_turn * intervened` beats a
    deterministic HMAC-seeded placebo firing at the exact same intervention rate.

    Args:
        raw_fail_probabilities: Raw failure probabilities `1 - p_complete` per evaluation attempt.
        no_control_passed: Whether the agent attempt succeeded without controller intervention.
        post_intervention_passed: Whether the attempt succeeded when given an extra controller turn.
        controller: Fitted `NetExpectedValueController` under evaluation.
        uncalibrated_fail_threshold: Naive raw failure threshold (default `0.50` per Vats 2026).
        seed_key: Deterministic HMAC seed key for reproducible placebo tie-breaking.

    Returns:
        `PlaceboControllerComparison` summarizing accuracy, cost, net value, and McNemar p-value.
    """
    total_n = len(raw_fail_probabilities)
    if (
        total_n == 0
        or len(no_control_passed) != total_n
        or len(post_intervention_passed) != total_n
    ):
        raise SchemaContractViolationError(
            "raw_fail_probabilities, no_control_passed, and post_intervention_passed must match length"
        )
    if (
        isinstance(uncalibrated_fail_threshold, bool)
        or not isinstance(uncalibrated_fail_threshold, (int, float))
        or not math.isfinite(float(uncalibrated_fail_threshold))
        or float(uncalibrated_fail_threshold) < 0.0
        or float(uncalibrated_fail_threshold) > 1.0
    ):
        raise SchemaContractViolationError("uncalibrated_fail_threshold must lie in [0, 1]")
    for idx, (base_ok, interv_ok) in enumerate(zip(no_control_passed, post_intervention_passed)):
        if not isinstance(base_ok, bool) or not isinstance(interv_ok, bool):
            raise SchemaContractViolationError(
                f"outcome flags at index {idx} must be booleans"
            )

    nevc_flags = [controller.evaluate(prob).should_intervene for prob in raw_fail_probabilities]
    uncal_flags = [float(prob) >= float(uncalibrated_fail_threshold) for prob in raw_fail_probabilities]
    nevc_rate = sum(1 for flag in nevc_flags if flag) / float(total_n)

    # Deterministic HMAC-ranked frequency-matched placebo with identical intervention count.
    k_intervene = sum(1 for flag in nevc_flags if flag)
    ranked_indices = sorted(
        range(total_n),
        key=lambda idx: hmac_uniform_float64(seed_key, f"placebo:{idx}", "u"),
    )
    placebo_set = set(ranked_indices[:k_intervene])
    placebo_flags = [idx in placebo_set for idx in range(total_n)]

    def _summarize_arm(flags: Sequence[bool]) -> tuple[list[bool], float, float, float]:
        outcomes: list[bool] = []
        net_sum = 0.0
        interventions = 0
        for flag, base_ok, interv_ok in zip(flags, no_control_passed, post_intervention_passed):
            if flag:
                interventions += 1
                final_ok = bool(interv_ok)
                cost = controller.intervention_cost
            else:
                final_ok = bool(base_ok)
                cost = 0.0
            outcomes.append(final_ok)
            net_sum += (controller.task_value if final_ok else 0.0) - cost
        acc = sum(1 for ok in outcomes if ok) / float(total_n)
        return outcomes, acc, float(interventions) / float(total_n), net_sum / float(total_n)

    no_ctrl_acc = sum(1 for ok in no_control_passed if ok) / float(total_n)
    _, uncal_acc, uncal_rate, uncal_net = _summarize_arm(uncal_flags)
    nevc_outcomes, nevc_acc, _, nevc_net = _summarize_arm(nevc_flags)
    placebo_outcomes, placebo_acc, placebo_rate, placebo_net = _summarize_arm(placebo_flags)
    mcnemar_res = paired_mcnemar_test(nevc_outcomes, placebo_outcomes)

    return PlaceboControllerComparison(
        n_items=total_n,
        no_control_accuracy=round(no_ctrl_acc, 4),
        uncalibrated_accuracy=round(uncal_acc, 4),
        uncalibrated_intervention_rate=round(uncal_rate, 4),
        uncalibrated_net_value=round(uncal_net, 4),
        nevc_accuracy=round(nevc_acc, 4),
        nevc_intervention_rate=round(nevc_rate, 4),
        nevc_net_value=round(nevc_net, 4),
        placebo_accuracy=round(placebo_acc, 4),
        placebo_intervention_rate=round(placebo_rate, 4),
        placebo_net_value=round(placebo_net, 4),
        nevc_vs_placebo_net_gain=round(nevc_net - placebo_net, 4),
        mcnemar_nevc_vs_placebo_p=round(float(mcnemar_res.p_two_sided), 6),
    )

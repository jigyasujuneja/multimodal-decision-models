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

"""Online Primal-Dual Budget-Paced Contextual & Dueling Bandit Router (ParetoBandit).

Implements online non-stationary multi-LLM routing grounded in:
  - Discounted Contextual LinUCB with geometric forgetting factor rho in (0, 1]
    (arXiv:2604.00136, ParetoBandit: Multi-Objective Non-Stationary LLM Routing)
  - Online Primal-Dual Lagrangian budget pacing:
        lambda_{t+1} = max(0, lambda_t + eta_lambda * (c(a_t) - B_max))
  - Pairwise Bradley-Terry Dueling Bandit preference updates (arXiv:2510.00841)
    for delayed or weak relative feedback when ground-truth labels are absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelArmSpec:
    """Cost and latency specification for a candidate decision or reasoning model arm."""

    arm_id: str
    cost_usd: float
    latency_ms: float


@dataclass(frozen=True)
class BanditRoutingDecision:
    """Single-step routing decision emitted by PrimalDualParetoBanditRouter."""

    chosen_arm: str
    ucb_quality_score: float
    primal_dual_acquisition: float
    dual_price_lambda: float
    dual_latency_mu: float
    arm_ucb_scores: dict[str, float]


@dataclass(frozen=True)
class BanditStreamSummary:
    """Aggregate performance, cost compliance, and regret metrics across a routed stream."""

    mean_quality: float
    mean_cost_usd: float
    mean_latency_ms: float
    cumulative_regret: float
    budget_violation_ratio: float
    final_dual_lambda: float
    final_dual_mu: float
    arm_pull_counts: dict[str, int]


def _validate_feature_vector(context: Sequence[float]) -> list[float]:
    if not context:
        raise ValueError("context feature vector must be non-empty")
    vec: list[float] = []
    for idx, val in enumerate(context):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"context[{idx}] must be a finite number, got {val!r}")
        vec.append(float(val))
    return vec


def _solve_spd_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve A x = b for symmetric positive-definite A via Gaussian elimination with partial pivoting."""
    d = len(rhs)
    aug: list[list[float]] = [
        [float(matrix[i][j]) for j in range(d)] + [float(rhs[i])]
        for i in range(d)
    ]
    for col in range(d):
        pivot_row = max(range(col, d), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            aug[col][col] += 1e-6
            pivot_row = col
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        piv = aug[col][col]
        for j in range(col, d + 1):
            aug[col][j] /= piv
        for r in range(d):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) > 1e-15:
                for j in range(col, d + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][d] for i in range(d)]


def _quadratic_form_with_inverse(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> float:
    """Compute v^T A^{-1} v >= 0 by solving A z = v and returning v^T z."""
    z = _solve_spd_linear_system(matrix, vec)
    val = sum(float(vec[i]) * z[i] for i in range(len(vec)))
    return max(0.0, val)


class PrimalDualParetoBanditRouter:
    """Online Primal-Dual Contextual & Dueling Bandit Router with geometric forgetting.

    Maintains per-arm discounted precision matrix A_{m,t} and reward-weighted context vector b_{m,t}:
        A_{m,t} = rho * A_{m,t-1} + (1 - rho) * ridge_lambda * I + phi_t phi_t^T
        b_{m,t} = rho * b_{m,t-1} + r_t * phi_t
    Selects arm maximizing:
        UCB_m(phi_t) + tau_pref * w_{pref,m}^T phi_t - lambda_t * c_m - mu_t * (l_m / 1000)
    and updates dual multipliers via projected primal-dual subgradient ascent.
    """

    def __init__(
        self,
        arms: Sequence[ModelArmSpec],
        context_dim: int,
        *,
        budget_per_query_usd: float = 0.005,
        latency_sla_ms: float | None = None,
        alpha_ucb: float = 0.35,
        rho_discount: float = 0.98,
        ridge_lambda: float = 1.0,
        eta_dual_cost: float = 2.5,
        eta_dual_latency: float = 0.05,
        dueling_lr: float = 0.15,
        dueling_weight: float = 0.20,
    ) -> None:
        if not arms:
            raise ValueError("arms must be a non-empty sequence of ModelArmSpec")
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 1:
            raise ValueError(f"context_dim must be an integer >= 1, got {context_dim!r}")
        if isinstance(budget_per_query_usd, bool) or not isinstance(budget_per_query_usd, (int, float)) or not math.isfinite(float(budget_per_query_usd)) or float(budget_per_query_usd) <= 0.0:
            raise ValueError(f"budget_per_query_usd must be finite and > 0, got {budget_per_query_usd!r}")
        if isinstance(alpha_ucb, bool) or not isinstance(alpha_ucb, (int, float)) or not math.isfinite(float(alpha_ucb)) or float(alpha_ucb) < 0.0:
            raise ValueError(f"alpha_ucb must be finite and >= 0, got {alpha_ucb!r}")
        if isinstance(rho_discount, bool) or not isinstance(rho_discount, (int, float)) or not math.isfinite(float(rho_discount)) or float(rho_discount) <= 0.0 or float(rho_discount) > 1.0:
            raise ValueError(f"rho_discount must lie in (0, 1], got {rho_discount!r}")
        if isinstance(ridge_lambda, bool) or not isinstance(ridge_lambda, (int, float)) or not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) <= 0.0:
            raise ValueError(f"ridge_lambda must be finite and > 0, got {ridge_lambda!r}")

        self.context_dim: int = int(context_dim)
        self.budget_per_query_usd: float = float(budget_per_query_usd)
        self.latency_sla_ms: float | None = None if latency_sla_ms is None else float(latency_sla_ms)
        self.alpha_ucb: float = float(alpha_ucb)
        self.rho_discount: float = float(rho_discount)
        self.ridge_lambda: float = float(ridge_lambda)
        self.eta_dual_cost: float = float(eta_dual_cost)
        self.eta_dual_latency: float = float(eta_dual_latency)
        self.dueling_lr: float = float(dueling_lr)
        self.dueling_weight: float = float(dueling_weight)

        self.arms: dict[str, ModelArmSpec] = {}
        for spec in arms:
            if not spec.arm_id or not spec.arm_id.strip():
                raise ValueError("ModelArmSpec.arm_id must be non-empty")
            if not math.isfinite(spec.cost_usd) or spec.cost_usd < 0.0:
                raise ValueError(f"cost_usd for {spec.arm_id!r} must be >= 0")
            if not math.isfinite(spec.latency_ms) or spec.latency_ms < 0.0:
                raise ValueError(f"latency_ms for {spec.arm_id!r} must be >= 0")
            self.arms[spec.arm_id] = spec

        self.arm_order: tuple[str, ...] = tuple(sorted(self.arms.keys()))
        self.dual_lambda: float = 0.0
        self.dual_mu: float = 0.0

        self._cov_a: dict[str, list[list[float]]] = {}
        self._vec_b: dict[str, list[float]] = {}
        self._duel_w: dict[str, list[float]] = {}
        self.pull_counts: dict[str, int] = {}

        for arm_id in self.arm_order:
            self._cov_a[arm_id] = [
                [self.ridge_lambda if i == j else 0.0 for j in range(self.context_dim)]
                for i in range(self.context_dim)
            ]
            self._vec_b[arm_id] = [0.0 for _ in range(self.context_dim)]
            self._duel_w[arm_id] = [0.0 for _ in range(self.context_dim)]
            self.pull_counts[arm_id] = 0

    def select_arm(
        self,
        context: Sequence[float],
        *,
        available_arms: Sequence[str] | None = None,
    ) -> BanditRoutingDecision:
        """Select the Pareto-optimal model arm for context feature vector phi_t."""
        phi = _validate_feature_vector(context)
        if len(phi) != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {len(phi)}")

        candidates = list(self.arm_order) if available_arms is None else [str(a) for a in available_arms]
        if not candidates:
            raise ValueError("available_arms must be non-empty")

        ucb_scores: dict[str, float] = {}
        acq_scores: dict[str, float] = {}

        for arm_id in candidates:
            if arm_id not in self.arms:
                raise ValueError(f"unknown arm {arm_id!r}")
            spec = self.arms[arm_id]
            mat_a = self._cov_a[arm_id]
            vec_b = self._vec_b[arm_id]
            theta_hat = _solve_spd_linear_system(mat_a, vec_b)
            mean_est = sum(theta_hat[i] * phi[i] for i in range(self.context_dim))
            bonus = self.alpha_ucb * math.sqrt(_quadratic_form_with_inverse(mat_a, phi))
            duel_term = self.dueling_weight * sum(self._duel_w[arm_id][i] * phi[i] for i in range(self.context_dim))
            ucb_val = mean_est + bonus + duel_term
            ucb_scores[arm_id] = ucb_val

            cost_penalty = self.dual_lambda * spec.cost_usd
            lat_penalty = 0.0
            if self.latency_sla_ms is not None:
                lat_penalty = self.dual_mu * (spec.latency_ms / 1000.0)
            acq_scores[arm_id] = ucb_val - cost_penalty - lat_penalty

        best_arm = sorted(candidates, key=lambda a: (-acq_scores[a], a))[0]
        return BanditRoutingDecision(
            chosen_arm=best_arm,
            ucb_quality_score=ucb_scores[best_arm],
            primal_dual_acquisition=acq_scores[best_arm],
            dual_price_lambda=self.dual_lambda,
            dual_latency_mu=self.dual_mu,
            arm_ucb_scores=ucb_scores,
        )

    def update(
        self,
        chosen_arm: str,
        context: Sequence[float],
        reward: float,
        *,
        incurred_cost_usd: float | None = None,
        incurred_latency_ms: float | None = None,
    ) -> float:
        """Update discounted LinUCB sufficient statistics and primal-dual Lagrangian multipliers."""
        if chosen_arm not in self.arms:
            raise ValueError(f"unknown chosen_arm {chosen_arm!r}")
        phi = _validate_feature_vector(context)
        if len(phi) != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {len(phi)}")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError(f"reward must be a finite number, got {reward!r}")

        r_val = float(reward)
        spec = self.arms[chosen_arm]
        cost_t = spec.cost_usd if incurred_cost_usd is None else float(incurred_cost_usd)
        lat_t = spec.latency_ms if incurred_latency_ms is None else float(incurred_latency_ms)

        # Apply geometric forgetting factor rho across all arms to track non-stationary drift
        rho = self.rho_discount
        prior_diag = (1.0 - rho) * self.ridge_lambda
        for arm_id in self.arm_order:
            mat_a = self._cov_a[arm_id]
            vec_b = self._vec_b[arm_id]
            for i in range(self.context_dim):
                vec_b[i] *= rho
                for j in range(self.context_dim):
                    mat_a[i][j] = rho * mat_a[i][j] + (prior_diag if i == j else 0.0)

        # Rank-1 precision and reward update for chosen arm
        chosen_a = self._cov_a[chosen_arm]
        chosen_b = self._vec_b[chosen_arm]
        for i in range(self.context_dim):
            chosen_b[i] += r_val * phi[i]
            for j in range(self.context_dim):
                chosen_a[i][j] += phi[i] * phi[j]

        self.pull_counts[chosen_arm] += 1

        # Online Primal-Dual Lagrangian multiplier update
        self.dual_lambda = max(
            0.0,
            self.dual_lambda + self.eta_dual_cost * (cost_t - self.budget_per_query_usd),
        )
        if self.latency_sla_ms is not None:
            self.dual_mu = max(
                0.0,
                self.dual_mu + self.eta_dual_latency * ((lat_t - self.latency_sla_ms) / 1000.0),
            )
        return self.dual_lambda

    def update_dueling_preference(
        self,
        winner_arm: str,
        loser_arm: str,
        context: Sequence[float],
    ) -> None:
        """Update pairwise Bradley-Terry contextual preference weights from a dueling comparison."""
        if winner_arm not in self.arms or loser_arm not in self.arms:
            raise ValueError(f"both winner_arm ({winner_arm!r}) and loser_arm ({loser_arm!r}) must be valid arms")
        phi = _validate_feature_vector(context)
        if len(phi) != self.context_dim:
            raise ValueError(f"expected context_dim={self.context_dim}, got {len(phi)}")
        if winner_arm == loser_arm:
            return

        w_win = self._duel_w[winner_arm]
        w_lose = self._duel_w[loser_arm]
        logit_diff = sum((w_win[i] - w_lose[i]) * phi[i] for i in range(self.context_dim))
        logit_clamped = min(20.0, max(-20.0, logit_diff))
        prob_win = 1.0 / (1.0 + math.exp(-logit_clamped))
        grad_scale = self.dueling_lr * (1.0 - prob_win)

        for i in range(self.context_dim):
            w_win[i] = self.rho_discount * w_win[i] + grad_scale * phi[i]
            w_lose[i] = self.rho_discount * w_lose[i] - grad_scale * phi[i]


def evaluate_pareto_bandit_stream(
    arms: Sequence[ModelArmSpec],
    stream_contexts: Sequence[Sequence[float]],
    stream_arm_rewards: Sequence[Mapping[str, float]],
    *,
    budget_per_query_usd: float = 0.005,
    latency_sla_ms: float | None = None,
    alpha_ucb: float = 0.35,
    rho_discount: float = 0.98,
    ridge_lambda: float = 1.0,
    eta_dual_cost: float = 2.5,
    dueling_pairs: Sequence[tuple[str, str] | None] | None = None,
) -> BanditStreamSummary:
    """Evaluate PrimalDualParetoBanditRouter over a stream of contexts and per-arm reward realizations."""
    if not stream_contexts or len(stream_contexts) != len(stream_arm_rewards):
        raise ValueError("stream_contexts and stream_arm_rewards must be non-empty sequences of equal length")
    if dueling_pairs is not None and len(dueling_pairs) != len(stream_contexts):
        raise ValueError("dueling_pairs must have the same length as stream_contexts when provided")

    context_dim = len(stream_contexts[0])
    router = PrimalDualParetoBanditRouter(
        arms=arms,
        context_dim=context_dim,
        budget_per_query_usd=budget_per_query_usd,
        latency_sla_ms=latency_sla_ms,
        alpha_ucb=alpha_ucb,
        rho_discount=rho_discount,
        ridge_lambda=ridge_lambda,
        eta_dual_cost=eta_dual_cost,
    )

    arm_specs = {a.arm_id: a for a in arms}
    feasible_arms = [a.arm_id for a in arms if a.cost_usd <= budget_per_query_usd * 1.5]
    if not feasible_arms:
        feasible_arms = [a.arm_id for a in arms]

    total_quality = 0.0
    total_cost = 0.0
    total_latency = 0.0
    cumulative_regret = 0.0

    for idx, (ctx, reward_map) in enumerate(zip(stream_contexts, stream_arm_rewards)):
        decision = router.select_arm(ctx)
        chosen = decision.chosen_arm
        if chosen not in reward_map:
            raise ValueError(f"chosen arm {chosen!r} missing from stream_arm_rewards at step {idx}")
        r_chosen = float(reward_map[chosen])
        spec = arm_specs[chosen]

        # Best budget-feasible oracle quality on this step
        best_feasible_r = max(float(reward_map[a]) for a in feasible_arms if a in reward_map)
        cumulative_regret += max(0.0, best_feasible_r - r_chosen)

        total_quality += r_chosen
        total_cost += spec.cost_usd
        total_latency += spec.latency_ms

        router.update(chosen, ctx, r_chosen)
        if dueling_pairs is not None and dueling_pairs[idx] is not None:
            pair = dueling_pairs[idx]
            if pair is not None:
                router.update_dueling_preference(pair[0], pair[1], ctx)

    n = float(len(stream_contexts))
    mean_cost = total_cost / n
    violation_ratio = max(0.0, (mean_cost - budget_per_query_usd) / budget_per_query_usd)

    return BanditStreamSummary(
        mean_quality=total_quality / n,
        mean_cost_usd=mean_cost,
        mean_latency_ms=total_latency / n,
        cumulative_regret=cumulative_regret,
        budget_violation_ratio=violation_ratio,
        final_dual_lambda=router.dual_lambda,
        final_dual_mu=router.dual_mu,
        arm_pull_counts=dict(router.pull_counts),
    )

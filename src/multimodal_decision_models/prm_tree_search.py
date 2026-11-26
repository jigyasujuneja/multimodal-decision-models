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

"""Calibrated Process Reward Verifiers (PRM) & Conformal Branch-and-Bound PUCT Tree Search.

Adapts System-1 8-slot parallel readout into a zero-generation-overhead Process Reward
Verifier scoring up to 8 candidate reasoning branches simultaneously on {0, 1, 2, 3}
rubrics, combined with:
  - Horizon-depth survival discount calibration (Lightman et al., ICLR 2024; Snell et al., 2024)
  - Conformal Branch-and-Bound PUCT tree search (Rosin, Ann. Math. Artif. Intell. 2011;
    Silver et al., Nature 2017) that prunes subtrees whenever their conformal upper bound
    falls strictly below the current incumbent lower bound: UCB_alpha(s, a) < LCB_alpha(s^*).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PRMBranchNode:
    """Candidate reasoning step node in a test-time compute tree."""

    node_id: str
    parent_id: str | None
    depth: int
    prior_prob: float
    rubric_probs: dict[str, float]
    is_terminal: bool = False
    ground_truth_correct: bool = False


@dataclass(frozen=True)
class PRMTreeSearchResult:
    """Result of Calibrated PRM + Conformal Branch-and-Bound PUCT tree search."""

    selected_leaf_id: str
    selected_path: tuple[str, ...]
    selected_value_mean: float
    selected_lcb: float
    selected_ucb: float
    total_nodes: int
    evaluated_nodes: int
    pruned_nodes: int
    pruning_ratio: float
    leaf_correct: bool


def score_prm_rubric_distribution(
    rubric_probs: Mapping[str, float],
    *,
    horizon_discount: float = 1.0,
) -> tuple[float, float]:
    """Compute calibrated step value expectation E[V] in [0, 1] and epistemic std sigma[V].

    Expects rubric keys in {"0", "1", "2", "3"} (or non-negative integer strings) normalized
    by maximum rubric grade G_max (default 3.0).
    """
    if not rubric_probs:
        raise ValueError("rubric_probs must be non-empty")
    if isinstance(horizon_discount, bool) or not isinstance(horizon_discount, (int, float)) or not math.isfinite(float(horizon_discount)) or float(horizon_discount) <= 0.0:
        raise ValueError(f"horizon_discount must be finite and > 0, got {horizon_discount!r}")

    total_p = 0.0
    parsed: list[tuple[float, float]] = []
    max_grade = 3.0
    for raw_k, raw_v in rubric_probs.items():
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)) or not math.isfinite(float(raw_v)) or float(raw_v) < 0.0:
            raise ValueError(f"invalid probability {raw_v!r} for rubric key {raw_k!r}")
        try:
            grade = float(int(str(raw_k).strip()))
        except ValueError as exc:
            raise ValueError(f"rubric key {raw_k!r} must be an integer string in '0'..'3'") from exc
        if grade < 0.0:
            raise ValueError(f"rubric grade must be >= 0, got {grade}")
        max_grade = max(max_grade, grade)
        p_val = float(raw_v)
        parsed.append((grade, p_val))
        total_p += p_val

    if total_p <= 0.0:
        raise ValueError("rubric_probs sum must be > 0")

    mean_norm = 0.0
    second_moment = 0.0
    for grade, p_val in parsed:
        p_norm = p_val / total_p
        g_unit = grade / max_grade
        mean_norm += p_norm * g_unit
        second_moment += p_norm * (g_unit * g_unit)

    var_unit = max(0.0, second_moment - (mean_norm * mean_norm))
    disc = min(1.5, max(0.05, float(horizon_discount)))
    calibrated_mean = min(1.0, max(0.0, mean_norm * disc))
    calibrated_std = math.sqrt(var_unit)
    return calibrated_mean, calibrated_std


def calibrate_prm_horizon_discount(
    calib_depths: Sequence[int],
    calib_rubric_probs: Sequence[Mapping[str, float]],
    calib_outcomes: Sequence[bool],
    *,
    max_depth: int = 6,
) -> dict[int, float]:
    """Calibrate per-depth survival discount factors gamma_d minimizing step-value Brier error.

    Counteracts step-level overconfidence on deep reasoning chains (Lightman et al., ICLR 2024).
    """
    if not calib_depths or len(calib_depths) != len(calib_rubric_probs) or len(calib_depths) != len(calib_outcomes):
        raise ValueError("calib_depths, calib_rubric_probs, and calib_outcomes must be non-empty sequences of equal length")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"max_depth must be an integer >= 1, got {max_depth!r}")

    num_by_depth: dict[int, float] = {}
    den_by_depth: dict[int, float] = {}

    for depth_raw, probs, outcome in zip(calib_depths, calib_rubric_probs, calib_outcomes):
        d = max(1, min(max_depth, int(depth_raw)))
        raw_v, _ = score_prm_rubric_distribution(probs, horizon_discount=1.0)
        y = 1.0 if bool(outcome) else 0.0
        num_by_depth[d] = num_by_depth.get(d, 0.0) + (raw_v * y)
        den_by_depth[d] = den_by_depth.get(d, 0.0) + (raw_v * raw_v)

    discounts: dict[int, float] = {}
    running_prev = 1.0
    for d in range(1, max_depth + 1):
        den = den_by_depth.get(d, 0.0)
        if den > 1e-9:
            ratio = num_by_depth[d] / den
            gamma_d = min(1.15, max(0.25, ratio))
        else:
            gamma_d = running_prev * 0.96
        discounts[d] = round(gamma_d, 6)
        running_prev = discounts[d]
    return discounts


def calibrate_prm_conformal_residual_quantile(
    calib_predicted_means: Sequence[float],
    calib_outcomes: Sequence[bool | float],
    *,
    alpha: float = 0.10,
) -> float:
    """Calibrate finite-sample split-conformal absolute residual quantile q_hat_{1-alpha}.

    Computes the order statistic at index ceil((n + 1) * (1 - alpha)) - 1 over held-out
    absolute step-value residuals |Y_i - V_hat_i| in [0, 1].

    Args:
        calib_predicted_means: Predicted step values V_hat_i in [0, 1].
        calib_outcomes: Realized binary or continuous outcomes Y_i in [0, 1].
        alpha: Miscoverage target in (0, 1).

    Returns:
        Finite-sample split-conformal residual quantile q_hat in [0, 1].

    Raises:
        ValueError: If inputs are empty, mismatched in length, or out of range.
    """
    if not calib_predicted_means or len(calib_predicted_means) != len(calib_outcomes):
        raise ValueError(
            "calib_predicted_means and calib_outcomes must be non-empty sequences of equal length"
        )
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")

    residuals: list[float] = []
    for pred_raw, out_raw in zip(calib_predicted_means, calib_outcomes):
        if isinstance(pred_raw, bool) or not isinstance(pred_raw, (int, float)) or not math.isfinite(float(pred_raw)):
            raise ValueError(f"predicted mean must be finite, got {pred_raw!r}")
        pred_val = float(pred_raw)
        if not (0.0 <= pred_val <= 1.0):
            raise ValueError(f"predicted mean must lie in [0, 1], got {pred_val}")
        y_val = (1.0 if out_raw else 0.0) if isinstance(out_raw, bool) else float(out_raw)
        if not math.isfinite(y_val) or not (0.0 <= y_val <= 1.0):
            raise ValueError(f"outcome must lie in [0, 1], got {out_raw!r}")
        residuals.append(abs(y_val - pred_val))

    residuals.sort()
    num_samples = len(residuals)
    rank = max(1, math.ceil((num_samples + 1) * (1.0 - float(alpha)) - 1e-12))
    if rank > num_samples:
        return float("inf")
    return float(residuals[rank - 1])


def run_calibrated_prm_puct_search(
    nodes: Sequence[PRMBranchNode],
    *,
    conformal_z: float = 1.28155,
    conformal_q_hat: float | None = None,
    c_puct: float = 0.75,
    horizon_discounts: Mapping[int, float] | None = None,
    enable_conformal_pruning: bool = True,
) -> PRMTreeSearchResult:
    """Execute PUCT + Conformal Branch-and-Bound tree search over a candidate reasoning tree.

    Scores candidate branches in batched 8-slot verifier passes and prunes any branch (s, a)
    whose conformal upper bound UCB_alpha(s, a) = V_hat(s, a) + half_width(s, a)
    is strictly below the best incumbent lower bound LCB_alpha(s^*) = V_hat(s^*) - half_width(s^*),
    where half_width combines empirical rubric standard error z_alpha * sigma_hat(s, a) with an
    optional split-conformal residual quantile q_hat_{1-alpha} / sqrt(steps).
    """
    if not nodes:
        raise ValueError("nodes must be non-empty")
    if isinstance(conformal_z, bool) or not isinstance(conformal_z, (int, float)) or not math.isfinite(float(conformal_z)) or float(conformal_z) < 0.0:
        raise ValueError(f"conformal_z must be finite and >= 0, got {conformal_z!r}")
    if conformal_q_hat is not None:
        if (
            isinstance(conformal_q_hat, bool)
            or not isinstance(conformal_q_hat, (int, float))
            or math.isnan(float(conformal_q_hat))
            or (
                math.isfinite(float(conformal_q_hat))
                and not (0.0 <= float(conformal_q_hat) <= 1.0)
            )
            or (math.isinf(float(conformal_q_hat)) and float(conformal_q_hat) < 0.0)
        ):
            raise ValueError(f"conformal_q_hat must lie in [0, 1] or be +inf, got {conformal_q_hat!r}")
    if isinstance(c_puct, bool) or not isinstance(c_puct, (int, float)) or not math.isfinite(float(c_puct)) or float(c_puct) < 0.0:
        raise ValueError(f"c_puct must be finite and >= 0, got {c_puct!r}")

    node_map: dict[str, PRMBranchNode] = {}
    children_map: dict[str | None, list[PRMBranchNode]] = {}
    for n in nodes:
        if n.node_id in node_map:
            raise ValueError(f"duplicate node_id {n.node_id!r}")
        node_map[n.node_id] = n
        children_map.setdefault(n.parent_id, []).append(n)

    # Reject cyclic parent_id chains before traversal.
    visit_state: dict[str, int] = {}
    for nid in node_map:
        if visit_state.get(nid, 0) != 0:
            continue
        chain: list[str] = []
        cur: str | None = nid
        while cur is not None and cur in node_map:
            st = visit_state.get(cur, 0)
            if st == 1:
                raise ValueError(f"cycle detected in reasoning tree involving node {cur!r}")
            if st == 2:
                break
            visit_state[cur] = 1
            chain.append(cur)
            cur = node_map[cur].parent_id
        for visited_id in chain:
            visit_state[visited_id] = 2

    for child_list in children_map.values():
        child_list.sort(key=lambda item: (-item.prior_prob, item.node_id))

    roots = children_map.get(None, [])
    if not roots:
        raise ValueError("reasoning tree must contain at least one root node with parent_id=None")

    z_val = float(conformal_z)
    q_hat_val = None if conformal_q_hat is None else float(conformal_q_hat)
    puct_c = float(c_puct)

    def _effective_step_discount(depth: int) -> float:
        raw_disc = 1.0 if horizon_discounts is None else float(horizon_discounts.get(depth, 1.0))
        return min(1.5, max(0.05, raw_disc))

    # Precompute bottom-up remaining-depth discount bounds on the validated acyclic tree
    # so branch-and-bound pruning is mathematically admissible for arbitrary tree depths.
    bfs_order: list[PRMBranchNode] = list(roots)
    bfs_head = 0
    while bfs_head < len(bfs_order):
        cur_node = bfs_order[bfs_head]
        bfs_head += 1
        bfs_order.extend(children_map.get(cur_node.node_id, []))

    rem_bounds: dict[str, dict[int, tuple[float, float]]] = {}
    for cur_node in reversed(bfs_order):
        ch_nodes = children_map.get(cur_node.node_id, [])
        if cur_node.is_terminal or not ch_nodes:
            rem_bounds[cur_node.node_id] = {0: (0.0, 0.0)}
            continue
        node_bounds: dict[int, tuple[float, float]] = {}
        for ch_node in ch_nodes:
            d_raw = _effective_step_discount(ch_node.depth)
            d_eff = min(1.0, d_raw)
            for r_child, (child_sum, child_max) in rem_bounds[ch_node.node_id].items():
                r_total = r_child + 1
                cand_sum = d_eff + child_sum
                cand_max = max(d_raw, child_max)
                if r_total in node_bounds:
                    prev_sum, prev_max = node_bounds[r_total]
                    node_bounds[r_total] = (max(prev_sum, cand_sum), max(prev_max, cand_max))
                else:
                    node_bounds[r_total] = (cand_sum, cand_max)
        rem_bounds[cur_node.node_id] = node_bounds

    def _bhatia_davis_leaf_ucb(
        val_sum: float,
        var_sum: float,
        steps_f: float,
        rem_steps: int,
        d_eff_sum: float,
        d_max: float,
    ) -> float:
        tot_steps = steps_f + float(rem_steps)
        if rem_steps <= 0 or d_max <= 0.0 or d_eff_sum <= 0.0:
            z_bound = (val_sum + z_val * math.sqrt(max(0.0, var_sum))) / tot_steps
        else:
            rf = float(rem_steps)
            mu_cap = min(1.0, d_eff_sum / (d_max * rf))
            mu_hi = max(0.5, mu_cap)
            cand_hi = d_eff_sum + z_val * math.sqrt(max(0.0, var_sum + rf * mu_hi * (1.0 - mu_hi)))
            denom = (z_val * z_val) + rf * (d_max * d_max)
            t_star = min(1.0, d_max * math.sqrt((4.0 * var_sum + rf) / denom)) if denom > 0.0 else 1.0
            mu_star = 0.5 * (1.0 + t_star)
            mu_lo = min(mu_cap, max(0.5, mu_star))
            cand_lo = d_max * rf * mu_lo + z_val * math.sqrt(max(0.0, var_sum + rf * mu_lo * (1.0 - mu_lo)))
            z_bound = (val_sum + max(cand_lo, cand_hi)) / tot_steps
        cand_ucb = min(1.0, z_bound)
        if q_hat_val is not None:
            mean_ub = (val_sum + d_eff_sum) / tot_steps
            cand_ucb = min(1.0, max(cand_ucb, mean_ub + q_hat_val / math.sqrt(tot_steps)))
        return cand_ucb

    def _subtree_max_ucb(node_id: str, val_sum: float, var_sum: float, steps_f: float) -> float:
        best_u = -1.0
        for r_steps, (d_eff_sum, d_max) in rem_bounds[node_id].items():
            if r_steps <= 0:
                continue
            cand_u = _bhatia_davis_leaf_ucb(val_sum, var_sum, steps_f, r_steps, d_eff_sum, d_max)
            if cand_u > best_u:
                best_u = cand_u
        return best_u

    def _child_subtree_max_ucb(ch_node: PRMBranchNode, val_sum: float, var_sum: float, steps_f: float) -> float:
        d_raw = _effective_step_discount(ch_node.depth)
        d_eff = min(1.0, d_raw)
        best_u = -1.0
        for r_child, (child_sum, child_max) in rem_bounds[ch_node.node_id].items():
            cand_u = _bhatia_davis_leaf_ucb(
                val_sum,
                var_sum,
                steps_f,
                r_child + 1,
                d_eff + child_sum,
                max(d_raw, child_max),
            )
            if cand_u > best_u:
                best_u = cand_u
        return best_u

    # Track evaluated vs pruned nodes via best-first PUCT + Conformal Branch-and-Bound traversal
    evaluated_ids: set[str] = set()
    pruned_ids: set[str] = set()

    best_leaf: PRMBranchNode | None = None
    best_leaf_path: tuple[str, ...] = ()
    best_leaf_val: float = -1.0
    best_leaf_lcb: float = -1.0
    best_leaf_ucb: float = -1.0
    best_leaf_score: float = -float("inf")

    def _count_descendants(nid: str) -> list[str]:
        desc: list[str] = [nid]
        seen_desc: set[str] = {nid}
        stack = [nid]
        while stack:
            cur = stack.pop()
            for ch in children_map.get(cur, []):
                if ch.node_id not in seen_desc:
                    seen_desc.add(ch.node_id)
                    desc.append(ch.node_id)
                    stack.append(ch.node_id)
        return desc

    def _explore(
        node: PRMBranchNode,
        path_so_far: tuple[str, ...],
        cum_val_sum: float,
        cum_var_sum: float,
        sibling_rank: int,
        sibling_count: int,
    ) -> None:
        nonlocal best_leaf, best_leaf_path, best_leaf_val, best_leaf_lcb, best_leaf_ucb, best_leaf_score

        disc = 1.0
        if horizon_discounts is not None:
            disc = float(horizon_discounts.get(node.depth, 1.0))

        step_v, step_std = score_prm_rubric_distribution(node.rubric_probs, horizon_discount=disc)
        evaluated_ids.add(node.node_id)

        steps = len(path_so_far) + 1
        new_val_sum = cum_val_sum + step_v
        new_var_sum = cum_var_sum + (step_std * step_std)
        path_mean = new_val_sum / float(steps)
        path_std = math.sqrt(new_var_sum) / float(steps)
        half_width = z_val * path_std
        if q_hat_val is not None:
            half_width = max(half_width, q_hat_val / math.sqrt(float(steps)))

        node_lcb = max(0.0, path_mean - half_width)
        node_ucb = min(1.0, path_mean + half_width)
        puct_bonus = puct_c * float(node.prior_prob) * (math.sqrt(float(sibling_count + 1)) / float(1 + sibling_rank))

        # Conformal Branch-and-Bound pruning condition: UCB_alpha(subtree) < best_incumbent_LCB
        ch_list = children_map.get(node.node_id, [])
        is_leaf = node.is_terminal or not ch_list
        if enable_conformal_pruning and (not is_leaf) and best_leaf is not None:
            if _subtree_max_ucb(node.node_id, new_val_sum, new_var_sum, float(steps)) < best_leaf_lcb:
                for ch in ch_list:
                    for d_id in _count_descendants(ch.node_id):
                        if d_id not in evaluated_ids:
                            pruned_ids.add(d_id)
                return

        cur_path = path_so_far + (node.node_id,)
        if is_leaf:
            leaf_acq = path_mean + 0.25 * node_lcb + 0.10 * puct_bonus
            if leaf_acq > best_leaf_score + 1e-12:
                best_leaf_score = leaf_acq
                best_leaf = node
                best_leaf_path = cur_path
                best_leaf_val = path_mean
                best_leaf_lcb = node_lcb
                best_leaf_ucb = node_ucb
            return

        # Order children by prior_prob + optimistic subtree UCB
        for rank_idx, ch in enumerate(ch_list):
            if enable_conformal_pruning and best_leaf is not None:
                if _child_subtree_max_ucb(ch, new_val_sum, new_var_sum, float(steps)) < best_leaf_lcb:
                    for d_id in _count_descendants(ch.node_id):
                        if d_id not in evaluated_ids:
                            pruned_ids.add(d_id)
                    continue
            _explore(ch, cur_path, new_val_sum, new_var_sum, rank_idx, len(ch_list))

    for r_idx, root in enumerate(roots):
        _explore(root, (), 0.0, 0.0, r_idx, len(roots))

    if best_leaf is None:
        # Fallback to first root if all branches were non-terminal
        first = roots[0]
        v0, s0 = score_prm_rubric_distribution(first.rubric_probs)
        hw0 = z_val * s0 if q_hat_val is None else max(z_val * s0, q_hat_val)
        best_leaf = first
        best_leaf_path = (first.node_id,)
        best_leaf_val = v0
        best_leaf_lcb = max(0.0, v0 - hw0)
        best_leaf_ucb = min(1.0, v0 + hw0)

    total_n = len(nodes)
    eval_n = len(evaluated_ids)
    pruned_n = len(pruned_ids - evaluated_ids)
    prune_ratio = float(pruned_n) / float(max(1, total_n))

    return PRMTreeSearchResult(
        selected_leaf_id=best_leaf.node_id,
        selected_path=best_leaf_path,
        selected_value_mean=best_leaf_val,
        selected_lcb=best_leaf_lcb,
        selected_ucb=best_leaf_ucb,
        total_nodes=total_n,
        evaluated_nodes=eval_n,
        pruned_nodes=pruned_n,
        pruning_ratio=prune_ratio,
        leaf_correct=bool(best_leaf.ground_truth_correct),
    )

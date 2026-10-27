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
  - Horizon-depth survival discount calibration (CondOT-PRM, 2025-2026)
  - Conformal Branch-and-Bound PUCT tree search (PATR, 2026) that prunes subtrees
    whenever their conformal upper bound falls strictly below the current incumbent
    lower bound: UCB_alpha(s, a) < LCB_alpha(s^*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


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

    Counteracts step-level overconfidence on deep reasoning chains (CondOT-PRM, 2026).
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


def run_calibrated_prm_puct_search(
    nodes: Sequence[PRMBranchNode],
    *,
    conformal_z: float = 1.28155,
    c_puct: float = 0.75,
    horizon_discounts: Mapping[int, float] | None = None,
    enable_conformal_pruning: bool = True,
) -> PRMTreeSearchResult:
    """Execute PUCT + Conformal Branch-and-Bound tree search over a candidate reasoning tree.

    Scores candidate branches in batched 8-slot verifier passes and prunes any branch (s, a)
    whose conformal upper bound UCB_alpha(s, a) = V_hat(s, a) + z_alpha * sigma_hat(s, a)
    is strictly below the best incumbent lower bound LCB_alpha(s^*) = V_hat(s^*) - z_alpha * sigma_hat(s^*).
    """
    if not nodes:
        raise ValueError("nodes must be non-empty")
    if isinstance(conformal_z, bool) or not isinstance(conformal_z, (int, float)) or not math.isfinite(float(conformal_z)) or float(conformal_z) < 0.0:
        raise ValueError(f"conformal_z must be finite and >= 0, got {conformal_z!r}")
    if isinstance(c_puct, bool) or not isinstance(c_puct, (int, float)) or not math.isfinite(float(c_puct)) or float(c_puct) < 0.0:
        raise ValueError(f"c_puct must be finite and >= 0, got {c_puct!r}")

    node_map: dict[str, PRMBranchNode] = {}
    children_map: dict[str | None, list[PRMBranchNode]] = {}
    for n in nodes:
        if n.node_id in node_map:
            raise ValueError(f"duplicate node_id {n.node_id!r}")
        node_map[n.node_id] = n
        children_map.setdefault(n.parent_id, []).append(n)

    for child_list in children_map.values():
        child_list.sort(key=lambda item: (-item.prior_prob, item.node_id))

    roots = children_map.get(None, [])
    if not roots:
        raise ValueError("reasoning tree must contain at least one root node with parent_id=None")

    z_val = float(conformal_z)
    puct_c = float(c_puct)

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
        stack = [nid]
        while stack:
            cur = stack.pop()
            for ch in children_map.get(cur, []):
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

        node_lcb = max(0.0, path_mean - z_val * path_std)
        node_ucb = min(1.0, path_mean + z_val * path_std)
        puct_bonus = puct_c * float(node.prior_prob) * (math.sqrt(float(sibling_count + 1)) / float(1 + sibling_rank))

        # Conformal Branch-and-Bound pruning condition: UCB_alpha(s, a) < best_incumbent_LCB
        ch_list = children_map.get(node.node_id, [])
        is_leaf = node.is_terminal or not ch_list
        if enable_conformal_pruning and (not is_leaf) and best_leaf is not None:
            if node_ucb + 0.02 * puct_bonus < best_leaf_lcb:
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

        # Order children by prior_prob + optimistic parent value
        for rank_idx, ch in enumerate(ch_list):
            if enable_conformal_pruning and best_leaf is not None:
                # Pre-screen child prior upper bound before expanding deep low-prior branches
                optimistic_child_ucb = (new_val_sum + min(1.0, 0.65 + 0.35 * ch.prior_prob)) / float(steps + 1) + z_val * path_std
                if optimistic_child_ucb < best_leaf_lcb:
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
        best_leaf = first
        best_leaf_path = (first.node_id,)
        best_leaf_val = v0
        best_leaf_lcb = max(0.0, v0 - z_val * s0)
        best_leaf_ucb = min(1.0, v0 + z_val * s0)

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

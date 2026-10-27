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

"""Transitive Triangle Consistency, Pack-Quantity Verifier, and Multi-Slot Lock Arbiters."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .eval_suite import brier_murphy_decomposition
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    PairwiseEdge,
    ProbabilisticTriangleScore,
    SwarmAgentProposal,
    SwarmLockDecision,
    TriangleContradiction,
    UnitMeasure,
)

_RE_MULTIPACK_PREFIX = re.compile(r"\b(\d+)\s*[x×]\s*\d+", re.IGNORECASE)
_RE_PACK_OF = re.compile(r"\b(?:pack\s+of|case\s+of|box\s+of)\s+(\d+)\b", re.IGNORECASE)
_RE_HYPHEN_PACK = re.compile(
    r"\b(\d+)\s*(?:-|\s)?\s*(?:pack|pk|count|ct|cans|bottles|bars|pods|pouches)\b",
    re.IGNORECASE,
)
_RE_UNIT_MEASURE = re.compile(
    r"(?:\b\d+\s*[x×]\s*)?(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|litres?|cl|g|grams?|kg|kilograms?|fl\s*oz|oz)\b",
    re.IGNORECASE,
)


def check_transitive_triangles(
    edges: Sequence[PairwiseEdge],
    high_thresh: float = 0.80,
    low_thresh: float = 0.25,
) -> list[TriangleContradiction]:
    """Detect 3-cycles (A, B, C) where two edges have P(equiv) >= high_thresh and the third <= low_thresh."""
    if high_thresh <= 0.0:
        high_thresh = 0.80
    if low_thresh <= 0.0:
        low_thresh = 0.25

    pair_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    node_set: set[str] = set()
    for e in edges:
        u, v = e.node_a.strip(), e.node_b.strip()
        if not u or not v or u == v:
            continue
        if u > v:
            u, v = v, u
        node_set.add(u)
        node_set.add(v)
        pair_sums[(u, v)].append(float(e.equiv_prob))

    prob_of = {k: sum(vals) / len(vals) for k, vals in pair_sums.items()}
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in prob_of:
        adj[u].append(v)
    for u in adj:
        adj[u].sort()

    contradictions: list[TriangleContradiction] = []
    for a in sorted(node_set):
        nbrs = adj[a]
        for i in range(len(nbrs)):
            b = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                c = nbrs[j]
                if (b, c) not in prob_of:
                    continue
                p_ab = prob_of[(a, b)]
                p_bc = prob_of[(b, c)]
                p_ac = prob_of[(a, c)]
                probs = (p_ab, p_bc, p_ac)
                pairs = ((a, b), (b, c), (a, c))

                high_indices = [idx for idx, p in enumerate(probs) if p >= high_thresh]
                low_indices = [idx for idx, p in enumerate(probs) if p <= low_thresh]
                if len(high_indices) == 2 and len(low_indices) == 1:
                    low_idx = low_indices[0]
                    min_strong = min(probs[idx] for idx in high_indices)
                    weak_p = probs[low_idx]
                    contradictions.append(
                        TriangleContradiction(
                            nodes=(a, b, c),
                            edge_probs=probs,
                            weak_edge=pairs[low_idx],
                            weak_prob=weak_p,
                            severity=round(min_strong - weak_p, 6),
                        )
                    )

    contradictions.sort(key=lambda c: (-c.severity, c.nodes))
    return contradictions


def probabilistic_triangle_contradiction(
    p_ab: float,
    p_bc: float,
    p_ac: float,
) -> tuple[float, int]:
    """Compute continuous soft transitive contradiction max(p_ij * p_jk * (1 - p_ik)) in [0, 1].

    Returns (max_contradiction_score, weak_edge_index) where weak_edge_index is 0 for (a,b),
    1 for (b,c), or 2 for (a,c).
    """
    p0 = min(1.0, max(0.0, float(p_ab)))
    p1 = min(1.0, max(0.0, float(p_bc)))
    p2 = min(1.0, max(0.0, float(p_ac)))
    # Index corresponds to the contradicted (weak) edge:
    # 0 -> (b,c) and (a,c) high while (a,b) low
    # 1 -> (a,b) and (a,c) high while (b,c) low
    # 2 -> (a,b) and (b,c) high while (a,c) low
    cands = (
        p1 * p2 * (1.0 - p0),
        p0 * p2 * (1.0 - p1),
        p0 * p1 * (1.0 - p2),
    )
    best_idx = max(range(3), key=lambda idx: cands[idx])
    return cands[best_idx], best_idx


def score_probabilistic_triangles(
    edges: Sequence[PairwiseEdge],
    min_score: float = 0.0,
) -> list[ProbabilisticTriangleScore]:
    """Rank 3-cycles by continuous transitive contradiction score p_ij * p_jk * (1 - p_ik)."""
    pair_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    node_set: set[str] = set()
    for e in edges:
        u, v = e.node_a.strip(), e.node_b.strip()
        if not u or not v or u == v:
            continue
        if u > v:
            u, v = v, u
        node_set.add(u)
        node_set.add(v)
        pair_sums[(u, v)].append(min(1.0, max(0.0, float(e.equiv_prob))))

    prob_of = {k: sum(vals) / len(vals) for k, vals in pair_sums.items()}
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in prob_of:
        adj[u].append(v)
    for u in adj:
        adj[u].sort()

    scored: list[ProbabilisticTriangleScore] = []
    for a in sorted(node_set):
        nbrs = adj[a]
        for i in range(len(nbrs)):
            b = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                c = nbrs[j]
                if (b, c) not in prob_of:
                    continue
                p_ab = prob_of[(a, b)]
                p_bc = prob_of[(b, c)]
                p_ac = prob_of[(a, c)]
                score, weak_idx = probabilistic_triangle_contradiction(p_ab, p_bc, p_ac)
                if score < min_score:
                    continue
                pairs = ((a, b), (b, c), (a, c))
                probs = (p_ab, p_bc, p_ac)
                scored.append(
                    ProbabilisticTriangleScore(
                        nodes=(a, b, c),
                        edge_probs=probs,
                        weak_edge=pairs[weak_idx],
                        weak_prob=probs[weak_idx],
                        contradiction_score=round(score, 8),
                    )
                )

    scored.sort(key=lambda s: (-s.contradiction_score, s.nodes))
    return scored


@dataclass(frozen=True)
class TransitiveProjectionResult:
    """Maximum-log-odds / minimum-disagreement projection onto the transitive equivalence polytope."""

    projected_edges: list[PairwiseEdge]
    clusters: list[tuple[str, ...]]
    num_violated_triangles_before: int
    num_violated_triangles_after: int
    num_edges_flipped: int
    objective_score: float


def _solve_component_partition(
    comp_nodes: Sequence[str],
    pair_weights: dict[tuple[int, int], float],
) -> tuple[list[int], float]:
    """Find partition assignment maximizing sum_{i<j, same_cluster} w_{ij} on a connected component."""
    m = len(comp_nodes)
    if m <= 1:
        return [0] * m, 0.0

    # Branch-and-bound exact set-partition solver for components of size <= 11
    if m <= 11:
        upper_rem = [0.0] * (m + 1)
        for i in range(m - 1, -1, -1):
            pos_i = sum(max(0.0, pair_weights.get((u, i), 0.0)) for u in range(i))
            upper_rem[i] = upper_rem[i + 1] + pos_i

        # Initialize with greedy agglomerative clustering for a strong initial lower bound
        init_assign = list(range(m))
        init_score = 0.0
        best_assign = list(init_assign)
        best_score = init_score

        cur_assign = [0] * m

        def _dfs(idx: int, num_clusters: int, cur_score: float) -> None:
            nonlocal best_score, best_assign
            if idx == m:
                if cur_score > best_score + 1e-12:
                    best_score = cur_score
                    best_assign = list(cur_assign)
                return
            if cur_score + upper_rem[idx] <= best_score + 1e-12:
                return

            # Try joining existing clusters in descending order of marginal gain
            gains: list[tuple[float, int]] = []
            for c_id in range(num_clusters):
                g = sum(
                    pair_weights.get((u, idx), 0.0)
                    for u in range(idx)
                    if cur_assign[u] == c_id
                )
                gains.append((g, c_id))
            gains.sort(key=lambda item: (-item[0], item[1]))

            for g_val, c_id in gains:
                cur_assign[idx] = c_id
                _dfs(idx + 1, num_clusters, cur_score + g_val)

            # Try opening a new singleton cluster
            cur_assign[idx] = num_clusters
            _dfs(idx + 1, num_clusters + 1, cur_score)

        _dfs(1, 1, 0.0)
        return best_assign, best_score

    # Average-linkage agglomerative correlation clustering + local move refinement for larger graphs
    clusters_map: dict[int, list[int]] = {i: [i] for i in range(m)}
    while len(clusters_map) > 1:
        best_gain = 0.0
        best_pair: tuple[int, int] | None = None
        c_ids = sorted(clusters_map.keys())
        for idx_a in range(len(c_ids)):
            ca = c_ids[idx_a]
            nodes_a = clusters_map[ca]
            for idx_b in range(idx_a + 1, len(c_ids)):
                cb = c_ids[idx_b]
                nodes_b = clusters_map[cb]
                gain = sum(
                    pair_weights.get((min(u, v), max(u, v)), 0.0)
                    for u in nodes_a
                    for v in nodes_b
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_pair = (ca, cb)
        if best_pair is None:
            break
        ca, cb = best_pair
        clusters_map[ca].extend(clusters_map.pop(cb))

    assign = [0] * m
    for new_cid, (_, members) in enumerate(sorted(clusters_map.items())):
        for u in members:
            assign[u] = new_cid

    total_score = sum(
        w for (u, v), w in pair_weights.items() if assign[u] == assign[v]
    )
    return assign, total_score


def project_triangles_to_transitive_consistency(
    edges: Sequence[PairwiseEdge],
    threshold: float = 0.50,
    high_thresh: float = 0.80,
    low_thresh: float = 0.25,
) -> TransitiveProjectionResult:
    """Project pairwise equivalence probabilities onto the transitive equivalence polytope.

    Solves weighted correlation clustering with log-odds edge weights
    `w_{uv} = logit(p_{uv}) - logit(threshold)` so that the resulting equivalence partition
    satisfies all triangle inequalities `E_{uv} + E_{vw} - E_{uw} <= 1` with zero 3-cycle
    contradictions. Exact branch-and-bound is used on connected components of size <= 11
    and agglomerative correlation clustering on larger components.
    """
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(f"threshold must lie strictly in (0, 1), got {threshold!r}")

    before_violations = len(
        check_transitive_triangles(edges, high_thresh=high_thresh, low_thresh=low_thresh)
    )

    pair_sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    node_set: set[str] = set()
    for e in edges:
        u, v = e.node_a.strip(), e.node_b.strip()
        if not u or not v or u == v:
            continue
        if u > v:
            u, v = v, u
        node_set.add(u)
        node_set.add(v)
        pair_sums[(u, v)].append(min(1.0 - 1e-6, max(1e-6, float(e.equiv_prob))))

    if not pair_sums:
        return TransitiveProjectionResult(
            projected_edges=[],
            clusters=[],
            num_violated_triangles_before=0,
            num_violated_triangles_after=0,
            num_edges_flipped=0,
            objective_score=0.0,
        )

    prob_of = {k: sum(vals) / len(vals) for k, vals in pair_sums.items()}
    thresh_clamped = min(1.0 - 1e-6, max(1e-6, float(threshold)))
    thresh_logit = math.log(thresh_clamped / (1.0 - thresh_clamped))

    # Build positive-edge connected components so disconnected subgraphs are solved independently
    pos_adj: dict[str, set[str]] = defaultdict(set)
    for (u, v), p in prob_of.items():
        if p >= thresh_clamped:
            pos_adj[u].add(v)
            pos_adj[v].add(u)

    visited: set[str] = set()
    all_clusters: list[tuple[str, ...]] = []
    node_to_cluster: dict[str, int] = {}
    total_obj = 0.0

    for root in sorted(node_set):
        if root in visited:
            continue
        comp: list[str] = []
        queue = [root]
        visited.add(root)
        while queue:
            cur = queue.pop(0)
            comp.append(cur)
            for nb in sorted(pos_adj[cur]):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp.sort()
        node_idx = {name: i for i, name in enumerate(comp)}
        pair_weights: dict[tuple[int, int], float] = {}
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                key = (comp[i], comp[j])
                if key in prob_of:
                    p_val = prob_of[key]
                    w_val = math.log(p_val / (1.0 - p_val)) - thresh_logit
                else:
                    # Missing edge defaults to slight non-equivalence prior below threshold
                    w_val = -abs(thresh_logit) - 0.5
                pair_weights[(i, j)] = w_val

        assign, comp_obj = _solve_component_partition(comp, pair_weights)
        total_obj += comp_obj

        grouped: dict[int, list[str]] = defaultdict(list)
        for idx_n, c_id in enumerate(assign):
            grouped[c_id].append(comp[idx_n])
        for members in grouped.values():
            members_tup = tuple(sorted(members))
            cid_global = len(all_clusters)
            all_clusters.append(members_tup)
            for m_name in members_tup:
                node_to_cluster[m_name] = cid_global

    all_clusters.sort()
    projected_edges: list[PairwiseEdge] = []
    flipped = 0

    for (u, v), orig_p in sorted(prob_of.items()):
        same_cluster = node_to_cluster[u] == node_to_cluster[v]
        was_positive = orig_p >= thresh_clamped
        if same_cluster != was_positive:
            flipped += 1
        if same_cluster:
            proj_p = max(orig_p, max(high_thresh, thresh_clamped + 0.5 * (1.0 - thresh_clamped)))
        else:
            proj_p = min(orig_p, min(low_thresh * 0.5, 0.5 * thresh_clamped))
        projected_edges.append(PairwiseEdge(node_a=u, node_b=v, equiv_prob=round(proj_p, 6)))

    after_violations = len(
        check_transitive_triangles(projected_edges, high_thresh=high_thresh, low_thresh=low_thresh)
    )
    return TransitiveProjectionResult(
        projected_edges=projected_edges,
        clusters=all_clusters,
        num_violated_triangles_before=before_violations,
        num_violated_triangles_after=after_violations,
        num_edges_flipped=flipped,
        objective_score=round(total_obj, 6),
    )


def parse_pack_multiplier(title: str) -> int:
    """Extract an explicit integer multipack multiplier from a catalog title (defaults to 1)."""
    for pattern in (_RE_MULTIPACK_PREFIX, _RE_PACK_OF, _RE_HYPHEN_PACK):
        m = pattern.search(title)
        if m:
            val = int(m.group(1))
            if val > 0:
                return val
    return 1


def parse_unit_measure(title: str) -> UnitMeasure | None:
    """Extract and normalize per-unit volume or mass from a catalog title (`0.33L -> 330ml`, `1kg -> 1000g`)."""
    m = _RE_UNIT_MEASURE.search(title)
    if not m:
        return None
    raw_val = float(m.group(1))
    if raw_val <= 0.0:
        return None
    raw_unit = re.sub(r"\s+", " ", m.group(2).strip().lower())
    if raw_unit in ("ml", "milliliter", "milliliters"):
        return UnitMeasure(round(raw_val, 3), "ml")
    if raw_unit in ("cl",):
        return UnitMeasure(round(raw_val * 10.0, 3), "ml")
    if raw_unit in ("l", "liter", "liters", "litre", "litres"):
        return UnitMeasure(round(raw_val * 1000.0, 3), "ml")
    if raw_unit in ("g", "gram", "grams"):
        return UnitMeasure(round(raw_val, 3), "g")
    if raw_unit in ("kg", "kilogram", "kilograms"):
        return UnitMeasure(round(raw_val * 1000.0, 3), "g")
    if raw_unit == "fl oz" or raw_unit == "floz":
        return UnitMeasure(round(raw_val, 3), "fl_oz")
    if raw_unit == "oz":
        return UnitMeasure(round(raw_val, 3), "oz")
    return None


def validate_pack_quantity_consistency(
    title_a: str,
    title_b: str,
    predicted_relation: str,
    predicted_pack_match: str,
    rel_tol: float = 0.02,
) -> bool:
    """Return False if multipack count OR normalized per-unit volume/mass differs on an exact match."""
    is_exact_claim = (
        predicted_relation == "exact_duplicate"
        or predicted_pack_match == "identical_pack_and_volume"
    )
    if not is_exact_claim:
        return True

    mult_a = parse_pack_multiplier(title_a)
    mult_b = parse_pack_multiplier(title_b)
    if mult_a != mult_b:
        return False

    unit_a = parse_unit_measure(title_a)
    unit_b = parse_unit_measure(title_b)
    if unit_a is not None and unit_b is not None:
        if unit_a.unit != unit_b.unit:
            return False
        denom = max(unit_a.value, unit_b.value, 1e-9)
        if abs(unit_a.value - unit_b.value) / denom > rel_tol:
            return False
    return True


def arbitrate_swarm_locks(
    proposals: Sequence[SwarmAgentProposal],
    max_hesitation: float = DEFAULT_HESITATION_THRESHOLD,
) -> SwarmLockDecision:
    """Resolve concurrent worker agent file-lock proposals in a single deterministic pass."""
    if max_hesitation <= 0.0:
        max_hesitation = DEFAULT_HESITATION_THRESHOLD
    if not proposals:
        return SwarmLockDecision(
            agent_actions={},
            write_collision=False,
            winning_agent="",
            escalate_to_human=False,
        )

    path_owners: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(proposals):
        seen: set[str] = set()
        for raw_p in p.target_paths:
            norm = raw_p.strip()
            if norm and norm not in seen:
                seen.add(norm)
                path_owners[norm].append(idx)

    colliding_indices: set[int] = set()
    write_collision = False
    for idxs in path_owners.values():
        if len(idxs) > 1:
            write_collision = True
            colliding_indices.update(idxs)

    ordered_indices = sorted(
        range(len(proposals)),
        key=lambda i: (-proposals[i].confidence, proposals[i].hesitation, proposals[i].agent_id),
    )
    winning_agent = proposals[ordered_indices[0]].agent_id

    actions: dict[str, str] = {}
    claimed_paths: dict[str, str] = {}
    escalate_to_human = False

    for idx in ordered_indices:
        p = proposals[idx]
        if idx not in colliding_indices:
            if p.hesitation >= max_hesitation:
                actions[p.agent_id] = "escalate_conflict"
                escalate_to_human = True
            else:
                actions[p.agent_id] = "grant_lock"
            for raw_p in p.target_paths:
                norm = raw_p.strip()
                if norm:
                    claimed_paths[norm] = p.agent_id
            continue

        if p.hesitation >= max_hesitation:
            actions[p.agent_id] = "escalate_conflict"
            escalate_to_human = True
            continue

        overlap = any(raw_p.strip() in claimed_paths for raw_p in p.target_paths if raw_p.strip())
        if not overlap:
            actions[p.agent_id] = "grant_lock"
            for raw_p in p.target_paths:
                norm = raw_p.strip()
                if norm:
                    claimed_paths[norm] = p.agent_id
        else:
            actions[p.agent_id] = "serialize_after_peer"

    return SwarmLockDecision(
        agent_actions=actions,
        write_collision=write_collision,
        winning_agent=winning_agent,
        escalate_to_human=escalate_to_human,
    )


@dataclass(frozen=True)
class CorrelatedEnsembleCalibrationResult:
    """Covariance-Discounted Supra-Bayesian Multi-Agent Pooling weights and diagnostics.

    Defends against correlated agent errors, model clones, and sycophantic majorities
    by weighting each agent m using inverse error-correlation matrix R_{reg}^{-1} times
    Murphy resolution-over-reliability skill s_m = (RES_m + 0.01) / (REL_m + 0.05).
    """

    agent_ids: tuple[str, ...]
    weights: dict[str, float]
    murphy_skill_scores: dict[str, float]
    effective_sample_size: float
    mean_pairwise_correlation: float
    shrinkage_delta: float


def _solve_linear_system_dense(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    m = len(rhs)
    aug: list[list[float]] = [
        [float(matrix[i][j]) for j in range(m)] + [float(rhs[i])]
        for i in range(m)
    ]
    for col in range(m):
        pivot_row = max(range(col, m), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            aug[col][col] += 1e-6
            pivot_row = col
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        piv = aug[col][col]
        for j in range(col, m + 1):
            aug[col][j] /= piv
        for r in range(m):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) > 1e-15:
                for j in range(col, m + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][m] for i in range(m)]


def calibrate_correlated_ensemble_weights(
    calib_agent_probs: Mapping[str, Sequence[Mapping[str, float]]],
    calib_labels: Sequence[str],
    *,
    shrinkage_delta: float = 0.15,
    num_bins: int = 10,
) -> CorrelatedEnsembleCalibrationResult:
    """Calibrate Covariance-Discounted Supra-Bayesian pooling weights across M agents.

    Computes signed error residuals e_{m,i} = 1 - p_{m,i}(y_i^*), estimates the M x M
    residual correlation matrix R with Ledoit-Wolf shrinkage R_reg = (1 - delta) * R + delta * I,
    and solves R_reg * u = s where s_m = (RES_m + 0.01) / (REL_m + 0.05).
    Consequently, K identical clones share weight 1/K rather than multiplying their log-odds by K.
    """
    if not calib_agent_probs:
        raise ValueError("calib_agent_probs must be non-empty")
    if not calib_labels:
        raise ValueError("calib_labels must be non-empty")
    if isinstance(shrinkage_delta, bool) or not isinstance(shrinkage_delta, (int, float)) or not math.isfinite(float(shrinkage_delta)) or float(shrinkage_delta) <= 0.0 or float(shrinkage_delta) > 1.0:
        raise ValueError(f"shrinkage_delta must lie in (0, 1], got {shrinkage_delta!r}")

    n = len(calib_labels)
    agent_ids = tuple(sorted(str(k) for k in calib_agent_probs.keys()))
    m_count = len(agent_ids)

    residuals: dict[str, list[float]] = {}
    skills: dict[str, float] = {}

    for aid in agent_ids:
        seq = calib_agent_probs[aid]
        if len(seq) != n:
            raise ValueError(f"agent {aid!r} has {len(seq)} calibration items, expected {n}")
        murphy = brier_murphy_decomposition(seq, calib_labels, num_bins=num_bins)
        # Higher resolution (RES) and lower reliability error (REL) yield higher skill
        acc_bonus = sum(
            1.0
            for dist, lbl in zip(seq, calib_labels)
            if sorted(dist.keys(), key=lambda k: (-float(dist[k]), str(k)))[0] == str(lbl)
        ) / float(n)
        skill_val = ((murphy.resolution + 0.02 * acc_bonus + 0.005) / (murphy.reliability + 0.04))
        skills[aid] = max(1e-4, skill_val)

        res_list: list[float] = []
        for dist, lbl_raw in zip(seq, calib_labels):
            lbl = str(lbl_raw)
            p_gold = float(dist.get(lbl, 0.0))
            res_list.append(1.0 - min(1.0, max(0.0, p_gold)))
        residuals[aid] = res_list

    # Compute pairwise residual correlation matrix R in [-1, 1]^{M x M}
    # Combine centered correlation with cosine similarity of raw residual vectors so identical
    # constant-error clones also exhibit correlation = 1.0.
    corr_mat: list[list[float]] = [[1.0 if i == j else 0.0 for j in range(m_count)] for i in range(m_count)]
    off_diag_sum = 0.0
    off_diag_pairs = 0

    for i in range(m_count):
        ri = residuals[agent_ids[i]]
        mean_i = sum(ri) / float(n)
        var_i = sum((x - mean_i) ** 2 for x in ri)
        norm_i = math.sqrt(sum(x * x for x in ri))
        for j in range(i + 1, m_count):
            rj = residuals[agent_ids[j]]
            mean_j = sum(rj) / float(n)
            var_j = sum((x - mean_j) ** 2 for x in rj)
            norm_j = math.sqrt(sum(x * x for x in rj))

            if var_i > 1e-12 and var_j > 1e-12:
                cov = sum((ri[k] - mean_i) * (rj[k] - mean_j) for k in range(n))
                pearson = cov / math.sqrt(var_i * var_j)
            else:
                pearson = 0.0

            if norm_i > 1e-12 and norm_j > 1e-12:
                cos_sim = sum(ri[k] * rj[k] for k in range(n)) / (norm_i * norm_j)
            else:
                cos_sim = 0.0

            # Positive error co-movement determines redundancy discounting
            rho_ij = max(0.0, min(1.0, 0.65 * max(0.0, pearson) + 0.35 * max(0.0, cos_sim)))
            corr_mat[i][j] = rho_ij
            corr_mat[j][i] = rho_ij
            off_diag_sum += rho_ij
            off_diag_pairs += 1

    delta = float(shrinkage_delta)
    r_reg: list[list[float]] = [
        [
            1.0 if i == j else (1.0 - delta) * corr_mat[i][j]
            for j in range(m_count)
        ]
        for i in range(m_count)
    ]
    skill_vec = [skills[aid] for aid in agent_ids]
    raw_u = _solve_linear_system_dense(r_reg, skill_vec)
    clipped_u = [max(0.0, val) for val in raw_u]
    u_sum = sum(clipped_u)
    if u_sum <= 1e-12:
        weights_list = [1.0 / float(m_count) for _ in range(m_count)]
    else:
        weights_list = [val / u_sum for val in clipped_u]

    weights_map = {aid: round(w, 8) for aid, w in zip(agent_ids, weights_list)}
    # Normalize rounding residual on largest weight
    top_aid = max(agent_ids, key=lambda a: weights_map[a])
    weights_map[top_aid] = round(weights_map[top_aid] + (1.0 - sum(weights_map.values())), 8)

    # Effective sample size N_eff = M / (1 + (M - 1) * mean_corr)
    mean_corr = (off_diag_sum / float(off_diag_pairs)) if off_diag_pairs > 0 else 0.0
    n_eff = float(m_count) / (1.0 + float(max(0, m_count - 1)) * mean_corr)

    return CorrelatedEnsembleCalibrationResult(
        agent_ids=agent_ids,
        weights=weights_map,
        murphy_skill_scores={k: round(v, 6) for k, v in skills.items()},
        effective_sample_size=round(n_eff, 6),
        mean_pairwise_correlation=round(mean_corr, 6),
        shrinkage_delta=delta,
    )


def covariance_discounted_logit_pool(
    agent_probs: Mapping[str, Mapping[str, float]],
    calibration: CorrelatedEnsembleCalibrationResult | None = None,
    *,
    prior_probs: Mapping[str, float] | None = None,
    prior_weight: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Pool multi-agent probability distributions via Covariance-Discounted Supra-Bayesian Logit Pooling.

    Computes:
        log P(y | d_1..d_M) propto (1 / temperature) * sum_{m=1}^M w_m * log p_m(y) + prior_weight * log pi(y)
    where sum_m w_m = 1 are the covariance-discounted weights. Unlike naive Product-of-Experts
    (which sums raw log-odds and suffers catastrophic overconfidence under correlated clones),
    convex covariance-discounted pooling is clone-invariant and resists sycophantic majorities.
    """
    if not agent_probs:
        raise ValueError("agent_probs must be non-empty")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    if isinstance(prior_weight, bool) or not isinstance(prior_weight, (int, float)) or not math.isfinite(float(prior_weight)) or float(prior_weight) < 0.0:
        raise ValueError(f"prior_weight must be finite and >= 0, got {prior_weight!r}")

    agent_ids = sorted(str(k) for k in agent_probs.keys())
    first_dist = agent_probs[agent_ids[0]]
    if not first_dist:
        raise ValueError("option probability mapping must be non-empty")
    option_keys = sorted(str(k) for k in first_dist.keys())

    if calibration is None:
        w_map = {aid: 1.0 / float(len(agent_ids)) for aid in agent_ids}
    else:
        raw_w = {aid: float(calibration.weights.get(aid, 0.0)) for aid in agent_ids}
        w_tot = sum(raw_w.values())
        if w_tot <= 1e-12:
            w_map = {aid: 1.0 / float(len(agent_ids)) for aid in agent_ids}
        else:
            w_map = {aid: raw_w[aid] / w_tot for aid in agent_ids}

    eps = 1e-9
    inv_temp = 1.0 / float(temperature)
    pooled_logits: dict[str, float] = {k: 0.0 for k in option_keys}

    for aid in agent_ids:
        dist = agent_probs[aid]
        w_m = w_map[aid]
        tot = sum(max(0.0, float(dist.get(k, 0.0))) for k in option_keys)
        if tot <= 0.0:
            raise ValueError(f"agent {aid!r} probability distribution sum must be > 0")
        for k in option_keys:
            p_mk = max(eps, float(dist.get(k, 0.0)) / tot)
            pooled_logits[k] += inv_temp * w_m * math.log(p_mk)

    if prior_probs is not None and prior_weight > 0.0:
        p_tot = sum(max(0.0, float(prior_probs.get(k, 0.0))) for k in option_keys)
        if p_tot > 0.0:
            for k in option_keys:
                pi_k = max(eps, float(prior_probs.get(k, 0.0)) / p_tot)
                pooled_logits[k] += float(prior_weight) * math.log(pi_k)

    max_l = max(pooled_logits.values())
    exps = {k: math.exp(pooled_logits[k] - max_l) for k in option_keys}
    z = sum(exps.values())
    return {k: exps[k] / z for k in option_keys}


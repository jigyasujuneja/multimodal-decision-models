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

import re
from collections import defaultdict
from typing import Sequence

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

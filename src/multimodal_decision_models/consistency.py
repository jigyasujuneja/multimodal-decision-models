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
    SwarmAgentProposal,
    SwarmLockDecision,
    TriangleContradiction,
)

_RE_MULTIPACK_PREFIX = re.compile(r"\b(\d+)\s*[x×]\s*\d+", re.IGNORECASE)
_RE_PACK_OF = re.compile(r"\b(?:pack\s+of|case\s+of|box\s+of)\s+(\d+)\b", re.IGNORECASE)
_RE_HYPHEN_PACK = re.compile(
    r"\b(\d+)\s*(?:-|\s)?\s*(?:pack|pk|count|ct|cans|bottles|bars|pods|pouches)\b",
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


def parse_pack_multiplier(title: str) -> int:
    """Extract an explicit integer multipack multiplier from a catalog title (defaults to 1)."""
    for pattern in (_RE_MULTIPACK_PREFIX, _RE_PACK_OF, _RE_HYPHEN_PACK):
        m = pattern.search(title)
        if m:
            val = int(m.group(1))
            if val > 0:
                return val
    return 1


def validate_pack_quantity_consistency(
    title_a: str,
    title_b: str,
    predicted_relation: str,
    predicted_pack_match: str,
) -> bool:
    """Return False if parsed multipack multipliers differ while the model predicts an exact duplicate."""
    mult_a = parse_pack_multiplier(title_a)
    mult_b = parse_pack_multiplier(title_b)
    if mult_a != mult_b:
        if predicted_relation == "exact_duplicate" or predicted_pack_match == "identical_pack_and_volume":
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

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

"""8-Slot Agent Trajectory Context Pruning via Exact Pareto-Frontier MCKP Dynamic Programming.

Scores up to 8 context blocks per forward pass over {"full", "stub", "drop"} and solves the
Multiple-Choice Knapsack Problem (MCKP) under an explicit token budget while preserving pinned
system or instruction blocks (minimum action = "stub").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .types import MAX_SLOTS_PER_PASS, DecisionQuestion

_PRUNE_ACTIONS: tuple[str, str, str] = ("full", "stub", "drop")
_RE_WORD_SHINGLE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class ContextBlock:
    """Single agent context block candidate with full and compressed stub token budgets."""

    id: str
    full_tokens: int
    stub_tokens: int
    full_text: str = ""
    stub_text: str = ""
    pinned: bool = False


@dataclass(frozen=True)
class ContextPruneResult:
    """Optimal per-block retention decisions, token compression stats, and rendered context."""

    decisions: dict[str, str]
    tokens_before: int
    tokens_after: int
    token_reduction_ratio: float
    expected_utility: float
    rendered_blocks: list[str] = field(default_factory=list)


def _normalize_prune_probs(block_id: str, dist: Mapping[str, float]) -> tuple[float, float, float]:
    if not dist:
        raise ValueError(f"missing slot_probabilities for block {block_id!r}")
    vals: dict[str, float] = {}
    tot = 0.0
    for act in _PRUNE_ACTIONS:
        raw_v = float(dist.get(act, 0.0))
        if not math.isfinite(raw_v) or raw_v < 0.0:
            raise ValueError(f"invalid probability for block {block_id!r} action {act!r}: {raw_v!r}")
        vals[act] = raw_v
        tot += raw_v
    if tot <= 0.0:
        raise ValueError(f"slot_probabilities for block {block_id!r} sum to zero over {_PRUNE_ACTIONS}")
    return vals["full"] / tot, vals["stub"] / tot, vals["drop"] / tot


def _pareto_prune_states(
    states: Sequence[tuple[int, float, tuple[str, ...]]],
    max_states: int = 4096,
) -> list[tuple[int, float, tuple[str, ...]]]:
    ordered = sorted(states, key=lambda item: (item[0], -item[1], item[2]))
    pareto: list[tuple[int, float, tuple[str, ...]]] = []
    best_u = -1.0
    for tok, util, choices in ordered:
        if util > best_u + 1e-12:
            pareto.append((tok, util, choices))
            best_u = util

    if len(pareto) <= max_states:
        return pareto

    min_t = pareto[0][0]
    max_t = pareto[-1][0]
    if max_t <= min_t:
        return [pareto[0]]

    span = float(max_t - min_t)
    buckets: dict[int, tuple[int, float, tuple[str, ...]]] = {}
    for state in pareto:
        b_idx = min(max_states - 1, int(((state[0] - min_t) / span) * (max_states - 1)))
        prev = buckets.get(b_idx)
        if prev is None or (state[1], -state[0]) > (prev[1], -prev[0]):
            buckets[b_idx] = state
    return [buckets[k] for k in sorted(buckets.keys())]


def _block_shingle_set(blk: ContextBlock) -> frozenset[str]:
    text = (blk.full_text or blk.stub_text or blk.id).lower()
    tokens = _RE_WORD_SHINGLE.findall(text)
    if not tokens:
        return frozenset()
    shingles = set(tokens)
    for i in range(len(tokens) - 1):
        shingles.add(f"{tokens[i]}_{tokens[i + 1]}")
    return frozenset(shingles)


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return float(inter) / float(union)


def prune_agent_context_blocks(
    blocks: Sequence[ContextBlock],
    slot_probabilities: Mapping[str, Mapping[str, float]],
    token_budget: int,
    *,
    redundancy_penalty_weight: float = 0.0,
) -> ContextPruneResult:
    """Select utility-maximizing {"full", "stub", "drop"} actions under `token_budget`.

    Action utilities given normalized (p_full, p_stub, p_drop):
      - u("full") = 1.0 * p_full + 0.5 * p_stub, cost = full_tokens
      - u("stub") = 0.6 * p_full + 1.0 * p_stub, cost = stub_tokens
      - u("drop") = 0.0, cost = 0 (disallowed when block.pinned is True)

    When `redundancy_penalty_weight > 0.0`, applies a submodular facility-location redundancy
    discount `1.0 - redundancy_penalty_weight * max_{j in retained} Jaccard(shingles_i, shingles_j)`
    so near-duplicate logs or stack traces do not crowd out orthogonal trajectory evidence.
    When `redundancy_penalty_weight == 0.0` (default), solves the exact additive MCKP DP.
    """
    if token_budget < 0:
        raise ValueError(f"token_budget must be >= 0, got {token_budget}")
    if (
        isinstance(redundancy_penalty_weight, bool)
        or not isinstance(redundancy_penalty_weight, (int, float))
        or not math.isfinite(float(redundancy_penalty_weight))
        or float(redundancy_penalty_weight) < 0.0
        or float(redundancy_penalty_weight) > 1.0
    ):
        raise ValueError(
            f"redundancy_penalty_weight must lie in [0.0, 1.0], got {redundancy_penalty_weight!r}"
        )
    red_weight = float(redundancy_penalty_weight)
    if not blocks:
        return ContextPruneResult(
            decisions={},
            tokens_before=0,
            tokens_after=0,
            token_reduction_ratio=0.0,
            expected_utility=0.0,
            rendered_blocks=[],
        )

    seen_ids: set[str] = set()
    min_pinned_tokens = 0
    for blk in blocks:
        if not blk.id:
            raise ValueError("ContextBlock.id must be non-empty")
        if blk.id in seen_ids:
            raise ValueError(f"duplicate ContextBlock.id: {blk.id!r}")
        seen_ids.add(blk.id)
        if blk.stub_tokens < 0 or blk.full_tokens < blk.stub_tokens:
            raise ValueError(
                f"block {blk.id!r} requires full_tokens >= stub_tokens >= 0, "
                f"got full={blk.full_tokens}, stub={blk.stub_tokens}"
            )
        if blk.id not in slot_probabilities:
            raise ValueError(f"missing slot_probabilities entry for block {blk.id!r}")
        if blk.pinned:
            min_pinned_tokens += blk.stub_tokens

    if token_budget < min_pinned_tokens:
        raise ValueError(
            f"token_budget ({token_budget}) is below minimum pinned stub token sum ({min_pinned_tokens})"
        )

    n_blocks = len(blocks)
    shingles = [_block_shingle_set(b) for b in blocks] if red_weight > 0.0 else []
    sim_matrix: list[list[float]] = []
    if red_weight > 0.0:
        sim_matrix = [
            [
                _jaccard_similarity(shingles[i], shingles[j]) if i != j else 1.0
                for j in range(n_blocks)
            ]
            for i in range(n_blocks)
        ]

    frontier: list[tuple[int, float, tuple[str, ...]]] = [(0, 0.0, ())]

    for blk_idx, blk in enumerate(blocks):
        p_full, p_stub, _ = _normalize_prune_probs(blk.id, slot_probabilities[blk.id])
        u_full = 1.0 * p_full + 0.5 * p_stub
        u_stub = 0.6 * p_full + 1.0 * p_stub

        options: list[tuple[str, int, float]] = [
            ("full", blk.full_tokens, u_full),
            ("stub", blk.stub_tokens, u_stub),
        ]
        if not blk.pinned:
            options.append(("drop", 0, 0.0))

        expanded: list[tuple[int, float, tuple[str, ...]]] = []
        for cur_tok, cur_u, cur_choices in frontier:
            max_sim = 0.0
            if red_weight > 0.0 and cur_choices:
                for prev_idx, prev_act in enumerate(cur_choices):
                    if prev_act == "full":
                        max_sim = max(max_sim, sim_matrix[blk_idx][prev_idx])
                    elif prev_act == "stub":
                        max_sim = max(max_sim, 0.5 * sim_matrix[blk_idx][prev_idx])
            discount = max(0.0, 1.0 - red_weight * max_sim)
            for act_name, act_cost, act_u in options:
                next_tok = cur_tok + act_cost
                if next_tok <= token_budget:
                    eff_u = (act_u * discount) if act_name != "drop" else 0.0
                    expanded.append((next_tok, cur_u + eff_u, cur_choices + (act_name,)))

        if not expanded:
            raise ValueError(f"no feasible pruning assignment found within token_budget={token_budget}")
        frontier = _pareto_prune_states(expanded)

    best_tok, best_u, best_choices = max(
        frontier,
        key=lambda item: (round(item[1], 12), -item[0], item[2]),
    )

    decisions: dict[str, str] = {}
    rendered: list[str] = []
    for blk, act in zip(blocks, best_choices):
        decisions[blk.id] = act
        if act == "full":
            rendered.append(blk.full_text)
        elif act == "stub":
            rendered.append(blk.stub_text)

    tokens_before = sum(b.full_tokens for b in blocks)
    reduction = (
        0.0
        if tokens_before == 0
        else float(tokens_before - best_tok) / float(tokens_before)
    )
    return ContextPruneResult(
        decisions=decisions,
        tokens_before=tokens_before,
        tokens_after=best_tok,
        token_reduction_ratio=reduction,
        expected_utility=best_u,
        rendered_blocks=rendered,
    )


def build_context_pruning_questions(
    blocks: Sequence[ContextBlock],
) -> list[dict[str, DecisionQuestion]]:
    """Pack trajectory context blocks into batches of up to MAX_SLOTS_PER_PASS (8) decision slots."""
    batches: list[dict[str, DecisionQuestion]] = []
    current: dict[str, DecisionQuestion] = {}

    for blk in blocks:
        q = DecisionQuestion(
            type="choice",
            instructions=(
                f"Select retention policy for context block {blk.id!r} "
                f"(full={blk.full_tokens} tok, stub={blk.stub_tokens} tok, pinned={blk.pinned})."
            ),
            criteria={
                "full": "Retain verbatim block text required for exact reasoning or edits",
                "stub": "Replace with compact summary stub preserving high-level state",
                "drop": "Omit superseded or irrelevant trajectory block",
            },
            order=["full", "stub", "drop"],
        )
        current[blk.id] = q
        if len(current) == MAX_SLOTS_PER_PASS:
            batches.append(current)
            current = {}

    if current:
        batches.append(current)
    return batches

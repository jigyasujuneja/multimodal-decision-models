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

"""Catalog-Scale Wide-Option Shortlist Routing and CSLS Hubness Correction.

Reduces high-cardinality classification (K > 26 up to 10,000+ classes) from multi-pass bracket
tournaments (ceil(K / 20) + 1 passes) to a single calibrated forward pass over the top-M
shortlisted candidates plus an out-of-distribution catch-all option ("none_of_the_above").
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .types import MAX_OPTIONS_PER_SLOT, DecisionQuestion


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must be non-empty and have matching dimensions")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def csls_hubness_scores(
    query_vec: Sequence[float],
    prototype_vecs: Mapping[str, Sequence[float]],
    neighborhood_means: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Compute Cross-Domain Similarity Local Scaling (CSLS) scores to prevent head-brand hubness.

    CSLS(q, p_k) = 2 * cos(q, p_k) - r_T(p_k), where r_T(p_k) is the mean similarity of prototype k
    to its top neighbors.
    """
    scores: dict[str, float] = {}
    for key, pvec in prototype_vecs.items():
        sim = cosine_similarity(query_vec, pvec)
        r_t = float(neighborhood_means.get(key, 0.0)) if neighborhood_means else 0.0
        scores[key] = 2.0 * sim - r_t
    return scores


def shortlist_wide_question(
    question: DecisionQuestion,
    scores: Mapping[str, float],
    top_k: int = 6,
    catch_all_key: str = "none_of_the_above",
    pairwise_diff: bool = True,
) -> DecisionQuestion:
    """Reduce a wide-option question (K > 26) to a single-pass top-M shortlist with catch-all.

    Leaves questions with K <= MAX_OPTIONS_PER_SLOT (26) unchanged.
    """
    if question.type not in ("choice", "noul"):
        return question
    keys = question.option_keys()
    if len(keys) <= MAX_OPTIONS_PER_SLOT:
        return question
    if top_k <= 0 or top_k >= MAX_OPTIONS_PER_SLOT:
        top_k = 6

    ranked = sorted(
        keys,
        key=lambda k: (float(scores.get(k, float("-inf"))), -keys.index(k)),
        reverse=True,
    )
    selected = ranked[:top_k]

    if catch_all_key and catch_all_key in question.criteria and catch_all_key not in selected:
        selected = selected[: max(1, top_k - 1)] + [catch_all_key]

    new_criteria: dict[str, str] = {}
    for idx, k in enumerate(selected):
        base_desc = question.criteria.get(k, k)
        if pairwise_diff and k != catch_all_key and len(selected) > 1:
            rival = selected[1] if idx == 0 else selected[0]
            rival_desc = question.criteria.get(rival, rival)
            new_criteria[k] = f"{base_desc} (distinguish from {rival}: {rival_desc})"
        else:
            new_criteria[k] = base_desc

    if catch_all_key and catch_all_key not in new_criteria and len(new_criteria) < MAX_OPTIONS_PER_SLOT:
        new_criteria[catch_all_key] = question.criteria.get(
            catch_all_key,
            "None of the shortlisted candidates match; escalate or abstain",
        )
        selected = list(selected) + [catch_all_key]

    return DecisionQuestion(
        type=question.type,
        instructions=question.instructions,
        criteria=new_criteria,
        order=list(selected),
    )

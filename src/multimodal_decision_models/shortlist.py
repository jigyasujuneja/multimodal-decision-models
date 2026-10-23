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

"""Catalog-Scale Wide-Option Shortlist Routing, Log-Softmax Cosine Priors, and CSLS Hubness Correction.

Reduces high-cardinality classification (K > 26) from multi-pass packed bracket tournaments
(1 + ceil(ceil(K / 20) / 8) passes) to a single calibrated forward pass over the top-M shortlisted
candidates (default M = 20, tested up to K = 202) plus an out-of-distribution
catch-all option ("none_of_the_above").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .types import MAX_OPTIONS_PER_SLOT, DecisionQuestion

DEFAULT_COSINE_PRIOR_TEMPERATURE = 0.07

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "user",
        "when",
        "with",
    }
)

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass(frozen=True)
class ShortlistConfig:
    """Configuration for single-pass wide-option shortlist reduction.

    contrastive_tags defaults to False because sibling-unique cues help boilerplate-heavy SKU
    titles while hurting terse intent labels. Enable explicitly on repetitive catalog schemas.
    """

    max_candidates: int = 20
    catch_all_key: str = "none_of_the_above"
    contrastive_tags: bool = False


def split_identifier_tokens(text: str) -> list[str]:
    """Split snake_case, camelCase, and prose text into lowercase informative tokens."""
    expanded = _CAMEL_RE.sub(r"\1 \2", text)
    raw_parts = _NON_ALNUM_RE.split(expanded.lower())
    out: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        if len(part) < 2 or part in _STOPWORDS or part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def build_contrastive_criteria(
    criteria: Mapping[str, str],
    selected_keys: Sequence[str],
    catch_all_key: str = "none_of_the_above",
    max_tags: int = 2,
) -> dict[str, str]:
    """Append sibling-unique [distinguishing: ...] cues to shortlisted option descriptions.

    Matches Go's BuildContrastiveCriteria: extracts tokens present in option k's key or
    description that appear fewest times across the other shortlisted siblings.
    """
    non_catch = [k for k in selected_keys if k != catch_all_key]
    token_sets: dict[str, list[str]] = {}
    doc_freq: dict[str, int] = {}
    for k in non_catch:
        toks = split_identifier_tokens(f"{k} {criteria.get(k, '')}")
        token_sets[k] = toks
        for tok in toks:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    out: dict[str, str] = {}
    for k in selected_keys:
        base_desc = criteria.get(k, k)
        if k == catch_all_key or len(non_catch) <= 1 or max_tags <= 0:
            out[k] = base_desc
            continue
        toks = token_sets.get(k, [])
        unique = [t for t in toks if doc_freq.get(t, 0) == 1]
        if not unique:
            # Fall back to rarest tokens among siblings (< half of shortlisted options)
            half = max(2, len(non_catch) // 2 + 1)
            unique = sorted(
                (t for t in toks if doc_freq.get(t, 0) < half),
                key=lambda t: (doc_freq.get(t, 0), toks.index(t)),
            )
        tags = unique[:max_tags]
        if tags:
            out[k] = f"{base_desc} [distinguishing: {', '.join(tags)}]"
        else:
            out[k] = base_desc
    return out


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


def normalize_cosine_prior_logits(
    cosines: Mapping[str, float],
    temperature: float = DEFAULT_COSINE_PRIOR_TEMPERATURE,
) -> dict[str, float]:
    """Convert raw cosine similarities into temperature-scaled log-softmax prior logits.

    Computes log P(k) = (cos_k / tau) - logsumexp_j(cos_j / tau) with default tau = 0.07
    so compressed dense embedding cosines (typically [0.25, 0.55]) form a calibrated
    log-probability prior before pre-gated Bayesian PoE fusion.
    """
    if not cosines:
        return {}
    tau = temperature if temperature > 0.0 and math.isfinite(temperature) else DEFAULT_COSINE_PRIOR_TEMPERATURE
    scaled: dict[str, float] = {}
    for k, v in cosines.items():
        fv = float(v)
        if not math.isfinite(fv):
            fv = -1.0
        scaled[k] = fv / tau
    max_s = max(scaled.values())
    lse = max_s + math.log(sum(math.exp(s - max_s) for s in scaled.values()))
    return {k: s - lse for k, s in scaled.items()}


def shortlist_wide_question(
    question: DecisionQuestion,
    scores: Mapping[str, float],
    top_k: int = 20,
    catch_all_key: str = "none_of_the_above",
    pairwise_diff: bool = False,
    contrastive_mode: str = "sibling_unique",
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
        top_k = 20

    non_catch = [k for k in keys if k != catch_all_key]
    pos_map = {k: i for i, k in enumerate(non_catch)}
    ranked = sorted(
        non_catch,
        key=lambda k: (float(scores.get(k, float("-inf"))), -pos_map[k]),
        reverse=True,
    )

    reserve_catch = bool(catch_all_key)
    budget = min(len(ranked), max(1, min(top_k, MAX_OPTIONS_PER_SLOT - (1 if reserve_catch else 0))))
    selected = list(ranked[:budget])
    if reserve_catch and catch_all_key not in selected and len(selected) < MAX_OPTIONS_PER_SLOT:
        selected.append(catch_all_key)

    base_criteria: dict[str, str] = {}
    for k in selected:
        if k == catch_all_key and k not in question.criteria:
            base_criteria[k] = "None of the shortlisted candidates match; escalate or abstain"
        else:
            base_criteria[k] = question.criteria.get(k, k)

    if not pairwise_diff:
        new_criteria = base_criteria
    elif contrastive_mode == "rival_pair":
        new_criteria = {}
        for idx, k in enumerate(selected):
            base_desc = base_criteria[k]
            if k != catch_all_key and len(selected) > 1:
                rival = selected[1] if idx == 0 else selected[0]
                rival_desc = base_criteria.get(rival, rival)
                new_criteria[k] = f"{base_desc} (distinguish from {rival}: {rival_desc})"
            else:
                new_criteria[k] = base_desc
    else:
        new_criteria = build_contrastive_criteria(
            base_criteria,
            selected,
            catch_all_key=catch_all_key,
        )

    return DecisionQuestion(
        type=question.type,
        instructions=question.instructions,
        criteria=new_criteria,
        order=list(selected),
        prior_logits={k: question.prior_logits[k] for k in selected if k in question.prior_logits},
        evidence_prompt=question.evidence_prompt,
        evidence_criteria=dict(question.evidence_criteria),
    )

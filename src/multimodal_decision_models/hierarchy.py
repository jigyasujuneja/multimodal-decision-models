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

"""Hierarchical Semantic Taxonomy Readout & Trie-Constrained Decoding.

Translates semantic-ID hierarchy routing (Rajput et al., NeurIPS 2023, arXiv:2305.05065) to
zero-shot single-token [A-Z] decision models without vocabulary expansion or 52-label OOS
entropy degradation:

  1. mtp_naive (reads=1): Simultaneous coarse + fine slots decoded via independent marginal argmax.
  2. mtp_trie (reads=1): Simultaneous coarse + fine slots decoded strictly over valid root-to-leaf
     paths e in T via the product-of-marginals pseudo-log-likelihood
     s(e | x) = log P_1(c_1(e) | x) + log P_2(c_2(e) | x). Under parallel single-pass readout,
     slot 2 marginalizes over coarse parents; when fine codes alias across coarse domains,
     hesitation_hybrid and beam_2pass restore exact chain-rule conditioning log P_2(c_2 | c_1, x).
  3. beam_2pass (reads=2): Pass-1 coarse domain readout (<= 26 options) -> Pass-2 fine readout over
     leaves in the top-b coarse domains (<= 26 options), fused with Pass-1 domain log-priors.
  4. hesitation_hybrid (reads ~ 1.2): Runs 1-pass trie decode when coarse normalized entropy
     H_1 < hesitation_threshold, and runs Pass-2 top-b conditional refinement only when hesitant.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .calibration import normalized_entropy, pre_gated_bayesian_poe_fusion
from .shortlist import split_identifier_tokens
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    MAX_SLOTS_PER_PASS,
    DecisionQuestion,
    DecisionTerminalState,
    FailureCategory,
    SlotReadout,
)

if TYPE_CHECKING:
    from .backends import DecisionBackend

_SPLIT_PREFIX_RE = re.compile(r"[._:/-]+")
_OPAQUE_KEY_RE = re.compile(r"^(?:option|opt|choice|item|label|class|idx|id)_?\d+$", re.IGNORECASE)


def packed_bracket_pass_count(
    num_options: int,
    bracket_size: int = 20,
    max_slots_per_pass: int = MAX_SLOTS_PER_PASS,
) -> int:
    """Compute exact forward passes for packed tournament brackets.

    Packs up to max_slots_per_pass Round-1 brackets onto a single multi-slot canvas pass,
    followed by 1 finals pass: 1 + ceil(ceil(K / bracket_size) / max_slots_per_pass).
    Yields 2 passes on K=59 and K=151, and 3 passes on K=202.
    """
    if num_options <= MAX_OPTIONS_PER_SLOT:
        return 1
    b_size = max(2, bracket_size)
    s_cap = max(1, max_slots_per_pass)
    round1_brackets = math.ceil(num_options / b_size)
    return 1 + math.ceil(round1_brackets / s_cap)


@dataclass(frozen=True)
class WideRoutingRecommendation:
    """Empirically validated wide-option routing recommendation for a catalog of size ``K``."""

    num_options: int
    strategy: str
    label_scheme: str
    expected_passes: float
    avoids_unpartitioned_trie_aliasing: bool
    rationale: str


def recommend_wide_routing_strategy(
    num_options: int,
    *,
    supports_az_aa_52: bool = True,
    supports_extended_255: bool = False,
    has_embeddings: bool = False,
    supports_multi_slot_canvas: bool = True,
) -> WideRoutingRecommendation:
    """Select the empirically optimal wide-option readout strategy for ``num_options`` candidates.

    Codifies the benchmarked routing ladder across ``dgem`` and ``MDM-Bench``:
      1. ``2 <= K <= 26``: ``direct_az`` (single-token ``[A-Z]`` readout, 1 pass).
      2. ``27 <= K <= 52`` (when ``supports_az_aa_52=True``): ``direct_az_aa`` (52-option ``A..Z, AA..AZ``, 1 pass).
      3. ``53 <= K <= 255`` (when ``supports_extended_255=True``): ``direct_universal_255`` (``UNIVERSAL_256_DECISION_CODES``, 1 pass).
      4. ``K > 26`` with dense embeddings (``has_embeddings=True``): ``csls_shortlist_1pass`` (CSLS Top-20 + cosine ``prior_logits``, 1 pass).
      5. ``K <= 176`` on multi-slot canvas (``supports_multi_slot_canvas=True``): ``mtp_partitioned_1pass`` (1 coarse + <= 7 disjoint fine bands, 1 pass).
      6. Larger taxonomies without embeddings: ``hesitation_hybrid`` (2-level ``[A-Z]`` taxonomy with conditional Pass-2 beam, ~1.2 passes).

    Unpartitioned 1-pass ``mtp_trie`` is intentionally omitted from recommendations because shared fine codes
    marginalize across coarse parents when ``K > 26``.
    """
    if isinstance(num_options, bool) or not isinstance(num_options, int) or num_options < 2:
        raise ValueError(f"num_options must be an integer >= 2, got {num_options!r}")

    if num_options <= MAX_OPTIONS_PER_SLOT:
        return WideRoutingRecommendation(
            num_options=num_options,
            strategy="direct_az",
            label_scheme="az",
            expected_passes=1.0,
            avoids_unpartitioned_trie_aliasing=True,
            rationale="Fits directly in a single-pass <= 26 [A-Z] decision slot.",
        )

    if supports_az_aa_52 and num_options <= 52:
        return WideRoutingRecommendation(
            num_options=num_options,
            strategy="direct_az_aa",
            label_scheme="az_aa",
            expected_passes=1.0,
            avoids_unpartitioned_trie_aliasing=True,
            rationale="Fits in a single-pass 52-option (az_aa) slot without hierarchical fine-code aliasing.",
        )

    if supports_extended_255 and num_options <= 255:
        return WideRoutingRecommendation(
            num_options=num_options,
            strategy="direct_universal_255",
            label_scheme="universal_256",
            expected_passes=1.0,
            avoids_unpartitioned_trie_aliasing=True,
            rationale="Uses 255 BPE-safe single-token uppercase codes (UNIVERSAL_256_DECISION_CODES) in 1 pass.",
        )

    if has_embeddings:
        return WideRoutingRecommendation(
            num_options=num_options,
            strategy="csls_shortlist_1pass",
            label_scheme="az",
            expected_passes=1.0,
            avoids_unpartitioned_trie_aliasing=True,
            rationale="Prunes wide catalog via CSLS Top-20 shortlist with log-softmax cosine prior_logits in 1 pass.",
        )

    max_partitioned_leaves = (MAX_SLOTS_PER_PASS - 1) * (MAX_OPTIONS_PER_SLOT - 1) + 1
    if supports_multi_slot_canvas and num_options <= max_partitioned_leaves:
        return WideRoutingRecommendation(
            num_options=num_options,
            strategy="mtp_partitioned_1pass",
            label_scheme="az",
            expected_passes=1.0,
            avoids_unpartitioned_trie_aliasing=True,
            rationale="Packs 1 coarse slot plus <= 7 disjoint fine bands onto an 8-slot canvas with zero aliasing.",
        )

    return WideRoutingRecommendation(
        num_options=num_options,
        strategy="hesitation_hybrid",
        label_scheme="az",
        expected_passes=1.2,
        avoids_unpartitioned_trie_aliasing=True,
        rationale="Executes Pass-2 conditional top-b beam refinement whenever coarse hesitation exceeds threshold.",
    )


@dataclass(frozen=True)
class TaxonomyLeaf:
    """Maps a leaf option key to its (coarse_code, fine_code) coordinates in a 2-level taxonomy."""

    leaf_key: str
    coarse_key: str
    fine_key: str
    description: str = ""


@dataclass
class TaxonomyTrie:
    """Two-level semantic taxonomy where every coarse and fine slot stays within <= 26 [A-Z] options."""

    coarse_criteria: dict[str, str]
    fine_criteria: dict[str, str]
    leaves: dict[str, TaxonomyLeaf]
    catch_all_key: str = ""
    coarse_order: list[str] = field(default_factory=list)
    fine_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.coarse_order:
            self.coarse_order = list(self.coarse_criteria.keys())
        if not self.fine_order:
            self.fine_order = list(self.fine_criteria.keys())
        if len(self.coarse_criteria) > MAX_OPTIONS_PER_SLOT:
            raise ValueError(
                f"coarse_criteria has {len(self.coarse_criteria)} options; must be <= {MAX_OPTIONS_PER_SLOT}"
            )
        if len(self.fine_criteria) > MAX_OPTIONS_PER_SLOT:
            raise ValueError(
                f"fine_criteria has {len(self.fine_criteria)} options; must be <= {MAX_OPTIONS_PER_SLOT}"
            )

    def domain_members(self, coarse_key: str) -> list[TaxonomyLeaf]:
        return [
            leaf
            for leaf in self.leaves.values()
            if leaf.coarse_key == coarse_key and leaf.leaf_key != self.catch_all_key
        ]


def _build_lexical_coarse_groups(
    items: Sequence[tuple[str, str]],
    cap_coarse: int,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """Bucket opaque keys (e.g., option_0 .. option_150) by informative description stems."""
    tok_lists: dict[str, list[str]] = {}
    doc_freq: dict[str, int] = {}
    for k, desc in items:
        toks = split_identifier_tokens(desc)
        tok_lists[k] = toks
        for t in toks:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    upper_df = max(2, len(items) // 3)
    raw_groups: dict[str, list[tuple[str, str]]] = {}
    raw_suffixes: dict[str, str] = {}
    for k, desc in items:
        toks = tok_lists[k]
        candidates = [t for t in toks if 2 <= doc_freq.get(t, 0) <= upper_df]
        if candidates:
            # Prefer moderately shared topic nouns (highest doc_freq below upper_df)
            candidates.sort(key=lambda t: (-doc_freq[t], toks.index(t)))
            dom = f"topic_{candidates[0]}"
            rem = [t for t in toks if t != candidates[0]]
            suf = rem[0] if rem else k
        elif toks:
            dom = f"topic_{toks[0]}"
            suf = toks[1] if len(toks) > 1 else k
        else:
            dom = "topic_misc"
            suf = k
        raw_groups.setdefault(dom, []).append((k, desc))
        raw_suffixes[k] = suf
    return raw_groups, raw_suffixes


def build_prefix_taxonomy_trie(
    criteria: Mapping[str, str],
    domain_map: Mapping[str, str] | None = None,
    catch_all_key: str = "",
    max_coarse: int = 25,
) -> TaxonomyTrie:
    """Construct a valid <=26 x <=26 TaxonomyTrie from domain prefixes, lexical stems, or domain_map.

    1. Groups leaves by explicit domain_map, prefix tokens (`iot_hue_lightdim`, `drive.files_list`),
       or natural-language description stems when keys are opaque (`option_0 .. option_150`).
    2. Aligns unique cross-domain action suffixes (`query`, `set`, `list`) into shared fine roles
       while packing any remaining domain-specific intents into compact residual slots (`member_01..M`),
       preserving 100% lossless root-to-leaf decoding within <= 26 options per slot.
    """
    detected_catch_all = catch_all_key
    if not detected_catch_all:
        for cand in ("oos", "none_of_the_above", "other", "unknown"):
            if cand in criteria:
                detected_catch_all = cand
                break

    non_catch_items = [(k, desc) for k, desc in criteria.items() if k != detected_catch_all]
    cap_coarse = min(max_coarse, MAX_OPTIONS_PER_SLOT - (1 if detected_catch_all else 0))

    opaque_count = sum(1 for k, _ in non_catch_items if _OPAQUE_KEY_RE.match(k))
    if not domain_map and non_catch_items and opaque_count >= max(2, int(0.8 * len(non_catch_items))):
        raw_groups, raw_suffixes = _build_lexical_coarse_groups(non_catch_items, cap_coarse)
    else:
        raw_groups = {}
        raw_suffixes = {}
        for key, desc in non_catch_items:
            if domain_map and key in domain_map:
                dom = domain_map[key]
                parts = _SPLIT_PREFIX_RE.split(key, maxsplit=1)
                suf = parts[1] if len(parts) > 1 else key
            else:
                parts = _SPLIT_PREFIX_RE.split(key, maxsplit=1)
                dom = parts[0] if len(parts) > 1 else key[:3].lower()
                suf = parts[1] if len(parts) > 1 else key
            raw_groups.setdefault(dom, []).append((key, desc))
            raw_suffixes[key] = suf

    # Merge smallest tail domains if domain count exceeds cap_coarse, ensuring each group <= 25 leaves.
    sorted_doms = sorted(raw_groups.keys(), key=lambda d: (-len(raw_groups[d]), d))
    groups: dict[str, list[tuple[str, str]]] = {}
    overflow_idx = 1
    max_per_group = MAX_OPTIONS_PER_SLOT - (1 if detected_catch_all else 0)
    for dom in sorted_doms:
        members = raw_groups[dom]
        for chunk_start in range(0, len(members), max_per_group):
            chunk = members[chunk_start : chunk_start + max_per_group]
            chunk_name = dom if chunk_start == 0 else f"{dom}_{chunk_start // max_per_group + 1}"
            if len(groups) < cap_coarse - 1 or len(sorted_doms) <= cap_coarse:
                groups[chunk_name] = list(chunk)
            else:
                misc_key = f"other_domain_{overflow_idx}"
                while misc_key in groups and len(groups[misc_key]) + len(chunk) > max_per_group:
                    overflow_idx += 1
                    misc_key = f"other_domain_{overflow_idx}"
                groups.setdefault(misc_key, []).extend(chunk)

    if len(groups) > cap_coarse:
        raise ValueError(
            f"cannot pack {len(criteria)} options into <= {cap_coarse} domains of <= {max_per_group} items"
        )

    coarse_criteria: dict[str, str] = {}
    for dom, members in groups.items():
        sample_names = ", ".join(k for k, _ in members[:5])
        more = f" (+{len(members) - 5} more)" if len(members) > 5 else ""
        coarse_criteria[dom] = f"Domain '{dom}' covering: {sample_names}{more}"
    if detected_catch_all and detected_catch_all in criteria:
        coarse_criteria[detected_catch_all] = criteria[detected_catch_all]

    fine_budget = MAX_OPTIONS_PER_SLOT - (1 if detected_catch_all else 0)
    distinct_suffixes: list[str] = []
    suffix_collision = False
    suffix_dom_counts: dict[str, int] = {}
    suffix_has_dup_in_dom: set[str] = set()

    for dom, members in groups.items():
        seen_in_dom: set[str] = set()
        for k, _ in members:
            s = raw_suffixes[k]
            if s in seen_in_dom:
                suffix_collision = True
                suffix_has_dup_in_dom.add(s)
            else:
                suffix_dom_counts[s] = suffix_dom_counts.get(s, 0) + 1
            seen_in_dom.add(s)
            if s not in distinct_suffixes:
                distinct_suffixes.append(s)

    fine_criteria: dict[str, str] = {}
    leaves: dict[str, TaxonomyLeaf] = {}

    if not suffix_collision and 2 <= len(distinct_suffixes) <= fine_budget:
        for s in distinct_suffixes:
            examples = [criteria[k] for k, s_val in raw_suffixes.items() if s_val == s][:3]
            fine_criteria[s] = f"Sub-intent '{s}': " + "; ".join(examples)
        for dom, members in groups.items():
            for k, desc in members:
                leaves[k] = TaxonomyLeaf(
                    leaf_key=k,
                    coarse_key=dom,
                    fine_key=raw_suffixes[k],
                    description=desc,
                )
    else:
        # Partial cross-domain role alignment: promote suffixes shared across >= 2 domains
        # (with no intra-domain collision) when shared roles + max residual per domain <= fine_budget.
        shared_candidates = sorted(
            (s for s, cnt in suffix_dom_counts.items() if cnt >= 2 and s not in suffix_has_dup_in_dom),
            key=lambda s: (-suffix_dom_counts[s], s),
        )
        promoted_roles: list[str] = []
        for s in shared_candidates:
            trial = set(promoted_roles) | {s}
            max_rem = max(
                sum(1 for k, _ in members if raw_suffixes[k] not in trial)
                for members in groups.values()
            )
            if len(trial) + max_rem <= fine_budget:
                promoted_roles.append(s)

        promoted_set = set(promoted_roles)
        for s in promoted_roles:
            role_key = f"role_{s}"
            examples = [criteria[k] for k, s_val in raw_suffixes.items() if s_val == s][:3]
            fine_criteria[role_key] = f"Action role '{s}': " + "; ".join(examples)
            for dom, members in groups.items():
                for k, desc in members:
                    if raw_suffixes[k] == s:
                        leaves[k] = TaxonomyLeaf(
                            leaf_key=k,
                            coarse_key=dom,
                            fine_key=role_key,
                            description=desc,
                        )

        residuals_by_dom: dict[str, list[tuple[str, str]]] = {
            dom: [(k, desc) for k, desc in members if raw_suffixes[k] not in promoted_set]
            for dom, members in groups.items()
        }
        max_res_len = max((len(m) for m in residuals_by_dom.values()), default=0)
        if not fine_criteria:
            max_res_len = max(2, max_res_len)

        for idx in range(max_res_len):
            slot_key = f"member_{idx + 1:02d}"
            slot_examples: list[str] = []
            for dom, rem_members in residuals_by_dom.items():
                if idx < len(rem_members):
                    leaf_k, leaf_desc = rem_members[idx]
                    short_desc = leaf_desc if len(leaf_desc) <= 48 else leaf_desc[:45] + "..."
                    slot_examples.append(f"[{dom}] {leaf_k} ({short_desc})")
                    leaves[leaf_k] = TaxonomyLeaf(
                        leaf_key=leaf_k,
                        coarse_key=dom,
                        fine_key=slot_key,
                        description=leaf_desc,
                    )
            if slot_examples or slot_key not in fine_criteria:
                fine_criteria[slot_key] = "; ".join(slot_examples) if slot_examples else f"Residual slot {idx + 1}"

    if detected_catch_all and detected_catch_all in criteria:
        fine_criteria[detected_catch_all] = criteria[detected_catch_all]
        leaves[detected_catch_all] = TaxonomyLeaf(
            leaf_key=detected_catch_all,
            coarse_key=detected_catch_all,
            fine_key=detected_catch_all,
            description=criteria[detected_catch_all],
        )

    return TaxonomyTrie(
        coarse_criteria=coarse_criteria,
        fine_criteria=fine_criteria,
        leaves=leaves,
        catch_all_key=detected_catch_all,
    )


def trie_constrained_joint_decode(
    coarse_probs: Mapping[str, float],
    fine_probs: Mapping[str, float],
    trie: TaxonomyTrie,
) -> tuple[dict[str, float], str]:
    """Decode joint leaf probabilities strictly over valid paths in `trie`.

    Computes s(e | x) = log P_coarse(c_1(e) | x) + log P_fine(c_2(e) | x) for every valid
    leaf e in T and normalizes via log-sum-exp, eliminating invalid cross-slot combinations.
    """
    eps = 1e-12
    c_tot = sum(max(0.0, float(v)) for v in coarse_probs.values()) or 1.0
    f_tot = sum(max(0.0, float(v)) for v in fine_probs.values()) or 1.0

    log_scores: dict[str, float] = {}
    for leaf_key, leaf in trie.leaves.items():
        if trie.catch_all_key and leaf_key == trie.catch_all_key:
            # Preserve direct coarse OOS probability calibration without squaring small probabilities
            p_c = max(eps, float(coarse_probs.get(leaf.coarse_key, 0.0)) / c_tot)
            p_f = max(eps, float(fine_probs.get(leaf.fine_key, p_c)) / f_tot)
            log_scores[leaf_key] = 0.5 * (math.log(p_c) + math.log(p_f))
        else:
            p_c = max(eps, float(coarse_probs.get(leaf.coarse_key, 0.0)) / c_tot)
            p_f = max(eps, float(fine_probs.get(leaf.fine_key, 0.0)) / f_tot)
            log_scores[leaf_key] = math.log(p_c) + math.log(p_f)

    if not log_scores:
        return {}, ""

    max_lp = max(log_scores.values())
    exps = {k: math.exp(lp - max_lp) for k, lp in log_scores.items()}
    denom = sum(exps.values())
    leaf_probs = {k: v / denom for k, v in exps.items()}
    winner = max(leaf_probs.items(), key=lambda kv: kv[1])[0]
    return leaf_probs, winner


def sinkhorn_dealiased_trie_decode(
    coarse_probs: Mapping[str, float],
    fine_probs: Mapping[str, float],
    trie: TaxonomyTrie,
    *,
    leaf_prior_weights: Mapping[str, float] | None = None,
    num_iters: int = 8,
) -> tuple[dict[str, float], str]:
    """De-alias 1-pass (coarse, fine) marginals via Sinkhorn-Knopp / IPF I-projection on `trie`.

    In 1-pass parallel readout, fine_probs[c_2] is the column marginal summed over all coarse
    domains sharing fine code c_2: P_fine(c_2) = sum_{c_1 : (c_1, c_2) in T} Q(c_1, c_2).
    Alternating Bregman KL row/column projections (Csiszar, Ann. Probab. 1975) onto the sparse
    trie support T recovers the minimum-KL coupling Q^*(c_1, c_2) consistent with both slot
    marginals while preventing high-collision fine codes from dominating low-collision roles.

    Args:
        coarse_probs: Coarse domain probability distribution P_1(c_1 | x).
        fine_probs: Fine slot marginal probability distribution P_2(c_2 | x).
        trie: Two-level TaxonomyTrie defining valid root-to-leaf edges (c_1, c_2).
        leaf_prior_weights: Optional positive prior affinity weights A(e) per leaf_key.
        num_iters: Number of alternating row/column IPF scaling iterations (>= 1).

    Returns:
        Tuple `(leaf_probs, winner_key)` where `leaf_probs` sums to 1.0 over `trie.leaves`.
    """
    if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError(f"num_iters must be an integer >= 1, got {num_iters!r}")
    if not trie.leaves:
        return {}, ""

    eps = 1e-12
    c_tot = sum(max(0.0, float(v)) for v in coarse_probs.values()) or 1.0
    f_tot = sum(max(0.0, float(v)) for v in fine_probs.values()) or 1.0
    p_coarse = {k: max(eps, float(coarse_probs.get(k, 0.0)) / c_tot) for k in trie.coarse_criteria}
    p_fine = {k: max(eps, float(fine_probs.get(k, 0.0)) / f_tot) for k in trie.fine_criteria}

    # Initialize sparse coupling Q^{(0)}(e) on valid trie leaves e = (c_1, c_2) in T
    coupling: dict[str, float] = {}
    for leaf_key, leaf in trie.leaves.items():
        prior_w = 1.0
        if leaf_prior_weights is not None:
            prior_w = max(eps, float(leaf_prior_weights.get(leaf_key, 1.0)))
        coupling[leaf_key] = prior_w * p_coarse.get(leaf.coarse_key, eps) * p_fine.get(leaf.fine_key, eps)

    # Alternating Bregman KL projection onto column (fine) and row (coarse) marginals
    for _ in range(num_iters):
        # 1. Column scaling: match fine slot marginal P_2(c_2 | x) across aliased leaves sharing c_2
        col_sums: dict[str, float] = {}
        for leaf_key, leaf in trie.leaves.items():
            col_sums[leaf.fine_key] = col_sums.get(leaf.fine_key, 0.0) + coupling[leaf_key]
        for leaf_key, leaf in trie.leaves.items():
            c_sum = col_sums.get(leaf.fine_key, 0.0)
            if c_sum > eps:
                coupling[leaf_key] *= p_fine.get(leaf.fine_key, eps) / c_sum

        # 2. Row scaling: match coarse domain marginal P_1(c_1 | x) across leaves in domain c_1
        row_sums: dict[str, float] = {}
        for leaf_key, leaf in trie.leaves.items():
            row_sums[leaf.coarse_key] = row_sums.get(leaf.coarse_key, 0.0) + coupling[leaf_key]
        for leaf_key, leaf in trie.leaves.items():
            r_sum = row_sums.get(leaf.coarse_key, 0.0)
            if r_sum > eps:
                coupling[leaf_key] *= p_coarse.get(leaf.coarse_key, eps) / r_sum

    total_q = sum(coupling.values())
    if total_q <= 0.0:
        return trie_constrained_joint_decode(coarse_probs, fine_probs, trie)

    leaf_probs = {k: v / total_q for k, v in coupling.items()}
    winner = max(leaf_probs.items(), key=lambda kv: kv[1])[0]
    return leaf_probs, winner


class HierarchicalDecisionRouter:
    """Executes wide-option (>26) taxonomy routing across 6 Semantic-ID decoding arms."""

    VALID_MODES = (
        "mtp_naive",
        "mtp_trie",
        "mtp_sinkhorn",
        "mtp_partitioned_1pass",
        "beam_2pass",
        "hesitation_hybrid",
    )

    def __init__(
        self,
        backend: DecisionBackend,
        mode: str = "hesitation_hybrid",
        beam_width: int = 2,
        hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD,
        prior_weight: float = 0.35,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        self.backend = backend
        self.mode = mode
        self.beam_width = beam_width
        self.hesitation_threshold = hesitation_threshold
        self.prior_weight = prior_weight

    def route(
        self,
        state: Any,
        slot_id: str,
        question: DecisionQuestion,
        trie: TaxonomyTrie | None = None,
        images: list[str] | None = None,
    ) -> SlotReadout:
        keys = question.option_keys()
        if len(keys) <= MAX_OPTIONS_PER_SLOT:
            dists = self.backend.score_questions(state, {slot_id: question}, images)
            raw = dists.get(slot_id, {})
            h = normalized_entropy(raw)
            pred = max(raw.items(), key=lambda kv: kv[1])[0] if raw else ""
            return SlotReadout(
                slot_id=slot_id,
                predicted=pred,
                probabilities=raw,
                raw_probabilities=raw,
                h_raw=h,
                confidence=1.0 - h,
                backend_name=self.backend.name,
                passes_used=1,
            )

        active_trie = trie or build_prefix_taxonomy_trie(question.criteria)
        coarse_q = DecisionQuestion(
            type="choice",
            instructions=f"{question.instructions} (Select the coarse category/domain.)",
            criteria=dict(active_trie.coarse_criteria),
            order=list(active_trie.coarse_order),
        )
        fine_q = DecisionQuestion(
            type="choice",
            instructions=f"{question.instructions} (Select the specific sub-intent within the category.)",
            criteria=dict(active_trie.fine_criteria),
            order=list(active_trie.fine_order),
        )

        if self.mode == "beam_2pass":
            p1 = self.backend.score_questions(state, {f"{slot_id}_coarse": coarse_q}, images)
            coarse_dist = p1.get(f"{slot_id}_coarse", {})
            h_coarse = normalized_entropy(coarse_dist)
            return self._run_pass2_beam(
                state=state,
                slot_id=slot_id,
                question=question,
                trie=active_trie,
                coarse_dist=coarse_dist,
                h_coarse=h_coarse,
                images=images,
                passes_so_far=1,
            )

        if self.mode == "mtp_partitioned_1pass":
            part_readout = self._try_partitioned_1pass(
                state=state,
                slot_id=slot_id,
                question=question,
                trie=active_trie,
                coarse_q=coarse_q,
                images=images,
            )
            if part_readout is not None:
                return part_readout

        # Single-pass parallel readout of (coarse, fine) slots on the same canvas
        p1 = self.backend.score_questions(
            state,
            {f"{slot_id}_coarse": coarse_q, f"{slot_id}_fine": fine_q},
            images,
        )
        coarse_dist = p1.get(f"{slot_id}_coarse", {})
        fine_dist = p1.get(f"{slot_id}_fine", {})
        h_coarse = normalized_entropy(coarse_dist)

        if self.mode == "mtp_naive":
            best_c = max(coarse_dist.items(), key=lambda kv: kv[1])[0] if coarse_dist else ""
            best_f = max(fine_dist.items(), key=lambda kv: kv[1])[0] if fine_dist else ""
            joint_probs, _ = trie_constrained_joint_decode(coarse_dist, fine_dist, active_trie)
            exact_match = ""
            for lk, leaf in active_trie.leaves.items():
                if leaf.coarse_key == best_c and leaf.fine_key == best_f:
                    exact_match = lk
                    break
            # Enforce Level-4 Total-Contract domain closure: never pollute predicted with out-of-catalog string sentinels.
            if exact_match:
                return SlotReadout(
                    slot_id=slot_id,
                    predicted=exact_match,
                    probabilities=joint_probs,
                    raw_probabilities=coarse_dist,
                    h_raw=h_coarse,
                    confidence=1.0 - h_coarse,
                    guard_applied="none",
                    backend_name=self.backend.name,
                    passes_used=1,
                    terminal_state=DecisionTerminalState.STAGE1_RESOLVED,
                    failure_category=FailureCategory.NONE,
                )
            return SlotReadout(
                slot_id=slot_id,
                predicted="",
                probabilities=joint_probs,
                raw_probabilities=coarse_dist,
                h_raw=h_coarse,
                confidence=1.0 - h_coarse,
                guard_applied="mtp_naive_unconstrained",
                backend_name=self.backend.name,
                passes_used=1,
                terminal_state=DecisionTerminalState.INVALID_TRIE_PATH,
                failure_category=FailureCategory.INVALID_UNCONSTRAINED_TUPLE,
                invalid_coordinates=(best_c, best_f),
            )

        if self.mode in ("mtp_sinkhorn", "mtp_partitioned_1pass"):
            joint_probs, trie_winner = sinkhorn_dealiased_trie_decode(
                coarse_dist,
                fine_dist,
                active_trie,
            )
            h_joint = normalized_entropy(joint_probs)
            return SlotReadout(
                slot_id=slot_id,
                predicted=trie_winner,
                probabilities=joint_probs,
                raw_probabilities=coarse_dist,
                h_raw=h_coarse,
                confidence=1.0 - h_joint,
                guard_applied="trie_sinkhorn_1pass",
                backend_name=self.backend.name,
                passes_used=1,
            )

        joint_probs, trie_winner = trie_constrained_joint_decode(coarse_dist, fine_dist, active_trie)
        h_joint = normalized_entropy(joint_probs)

        if self.mode == "mtp_trie" or h_coarse < self.hesitation_threshold:
            return SlotReadout(
                slot_id=slot_id,
                predicted=trie_winner,
                probabilities=joint_probs,
                raw_probabilities=coarse_dist,
                h_raw=h_coarse,
                confidence=1.0 - h_joint,
                guard_applied="trie_1pass",
                backend_name=self.backend.name,
                passes_used=1,
            )

        # hesitation_hybrid with H_coarse >= threshold: run Pass-2 top-b conditional expansion
        return self._run_pass2_beam(
            state=state,
            slot_id=slot_id,
            question=question,
            trie=active_trie,
            coarse_dist=coarse_dist,
            h_coarse=h_coarse,
            images=images,
            passes_so_far=1,
        )

    def _try_partitioned_1pass(
        self,
        state: Any,
        slot_id: str,
        question: DecisionQuestion,
        trie: TaxonomyTrie,
        coarse_q: DecisionQuestion,
        images: list[str] | None,
    ) -> SlotReadout | None:
        """Pack up to 7 disjoint fine sub-slots alongside 1 coarse slot in a single 8-slot pass.

        Eliminates cross-domain fine-code aliasing for catalogs up to 7 * 25 + 1 = 176 options
        while keeping passes_used = 1.
        """
        max_bands = MAX_SLOTS_PER_PASS - 1
        band_cap = MAX_OPTIONS_PER_SLOT - 1
        bands: list[list[TaxonomyLeaf]] = []
        current_band: list[TaxonomyLeaf] = []

        for dom in trie.coarse_order:
            if dom == trie.catch_all_key:
                continue
            members = trie.domain_members(dom)
            if not members:
                continue
            if len(members) > band_cap:
                return None
            if current_band and len(current_band) + len(members) > band_cap:
                bands.append(current_band)
                current_band = list(members)
            else:
                current_band.extend(members)
        if current_band:
            bands.append(current_band)

        if not bands or len(bands) > max_bands:
            return None

        complement_key = trie.catch_all_key or "__other_band__"
        complement_desc = (
            question.criteria.get(trie.catch_all_key, "None of the above in this band")
            if trie.catch_all_key
            else "Intent belongs to another coarse domain band"
        )

        questions_1pass: dict[str, DecisionQuestion] = {f"{slot_id}_coarse": coarse_q}
        leaf_to_band: dict[str, str] = {}
        for b_idx, band_leaves in enumerate(bands):
            b_slot = f"{slot_id}_band_{b_idx + 1}"
            b_crit: dict[str, str] = {}
            b_order: list[str] = []
            for leaf in band_leaves:
                b_crit[leaf.leaf_key] = leaf.description or question.criteria.get(leaf.leaf_key, leaf.leaf_key)
                b_order.append(leaf.leaf_key)
                leaf_to_band[leaf.leaf_key] = b_slot
            b_crit[complement_key] = complement_desc
            b_order.append(complement_key)
            questions_1pass[b_slot] = DecisionQuestion(
                type="choice",
                instructions=f"{question.instructions} (Select the matching leaf intent in band {b_idx + 1}.)",
                criteria=b_crit,
                order=b_order,
            )

        dists = self.backend.score_questions(state, questions_1pass, images)
        coarse_dist = dists.get(f"{slot_id}_coarse", {})
        h_coarse = normalized_entropy(coarse_dist)

        eps = 1e-12
        c_tot = sum(max(0.0, float(v)) for v in coarse_dist.values()) or 1.0
        domain_band_totals: dict[str, float] = {}
        for leaf_key, leaf in trie.leaves.items():
            if trie.catch_all_key and leaf_key == trie.catch_all_key:
                continue
            b_slot = leaf_to_band.get(leaf_key, "")
            b_dist = dists.get(b_slot, {})
            domain_band_totals[leaf.coarse_key] = domain_band_totals.get(leaf.coarse_key, 0.0) + max(
                eps, float(b_dist.get(leaf_key, 0.0))
            )

        raw_joint: dict[str, float] = {}
        for leaf_key, leaf in trie.leaves.items():
            p_c = max(eps, float(coarse_dist.get(leaf.coarse_key, 0.0)) / c_tot)
            if trie.catch_all_key and leaf_key == trie.catch_all_key:
                raw_joint[leaf_key] = p_c
            else:
                b_slot = leaf_to_band.get(leaf_key, "")
                b_dist = dists.get(b_slot, {})
                dom_tot = domain_band_totals.get(leaf.coarse_key, 0.0) or 1.0
                p_cond = max(eps, float(b_dist.get(leaf_key, 0.0))) / dom_tot
                raw_joint[leaf_key] = p_c * p_cond

        denom = sum(raw_joint.values()) or 1.0
        joint_probs = {k: v / denom for k, v in raw_joint.items()}
        winner = max(joint_probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        h_joint = normalized_entropy(joint_probs)

        return SlotReadout(
            slot_id=slot_id,
            predicted=winner,
            probabilities=joint_probs,
            raw_probabilities=coarse_dist,
            h_raw=h_coarse,
            confidence=1.0 - h_joint,
            guard_applied=f"partitioned_1pass_{len(bands)}bands",
            backend_name=self.backend.name,
            passes_used=1,
        )

    def _run_pass2_beam(
        self,
        state: Any,
        slot_id: str,
        question: DecisionQuestion,
        trie: TaxonomyTrie,
        coarse_dist: Mapping[str, float],
        h_coarse: float,
        images: list[str] | None,
        passes_so_far: int,
    ) -> SlotReadout:
        non_catch_doms = [d for d in trie.coarse_order if d != trie.catch_all_key]
        ranked_doms = sorted(
            non_catch_doms,
            key=lambda d: float(coarse_dist.get(d, 0.0)),
            reverse=True,
        )
        max_leaf_budget = MAX_OPTIONS_PER_SLOT - (1 if trie.catch_all_key else 0)
        candidate_leaves: list[str] = []
        domain_log_priors: dict[str, float] = {}

        active_doms = [d for d in ranked_doms[: self.beam_width] if trie.domain_members(d)]
        dom_members = {d: trie.domain_members(d) for d in active_doms}
        total_avail = sum(len(dom_members[d]) for d in active_doms)

        if total_avail <= max_leaf_budget:
            for dom in active_doms:
                p_dom = max(1e-9, float(coarse_dist.get(dom, 1e-9)))
                for leaf in dom_members[dom]:
                    candidate_leaves.append(leaf.leaf_key)
                    domain_log_priors[leaf.leaf_key] = math.log(p_dom)
        elif active_doms:
            # Probability-proportional per-domain leaf quota with a guaranteed floor so a 25-leaf rank-1
            # domain never starves rank-2..b domains in Pass 2.
            prob_mass = sum(max(1e-6, float(coarse_dist.get(d, 1e-6))) for d in active_doms)
            min_floor = max(2, max_leaf_budget // (2 * len(active_doms)))
            if min_floor * len(active_doms) > max_leaf_budget:
                min_floor = max(1, max_leaf_budget // len(active_doms))

            taken: dict[str, int] = {}
            remaining = max_leaf_budget
            for dom in active_doms:
                raw_share = round(max_leaf_budget * (max(1e-6, float(coarse_dist.get(dom, 1e-6))) / prob_mass))
                share = max(min_floor, int(raw_share))
                share = min(share, len(dom_members[dom]), remaining)
                taken[dom] = share
                remaining -= share

            while remaining > 0:
                progressed = False
                for dom in active_doms:
                    if remaining == 0:
                        break
                    if taken[dom] < len(dom_members[dom]):
                        taken[dom] += 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    break

            for dom in active_doms:
                p_dom = max(1e-9, float(coarse_dist.get(dom, 1e-9)))
                for leaf in dom_members[dom][: taken[dom]]:
                    candidate_leaves.append(leaf.leaf_key)
                    domain_log_priors[leaf.leaf_key] = math.log(p_dom)

        if trie.catch_all_key and trie.catch_all_key in question.criteria:
            candidate_leaves.append(trie.catch_all_key)
            p_oos = max(1e-9, float(coarse_dist.get(trie.catch_all_key, 1e-9)))
            domain_log_priors[trie.catch_all_key] = math.log(p_oos)

        sub_crit = {k: question.criteria[k] for k in candidate_leaves if k in question.criteria}
        sub_q = DecisionQuestion(
            type="choice",
            instructions=question.instructions,
            criteria=sub_crit,
            order=list(candidate_leaves),
        )
        p2 = self.backend.score_questions(state, {slot_id: sub_q}, images)
        raw_p2 = p2.get(slot_id, {})
        fused_p2, h_p2, _ = pre_gated_bayesian_poe_fusion(
            raw_probs=raw_p2,
            prior_logits=domain_log_priors,
            prior_weight=self.prior_weight,
            hesitation_threshold=self.hesitation_threshold,
            gate_mode="hesitant",
        )
        winner = max(fused_p2.items(), key=lambda kv: kv[1])[0] if fused_p2 else ""
        # Preserve max(h_coarse, h_p2) so coarse OOS uncertainty is never hidden
        effective_h = max(h_coarse, h_p2)
        return SlotReadout(
            slot_id=slot_id,
            predicted=winner,
            probabilities=fused_p2,
            raw_probabilities=raw_p2,
            h_raw=effective_h,
            confidence=1.0 - effective_h,
            escalated=True,
            guard_applied=f"beam_top{self.beam_width}_pass2",
            backend_name=self.backend.name,
            passes_used=passes_so_far + 1,
        )

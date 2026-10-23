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

Translates semantic-ID hierarchy routing (arXiv:2610.15864v1) to zero-shot single-token [A-Z]
decision models without vocabulary expansion or 52-label OOS entropy degradation:

  1. mtp_naive (reads=1): Simultaneous coarse + fine slots decoded via independent marginal argmax.
  2. mtp_trie (reads=1): Simultaneous coarse + fine slots decoded strictly over valid root-to-leaf
     paths e in T via exact joint log-probability s(e | x) = log P_1(c_1(e) | x) + log P_2(c_2(e) | x).
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
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    DecisionQuestion,
    SlotReadout,
)

if TYPE_CHECKING:
    from .backends import DecisionBackend

_SPLIT_PREFIX_RE = re.compile(r"[._:/-]+")


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


def build_prefix_taxonomy_trie(
    criteria: Mapping[str, str],
    domain_map: Mapping[str, str] | None = None,
    catch_all_key: str = "",
    max_coarse: int = 25,
) -> TaxonomyTrie:
    """Construct a valid <=26 x <=26 TaxonomyTrie from domain prefixes or an explicit domain_map.

    For schemas like MASSIVE (`iot_hue_lightdim`, `calendar_set`), CLINC150 (10 domains x 15 intents + `oos`),
    or `workspace-tools-202` (`drive.files_list`), groups leaves by coarse domain and assigns each member
    either a shared functional action role (when <= 25 distinct suffixes exist) or compact slot indices
    (`slot_01`..`slot_M`) with aggregated per-slot semantic summaries.
    """
    detected_catch_all = catch_all_key
    if not detected_catch_all:
        for cand in ("oos", "none_of_the_above", "other", "unknown"):
            if cand in criteria:
                detected_catch_all = cand
                break

    raw_groups: dict[str, list[tuple[str, str]]] = {}
    raw_suffixes: dict[str, str] = {}

    for key, desc in criteria.items():
        if key == detected_catch_all:
            continue
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

    # Merge smallest tail domains if domain count exceeds max_coarse, ensuring every group <= 25 leaves.
    cap_coarse = min(max_coarse, MAX_OPTIONS_PER_SLOT - (1 if detected_catch_all else 0))
    sorted_doms = sorted(raw_groups.keys(), key=lambda d: (-len(raw_groups[d]), d))
    groups: dict[str, list[tuple[str, str]]] = {}
    overflow_idx = 1
    for dom in sorted_doms:
        members = raw_groups[dom]
        # Split oversized domains into <=25 chunks if needed
        for chunk_start in range(0, len(members), MAX_OPTIONS_PER_SLOT - 1):
            chunk = members[chunk_start : chunk_start + (MAX_OPTIONS_PER_SLOT - 1)]
            chunk_name = dom if chunk_start == 0 else f"{dom}_{chunk_start // (MAX_OPTIONS_PER_SLOT - 1) + 1}"
            if len(groups) < cap_coarse - 1 or len(sorted_doms) <= cap_coarse:
                groups[chunk_name] = chunk
            else:
                misc_key = f"other_domain_{overflow_idx}"
                while misc_key in groups and len(groups[misc_key]) + len(chunk) > MAX_OPTIONS_PER_SLOT - 1:
                    overflow_idx += 1
                    misc_key = f"other_domain_{overflow_idx}"
                groups.setdefault(misc_key, []).extend(chunk)

    if len(groups) > cap_coarse:
        raise ValueError(
            f"cannot pack {len(criteria)} options into <= {cap_coarse} domains of <= {MAX_OPTIONS_PER_SLOT - 1} items"
        )

    coarse_criteria: dict[str, str] = {}
    for dom, members in groups.items():
        sample_names = ", ".join(k for k, _ in members[:5])
        more = f" (+{len(members) - 5} more)" if len(members) > 5 else ""
        coarse_criteria[dom] = f"Domain '{dom}' covering: {sample_names}{more}"
    if detected_catch_all and detected_catch_all in criteria:
        coarse_criteria[detected_catch_all] = criteria[detected_catch_all]

    # Check if natural suffixes are unique within each domain AND fit in <= 25 total fine options
    distinct_suffixes: list[str] = []
    suffix_collision = False
    for dom, members in groups.items():
        seen_in_dom: set[str] = set()
        for k, _ in members:
            s = raw_suffixes[k]
            if s in seen_in_dom:
                suffix_collision = True
            seen_in_dom.add(s)
            if s not in distinct_suffixes:
                distinct_suffixes.append(s)

    fine_criteria: dict[str, str] = {}
    leaves: dict[str, TaxonomyLeaf] = {}

    if not suffix_collision and 2 <= len(distinct_suffixes) <= (
        MAX_OPTIONS_PER_SLOT - (1 if detected_catch_all else 0)
    ):
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
        max_member_len = max(len(m) for m in groups.values())
        max_member_len = max(2, max_member_len)
        for idx in range(max_member_len):
            slot_key = f"member_{idx + 1:02d}"
            slot_examples: list[str] = []
            for dom, members in groups.items():
                if idx < len(members):
                    leaf_k, leaf_desc = members[idx]
                    short_desc = leaf_desc if len(leaf_desc) <= 48 else leaf_desc[:45] + "..."
                    slot_examples.append(f"[{dom}] {leaf_k} ({short_desc})")
                    leaves[leaf_k] = TaxonomyLeaf(
                        leaf_key=leaf_k,
                        coarse_key=dom,
                        fine_key=slot_key,
                        description=leaf_desc,
                    )
            fine_criteria[slot_key] = "; ".join(slot_examples)

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
    """Decode exact joint leaf probabilities strictly over valid paths in `trie`.

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


class HierarchicalDecisionRouter:
    """Executes wide-option (>26) taxonomy routing across 4 Semantic-ID decoding arms."""

    VALID_MODES = ("mtp_naive", "mtp_trie", "beam_2pass", "hesitation_hybrid")

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
            joint_probs, trie_winner = trie_constrained_joint_decode(coarse_dist, fine_dist, active_trie)
            exact_match = ""
            for lk, leaf in active_trie.leaves.items():
                if leaf.coarse_key == best_c and leaf.fine_key == best_f:
                    exact_match = lk
                    break
            # Unrestricted MTP emits invalid (c1, c2) when no leaf exists; record fallback marker or valid leaf
            pred = exact_match if exact_match else f"__invalid_tuple__:{best_c}/{best_f}"
            return SlotReadout(
                slot_id=slot_id,
                predicted=pred,
                probabilities=joint_probs,
                raw_probabilities=coarse_dist,
                h_raw=h_coarse,
                confidence=1.0 - h_coarse,
                guard_applied="mtp_naive_unconstrained" if not exact_match else "none",
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

        for dom in ranked_doms[: self.beam_width]:
            p_dom = max(1e-9, float(coarse_dist.get(dom, 1e-9)))
            for leaf in trie.domain_members(dom):
                if len(candidate_leaves) >= max_leaf_budget:
                    break
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

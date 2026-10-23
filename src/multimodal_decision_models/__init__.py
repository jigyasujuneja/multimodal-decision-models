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

"""Backend-agnostic Calibrated Multimodal Decision Models."""

from .backends import (
    CascadeDecisionOrchestrator,
    DecisionBackend,
    DgemSystemOneBackend,
    MockCallableBackend,
    OpenAILogprobBackend,
    VertexGeminiBackend,
)
from .calibration import (
    DEFAULT_EVIDENCE_CRITERIA,
    apply_layer5_deterministic_guards,
    inject_evidence_slot,
    normalized_entropy,
    pre_gated_bayesian_poe_fusion,
    stage2a_fast_exit_gate,
)
from .consistency import (
    arbitrate_swarm_locks,
    check_transitive_triangles,
    parse_pack_multiplier,
    probabilistic_triangle_contradiction,
    score_probabilistic_triangles,
    validate_pack_quantity_consistency,
)
from .hierarchy import (
    HierarchicalDecisionRouter,
    TaxonomyLeaf,
    TaxonomyTrie,
    build_prefix_taxonomy_trie,
    packed_bracket_pass_count,
    trie_constrained_joint_decode,
)
from .shortlist import (
    DEFAULT_COSINE_PRIOR_TEMPERATURE,
    ShortlistConfig,
    build_contrastive_criteria,
    cosine_similarity,
    csls_hubness_scores,
    normalize_cosine_prior_logits,
    shortlist_wide_question,
    split_identifier_tokens,
)
from .som_navigator import (
    assign_box_to_grid_4x4,
    assign_box_to_grid_cell,
    assign_point_to_grid_4x4,
    assign_point_to_grid_cell,
    box_area,
    box_center,
    box_iou,
    build_set_of_marks_questions,
    expected_mark_center_from_probabilities,
    filter_marks_by_grid_cell,
    grid_4x4_cell_bounds,
    point_in_box,
    select_containing_or_nearest_mark,
)
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    MAX_SLOTS_PER_PASS,
    CandidateMark,
    DecisionQuestion,
    DecisionResult,
    PairwiseEdge,
    ProbabilisticTriangleScore,
    SlotReadout,
    SwarmAgentProposal,
    SwarmLockDecision,
    TriangleContradiction,
)

__version__ = "0.2.1"

__all__ = [
    "DEFAULT_COSINE_PRIOR_TEMPERATURE",
    "DEFAULT_EVIDENCE_CRITERIA",
    "DEFAULT_HESITATION_THRESHOLD",
    "MAX_OPTIONS_PER_SLOT",
    "MAX_SLOTS_PER_PASS",
    "CandidateMark",
    "CascadeDecisionOrchestrator",
    "DecisionBackend",
    "DecisionQuestion",
    "DecisionResult",
    "DgemSystemOneBackend",
    "HierarchicalDecisionRouter",
    "MockCallableBackend",
    "OpenAILogprobBackend",
    "PairwiseEdge",
    "ProbabilisticTriangleScore",
    "ShortlistConfig",
    "SlotReadout",
    "SwarmAgentProposal",
    "SwarmLockDecision",
    "TaxonomyLeaf",
    "TaxonomyTrie",
    "TriangleContradiction",
    "VertexGeminiBackend",
    "__version__",
    "apply_layer5_deterministic_guards",
    "arbitrate_swarm_locks",
    "assign_box_to_grid_4x4",
    "assign_box_to_grid_cell",
    "assign_point_to_grid_4x4",
    "assign_point_to_grid_cell",
    "box_area",
    "box_center",
    "box_iou",
    "build_contrastive_criteria",
    "build_prefix_taxonomy_trie",
    "build_set_of_marks_questions",
    "check_transitive_triangles",
    "cosine_similarity",
    "csls_hubness_scores",
    "expected_mark_center_from_probabilities",
    "filter_marks_by_grid_cell",
    "grid_4x4_cell_bounds",
    "inject_evidence_slot",
    "normalize_cosine_prior_logits",
    "normalized_entropy",
    "packed_bracket_pass_count",
    "parse_pack_multiplier",
    "point_in_box",
    "pre_gated_bayesian_poe_fusion",
    "probabilistic_triangle_contradiction",
    "score_probabilistic_triangles",
    "select_containing_or_nearest_mark",
    "shortlist_wide_question",
    "split_identifier_tokens",
    "stage2a_fast_exit_gate",
    "trie_constrained_joint_decode",
    "validate_pack_quantity_consistency",
]

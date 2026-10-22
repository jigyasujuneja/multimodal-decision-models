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
    validate_pack_quantity_consistency,
)
from .shortlist import (
    cosine_similarity,
    csls_hubness_scores,
    shortlist_wide_question,
)
from .som_navigator import (
    assign_box_to_grid_cell,
    assign_point_to_grid_cell,
    box_area,
    box_center,
    box_iou,
    build_set_of_marks_questions,
    filter_marks_by_grid_cell,
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
    SlotReadout,
    SwarmAgentProposal,
    SwarmLockDecision,
    TriangleContradiction,
)

__all__ = [
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
    "MockCallableBackend",
    "OpenAILogprobBackend",
    "PairwiseEdge",
    "SlotReadout",
    "SwarmAgentProposal",
    "SwarmLockDecision",
    "TriangleContradiction",
    "VertexGeminiBackend",
    "apply_layer5_deterministic_guards",
    "arbitrate_swarm_locks",
    "assign_box_to_grid_cell",
    "assign_point_to_grid_cell",
    "box_area",
    "box_center",
    "box_iou",
    "build_set_of_marks_questions",
    "check_transitive_triangles",
    "cosine_similarity",
    "csls_hubness_scores",
    "filter_marks_by_grid_cell",
    "inject_evidence_slot",
    "normalized_entropy",
    "parse_pack_multiplier",
    "point_in_box",
    "pre_gated_bayesian_poe_fusion",
    "select_containing_or_nearest_mark",
    "shortlist_wide_question",
    "stage2a_fast_exit_gate",
    "validate_pack_quantity_consistency",
]

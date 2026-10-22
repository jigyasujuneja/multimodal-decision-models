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

"""Core data structures for backend-agnostic Multimodal Decision Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_OPTIONS_PER_SLOT = 26
MAX_SLOTS_PER_PASS = 8
DEFAULT_HESITATION_THRESHOLD = 0.16


@dataclass
class DecisionQuestion:
    """Represents a single decision slot (choice, noul, or ordinal score)."""

    type: str  # "choice", "noul", or "score"
    instructions: str
    criteria: dict[str, str] = field(default_factory=dict)
    levels: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)

    def option_keys(self) -> list[str]:
        if self.type == "score":
            return list(self.levels)
        if self.order:
            return [k for k in self.order if k in self.criteria]
        return list(self.criteria.keys())


@dataclass
class SlotReadout:
    """Calibrated probability distribution and hesitation metrics for a single decision slot."""

    slot_id: str
    predicted: str
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    h_raw: float
    confidence: float
    escalated: bool = False
    guard_applied: str = ""
    backend_name: str = ""


@dataclass
class DecisionResult:
    """Multi-slot decision result returned by any DecisionBackend or CascadeDecisionOrchestrator."""

    slots: dict[str, SlotReadout]
    evidence_slot_id: str | None = None
    escalated: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateMark:
    """Interactive UI element or DOM node annotated with a Set-of-Marks ID and [y_min, x_min, y_max, x_max] box."""

    id: str
    label: str
    box: tuple[float, float, float, float]  # (y_min, x_min, y_max, x_max) in [0, 100]


@dataclass(frozen=True)
class PairwiseEdge:
    """Pairwise entity-resolution equivalence judgment between node_a and node_b."""

    node_a: str
    node_b: str
    equiv_prob: float


@dataclass(frozen=True)
class TriangleContradiction:
    """3-cycle (A, B, C) violating transitive equivalence: two edges >= high_thresh and one <= low_thresh."""

    nodes: tuple[str, str, str]
    edge_probs: tuple[float, float, float]
    weak_edge: tuple[str, str]
    weak_prob: float
    severity: float


@dataclass(frozen=True)
class SwarmAgentProposal:
    """Proposed file/resource lock acquisition by a concurrent worker agent."""

    agent_id: str
    target_paths: tuple[str, ...]
    confidence: float
    hesitation: float


@dataclass
class SwarmLockDecision:
    """Deterministic lock arbitration outcome across concurrent worker proposals."""

    agent_actions: dict[str, str]
    write_collision: bool
    winning_agent: str
    escalate_to_human: bool

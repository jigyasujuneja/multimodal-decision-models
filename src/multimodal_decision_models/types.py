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

"""Core data structures and closed contract enums for Multimodal Decision Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_OPTIONS_PER_SLOT = 26
MAX_EXTENDED_OPTIONS_PER_SLOT = 256
MAX_SLOTS_PER_PASS = 8
DEFAULT_HESITATION_THRESHOLD = 0.16

UNIVERSAL_256_DECISION_CODES: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF",
    "AG", "AH", "AI", "AK", "AL", "AM", "AN", "AP", "AR", "AS", "AT", "AU", "AV", "AW", "BA", "BB",
    "BC", "BD", "BE", "BI", "BL", "BM", "BO", "BP", "BR", "BS", "BT", "BU", "BY", "CA", "CB", "CC",
    "CD", "CE", "CF", "CG", "CH", "CI", "CL", "CM", "CN", "CO", "CP", "CR", "CS", "CT", "CU", "CV",
    "DA", "DB", "DC", "DD", "DE", "DF", "DH", "DI", "DL", "DM", "DN", "DO", "DP", "DR", "DS", "DT",
    "EA", "EC", "ED", "EE", "EF", "EG", "EL", "EM", "EN", "EP", "ER", "ES", "ET", "EU", "EV", "EX",
    "FA", "FC", "FD", "FE", "FI", "FL", "FM", "FO", "FP", "FR", "FS", "FT", "GA", "GB", "GC", "GD",
    "GE", "GH", "GI", "GL", "GM", "GN", "GO", "GP", "GR", "GS", "GT", "GW", "HA", "HC", "HD", "HE",
    "HF", "HI", "HL", "HO", "HP", "HR", "HS", "HT", "HV", "IA", "IB", "IC", "ID", "IE", "IF", "IG",
    "II", "IL", "IM", "IN", "IO", "IP", "IR", "IS", "IT", "IU", "IV", "IX", "JC", "JD", "JO", "JS",
    "KA", "KB", "KE", "KS", "LA", "LC", "LD", "LE", "LG", "LI", "LL", "LM", "LO", "LP", "LS", "LT",
    "MA", "MB", "MC", "MD", "ME", "MH", "MI", "ML", "MM", "MN", "MO", "MP", "MR", "MS", "MT", "MW",
    "MY", "NA", "NC", "ND", "NE", "NF", "NG", "NH", "NI", "NL", "NM", "NO", "NP", "NR", "NS", "NT",
    "NV", "NW", "NY", "OA", "OB", "OC", "OD", "OF", "OH", "OK", "OM", "ON", "OP", "OR", "OS", "OT",
    "PA", "PC", "PD", "PE", "PF", "PG", "PH", "PI", "PK", "PL", "PM", "PO", "PP", "PR", "PS", "PT",
)

INTEGRATION_MODES: tuple[str, ...] = (
    "replace",
    "cascade",
    "assist",
    "guard",
    "trigger",
)

HARNESS_DECISION_POINTS: tuple[str, ...] = (
    "tool_select",
    "tool_args",
    "action",
    "memory_write",
    "memory_read",
    "reflection",
    "stop",
    "routing",
)



class DecisionTerminalState(str, Enum):
    """Closed Level-4 Total-Contract terminal states for a slot or decision request."""

    STAGE2A_FAST_EXIT = "STAGE2A_FAST_EXIT"
    STAGE1_RESOLVED = "STAGE1_RESOLVED"
    STAGE2_VERIFIED = "STAGE2_VERIFIED"
    GUARD_OVERRIDDEN = "GUARD_OVERRIDDEN"
    ABSTAIN_OOS = "ABSTAIN_OOS"
    INVALID_TRIE_PATH = "INVALID_TRIE_PATH"
    RETRY_PENDING = "RETRY_PENDING"
    REJECTED_CONTRACT_VIOLATION = "REJECTED_CONTRACT_VIOLATION"


class FailureCategory(str, Enum):
    """Structured failure taxonomy distinguishing transient upstream faults from contract errors."""

    NONE = "NONE"
    UPSTREAM_TIMEOUT_OR_THROTTLE = "UPSTREAM_TIMEOUT_OR_THROTTLE"
    UNALIGNED_LOGPROBS = "UNALIGNED_LOGPROBS"
    SCHEMA_CONTRACT_VIOLATION = "SCHEMA_CONTRACT_VIOLATION"
    INVALID_UNCONSTRAINED_TUPLE = "INVALID_UNCONSTRAINED_TUPLE"
    ABSTAINED_OOS = "ABSTAINED_OOS"


@dataclass(frozen=True)
class UnitMeasure:
    """Normalized physical unit volume or mass extracted from a catalog item title."""

    value: float
    unit: str  # canonical base unit: "ml", "g", "fl_oz", "oz"


@dataclass
class DecisionQuestion:
    """Represents a single decision slot (choice, noul, or ordinal score)."""

    type: str  # "choice", "noul", or "score"
    instructions: str
    criteria: dict[str, str] = field(default_factory=dict)
    levels: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    prior_logits: dict[str, float] = field(default_factory=dict)
    evidence_prompt: str = ""
    evidence_criteria: dict[str, str] = field(default_factory=dict)

    def option_keys(self) -> list[str]:
        if self.type == "score":
            return list(self.levels)
        if self.order:
            return [k for k in self.order if k in self.criteria]
        return list(self.criteria.keys())


@dataclass
class SlotReadout:
    """Calibrated probability distribution, hesitation metrics, and contract state for a slot."""

    slot_id: str
    predicted: str
    probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    h_raw: float
    confidence: float
    escalated: bool = False
    guard_applied: str = ""
    backend_name: str = ""
    evidence: str = ""
    passes_used: int = 1
    terminal_state: DecisionTerminalState = DecisionTerminalState.STAGE1_RESOLVED
    failure_category: FailureCategory = FailureCategory.NONE
    invalid_coordinates: tuple[str, str] | None = None
    conformal_set: list[str] = field(default_factory=list)
    epistemic_variance: float = 0.0


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
    terminal_state: DecisionTerminalState = DecisionTerminalState.STAGE1_RESOLVED
    failure_category: FailureCategory = FailureCategory.NONE
    cache_hit: bool = False
    cache_key: str = ""


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
class ProbabilisticTriangleScore:
    """Continuous transitive contradiction score p_ij * p_jk * (1 - p_ik) in [0, 1] for a 3-cycle."""

    nodes: tuple[str, str, str]
    edge_probs: tuple[float, float, float]
    weak_edge: tuple[str, str]
    weak_prob: float
    contradiction_score: float


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

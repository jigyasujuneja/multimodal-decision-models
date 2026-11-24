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

"""Two-phase and speculative cascade orchestration, top-k pruning, and uniform guards."""

from __future__ import annotations

from .backends import CascadeDecisionOrchestrator
from .calibration import DomainCompetenceProfile, compute_dawid_skene_poe_weights, is_uniform_distribution
from .shortlist import prune_choice_top_k
from .types import DecisionQuestion, DecisionResult, SlotReadout

__all__ = [
    "CascadeDecisionOrchestrator",
    "DecisionQuestion",
    "DecisionResult",
    "DomainCompetenceProfile",
    "SlotReadout",
    "compute_dawid_skene_poe_weights",
    "is_uniform_distribution",
    "prune_choice_top_k",
]

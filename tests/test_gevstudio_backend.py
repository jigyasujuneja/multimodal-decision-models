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

"""Unit tests for GevStudioSystemOneBackend (/api/gev/systemone adapter)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from multimodal_decision_models import (
    CascadeDecisionOrchestrator,
    DecisionQuestion,
    DecisionTerminalState,
    GevStudioSystemOneBackend,
    LogprobAlignmentError,
    SchemaContractViolationError,
)


def test_gevstudio_build_url_encodes_target_query_param() -> None:
    backend = GevStudioSystemOneBackend(
        endpoint_url="https://gevstudio.corp.google.com",
        target="beyond-decision:/models/gemini-v4xs-decisions?hc=true;temp=1.0",
    )
    url = backend.build_url()
    assert url == (
        "https://gevstudio.corp.google.com/api/gev/systemone"
        "?target=beyond-decision%3A%2Fmodels%2Fgemini-v4xs-decisions%3Fhc%3Dtrue%3Btemp%3D1.0"
    )


def test_gevstudio_build_payload_formats_multimodal_media_and_question_types() -> None:
    backend = GevStudioSystemOneBackend(
        target="gemini-v4xs-decisions",
        model="gemini-v4xs-decisions",
        temperature=0.0,
    )
    questions = {
        "q_choice": DecisionQuestion(
            type="choice",
            instructions="Choose next action.",
            criteria={"stop": "Full stop", "slow": "Slow down"},
        ),
        "q_noul": DecisionQuestion(
            type="noul",
            instructions="Is a pedestrian visible?",
            criteria={"no": "No pedestrian", "yes": "Pedestrian visible"},
        ),
        "q_score": DecisionQuestion(
            type="score",
            instructions="Rate urgency.",
            levels=["low", "medium", "high"],
        ),
    }
    state = {
        "zone": "aisle_4",
        "__media__": [{"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"}],
    }
    payload = backend.build_payload(
        state=state,
        questions=questions,
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==", "/9j/4AAQSkZJRg=="],
    )
    assert payload["state"] == {"zone": "aisle_4"}
    assert payload["model"] == "gemini-v4xs-decisions"
    assert payload["temperature"] == 0.0
    assert len(payload["media"]) == 3
    assert payload["media"][0] == {"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"}
    assert payload["media"][1] == {"mime_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUg=="}
    assert payload["media"][2] == {"mime_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="}
    assert payload["questions"]["q_choice"]["type"] == "choice"
    assert payload["questions"]["q_noul"] == {
        "type": "noul",
        "instructions": "Is a pedestrian visible?",
    }
    assert payload["questions"]["q_score"] == {
        "type": "score",
        "instructions": "Rate urgency.",
        "criteria": ["low", "medium", "high"],
    }


def test_gevstudio_parse_response_choice_noul_and_score_probabilities() -> None:
    backend = GevStudioSystemOneBackend(target="gemini-v4xs-decisions")
    questions = {
        "action": DecisionQuestion(
            type="choice",
            instructions="Pick action.",
            criteria={"stop": "Stop", "slow": "Slow", "proceed": "Proceed"},
        ),
        "pedestrian": DecisionQuestion(
            type="noul",
            instructions="Pedestrian in path?",
            criteria={"no": "No", "yes": "Yes"},
        ),
        "risk": DecisionQuestion(
            type="score",
            instructions="Risk level.",
            levels=["low", "med", "high"],
        ),
    }
    wire_response = {
        "answers": {
            "action": {
                "choice": "stop",
                "probabilities": {"stop": 0.90, "slow": 0.08, "proceed": 0.02},
            },
            "pedestrian": {
                "noul": True,
                "noul_prob": 0.96,
                "probabilities": {"0": 0.04, "1": 0.96},
            },
            "risk": {
                "score": 2,
                "score_value": 1.88,
                "probabilities": {"0": 0.02, "1": 0.08, "2": 0.90},
            },
        }
    }
    dists, ev, trace = backend.parse_response(wire_response, questions)
    assert pytest.approx(dists["action"]["stop"], rel=1e-5) == 0.90
    assert pytest.approx(dists["pedestrian"]["yes"], rel=1e-5) == 0.96
    assert pytest.approx(dists["pedestrian"]["no"], rel=1e-5) == 0.04
    assert pytest.approx(dists["risk"]["high"], rel=1e-5) == 0.90
    assert trace["discrete_fallback_slots"] == []
    assert ev == {}


def test_gevstudio_parse_response_discrete_fallback_and_strict_mode() -> None:
    questions = {
        "action": DecisionQuestion(
            type="choice",
            instructions="Pick action.",
            criteria={"stop": "Stop", "proceed": "Proceed"},
        )
    }
    wire_discrete = {"answers": {"action": {"choice": "stop"}}}

    lenient = GevStudioSystemOneBackend(
        target="gemini-3.6-flash",
        allow_discrete_fallback=True,
        label_smoothing_eps=1e-3,
    )
    dists, _, trace = lenient.parse_response(wire_discrete, questions)
    assert dists["action"]["stop"] > 0.998
    assert trace["discrete_fallback_slots"] == ["action"]

    strict = GevStudioSystemOneBackend(
        target="gemini-3.6-flash",
        allow_discrete_fallback=False,
    )
    with pytest.raises(LogprobAlignmentError):
        strict.parse_response(wire_discrete, questions)


def test_gevstudio_cascade_orchestrator_end_to_end() -> None:
    backend = GevStudioSystemOneBackend(
        endpoint_url="https://gevstudio.corp.google.com/api/gev/systemone",
        target="gemini-v4xs-decisions",
    )
    questions = {
        "sku": DecisionQuestion(
            type="choice",
            instructions="Select matching SKU.",
            criteria={"sku_a": "SKU A", "sku_b": "SKU B"},
            prior_logits={"sku_a": 1.2, "sku_b": -0.4},
        )
    }
    mock_wire = {
        "answers": {
            "sku": {
                "choice": "sku_a",
                "probabilities": {"sku_a": 0.88, "sku_b": 0.12},
            }
        }
    }
    with patch("multimodal_decision_models.backends._post_json_with_retry", return_value=mock_wire):
        orch = CascadeDecisionOrchestrator(
            stage1_backend=backend,
            prior_weight=0.25,
            hesitation_threshold=0.30,
        )
        res = orch.evaluate(state={"shelf": "bay_1"}, questions=questions)
    assert res.terminal_state == DecisionTerminalState.STAGE1_RESOLVED
    assert res.slots["sku"].predicted == "sku_a"
    assert res.slots["sku"].probabilities["sku_a"] > 0.88


def test_gevstudio_validation_errors() -> None:
    with pytest.raises(SchemaContractViolationError):
        GevStudioSystemOneBackend(endpoint_url="file:///etc/passwd")
    with pytest.raises(SchemaContractViolationError):
        GevStudioSystemOneBackend(label_smoothing_eps=0.75)

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

"""Backend-agnostic adapters and Cascade Decision Orchestrator.

Supports:
  1. DgemSystemOneBackend: /v1/systemone parallel masked diffusion endpoints.
  2. OpenAILogprobBackend: /v1/chat/completions single-token logprob readout (vLLM, SGLang, OpenAI, Gemma 4).
  3. VertexGeminiBackend: Vertex AI generateContent with responseLogprobs and Evidence-First JSON schema.
  4. MockCallableBackend: Deterministic in-memory adapter for unit testing and offline replay.
  5. CascadeDecisionOrchestrator: Unifies Stage-2A Fast-Exit, Wide-Option Shortlist, Co-Denoised/Causal
     Evidence Slot, Pre-Gated Bayesian PoE Prior Fusion, Layer-5 Guards, and Hesitation-Gated Escalation.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Protocol

from .calibration import (
    apply_layer5_deterministic_guards,
    inject_evidence_slot,
    normalized_entropy,
    pre_gated_bayesian_poe_fusion,
    stage2a_fast_exit_gate,
)
from .shortlist import shortlist_wide_question
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    DecisionQuestion,
    DecisionResult,
    SlotReadout,
)

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_INFERRED_EVIDENCE_LABELS = frozenset({"inferred", "occluded_or_unreadable", "unreadable", "absent"})


def _validate_http_url(url: str) -> str:
    """Validate that an endpoint URL uses http or https with a non-empty host (blocks file:// SSRF)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"endpoint URL must use http:// or https:// with a valid host, got {url!r}")
    return url


def _validate_resource_segment(label: str, value: str) -> str:
    if not value or not _SAFE_SEGMENT_RE.match(value):
        raise ValueError(f"invalid {label} identifier: {value!r}")
    return value


class DecisionBackend(Protocol):
    """Protocol implemented by all decision model backends."""

    name: str

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Return raw probability distributions {slot_id: {option_key: prob}} for each question."""
        ...


class MockCallableBackend:
    """In-memory backend driven by a callable or static probability dictionary for testing & replay."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Any, Mapping[str, DecisionQuestion], list[str] | None], dict[str, dict[str, float]]],
    ) -> None:
        self.name = name
        self._fn = fn
        self.last_evidence: dict[str, str] = {}

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        return self._fn(state, questions, images)


class DgemSystemOneBackend:
    """Adapter for /v1/systemone parallel masked diffusion decision endpoints."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        name: str = "dgem_systemone",
    ) -> None:
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        self.endpoint_url = clean_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.name = name
        self.last_evidence: dict[str, str] = {}
        self.last_trace: dict[str, Any] = {}

    def build_payload(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
        prior_logits: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        q_payload: dict[str, Any] = {}
        for qid, q in questions.items():
            if q.type == "score":
                entry: dict[str, Any] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": list(q.levels),
                }
            else:
                ordered_crit = {k: q.criteria[k] for k in q.option_keys() if k in q.criteria}
                entry = {
                    "type": q.type,
                    "instructions": q.instructions,
                    "criteria": ordered_crit,
                }
            merged_priors = dict(q.prior_logits)
            if prior_logits and qid in prior_logits:
                merged_priors.update(prior_logits[qid])
            if merged_priors:
                entry["prior_logits"] = merged_priors
            if q.evidence_prompt.strip():
                entry["evidence_prompt"] = q.evidence_prompt.strip()
                if q.evidence_criteria:
                    entry["evidence_criteria"] = dict(q.evidence_criteria)
            q_payload[qid] = entry

        payload: dict[str, Any] = {"state": state, "questions": q_payload}
        if images:
            payload["images"] = list(images)
        return payload

    @staticmethod
    def parse_response(
        data: Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, Any]]:
        """Parse a /v1/systemone response envelope (or legacy flat predictions map)."""
        evidence_map: dict[str, str] = {}
        trace = dict(data.get("evaluation_trace") or {})

        if "answers" in data and isinstance(data["answers"], Mapping):
            answers = data["answers"]
            out: dict[str, dict[str, float]] = {}
            for qid, q in questions.items():
                ans = answers.get(qid)
                if not isinstance(ans, Mapping):
                    continue
                probs = dict(ans.get("probabilities") or {})
                if q.type == "score" and probs:
                    legend = ans.get("legend") or {}
                    mapped_probs: dict[str, float] = {}
                    for idx_str, p_val in probs.items():
                        if isinstance(legend, Mapping) and idx_str in legend:
                            mapped_probs[str(legend[idx_str])] = float(p_val)
                        elif idx_str.isdigit() and int(idx_str) < len(q.levels):
                            mapped_probs[q.levels[int(idx_str)]] = float(p_val)
                        else:
                            mapped_probs[str(idx_str)] = float(p_val)
                    out[qid] = mapped_probs
                else:
                    out[qid] = {str(k): float(v) for k, v in probs.items()}
                ev = ans.get("evidence")
                if isinstance(ev, str) and ev:
                    evidence_map[qid] = ev
            return out, evidence_map, trace

        preds = data.get("predictions", data)
        if isinstance(preds, Mapping):
            return (
                {
                    str(qid): {str(k): float(v) for k, v in dist.items()}
                    for qid, dist in preds.items()
                    if isinstance(dist, Mapping)
                },
                evidence_map,
                trace,
            )
        return {}, evidence_map, trace

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        url = (
            self.endpoint_url
            if self.endpoint_url.endswith("/v1/systemone")
            else f"{self.endpoint_url}/v1/systemone"
        )
        _validate_http_url(url)
        body = json.dumps(self.build_payload(state, questions, images)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.load(resp)
        dists, ev_map, trace = self.parse_response(data, questions)
        self.last_evidence = ev_map
        self.last_trace = trace
        return dists


class OpenAILogprobBackend:
    """Adapter for OpenAI-compatible /v1/chat/completions endpoints (vLLM, SGLang, TGI, OpenAI).

    Maps each question's options to single uppercase letters [A..Z], requests `logprobs=True`
    and `top_logprobs=20`, and extracts a calibrated restricted softmax over the valid letters.
    """

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        name: str | None = None,
    ) -> None:
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        self.endpoint_url = clean_url
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.name = name or f"openai_logprob:{model}"
        self.last_evidence: dict[str, str] = {}

    @staticmethod
    def build_letter_prompt(state: Any, q: DecisionQuestion) -> tuple[str, dict[str, str]]:
        keys = q.option_keys()
        if len(keys) > MAX_OPTIONS_PER_SLOT:
            raise ValueError(f"question has {len(keys)} options; shortlist to <= {MAX_OPTIONS_PER_SLOT} first")
        letters = string.ascii_uppercase[: len(keys)]
        letter_to_key = {letters[i]: keys[i] for i in range(len(keys))}
        lines = [
            "State / Context:",
            json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state,
            "",
            f"Question: {q.instructions}",
            "Options:",
        ]
        for letter, key in letter_to_key.items():
            desc = q.criteria.get(key, key) if q.type != "score" else key
            lines.append(f"  {letter}) {key}: {desc}")
        lines.append("\nReply with the single uppercase option letter only.")
        return "\n".join(lines), letter_to_key

    @staticmethod
    def extract_letter_distribution(
        top_logprobs: Mapping[str, float],
        letter_to_key: Mapping[str, str],
    ) -> dict[str, float]:
        """Convert token top_logprobs into a normalized probability distribution over option keys."""
        letter_logits: dict[str, float] = {}
        for letter, key in letter_to_key.items():
            best_lp = float("-inf")
            for tok, lp in top_logprobs.items():
                if tok.strip() == letter and float(lp) > best_lp:
                    best_lp = float(lp)
            letter_logits[key] = best_lp

        finite_vals = [v for v in letter_logits.values() if math.isfinite(v)]
        if not finite_vals:
            return {k: 1.0 / len(letter_to_key) for k in letter_to_key.values()}
        floor_lp = min(finite_vals) - 10.0
        max_lp = max(finite_vals)
        exps = {
            k: math.exp((v if math.isfinite(v) else floor_lp) - max_lp)
            for k, v in letter_logits.items()
        }
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        url = (
            self.endpoint_url
            if self.endpoint_url.endswith("/v1/chat/completions")
            else f"{self.endpoint_url}/v1/chat/completions"
        )
        _validate_http_url(url)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        out: dict[str, dict[str, float]] = {}
        for qid, q in questions.items():
            prompt_text, letter_to_key = self.build_letter_prompt(state, q)
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": min(20, len(letter_to_key) + 4),
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.load(resp)
            choice0 = data["choices"][0]
            content_lp = choice0.get("logprobs", {}).get("content", [])
            top_map: dict[str, float] = {}
            if content_lp:
                for entry in content_lp[0].get("top_logprobs", []):
                    top_map[entry["token"]] = float(entry["logprob"])
            out[qid] = self.extract_letter_distribution(top_map, letter_to_key)
        return out


class VertexGeminiBackend:
    """Adapter for Vertex AI Gemini models using Causal Evidence-First structured JSON schemas."""

    def __init__(
        self,
        project: str,
        model: str = "gemini-3.8-flash",
        location: str = "global",
        access_token_fn: Callable[[], str] | None = None,
        timeout_sec: float = 60.0,
        endpoint_override: str | None = None,
    ) -> None:
        self.project = _validate_resource_segment("project", project)
        self.model = _validate_resource_segment("model", model)
        self.location = _validate_resource_segment("location", location)
        self.access_token_fn = access_token_fn
        self.timeout_sec = timeout_sec
        if endpoint_override is not None:
            _validate_http_url(endpoint_override)
        self.endpoint_override = endpoint_override
        self.name = f"vertex_gemini:{self.model}"
        self.last_evidence: dict[str, str] = {}

    def endpoint_url(self) -> str:
        if self.endpoint_override:
            return self.endpoint_override
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{self.model}:generateContent"
        )

    @staticmethod
    def build_evidence_first_response_schema(
        questions: Mapping[str, DecisionQuestion],
    ) -> dict[str, Any]:
        """Build a Gemini JSON responseSchema that places evidence_read FIRST before choice slots."""
        properties: dict[str, Any] = {
            "evidence_read": {
                "type": "STRING",
                "description": "Concise factual observation (3-8 words, such as printed pack size or visual cue) decoded before selecting slot choices.",
            }
        }
        required = ["evidence_read"]
        for qid, q in questions.items():
            properties[qid] = {
                "type": "STRING",
                "enum": q.option_keys(),
                "description": q.instructions,
            }
            required.append(qid)
        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
            "propertyOrdering": required,
        }

    def build_generate_content_payload(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for img in images or []:
            if img.startswith("data:") and ";base64," in img:
                header, b64_data = img.split(";base64,", 1)
                mime_type = header[len("data:") :] or "image/png"
                parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})
        prompt_lines = [
            "Evaluate the state and answer each question strictly using the allowed enum keys.",
            "State:",
            json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state,
            "Questions:",
        ]
        for qid, q in questions.items():
            opts_desc = ", ".join(f"{k} ({q.criteria.get(k, k)})" for k in q.option_keys())
            prompt_lines.append(f"- {qid}: {q.instructions} | Options: {opts_desc}")
        parts.append({"text": "\n".join(prompt_lines)})

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": self.build_evidence_first_response_schema(questions),
                "responseLogprobs": True,
                "logprobs": 5,
            },
        }

    @staticmethod
    def parse_generate_content_response(
        data: Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> tuple[dict[str, dict[str, float]], str]:
        candidates = data.get("candidates") or []
        if not candidates:
            return {}, ""
        cand0 = candidates[0]
        parts = (cand0.get("content") or {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, Mapping))
        parsed_json: dict[str, Any] = {}
        if raw_text.strip():
            parsed_json = json.loads(raw_text)

        evidence_read = str(parsed_json.get("evidence_read", ""))
        chosen_steps = ((cand0.get("logprobsResult") or {}).get("chosenCandidates") or [])
        top_steps = ((cand0.get("logprobsResult") or {}).get("topCandidates") or [])

        out: dict[str, dict[str, float]] = {}
        for qid, q in questions.items():
            keys = q.option_keys()
            selected = str(parsed_json.get(qid, keys[0] if keys else ""))
            # Extract token-level probability if logprobsResult matched the enum value token
            matched_probs: dict[str, float] = {}
            for step_idx, chosen_tok in enumerate(chosen_steps):
                tok_str = str(chosen_tok.get("token", "")).strip('" ')
                if tok_str == selected and step_idx < len(top_steps):
                    cands = top_steps[step_idx].get("candidates") or []
                    logits: dict[str, float] = {}
                    for k in keys:
                        best_lp = float("-inf")
                        for c_entry in cands:
                            ct = str(c_entry.get("token", "")).strip('" ')
                            if ct == k or (k.startswith(ct) and len(ct) >= 2):
                                lp = float(c_entry.get("logProbability", float("-inf")))
                                if lp > best_lp:
                                    best_lp = lp
                        logits[k] = best_lp
                    finite = [v for v in logits.values() if math.isfinite(v)]
                    if finite:
                        floor_lp = min(finite) - 8.0
                        max_lp = max(finite)
                        exps = {
                            k: math.exp((v if math.isfinite(v) else floor_lp) - max_lp)
                            for k, v in logits.items()
                        }
                        tot = sum(exps.values())
                        matched_probs = {k: v / tot for k, v in exps.items()}
                    break

            if not matched_probs and keys:
                # Smooth deterministic distribution when token boundaries split enum strings
                eps_share = 0.02 / max(1, len(keys) - 1) if len(keys) > 1 else 0.0
                matched_probs = {
                    k: (0.98 if k == selected and len(keys) > 1 else (1.0 if len(keys) == 1 else eps_share))
                    for k in keys
                }
            out[qid] = matched_probs
        return out, evidence_read

    def _resolve_access_token(self) -> str:
        if self.access_token_fn is not None:
            tok = self.access_token_fn().strip()
            if tok:
                return tok
        env_tok = (
            os.environ.get("VERTEX_ACCESS_TOKEN", "").strip()
            or os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
        )
        if env_tok:
            return env_tok
        raise RuntimeError(
            "VertexGeminiBackend requires access_token_fn or VERTEX_ACCESS_TOKEN / GOOGLE_OAUTH_ACCESS_TOKEN"
        )

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        url = self.endpoint_url()
        _validate_http_url(url)
        token = self._resolve_access_token()
        payload = self.build_generate_content_payload(state, questions, images)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.load(resp)
        dists, evidence_read = self.parse_generate_content_response(data, questions)
        self.last_evidence = {qid: evidence_read for qid in questions if evidence_read}
        return dists


class CascadeDecisionOrchestrator:
    """Backend-agnostic Multimodal Decision Funnel Orchestrator.

    Combines:
      1. Stage 2A Adaptive Fast-Exit Gate (0-VLM resolution when P_1 - P_2 >= 0.14 and Z >= 1.65).
      2. Wide-option Shortlist Routing (K > 26 -> top_k + none_of_the_above with sibling-unique tags).
      3. Co-denoised / Causal Evidence-First Slot Injection.
      4. Stage-1 Fast Decision Readout (for example, DgemSystemOneBackend or local vLLM).
      5. Pre-Gated Bayesian PoE Prior Fusion (default gate_mode="hesitant").
      6. Layer-5 Deterministic Post-Generation Guards (Pack-Size Sister Switch, Energy Veto, Competitor Guard).
      7. Hesitation-Gated Escalation (H_raw >= hesitation_threshold or inferred evidence) to Stage-2 Frontier Backend.
    """

    def __init__(
        self,
        stage1_backend: DecisionBackend,
        stage2_backend: DecisionBackend | None = None,
        hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD,
        prior_weight: float = 0.0,
        prior_gate_mode: str = "hesitant",
        temperature: float = 1.0,
        shortlist_top_k: int = 0,
        shortlist_pairwise_diff: bool = True,
        evidence_prompt: str = "",
        evidence_criteria: Mapping[str, str] | None = None,
        enable_stage2a_fast_exit: bool = False,
        escalate_on_inferred_evidence: bool = False,
    ) -> None:
        self.stage1_backend = stage1_backend
        self.stage2_backend = stage2_backend
        self.hesitation_threshold = hesitation_threshold
        self.prior_weight = prior_weight
        self.prior_gate_mode = prior_gate_mode
        self.temperature = temperature
        self.shortlist_top_k = shortlist_top_k
        self.shortlist_pairwise_diff = shortlist_pairwise_diff
        self.evidence_prompt = evidence_prompt
        self.evidence_criteria = evidence_criteria
        self.enable_stage2a_fast_exit = enable_stage2a_fast_exit
        self.escalate_on_inferred_evidence = escalate_on_inferred_evidence

    def evaluate(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
        prior_logits: Mapping[str, Mapping[str, float]] | None = None,
        shortlist_scores: Mapping[str, Mapping[str, float]] | None = None,
        candidate_pack_sizes: Mapping[str, int] | None = None,
        decoded_pack_size: int | None = None,
        visual_energies: Mapping[str, float] | None = None,
        rank1_id: str | None = None,
        rank1_visual_similarity: float | None = None,
        sister_pack_collision: bool = False,
    ) -> DecisionResult:
        t0 = time.perf_counter()

        # 1. Stage 2A Adaptive Fast-Exit check for single-slot retrieval verification
        if (
            self.enable_stage2a_fast_exit
            and len(questions) == 1
            and prior_logits
        ):
            only_qid = next(iter(questions.keys()))
            slot_priors = prior_logits.get(only_qid, {})
            if slot_priors:
                max_lp = max(slot_priors.values())
                exps = {k: math.exp(v - max_lp) for k, v in slot_priors.items()}
                tot = sum(exps.values())
                prior_probs = {k: v / tot for k, v in exps.items()}
                fast_ok, winner = stage2a_fast_exit_gate(
                    prior_probs,
                    sister_pack_collision=sister_pack_collision,
                )
                if fast_ok and winner is not None:
                    h_fast = normalized_entropy(prior_probs)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    return DecisionResult(
                        slots={
                            only_qid: SlotReadout(
                                slot_id=only_qid,
                                predicted=winner,
                                probabilities=prior_probs,
                                raw_probabilities=prior_probs,
                                h_raw=h_fast,
                                confidence=1.0 - h_fast,
                                escalated=False,
                                guard_applied="stage2a_fast_exit",
                                backend_name="stage2a_prior_fast_exit",
                            )
                        },
                        escalated=False,
                        latency_ms=elapsed,
                        backend_name="stage2a_prior_fast_exit",
                    )

        # 2. Optional Wide-Option Shortlist for questions with K > 26
        active_questions: dict[str, DecisionQuestion] = {}
        for qid, q in questions.items():
            if self.shortlist_top_k > 0 and len(q.option_keys()) > MAX_OPTIONS_PER_SLOT:
                s_map = (
                    (shortlist_scores or {}).get(qid)
                    or (prior_logits or {}).get(qid)
                    or q.prior_logits
                    or {}
                )
                if s_map:
                    active_questions[qid] = shortlist_wide_question(
                        q,
                        s_map,
                        top_k=self.shortlist_top_k,
                        pairwise_diff=self.shortlist_pairwise_diff,
                    )
                else:
                    active_questions[qid] = q
            else:
                active_questions[qid] = q

        # 3. Optional Co-Denoised / Causal Evidence Slot Injection
        evidence_slot_id: str | None = None
        if self.evidence_prompt.strip():
            active_questions = inject_evidence_slot(
                active_questions,
                self.evidence_prompt,
                self.evidence_criteria,
            )
            evidence_slot_id = "evidence_feature"

        # 4. Run Stage-1 Fast Backend
        raw_stage1 = self.stage1_backend.score_questions(state, active_questions, images)
        backend_evidence: Mapping[str, str] = getattr(self.stage1_backend, "last_evidence", {}) or {}

        injected_ev_choice = ""
        if evidence_slot_id and evidence_slot_id in raw_stage1 and raw_stage1[evidence_slot_id]:
            injected_ev_choice = max(raw_stage1[evidence_slot_id].items(), key=lambda kv: kv[1])[0]

        slots_out: dict[str, SlotReadout] = {}
        escalated_qids: list[str] = []
        escalation_reasons: list[str] = []

        for qid, q in active_questions.items():
            raw_dist = raw_stage1.get(qid, {})
            slot_prior = (
                ((prior_logits or {}).get(qid) or q.prior_logits or None)
                if qid != evidence_slot_id
                else None
            )
            fused_dist, h_raw, should_esc = pre_gated_bayesian_poe_fusion(
                raw_dist,
                prior_logits=slot_prior,
                prior_weight=self.prior_weight if slot_prior else 0.0,
                temperature=self.temperature,
                hesitation_threshold=self.hesitation_threshold,
                gate_mode=self.prior_gate_mode,
            )
            predicted = max(fused_dist.items(), key=lambda kv: kv[1])[0] if fused_dist else ""
            guard_name = "none"
            if qid != evidence_slot_id and (
                candidate_pack_sizes or visual_energies or rank1_visual_similarity is not None
            ):
                predicted, guard_name = apply_layer5_deterministic_guards(
                    chosen_id=predicted,
                    raw_probs=raw_dist,
                    candidate_pack_sizes=candidate_pack_sizes,
                    decoded_pack_size=decoded_pack_size,
                    visual_energies=visual_energies,
                    rank1_id=rank1_id,
                    rank1_visual_similarity=rank1_visual_similarity,
                )

            slot_ev = backend_evidence.get(qid, injected_ev_choice if qid != evidence_slot_id else "")
            inferred_flag = (
                self.escalate_on_inferred_evidence
                and qid != evidence_slot_id
                and slot_ev in _INFERRED_EVIDENCE_LABELS
            )

            if qid != evidence_slot_id and (should_esc or inferred_flag):
                escalated_qids.append(qid)
                if should_esc:
                    escalation_reasons.append(f"{qid}:h_raw={h_raw:.4f}>={self.hesitation_threshold}")
                if inferred_flag:
                    escalation_reasons.append(f"{qid}:evidence={slot_ev}")

            slots_out[qid] = SlotReadout(
                slot_id=qid,
                predicted=predicted,
                probabilities=fused_dist,
                raw_probabilities=raw_dist,
                h_raw=h_raw,
                confidence=1.0 - h_raw,
                escalated=False,
                guard_applied=guard_name,
                backend_name=self.stage1_backend.name,
                evidence=slot_ev,
            )

        # 5. Optional Stage-2 Frontier Escalation on Hesitant or Inferred-Evidence Slots
        did_escalate = False
        if escalated_qids and self.stage2_backend is not None:
            did_escalate = True
            esc_questions = {qid: active_questions[qid] for qid in escalated_qids}
            stage2_dists = self.stage2_backend.score_questions(state, esc_questions, images)
            s2_evidence: Mapping[str, str] = getattr(self.stage2_backend, "last_evidence", {}) or {}
            for qid in escalated_qids:
                if qid in stage2_dists and stage2_dists[qid]:
                    s2_dist = stage2_dists[qid]
                    s2_h = normalized_entropy(s2_dist)
                    s2_pred = max(s2_dist.items(), key=lambda kv: kv[1])[0]
                    slots_out[qid] = SlotReadout(
                        slot_id=qid,
                        predicted=s2_pred,
                        probabilities=s2_dist,
                        raw_probabilities=s2_dist,
                        h_raw=s2_h,
                        confidence=1.0 - s2_h,
                        escalated=True,
                        guard_applied=slots_out[qid].guard_applied,
                        backend_name=self.stage2_backend.name,
                        evidence=s2_evidence.get(qid, slots_out[qid].evidence),
                        passes_used=2,
                    )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return DecisionResult(
            slots=slots_out,
            evidence_slot_id=evidence_slot_id,
            escalated=did_escalate,
            escalation_reasons=escalation_reasons,
            latency_ms=elapsed_ms,
            backend_name=self.stage2_backend.name if did_escalate and self.stage2_backend else self.stage1_backend.name,
        )

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

"""Backend-agnostic adapters and Level-4 Total-Contract Cascade Decision Orchestrator.

Supports:
  1. DgemSystemOneBackend: /v1/systemone parallel masked diffusion endpoints.
  2. OpenAILogprobBackend: /v1/chat/completions single-token logprob readout (vLLM, SGLang, OpenAI, Gemma 4)
     with Vision `image_url` blocks, bounded multi-slot concurrency, and zero uniform fallback.
  3. VertexGeminiBackend: Vertex AI generateContent with single-token `[A..Z]` enum binding,
     `responseLogprobs` validation, and zero fabricated fallback distributions.
  4. MockCallableBackend: Deterministic in-memory adapter for unit testing and offline replay.
  5. CascadeDecisionOrchestrator: Unifies Stage-2A Fast-Exit, Wide-Option Shortlist, Co-Denoised/Causal
     Evidence Slot, Pre-Gated Bayesian PoE Prior Fusion, Layer-5 Guards, Hesitation-Gated Escalation,
     SHA-256 Content-Addressed Caching, and `RETRY_PENDING` / `REJECTED_CONTRACT_VIOLATION` Total Contracts.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import random
import re
import socket
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Collection, Mapping, Protocol

from .calibration import (
    apply_layer5_deterministic_guards,
    inject_evidence_slot,
    normalized_entropy,
    pre_gated_bayesian_poe_fusion,
    stage2a_fast_exit_gate,
)
from .contracts import (
    UNTRUSTED_BOUNDARY_NOTICE,
    ContentAddressedDecisionCache,
    DecisionContractError,
    LogprobAlignmentError,
    SchemaContractViolationError,
    UpstreamTransientError,
    _sanitize_value,
    compute_decision_cache_key,
    wrap_untrusted_state,
)
from .shortlist import shortlist_wide_question
from .types import (
    DEFAULT_HESITATION_THRESHOLD,
    MAX_OPTIONS_PER_SLOT,
    DecisionQuestion,
    DecisionResult,
    DecisionTerminalState,
    FailureCategory,
    SlotReadout,
)

OPENAI_MAX_TOP_LOGPROBS = 20
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_INFERRED_EVIDENCE_LABELS = frozenset({"inferred", "occluded_or_unreadable", "unreadable", "absent"})
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_OOS_KEYS = frozenset({"none_of_the_above", "none_visible", "oos"})


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


def _post_json_with_retry(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    timeout_sec: float,
    max_retries: int = 2,
    base_backoff_sec: float = 0.05,
    sla_deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Execute a JSON POST with bounded jittered retries on transient HTTP/network errors."""
    raw_body = json.dumps(payload).encode("utf-8")
    attempts = max(0, int(max_retries)) + 1
    last_err: Exception | None = None

    for attempt in range(attempts):
        now = time.monotonic()
        eff_timeout = timeout_sec
        if sla_deadline_monotonic is not None:
            rem = sla_deadline_monotonic - now
            if rem <= 0.0:
                raise UpstreamTransientError("SLA deadline exhausted before upstream call")
            eff_timeout = min(timeout_sec, rem)

        req = urllib.request.Request(url, data=raw_body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                raw_resp = resp.read().decode("utf-8")
            try:
                parsed = json.loads(raw_resp)
            except json.JSONDecodeError as exc:
                raise SchemaContractViolationError(f"upstream returned invalid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise SchemaContractViolationError("upstream JSON root must be an object")
            return parsed
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            if code in _RETRYABLE_HTTP_STATUS:
                last_err = UpstreamTransientError(
                    f"transient upstream HTTP {code} from {url}",
                    status_code=code,
                )
            else:
                raise SchemaContractViolationError(
                    f"non-retryable upstream HTTP {code} from {url}",
                    status_code=code,
                ) from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_err = UpstreamTransientError(f"upstream timeout or transport error: {exc}")

        if attempt + 1 < attempts:
            backoff = base_backoff_sec * (2**attempt) * (0.75 + 0.5 * random.random())
            if sla_deadline_monotonic is not None:
                rem = sla_deadline_monotonic - time.monotonic()
                if rem <= backoff:
                    break
            time.sleep(backoff)

    assert last_err is not None
    if isinstance(last_err, DecisionContractError):
        raise last_err
    raise UpstreamTransientError(str(last_err))


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
    """In-memory backend driven by a callable for deterministic unit testing and offline replay."""

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
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        name: str = "dgem_systemone",
    ) -> None:
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        self.endpoint_url = clean_url
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
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

        payload: dict[str, Any] = {"state": _sanitize_value(state), "questions": q_payload}
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
                    raise SchemaContractViolationError(f"missing answer object for question {qid!r}")
                probs = dict(ans.get("probabilities") or {})
                if not probs:
                    raise LogprobAlignmentError(f"empty probabilities map for question {qid!r}")
                if q.type == "score":
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
            parsed_preds = {
                str(qid): {str(k): float(v) for k, v in dist.items()}
                for qid, dist in preds.items()
                if isinstance(dist, Mapping) and dist
            }
            for qid in questions:
                if qid not in parsed_preds:
                    raise SchemaContractViolationError(f"missing prediction probabilities for {qid!r}")
            return parsed_preds, evidence_map, trace
        raise SchemaContractViolationError("invalid /v1/systemone response structure")

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
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = _post_json_with_retry(
            url,
            self.build_payload(state, questions, images),
            headers,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            base_backoff_sec=self.base_backoff_sec,
        )
        dists, ev_map, trace = self.parse_response(data, questions)
        self.last_evidence = ev_map
        self.last_trace = trace
        return dists


class OpenAILogprobBackend:
    """Adapter for OpenAI-compatible /v1/chat/completions endpoints (vLLM, SGLang, TGI, OpenAI).

    Maps each question's options to single uppercase letters [A..Z], fences untrusted state inside
    `<untrusted_data>`, supports Vision `image_url` blocks, runs multi-slot schemas concurrently,
    and fails closed (`LogprobAlignmentError`) instead of returning uniform `1/K` when valid option
    letters are absent from `top_logprobs`.
    """

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        max_concurrency: int = 8,
        letter_token_ids: Mapping[str, int] | None = None,
        min_captured_mass: float = 1e-6,
        name: str | None = None,
    ) -> None:
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        self.endpoint_url = clean_url
        self.model = model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_concurrency = max(1, int(max_concurrency))
        self.letter_token_ids = dict(letter_token_ids) if letter_token_ids else None
        self.min_captured_mass = min_captured_mass
        self.name = name or f"openai_logprob:{model}"
        self.last_evidence: dict[str, str] = {}

    @staticmethod
    def build_letter_prompt(
        state: Any,
        q: DecisionQuestion,
        *,
        allow_above_20: bool = False,
    ) -> tuple[str, dict[str, str]]:
        keys = q.option_keys()
        if len(keys) < 2:
            raise SchemaContractViolationError(f"question requires >= 2 options, got {len(keys)}")
        if len(keys) > MAX_OPTIONS_PER_SLOT:
            raise SchemaContractViolationError(
                f"question has {len(keys)} options; shortlist to <= {MAX_OPTIONS_PER_SLOT} first"
            )
        if len(keys) > OPENAI_MAX_TOP_LOGPROBS and not allow_above_20:
            raise SchemaContractViolationError(
                f"question has {len(keys)} options, exceeding OpenAI top_logprobs={OPENAI_MAX_TOP_LOGPROBS} "
                "ceiling without letter_token_ids logit_bias; shortlist to <= 20 options or supply letter_token_ids"
            )
        letters = string.ascii_uppercase[: len(keys)]
        letter_to_key = {letters[i]: keys[i] for i in range(len(keys))}
        lines = [
            UNTRUSTED_BOUNDARY_NOTICE,
            "State / Context:",
            wrap_untrusted_state(state),
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
        min_captured_mass: float = 1e-6,
    ) -> dict[str, float]:
        """Convert token top_logprobs into a normalized distribution; fail closed when unaligned."""
        if not top_logprobs:
            raise LogprobAlignmentError("upstream chat completion omitted top_logprobs")

        letter_logits: dict[str, float] = {}
        for letter, key in letter_to_key.items():
            best_lp = float("-inf")
            for tok, lp in top_logprobs.items():
                val = float(lp)
                if math.isnan(val):
                    raise LogprobAlignmentError(f"NaN logprob encountered on token {tok!r}")
                if tok.strip() == letter and val > best_lp:
                    best_lp = val
            letter_logits[key] = best_lp

        finite_vals = [v for v in letter_logits.values() if math.isfinite(v)]
        if not finite_vals:
            raise LogprobAlignmentError(
                f"zero valid option letters {list(letter_to_key.keys())} present in top_logprobs"
            )

        captured_mass = sum(math.exp(min(0.0, v)) for v in finite_vals)
        if captured_mass < min_captured_mass:
            raise LogprobAlignmentError(
                f"captured valid-letter probability mass {captured_mass:.3e} below {min_captured_mass:.3e}"
            )

        floor_lp = min(finite_vals) - 12.0
        max_lp = max(finite_vals)
        exps = {
            k: math.exp((v if math.isfinite(v) else floor_lp) - max_lp)
            for k, v in letter_logits.items()
        }
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    def _build_user_content(self, prompt_text: str, images: list[str] | None) -> Any:
        if not images:
            return prompt_text
        content_parts: list[dict[str, Any]] = []
        for img in images:
            clean_img = img.strip()
            if clean_img:
                content_parts.append({"type": "image_url", "image_url": {"url": clean_img}})
        content_parts.append({"type": "text", "text": prompt_text})
        return content_parts

    def _score_single_slot(
        self,
        url: str,
        headers: Mapping[str, str],
        state: Any,
        q: DecisionQuestion,
        images: list[str] | None,
    ) -> dict[str, float]:
        has_bias = self.letter_token_ids is not None
        prompt_text, letter_to_key = self.build_letter_prompt(state, q, allow_above_20=has_bias)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._build_user_content(prompt_text, images)}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(OPENAI_MAX_TOP_LOGPROBS, len(letter_to_key) + 4),
        }
        if self.letter_token_ids:
            payload["logit_bias"] = {
                str(self.letter_token_ids[ltr]): 100
                for ltr in letter_to_key
                if ltr in self.letter_token_ids
            }
        data = _post_json_with_retry(
            url,
            payload,
            headers,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            base_backoff_sec=self.base_backoff_sec,
        )
        choices = data.get("choices") or []
        if not choices:
            raise SchemaContractViolationError("upstream chat completion returned empty choices")
        choice0 = choices[0]
        content_lp = (choice0.get("logprobs") or {}).get("content") or []
        if not content_lp:
            raise LogprobAlignmentError("upstream chat completion missing logprobs.content")
        top_map: dict[str, float] = {}
        for entry in content_lp[0].get("top_logprobs") or []:
            top_map[str(entry["token"])] = float(entry["logprob"])
        return self.extract_letter_distribution(
            top_map,
            letter_to_key,
            min_captured_mass=self.min_captured_mass,
        )

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if not questions:
            return {}
        url = (
            self.endpoint_url
            if self.endpoint_url.endswith("/v1/chat/completions")
            else f"{self.endpoint_url}/v1/chat/completions"
        )
        _validate_http_url(url)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        q_items = list(questions.items())
        if len(q_items) == 1 or self.max_concurrency == 1:
            return {
                qid: self._score_single_slot(url, headers, state, q, images)
                for qid, q in q_items
            }

        out: dict[str, dict[str, float]] = {}
        workers = min(self.max_concurrency, len(q_items))
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(self._score_single_slot, url, headers, state, q, images): qid
                for qid, q in q_items
            }
            for fut in cf.as_completed(fut_map):
                qid = fut_map[fut]
                out[qid] = fut.result()
        return {qid: out[qid] for qid, _ in q_items}


class VertexGeminiBackend:
    """Adapter for Vertex AI Gemini models using Causal Evidence-First single-token enum schemas."""

    def __init__(
        self,
        project: str,
        model: str = "gemini-3.8-flash",
        location: str = "global",
        access_token_fn: Callable[[], str] | None = None,
        timeout_sec: float = 60.0,
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        endpoint_override: str | None = None,
    ) -> None:
        self.project = _validate_resource_segment("project", project)
        self.model = _validate_resource_segment("model", model)
        self.location = _validate_resource_segment("location", location)
        self.access_token_fn = access_token_fn
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
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
    def slot_letter_bijection(q: DecisionQuestion) -> tuple[dict[str, str], dict[str, str]]:
        """Map a slot's option_keys() bijectively onto single-token uppercase letters A..Z."""
        keys = q.option_keys()
        if len(keys) < 2 or len(keys) > MAX_OPTIONS_PER_SLOT:
            raise SchemaContractViolationError(
                f"VertexGeminiBackend requires 2..{MAX_OPTIONS_PER_SLOT} options per slot, got {len(keys)}"
            )
        letters = string.ascii_uppercase[: len(keys)]
        letter_to_key = {letters[i]: keys[i] for i in range(len(keys))}
        key_to_letter = {keys[i]: letters[i] for i in range(len(keys))}
        return letter_to_key, key_to_letter

    @classmethod
    def build_evidence_first_response_schema(
        cls,
        questions: Mapping[str, DecisionQuestion],
    ) -> dict[str, Any]:
        """Build a Gemini JSON responseSchema placing `evidence_read` FIRST and single-token letters per slot."""
        properties: dict[str, Any] = {
            "evidence_read": {
                "type": "STRING",
                "description": (
                    "Concise factual observation (3-8 words, such as printed pack size or visual cue) "
                    "decoded before selecting slot choices."
                ),
            }
        }
        required = ["evidence_read"]
        for qid, q in questions.items():
            letter_to_key, _ = cls.slot_letter_bijection(q)
            legend = ", ".join(f"{ltr}={key}" for ltr, key in letter_to_key.items())
            properties[qid] = {
                "type": "STRING",
                "enum": list(letter_to_key.keys()),
                "description": f"{q.instructions} (Select single uppercase letter: {legend})",
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
            UNTRUSTED_BOUNDARY_NOTICE,
            "Evaluate the state and answer each question using its assigned uppercase option letter.",
            "State:",
            wrap_untrusted_state(state),
            "Questions:",
        ]
        max_opts = 2
        for qid, q in questions.items():
            letter_to_key, _ = self.slot_letter_bijection(q)
            max_opts = max(max_opts, len(letter_to_key))
            opts_desc = ", ".join(
                f"{ltr}={k} ({q.criteria.get(k, k) if q.type != 'score' else k})"
                for ltr, k in letter_to_key.items()
            )
            prompt_lines.append(f"- {qid}: {q.instructions} | Options: {opts_desc}")
        parts.append({"text": "\n".join(prompt_lines)})

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": self.build_evidence_first_response_schema(questions),
                "responseLogprobs": True,
                "logprobs": min(20, max_opts + 2),
            },
        }

    @classmethod
    def parse_generate_content_response(
        cls,
        data: Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> tuple[dict[str, dict[str, float]], str]:
        """Parse Vertex Gemini response + exact token logprobs; fail closed (`LogprobAlignmentError`) if missing."""
        candidates = data.get("candidates") or []
        if not candidates:
            raise SchemaContractViolationError("Vertex Gemini response contained zero candidates")
        cand0 = candidates[0]
        parts = (cand0.get("content") or {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, Mapping))
        if not raw_text.strip():
            raise SchemaContractViolationError("Vertex Gemini candidate returned empty content text")
        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SchemaContractViolationError(f"Vertex Gemini JSON decode failed: {exc}") from exc
        if not isinstance(parsed_json, Mapping):
            raise SchemaContractViolationError("Vertex Gemini structured output must be a JSON object")

        evidence_read = str(parsed_json.get("evidence_read", ""))
        chosen_steps = (cand0.get("logprobsResult") or {}).get("chosenCandidates") or []
        top_steps = (cand0.get("logprobsResult") or {}).get("topCandidates") or []
        if not chosen_steps or not top_steps:
            raise LogprobAlignmentError(
                "Vertex Gemini response omitted logprobsResult; synthetic probability fallbacks are prohibited"
            )

        out: dict[str, dict[str, float]] = {}
        used_step_indices: set[int] = set()

        for qid, q in questions.items():
            letter_to_key, key_to_letter = cls.slot_letter_bijection(q)
            keys = q.option_keys()
            if qid not in parsed_json:
                raise SchemaContractViolationError(f"Vertex Gemini JSON missing required slot {qid!r}")
            raw_val = str(parsed_json[qid]).strip()
            if raw_val in letter_to_key:
                selected_letter = raw_val
                selected_key = letter_to_key[raw_val]
            elif raw_val in key_to_letter:
                selected_key = raw_val
                selected_letter = key_to_letter[raw_val]
            else:
                raise SchemaContractViolationError(
                    f"slot {qid!r} returned {raw_val!r}, outside valid letters {list(letter_to_key)} or keys {keys}"
                )

            matched_probs: dict[str, float] = {}
            for step_idx, chosen_tok in enumerate(chosen_steps):
                if step_idx in used_step_indices or step_idx >= len(top_steps):
                    continue
                tok_str = str(chosen_tok.get("token", "")).strip('" ')
                if tok_str not in (selected_letter, selected_key):
                    continue
                cands = top_steps[step_idx].get("candidates") or []
                logits: dict[str, float] = {}
                for ltr, k in letter_to_key.items():
                    best_lp = float("-inf")
                    for c_entry in cands:
                        ct = str(c_entry.get("token", "")).strip('" ')
                        if ct == ltr or ct == k:
                            lp = float(c_entry.get("logProbability", float("-inf")))
                            if math.isnan(lp):
                                raise LogprobAlignmentError(f"NaN logProbability on slot {qid!r}")
                            if lp > best_lp:
                                best_lp = lp
                    logits[k] = best_lp

                finite = [v for v in logits.values() if math.isfinite(v)]
                if finite and math.isfinite(logits.get(selected_key, float("-inf"))):
                    used_step_indices.add(step_idx)
                    floor_lp = min(finite) - 12.0
                    max_lp = max(finite)
                    exps = {
                        k: math.exp((v if math.isfinite(v) else floor_lp) - max_lp)
                        for k, v in logits.items()
                    }
                    tot = sum(exps.values())
                    matched_probs = {k: v / tot for k, v in exps.items()}
                    break

            if not matched_probs:
                raise LogprobAlignmentError(
                    f"failed to align single-token logprobs for slot {qid!r} (value={raw_val!r}); "
                    "multi-token split or missing token step rejected without fallback"
                )
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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        data = _post_json_with_retry(
            url,
            payload,
            headers,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            base_backoff_sec=self.base_backoff_sec,
        )
        dists, evidence_read = self.parse_generate_content_response(data, questions)
        self.last_evidence = {qid: evidence_read for qid in questions if evidence_read}
        return dists


class CascadeDecisionOrchestrator:
    """Level-4 Total-Contract Multimodal Decision Funnel Orchestrator."""

    def __init__(
        self,
        stage1_backend: DecisionBackend,
        stage2_backend: DecisionBackend | None = None,
        hesitation_threshold: float = DEFAULT_HESITATION_THRESHOLD,
        prior_weight: float = 0.0,
        prior_gate_mode: str = "hesitant",
        temperature: float = 1.0,
        shortlist_top_k: int = 0,
        shortlist_pairwise_diff: bool = False,
        evidence_prompt: str = "",
        evidence_criteria: Mapping[str, str] | None = None,
        enable_stage2a_fast_exit: bool = False,
        escalate_on_inferred_evidence: bool = False,
        total_contract_mode: bool = True,
        sla_timeout_sec: float = 180.0,
        response_cache: ContentAddressedDecisionCache | None = None,
        oos_keys: Collection[str] = _DEFAULT_OOS_KEYS,
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
        self.total_contract_mode = total_contract_mode
        self.sla_timeout_sec = sla_timeout_sec
        self.response_cache = response_cache
        self.oos_keys = frozenset(oos_keys)

    def _quarantine_result(
        self,
        questions: Mapping[str, DecisionQuestion],
        exc: DecisionContractError,
        t0: float,
        cache_key: str,
    ) -> DecisionResult:
        term = (
            DecisionTerminalState.RETRY_PENDING
            if exc.retryable
            else DecisionTerminalState.REJECTED_CONTRACT_VIOLATION
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        slots_err = {
            qid: SlotReadout(
                slot_id=qid,
                predicted="",
                probabilities={},
                raw_probabilities={},
                h_raw=1.0,
                confidence=0.0,
                escalated=False,
                guard_applied="contract_quarantine",
                backend_name=self.stage1_backend.name,
                passes_used=0,
                terminal_state=term,
                failure_category=exc.failure_category,
            )
            for qid in questions
        }
        return DecisionResult(
            slots=slots_err,
            escalated=False,
            escalation_reasons=[f"quarantine:{exc.failure_category.value}:{exc}"],
            latency_ms=elapsed_ms,
            backend_name=self.stage1_backend.name,
            raw_metadata={"error": str(exc), "status_code": exc.status_code},
            terminal_state=term,
            failure_category=exc.failure_category,
            cache_hit=False,
            cache_key=cache_key,
        )

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
        s2_name = self.stage2_backend.name if self.stage2_backend else "none"
        cache_key = compute_decision_cache_key(
            f"{self.stage1_backend.name}->{s2_name}",
            state,
            questions,
            images,
            extra_config={
                "hesitation_threshold": self.hesitation_threshold,
                "prior_weight": self.prior_weight,
                "prior_gate_mode": self.prior_gate_mode,
                "temperature": self.temperature,
                "shortlist_top_k": self.shortlist_top_k,
                "shortlist_pairwise_diff": self.shortlist_pairwise_diff,
                "evidence_prompt": self.evidence_prompt,
                "enable_stage2a_fast_exit": self.enable_stage2a_fast_exit,
                "sister_pack_collision": sister_pack_collision,
                "decoded_pack_size": decoded_pack_size,
            },
        )
        if self.response_cache is not None:
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                cached.latency_ms = (time.perf_counter() - t0) * 1000.0
                return cached

        try:
            return self._evaluate_inner(
                t0=t0,
                cache_key=cache_key,
                state=state,
                questions=questions,
                images=images,
                prior_logits=prior_logits,
                shortlist_scores=shortlist_scores,
                candidate_pack_sizes=candidate_pack_sizes,
                decoded_pack_size=decoded_pack_size,
                visual_energies=visual_energies,
                rank1_id=rank1_id,
                rank1_visual_similarity=rank1_visual_similarity,
                sister_pack_collision=sister_pack_collision,
            )
        except DecisionContractError as exc:
            if not self.total_contract_mode:
                raise
            return self._quarantine_result(questions, exc, t0, cache_key)

    def _evaluate_inner(
        self,
        *,
        t0: float,
        cache_key: str,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None,
        prior_logits: Mapping[str, Mapping[str, float]] | None,
        shortlist_scores: Mapping[str, Mapping[str, float]] | None,
        candidate_pack_sizes: Mapping[str, int] | None,
        decoded_pack_size: int | None,
        visual_energies: Mapping[str, float] | None,
        rank1_id: str | None,
        rank1_visual_similarity: float | None,
        sister_pack_collision: bool,
    ) -> DecisionResult:
        # 1. Stage 2A Adaptive Fast-Exit check for single-slot retrieval verification
        if self.enable_stage2a_fast_exit and len(questions) == 1 and prior_logits:
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
                    term = (
                        DecisionTerminalState.ABSTAIN_OOS
                        if winner in self.oos_keys
                        else DecisionTerminalState.STAGE2A_FAST_EXIT
                    )
                    res = DecisionResult(
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
                                passes_used=0,
                                terminal_state=term,
                                failure_category=(
                                    FailureCategory.ABSTAINED_OOS
                                    if term == DecisionTerminalState.ABSTAIN_OOS
                                    else FailureCategory.NONE
                                ),
                            )
                        },
                        escalated=False,
                        latency_ms=elapsed,
                        backend_name="stage2a_prior_fast_exit",
                        terminal_state=term,
                        cache_key=cache_key,
                    )
                    if self.response_cache is not None:
                        self.response_cache.put(cache_key, res)
                    return res

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

        # 4. Run Stage-1 Fast Backend with SLA check
        raw_stage1 = self.stage1_backend.score_questions(state, active_questions, images)
        if (time.perf_counter() - t0) > self.sla_timeout_sec:
            raise UpstreamTransientError(
                f"Stage-1 exceeded SLA budget ({self.sla_timeout_sec:.1f}s)"
            )
        backend_evidence: Mapping[str, str] = getattr(self.stage1_backend, "last_evidence", {}) or {}

        injected_ev_choice = ""
        if evidence_slot_id and evidence_slot_id in raw_stage1 and raw_stage1[evidence_slot_id]:
            injected_ev_choice = max(raw_stage1[evidence_slot_id].items(), key=lambda kv: kv[1])[0]

        slots_out: dict[str, SlotReadout] = {}
        escalated_qids: list[str] = []
        escalation_reasons: list[str] = []

        for qid, q in active_questions.items():
            raw_dist = raw_stage1.get(qid, {})
            if not raw_dist:
                raise SchemaContractViolationError(f"Stage-1 backend returned empty distribution for {qid!r}")
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
            predicted = max(fused_dist.items(), key=lambda kv: kv[1])[0]
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

            if predicted in self.oos_keys:
                slot_state = DecisionTerminalState.ABSTAIN_OOS
                f_cat = FailureCategory.ABSTAINED_OOS
            elif guard_name != "none":
                slot_state = DecisionTerminalState.GUARD_OVERRIDDEN
                f_cat = FailureCategory.NONE
            else:
                slot_state = DecisionTerminalState.STAGE1_RESOLVED
                f_cat = FailureCategory.NONE

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
                terminal_state=slot_state,
                failure_category=f_cat,
            )

        # 5. Optional Stage-2 Frontier Escalation on Hesitant or Inferred-Evidence Slots
        did_escalate = False
        if escalated_qids and self.stage2_backend is not None:
            if (time.perf_counter() - t0) >= self.sla_timeout_sec:
                raise UpstreamTransientError(
                    f"SLA budget ({self.sla_timeout_sec:.1f}s) exhausted prior to Stage-2 escalation"
                )
            did_escalate = True
            esc_questions = {qid: active_questions[qid] for qid in escalated_qids}
            stage2_dists = self.stage2_backend.score_questions(state, esc_questions, images)
            if (time.perf_counter() - t0) > self.sla_timeout_sec:
                raise UpstreamTransientError(
                    f"Stage-2 exceeded SLA budget ({self.sla_timeout_sec:.1f}s)"
                )
            s2_evidence: Mapping[str, str] = getattr(self.stage2_backend, "last_evidence", {}) or {}
            for qid in escalated_qids:
                s2_dist = stage2_dists.get(qid) or {}
                if not s2_dist:
                    raise SchemaContractViolationError(f"Stage-2 backend returned empty distribution for {qid!r}")
                s2_h = normalized_entropy(s2_dist)
                s2_pred = max(s2_dist.items(), key=lambda kv: kv[1])[0]
                s2_term = (
                    DecisionTerminalState.ABSTAIN_OOS
                    if s2_pred in self.oos_keys
                    else DecisionTerminalState.STAGE2_VERIFIED
                )
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
                    terminal_state=s2_term,
                    failure_category=(
                        FailureCategory.ABSTAINED_OOS
                        if s2_term == DecisionTerminalState.ABSTAIN_OOS
                        else FailureCategory.NONE
                    ),
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        non_ev_states = [
            s.terminal_state for sid, s in slots_out.items() if sid != evidence_slot_id
        ]
        if did_escalate:
            overall_state = DecisionTerminalState.STAGE2_VERIFIED
        elif any(st == DecisionTerminalState.ABSTAIN_OOS for st in non_ev_states):
            overall_state = DecisionTerminalState.ABSTAIN_OOS
        elif any(st == DecisionTerminalState.GUARD_OVERRIDDEN for st in non_ev_states):
            overall_state = DecisionTerminalState.GUARD_OVERRIDDEN
        else:
            overall_state = DecisionTerminalState.STAGE1_RESOLVED

        result = DecisionResult(
            slots=slots_out,
            evidence_slot_id=evidence_slot_id,
            escalated=did_escalate,
            escalation_reasons=escalation_reasons,
            latency_ms=elapsed_ms,
            backend_name=(
                self.stage2_backend.name if did_escalate and self.stage2_backend else self.stage1_backend.name
            ),
            terminal_state=overall_state,
            failure_category=(
                FailureCategory.ABSTAINED_OOS
                if overall_state == DecisionTerminalState.ABSTAIN_OOS
                else FailureCategory.NONE
            ),
            cache_hit=False,
            cache_key=cache_key,
        )
        if self.response_cache is not None:
            self.response_cache.put(cache_key, result)
        return result

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
  4. BlackBoxSampledBackend: Logprob-free multi-sample and verbalized-vote adapter with symmetric
     Dirichlet/Jeffreys prior smoothing and posterior epistemic variance.
  5. MockCallableBackend: Deterministic in-memory adapter for unit testing and offline replay.
  6. CascadeDecisionOrchestrator: Unifies Stage-2A Fast-Exit, Wide-Option Shortlist, Co-Denoised/Causal
     Evidence Slot, Pre-Fusion Split Conformal / RAPS Set-Size Gating, Pre-Gated Bayesian PoE Prior Fusion,
     Layer-5 Guards, Hesitation-Gated Escalation, SHA-256 Content-Addressed Caching, and
     `RETRY_PENDING` / `REJECTED_CONTRACT_VIOLATION` Total Contracts.
"""

from __future__ import annotations

import concurrent.futures as cf
import inspect
import json
import math
import os
import random
import re
import socket
import string
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Collection, Mapping, Protocol

from .calibration import (
    VALID_PRIOR_GATE_MODES,
    apply_layer5_deterministic_guards,
    inject_evidence_slot,
    normalized_entropy,
    pre_gated_bayesian_poe_fusion,
    stage2a_fast_exit_gate,
)
from .conformal import (
    DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    conformal_prediction_set,
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
    MAX_EXTENDED_OPTIONS_PER_SLOT,
    MAX_OPTIONS_PER_SLOT,
    UNIVERSAL_256_DECISION_CODES,
    DecisionQuestion,
    DecisionResult,
    DecisionTerminalState,
    FailureCategory,
    SlotReadout,
)

OPENAI_MAX_TOP_LOGPROBS = 20
_THINKING_TAG_TOKENS = frozenset({"<think>", "</think>", "<|thought|>", "<thought>", "thought"})


def build_extended_255_code_bijection(
    option_keys: Sequence[str],
    vocab: Collection[str] | None = None,
) -> dict[str, str]:
    """Map up to 256 option keys to single-token uppercase codes (A..Z, AA..PT).

    Follows arXiv:2610.03935 Section 6.1 and DecisionBench 1.0 (imajev-4b / mojev)
    while filtering out rare two-letter pairs (e.g., BQ, CJ, FQ, GZ) that split into
    2 BPE sub-tokens in Qwen3, DeepSeek-V3, Gemma-3, Llama-3.2, or SmolLM2 tokenizers.
    When a custom `vocab` is supplied, dynamically selects single-token codes verified
    to exist in `vocab`.
    """
    keys = [str(k) for k in option_keys]
    if len(keys) < 2:
        raise SchemaContractViolationError(f"question requires >= 2 options, got {len(keys)}")
    if len(keys) > MAX_EXTENDED_OPTIONS_PER_SLOT:
        raise SchemaContractViolationError(
            f"question has {len(keys)} options, exceeding extended single-token ceiling {MAX_EXTENDED_OPTIONS_PER_SLOT}"
        )
    if len(set(keys)) != len(keys):
        raise SchemaContractViolationError("option_keys must not contain duplicate keys")

    if vocab is not None:
        all_candidates = list(string.ascii_uppercase) + [
            a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
        ]
        valid_codes = [c for c in all_candidates if c in vocab]
        if len(valid_codes) < len(keys):
            raise SchemaContractViolationError(
                f"supplied vocabulary contains only {len(valid_codes)} single-token uppercase codes, need {len(keys)}"
            )
        selected_codes = valid_codes[: len(keys)]
    elif len(keys) <= MAX_OPTIONS_PER_SLOT:
        selected_codes = list(string.ascii_uppercase[: len(keys)])
    else:
        selected_codes = list(UNIVERSAL_256_DECISION_CODES[: len(keys)])

    return {selected_codes[i]: keys[i] for i in range(len(keys))}
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_INFERRED_EVIDENCE_LABELS = frozenset({"inferred", "occluded_or_unreadable", "unreadable", "absent"})
_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DEFAULT_OOS_KEYS = frozenset({"none_of_the_above", "none_visible", "oos"})
_EXPLICIT_ABSTAIN_ALIASES = frozenset(
    {
        "abstain",
        "insufficient_evidence",
        "none",
        "none_of_the_above",
        "none_visible",
        "oos",
        "out_of_scope",
        "out_of_stock",
        "uncertain",
        "unknown",
    }
)
_ALLOWED_DRAW_META_KEYS = frozenset(
    {"evidence", "evidence_read", "explanation", "rationale", "reasoning"}
)
_CHOICE_ALIAS_KEYS = ("choice", "answer", "selected", "predicted")


def _validate_http_url(url: str) -> str:
    """Validate that an endpoint URL uses http or https with a non-empty host (blocks file:// SSRF)."""
    if not isinstance(url, str):
        raise SchemaContractViolationError(f"endpoint URL must be a string, got {url!r}")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SchemaContractViolationError(
            f"endpoint URL must use http:// or https:// with a valid host, got {url!r}"
        )
    return url


def _validate_resource_segment(label: str, value: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_SEGMENT_RE.match(value):
        raise SchemaContractViolationError(f"invalid {label} identifier: {value!r}")
    return value


def _require_finite_number(val: Any, field: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise SchemaContractViolationError(
            f"{field} must be a finite non-boolean number, got {val!r}"
        )
    out = float(val)
    if not math.isfinite(out):
        raise SchemaContractViolationError(f"{field} must be finite, got {val!r}")
    return 0.0 if out == 0.0 else out


def _require_nonneg_finite_number(val: Any, field: str) -> float:
    out = _require_finite_number(val, field)
    if out < 0.0:
        raise SchemaContractViolationError(f"{field} must be >= 0, got {val!r}")
    return out


def _require_positive_finite_number(val: Any, field: str) -> float:
    out = _require_finite_number(val, field)
    if out <= 0.0:
        raise SchemaContractViolationError(f"{field} must be > 0, got {val!r}")
    return out


def _require_strict_int(val: Any, field: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise SchemaContractViolationError(
            f"{field} must be a non-boolean integer, got {val!r}"
        )
    return int(val)


def _require_nonneg_int(val: Any, field: str) -> int:
    out = _require_strict_int(val, field)
    if out < 0:
        raise SchemaContractViolationError(f"{field} must be >= 0, got {val!r}")
    return out


def _require_positive_int(val: Any, field: str) -> int:
    out = _require_strict_int(val, field)
    if out < 1:
        raise SchemaContractViolationError(f"{field} must be >= 1, got {val!r}")
    return out


def _validate_slot_distribution(dist: Any, label: str) -> dict[str, float]:
    if not isinstance(dist, Mapping) or not dist:
        raise SchemaContractViolationError(f"{label} returned empty or non-mapping distribution")
    clean: dict[str, float] = {}
    total = 0.0
    for raw_k, raw_v in dist.items():
        if not isinstance(raw_k, str) or not raw_k:
            raise SchemaContractViolationError(
                f"{label} distribution keys must be non-empty strings, got {raw_k!r}"
            )
        prob = _require_nonneg_finite_number(raw_v, f"{label}[{raw_k!r}]")
        clean[raw_k] = prob
        total += prob
    if total <= 0.0 or not math.isfinite(total):
        raise SchemaContractViolationError(f"{label} probability mass must be positive and finite")
    return clean


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
    try:
        raw_body = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaContractViolationError(f"failed to serialize request payload: {exc}") from exc
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
        if not isinstance(name, str) or not name.strip():
            raise SchemaContractViolationError("MockCallableBackend name must be a non-empty string")
        if not callable(fn):
            raise SchemaContractViolationError("MockCallableBackend fn must be callable")
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
        if not isinstance(endpoint_url, str):
            raise SchemaContractViolationError(
                f"endpoint_url must be a string, got {endpoint_url!r}"
            )
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        if api_key is not None and not isinstance(api_key, str):
            raise SchemaContractViolationError(
                f"api_key must be a string or None, got {api_key!r}"
            )
        if not isinstance(name, str) or not name.strip():
            raise SchemaContractViolationError("DgemSystemOneBackend name must be a non-empty string")
        self.endpoint_url = clean_url
        self.api_key = api_key
        self.timeout_sec = _require_positive_finite_number(timeout_sec, "timeout_sec")
        self.max_retries = _require_nonneg_int(max_retries, "max_retries")
        self.base_backoff_sec = _require_nonneg_finite_number(base_backoff_sec, "base_backoff_sec")
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
            merged_priors: dict[str, float] = {
                str(pk): _require_finite_number(pv, f"prior_logits[{qid!r}][{pk!r}]")
                for pk, pv in q.prior_logits.items()
            }
            if prior_logits and qid in prior_logits:
                slot_pl = prior_logits[qid]
                if not isinstance(slot_pl, Mapping):
                    raise SchemaContractViolationError(f"prior_logits[{qid!r}] must be a mapping")
                for pk, pv in slot_pl.items():
                    merged_priors[str(pk)] = _require_finite_number(
                        pv, f"prior_logits[{qid!r}][{pk!r}]"
                    )
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
        if not isinstance(data, Mapping):
            raise SchemaContractViolationError("invalid /v1/systemone response root")
        evidence_map: dict[str, str] = {}
        raw_trace = data.get("evaluation_trace")
        if raw_trace is not None and not isinstance(raw_trace, Mapping):
            raise SchemaContractViolationError("evaluation_trace must be a JSON object")
        trace = dict(raw_trace) if isinstance(raw_trace, Mapping) else {}

        if "answers" in data:
            answers = data["answers"]
            if not isinstance(answers, Mapping):
                raise SchemaContractViolationError("answers field must be a JSON object")
            out: dict[str, dict[str, float]] = {}
            for qid, q in questions.items():
                ans = answers.get(qid)
                if not isinstance(ans, Mapping):
                    raise SchemaContractViolationError(f"missing answer object for question {qid!r}")
                raw_probs = ans.get("probabilities")
                if raw_probs is None or (isinstance(raw_probs, Mapping) and not raw_probs):
                    raise LogprobAlignmentError(f"empty probabilities map for question {qid!r}")
                if not isinstance(raw_probs, Mapping):
                    raise SchemaContractViolationError(
                        f"probabilities for question {qid!r} must be a JSON object"
                    )
                if q.type == "score":
                    legend = ans.get("legend")
                    if legend is not None and not isinstance(legend, Mapping):
                        raise SchemaContractViolationError(
                            f"legend for score question {qid!r} must be a JSON object"
                        )
                    mapped_probs: dict[str, float] = {}
                    for idx_key, p_val in raw_probs.items():
                        idx_str = str(idx_key)
                        prob = _require_nonneg_finite_number(
                            p_val, f"answers[{qid!r}].probabilities[{idx_str!r}]"
                        )
                        if isinstance(legend, Mapping) and idx_str in legend:
                            mapped_probs[str(legend[idx_str])] = prob
                        elif idx_str.isdigit() and int(idx_str) < len(q.levels):
                            mapped_probs[q.levels[int(idx_str)]] = prob
                        else:
                            mapped_probs[idx_str] = prob
                    out[qid] = _validate_slot_distribution(mapped_probs, f"SystemOne slot {qid!r}")
                else:
                    out[qid] = _validate_slot_distribution(raw_probs, f"SystemOne slot {qid!r}")
                ev = ans.get("evidence")
                if isinstance(ev, str) and ev:
                    evidence_map[qid] = ev
            return out, evidence_map, trace

        preds = data.get("predictions", data)
        if isinstance(preds, Mapping):
            parsed_preds: dict[str, dict[str, float]] = {}
            for qid in questions:
                if qid not in preds:
                    raise SchemaContractViolationError(f"missing prediction probabilities for {qid!r}")
                raw_dist = preds[qid]
                if raw_dist is None or (isinstance(raw_dist, Mapping) and not raw_dist):
                    raise LogprobAlignmentError(f"empty probabilities map for question {qid!r}")
                parsed_preds[qid] = _validate_slot_distribution(raw_dist, f"SystemOne slot {qid!r}")
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


class GevStudioSystemOneBackend:
    """Adapter for Gev Studio (/api/gev/systemone) multi-target System 1 decision endpoints.

    Supports all Gev Studio backend targets via `?target=...` (e.g., `gemini-v4xs-decisions`,
    `beyond-decision:/models/gemini-v4xs-decisions?hc=true`, `gemini-3.5-flash-lite`,
    `gemini-3.1-flash-lite`, `dgem:vertex`, `saturn-moe-3p8b-qat-v0-fst-genmedia`),
    multimodal `media` payloads (`image/png`, `image/jpeg`, `audio/wav`, `video/mp4`),
    and all 3 question types (`choice`, `noul`, `score`).
    """

    def __init__(
        self,
        endpoint_url: str = "https://gevstudio.corp.google.com/api/gev/systemone",
        target: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        use_sso_client: bool = False,
        allow_discrete_fallback: bool = True,
        label_smoothing_eps: float = 1e-4,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        name: str | None = None,
    ) -> None:
        if not isinstance(endpoint_url, str):
            raise SchemaContractViolationError(
                f"endpoint_url must be a string, got {endpoint_url!r}"
            )
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise SchemaContractViolationError(f"target must be a non-empty string or None, got {target!r}")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise SchemaContractViolationError(f"model must be a non-empty string or None, got {model!r}")
        if temperature is not None:
            temperature = _require_nonneg_finite_number(temperature, "temperature")
        if api_key is not None and not isinstance(api_key, str):
            raise SchemaContractViolationError(f"api_key must be a string or None, got {api_key!r}")
        if not isinstance(use_sso_client, bool):
            raise SchemaContractViolationError("use_sso_client must be a boolean")
        if not isinstance(allow_discrete_fallback, bool):
            raise SchemaContractViolationError("allow_discrete_fallback must be a boolean")
        clean_eps = _require_positive_finite_number(label_smoothing_eps, "label_smoothing_eps")
        if clean_eps >= 0.5:
            raise SchemaContractViolationError(
                f"label_smoothing_eps must be in (0, 0.5), got {label_smoothing_eps!r}"
            )
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise SchemaContractViolationError(
                f"GevStudioSystemOneBackend name must be a non-empty string when provided, got {name!r}"
            )
        self.endpoint_url = clean_url
        self.target = target.strip() if target else None
        self.model = model.strip() if model else None
        self.temperature = temperature
        self.api_key = api_key
        self.use_sso_client = use_sso_client
        self.allow_discrete_fallback = allow_discrete_fallback
        self.label_smoothing_eps = clean_eps
        self.timeout_sec = _require_positive_finite_number(timeout_sec, "timeout_sec")
        self.max_retries = _require_nonneg_int(max_retries, "max_retries")
        self.base_backoff_sec = _require_nonneg_finite_number(base_backoff_sec, "base_backoff_sec")
        default_suffix = self.target or self.model or "default"
        self.name = name or f"gev_studio:{default_suffix}"
        self.last_evidence: dict[str, str] = {}
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _infer_mime_and_b64(raw_item: str) -> dict[str, str]:
        """Parse a data URI or raw base64 string into Gev Studio's {mime_type, data} media block."""
        s = raw_item.strip()
        if s.startswith("data:"):
            semi = s.find(";base64,")
            if semi > 5:
                mime = s[5:semi].strip() or "image/png"
                b64 = s[semi + 8 :].strip()
                return {"mime_type": mime, "data": b64}
        if s.startswith("/9j/"):
            return {"mime_type": "image/jpeg", "data": s}
        if s.startswith("iVBOR"):
            return {"mime_type": "image/png", "data": s}
        if s.startswith("R0lGR"):
            return {"mime_type": "image/webp", "data": s}
        if s.startswith("UklGR"):
            return {"mime_type": "audio/wav", "data": s}
        return {"mime_type": "image/png", "data": s}

    def build_url(self) -> str:
        """Return the resolved /api/gev/systemone URL with URL-encoded ?target=... query parameter."""
        parsed = urllib.parse.urlparse(self.endpoint_url)
        base = (
            self.endpoint_url
            if parsed.path.endswith("/api/gev/systemone") or parsed.path.endswith("/v1/systemone")
            else f"{self.endpoint_url}/api/gev/systemone"
        )
        if not self.target:
            return base
        encoded_target = urllib.parse.quote(self.target, safe="")
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}target={encoded_target}"

    def build_payload(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
        media: list[Mapping[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Construct a Gev Studio systemone.Request JSON payload."""
        clean_state = state
        extracted_media: list[dict[str, str]] = []
        if isinstance(state, Mapping) and "__media__" in state:
            clean_state = {k: v for k, v in state.items() if k != "__media__"}
            raw_m = state.get("__media__")
            if isinstance(raw_m, list):
                for m_item in raw_m:
                    if isinstance(m_item, Mapping) and m_item.get("data"):
                        extracted_media.append(
                            {
                                "mime_type": str(m_item.get("mime_type") or "image/png"),
                                "data": str(m_item["data"]),
                            }
                        )
                    elif isinstance(m_item, str) and m_item.strip():
                        extracted_media.append(self._infer_mime_and_b64(m_item))

        if images:
            for img in images:
                if isinstance(img, str) and img.strip():
                    extracted_media.append(self._infer_mime_and_b64(img))

        if media:
            for m_item in media:
                if isinstance(m_item, Mapping) and m_item.get("data"):
                    extracted_media.append(
                        {
                            "mime_type": str(m_item.get("mime_type") or "image/png"),
                            "data": str(m_item["data"]),
                        }
                    )

        q_payload: dict[str, Any] = {}
        for qid, q in questions.items():
            if q.type == "score":
                q_payload[qid] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": list(q.levels),
                }
            elif q.type == "noul":
                q_payload[qid] = {
                    "type": "noul",
                    "instructions": q.instructions,
                }
            else:
                ordered_crit = {k: q.criteria[k] for k in q.option_keys() if k in q.criteria}
                q_payload[qid] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": ordered_crit,
                }

        payload: dict[str, Any] = {
            "state": _sanitize_value(clean_state),
            "questions": q_payload,
        }
        if self.model:
            payload["model"] = self.model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if extracted_media:
            payload["media"] = extracted_media
        return payload

    @staticmethod
    def _resolve_noul_keys(q: DecisionQuestion) -> tuple[str, str]:
        """Resolve (negative_key, positive_key) for a noul question."""
        keys = q.option_keys()
        if len(keys) == 2:
            k0, k1 = str(keys[0]), str(keys[1])
            pos_aliases = {"1", "true", "yes", "positive", "pass", "valid"}
            neg_aliases = {"0", "false", "no", "negative", "fail", "invalid"}
            if k0.lower() in pos_aliases and k1.lower() in neg_aliases:
                return k1, k0
            return k0, k1
        return "0", "1"

    @staticmethod
    def _smoothed_discrete_distribution(
        chosen_key: str,
        option_keys: list[str],
        eps: float,
    ) -> dict[str, float]:
        if len(option_keys) < 2:
            return {chosen_key: 1.0}
        rem = eps / float(len(option_keys) - 1)
        return {k: (1.0 - eps if k == chosen_key else rem) for k in option_keys}

    def parse_response(
        self,
        data: Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
    ) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, Any]]:
        """Parse Gev Studio /api/gev/systemone response across choice, noul, and score slots."""
        if not isinstance(data, Mapping):
            raise SchemaContractViolationError("invalid Gev Studio /api/gev/systemone response root")
        if "error" in data and data["error"] and "answers" not in data:
            raise UpstreamTransientError(f"Gev Studio error response: {data['error']}")
        answers = data.get("answers")
        if not isinstance(answers, Mapping):
            raise SchemaContractViolationError("Gev Studio response missing 'answers' object")

        out: dict[str, dict[str, float]] = {}
        evidence_map: dict[str, str] = {}
        discrete_fallback_slots: list[str] = []

        for qid, q in questions.items():
            ans = answers.get(qid)
            if not isinstance(ans, Mapping):
                raise SchemaContractViolationError(f"missing answer object for question {qid!r}")
            raw_probs = ans.get("probabilities")

            if q.type == "noul":
                neg_key, pos_key = self._resolve_noul_keys(q)
                p1: float | None = None
                if isinstance(raw_probs, Mapping) and raw_probs:
                    if "1" in raw_probs:
                        p1 = _require_nonneg_finite_number(raw_probs["1"], f"answers[{qid!r}].probabilities['1']")
                    elif pos_key in raw_probs:
                        p1 = _require_nonneg_finite_number(
                            raw_probs[pos_key], f"answers[{qid!r}].probabilities[{pos_key!r}]"
                        )
                    elif "true" in raw_probs:
                        p1 = _require_nonneg_finite_number(
                            raw_probs["true"], f"answers[{qid!r}].probabilities['true']"
                        )
                if p1 is None and ans.get("noul_prob") is not None:
                    p1 = _require_nonneg_finite_number(ans["noul_prob"], f"answers[{qid!r}].noul_prob")
                if p1 is None and isinstance(ans.get("noul"), bool):
                    if not self.allow_discrete_fallback:
                        raise LogprobAlignmentError(f"missing noul probabilities for slot {qid!r}")
                    p1 = (1.0 - self.label_smoothing_eps) if ans["noul"] else self.label_smoothing_eps
                    discrete_fallback_slots.append(qid)
                if p1 is None:
                    raise LogprobAlignmentError(f"unable to extract noul probability for slot {qid!r}")
                p1_clamped = min(1.0 - 1e-9, max(1e-9, float(p1)))
                out[qid] = {neg_key: 1.0 - p1_clamped, pos_key: p1_clamped}

            elif q.type == "score":
                levels = list(q.levels)
                if not levels:
                    raise SchemaContractViolationError(f"score question {qid!r} has empty levels")
                if isinstance(raw_probs, Mapping) and raw_probs:
                    legend = ans.get("legend")
                    mapped_probs: dict[str, float] = {}
                    for idx_key, p_val in raw_probs.items():
                        idx_str = str(idx_key)
                        prob = _require_nonneg_finite_number(
                            p_val, f"answers[{qid!r}].probabilities[{idx_str!r}]"
                        )
                        if isinstance(legend, Mapping) and idx_str in legend:
                            mapped_probs[str(legend[idx_str])] = prob
                        elif idx_str.isdigit() and int(idx_str) < len(levels):
                            mapped_probs[levels[int(idx_str)]] = prob
                        elif idx_str in levels:
                            mapped_probs[idx_str] = prob
                    for lvl in levels:
                        mapped_probs.setdefault(lvl, 1e-9)
                    total = sum(mapped_probs.values())
                    out[qid] = {k: v / total for k, v in mapped_probs.items()}
                elif ans.get("score") is not None and self.allow_discrete_fallback:
                    s_idx = _require_nonneg_int(ans["score"], f"answers[{qid!r}].score")
                    chosen_lvl = levels[min(s_idx, len(levels) - 1)]
                    out[qid] = self._smoothed_discrete_distribution(
                        chosen_lvl, levels, self.label_smoothing_eps
                    )
                    discrete_fallback_slots.append(qid)
                else:
                    raise LogprobAlignmentError(f"empty probabilities map for score question {qid!r}")

            else:
                opt_keys = q.option_keys()
                if isinstance(raw_probs, Mapping) and raw_probs:
                    cleaned = _validate_slot_distribution(raw_probs, f"GevStudio slot {qid!r}")
                    if opt_keys:
                        aligned: dict[str, float] = {}
                        for k in opt_keys:
                            aligned[k] = float(cleaned.get(k, 1e-9))
                        total = sum(aligned.values())
                        out[qid] = {k: v / total for k, v in aligned.items()}
                    else:
                        total = sum(cleaned.values())
                        out[qid] = {k: v / total for k, v in cleaned.items()}
                elif isinstance(ans.get("choice"), str) and ans["choice"] and self.allow_discrete_fallback:
                    chosen = str(ans["choice"])
                    if opt_keys and chosen not in opt_keys:
                        raise SchemaContractViolationError(
                            f"discrete choice {chosen!r} for slot {qid!r} not in option_keys {opt_keys}"
                        )
                    keys_for_dist = opt_keys if opt_keys else [chosen]
                    out[qid] = self._smoothed_discrete_distribution(
                        chosen, keys_for_dist, self.label_smoothing_eps
                    )
                    discrete_fallback_slots.append(qid)
                else:
                    raise LogprobAlignmentError(f"empty probabilities map for choice question {qid!r}")

            ev = ans.get("evidence")
            if isinstance(ev, str) and ev:
                evidence_map[qid] = ev

        trace: dict[str, Any] = {
            "target": self.target or "default",
            "discrete_fallback_slots": discrete_fallback_slots,
        }
        return out, evidence_map, trace

    def _post_via_sso_client(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute POST via Google corp `sso_client` CLI using a /tmp payload file."""
        try:
            raw_json = json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaContractViolationError(f"failed to serialize request payload: {exc}") from exc

        attempts = max(0, int(self.max_retries)) + 1
        last_err: Exception | None = None
        timeout_int = max(5, int(math.ceil(self.timeout_sec)))

        for attempt in range(attempts):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="gev_req_", dir="/tmp", delete=True
            ) as tmp_f:
                tmp_f.write(raw_json)
                tmp_f.flush()
                cmd = [
                    "sso_client",
                    f"--url={url}",
                    "--method=POST",
                    "--headers=Content-Type: application/json",
                    f"--data_file={tmp_f.name}",
                    f"--request_timeout={timeout_int}",
                ]
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_sec + 5.0,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    last_err = UpstreamTransientError(f"sso_client timed out after {self.timeout_sec}s: {exc}")
                    proc = None

            if proc is not None:
                stdout = (proc.stdout or "").strip()
                if stdout.startswith("{"):
                    try:
                        parsed = json.loads(stdout)
                    except json.JSONDecodeError as exc:
                        last_err = UpstreamTransientError(f"invalid JSON from sso_client: {exc}")
                        parsed = None
                    if isinstance(parsed, dict):
                        if "error" in parsed and "answers" not in parsed:
                            last_err = UpstreamTransientError(
                                f"Gev Studio error from {url}: {parsed.get('error')}"
                            )
                        else:
                            return parsed
                else:
                    stderr_snip = (proc.stderr or stdout)[:240]
                    last_err = UpstreamTransientError(
                        f"sso_client non-JSON response (rc={proc.returncode}): {stderr_snip}"
                    )

            if attempt + 1 < attempts:
                backoff = self.base_backoff_sec * (2**attempt) * (0.75 + 0.5 * random.random())
                time.sleep(backoff)

        assert last_err is not None
        if isinstance(last_err, DecisionContractError):
            raise last_err
        raise UpstreamTransientError(str(last_err))

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if not questions:
            return {}
        url = self.build_url()
        _validate_http_url(url)
        payload = self.build_payload(state, questions, images)
        if self.use_sso_client:
            data = self._post_via_sso_client(url, payload)
        else:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            data = _post_json_with_retry(
                url,
                payload,
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
        if not isinstance(endpoint_url, str):
            raise SchemaContractViolationError(
                f"endpoint_url must be a string, got {endpoint_url!r}"
            )
        clean_url = endpoint_url.rstrip("/")
        _validate_http_url(clean_url)
        if not isinstance(model, str) or not model.strip():
            raise SchemaContractViolationError(f"model must be a non-empty string, got {model!r}")
        if api_key is not None and not isinstance(api_key, str):
            raise SchemaContractViolationError(f"api_key must be a string or None, got {api_key!r}")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise SchemaContractViolationError(
                f"OpenAILogprobBackend name must be a non-empty string when provided, got {name!r}"
            )
        clean_mass = _require_positive_finite_number(min_captured_mass, "min_captured_mass")
        if clean_mass > 1.0:
            raise SchemaContractViolationError(
                f"min_captured_mass must be in (0, 1], got {min_captured_mass!r}"
            )
        clean_token_ids: dict[str, int] | None = None
        if letter_token_ids is not None:
            if not isinstance(letter_token_ids, Mapping):
                raise SchemaContractViolationError("letter_token_ids must be a mapping when provided")
            if letter_token_ids:
                clean_token_ids = {}
                for ltr, tid in letter_token_ids.items():
                    if not isinstance(ltr, str) or not ltr.strip():
                        raise SchemaContractViolationError(
                            f"letter_token_ids keys must be non-empty strings, got {ltr!r}"
                        )
                    clean_token_ids[ltr] = _require_nonneg_int(tid, f"letter_token_ids[{ltr!r}]")
        self.endpoint_url = clean_url
        self.model = model
        self.api_key = api_key
        self.timeout_sec = _require_positive_finite_number(timeout_sec, "timeout_sec")
        self.max_retries = _require_nonneg_int(max_retries, "max_retries")
        self.base_backoff_sec = _require_nonneg_finite_number(base_backoff_sec, "base_backoff_sec")
        self.max_concurrency = _require_positive_int(max_concurrency, "max_concurrency")
        self.letter_token_ids = clean_token_ids
        self.min_captured_mass = clean_mass
        self.name = name or f"openai_logprob:{model}"
        self.last_evidence: dict[str, str] = {}

    @staticmethod
    def build_letter_prompt(
        state: Any,
        q: DecisionQuestion,
        *,
        allow_above_20: bool = False,
        allow_extended_255: bool = False,
        vocab: Collection[str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        keys = q.option_keys()
        if len(keys) < 2:
            raise SchemaContractViolationError(f"question requires >= 2 options, got {len(keys)}")
        max_allowed = MAX_EXTENDED_OPTIONS_PER_SLOT if allow_extended_255 else MAX_OPTIONS_PER_SLOT
        if len(keys) > max_allowed:
            raise SchemaContractViolationError(
                f"question has {len(keys)} options; shortlist to <= {max_allowed} first"
            )
        if len(keys) > OPENAI_MAX_TOP_LOGPROBS and not (allow_above_20 or allow_extended_255):
            raise SchemaContractViolationError(
                f"question has {len(keys)} options, exceeding OpenAI top_logprobs={OPENAI_MAX_TOP_LOGPROBS} "
                "ceiling without letter_token_ids logit_bias; shortlist to <= 20 options or supply letter_token_ids"
            )
        if len(keys) <= MAX_OPTIONS_PER_SLOT and vocab is None:
            letters = string.ascii_uppercase[: len(keys)]
            letter_to_key = {letters[i]: keys[i] for i in range(len(keys))}
        else:
            letter_to_key = build_extended_255_code_bijection(keys, vocab=vocab)
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
        code_kind = "option code" if len(keys) > MAX_OPTIONS_PER_SLOT else "single uppercase option letter"
        lines.append(f"\nReply with the {code_kind} only.")
        return "\n".join(lines), letter_to_key

    @staticmethod
    def extract_letter_distribution(
        top_logprobs: Mapping[str, float],
        letter_to_key: Mapping[str, str],
        min_captured_mass: float = 1e-6,
    ) -> dict[str, float]:
        """Convert token top_logprobs into a normalized distribution; fail closed when unaligned."""
        if not isinstance(top_logprobs, Mapping) or not top_logprobs:
            raise LogprobAlignmentError("upstream chat completion omitted top_logprobs")

        # Detect unstripped thinking-tag leakage (e.g., '<think>' occupying top-1 token when
        # enable_thinking=False was omitted on hybrid thinking models; arXiv:2610.03935 Sec 6.2).
        best_tok_overall: str | None = None
        best_lp_overall = float("-inf")
        letter_logits: dict[str, float] = {}
        for tok, lp in top_logprobs.items():
            if not isinstance(tok, str):
                raise SchemaContractViolationError("top_logprobs token key must be a string")
            if isinstance(lp, bool) or not isinstance(lp, (int, float)):
                raise SchemaContractViolationError(
                    f"non-numeric logprob encountered on token {tok!r}: {lp!r}"
                )
            val = float(lp)
            if math.isnan(val) or (math.isinf(val) and val > 0.0):
                raise LogprobAlignmentError(f"invalid non-finite logprob {val!r} on token {tok!r}")
            if val > best_lp_overall:
                best_lp_overall = val
                best_tok_overall = tok.strip().lower()

        if (
            best_tok_overall in _THINKING_TAG_TOKENS
            and math.exp(min(0.0, best_lp_overall)) >= 0.50
        ):
            raise LogprobAlignmentError(
                f"thinking token {best_tok_overall!r} dominates first-token logprobs "
                f"(prob={math.exp(min(0.0, best_lp_overall)):.3f}); set enable_thinking=False "
                "or extract post-reasoning token after </think>"
            )

        for letter, key in letter_to_key.items():
            best_lp = float("-inf")
            for tok, lp in top_logprobs.items():
                val = float(lp)
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
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise SchemaContractViolationError("upstream chat completion returned empty or invalid choices")
        choice0 = choices[0]
        lp_root = choice0.get("logprobs")
        if lp_root is None:
            raise LogprobAlignmentError("upstream chat completion missing logprobs")
        if not isinstance(lp_root, Mapping):
            raise SchemaContractViolationError("upstream logprobs must be a JSON object")
        content_lp = lp_root.get("content")
        if content_lp is None or (isinstance(content_lp, list) and not content_lp):
            raise LogprobAlignmentError("upstream chat completion missing logprobs.content")
        if not isinstance(content_lp, list) or not isinstance(content_lp[0], Mapping):
            raise SchemaContractViolationError("upstream logprobs.content must be a non-empty array of objects")
        top_entries = content_lp[0].get("top_logprobs")
        if top_entries is None or (isinstance(top_entries, list) and not top_entries):
            raise LogprobAlignmentError("upstream chat completion omitted top_logprobs")
        if not isinstance(top_entries, list):
            raise SchemaContractViolationError("upstream top_logprobs must be a JSON array")
        top_map: dict[str, float] = {}
        for entry in top_entries:
            if not isinstance(entry, Mapping) or "token" not in entry or "logprob" not in entry:
                raise SchemaContractViolationError("malformed top_logprobs entry in chat completion")
            raw_tok = entry["token"]
            raw_lp = entry["logprob"]
            if not isinstance(raw_tok, str):
                raise SchemaContractViolationError("top_logprobs entry token must be a string")
            if isinstance(raw_lp, bool) or not isinstance(raw_lp, (int, float)):
                raise SchemaContractViolationError(
                    f"top_logprobs entry logprob must be numeric, got {raw_lp!r}"
                )
            lp_val = float(raw_lp)
            if math.isnan(lp_val) or (math.isinf(lp_val) and lp_val > 0.0):
                raise LogprobAlignmentError(f"invalid non-finite logprob {lp_val!r} on token {raw_tok!r}")
            if raw_tok not in top_map or lp_val > top_map[raw_tok]:
                top_map[raw_tok] = lp_val
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
        if access_token_fn is not None and not callable(access_token_fn):
            raise SchemaContractViolationError("access_token_fn must be callable when provided")
        self.access_token_fn = access_token_fn
        self.timeout_sec = _require_positive_finite_number(timeout_sec, "timeout_sec")
        self.max_retries = _require_nonneg_int(max_retries, "max_retries")
        self.base_backoff_sec = _require_nonneg_finite_number(base_backoff_sec, "base_backoff_sec")
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
        if not isinstance(data, Mapping):
            raise SchemaContractViolationError("Vertex Gemini response root must be a JSON object")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], Mapping):
            raise SchemaContractViolationError("Vertex Gemini response contained zero or invalid candidates")
        cand0 = candidates[0]
        content_obj = cand0.get("content")
        if content_obj is not None and not isinstance(content_obj, Mapping):
            raise SchemaContractViolationError("Vertex Gemini candidate content must be a JSON object")
        parts = (content_obj or {}).get("parts")
        if parts is not None and not isinstance(parts, list):
            raise SchemaContractViolationError("Vertex Gemini content.parts must be a JSON array")
        raw_text = "".join(str(p.get("text", "")) for p in (parts or []) if isinstance(p, Mapping))
        if not raw_text.strip():
            raise SchemaContractViolationError("Vertex Gemini candidate returned empty content text")
        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SchemaContractViolationError(f"Vertex Gemini JSON decode failed: {exc}") from exc
        if not isinstance(parsed_json, Mapping):
            raise SchemaContractViolationError("Vertex Gemini structured output must be a JSON object")

        evidence_read = str(parsed_json.get("evidence_read", ""))
        lp_res = cand0.get("logprobsResult")
        if lp_res is None:
            raise LogprobAlignmentError(
                "Vertex Gemini response omitted logprobsResult; synthetic probability fallbacks are prohibited"
            )
        if not isinstance(lp_res, Mapping):
            raise SchemaContractViolationError("Vertex Gemini logprobsResult must be a JSON object")
        chosen_steps = lp_res.get("chosenCandidates")
        top_steps = lp_res.get("topCandidates")
        if (
            chosen_steps is None
            or top_steps is None
            or (isinstance(chosen_steps, list) and not chosen_steps)
            or (isinstance(top_steps, list) and not top_steps)
        ):
            raise LogprobAlignmentError(
                "Vertex Gemini response omitted logprobsResult; synthetic probability fallbacks are prohibited"
            )
        if not isinstance(chosen_steps, list) or not isinstance(top_steps, list):
            raise SchemaContractViolationError(
                "Vertex Gemini chosenCandidates and topCandidates must be JSON arrays"
            )

        out: dict[str, dict[str, float]] = {}
        used_step_indices: set[int] = set()

        for qid, q in questions.items():
            letter_to_key, key_to_letter = cls.slot_letter_bijection(q)
            keys = q.option_keys()
            if qid not in parsed_json:
                raise SchemaContractViolationError(f"Vertex Gemini JSON missing required slot {qid!r}")
            raw_slot_val = parsed_json[qid]
            if raw_slot_val is None or isinstance(raw_slot_val, (bool, Mapping, list)):
                raise SchemaContractViolationError(
                    f"slot {qid!r} must be a string or scalar token, got {raw_slot_val!r}"
                )
            raw_val = str(raw_slot_val).strip()
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
                if not isinstance(chosen_tok, Mapping) or not isinstance(top_steps[step_idx], Mapping):
                    raise SchemaContractViolationError(
                        "Vertex Gemini logprob step entries must be JSON objects"
                    )
                tok_str = str(chosen_tok.get("token", "")).strip('" ')
                if tok_str not in (selected_letter, selected_key):
                    continue
                cands = top_steps[step_idx].get("candidates")
                if cands is not None and not isinstance(cands, list):
                    raise SchemaContractViolationError(
                        "Vertex Gemini topCandidates.candidates must be a JSON array"
                    )
                logits: dict[str, float] = {}
                for ltr, k in letter_to_key.items():
                    best_lp = float("-inf")
                    for c_entry in cands or []:
                        if not isinstance(c_entry, Mapping):
                            raise SchemaContractViolationError(
                                "Vertex Gemini topCandidates entry must be a JSON object"
                            )
                        ct = str(c_entry.get("token", "")).strip('" ')
                        if ct == ltr or ct == k:
                            raw_lp = c_entry.get("logProbability", float("-inf"))
                            if isinstance(raw_lp, bool) or not isinstance(raw_lp, (int, float)):
                                raise SchemaContractViolationError(
                                    f"non-numeric logProbability on slot {qid!r}: {raw_lp!r}"
                                )
                            lp = float(raw_lp)
                            if math.isnan(lp) or (math.isinf(lp) and lp > 0.0):
                                raise LogprobAlignmentError(
                                    f"invalid non-finite logProbability {lp!r} on slot {qid!r}"
                                )
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


SamplerFn = Callable[..., Mapping[str, Any] | str]
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise SchemaContractViolationError(f"non-finite JSON constant {token!r} in sampler output")


def _extract_json_dict_if_present(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    if not stripped:
        return None
    fenced = _JSON_FENCE_RE.search(stripped)
    candidate_str = fenced.group(1) if fenced else ""
    if not candidate_str and "{" in stripped and "}" in stripped:
        i0 = stripped.find("{")
        i1 = stripped.rfind("}")
        if i1 > i0:
            candidate_str = stripped[i0 : i1 + 1]
    if not candidate_str:
        return None
    try:
        parsed = json.loads(candidate_str, parse_constant=_reject_nonfinite_json_constant)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_option_token(
    raw_token: str,
    valid_keys: list[str],
    letter_to_key: Mapping[str, str],
    fallback_key: str | None = None,
) -> str | None:
    cleaned = raw_token.strip()
    if not cleaned:
        return None
    if cleaned in valid_keys:
        return cleaned
    unquoted = cleaned.strip(" \t\r\n\"'")
    if not unquoted:
        return None
    if unquoted in valid_keys:
        return unquoted
    ci_matches = [k for k in valid_keys if k.lower() == unquoted.lower()]
    if len(ci_matches) == 1:
        return ci_matches[0]
    m_opt = re.fullmatch(r"\(?\s*(?:[Oo]ption\s+)?([A-Za-z])\s*[).:]?", unquoted)
    if m_opt:
        ltr = m_opt.group(1).upper()
        if ltr in letter_to_key:
            return letter_to_key[ltr]
    m_pair = re.match(r"^\(?\s*(?:[Oo]ption\s+)?([A-Z])\s*[)=:\u2014-]\s*(.+)$", unquoted)
    if m_pair and m_pair.group(1) in letter_to_key:
        expected_key = letter_to_key[m_pair.group(1)]
        rest = m_pair.group(2).strip()
        if (
            rest == expected_key
            or rest.lower().startswith(f"{expected_key.lower()}:")
            or rest.lower().startswith(f"{expected_key.lower()} ")
            or rest.lower().startswith(f"{expected_key.lower()}(")
        ):
            return expected_key
    if fallback_key is not None and unquoted.lower() in _EXPLICIT_ABSTAIN_ALIASES:
        return fallback_key
    return None


def _resolve_choice_scalar(
    raw_val: Any,
    valid_keys: list[str],
    letter_to_key: Mapping[str, str],
    fallback_key: str | None,
    field: str,
) -> str:
    if isinstance(raw_val, bool) or raw_val is None:
        raise SchemaContractViolationError(
            f"{field} must be a valid option string, got {raw_val!r}"
        )
    if isinstance(raw_val, int):
        int_str = str(raw_val)
        if int_str in valid_keys:
            return int_str
        raise SchemaContractViolationError(
            f"{field} integer token {raw_val!r} is not in valid option keys {valid_keys!r}"
        )
    if not isinstance(raw_val, str):
        raise SchemaContractViolationError(
            f"{field} must be a valid option string, got {raw_val!r}"
        )
    matched = _resolve_option_token(raw_val, valid_keys, letter_to_key, fallback_key)
    if matched is None:
        raise SchemaContractViolationError(
            f"{field} value {raw_val!r} does not match any valid option in {valid_keys!r}"
        )
    return matched


def _validate_draw_mapping_keys_and_meta(mapping_obj: Mapping[str, Any], context: str) -> str:
    for raw_k in mapping_obj.keys():
        if not isinstance(raw_k, str) or not raw_k.strip():
            raise SchemaContractViolationError(
                f"{context} keys must be non-empty strings, got {raw_k!r}"
            )
    for meta_k in _ALLOWED_DRAW_META_KEYS:
        if meta_k in mapping_obj and mapping_obj[meta_k] is not None and not isinstance(
            mapping_obj[meta_k], str
        ):
            raise SchemaContractViolationError(
                f"{context}[{meta_k!r}] must be a string when provided, got {mapping_obj[meta_k]!r}"
            )
    ev_raw = mapping_obj.get("evidence_read") or mapping_obj.get("evidence") or ""
    return str(ev_raw).strip()


def _sampler_accepts_sample_idx(fn: SamplerFn) -> tuple[int, bool]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 3, False
    pos_count = 0
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return 4, True
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos_count += 1
    return pos_count, pos_count >= 4


class BlackBoxSampledBackend:
    """Logprob-free multi-sample and verbalized-vote DecisionBackend with Dirichlet/Jeffreys smoothing.

    Supports black-box frontier models, reasoning endpoints, and callable samplers that emit discrete
    choices or verbalized vote counts without exposing token log-probabilities. Applies a symmetric
    Dirichlet prior `alpha_0 = prior_alpha` (default 0.5 Jeffreys prior) so finite M-draw consensus
    yields positive normalized entropy, finite NLL, and analytic epistemic variance:
      p_k = (c_k + alpha_0) / A_0,  Var(theta_k | c) = p_k * (1 - p_k) / (A_0 + 1).
    """

    def __init__(
        self,
        name: str,
        sampler: SamplerFn | None = None,
        *,
        num_samples: int = 5,
        prior_alpha: float = 0.5,
        endpoint_url: str | None = None,
        model: str = "blackbox-sampled",
        api_key: str | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        max_concurrency: int = 8,
        catch_all_keys: Collection[str] = _DEFAULT_OOS_KEYS,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise SchemaContractViolationError("BlackBoxSampledBackend name must be non-empty")
        if sampler is not None and not callable(sampler):
            raise SchemaContractViolationError(
                "BlackBoxSampledBackend sampler must be callable when provided"
            )
        clean_samples = _require_positive_int(num_samples, "num_samples")
        clean_alpha = _require_positive_finite_number(prior_alpha, "prior_alpha")
        if sampler is None and not endpoint_url:
            raise SchemaContractViolationError(
                "BlackBoxSampledBackend requires either a callable `sampler` or `endpoint_url`"
            )
        if not isinstance(model, str) or not model.strip():
            raise SchemaContractViolationError(f"model must be a non-empty string, got {model!r}")
        if api_key is not None and not isinstance(api_key, str):
            raise SchemaContractViolationError(f"api_key must be a string or None, got {api_key!r}")
        if (
            not isinstance(catch_all_keys, Collection)
            or isinstance(catch_all_keys, (str, bytes, bytearray))
            or not all(isinstance(k, str) and bool(k.strip()) for k in catch_all_keys)
        ):
            raise SchemaContractViolationError(
                "catch_all_keys must be a collection of non-empty strings"
            )

        self.name = str(name).strip()
        self.sampler = sampler
        self.num_samples = clean_samples
        self.prior_alpha = clean_alpha
        self.endpoint_url: str | None = None
        if endpoint_url is not None:
            if not isinstance(endpoint_url, str):
                raise SchemaContractViolationError(
                    f"endpoint_url must be a string, got {endpoint_url!r}"
                )
            clean_url = endpoint_url.rstrip("/")
            _validate_http_url(clean_url)
            self.endpoint_url = clean_url
        self.model = model
        self.api_key = api_key
        self.timeout_sec = _require_positive_finite_number(timeout_sec, "timeout_sec")
        self.max_retries = _require_nonneg_int(max_retries, "max_retries")
        self.base_backoff_sec = _require_nonneg_finite_number(base_backoff_sec, "base_backoff_sec")
        self.max_concurrency = _require_positive_int(max_concurrency, "max_concurrency")
        default_zero = {"0"} if catch_all_keys is _DEFAULT_OOS_KEYS else set()
        self.catch_all_keys = frozenset(str(k) for k in catch_all_keys) | default_zero

        self.last_epistemic_variance: dict[str, float] = {}
        self.last_option_epistemic_variance: dict[str, dict[str, float]] = {}
        self.last_evidence: dict[str, str] = {}

    def _build_http_sample_prompt(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        bijections: Mapping[str, dict[str, str]],
    ) -> str:
        lines = [
            UNTRUSTED_BOUNDARY_NOTICE,
            "State:",
            wrap_untrusted_state(state),
            "",
            "Return a single JSON object containing concise `evidence_read` and the chosen uppercase option letter for each slot:",
        ]
        for qid, q in questions.items():
            letter_to_key = bijections[qid]
            opts = ", ".join(
                f"{ltr}={k} ({q.criteria.get(k, k) if q.type != 'score' else k})"
                for ltr, k in letter_to_key.items()
            )
            lines.append(f"- {qid}: {q.instructions} | Options: {opts}")
        return "\n".join(lines)

    def _fetch_single_http_draw(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any] | str:
        data = _post_json_with_retry(
            url,
            payload,
            headers,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            base_backoff_sec=self.base_backoff_sec,
        )
        if isinstance(data.get("choices"), list) and data["choices"]:
            c0 = data["choices"][0]
            if isinstance(c0, Mapping):
                msg = c0.get("message")
                if isinstance(msg, Mapping) and msg.get("content") is not None:
                    content_val = msg["content"]
                    if isinstance(content_val, (str, Mapping)):
                        return content_val
                    raise SchemaContractViolationError(
                        f"upstream message.content must be a string or object, got {content_val!r}"
                    )
                if c0.get("text") is not None:
                    text_val = c0["text"]
                    if isinstance(text_val, str):
                        return text_val
                    raise SchemaContractViolationError(
                        f"upstream choice.text must be a string, got {text_val!r}"
                    )
        if isinstance(data.get("candidates"), list) and data["candidates"]:
            cand0 = data["candidates"][0]
            if isinstance(cand0, Mapping):
                content_obj = cand0.get("content")
                if isinstance(content_obj, Mapping):
                    parts = content_obj.get("parts")
                    if isinstance(parts, list):
                        txt = "".join(str(p.get("text", "")) for p in parts if isinstance(p, Mapping))
                        if txt:
                            return txt
        return data

    def _invoke_callable_sampler(
        self,
        pos_arity: int,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None,
        sample_idx: int,
    ) -> Mapping[str, Any] | str:
        assert self.sampler is not None
        if pos_arity >= 4:
            return self.sampler(state, questions, images, sample_idx)
        if pos_arity == 3:
            return self.sampler(state, questions, images)
        if pos_arity == 2:
            return self.sampler(state, questions)
        if pos_arity == 1:
            return self.sampler(state)
        return self.sampler()

    def _accumulate_votes_map(
        self,
        votes_raw: Mapping[str, Any],
        counts_q: dict[str, float],
        valid_keys: list[str],
        letter_to_key: Mapping[str, str],
        fallback_key: str | None,
    ) -> None:
        if not isinstance(votes_raw, Mapping) or not votes_raw:
            raise SchemaContractViolationError(
                "verbalized votes map must be a non-empty JSON object"
            )
        added_weight = 0.0
        for raw_opt, raw_w in votes_raw.items():
            if not isinstance(raw_opt, str) or not raw_opt.strip():
                raise SchemaContractViolationError(
                    f"verbalized vote option key must be a non-empty string, got {raw_opt!r}"
                )
            weight = _require_nonneg_finite_number(
                raw_w, f"verbalized vote count for {raw_opt!r}"
            )
            matched = _resolve_option_token(raw_opt, valid_keys, letter_to_key, fallback_key)
            if matched is None:
                raise SchemaContractViolationError(
                    f"unrecognized option token {raw_opt!r} in verbalized votes (valid={valid_keys!r})"
                )
            if weight > 0.0:
                counts_q[matched] += weight
                added_weight += weight
        if not math.isfinite(added_weight) or added_weight <= 0.0:
            raise SchemaContractViolationError(
                "verbalized votes map must have strictly positive finite total vote mass"
            )

    def _accumulate_slot_entry(
        self,
        slot_entry: Any,
        qid: str,
        counts_q: dict[str, float],
        evidences_q: list[str],
        valid_keys: list[str],
        letter_to_key: Mapping[str, str],
        fallback_key: str | None,
    ) -> bool:
        if isinstance(slot_entry, Mapping):
            slot_ev = _validate_draw_mapping_keys_and_meta(slot_entry, f"slot {qid!r}")
            if slot_ev:
                evidences_q.append(slot_ev)
            slot_payload_keys = [k for k in slot_entry.keys() if k not in _ALLOWED_DRAW_META_KEYS]
            if not slot_payload_keys:
                raise SchemaContractViolationError(
                    f"slot {qid!r} object contains no choice or votes field"
                )
            has_votes_key = "votes" in slot_entry
            has_counts_key = "counts" in slot_entry
            if has_votes_key and has_counts_key:
                raise SchemaContractViolationError(
                    f"slot {qid!r} object cannot specify both 'votes' and 'counts'"
                )
            if has_votes_key or has_counts_key:
                sv_field = "votes" if has_votes_key else "counts"
                if slot_payload_keys != [sv_field]:
                    raise SchemaContractViolationError(
                        f"slot {qid!r} object has unexpected keys alongside {sv_field!r}: {slot_payload_keys!r}"
                    )
                sub_votes = slot_entry[sv_field]
                if not isinstance(sub_votes, Mapping):
                    raise SchemaContractViolationError(
                        f"slot {qid!r} {sv_field!r} field must be a JSON object, got {sub_votes!r}"
                    )
                self._accumulate_votes_map(
                    sub_votes, counts_q, valid_keys, letter_to_key, fallback_key
                )
                return True

            present_aliases = [a for a in _CHOICE_ALIAS_KEYS if a in slot_entry]
            if present_aliases:
                if len(present_aliases) > 1 or slot_payload_keys != [present_aliases[0]]:
                    raise SchemaContractViolationError(
                        f"slot {qid!r} object has ambiguous or unrecognized keys: {slot_payload_keys!r}"
                    )
                chosen = _resolve_choice_scalar(
                    slot_entry[present_aliases[0]],
                    valid_keys,
                    letter_to_key,
                    fallback_key,
                    f"slot {qid!r}.{present_aliases[0]}",
                )
                counts_q[chosen] += 1.0
                return False

            flat_sub_votes = {k: slot_entry[k] for k in slot_payload_keys}
            self._accumulate_votes_map(
                flat_sub_votes, counts_q, valid_keys, letter_to_key, fallback_key
            )
            return True

        chosen_scalar = _resolve_choice_scalar(
            slot_entry,
            valid_keys,
            letter_to_key,
            fallback_key,
            f"slot {qid!r}",
        )
        counts_q[chosen_scalar] += 1.0
        return False

    def _accumulate_draw(
        self,
        draw: Mapping[str, Any] | str,
        questions: Mapping[str, DecisionQuestion],
        valid_keys_map: Mapping[str, list[str]],
        bijections: Mapping[str, dict[str, str]],
        fallback_keys: Mapping[str, str | None],
        counts: dict[str, dict[str, float]],
        evidences: dict[str, list[str]],
    ) -> bool:
        before_totals = {qid: sum(counts[qid].values()) for qid in questions}
        saw_explicit_votes = False
        parsed_map: Mapping[str, Any] | None = None

        if isinstance(draw, str):
            extracted = _extract_json_dict_if_present(draw)
            if extracted is not None:
                parsed_map = extracted
            else:
                for qid in questions:
                    matched = _resolve_option_token(
                        draw,
                        valid_keys_map[qid],
                        bijections[qid],
                        fallback_keys[qid],
                    )
                    if matched is None:
                        raise SchemaContractViolationError(
                            f"unparseable sampler token {draw!r} for slot {qid!r} (valid={valid_keys_map[qid]!r})"
                        )
                    counts[qid][matched] += 1.0
                return False
        elif isinstance(draw, Mapping):
            parsed_map = draw
        else:
            raise SchemaContractViolationError(
                f"unsupported sampler draw type: {type(draw).__name__}"
            )

        assert parsed_map is not None
        if not parsed_map:
            raise SchemaContractViolationError(
                "BlackBoxSampledBackend received empty JSON object draw"
            )
        top_ev = _validate_draw_mapping_keys_and_meta(parsed_map, "sampler draw")
        has_top_votes = "votes" in parsed_map
        has_top_counts = "counts" in parsed_map
        if has_top_votes and has_top_counts:
            raise SchemaContractViolationError(
                "sampler draw cannot specify both 'votes' and 'counts'"
            )

        if has_top_votes or has_top_counts:
            v_field = "votes" if has_top_votes else "counts"
            extra_top = [
                k for k in parsed_map.keys() if k != v_field and k not in _ALLOWED_DRAW_META_KEYS
            ]
            if extra_top:
                raise SchemaContractViolationError(
                    f"sampler draw has unexpected top-level keys alongside {v_field!r}: {extra_top!r}"
                )
            raw_top_votes = parsed_map[v_field]
            if not isinstance(raw_top_votes, Mapping) or not raw_top_votes:
                raise SchemaContractViolationError(
                    f"BlackBoxSampledBackend {v_field!r} field must be a non-empty JSON object, got {raw_top_votes!r}"
                )
            if any(isinstance(v, Mapping) for v in raw_top_votes.values()):
                if not all(isinstance(v, Mapping) for v in raw_top_votes.values()):
                    raise SchemaContractViolationError(
                        f"BlackBoxSampledBackend {v_field!r} mixes nested slot objects and scalar values"
                    )
                if set(raw_top_votes.keys()) != set(questions.keys()):
                    raise SchemaContractViolationError(
                        f"BlackBoxSampledBackend {v_field!r} slot keys {list(raw_top_votes.keys())!r} "
                        f"do not match expected slots {list(questions.keys())!r}"
                    )
                for qid in questions:
                    q_votes = raw_top_votes[qid]
                    assert isinstance(q_votes, Mapping)
                    self._accumulate_votes_map(
                        q_votes,
                        counts[qid],
                        valid_keys_map[qid],
                        bijections[qid],
                        fallback_keys[qid],
                    )
                    if top_ev:
                        evidences[qid].append(top_ev)
                saw_explicit_votes = True
            elif len(questions) == 1:
                only_qid = next(iter(questions.keys()))
                self._accumulate_votes_map(
                    raw_top_votes,
                    counts[only_qid],
                    valid_keys_map[only_qid],
                    bijections[only_qid],
                    fallback_keys[only_qid],
                )
                if top_ev:
                    evidences[only_qid].append(top_ev)
                saw_explicit_votes = True
            else:
                raise SchemaContractViolationError(
                    f"multi-slot {v_field!r} must map each slot_id in {list(questions.keys())!r} to an option-weight object"
                )
        elif "answers" in parsed_map:
            extra_ans = [
                k for k in parsed_map.keys() if k != "answers" and k not in _ALLOWED_DRAW_META_KEYS
            ]
            if extra_ans:
                raise SchemaContractViolationError(
                    f"sampler draw has unexpected top-level keys alongside 'answers': {extra_ans!r}"
                )
            answers_obj = parsed_map["answers"]
            if not isinstance(answers_obj, Mapping) or not answers_obj:
                raise SchemaContractViolationError(
                    "BlackBoxSampledBackend 'answers' field must be a non-empty JSON object"
                )
            ans_ev = _validate_draw_mapping_keys_and_meta(answers_obj, "answers")
            ans_payload_keys = [k for k in answers_obj.keys() if k not in _ALLOWED_DRAW_META_KEYS]
            if set(ans_payload_keys) != set(questions.keys()):
                raise SchemaContractViolationError(
                    f"BlackBoxSampledBackend 'answers' keys {ans_payload_keys!r} "
                    f"do not match expected slots {list(questions.keys())!r}"
                )
            for qid in questions:
                if top_ev:
                    evidences[qid].append(top_ev)
                if ans_ev:
                    evidences[qid].append(ans_ev)
                if self._accumulate_slot_entry(
                    answers_obj[qid],
                    qid,
                    counts[qid],
                    evidences[qid],
                    valid_keys_map[qid],
                    bijections[qid],
                    fallback_keys[qid],
                ):
                    saw_explicit_votes = True
        else:
            payload_keys = [k for k in parsed_map.keys() if k not in _ALLOWED_DRAW_META_KEYS]
            if not payload_keys:
                raise SchemaContractViolationError(
                    "sampler draw object contains no slot answers, choice alias, or votes"
                )
            if set(payload_keys) == set(questions.keys()):
                for qid in questions:
                    if top_ev:
                        evidences[qid].append(top_ev)
                    if self._accumulate_slot_entry(
                        parsed_map[qid],
                        qid,
                        counts[qid],
                        evidences[qid],
                        valid_keys_map[qid],
                        bijections[qid],
                        fallback_keys[qid],
                    ):
                        saw_explicit_votes = True
            elif len(questions) == 1:
                only_qid = next(iter(questions.keys()))
                if top_ev:
                    evidences[only_qid].append(top_ev)
                present_top_aliases = [a for a in _CHOICE_ALIAS_KEYS if a in parsed_map]
                if present_top_aliases:
                    if len(present_top_aliases) > 1 or payload_keys != [present_top_aliases[0]]:
                        raise SchemaContractViolationError(
                            f"sampler draw has ambiguous or unrecognized keys: {payload_keys!r}"
                        )
                    chosen_top = _resolve_choice_scalar(
                        parsed_map[present_top_aliases[0]],
                        valid_keys_map[only_qid],
                        bijections[only_qid],
                        fallback_keys[only_qid],
                        f"draw.{present_top_aliases[0]}",
                    )
                    counts[only_qid][chosen_top] += 1.0
                else:
                    flat_votes = {k: parsed_map[k] for k in payload_keys}
                    self._accumulate_votes_map(
                        flat_votes,
                        counts[only_qid],
                        valid_keys_map[only_qid],
                        bijections[only_qid],
                        fallback_keys[only_qid],
                    )
                    saw_explicit_votes = True
            else:
                raise SchemaContractViolationError(
                    f"multi-slot sampler draw keys {payload_keys!r} do not match expected slots {list(questions.keys())!r}"
                )

        for qid in questions:
            delta = sum(counts[qid].values()) - before_totals[qid]
            if not math.isfinite(delta) or delta <= 0.0:
                raise SchemaContractViolationError(
                    f"sampler draw failed to accumulate positive vote mass for slot {qid!r}"
                )
        return saw_explicit_votes

    def score_questions(
        self,
        state: Any,
        questions: Mapping[str, DecisionQuestion],
        images: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        if not questions:
            self.last_epistemic_variance = {}
            self.last_option_epistemic_variance = {}
            self.last_evidence = {}
            return {}

        valid_keys_map: dict[str, list[str]] = {}
        bijections: dict[str, dict[str, str]] = {}
        fallback_keys: dict[str, str | None] = {}
        counts: dict[str, dict[str, float]] = {}
        evidences: dict[str, list[str]] = {}

        for qid, q in questions.items():
            keys = q.option_keys()
            if len(keys) < 2 or len(keys) > MAX_OPTIONS_PER_SLOT:
                raise SchemaContractViolationError(
                    f"BlackBoxSampledBackend requires 2..{MAX_OPTIONS_PER_SLOT} options for {qid!r}, got {len(keys)}"
                )
            valid_keys_map[qid] = keys
            letters = string.ascii_uppercase[: len(keys)]
            bijections[qid] = {letters[i]: keys[i] for i in range(len(keys))}
            fb: str | None = None
            for k in keys:
                if k in self.catch_all_keys:
                    fb = k
                    break
            fallback_keys[qid] = fb
            counts[qid] = {k: 0.0 for k in keys}
            evidences[qid] = []

        if self.sampler is not None:
            pos_arity, has_sample_idx = _sampler_accepts_sample_idx(self.sampler)
            for s_idx in range(self.num_samples):
                draw = self._invoke_callable_sampler(pos_arity, state, questions, images, s_idx)
                explicit_votes = self._accumulate_draw(
                    draw,
                    questions,
                    valid_keys_map,
                    bijections,
                    fallback_keys,
                    counts,
                    evidences,
                )
                if explicit_votes and not has_sample_idx:
                    break
        else:
            assert self.endpoint_url is not None
            url = (
                self.endpoint_url
                if self.endpoint_url.endswith("/v1/chat/completions")
                else f"{self.endpoint_url}/v1/chat/completions"
            )
            _validate_http_url(url)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            prompt_text = self._build_http_sample_prompt(state, questions, bijections)
            user_content: Any = prompt_text
            if images:
                parts_list: list[dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": img.strip()}}
                    for img in images
                    if img.strip()
                ]
                parts_list.append({"type": "text", "text": prompt_text})
                user_content = parts_list
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.7 if self.num_samples > 1 else 0.0,
            }

            if self.num_samples == 1 or self.max_concurrency == 1:
                draws = [self._fetch_single_http_draw(url, headers, payload) for _ in range(self.num_samples)]
            else:
                workers = min(self.max_concurrency, self.num_samples)
                with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = [
                        pool.submit(self._fetch_single_http_draw, url, headers, payload)
                        for _ in range(self.num_samples)
                    ]
                    draws = [f.result() for f in futs]

            for draw in draws:
                self._accumulate_draw(
                    draw,
                    questions,
                    valid_keys_map,
                    bijections,
                    fallback_keys,
                    counts,
                    evidences,
                )

        out_probs: dict[str, dict[str, float]] = {}
        epistemic_tot: dict[str, float] = {}
        epistemic_opt: dict[str, dict[str, float]] = {}
        evidence_out: dict[str, str] = {}

        for qid, keys in valid_keys_map.items():
            alpha_post = {k: counts[qid][k] + self.prior_alpha for k in keys}
            a0 = sum(alpha_post.values())
            probs_q = {k: alpha_post[k] / a0 for k in keys}
            var_q = {k: (probs_q[k] * (1.0 - probs_q[k])) / (a0 + 1.0) for k in keys}
            out_probs[qid] = probs_q
            epistemic_opt[qid] = var_q
            epistemic_tot[qid] = sum(var_q.values())
            if evidences[qid]:
                evidence_out[qid] = evidences[qid][0]

        self.last_epistemic_variance = epistemic_tot
        self.last_option_epistemic_variance = epistemic_opt
        self.last_evidence = evidence_out
        return out_probs


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
        conformal_q_hat: float | Mapping[str, float] | None = None,
        conformal_method: str = "raps",
        conformal_lam_reg: float = 0.01,
        conformal_k_reg: int = 1,
        escalate_on_conformal_ambiguous: bool = True,
        layer5_slot_context: Mapping[str, Mapping[str, Any]] | None = None,
        conformal_randomized: bool = False,
        conformal_seed: int | None = None,
        conformal_rng: random.Random | None = None,
    ) -> None:
        if stage1_backend is None or not callable(getattr(stage1_backend, "score_questions", None)):
            raise SchemaContractViolationError(
                "stage1_backend must implement score_questions(state, questions, images)"
            )
        if stage2_backend is not None and not callable(
            getattr(stage2_backend, "score_questions", None)
        ):
            raise SchemaContractViolationError(
                "stage2_backend must implement score_questions(state, questions, images) when provided"
            )
        for flag_name, flag_val in (
            ("shortlist_pairwise_diff", shortlist_pairwise_diff),
            ("enable_stage2a_fast_exit", enable_stage2a_fast_exit),
            ("escalate_on_inferred_evidence", escalate_on_inferred_evidence),
            ("total_contract_mode", total_contract_mode),
            ("escalate_on_conformal_ambiguous", escalate_on_conformal_ambiguous),
            ("conformal_randomized", conformal_randomized),
        ):
            if not isinstance(flag_val, bool):
                raise SchemaContractViolationError(
                    f"{flag_name} must be bool, got {flag_val!r}"
                )
        self.stage1_backend = stage1_backend
        self.stage2_backend = stage2_backend
        self.hesitation_threshold = _require_nonneg_finite_number(
            hesitation_threshold, "hesitation_threshold"
        )
        self.prior_weight = _require_nonneg_finite_number(prior_weight, "prior_weight")
        self.prior_gate_mode = prior_gate_mode
        self.temperature = _require_positive_finite_number(temperature, "temperature")
        self.shortlist_top_k = _require_nonneg_int(shortlist_top_k, "shortlist_top_k")
        self.shortlist_pairwise_diff = shortlist_pairwise_diff
        self.evidence_prompt = evidence_prompt
        self.evidence_criteria: dict[str, str] | Any = (
            {str(k): str(v) for k, v in evidence_criteria.items()}
            if isinstance(evidence_criteria, Mapping)
            else evidence_criteria
        )
        self.enable_stage2a_fast_exit = enable_stage2a_fast_exit
        self.escalate_on_inferred_evidence = escalate_on_inferred_evidence
        self.total_contract_mode = total_contract_mode
        self.sla_timeout_sec = _require_positive_finite_number(
            sla_timeout_sec, "sla_timeout_sec"
        )
        self.response_cache = response_cache
        self.oos_keys: frozenset[str] | Any = (
            frozenset(str(k) for k in oos_keys)
            if isinstance(oos_keys, Collection) and not isinstance(oos_keys, (str, bytes, bytearray))
            else oos_keys
        )
        self.conformal_q_hat: float | dict[str, float] | Any = (
            dict(conformal_q_hat) if isinstance(conformal_q_hat, Mapping) else conformal_q_hat
        )
        self.conformal_q_hat = self._validated_conformal_q_hat()
        self.conformal_method = conformal_method
        self.conformal_lam_reg = _require_nonneg_finite_number(
            conformal_lam_reg, "conformal_lam_reg"
        )
        self.conformal_k_reg = _require_nonneg_int(conformal_k_reg, "conformal_k_reg")
        self.escalate_on_conformal_ambiguous = escalate_on_conformal_ambiguous
        self.conformal_randomized = conformal_randomized
        self.conformal_seed = (
            _require_strict_int(conformal_seed, "conformal_seed")
            if conformal_seed is not None
            else None
        )
        self.conformal_rng = conformal_rng
        self.layer5_slot_context: dict[str, dict[str, Any]] | Any = (
            {
                str(qid): (dict(ctx) if isinstance(ctx, Mapping) else ctx)
                for qid, ctx in layer5_slot_context.items()
            }
            if isinstance(layer5_slot_context, Mapping)
            else layer5_slot_context
        )

    @staticmethod
    def _validate_conformal_q_hat_value(raw_q: Any, field: str) -> float:
        if isinstance(raw_q, bool) or not isinstance(raw_q, (int, float)):
            raise SchemaContractViolationError(
                f"{field} must be a non-boolean number >= 0, got {raw_q!r}"
            )
        val = float(raw_q)
        if math.isnan(val) or val < 0.0:
            raise SchemaContractViolationError(f"{field} must be >= 0 and not NaN, got {raw_q!r}")
        return 0.0 if val == 0.0 else val

    def _validated_conformal_q_hat(self) -> float | dict[str, float] | None:
        if self.conformal_q_hat is None:
            return None
        if isinstance(self.conformal_q_hat, Mapping):
            out: dict[str, float] = {}
            for raw_k, raw_v in self.conformal_q_hat.items():
                if not isinstance(raw_k, str) or not raw_k:
                    raise SchemaContractViolationError(
                        "conformal_q_hat mapping keys must be non-empty strings"
                    )
                out[raw_k] = self._validate_conformal_q_hat_value(
                    raw_v, f"conformal_q_hat[{raw_k!r}]"
                )
            return out
        return self._validate_conformal_q_hat_value(self.conformal_q_hat, "conformal_q_hat")

    def _resolve_slot_q_hat(self, qid: str) -> float | None:
        clean_q = self._validated_conformal_q_hat()
        if clean_q is None:
            return None
        if isinstance(clean_q, Mapping):
            return clean_q.get(qid)
        return clean_q

    @staticmethod
    def _validate_nested_float_map(
        raw: Mapping[str, Mapping[str, float]] | None,
        field: str,
    ) -> dict[str, dict[str, float]] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise SchemaContractViolationError(f"{field} must be a mapping")
        out: dict[str, dict[str, float]] = {}
        for qid, inner in raw.items():
            if not isinstance(qid, str) or not qid:
                raise SchemaContractViolationError(f"{field} slot keys must be non-empty strings")
            if not isinstance(inner, Mapping):
                raise SchemaContractViolationError(f"{field}[{qid!r}] must be a mapping")
            slot_out: dict[str, float] = {}
            for opt_k, opt_v in inner.items():
                if not isinstance(opt_k, str) or not opt_k:
                    raise SchemaContractViolationError(
                        f"{field}[{qid!r}] option keys must be non-empty strings"
                    )
                slot_out[opt_k] = _require_finite_number(opt_v, f"{field}[{qid!r}][{opt_k!r}]")
            out[qid] = slot_out
        return out

    @staticmethod
    def _validate_pack_map(
        raw: Mapping[str, int] | None,
        field: str,
    ) -> dict[str, int] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise SchemaContractViolationError(f"{field} must be a mapping")
        out: dict[str, int] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k:
                raise SchemaContractViolationError(f"{field} keys must be non-empty strings")
            out[k] = _require_strict_int(v, f"{field}[{k!r}]")
        return out

    @staticmethod
    def _validate_flat_float_map(
        raw: Mapping[str, float] | None,
        field: str,
    ) -> dict[str, float] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise SchemaContractViolationError(f"{field} must be a mapping")
        out: dict[str, float] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k:
                raise SchemaContractViolationError(f"{field} keys must be non-empty strings")
            out[k] = _require_finite_number(v, f"{field}[{k!r}]")
        return out

    @classmethod
    def _validate_slot_context_map(
        cls,
        raw: Mapping[str, Mapping[str, Any]] | None,
        field: str,
    ) -> dict[str, dict[str, Any]] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise SchemaContractViolationError(f"{field} must be a mapping")
        out: dict[str, dict[str, Any]] = {}
        for qid, ctx in raw.items():
            if not isinstance(qid, str) or not qid:
                raise SchemaContractViolationError(f"{field} slot keys must be non-empty strings")
            if not isinstance(ctx, Mapping):
                raise SchemaContractViolationError(f"{field}[{qid!r}] must be a mapping")
            clean_ctx: dict[str, Any] = dict(ctx)
            if "candidate_pack_sizes" in ctx and ctx["candidate_pack_sizes"] is not None:
                clean_ctx["candidate_pack_sizes"] = cls._validate_pack_map(
                    ctx["candidate_pack_sizes"], f"{field}[{qid!r}].candidate_pack_sizes"
                )
            if "decoded_pack_size" in ctx and ctx["decoded_pack_size"] is not None:
                clean_ctx["decoded_pack_size"] = _require_strict_int(
                    ctx["decoded_pack_size"], f"{field}[{qid!r}].decoded_pack_size"
                )
            if "visual_energies" in ctx and ctx["visual_energies"] is not None:
                clean_ctx["visual_energies"] = cls._validate_flat_float_map(
                    ctx["visual_energies"], f"{field}[{qid!r}].visual_energies"
                )
            if "rank1_id" in ctx and ctx["rank1_id"] is not None:
                if not isinstance(ctx["rank1_id"], str):
                    raise SchemaContractViolationError(
                        f"{field}[{qid!r}].rank1_id must be a string"
                    )
                clean_ctx["rank1_id"] = ctx["rank1_id"]
            if "rank1_visual_similarity" in ctx and ctx["rank1_visual_similarity"] is not None:
                clean_ctx["rank1_visual_similarity"] = _require_finite_number(
                    ctx["rank1_visual_similarity"], f"{field}[{qid!r}].rank1_visual_similarity"
                )
            out[qid] = clean_ctx
        return out

    def _resolve_slot_layer5_args(
        self,
        qid: str,
        *,
        slot_context_override: Mapping[str, Mapping[str, Any]] | None,
        candidate_pack_sizes: Mapping[str, int] | None,
        decoded_pack_size: int | None,
        visual_energies: Mapping[str, float] | None,
        rank1_id: str | None,
        rank1_visual_similarity: float | None,
    ) -> tuple[
        Mapping[str, int] | None,
        int | None,
        Mapping[str, float] | None,
        str | None,
        float | None,
    ]:
        ctx: Mapping[str, Any] = {}
        if slot_context_override is not None and qid in slot_context_override:
            ctx = slot_context_override[qid]
        elif isinstance(self.layer5_slot_context, Mapping) and qid in self.layer5_slot_context:
            ctx = self.layer5_slot_context[qid]
        if not ctx:
            return (
                candidate_pack_sizes,
                decoded_pack_size,
                visual_energies,
                rank1_id,
                rank1_visual_similarity,
            )
        c_packs = ctx.get("candidate_pack_sizes", candidate_pack_sizes)
        d_pack = ctx.get("decoded_pack_size", decoded_pack_size)
        v_en = ctx.get("visual_energies", visual_energies)
        r1_id = ctx.get("rank1_id", rank1_id)
        r1_sim = ctx.get("rank1_visual_similarity", rank1_visual_similarity)
        if r1_id is not None and not isinstance(r1_id, str):
            raise SchemaContractViolationError(f"layer5_slot_context[{qid!r}].rank1_id must be str")
        return (
            self._validate_pack_map(c_packs, f"layer5_slot_context[{qid!r}].candidate_pack_sizes"),
            _require_strict_int(d_pack, f"layer5_slot_context[{qid!r}].decoded_pack_size")
            if d_pack is not None
            else None,
            self._validate_flat_float_map(v_en, f"layer5_slot_context[{qid!r}].visual_energies"),
            r1_id,
            _require_finite_number(r1_sim, f"layer5_slot_context[{qid!r}].rank1_visual_similarity")
            if r1_sim is not None
            else None,
        )

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
        safe_qids = (
            [str(qid) for qid in questions.keys()] if isinstance(questions, Mapping) else []
        )
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
            for qid in safe_qids
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
        layer5_slot_context: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        cache_key = ""
        try:
            clean_hes = _require_finite_number(self.hesitation_threshold, "hesitation_threshold")
            clean_pw = _require_nonneg_finite_number(self.prior_weight, "prior_weight")
            clean_temp = _require_finite_number(self.temperature, "temperature")
            if clean_temp <= 0.0:
                raise SchemaContractViolationError(f"temperature must be > 0, got {self.temperature!r}")
            clean_top_k = _require_strict_int(self.shortlist_top_k, "shortlist_top_k")
            clean_sla = _require_finite_number(self.sla_timeout_sec, "sla_timeout_sec")
            if clean_sla <= 0.0:
                raise SchemaContractViolationError(
                    f"sla_timeout_sec must be > 0, got {self.sla_timeout_sec!r}"
                )
            if not isinstance(sister_pack_collision, bool):
                raise SchemaContractViolationError(
                    f"sister_pack_collision must be bool, got {sister_pack_collision!r}"
                )
            if not isinstance(self.oos_keys, frozenset):
                raise SchemaContractViolationError("oos_keys must be a collection of strings")
            clean_dec_pack = (
                _require_strict_int(decoded_pack_size, "decoded_pack_size")
                if decoded_pack_size is not None
                else None
            )
            if rank1_id is not None and not isinstance(rank1_id, str):
                raise SchemaContractViolationError(
                    f"rank1_id must be a string when provided, got {rank1_id!r}"
                )
            clean_r1_sim = (
                _require_finite_number(rank1_visual_similarity, "rank1_visual_similarity")
                if rank1_visual_similarity is not None
                else None
            )
            clean_prior_logits = self._validate_nested_float_map(prior_logits, "prior_logits")
            clean_shortlist_scores = self._validate_nested_float_map(
                shortlist_scores, "shortlist_scores"
            )
            clean_pack_sizes = self._validate_pack_map(candidate_pack_sizes, "candidate_pack_sizes")
            clean_visual_energies = self._validate_flat_float_map(visual_energies, "visual_energies")
            clean_init_slot_ctx = self._validate_slot_context_map(
                self.layer5_slot_context, "layer5_slot_context"
            )
            clean_call_slot_ctx = self._validate_slot_context_map(
                layer5_slot_context, "call_layer5_slot_context"
            )
            if not isinstance(self.prior_gate_mode, str) or self.prior_gate_mode not in VALID_PRIOR_GATE_MODES:
                raise SchemaContractViolationError(
                    f"prior_gate_mode must be one of {VALID_PRIOR_GATE_MODES}, got {self.prior_gate_mode!r}"
                )
            if not isinstance(self.enable_stage2a_fast_exit, bool):
                raise SchemaContractViolationError(
                    f"enable_stage2a_fast_exit must be bool, got {self.enable_stage2a_fast_exit!r}"
                )
            if not isinstance(self.shortlist_pairwise_diff, bool):
                raise SchemaContractViolationError(
                    f"shortlist_pairwise_diff must be bool, got {self.shortlist_pairwise_diff!r}"
                )
            if not isinstance(self.escalate_on_inferred_evidence, bool):
                raise SchemaContractViolationError(
                    f"escalate_on_inferred_evidence must be bool, got {self.escalate_on_inferred_evidence!r}"
                )
            if not isinstance(self.escalate_on_conformal_ambiguous, bool):
                raise SchemaContractViolationError(
                    f"escalate_on_conformal_ambiguous must be bool, got {self.escalate_on_conformal_ambiguous!r}"
                )
            if not isinstance(self.evidence_prompt, str):
                raise SchemaContractViolationError(
                    f"evidence_prompt must be str, got {self.evidence_prompt!r}"
                )
            if self.evidence_criteria is not None and not isinstance(self.evidence_criteria, Mapping):
                raise SchemaContractViolationError("evidence_criteria must be a mapping when provided")
            clean_q_hat = self._validated_conformal_q_hat()
            if not isinstance(self.conformal_randomized, bool):
                raise SchemaContractViolationError(
                    f"conformal_randomized must be bool, got {self.conformal_randomized!r}"
                )
            if self.conformal_seed is not None:
                _require_strict_int(self.conformal_seed, "conformal_seed")

            s1_name = getattr(self.stage1_backend, "name", None)
            if not isinstance(s1_name, str) or not s1_name.strip():
                raise SchemaContractViolationError("stage1_backend.name must be a non-empty string")
            if self.stage2_backend is not None:
                s2_raw_name = getattr(self.stage2_backend, "name", None)
                if not isinstance(s2_raw_name, str) or not s2_raw_name.strip():
                    raise SchemaContractViolationError("stage2_backend.name must be a non-empty string")
                s2_name = s2_raw_name
            else:
                s2_name = "none"
            extra_cfg: dict[str, Any] = {
                "hesitation_threshold": clean_hes,
                "prior_weight": clean_pw,
                "prior_gate_mode": self.prior_gate_mode,
                "temperature": clean_temp,
                "shortlist_top_k": clean_top_k,
                "shortlist_pairwise_diff": self.shortlist_pairwise_diff,
                "evidence_prompt": self.evidence_prompt,
                "enable_stage2a_fast_exit": self.enable_stage2a_fast_exit,
                "sister_pack_collision": sister_pack_collision,
                "decoded_pack_size": clean_dec_pack,
            }
            if (
                (self.stage2_backend is not None and s2_name == "none")
                or "->" in s1_name
                or (self.stage2_backend is not None and "->" in s2_name)
            ):
                extra_cfg["backend_pair"] = [
                    s1_name,
                    s2_name if self.stage2_backend is not None else None,
                ]
            if self.oos_keys != _DEFAULT_OOS_KEYS:
                extra_cfg["oos_keys"] = sorted(self.oos_keys)
            if self.evidence_criteria is not None:
                extra_cfg["evidence_criteria"] = dict(self.evidence_criteria)
            if self.escalate_on_inferred_evidence:
                extra_cfg["escalate_on_inferred_evidence"] = True
            if clean_init_slot_ctx is not None and clean_call_slot_ctx is None:
                extra_cfg["layer5_slot_context"] = clean_init_slot_ctx
            elif clean_init_slot_ctx is None and clean_call_slot_ctx is not None:
                extra_cfg["call_layer5_slot_context"] = clean_call_slot_ctx
            elif clean_init_slot_ctx is not None and clean_call_slot_ctx is not None:
                extra_cfg["layer5_slot_context"] = clean_init_slot_ctx
                extra_cfg["call_layer5_slot_context"] = clean_call_slot_ctx

            if clean_prior_logits is not None:
                extra_cfg["prior_logits"] = clean_prior_logits
            if clean_shortlist_scores is not None:
                extra_cfg["shortlist_scores"] = clean_shortlist_scores
            if clean_pack_sizes is not None:
                extra_cfg["candidate_pack_sizes"] = clean_pack_sizes
            if clean_visual_energies is not None:
                extra_cfg["visual_energies"] = clean_visual_energies
            if rank1_id is not None:
                extra_cfg["rank1_id"] = rank1_id
            if clean_r1_sim is not None:
                extra_cfg["rank1_visual_similarity"] = clean_r1_sim

            if clean_q_hat is not None:
                if not isinstance(self.conformal_method, str) or self.conformal_method not in (
                    "lac",
                    "aps",
                    "raps",
                ):
                    raise SchemaContractViolationError(
                        f"conformal_method must be 'lac', 'aps', or 'raps', got {self.conformal_method!r}"
                    )
                clean_lam = _require_nonneg_finite_number(self.conformal_lam_reg, "conformal_lam_reg")
                clean_k = _require_strict_int(self.conformal_k_reg, "conformal_k_reg")
                if clean_k < 0:
                    raise SchemaContractViolationError(
                        f"conformal_k_reg must be >= 0, got {self.conformal_k_reg!r}"
                    )
                extra_cfg["conformal_q_hat"] = clean_q_hat
                extra_cfg["conformal_method"] = self.conformal_method
                extra_cfg["conformal_lam_reg"] = clean_lam
                extra_cfg["conformal_k_reg"] = clean_k
                extra_cfg["escalate_on_conformal_ambiguous"] = bool(
                    self.escalate_on_conformal_ambiguous
                )
                if self.conformal_randomized or self.conformal_seed is not None:
                    extra_cfg["conformal_randomized"] = self.conformal_randomized
                    extra_cfg["conformal_seed"] = self.conformal_seed

            cache_key = compute_decision_cache_key(
                f"{s1_name}->{s2_name}",
                state,
                questions,
                images,
                extra_config=extra_cfg,
            )
            can_use_cache = self.response_cache is not None and not (
                clean_q_hat is not None and self.conformal_randomized
            )
            if can_use_cache and self.response_cache is not None:
                cached = self.response_cache.get(cache_key)
                if cached is not None:
                    cached.latency_ms = (time.perf_counter() - t0) * 1000.0
                    return cached

            return self._evaluate_inner(
                t0=t0,
                cache_key=cache_key,
                can_use_cache=can_use_cache,
                state=state,
                questions=questions,
                images=images,
                prior_logits=clean_prior_logits,
                shortlist_scores=clean_shortlist_scores,
                candidate_pack_sizes=clean_pack_sizes,
                decoded_pack_size=clean_dec_pack,
                visual_energies=clean_visual_energies,
                rank1_id=rank1_id,
                rank1_visual_similarity=clean_r1_sim,
                sister_pack_collision=sister_pack_collision,
                layer5_slot_context=clean_call_slot_ctx,
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
        can_use_cache: bool = True,
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
        layer5_slot_context: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> DecisionResult:
        # 1. Stage 2A Adaptive Fast-Exit check for single-slot retrieval verification
        if self.enable_stage2a_fast_exit and len(questions) == 1 and prior_logits:
            only_qid = next(iter(questions.keys()))
            only_question = questions[only_qid]
            allowed_opts = only_question.option_keys()
            allowed_set = set(allowed_opts)
            slot_priors = prior_logits.get(only_qid, {})
            if slot_priors and allowed_opts:
                max_lp = max(slot_priors.values())
                exps = {k: math.exp(v - max_lp) for k, v in slot_priors.items()}
                tot = sum(exps.values())
                prior_probs = {k: v / tot for k, v in exps.items()}
                fast_ok, winner = stage2a_fast_exit_gate(
                    prior_probs,
                    sister_pack_collision=sister_pack_collision,
                )
                if fast_ok and winner is not None and winner in allowed_set:
                    closed_raw = {k: prior_probs.get(k, 0.0) for k in allowed_opts}
                    closed_tot = sum(closed_raw.values())
                    closed_probs = (
                        {k: v / closed_tot for k, v in closed_raw.items()}
                        if closed_tot > 0.0
                        else {k: 1.0 / len(allowed_opts) for k in allowed_opts}
                    )
                    h_fast = normalized_entropy(closed_probs)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    term = (
                        DecisionTerminalState.ABSTAIN_OOS
                        if winner in self.oos_keys
                        else DecisionTerminalState.STAGE2A_FAST_EXIT
                    )
                    f_cat_fast = (
                        FailureCategory.ABSTAINED_OOS
                        if term == DecisionTerminalState.ABSTAIN_OOS
                        else FailureCategory.NONE
                    )
                    res = DecisionResult(
                        slots={
                            only_qid: SlotReadout(
                                slot_id=only_qid,
                                predicted=winner,
                                probabilities=closed_probs,
                                raw_probabilities=closed_probs,
                                h_raw=h_fast,
                                confidence=1.0 - h_fast,
                                escalated=False,
                                guard_applied="stage2a_fast_exit",
                                backend_name="stage2a_prior_fast_exit",
                                passes_used=0,
                                terminal_state=term,
                                failure_category=f_cat_fast,
                            )
                        },
                        escalated=False,
                        latency_ms=elapsed,
                        backend_name="stage2a_prior_fast_exit",
                        terminal_state=term,
                        failure_category=f_cat_fast,
                        cache_key=cache_key,
                    )
                    if can_use_cache and self.response_cache is not None:
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
                    try:
                        active_questions[qid] = shortlist_wide_question(
                            q,
                            s_map,
                            top_k=self.shortlist_top_k,
                            pairwise_diff=self.shortlist_pairwise_diff,
                        )
                    except DecisionContractError:
                        raise
                    except (ValueError, TypeError) as exc:
                        raise SchemaContractViolationError(str(exc)) from exc
                else:
                    active_questions[qid] = q
            else:
                active_questions[qid] = q

        # 3. Optional Co-Denoised / Causal Evidence Slot Injection
        evidence_slot_id: str | None = None
        if self.evidence_prompt.strip():
            try:
                active_questions = inject_evidence_slot(
                    active_questions,
                    self.evidence_prompt,
                    self.evidence_criteria,
                )
            except DecisionContractError:
                raise
            except (ValueError, TypeError) as exc:
                raise SchemaContractViolationError(str(exc)) from exc
            evidence_slot_id = "evidence_feature"

        # 4. Run Stage-1 Fast Backend with SLA check
        raw_stage1 = self.stage1_backend.score_questions(state, active_questions, images)
        if (time.perf_counter() - t0) > self.sla_timeout_sec:
            raise UpstreamTransientError(
                f"Stage-1 exceeded SLA budget ({self.sla_timeout_sec:.1f}s)"
            )
        if not isinstance(raw_stage1, Mapping):
            raise SchemaContractViolationError("Stage-1 backend returned non-mapping response")
        backend_evidence: Mapping[str, str] = getattr(self.stage1_backend, "last_evidence", {}) or {}
        stage1_epistemic: Mapping[str, float] = getattr(self.stage1_backend, "last_epistemic_variance", {}) or {}

        injected_ev_choice = ""
        if evidence_slot_id and evidence_slot_id in raw_stage1 and raw_stage1[evidence_slot_id]:
            ev_dist = _validate_slot_distribution(
                raw_stage1[evidence_slot_id], f"Stage-1 slot {evidence_slot_id!r}"
            )
            injected_ev_choice = max(ev_dist.items(), key=lambda kv: kv[1])[0]

        slots_out: dict[str, SlotReadout] = {}
        escalated_qids: list[str] = []
        escalation_reasons: list[str] = []

        for qid, q in active_questions.items():
            raw_dist = _validate_slot_distribution(raw_stage1.get(qid), f"Stage-1 slot {qid!r}")

            # Evaluate Split Conformal / RAPS set strictly on raw_dist BEFORE Stage-1 prior fusion.
            slot_conformal_set: list[str] = []
            conformal_esc = False
            slot_q_hat = self._resolve_slot_q_hat(qid) if qid != evidence_slot_id else None
            if slot_q_hat is not None:
                try:
                    conf_res = conformal_prediction_set(
                        raw_dist,
                        slot_q_hat,
                        method=self.conformal_method,
                        lam_reg=self.conformal_lam_reg,
                        k_reg=self.conformal_k_reg,
                        abstain_keys=self.oos_keys | DEFAULT_CONFORMAL_ABSTAIN_KEYS,
                        randomized=self.conformal_randomized,
                        seed=self.conformal_seed,
                        rng=self.conformal_rng,
                    )
                except DecisionContractError:
                    raise
                except (ValueError, TypeError) as exc:
                    raise SchemaContractViolationError(str(exc)) from exc
                slot_conformal_set = conf_res.prediction_set
                if self.escalate_on_conformal_ambiguous:
                    conformal_esc = conf_res.should_escalate

            slot_prior = (
                ((prior_logits or {}).get(qid) or q.prior_logits or None)
                if qid != evidence_slot_id
                else None
            )
            try:
                fused_dist, h_raw, should_esc = pre_gated_bayesian_poe_fusion(
                    raw_dist,
                    prior_logits=slot_prior,
                    prior_weight=self.prior_weight if slot_prior else 0.0,
                    temperature=self.temperature,
                    hesitation_threshold=self.hesitation_threshold,
                    gate_mode=self.prior_gate_mode,
                )
            except DecisionContractError:
                raise
            except (ValueError, TypeError) as exc:
                raise SchemaContractViolationError(str(exc)) from exc
            predicted = max(fused_dist.items(), key=lambda kv: kv[1])[0]
            guard_name = "none"
            if qid != evidence_slot_id:
                s1_packs, s1_dec_pack, s1_ven, s1_r1_id, s1_r1_sim = self._resolve_slot_layer5_args(
                    qid,
                    slot_context_override=layer5_slot_context,
                    candidate_pack_sizes=candidate_pack_sizes,
                    decoded_pack_size=decoded_pack_size,
                    visual_energies=visual_energies,
                    rank1_id=rank1_id,
                    rank1_visual_similarity=rank1_visual_similarity,
                )
                slot_allowed_s1 = set(q.option_keys())
                if s1_packs:
                    s1_packs = {k: v for k, v in s1_packs.items() if k in slot_allowed_s1} or None
                if s1_r1_id is not None and s1_r1_id not in slot_allowed_s1:
                    s1_r1_id = None
                if s1_packs or s1_ven or s1_r1_sim is not None:
                    try:
                        predicted, guard_name = apply_layer5_deterministic_guards(
                            chosen_id=predicted,
                            raw_probs=raw_dist,
                            candidate_pack_sizes=s1_packs,
                            decoded_pack_size=s1_dec_pack,
                            visual_energies=s1_ven,
                            rank1_id=s1_r1_id,
                            rank1_visual_similarity=s1_r1_sim,
                        )
                    except DecisionContractError:
                        raise
                    except (ValueError, TypeError) as exc:
                        raise SchemaContractViolationError(str(exc)) from exc

            slot_ev = backend_evidence.get(qid, injected_ev_choice if qid != evidence_slot_id else "")
            inferred_flag = (
                self.escalate_on_inferred_evidence
                and qid != evidence_slot_id
                and slot_ev in _INFERRED_EVIDENCE_LABELS
            )

            if qid != evidence_slot_id and (should_esc or inferred_flag or conformal_esc):
                escalated_qids.append(qid)
                if should_esc:
                    escalation_reasons.append(f"{qid}:h_raw={h_raw:.4f}>={self.hesitation_threshold}")
                if inferred_flag:
                    escalation_reasons.append(f"{qid}:evidence={slot_ev}")
                if conformal_esc:
                    escalation_reasons.append(f"{qid}:conformal_set_size={len(slot_conformal_set)}")

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
                conformal_set=slot_conformal_set,
                epistemic_variance=float(stage1_epistemic.get(qid, 0.0)),
            )

        # 5. Optional Stage-2 Frontier Escalation on Hesitant, Conformal-Ambiguous, or Inferred-Evidence Slots
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
            if not isinstance(stage2_dists, Mapping):
                raise SchemaContractViolationError("Stage-2 backend returned non-mapping response")
            s2_evidence: Mapping[str, str] = getattr(self.stage2_backend, "last_evidence", {}) or {}
            s2_epistemic: Mapping[str, float] = getattr(self.stage2_backend, "last_epistemic_variance", {}) or {}
            for qid in escalated_qids:
                s2_dist = _validate_slot_distribution(stage2_dists.get(qid), f"Stage-2 slot {qid!r}")
                s2_h = normalized_entropy(s2_dist)
                s2_pred = max(s2_dist.items(), key=lambda kv: kv[1])[0]
                s2_guard_name = "none"
                if qid != evidence_slot_id:
                    s2_packs, s2_dec_pack, s2_ven, s2_r1_id, s2_r1_sim = self._resolve_slot_layer5_args(
                        qid,
                        slot_context_override=layer5_slot_context,
                        candidate_pack_sizes=candidate_pack_sizes,
                        decoded_pack_size=decoded_pack_size,
                        visual_energies=visual_energies,
                        rank1_id=rank1_id,
                        rank1_visual_similarity=rank1_visual_similarity,
                    )
                    slot_allowed_s2 = set(active_questions[qid].option_keys())
                    if s2_packs:
                        s2_packs = {k: v for k, v in s2_packs.items() if k in slot_allowed_s2} or None
                    if s2_r1_id is not None and s2_r1_id not in slot_allowed_s2:
                        s2_r1_id = None
                    if s2_packs or s2_ven or s2_r1_sim is not None:
                        try:
                            s2_pred, s2_guard_name = apply_layer5_deterministic_guards(
                                chosen_id=s2_pred,
                                raw_probs=s2_dist,
                                candidate_pack_sizes=s2_packs,
                                decoded_pack_size=s2_dec_pack,
                                visual_energies=s2_ven,
                                rank1_id=s2_r1_id,
                                rank1_visual_similarity=s2_r1_sim,
                            )
                        except DecisionContractError:
                            raise
                        except (ValueError, TypeError) as exc:
                            raise SchemaContractViolationError(str(exc)) from exc
                if s2_pred in self.oos_keys:
                    s2_term = DecisionTerminalState.ABSTAIN_OOS
                elif s2_guard_name != "none":
                    s2_term = DecisionTerminalState.GUARD_OVERRIDDEN
                else:
                    s2_term = DecisionTerminalState.STAGE2_VERIFIED
                slots_out[qid] = SlotReadout(
                    slot_id=qid,
                    predicted=s2_pred,
                    probabilities=s2_dist,
                    raw_probabilities=s2_dist,
                    h_raw=s2_h,
                    confidence=1.0 - s2_h,
                    escalated=True,
                    guard_applied=s2_guard_name,
                    backend_name=self.stage2_backend.name,
                    evidence=s2_evidence.get(qid, slots_out[qid].evidence),
                    passes_used=2,
                    terminal_state=s2_term,
                    failure_category=(
                        FailureCategory.ABSTAINED_OOS
                        if s2_term == DecisionTerminalState.ABSTAIN_OOS
                        else FailureCategory.NONE
                    ),
                    conformal_set=list(slots_out[qid].conformal_set),
                    epistemic_variance=float(s2_epistemic.get(qid, slots_out[qid].epistemic_variance)),
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        non_ev_states = [
            s.terminal_state for sid, s in slots_out.items() if sid != evidence_slot_id
        ]
        if any(st == DecisionTerminalState.ABSTAIN_OOS for st in non_ev_states):
            overall_state = DecisionTerminalState.ABSTAIN_OOS
        elif any(st == DecisionTerminalState.GUARD_OVERRIDDEN for st in non_ev_states):
            overall_state = DecisionTerminalState.GUARD_OVERRIDDEN
        elif did_escalate:
            overall_state = DecisionTerminalState.STAGE2_VERIFIED
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
        if can_use_cache and self.response_cache is not None:
            self.response_cache.put(cache_key, result)
        return result

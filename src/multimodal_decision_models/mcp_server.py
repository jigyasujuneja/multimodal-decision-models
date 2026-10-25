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

"""Zero-Dependency Model Context Protocol (MCP) JSON-RPC 2.0 `stdio` Server (`dgem-catalog`).

Run via:
    python3 -m multimodal_decision_models.mcp_server

Supports both standard MCP newline-delimited JSON-RPC 2.0 messages and LSP-style
`Content-Length: <N>\\r\\n\\r\\n` framed messages over `stdin` / `stdout`, exposing four tools:
  1. `shortlist_candidates`
  2. `evaluate_conformal_decision`
  3. `prune_context_blocks`
  4. `compile_contact_sheet`
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import sys
from typing import Any, BinaryIO, Mapping, Sequence

from .calibration import VALID_PRIOR_GATE_MODES, pre_gated_bayesian_poe_fusion
from .conformal import (
    DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    calibrate_conformal_quantile,
    conformal_prediction_set,
)
from .context_pruner import ContextBlock, prune_agent_context_blocks
from .contracts import DecisionContractError
from .embedding_index import MultimodalEmbeddingIndex
from .visual_compiler import (
    ImageBuffer,
    compile_dual_buffer_contact_sheet,
    decode_png,
    decode_ppm,
)

MCP_SERVER_NAME = "dgem-catalog"
MCP_SERVER_VERSION = "0.4.0"
DEFAULT_MCP_PROTOCOL_VERSION: str = "2024-11-05"
MAX_FRAME_BYTES: int = 16 * 1024 * 1024
MAX_CONTENT_LENGTH_BYTES: int = MAX_FRAME_BYTES
MAX_HEADER_LINES: int = 64
_MAX_DRAIN_HEADER_LINES: int = 1024
_MIME_HEADER_LINE_RE = re.compile(rb"^([A-Za-z][A-Za-z0-9_-]*)[ \t]*:(.*)$")


class _InvalidParamsError(ValueError):
    """Raised when JSON-RPC params or MCP tool arguments violate schema types."""


class _FramingInvalidRequestError(ValueError):
    """Raised when LSP Content-Length header bounds or header line limits are violated."""


def _reject_json_constant(token: str) -> Any:
    """Reject non-RFC-8259 NaN/Infinity tokens during JSON parsing."""
    raise ValueError(f"Non-RFC-8259 JSON constant not permitted: {token}")


def _require_mapping(val: Any, field: str, *, non_empty: bool = False) -> Mapping[str, Any]:
    if not isinstance(val, Mapping):
        raise _InvalidParamsError(f"{field} must be an object")
    if non_empty and not val:
        raise _InvalidParamsError(f"{field} must be a non-empty object")
    return val


def _require_sequence(val: Any, field: str, *, non_empty: bool = False) -> Sequence[Any]:
    if not isinstance(val, list):
        raise _InvalidParamsError(f"{field} must be an array")
    if non_empty and not val:
        raise _InvalidParamsError(f"{field} must be a non-empty array")
    return val


def _require_number(val: Any, field: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise _InvalidParamsError(f"{field} must be a finite number")
    out = float(val)
    if not math.isfinite(out):
        raise _InvalidParamsError(f"{field} must be a finite number")
    return out


def _require_int(val: Any, field: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise _InvalidParamsError(f"{field} must be an integer")
    return int(val)


def _require_str(val: Any, field: str, *, non_empty: bool = False) -> str:
    if not isinstance(val, str):
        raise _InvalidParamsError(f"{field} must be a string")
    if non_empty and not val:
        raise _InvalidParamsError(f"{field} must be a non-empty string")
    return val


def _require_bool(val: Any, field: str) -> bool:
    if not isinstance(val, bool):
        raise _InvalidParamsError(f"{field} must be a boolean")
    return val


def _require_float_map(val: Any, field: str, *, non_empty: bool = False) -> dict[str, float]:
    m = _require_mapping(val, field, non_empty=non_empty)
    out: dict[str, float] = {}
    for k, v in m.items():
        if not isinstance(k, str) or not k:
            raise _InvalidParamsError(f"{field} keys must be non-empty strings")
        out[k] = _require_number(v, f"{field}.{k}")
    return out


def _parse_lab_triple(val: Any, field: str) -> tuple[float, float, float] | None:
    if val is None:
        return None
    seq = _require_sequence(val, field)
    if len(seq) != 3:
        raise _InvalidParamsError(f"{field} must have length 3")
    return (
        _require_number(seq[0], f"{field}[0]"),
        _require_number(seq[1], f"{field}[1]"),
        _require_number(seq[2], f"{field}[2]"),
    )


_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "shortlist_candidates",
        "description": (
            "Rank multimodal catalog items using cosine/CSLS similarity, CIE Lab color, "
            "and Lidstone container priors, returning calibrated prior logits and Stage-2A fast-exit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "embedding": {"type": "array", "items": {"type": "number"}},
                            "title": {"type": "string"},
                            "lab_color": {"type": "array", "items": {"type": "number"}},
                            "aspect_bin": {"type": "string"},
                            "pack_size": {"type": "integer"},
                        },
                        "required": ["id", "embedding"],
                    },
                },
                "query_embedding": {"type": "array", "items": {"type": "number"}},
                "query_lab_color": {"type": "array", "items": {"type": "number"}},
                "query_aspect_bin": {"type": "string"},
                "top_k": {"type": "integer", "default": 6},
                "use_csls": {"type": "boolean", "default": False},
            },
            "required": ["items", "query_embedding"],
        },
    },
    {
        "name": "evaluate_conformal_decision",
        "description": (
            "Evaluate pre-fusion Split Conformal / RAPS prediction sets and pre-gated Bayesian PoE "
            "fusion on raw option probabilities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "q_hat": {
                    "type": ["number", "null"],
                    "description": (
                        "Calibrated nonconformity quantile threshold, or null when "
                        "q_hat_infinite is true (trivial full-set coverage)."
                    ),
                },
                "q_hat_infinite": {
                    "type": "boolean",
                    "description": (
                        "Set to true (with q_hat null or omitted) when the calibrated "
                        "quantile is +Infinity so every candidate is retained."
                    ),
                },
                "calib_probs": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "calib_labels": {"type": "array", "items": {"type": "string"}},
                "alpha": {"type": "number", "default": 0.10},
                "method": {"type": "string", "enum": ["lac", "aps", "raps"], "default": "raps"},
                "lam_reg": {"type": "number", "default": 0.01},
                "k_reg": {"type": "integer", "default": 1},
                "prior_logits": {"type": "object", "additionalProperties": {"type": "number"}},
                "prior_weight": {"type": "number", "default": 0.0},
                "prior_gate_mode": {
                    "type": "string",
                    "enum": list(VALID_PRIOR_GATE_MODES),
                    "default": "hesitant",
                },
                "hesitation_threshold": {"type": "number", "default": 0.16},
            },
            "required": ["probabilities"],
        },
    },
    {
        "name": "prune_context_blocks",
        "description": (
            "Solve exact MCKP utility optimization over {full, stub, drop} actions for agent "
            "context blocks under a token budget while preserving pinned blocks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "full_tokens": {"type": "integer"},
                            "stub_tokens": {"type": "integer"},
                            "full_text": {"type": "string"},
                            "stub_text": {"type": "string"},
                            "pinned": {"type": "boolean"},
                        },
                        "required": ["id", "full_tokens", "stub_tokens"],
                    },
                },
                "slot_probabilities": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "token_budget": {"type": "integer"},
            },
            "required": ["blocks", "slot_probabilities", "token_budget"],
        },
    },
    {
        "name": "compile_contact_sheet",
        "description": (
            "Compile a dual-buffer Stage-2B contact sheet PNG with automatic inter-candidate "
            "saliency zoom, bottom-30% pack-weight zoom, and optional cylindrical unwarp."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "object"},
                "query_png_base64": {"type": "string"},
                "candidates": {"type": "array", "items": {"type": "object"}},
                "candidate_metadata": {"type": "array", "items": {"type": "object"}},
                "curved_container": {"type": "boolean", "default": False},
                "band_height_fraction": {"type": "number"},
            },
            "required": ["candidates"],
        },
    },
]


def _decode_image_spec(spec: Any, field: str = "image") -> ImageBuffer:
    if isinstance(spec, str):
        if not spec:
            raise _InvalidParamsError(f"{field} string must be non-empty")
        try:
            raw_bytes = base64.b64decode(spec, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"invalid base64 in {field}: {exc}") from exc
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return decode_png(raw_bytes)
        return decode_ppm(raw_bytes)

    if not isinstance(spec, Mapping):
        raise _InvalidParamsError(f"{field} must be an image object or base64 string")

    if "png_base64" in spec and spec["png_base64"] is not None:
        png_b64 = _require_str(spec["png_base64"], f"{field}.png_base64", non_empty=True)
        return decode_png(base64.b64decode(png_b64, validate=True))
    if "ppm_base64" in spec and spec["ppm_base64"] is not None:
        ppm_b64 = _require_str(spec["ppm_base64"], f"{field}.ppm_base64", non_empty=True)
        return decode_ppm(base64.b64decode(ppm_b64, validate=True))
    if "width" in spec and "height" in spec:
        w = _require_int(spec["width"], f"{field}.width")
        h = _require_int(spec["height"], f"{field}.height")
        if "rgb_hex" in spec and spec["rgb_hex"] is not None:
            rgb_hex = _require_str(spec["rgb_hex"], f"{field}.rgb_hex", non_empty=True)
            rgb = bytes.fromhex(rgb_hex)
        elif "rgb_base64" in spec and spec["rgb_base64"] is not None:
            rgb_b64 = _require_str(spec["rgb_base64"], f"{field}.rgb_base64", non_empty=True)
            rgb = base64.b64decode(rgb_b64, validate=True)
        else:
            raise _InvalidParamsError(f"{field} raw ImageBuffer spec requires rgb_hex or rgb_base64")
        return ImageBuffer(w, h, rgb)
    raise _InvalidParamsError(
        f"unsupported {field} specification; provide png_base64, ppm_base64, or width/height/rgb_hex"
    )


def _tool_shortlist_candidates(args: Mapping[str, Any]) -> dict[str, Any]:
    if "items" not in args:
        raise _InvalidParamsError("items is required")
    items = _require_sequence(args.get("items"), "items", non_empty=True)
    if "query_embedding" not in args:
        raise _InvalidParamsError("query_embedding is required")
    query_emb_seq = _require_sequence(args.get("query_embedding"), "query_embedding", non_empty=True)
    query_vec = [
        _require_number(x, f"query_embedding[{j}]") for j, x in enumerate(query_emb_seq)
    ]

    idx = MultimodalEmbeddingIndex()
    for i, raw_item in enumerate(items):
        item_obj = _require_mapping(raw_item, f"items[{i}]")
        if "id" not in item_obj or "embedding" not in item_obj:
            raise _InvalidParamsError(f"items[{i}] requires id and embedding")
        item_id = _require_str(item_obj["id"], f"items[{i}].id", non_empty=True)
        emb_seq = _require_sequence(item_obj["embedding"], f"items[{i}].embedding", non_empty=True)
        emb_vec = [
            _require_number(x, f"items[{i}].embedding[{j}]") for j, x in enumerate(emb_seq)
        ]
        title = (
            _require_str(item_obj["title"], f"items[{i}].title")
            if item_obj.get("title") is not None
            else ""
        )
        lab_t = _parse_lab_triple(item_obj.get("lab_color"), f"items[{i}].lab_color")
        aspect_bin = (
            _require_str(item_obj["aspect_bin"], f"items[{i}].aspect_bin", non_empty=True)
            if item_obj.get("aspect_bin") is not None
            else "standard"
        )
        pack_sz = (
            _require_int(item_obj["pack_size"], f"items[{i}].pack_size")
            if item_obj.get("pack_size") is not None
            else None
        )
        meta_raw = item_obj.get("metadata")
        meta_clean: dict[str, str] = {}
        if meta_raw is not None:
            meta_map = _require_mapping(meta_raw, f"items[{i}].metadata")
            for mk, mv in meta_map.items():
                if not isinstance(mk, str):
                    raise _InvalidParamsError(f"items[{i}].metadata keys must be strings")
                if mv is None or isinstance(mv, (Mapping, list)):
                    raise _InvalidParamsError(f"items[{i}].metadata.{mk} must be scalar")
                meta_clean[mk] = str(mv)
        idx.upsert(
            item_id,
            emb_vec,
            title=title,
            lab_color=lab_t,
            aspect_bin=aspect_bin,
            pack_size=pack_sz,
            metadata=meta_clean,
        )

    q_lab = _parse_lab_triple(args.get("query_lab_color"), "query_lab_color")
    q_aspect = (
        _require_str(args["query_aspect_bin"], "query_aspect_bin")
        if args.get("query_aspect_bin") is not None
        else None
    )
    top_k = _require_int(args["top_k"], "top_k") if "top_k" in args and args["top_k"] is not None else 6
    use_csls = (
        _require_bool(args["use_csls"], "use_csls")
        if "use_csls" in args and args["use_csls"] is not None
        else False
    )
    res = idx.query(
        query_vec,
        query_lab_color=q_lab,
        query_aspect_bin=q_aspect or None,
        top_k=top_k,
        use_csls=use_csls,
    )
    return {
        "shortlist_ids": res.shortlist_ids,
        "stage2a_energies": res.stage2a_energies,
        "stage2a_probabilities": res.stage2a_probabilities,
        "prior_logits": res.prior_logits,
        "fast_exit": res.fast_exit,
        "fast_exit_id": res.fast_exit_id,
        "top1_margin": res.top1_margin,
        "z_score": res.z_score,
    }


def _tool_evaluate_conformal_decision(args: Mapping[str, Any]) -> dict[str, Any]:
    if "probabilities" not in args:
        raise _InvalidParamsError("probabilities is required")
    probs = _require_float_map(args.get("probabilities"), "probabilities", non_empty=True)

    method = (
        _require_str(args["method"], "method", non_empty=True)
        if "method" in args and args["method"] is not None
        else "raps"
    )
    lam_reg = (
        _require_number(args["lam_reg"], "lam_reg")
        if "lam_reg" in args and args["lam_reg"] is not None
        else 0.01
    )
    k_reg = (
        _require_int(args["k_reg"], "k_reg")
        if "k_reg" in args and args["k_reg"] is not None
        else 1
    )

    c_probs: list[dict[str, float]] | None = None
    c_labels: list[str] | None = None
    if "calib_probs" in args and args["calib_probs"] is not None:
        cp_seq = _require_sequence(args["calib_probs"], "calib_probs", non_empty=True)
        c_probs = [
            _require_float_map(row, f"calib_probs[{i}]", non_empty=True)
            for i, row in enumerate(cp_seq)
        ]
    if "calib_labels" in args and args["calib_labels"] is not None:
        cl_seq = _require_sequence(args["calib_labels"], "calib_labels", non_empty=True)
        c_labels = [
            _require_str(lbl, f"calib_labels[{i}]", non_empty=True)
            for i, lbl in enumerate(cl_seq)
        ]

    q_hat_present = "q_hat" in args
    q_hat_raw = args.get("q_hat")
    q_hat_inf_raw = args.get("q_hat_infinite")
    q_hat_inf_flag: bool | None = (
        _require_bool(q_hat_inf_raw, "q_hat_infinite")
        if q_hat_inf_raw is not None
        else None
    )

    if q_hat_inf_flag is True:
        if q_hat_raw is not None:
            raise _InvalidParamsError(
                "q_hat must be null or omitted when q_hat_infinite is true"
            )
        q_hat = float("inf")
    elif q_hat_raw is not None:
        q_hat = _require_number(q_hat_raw, "q_hat")
    elif q_hat_present and q_hat_raw is None and q_hat_inf_flag is False:
        raise _InvalidParamsError(
            "contradictory parameters: q_hat is null while q_hat_infinite is false"
        )
    elif q_hat_present and q_hat_raw is None and (c_probs is None or c_labels is None):
        raise _InvalidParamsError(
            "q_hat cannot be null unless q_hat_infinite is true or (calib_probs and calib_labels) are provided"
        )
    elif c_probs is not None and c_labels is not None:
        alpha = (
            _require_number(args["alpha"], "alpha")
            if "alpha" in args and args["alpha"] is not None
            else 0.10
        )
        cal_res = calibrate_conformal_quantile(
            c_probs,
            c_labels,
            alpha=alpha,
            method=method,
            lam_reg=lam_reg,
            k_reg=k_reg,
        )
        q_hat = cal_res.q_hat
    else:
        raise _InvalidParamsError(
            "provide either q_hat, q_hat_infinite=true, or (calib_probs and calib_labels)"
        )

    # Evaluate conformal prediction set strictly on raw_probs BEFORE prior fusion.
    conf_res = conformal_prediction_set(
        probs,
        q_hat,
        method=method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        abstain_keys=DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    )

    prior_logits_raw = args.get("prior_logits")
    prior_logits = (
        _require_float_map(prior_logits_raw, "prior_logits", non_empty=True)
        if prior_logits_raw is not None
        else None
    )
    prior_weight = (
        _require_number(args["prior_weight"], "prior_weight")
        if "prior_weight" in args and args["prior_weight"] is not None
        else 0.0
    )
    prior_gate_mode = (
        _require_str(args["prior_gate_mode"], "prior_gate_mode", non_empty=True)
        if "prior_gate_mode" in args and args["prior_gate_mode"] is not None
        else "hesitant"
    )
    if prior_gate_mode not in VALID_PRIOR_GATE_MODES:
        raise _InvalidParamsError(
            f"prior_gate_mode must be one of {list(VALID_PRIOR_GATE_MODES)}, got {prior_gate_mode!r}"
        )
    hesitation_threshold = (
        _require_number(args["hesitation_threshold"], "hesitation_threshold")
        if "hesitation_threshold" in args and args["hesitation_threshold"] is not None
        else 0.16
    )

    fused_probs, h_raw, hes_escalate = pre_gated_bayesian_poe_fusion(
        probs,
        prior_logits=prior_logits,
        prior_weight=prior_weight if prior_logits else 0.0,
        hesitation_threshold=hesitation_threshold,
        gate_mode=prior_gate_mode,
    )
    predicted = max(fused_probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
    q_hat_is_inf = math.isinf(q_hat) and q_hat > 0.0
    if not q_hat_is_inf and not math.isfinite(q_hat):
        raise ValueError("calibrated q_hat must be finite or +inf")
    return {
        "predicted": predicted,
        "probabilities": fused_probs,
        "raw_probabilities": probs,
        "h_raw": h_raw,
        "q_hat": None if q_hat_is_inf else float(q_hat),
        "q_hat_infinite": q_hat_is_inf,
        "prediction_set": conf_res.prediction_set,
        "singleton": conf_res.singleton,
        "contains_abstain": conf_res.contains_abstain,
        "conformal_escalate": conf_res.should_escalate,
        "hesitation_escalate": hes_escalate,
        "should_escalate": bool(conf_res.should_escalate or hes_escalate),
    }


def _tool_prune_context_blocks(args: Mapping[str, Any]) -> dict[str, Any]:
    if "blocks" not in args:
        raise _InvalidParamsError("blocks is required")
    raw_blocks = _require_sequence(args.get("blocks"), "blocks")
    blocks: list[ContextBlock] = []
    for i, b_raw in enumerate(raw_blocks):
        b = _require_mapping(b_raw, f"blocks[{i}]")
        if "id" not in b or "full_tokens" not in b or "stub_tokens" not in b:
            raise _InvalidParamsError(f"blocks[{i}] requires id, full_tokens, and stub_tokens")
        blocks.append(
            ContextBlock(
                id=_require_str(b["id"], f"blocks[{i}].id", non_empty=True),
                full_tokens=_require_int(b["full_tokens"], f"blocks[{i}].full_tokens"),
                stub_tokens=_require_int(b["stub_tokens"], f"blocks[{i}].stub_tokens"),
                full_text=(
                    _require_str(b["full_text"], f"blocks[{i}].full_text")
                    if b.get("full_text") is not None
                    else ""
                ),
                stub_text=(
                    _require_str(b["stub_text"], f"blocks[{i}].stub_text")
                    if b.get("stub_text") is not None
                    else ""
                ),
                pinned=(
                    _require_bool(b["pinned"], f"blocks[{i}].pinned")
                    if b.get("pinned") is not None
                    else False
                ),
            )
        )
    if "slot_probabilities" not in args:
        raise _InvalidParamsError("slot_probabilities is required")
    raw_sp = _require_mapping(args.get("slot_probabilities"), "slot_probabilities")
    slot_probs: dict[str, dict[str, float]] = {}
    for bid, dist in raw_sp.items():
        if not isinstance(bid, str) or not bid:
            raise _InvalidParamsError("slot_probabilities keys must be non-empty strings")
        slot_probs[bid] = _require_float_map(dist, f"slot_probabilities.{bid}", non_empty=True)
    if "token_budget" not in args:
        raise _InvalidParamsError("token_budget is required")
    budget = _require_int(args.get("token_budget"), "token_budget")
    res = prune_agent_context_blocks(blocks, slot_probs, budget)
    return {
        "decisions": res.decisions,
        "tokens_before": res.tokens_before,
        "tokens_after": res.tokens_after,
        "token_reduction_ratio": res.token_reduction_ratio,
        "expected_utility": res.expected_utility,
        "rendered_blocks": res.rendered_blocks,
    }


def _tool_compile_contact_sheet(args: Mapping[str, Any]) -> dict[str, Any]:
    if "query_png_base64" in args and args["query_png_base64"] is not None:
        q_b64 = _require_str(args["query_png_base64"], "query_png_base64", non_empty=True)
        query_buf = _decode_image_spec(q_b64, "query_png_base64")
    elif "query" in args and args["query"] is not None:
        query_buf = _decode_image_spec(args["query"], "query")
    elif (
        "query_width" in args
        and "query_height" in args
        and ("query_rgb_hex" in args or "query_rgb_base64" in args)
    ):
        query_buf = _decode_image_spec(
            {
                "width": args["query_width"],
                "height": args["query_height"],
                "rgb_hex": args.get("query_rgb_hex"),
                "rgb_base64": args.get("query_rgb_base64"),
            },
            "query",
        )
    else:
        raise _InvalidParamsError("compile_contact_sheet requires query_png_base64 or query object")

    if "candidates" not in args:
        raise _InvalidParamsError("candidates is required")
    raw_cands = _require_sequence(args.get("candidates"), "candidates", non_empty=True)
    cand_bufs = [_decode_image_spec(c, f"candidates[{i}]") for i, c in enumerate(raw_cands)]

    raw_meta = args.get("candidate_metadata")
    meta_list: list[dict[str, str]] | None = None
    if raw_meta is not None:
        meta_seq = _require_sequence(raw_meta, "candidate_metadata")
        meta_list = []
        for i, m in enumerate(meta_seq):
            m_obj = _require_mapping(m, f"candidate_metadata[{i}]")
            clean_m: dict[str, str] = {}
            for k, v in m_obj.items():
                if not isinstance(k, str):
                    raise _InvalidParamsError(f"candidate_metadata[{i}] keys must be strings")
                if v is None or isinstance(v, (Mapping, list)):
                    raise _InvalidParamsError(f"candidate_metadata[{i}].{k} must be scalar")
                clean_m[k] = str(v)
            meta_list.append(clean_m)

    band_frac = (
        _require_number(args["band_height_fraction"], "band_height_fraction")
        if args.get("band_height_fraction") is not None
        else None
    )
    curved = (
        _require_bool(args["curved_container"], "curved_container")
        if args.get("curved_container") is not None
        else False
    )
    sheet = compile_dual_buffer_contact_sheet(
        query_buf,
        cand_bufs,
        candidate_metadata=meta_list,
        curved_container=curved,
        band_height_fraction=band_frac,
    )
    return {
        "png_base64": base64.b64encode(sheet.png_bytes).decode("ascii"),
        "saliency_band": [sheet.saliency_band[0], sheet.saliency_band[1]],
        "variance_profile": sheet.variance_profile,
        "diff_hints": sheet.diff_hints,
        "panel_labels": sheet.panel_labels,
        "stage2a_query_size": [sheet.stage2a_query_buffer.width, sheet.stage2a_query_buffer.height],
    }


def handle_jsonrpc_message(msg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 request or notification and return response dict (or None)."""
    has_id = "id" in msg
    req_id = msg.get("id")
    if has_id and req_id is not None and (
        isinstance(req_id, bool)
        or not isinstance(req_id, (str, int, float))
        or (isinstance(req_id, float) and not math.isfinite(req_id))
    ):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "Invalid Request: id must be a string, finite number, or null",
            },
        }
    method = msg.get("method")

    if not isinstance(method, str):
        if not has_id:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid JSON-RPC request: missing string method"},
        }

    # Notifications carry no id and produce no response frame.
    if not has_id:
        return None

    if "params" not in msg or msg["params"] is None:
        params: Mapping[str, Any] = {}
    elif isinstance(msg["params"], Mapping):
        params = msg["params"]
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": "Invalid params: expected object"},
        }

    if method == "initialize":
        raw_proto = params.get("protocolVersion")
        if raw_proto is not None and not isinstance(raw_proto, str):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "Invalid params: protocolVersion must be string"},
            }
        proto_ver = raw_proto if raw_proto else DEFAULT_MCP_PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": proto_ver,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": MCP_SERVER_NAME,
                    "version": MCP_SERVER_VERSION,
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOL_SCHEMAS}}

    if method == "tools/call":
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "tools/call requires non-empty string name"},
            }
        if "arguments" not in params:
            tool_args: Mapping[str, Any] = {}
        elif isinstance(params["arguments"], Mapping):
            tool_args = params["arguments"]
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "tools/call requires object arguments"},
            }
        dispatch = {
            "shortlist_candidates": _tool_shortlist_candidates,
            "evaluate_conformal_decision": _tool_evaluate_conformal_decision,
            "prune_context_blocks": _tool_prune_context_blocks,
            "compile_contact_sheet": _tool_compile_contact_sheet,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name!r}"},
            }
        try:
            payload = handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True, allow_nan=False)}],
                    "isError": False,
                },
            }
        except _InvalidParamsError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Invalid params: {exc}"},
            }
        except (DecisionContractError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Invalid tool arguments: {exc}"},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method!r}"},
    }


def _drain_mime_headers(stream: BinaryIO, max_lines: int = _MAX_DRAIN_HEADER_LINES) -> None:
    """Consume remaining lines of an oversized or malformed MIME header block up to the blank separator."""
    for _ in range(max_lines):
        hdr_line = stream.readline()
        if not hdr_line or hdr_line in (b"\r\n", b"\n"):
            break


def _read_exact_bytes(stream: BinaryIO, n: int) -> bytes | None:
    """Read exactly `n` bytes from `stream`, looping across partial pipe reads; return `None` on premature EOF."""
    if n == 0:
        return b""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        piece = stream.read(remaining)
        if not piece:
            return None
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _parse_content_length_value(raw_len_text: bytes | None) -> int:
    """Validate and parse `Content-Length` header value within `[0, MAX_CONTENT_LENGTH_BYTES]`."""
    if raw_len_text is None or not raw_len_text:
        raise _FramingInvalidRequestError("missing Content-Length header in MIME block")
    if raw_len_text.startswith(b"-") and len(raw_len_text) > 1 and raw_len_text[1:].isdigit():
        neg_val = -int(raw_len_text[1:])
        raise _FramingInvalidRequestError(
            f"Content-Length {neg_val} out of bounds [0, {MAX_CONTENT_LENGTH_BYTES}]"
        )
    if not raw_len_text.isdigit():
        bad_txt = raw_len_text.decode("latin-1", errors="replace")
        raise _FramingInvalidRequestError(f"invalid Content-Length header value {bad_txt!r}")
    content_len = int(raw_len_text)
    if content_len > MAX_CONTENT_LENGTH_BYTES:
        raise _FramingInvalidRequestError(
            f"Content-Length {content_len} out of bounds [0, {MAX_CONTENT_LENGTH_BYTES}]"
        )
    return content_len


def _read_next_frame(stream: BinaryIO) -> tuple[bytes | None, bool]:
    """Read next JSON-RPC payload from binary stream, returning `(raw_json_bytes, is_content_length_framed)`."""
    while True:
        line = stream.readline()
        if not line:
            return None, False
        stripped = line.strip()
        if not stripped:
            continue

        line_no_crlf = line.rstrip(b"\r\n")
        first_hdr = _MIME_HEADER_LINE_RE.match(line_no_crlf)
        if first_hdr is not None:
            raw_len_text: bytes | None = None
            framing_err: str | None = None
            first_name = first_hdr.group(1).lower()
            if first_name == b"content-length":
                raw_len_text = first_hdr.group(2).strip()

            header_lines_seen = 1
            while True:
                hdr_line = stream.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break
                header_lines_seen += 1
                if header_lines_seen > MAX_HEADER_LINES:
                    _drain_mime_headers(stream)
                    raise _FramingInvalidRequestError(
                        f"exceeded MAX_HEADER_LINES ({MAX_HEADER_LINES})"
                    )
                hdr_match = _MIME_HEADER_LINE_RE.match(hdr_line.rstrip(b"\r\n"))
                if hdr_match is None:
                    _drain_mime_headers(stream)
                    raise _FramingInvalidRequestError("malformed MIME header line")
                hdr_name = hdr_match.group(1).lower()
                if hdr_name == b"content-length":
                    if raw_len_text is not None:
                        framing_err = "duplicate Content-Length header"
                    else:
                        raw_len_text = hdr_match.group(2).strip()

            if framing_err is not None:
                raise _FramingInvalidRequestError(framing_err)
            content_len = _parse_content_length_value(raw_len_text)
            body = _read_exact_bytes(stream, content_len)
            if body is None:
                return None, True
            return body, True

        return stripped, False


def _write_frame(stream: BinaryIO, response: Mapping[str, Any], use_content_length: bool) -> None:
    encoded = json.dumps(response, sort_keys=True, allow_nan=False).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        stream.write(header + encoded)
    else:
        stream.write(encoded + b"\n")
    stream.flush()


def serve_stdio(stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Run the synchronous `dgem-catalog` MCP JSON-RPC 2.0 server loop until EOF."""
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer

    while True:
        try:
            raw_frame, framed = _read_next_frame(in_stream)
        except _FramingInvalidRequestError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": f"Invalid Request: {exc}"},
            }
            _write_frame(out_stream, err_resp, True)
            continue
        except ValueError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Framing error: {exc}"},
            }
            _write_frame(out_stream, err_resp, True)
            continue

        if raw_frame is None:
            break

        try:
            parsed = json.loads(
                raw_frame.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            _write_frame(out_stream, err_resp, framed)
            continue

        if not isinstance(parsed, Mapping):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request: expected JSON object"},
            }
            _write_frame(out_stream, err_resp, framed)
            continue

        response = handle_jsonrpc_message(parsed)
        if response is not None:
            _write_frame(out_stream, response, framed)

    return 0


def main() -> int:
    """CLI entrypoint for `python3 -m multimodal_decision_models.mcp_server`."""
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())

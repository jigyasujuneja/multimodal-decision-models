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
import json
import sys
from typing import Any, BinaryIO, Mapping

from .calibration import pre_gated_bayesian_poe_fusion
from .conformal import (
    DEFAULT_CONFORMAL_ABSTAIN_KEYS,
    calibrate_conformal_quantile,
    conformal_prediction_set,
)
from .context_pruner import ContextBlock, prune_agent_context_blocks
from .embedding_index import MultimodalEmbeddingIndex
from .visual_compiler import (
    ImageBuffer,
    compile_dual_buffer_contact_sheet,
    decode_png,
    decode_ppm,
)

MCP_SERVER_NAME = "dgem-catalog"
MCP_SERVER_VERSION = "0.4.0"
DEFAULT_MCP_PROTOCOL_VERSION = "2024-11-05"

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
                "q_hat": {"type": "number"},
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
                "prior_gate_mode": {"type": "string", "default": "hesitant"},
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


def _decode_image_spec(spec: Mapping[str, Any] | str) -> ImageBuffer:
    if isinstance(spec, str):
        raw_bytes = base64.b64decode(spec)
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return decode_png(raw_bytes)
        return decode_ppm(raw_bytes)

    if "png_base64" in spec and spec["png_base64"]:
        return decode_png(base64.b64decode(str(spec["png_base64"])))
    if "ppm_base64" in spec and spec["ppm_base64"]:
        return decode_ppm(base64.b64decode(str(spec["ppm_base64"])))
    if "width" in spec and "height" in spec:
        w = int(spec["width"])
        h = int(spec["height"])
        if "rgb_hex" in spec:
            rgb = bytes.fromhex(str(spec["rgb_hex"]))
        elif "rgb_base64" in spec:
            rgb = base64.b64decode(str(spec["rgb_base64"]))
        else:
            raise ValueError("raw ImageBuffer spec requires rgb_hex or rgb_base64")
        return ImageBuffer(w, h, rgb)
    raise ValueError("unsupported image specification; provide png_base64, ppm_base64, or width/height/rgb_hex")


def _tool_shortlist_candidates(args: Mapping[str, Any]) -> dict[str, Any]:
    items = args.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list")
    query_emb = args.get("query_embedding")
    if not isinstance(query_emb, list) or not query_emb:
        raise ValueError("query_embedding must be a non-empty float list")

    idx = MultimodalEmbeddingIndex()
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("each item in items must be an object")
        lab_raw = raw_item.get("lab_color")
        lab_t = (
            (float(lab_raw[0]), float(lab_raw[1]), float(lab_raw[2]))
            if isinstance(lab_raw, list) and len(lab_raw) == 3
            else None
        )
        idx.upsert(
            str(raw_item["id"]),
            [float(x) for x in raw_item["embedding"]],
            title=str(raw_item.get("title", "")),
            lab_color=lab_t,
            aspect_bin=str(raw_item.get("aspect_bin", "standard")),
            pack_size=int(raw_item["pack_size"]) if raw_item.get("pack_size") is not None else None,
            metadata={str(k): str(v) for k, v in (raw_item.get("metadata") or {}).items()},
        )

    q_lab_raw = args.get("query_lab_color")
    q_lab = (
        (float(q_lab_raw[0]), float(q_lab_raw[1]), float(q_lab_raw[2]))
        if isinstance(q_lab_raw, list) and len(q_lab_raw) == 3
        else None
    )
    res = idx.query(
        [float(x) for x in query_emb],
        query_lab_color=q_lab,
        query_aspect_bin=str(args["query_aspect_bin"]) if args.get("query_aspect_bin") else None,
        top_k=int(args.get("top_k", 6)),
        use_csls=bool(args.get("use_csls", False)),
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
    raw_probs_in = args.get("probabilities")
    if not isinstance(raw_probs_in, Mapping) or not raw_probs_in:
        raise ValueError("probabilities must be a non-empty object")
    probs = {str(k): float(v) for k, v in raw_probs_in.items()}

    method = str(args.get("method", "raps"))
    lam_reg = float(args.get("lam_reg", 0.01))
    k_reg = int(args.get("k_reg", 1))

    if "q_hat" in args and args["q_hat"] is not None:
        q_hat = float(args["q_hat"])
    elif "calib_probs" in args and "calib_labels" in args:
        c_probs = [{str(k): float(v) for k, v in row.items()} for row in args["calib_probs"]]
        c_labels = [str(lbl) for lbl in args["calib_labels"]]
        alpha = float(args.get("alpha", 0.10))
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
        raise ValueError("provide either q_hat or (calib_probs and calib_labels)")

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
        {str(k): float(v) for k, v in prior_logits_raw.items()}
        if isinstance(prior_logits_raw, Mapping)
        else None
    )
    prior_weight = float(args.get("prior_weight", 0.0))
    prior_gate_mode = str(args.get("prior_gate_mode", "hesitant"))
    hesitation_threshold = float(args.get("hesitation_threshold", 0.16))

    fused_probs, h_raw, hes_escalate = pre_gated_bayesian_poe_fusion(
        probs,
        prior_logits=prior_logits,
        prior_weight=prior_weight if prior_logits else 0.0,
        hesitation_threshold=hesitation_threshold,
        gate_mode=prior_gate_mode,
    )
    predicted = max(fused_probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return {
        "predicted": predicted,
        "probabilities": fused_probs,
        "raw_probabilities": probs,
        "h_raw": h_raw,
        "q_hat": q_hat,
        "prediction_set": conf_res.prediction_set,
        "singleton": conf_res.singleton,
        "contains_abstain": conf_res.contains_abstain,
        "conformal_escalate": conf_res.should_escalate,
        "hesitation_escalate": hes_escalate,
        "should_escalate": bool(conf_res.should_escalate or hes_escalate),
    }


def _tool_prune_context_blocks(args: Mapping[str, Any]) -> dict[str, Any]:
    raw_blocks = args.get("blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("blocks must be a list")
    blocks = [
        ContextBlock(
            id=str(b["id"]),
            full_tokens=int(b["full_tokens"]),
            stub_tokens=int(b["stub_tokens"]),
            full_text=str(b.get("full_text", "")),
            stub_text=str(b.get("stub_text", "")),
            pinned=bool(b.get("pinned", False)),
        )
        for b in raw_blocks
    ]
    raw_sp = args.get("slot_probabilities")
    if not isinstance(raw_sp, Mapping):
        raise ValueError("slot_probabilities must be an object")
    slot_probs = {
        str(bid): {str(act): float(p) for act, p in dist.items()}
        for bid, dist in raw_sp.items()
    }
    budget = int(args["token_budget"])
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
    if "query_png_base64" in args and args["query_png_base64"]:
        query_buf = _decode_image_spec(str(args["query_png_base64"]))
    elif "query" in args and isinstance(args["query"], Mapping):
        query_buf = _decode_image_spec(args["query"])
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
            }
        )
    else:
        raise ValueError("compile_contact_sheet requires query_png_base64 or query object")

    raw_cands = args.get("candidates")
    if not isinstance(raw_cands, list) or not raw_cands:
        raise ValueError("candidates must be a non-empty list")
    cand_bufs = [_decode_image_spec(c) for c in raw_cands]

    raw_meta = args.get("candidate_metadata")
    meta_list = (
        [{str(k): str(v) for k, v in m.items()} for m in raw_meta]
        if isinstance(raw_meta, list)
        else None
    )
    band_frac = (
        float(args["band_height_fraction"])
        if args.get("band_height_fraction") is not None
        else None
    )
    sheet = compile_dual_buffer_contact_sheet(
        query_buf,
        cand_bufs,
        candidate_metadata=meta_list,
        curved_container=bool(args.get("curved_container", False)),
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

    params = msg.get("params") or {}
    if not isinstance(params, Mapping):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": "Invalid params: expected object"},
        }

    if method == "initialize":
        proto_ver = str(params.get("protocolVersion") or DEFAULT_MCP_PROTOCOL_VERSION)
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
        tool_args = params.get("arguments") or {}
        if not isinstance(tool_name, str) or not isinstance(tool_args, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": "tools/call requires string name and object arguments"},
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
                    "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
                    "isError": False,
                },
            }
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method!r}"},
    }


def _read_next_frame(stream: BinaryIO) -> tuple[bytes | None, bool]:
    """Read next JSON-RPC payload from binary stream, returning `(raw_json_bytes, is_content_length_framed)`."""
    while True:
        line = stream.readline()
        if not line:
            return None, False
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length:"):
            parts = stripped.split(b":", 1)
            content_len = int(parts[1].strip())
            while True:
                hdr_line = stream.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break
            body = stream.read(content_len)
            if len(body) < content_len:
                return None, True
            return body, True
        return stripped, False


def _write_frame(stream: BinaryIO, response: Mapping[str, Any], use_content_length: bool) -> None:
    encoded = json.dumps(response, sort_keys=True).encode("utf-8")
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
        except ValueError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Framing error: {exc}"},
            }
            _write_frame(out_stream, err_resp, False)
            continue

        if raw_frame is None:
            break

        try:
            parsed = json.loads(raw_frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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

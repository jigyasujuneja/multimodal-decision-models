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

"""Pluggable Multimodal Embedding Index and Vertex AI `gemini-embedding-2` Wire Adapter.

Implements:
  - `VertexMultimodalEmbeddingAdapter`: Builds and parses Vertex AI `:predict` multimodal embedding
    wire payloads for `gemini-embedding-2` (text, PNG/JPEG image bytes, and optional Matryoshka
    `outputDimensionality`) using pure stdlib HTTP with bounded retries.
  - `MultimodalEmbeddingIndex`: Zero-dependency in-memory and JSONL-persisted catalog vector store
    combining residual-adapted cosine (or CSLS hubness-adjusted) similarity, Stage-2A inner-80%
    CIE L*a*b* color similarity, and Lidstone-smoothed container aspect priors with Stage-2A
    Adaptive Fast-Exit gating.
"""

from __future__ import annotations

import base64
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backends import _post_json_with_retry, _validate_http_url, _validate_resource_segment
from .calibration import stage2a_fast_exit_gate
from .contracts import SchemaContractViolationError
from .shortlist import cosine_similarity, csls_hubness_scores, split_identifier_tokens
from .visual_compiler import (
    ImageBuffer,
    apply_residual_mlp_adapter,
    mean_inner_crop_cie_lab,
    shades_of_gray_white_balance,
)


@dataclass(frozen=True)
class CatalogItem:
    """Indexed multimodal catalog item with dense vector, optional Lab color, and physical container traits."""

    id: str
    embedding: tuple[float, ...]
    title: str = ""
    lab_color: tuple[float, float, float] | None = None
    aspect_bin: str = "standard"
    pack_size: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingIndexQueryResult:
    """Shortlist ranking, calibrated Stage-2A priors, and adaptive fast-exit decision."""

    shortlist_ids: list[str]
    stage2a_energies: dict[str, float]
    stage2a_probabilities: dict[str, float]
    prior_logits: dict[str, float]
    fast_exit: bool
    fast_exit_id: str | None
    top1_margin: float
    z_score: float


def _normalize_vector(vec: Sequence[float], label: str = "embedding") -> tuple[float, ...]:
    if not vec:
        raise ValueError(f"{label} must be non-empty")
    vals = [float(v) for v in vec]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError(f"{label} contains non-finite values")
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 1e-12:
        raise ValueError(f"{label} L2 norm must be > 0")
    return tuple(v / norm for v in vals)


def _aspect_bin_from_buffer(buf: ImageBuffer) -> str:
    ratio = float(buf.height) / float(max(1, buf.width))
    if ratio >= 1.65:
        return "tall_cylinder"
    if ratio <= 0.72:
        return "wide_tub"
    return "standard"


class VertexMultimodalEmbeddingAdapter:
    """Zero-dependency wire adapter for Vertex AI `gemini-embedding-2` multimodal embeddings."""

    def __init__(
        self,
        project: str,
        model: str = "gemini-embedding-2",
        location: str = "us-central1",
        *,
        output_dimensionality: int | None = None,
        access_token_fn: Callable[[], str] | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        base_backoff_sec: float = 0.05,
        endpoint_override: str | None = None,
    ) -> None:
        self.project = _validate_resource_segment("project", project)
        self.model = _validate_resource_segment("model", model)
        self.location = _validate_resource_segment("location", location)
        if output_dimensionality is not None and output_dimensionality < 1:
            raise ValueError(f"output_dimensionality must be >= 1, got {output_dimensionality}")
        self.output_dimensionality = output_dimensionality
        self.access_token_fn = access_token_fn
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        if endpoint_override is not None:
            _validate_http_url(endpoint_override)
        self.endpoint_override = endpoint_override

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
            f"/publishers/google/models/{self.model}:predict"
        )

    def build_predict_payload(
        self,
        *,
        text: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        clean_text = text.strip() if isinstance(text, str) else ""
        has_image = image_bytes is not None and len(image_bytes) > 0
        if not clean_text and not has_image:
            raise ValueError("build_predict_payload requires non-empty text or image_bytes")

        instance: dict[str, Any] = {}
        if clean_text:
            instance["text"] = clean_text
        if has_image and image_bytes is not None:
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(image_bytes).decode("ascii"),
                "mimeType": image_mime_type,
            }

        payload: dict[str, Any] = {"instances": [instance]}
        if self.output_dimensionality is not None:
            payload["parameters"] = {"outputDimensionality": int(self.output_dimensionality)}
        return payload

    @staticmethod
    def parse_predict_response(data: Mapping[str, Any]) -> list[float]:
        raw_vec: Any = None
        preds = data.get("predictions")
        if isinstance(preds, Sequence) and len(preds) > 0 and isinstance(preds[0], Mapping):
            p0 = preds[0]
            for key in ("multimodalEmbedding", "imageEmbedding", "textEmbedding", "embedding"):
                if key in p0 and isinstance(p0[key], Sequence):
                    raw_vec = p0[key]
                    break
            if raw_vec is None and isinstance(p0.get("embeddings"), Mapping):
                vals = p0["embeddings"].get("values")
                if isinstance(vals, Sequence):
                    raw_vec = vals
        elif isinstance(data.get("embedding"), Mapping):
            vals = data["embedding"].get("values")
            if isinstance(vals, Sequence):
                raw_vec = vals

        if not isinstance(raw_vec, Sequence) or len(raw_vec) == 0:
            raise SchemaContractViolationError(
                "Vertex embedding response omitted non-empty embedding vector"
            )

        out: list[float] = []
        for idx, v in enumerate(raw_vec):
            fv = float(v)
            if not math.isfinite(fv):
                raise SchemaContractViolationError(f"non-finite embedding value at index {idx}")
            out.append(fv)
        norm = math.sqrt(sum(x * x for x in out))
        if norm <= 1e-12:
            raise SchemaContractViolationError("Vertex embedding response vector has zero L2 norm")
        return [x / norm for x in out]

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
            "VertexMultimodalEmbeddingAdapter requires access_token_fn or VERTEX_ACCESS_TOKEN / GOOGLE_OAUTH_ACCESS_TOKEN"
        )

    def embed(
        self,
        *,
        text: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/png",
    ) -> list[float]:
        url = self.endpoint_url()
        _validate_http_url(url)
        token = self._resolve_access_token()
        payload = self.build_predict_payload(
            text=text,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        resp = _post_json_with_retry(
            url,
            payload,
            headers,
            timeout_sec=self.timeout_sec,
            max_retries=self.max_retries,
            base_backoff_sec=self.base_backoff_sec,
        )
        return self.parse_predict_response(resp)


class MultimodalEmbeddingIndex:
    """In-memory and JSONL-persisted multimodal shortlist index with cosine, CIE Lab, and shape priors."""

    def __init__(
        self,
        *,
        w_cos: float = 0.80,
        w_lab: float = 0.15,
        w_shape: float = 0.05,
        inv_temperature: float = 28.0,
        lidstone_eps: float = 0.01,
        fast_exit_margin: float = 0.14,
        fast_exit_min_z: float = 1.65,
        adapter_w1: Sequence[Sequence[float]] | None = None,
        adapter_w2: Sequence[Sequence[float]] | None = None,
        adapter_alpha: float = 0.35,
    ) -> None:
        if w_cos < 0.0 or w_lab < 0.0 or w_shape < 0.0 or (w_cos + w_lab + w_shape) <= 0.0:
            raise ValueError("weights w_cos, w_lab, w_shape must be >= 0 with positive sum")
        if not math.isfinite(inv_temperature) or inv_temperature <= 0.0:
            raise ValueError(f"inv_temperature must be finite and > 0, got {inv_temperature!r}")
        if not math.isfinite(lidstone_eps) or lidstone_eps <= 0.0:
            raise ValueError(f"lidstone_eps must be finite and > 0, got {lidstone_eps!r}")

        self.w_cos = float(w_cos)
        self.w_lab = float(w_lab)
        self.w_shape = float(w_shape)
        self.inv_temperature = float(inv_temperature)
        self.lidstone_eps = float(lidstone_eps)
        self.fast_exit_margin = float(fast_exit_margin)
        self.fast_exit_min_z = float(fast_exit_min_z)
        self.adapter_w1 = [[float(x) for x in row] for row in adapter_w1] if adapter_w1 is not None else None
        self.adapter_w2 = [[float(x) for x in row] for row in adapter_w2] if adapter_w2 is not None else None
        self.adapter_alpha = float(adapter_alpha)

        self._items: dict[str, CatalogItem] = {}
        self._dim: int | None = None

    def __len__(self) -> int:
        return len(self._items)

    def get(self, item_id: str) -> CatalogItem | None:
        return self._items.get(item_id)

    def _build_catalog_item(
        self,
        item_id: str,
        embedding: Sequence[float],
        *,
        title: str = "",
        image: ImageBuffer | None = None,
        lab_color: tuple[float, float, float] | None = None,
        aspect_bin: str = "standard",
        pack_size: int | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> CatalogItem:
        clean_id = str(item_id).strip()
        if not clean_id:
            raise ValueError("item_id must be non-empty")
        norm_emb = _normalize_vector(embedding, label=f"embedding[{clean_id}]")
        if self._dim is not None and len(norm_emb) != self._dim:
            raise ValueError(
                f"embedding dimension mismatch for {clean_id!r}: expected {self._dim}, got {len(norm_emb)}"
            )

        resolved_lab: tuple[float, float, float] | None = None
        if lab_color is not None:
            if len(lab_color) != 3 or any(not math.isfinite(float(c)) for c in lab_color):
                raise ValueError(f"lab_color for {clean_id!r} must be a finite 3-tuple")
            resolved_lab = (float(lab_color[0]), float(lab_color[1]), float(lab_color[2]))
        elif image is not None:
            wb_img = shades_of_gray_white_balance(image, p=6.0)
            resolved_lab = mean_inner_crop_cie_lab(wb_img, crop_fraction=0.80)

        resolved_aspect = aspect_bin.strip() if aspect_bin.strip() else "standard"
        if image is not None and aspect_bin == "standard":
            resolved_aspect = _aspect_bin_from_buffer(image)

        return CatalogItem(
            id=clean_id,
            embedding=norm_emb,
            title=str(title),
            lab_color=resolved_lab,
            aspect_bin=resolved_aspect,
            pack_size=int(pack_size) if pack_size is not None else None,
            metadata={str(k): str(v) for k, v in (metadata or {}).items()},
        )

    def add(
        self,
        item_id: str,
        embedding: Sequence[float],
        *,
        title: str = "",
        image: ImageBuffer | None = None,
        lab_color: tuple[float, float, float] | None = None,
        aspect_bin: str = "standard",
        pack_size: int | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        clean_id = str(item_id).strip()
        if clean_id in self._items:
            raise ValueError(f"item_id {clean_id!r} already exists; call upsert() to overwrite")
        item = self._build_catalog_item(
            clean_id,
            embedding,
            title=title,
            image=image,
            lab_color=lab_color,
            aspect_bin=aspect_bin,
            pack_size=pack_size,
            metadata=metadata,
        )
        self._dim = len(item.embedding)
        self._items[item.id] = item

    def upsert(
        self,
        item_id: str,
        embedding: Sequence[float],
        *,
        title: str = "",
        image: ImageBuffer | None = None,
        lab_color: tuple[float, float, float] | None = None,
        aspect_bin: str = "standard",
        pack_size: int | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        item = self._build_catalog_item(
            item_id,
            embedding,
            title=title,
            image=image,
            lab_color=lab_color,
            aspect_bin=aspect_bin,
            pack_size=pack_size,
            metadata=metadata,
        )
        self._dim = len(item.embedding)
        self._items[item.id] = item

    def _compute_neighborhood_means(self, k_neighbors: int = 4) -> dict[str, float]:
        keys = sorted(self._items.keys())
        if len(keys) <= 1:
            return {k: 0.0 for k in keys}
        out: dict[str, float] = {}
        eff_k = min(k_neighbors, len(keys) - 1)
        for k_a in keys:
            emb_a = self._items[k_a].embedding
            sims = sorted(
                (cosine_similarity(emb_a, self._items[k_b].embedding) for k_b in keys if k_b != k_a),
                reverse=True,
            )
            out[k_a] = sum(sims[:eff_k]) / float(eff_k)
        return out

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        query_image: ImageBuffer | None = None,
        query_lab_color: tuple[float, float, float] | None = None,
        query_aspect_bin: str | None = None,
        top_k: int = 6,
        use_csls: bool = False,
    ) -> EmbeddingIndexQueryResult:
        if not self._items:
            raise ValueError("cannot query an empty MultimodalEmbeddingIndex")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        q_norm = _normalize_vector(query_embedding, label="query_embedding")
        if self._dim is not None and len(q_norm) != self._dim:
            raise ValueError(f"query_embedding dimension {len(q_norm)} != index dimension {self._dim}")

        if self.adapter_w1 is not None and self.adapter_w2 is not None:
            q_vec: Sequence[float] = apply_residual_mlp_adapter(
                q_norm,
                self.adapter_w1,
                self.adapter_w2,
                alpha=self.adapter_alpha,
            )
        else:
            q_vec = q_norm

        resolved_q_lab: tuple[float, float, float] | None = None
        if query_lab_color is not None:
            resolved_q_lab = (
                float(query_lab_color[0]),
                float(query_lab_color[1]),
                float(query_lab_color[2]),
            )
        elif query_image is not None:
            wb_q = shades_of_gray_white_balance(query_image, p=6.0)
            resolved_q_lab = mean_inner_crop_cie_lab(wb_q, crop_fraction=0.80)

        resolved_q_aspect = query_aspect_bin
        if resolved_q_aspect is None and query_image is not None:
            resolved_q_aspect = _aspect_bin_from_buffer(query_image)

        proto_map = {k: item.embedding for k, item in self._items.items()}
        if use_csls and len(self._items) > 2:
            neigh_means = self._compute_neighborhood_means()
            cos_scores = csls_hubness_scores(q_vec, proto_map, neighborhood_means=neigh_means)
        else:
            cos_scores = {k: cosine_similarity(q_vec, emb) for k, emb in proto_map.items()}

        aspect_counts: dict[str, int] = {}
        for item in self._items.values():
            aspect_counts[item.aspect_bin] = aspect_counts.get(item.aspect_bin, 0) + 1
        num_bins = max(2, len(aspect_counts))
        total_items = float(len(self._items))

        energies: dict[str, float] = {}
        for item_id, item in self._items.items():
            s_cos = cos_scores[item_id]

            if resolved_q_lab is not None and item.lab_color is not None:
                dl = resolved_q_lab[0] - item.lab_color[0]
                da = resolved_q_lab[1] - item.lab_color[1]
                db = resolved_q_lab[2] - item.lab_color[2]
                delta_e = math.sqrt(dl * dl + da * da + db * db)
                s_lab = math.exp(-delta_e / 25.0)
                eff_w_cos = self.w_cos
                eff_w_lab = self.w_lab
            else:
                s_lab = 0.0
                eff_w_cos = self.w_cos + self.w_lab
                eff_w_lab = 0.0

            if resolved_q_aspect is not None:
                bin_cnt = float(aspect_counts.get(item.aspect_bin, 0))
                p_lidstone = (bin_cnt + self.lidstone_eps) / (total_items + self.lidstone_eps * num_bins)
                match_bonus = 1.0 if item.aspect_bin == resolved_q_aspect else 0.25
                s_shape = match_bonus * (1.0 / (1.0 - math.log(max(1e-6, p_lidstone)) * 0.1))
                eff_w_shape = self.w_shape
            else:
                s_shape = 0.0
                eff_w_cos += self.w_shape
                eff_w_shape = 0.0

            w_sum = eff_w_cos + eff_w_lab + eff_w_shape
            energies[item_id] = (
                (eff_w_cos * s_cos) + (eff_w_lab * s_lab) + (eff_w_shape * s_shape)
            ) / w_sum

        ranked_ids = sorted(energies.keys(), key=lambda k: (-energies[k], k))
        shortlist_ids = ranked_ids[: min(top_k, len(ranked_ids))]

        short_energies = {k: energies[k] for k in shortlist_ids}
        scaled_logits = {k: short_energies[k] * self.inv_temperature for k in shortlist_ids}
        max_s = max(scaled_logits.values())
        lse = max_s + math.log(sum(math.exp(v - max_s) for v in scaled_logits.values()))
        prior_logits = {k: v - lse for k, v in scaled_logits.items()}
        probs = {k: math.exp(prior_logits[k]) for k in shortlist_ids}
        prob_tot = sum(probs.values())
        probs = {k: v / prob_tot for k, v in probs.items()}

        if len(shortlist_ids) >= 2:
            top1_id = shortlist_ids[0]
            top2_id = shortlist_ids[1]
            p1 = probs[top1_id]
            p2 = probs[top2_id]
            margin = p1 - p2
            p_vals = [probs[k] for k in shortlist_ids]
            mean_p = statistics.fmean(p_vals)
            std_p = statistics.pstdev(p_vals)
            z_val = ((p1 - mean_p) / std_p) if std_p > 1e-9 else 0.0

            item1 = self._items[top1_id]
            item2 = self._items[top2_id]
            sister_collision = False
            if (
                item1.pack_size is not None
                and item2.pack_size is not None
                and item1.pack_size != item2.pack_size
            ):
                toks1 = set(split_identifier_tokens(item1.title or item1.id))
                toks2 = set(split_identifier_tokens(item2.title or item2.id))
                if not toks1 or not toks2 or len(toks1 & toks2) > 0:
                    sister_collision = True

            fast_ok, fast_winner = stage2a_fast_exit_gate(
                probs,
                min_margin=self.fast_exit_margin,
                min_z=self.fast_exit_min_z,
                sister_pack_collision=sister_collision,
            )
        else:
            only_id = shortlist_ids[0]
            margin = 1.0
            z_val = 0.0
            fast_ok = False
            fast_winner = None

        return EmbeddingIndexQueryResult(
            shortlist_ids=shortlist_ids,
            stage2a_energies=short_energies,
            stage2a_probabilities=probs,
            prior_logits=prior_logits,
            fast_exit=fast_ok,
            fast_exit_id=fast_winner,
            top1_margin=margin,
            z_score=z_val,
        )

    def save_jsonl(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            json.dumps(
                {
                    "_record_type": "index_config",
                    "w_cos": self.w_cos,
                    "w_lab": self.w_lab,
                    "w_shape": self.w_shape,
                    "inv_temperature": self.inv_temperature,
                    "lidstone_eps": self.lidstone_eps,
                    "fast_exit_margin": self.fast_exit_margin,
                    "fast_exit_min_z": self.fast_exit_min_z,
                    "adapter_w1": self.adapter_w1,
                    "adapter_w2": self.adapter_w2,
                    "adapter_alpha": self.adapter_alpha,
                },
                sort_keys=True,
            )
        ]
        for item_id in sorted(self._items.keys()):
            item = self._items[item_id]
            lines.append(
                json.dumps(
                    {
                        "_record_type": "item",
                        "id": item.id,
                        "embedding": list(item.embedding),
                        "title": item.title,
                        "lab_color": list(item.lab_color) if item.lab_color is not None else None,
                        "aspect_bin": item.aspect_bin,
                        "pack_size": item.pack_size,
                        "metadata": item.metadata,
                    },
                    sort_keys=True,
                )
            )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> MultimodalEmbeddingIndex:
        in_path = Path(path)
        cfg_kwargs: dict[str, Any] = {}
        item_rows: list[dict[str, Any]] = []

        for raw_line in in_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, Mapping):
                raise ValueError("JSONL row in MultimodalEmbeddingIndex must be a JSON object")
            rtype = row.get("_record_type", "item")
            if rtype == "index_config":
                for key in (
                    "w_cos",
                    "w_lab",
                    "w_shape",
                    "inv_temperature",
                    "lidstone_eps",
                    "fast_exit_margin",
                    "fast_exit_min_z",
                    "adapter_w1",
                    "adapter_w2",
                    "adapter_alpha",
                ):
                    if key in row:
                        cfg_kwargs[key] = row[key]
            else:
                item_rows.append(dict(row))

        index = cls(**cfg_kwargs)
        for row in item_rows:
            lab_raw = row.get("lab_color")
            lab_tuple = (
                (float(lab_raw[0]), float(lab_raw[1]), float(lab_raw[2]))
                if isinstance(lab_raw, Sequence) and len(lab_raw) == 3
                else None
            )
            index.upsert(
                str(row["id"]),
                [float(x) for x in row["embedding"]],
                title=str(row.get("title", "")),
                lab_color=lab_tuple,
                aspect_bin=str(row.get("aspect_bin", "standard")),
                pack_size=int(row["pack_size"]) if row.get("pack_size") is not None else None,
                metadata=dict(row.get("metadata") or {}),
            )
        return index

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

"""Five-Rung & 2^4 Factorial Ablation Protocol, Split-Lock Hygiene, OOF Calibration, and Controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from .audio_video_decision import SpatioTemporalFrameCandidate
from .calibration import TemperatureScalingModel, VectorScalingModel, fit_temperature_scaling, fit_vector_scaling, normalized_entropy
from .conformal import ConformalCalibrationResult, ConformalSetResult, MondrianCalibrationResult, calibrate_conformal_quantile, calibrate_mondrian_conformal_quantiles, conformal_prediction_set, hmac_uint64, hmac_uniform_float64, mondrian_conformal_prediction_set
from .context_pruner import ContextBlock, ContextPruneResult
from .contracts import SchemaContractViolationError

R0_GENERATIVE_CEILING_ID: str = "R0"
FACTOR_NAMES: tuple[str, str, str, str] = ("C", "M", "K", "P")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_BIT4_RE = re.compile(r"^[01]{4}$")
_VALID_SPLITS: tuple[str, str, str] = ("cal", "val", "test")

_LADDER_ALIASES: dict[str, str] = {
    "0000": "R1",
    "1000": "R2",
    "1100": "R3",
    "1110": "R3_5",
    "1111": "R4",
}
_ROLE_LABELS: dict[str, str] = {
    "0000": "R1_0000",
    "1000": "R2_KI_C_1000",
    "0100": "KI_M_0100",
    "0010": "KI_K_0010",
    "0001": "KI_P_0001",
    "1100": "R3_SYN_CM_1100",
    "1010": "SYN_CK_1010",
    "1001": "SYN_CP_1001",
    "0110": "SYN_MK_0110",
    "0101": "SYN_MP_0101",
    "0011": "SYN_KP_0011",
    "0111": "LOO_noC_0111",
    "1011": "LOO_noM_1011",
    "1101": "LOO_noK_1101",
    "1110": "R3_5_LOO_noP_1110",
    "1111": "R4_1111",
}


@dataclass(frozen=True)
class FactorialConfig:
    """Binary factor assignment (C, M, K, P) for the MDM ablation cube."""

    calibrate_conformal: bool
    multimodal_compiler: bool
    context_trajectory_budget: bool
    pareto_bandit_router: bool

    def __post_init__(self) -> None:
        for field_name in (
            "calibrate_conformal",
            "multimodal_compiler",
            "context_trajectory_budget",
            "pareto_bandit_router",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, bool):
                raise SchemaContractViolationError(
                    f"FactorialConfig.{field_name} must be a bool, got {val!r}"
                )

    @property
    def bit_code(self) -> str:
        """Return the 4-character binary code 'CMKP' in {'0000'..'1111'}."""
        bits = (
            self.calibrate_conformal,
            self.multimodal_compiler,
            self.context_trajectory_budget,
            self.pareto_bandit_router,
        )
        return "".join("1" if flag else "0" for flag in bits)

    @property
    def active_factors(self) -> tuple[str, ...]:
        """Return ordered tuple of active factor codes among ('C', 'M', 'K', 'P')."""
        flags = (
            self.calibrate_conformal,
            self.multimodal_compiler,
            self.context_trajectory_budget,
            self.pareto_bandit_router,
        )
        return tuple(name for name, active in zip(FACTOR_NAMES, flags) if active)

    @property
    def hamming_weight(self) -> int:
        """Return the number of enabled factors in {0, 1, 2, 3, 4}."""
        return len(self.active_factors)

    @property
    def ladder_alias(self) -> str | None:
        """Return Five-Rung alias ('R1', 'R2', 'R3', 'R3_5', 'R4') when on the cumulative ladder."""
        return _LADDER_ALIASES.get(self.bit_code)

    @property
    def canonical_label(self) -> str:
        """Return canonical regime identifier combining role and 4-bit code."""
        return _ROLE_LABELS[self.bit_code]

    @classmethod
    def from_bit_code(cls, bit_code: str) -> FactorialConfig:
        """Construct a ``FactorialConfig`` from a 4-character binary string ('0000'..'1111').

        Args:
            bit_code: Four-character binary string ordered as 'CMKP'.

        Returns:
            Parsed ``FactorialConfig`` instance.

        Raises:
            SchemaContractViolationError: If ``bit_code`` is not a 4-character binary string.
        """
        if not isinstance(bit_code, str) or not _BIT4_RE.match(bit_code):
            raise SchemaContractViolationError(
                f"bit_code must be a 4-character binary string in '0000'..'1111', got {bit_code!r}"
            )
        return cls(
            calibrate_conformal=(bit_code[0] == "1"),
            multimodal_compiler=(bit_code[1] == "1"),
            context_trajectory_budget=(bit_code[2] == "1"),
            pareto_bandit_router=(bit_code[3] == "1"),
        )


def build_five_rung_ladder() -> tuple[FactorialConfig, ...]:
    """Build the 1-factor-per-step cumulative ladder R1(0000)->R2(1000)->R3(1100)->R3_5(1110)->R4(1111)."""
    return tuple(
        FactorialConfig.from_bit_code(code)
        for code in ("0000", "1000", "1100", "1110", "1111")
    )


def build_single_factor_knock_ins() -> dict[str, FactorialConfig]:
    """Build single-factor knock-in configurations onto R1 (1000, 0100, 0010, 0001)."""
    return {
        "KI_C": FactorialConfig.from_bit_code("1000"),
        "KI_M": FactorialConfig.from_bit_code("0100"),
        "KI_K": FactorialConfig.from_bit_code("0010"),
        "KI_P": FactorialConfig.from_bit_code("0001"),
    }


def build_loo_knockouts() -> dict[str, FactorialConfig]:
    """Build Leave-One-Out (LOO) knockout configurations from R4 (0111, 1011, 1101, 1110)."""
    return {
        "LOO_noC": FactorialConfig.from_bit_code("0111"),
        "LOO_noM": FactorialConfig.from_bit_code("1011"),
        "LOO_noK": FactorialConfig.from_bit_code("1101"),
        "LOO_noP": FactorialConfig.from_bit_code("1110"),
    }


def build_non_adjacent_pairwise_synergies() -> dict[str, FactorialConfig]:
    """Build non-adjacent pairwise synergy configurations (1010, 1001, 0110, 0101, 0011)."""
    return {
        "SYN_CK": FactorialConfig.from_bit_code("1010"),
        "SYN_CP": FactorialConfig.from_bit_code("1001"),
        "SYN_MK": FactorialConfig.from_bit_code("0110"),
        "SYN_MP": FactorialConfig.from_bit_code("0101"),
        "SYN_KP": FactorialConfig.from_bit_code("0011"),
    }


def build_orthogonal_l8_array() -> tuple[FactorialConfig, ...]:
    """Build the Resolution-IV 2^(4-1) orthogonal L8 array (even-parity defining relation)."""
    return tuple(
        FactorialConfig.from_bit_code(code)
        for code in ("0000", "1100", "1010", "1001", "0110", "0101", "0011", "1111")
    )


def build_full_factorial_cube() -> tuple[FactorialConfig, ...]:
    """Build all 2^4 = 16 factorial configurations ordered from '0000' to '1111'."""
    return tuple(FactorialConfig.from_bit_code(format(mask, "04b")) for mask in range(16))


def compute_episode_content_sha256(payload: Mapping[str, object] | str | bytes) -> str:
    """Compute a canonical SHA-256 hex digest for an episode payload (Gate G03)."""
    if isinstance(payload, bytes):
        raw_bytes = payload
    elif isinstance(payload, str):
        raw_bytes = payload.encode("utf-8")
    elif isinstance(payload, Mapping):
        canonical_json = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        raw_bytes = canonical_json.encode("utf-8")
    else:
        raise SchemaContractViolationError(
            f"unsupported episode payload type for SHA-256 hashing: {type(payload).__qualname__}"
        )
    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass(frozen=True)
class EpisodeSplitRecord:
    """Locked metadata record for a single multi-turn multimodal evaluation episode."""

    episode_id: str
    cluster_id: str
    track_id: str
    content_sha256: str
    num_turns: int = 6
    split: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise SchemaContractViolationError("episode_id must be a non-empty string")
        if not isinstance(self.cluster_id, str) or not self.cluster_id.strip():
            raise SchemaContractViolationError("cluster_id must be a non-empty string")
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise SchemaContractViolationError("track_id must be a non-empty string")
        if not isinstance(self.content_sha256, str) or not _HEX64_RE.match(self.content_sha256.lower()):
            raise SchemaContractViolationError(
                f"content_sha256 must be a 64-character hex SHA-256 digest, got {self.content_sha256!r}"
            )
        if isinstance(self.num_turns, bool) or not isinstance(self.num_turns, int) or self.num_turns < 1:
            raise SchemaContractViolationError(f"num_turns must be an integer >= 1, got {self.num_turns!r}")
        if self.split != "" and self.split not in _VALID_SPLITS:
            raise SchemaContractViolationError(
                f"split must be one of {_VALID_SPLITS} (or empty before assignment), got {self.split!r}"
            )


def verify_disjoint_episode_splits(records: Sequence[EpisodeSplitRecord]) -> dict[str, int]:
    """Verify Gate G03 (SHA-256 dedup) and Gate G13 (episode + cluster disjointness) across splits.

    Args:
        records: Sequence of assigned ``EpisodeSplitRecord`` instances.

    Returns:
        Dictionary mapping ``'cal'``, ``'val'``, and ``'test'`` to episode counts.

    Raises:
        SchemaContractViolationError: If any episode_id, cluster_id, or content_sha256 leaks or
            duplicates across splits, or if any split is empty.
    """
    if not records:
        raise SchemaContractViolationError("records must be non-empty")

    seen_episode_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    cluster_to_split: dict[str, str] = {}
    counts: dict[str, int] = {"cal": 0, "val": 0, "test": 0}

    for rec in records:
        if not isinstance(rec, EpisodeSplitRecord):
            raise SchemaContractViolationError("all entries must be EpisodeSplitRecord instances")
        if rec.split not in _VALID_SPLITS:
            raise SchemaContractViolationError(
                f"episode {rec.episode_id!r} has unassigned or invalid split {rec.split!r}"
            )

        prev_ep_split = seen_episode_ids.get(rec.episode_id)
        if prev_ep_split is not None:
            raise SchemaContractViolationError(
                f"duplicate episode_id {rec.episode_id!r} across splits ({prev_ep_split!r} vs {rec.split!r})"
            )
        seen_episode_ids[rec.episode_id] = rec.split

        digest_lower = rec.content_sha256.lower()
        prev_hash_ep = seen_hashes.get(digest_lower)
        if prev_hash_ep is not None:
            raise SchemaContractViolationError(
                f"G03 SHA-256 duplicate content leak between {prev_hash_ep!r} and {rec.episode_id!r}"
            )
        seen_hashes[digest_lower] = rec.episode_id

        bound_split = cluster_to_split.get(rec.cluster_id)
        if bound_split is not None and bound_split != rec.split:
            raise SchemaContractViolationError(
                f"G13 cluster leakage: cluster_id {rec.cluster_id!r} spans {bound_split!r} and {rec.split!r}"
            )
        cluster_to_split[rec.cluster_id] = rec.split
        counts[rec.split] += 1

    for split_name, split_count in counts.items():
        if split_count <= 0:
            raise SchemaContractViolationError(f"required split {split_name!r} has zero assigned episodes")

    return counts


def _compute_manifest_sha256(records: Sequence[EpisodeSplitRecord]) -> str:
    canonical_rows = [
        {
            "episode_id": rec.episode_id,
            "cluster_id": rec.cluster_id,
            "track_id": rec.track_id,
            "content_sha256": rec.content_sha256.lower(),
            "num_turns": rec.num_turns,
            "split": rec.split,
        }
        for rec in sorted(records, key=lambda item: item.episode_id)
    ]
    serialized = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSplitLock:
    """Immutable SHA-256 split manifest locking 30% D_cal / 20% D_val / 50% D_test assignments."""

    benchmark_id: str
    version: str
    seed_key: str
    records: tuple[EpisodeSplitRecord, ...]
    manifest_sha256: str
    deduplicated_count: int = 0

    def split_counts(self) -> dict[str, int]:
        """Validate and return episode counts per split."""
        return verify_disjoint_episode_splits(self.records)

    def episodes_for_split(self, split: str) -> tuple[EpisodeSplitRecord, ...]:
        """Return ordered tuple of records belonging to ``split`` ('cal', 'val', or 'test')."""
        if split not in _VALID_SPLITS:
            raise SchemaContractViolationError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
        return tuple(rec for rec in self.records if rec.split == split)

    def verify(self, live_records: Sequence[EpisodeSplitRecord] | None = None) -> dict[str, int]:
        """Verify internal G03/G13 disjointness, manifest SHA-256, and optional live corpus match.

        Args:
            live_records: Optional live episode records to verify against this committed lockfile.

        Returns:
            Verified episode count map ``{'cal': ..., 'val': ..., 'test': ...}``.

        Raises:
            SchemaContractViolationError: On manifest tampering, split leakage, or live mismatch.
        """
        counts = verify_disjoint_episode_splits(self.records)
        expected_manifest = _compute_manifest_sha256(self.records)
        if self.manifest_sha256 != expected_manifest:
            raise SchemaContractViolationError(
                f"lockfile manifest_sha256 mismatch: expected {expected_manifest}, got {self.manifest_sha256}"
            )
        if live_records is not None:
            if len(live_records) != len(self.records):
                raise SchemaContractViolationError(
                    f"live episode count {len(live_records)} does not match lockfile count {len(self.records)}"
                )
            locked_by_id = {rec.episode_id: rec for rec in self.records}
            for live_rec in live_records:
                locked_rec = locked_by_id.get(live_rec.episode_id)
                if locked_rec is None:
                    raise SchemaContractViolationError(
                        f"live episode {live_rec.episode_id!r} absent from DatasetSplitLock"
                    )
                if (
                    locked_rec.cluster_id != live_rec.cluster_id
                    or locked_rec.track_id != live_rec.track_id
                    or locked_rec.content_sha256.lower() != live_rec.content_sha256.lower()
                    or locked_rec.num_turns != live_rec.num_turns
                ):
                    raise SchemaContractViolationError(
                        f"live episode {live_rec.episode_id!r} metadata/SHA-256 diverges from DatasetSplitLock"
                    )
                if live_rec.split and live_rec.split != locked_rec.split:
                    raise SchemaContractViolationError(
                        f"live episode {live_rec.episode_id!r} split {live_rec.split!r} != locked {locked_rec.split!r}"
                    )
        return counts

    def to_dict(self) -> dict[str, object]:
        """Serialize this lockfile into a deterministic JSON-compatible dictionary."""
        counts = self.split_counts()
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "seed_key": self.seed_key,
            "manifest_sha256": self.manifest_sha256,
            "deduplicated_count": self.deduplicated_count,
            "split_counts": counts,
            "total_episodes": len(self.records),
            "total_turns": sum(rec.num_turns for rec in self.records),
            "records": [
                {
                    "episode_id": rec.episode_id,
                    "cluster_id": rec.cluster_id,
                    "track_id": rec.track_id,
                    "content_sha256": rec.content_sha256.lower(),
                    "num_turns": rec.num_turns,
                    "split": rec.split,
                }
                for rec in sorted(self.records, key=lambda item: item.episode_id)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetSplitLock:
        """Deserialize and immediately verify a ``DatasetSplitLock`` dictionary."""
        if not isinstance(payload, Mapping):
            raise SchemaContractViolationError("DatasetSplitLock payload must be a mapping")
        raw_records = payload.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise SchemaContractViolationError("DatasetSplitLock 'records' must be a list")
        parsed_records: list[EpisodeSplitRecord] = []
        for entry in raw_records:
            if not isinstance(entry, Mapping):
                raise SchemaContractViolationError("each lockfile record must be a mapping")
            parsed_records.append(
                EpisodeSplitRecord(
                    episode_id=str(entry["episode_id"]),
                    cluster_id=str(entry["cluster_id"]),
                    track_id=str(entry["track_id"]),
                    content_sha256=str(entry["content_sha256"]).lower(),
                    num_turns=int(entry.get("num_turns", 6)),
                    split=str(entry["split"]),
                )
            )
        ordered = tuple(sorted(parsed_records, key=lambda item: item.episode_id))
        lock = cls(
            benchmark_id=str(payload["benchmark_id"]),
            version=str(payload["version"]),
            seed_key=str(payload["seed_key"]),
            records=ordered,
            manifest_sha256=str(payload["manifest_sha256"]),
            deduplicated_count=int(payload.get("deduplicated_count", 0)),
        )
        lock.verify()
        return lock

    def save_json(self, path: str | Path) -> None:
        """Write deterministic JSON lockfile to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        target.write_text(serialized, encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> DatasetSplitLock:
        """Load and cryptographically verify a JSON lockfile from ``path``."""
        target = Path(path)
        raw_obj = json.loads(target.read_text(encoding="utf-8"))
        return cls.from_dict(raw_obj)


def partition_episodes_cluster_disjoint(
    records: Sequence[EpisodeSplitRecord],
    *,
    cal_fraction: float = 0.30,
    val_fraction: float = 0.20,
    test_fraction: float = 0.50,
    benchmark_id: str = "MDM-EpisodeBench",
    version: str = "1.0.0",
    seed_key: str = "mdm-episodebench-split-v1",
    deduplicate_sha256: bool = True,
) -> DatasetSplitLock:
    """Partition episodes into cluster-disjoint 30% D_cal / 20% D_val / 50% D_test splits.

    Args:
        records: Input ``EpisodeSplitRecord`` sequence (split field is overwritten cleanly).
        cal_fraction: Target calibration split proportion (default 0.30).
        val_fraction: Target validation split proportion (default 0.20).
        test_fraction: Target locked test split proportion (default 0.50).
        benchmark_id: Canonical benchmark identifier stored in the lockfile.
        version: Manifest schema/corpus version string.
        seed_key: Deterministic HMAC key governing cluster ordering.
        deduplicate_sha256: When True, drops duplicate content_sha256 records deterministically
            (Gate G03); when False, raises ``SchemaContractViolationError`` on duplicate hashes.

    Returns:
        Verified ``DatasetSplitLock`` instance.

    Raises:
        SchemaContractViolationError: On invalid fractions, duplicate episode_ids, cross-track
            cluster collisions, or insufficient clusters for 3-way partitioning.
    """
    for frac_name, frac_val in (
        ("cal_fraction", cal_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        if isinstance(frac_val, bool) or not isinstance(frac_val, (int, float)) or float(frac_val) <= 0.0:
            raise SchemaContractViolationError(f"{frac_name} must be > 0, got {frac_val!r}")
    total_frac = float(cal_fraction) + float(val_fraction) + float(test_fraction)
    if abs(total_frac - 1.0) > 1e-6:
        raise SchemaContractViolationError(
            f"cal_fraction + val_fraction + test_fraction must equal 1.0, got {total_frac:.6f}"
        )
    if not records:
        raise SchemaContractViolationError("records must be non-empty")

    ordered_input = sorted(records, key=lambda item: item.episode_id)
    seen_ep_ids: set[str] = set()
    seen_hashes: set[str] = set()
    deduped: list[EpisodeSplitRecord] = []
    dropped_duplicates = 0

    for rec in ordered_input:
        if rec.episode_id in seen_ep_ids:
            raise SchemaContractViolationError(f"duplicate episode_id {rec.episode_id!r} in input")
        seen_ep_ids.add(rec.episode_id)
        digest_lower = rec.content_sha256.lower()
        if digest_lower in seen_hashes:
            if not deduplicate_sha256:
                raise SchemaContractViolationError(
                    f"duplicate content_sha256 detected on episode {rec.episode_id!r} (G03)"
                )
            dropped_duplicates += 1
            continue
        seen_hashes.add(digest_lower)
        deduped.append(rec)

    clusters_map: dict[str, list[EpisodeSplitRecord]] = {}
    for rec in deduped:
        clusters_map.setdefault(rec.cluster_id, []).append(rec)

    if len(clusters_map) < 3:
        raise SchemaContractViolationError(
            f"at least 3 distinct clusters are required for 3-way disjoint split, got {len(clusters_map)}"
        )

    total_eps = len(deduped)
    target_cal = max(1, int(round(total_eps * float(cal_fraction))))
    target_val = max(1, int(round(total_eps * float(val_fraction))))
    target_test = max(1, total_eps - target_cal - target_val)

    by_primary_track: dict[str, list[tuple[str, list[EpisodeSplitRecord]]]] = {}
    for cid, c_eps in clusters_map.items():
        primary_track = sorted({ep.track_id for ep in c_eps})[0]
        by_primary_track.setdefault(primary_track, []).append((cid, c_eps))

    interleaved_clusters: list[tuple[str, list[EpisodeSplitRecord]]] = []
    sorted_tracks = sorted(by_primary_track.keys())
    per_track_ranked: list[list[tuple[str, list[EpisodeSplitRecord]]]] = []
    for track_name in sorted_tracks:
        ranked_group = sorted(
            by_primary_track[track_name],
            key=lambda pair: (hmac_uint64(seed_key, pair[0], f"split:{track_name}"), pair[0]),
        )
        per_track_ranked.append(ranked_group)

    max_len = max(len(grp) for grp in per_track_ranked)
    for pos in range(max_len):
        for grp in per_track_ranked:
            if pos < len(grp):
                interleaved_clusters.append(grp[pos])

    split_targets = {"cal": target_cal, "val": target_val, "test": target_test}
    split_accum = {"cal": 0, "val": 0, "test": 0}
    split_cluster_counts = {"cal": 0, "val": 0, "test": 0}
    assigned_records: list[EpisodeSplitRecord] = []

    total_clusters = len(interleaved_clusters)
    for idx, (_cid, c_eps) in enumerate(interleaved_clusters):
        remaining_clusters_after = total_clusters - idx - 1
        empty_splits = [sp for sp in _VALID_SPLITS if split_cluster_counts[sp] == 0]
        if empty_splits and len(empty_splits) > remaining_clusters_after:
            chosen_split = empty_splits[0]
        else:
            def _deficit_score(sp_name: str, batch_size: int = len(c_eps)) -> tuple[float, int]:
                projected = split_accum[sp_name] + batch_size
                target = split_targets[sp_name]
                if split_accum[sp_name] < target and projected <= target:
                    rel_fill = float(split_accum[sp_name]) / float(target)
                    return (rel_fill, 0)
                overshoot = abs(projected - target) / float(target)
                return (1.0 + overshoot, 1)

            chosen_split = min(_VALID_SPLITS, key=_deficit_score)

        split_accum[chosen_split] += len(c_eps)
        split_cluster_counts[chosen_split] += 1
        for ep in c_eps:
            assigned_records.append(
                EpisodeSplitRecord(
                    episode_id=ep.episode_id,
                    cluster_id=ep.cluster_id,
                    track_id=ep.track_id,
                    content_sha256=ep.content_sha256.lower(),
                    num_turns=ep.num_turns,
                    split=chosen_split,
                )
            )

    ordered_assigned = tuple(sorted(assigned_records, key=lambda item: item.episode_id))
    verify_disjoint_episode_splits(ordered_assigned)
    manifest_digest = _compute_manifest_sha256(ordered_assigned)
    return DatasetSplitLock(
        benchmark_id=benchmark_id,
        version=version,
        seed_key=seed_key,
        records=ordered_assigned,
        manifest_sha256=manifest_digest,
        deduplicated_count=dropped_duplicates,
    )


@dataclass(frozen=True)
class OOFCalibratedConformalBundle:
    """5-fold out-of-fold calibrated post-processor + split/Mondrian conformal thresholds."""

    scaling_method: str
    conformal_method: str
    alpha: float
    num_folds: int
    post_processor: TemperatureScalingModel | VectorScalingModel
    oof_calibrated_probs: tuple[dict[str, float], ...]
    conformal_calibration: ConformalCalibrationResult
    mondrian_calibration: MondrianCalibrationResult | None
    in_sample_resubstituted_q_hat: float

    @property
    def q_hat_global(self) -> float:
        """Return the OOF-calibrated global conformal nonconformity quantile."""
        return self.conformal_calibration.q_hat

    def predict_proba(
        self,
        probs_or_logits: Sequence[Mapping[str, float]],
        *,
        is_logits: bool = False,
    ) -> list[dict[str, float]]:
        """Apply the full-D_cal post-hoc temperature/vector scaling model to new inputs."""
        return self.post_processor.predict_proba(probs_or_logits, is_logits=is_logits)

    def predict_set(
        self,
        raw_probs_or_logits: Mapping[str, float],
        *,
        is_logits: bool = False,
        group: str | None = None,
        use_mondrian: bool = True,
    ) -> ConformalSetResult:
        """Compute a calibrated conformal prediction set on a single unseen example."""
        cal_dist = self.post_processor.predict_proba([raw_probs_or_logits], is_logits=is_logits)[0]
        if use_mondrian and self.mondrian_calibration is not None:
            return mondrian_conformal_prediction_set(
                cal_dist,
                self.mondrian_calibration,
                group=group,
            )
        return conformal_prediction_set(
            cal_dist,
            self.conformal_calibration.q_hat,
            method=self.conformal_method,
            lam_reg=self.conformal_calibration.lam_reg,
            k_reg=self.conformal_calibration.k_reg,
            randomized=False,
        )


def fit_oof_calibrated_conformal_bundle(
    calib_probs_or_logits: Sequence[Mapping[str, float]],
    calib_labels: Sequence[str],
    *,
    calib_groups: Sequence[str] | None = None,
    cluster_ids: Sequence[str] | None = None,
    is_logits: bool = False,
    scaling_method: str = "vector",
    l2_reg: float = 1e-3,
    alpha: float = 0.10,
    conformal_method: str = "raps",
    lam_reg: float = 0.01,
    k_reg: int = 1,
    num_folds: int = 5,
    seed_key: str = "mdm-oof-calibration-v1",
) -> OOFCalibratedConformalBundle:
    """Fit 5-fold out-of-fold temperature/vector scaling on D_cal prior to conformal quantiles.

    Computes out-of-fold calibrated probabilities across ``D_cal`` so conformal nonconformity
    scores never suffer from in-sample resubstitution sharpness bias, then fits the deployment
    scaling model on full ``D_cal``.

    Args:
        calib_probs_or_logits: Sequence of D_cal probability or logit dicts.
        calib_labels: Ground-truth labels aligned with ``calib_probs_or_logits``.
        calib_groups: Optional group/stratum labels for Mondrian conformal thresholds.
        cluster_ids: Optional episode/domain cluster IDs kept intact within OOF folds.
        is_logits: Whether ``calib_probs_or_logits`` contains unnormalized logits.
        scaling_method: Post-hoc calibrator ('temperature' or 'vector').
        l2_reg: L2 regularization weight for temperature/vector scaling.
        alpha: Target miscoverage rate in (0, 1).
        conformal_method: Nonconformity score ('lac', 'aps', or 'raps').
        lam_reg: RAPS penalty weight.
        k_reg: RAPS unpenalized rank cutoff.
        num_folds: Number of cross-fitting folds inside D_cal (default 5, minimum 2).
        seed_key: Deterministic HMAC key for fold assignment.

    Returns:
        ``OOFCalibratedConformalBundle`` containing OOF conformal quantiles and full scaling model.

    Raises:
        SchemaContractViolationError: If inputs have invalid lengths, folds, or scaling method.
    """
    n_items = len(calib_probs_or_logits)
    if n_items < 4 or n_items != len(calib_labels):
        raise SchemaContractViolationError(
            "calib_probs_or_logits and calib_labels must have matching length >= 4"
        )
    if isinstance(num_folds, bool) or not isinstance(num_folds, int) or num_folds < 2 or num_folds > n_items:
        raise SchemaContractViolationError(
            f"num_folds must be an integer in [2, {n_items}], got {num_folds!r}"
        )
    if scaling_method not in ("temperature", "vector"):
        raise SchemaContractViolationError(
            f"scaling_method must be 'temperature' or 'vector', got {scaling_method!r}"
        )
    if calib_groups is not None and len(calib_groups) != n_items:
        raise SchemaContractViolationError("calib_groups must match calib_labels length")
    if cluster_ids is not None and len(cluster_ids) != n_items:
        raise SchemaContractViolationError("cluster_ids must match calib_labels length")

    def _fit_scaler(
        train_rows: Sequence[Mapping[str, float]],
        train_lbls: Sequence[str],
    ) -> TemperatureScalingModel | VectorScalingModel:
        if scaling_method == "temperature":
            return fit_temperature_scaling(
                train_rows, train_lbls, is_logits=is_logits, l2_reg=l2_reg
            )
        return fit_vector_scaling(train_rows, train_lbls, is_logits=is_logits, l2_reg=l2_reg)

    resolved_clusters = (
        [str(cid) for cid in cluster_ids]
        if cluster_ids is not None
        else [f"item_{idx:06d}" for idx in range(n_items)]
    )
    cluster_to_indices: dict[str, list[int]] = {}
    for idx, cid in enumerate(resolved_clusters):
        cluster_to_indices.setdefault(cid, []).append(idx)

    if len(cluster_to_indices) < num_folds:
        raise SchemaContractViolationError(
            f"need at least num_folds={num_folds} distinct clusters in D_cal, got {len(cluster_to_indices)}"
        )

    ranked_clusters = sorted(
        cluster_to_indices.keys(),
        key=lambda cid: (hmac_uint64(seed_key, cid, "oof_fold"), cid),
    )
    fold_buckets: list[list[int]] = [[] for _ in range(num_folds)]
    for c_idx, cid in enumerate(ranked_clusters):
        target_fold = c_idx % num_folds
        fold_buckets[target_fold].extend(cluster_to_indices[cid])

    oof_predictions: list[dict[str, float] | None] = [None] * n_items
    for fold_idx in range(num_folds):
        heldout_set = set(fold_buckets[fold_idx])
        train_rows = [calib_probs_or_logits[i] for i in range(n_items) if i not in heldout_set]
        train_lbls = [str(calib_labels[i]) for i in range(n_items) if i not in heldout_set]
        heldout_indices = fold_buckets[fold_idx]
        heldout_rows = [calib_probs_or_logits[i] for i in heldout_indices]

        fold_scaler = _fit_scaler(train_rows, train_lbls)
        fold_preds = fold_scaler.predict_proba(heldout_rows, is_logits=is_logits)
        for item_pos, pred_dist in zip(heldout_indices, fold_preds):
            oof_predictions[item_pos] = pred_dist

    final_oof_probs: list[dict[str, float]] = []
    for pos, maybe_dist in enumerate(oof_predictions):
        if maybe_dist is None:
            raise SchemaContractViolationError(f"internal OOF assignment missed index {pos}")
        final_oof_probs.append(maybe_dist)

    clean_labels = [str(lbl) for lbl in calib_labels]
    oof_conformal = calibrate_conformal_quantile(
        final_oof_probs,
        clean_labels,
        alpha=alpha,
        method=conformal_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        randomized=False,
    )
    oof_mondrian = calibrate_mondrian_conformal_quantiles(
        final_oof_probs,
        clean_labels,
        calib_groups=calib_groups,
        alpha=alpha,
        method=conformal_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
    )

    full_scaler = _fit_scaler(calib_probs_or_logits, clean_labels)
    resubstituted_probs = full_scaler.predict_proba(calib_probs_or_logits, is_logits=is_logits)
    resubstituted_conformal = calibrate_conformal_quantile(
        resubstituted_probs,
        clean_labels,
        alpha=alpha,
        method=conformal_method,
        lam_reg=lam_reg,
        k_reg=k_reg,
        randomized=False,
    )

    return OOFCalibratedConformalBundle(
        scaling_method=scaling_method,
        conformal_method=oof_conformal.method,
        alpha=float(alpha),
        num_folds=num_folds,
        post_processor=full_scaler,
        oof_calibrated_probs=tuple(final_oof_probs),
        conformal_calibration=oof_conformal,
        mondrian_calibration=oof_mondrian,
        in_sample_resubstituted_q_hat=resubstituted_conformal.q_hat,
    )


@dataclass(frozen=True)
class RandomEscalationControl:
    """Control C1: Iso-escalation deterministic HMAC-seeded random gate."""

    target_escalation_rate: float
    seed_key: str = "mdm-control-c1-random-escalation"

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_escalation_rate, bool)
            or not isinstance(self.target_escalation_rate, (int, float))
            or not math.isfinite(float(self.target_escalation_rate))
            or not (0.0 <= float(self.target_escalation_rate) <= 1.0)
        ):
            raise SchemaContractViolationError(
                f"target_escalation_rate must lie in [0, 1], got {self.target_escalation_rate!r}"
            )
        if not isinstance(self.seed_key, str) or not self.seed_key.strip():
            raise SchemaContractViolationError("seed_key must be a non-empty string")

    def should_escalate(self, item_key: str) -> bool:
        """Return stateless deterministic Bernoulli(target_escalation_rate) draw for ``item_key``."""
        u_val = hmac_uniform_float64(self.seed_key, str(item_key), "c1_escalate")
        return u_val < float(self.target_escalation_rate)

    def select_exact_quota(self, item_keys: Sequence[str]) -> dict[str, bool]:
        """Select exact round(N * target_escalation_rate) items via HMAC rank order."""
        unique_keys = [str(key) for key in item_keys]
        if len(set(unique_keys)) != len(unique_keys):
            raise SchemaContractViolationError("item_keys must be unique for exact quota matching")
        n_total = len(unique_keys)
        if n_total == 0:
            return {}
        quota = min(
            n_total,
            max(0, int(round(float(n_total) * float(self.target_escalation_rate)))),
        )
        ranked = sorted(
            unique_keys,
            key=lambda key: (hmac_uniform_float64(self.seed_key, key, "c1_quota"), key),
        )
        chosen_set = set(ranked[:quota])
        return {key: (key in chosen_set) for key in unique_keys}


@dataclass(frozen=True)
class MatchedRateEntropyControl:
    """Control C2: Uncalibrated raw normalized_entropy gate tuned on D_val to match spend/rate."""

    entropy_threshold: float
    tie_fraction: float = 0.0
    target_escalation_rate: float = 0.0
    seed_key: str = "mdm-control-c2-entropy-gate"

    def __post_init__(self) -> None:
        if (
            isinstance(self.entropy_threshold, bool)
            or not isinstance(self.entropy_threshold, (int, float))
            or not math.isfinite(float(self.entropy_threshold))
        ):
            raise SchemaContractViolationError("entropy_threshold must be finite")
        if (
            isinstance(self.tie_fraction, bool)
            or not isinstance(self.tie_fraction, (int, float))
            or not (0.0 <= float(self.tie_fraction) <= 1.0)
        ):
            raise SchemaContractViolationError("tie_fraction must lie in [0, 1]")

    @classmethod
    def fit_on_validation(
        cls,
        val_raw_probs: Sequence[Mapping[str, float]],
        *,
        target_escalation_rate: float | None = None,
        target_episode_cost_usd: float | None = None,
        stage1_cost_usd: float = 0.0001,
        stage2_incremental_cost_usd: float = 0.0020,
        seed_key: str = "mdm-control-c2-entropy-gate",
    ) -> MatchedRateEntropyControl:
        """Fit raw H_norm threshold on D_val to match a target escalation rate or episode spend."""
        if not val_raw_probs:
            raise SchemaContractViolationError("val_raw_probs must be non-empty")
        if target_escalation_rate is None and target_episode_cost_usd is None:
            raise SchemaContractViolationError(
                "provide either target_escalation_rate or target_episode_cost_usd"
            )
        if target_escalation_rate is not None:
            rate = float(target_escalation_rate)
        else:
            assert target_episode_cost_usd is not None
            if stage2_incremental_cost_usd <= 0.0:
                raise SchemaContractViolationError("stage2_incremental_cost_usd must be > 0")
            rate = (float(target_episode_cost_usd) - float(stage1_cost_usd)) / float(
                stage2_incremental_cost_usd
            )
        clamped_rate = min(1.0, max(0.0, rate))
        n_val = len(val_raw_probs)
        target_count = int(round(float(n_val) * clamped_rate))
        if target_count <= 0:
            return cls(
                entropy_threshold=1.0 + 1e-9,
                tie_fraction=0.0,
                target_escalation_rate=clamped_rate,
                seed_key=seed_key,
            )
        if target_count >= n_val:
            return cls(
                entropy_threshold=-1e-9,
                tie_fraction=1.0,
                target_escalation_rate=clamped_rate,
                seed_key=seed_key,
            )

        entropies_desc = sorted((normalized_entropy(dist) for dist in val_raw_probs), reverse=True)
        tau_star = entropies_desc[target_count - 1]
        strictly_above = sum(1 for h_val in entropies_desc if h_val > tau_star + 1e-12)
        tied_at_tau = sum(1 for h_val in entropies_desc if abs(h_val - tau_star) <= 1e-12)
        needed_from_ties = max(0, target_count - strictly_above)
        tie_frac = (
            min(1.0, float(needed_from_ties) / float(tied_at_tau)) if tied_at_tau > 0 else 0.0
        )
        return cls(
            entropy_threshold=tau_star,
            tie_fraction=tie_frac,
            target_escalation_rate=clamped_rate,
            seed_key=seed_key,
        )

    def should_escalate(self, raw_probs: Mapping[str, float], item_key: str = "item") -> bool:
        """Evaluate whether raw normalized entropy exceeds the validation-tuned threshold."""
        h_val = normalized_entropy(raw_probs)
        if h_val > self.entropy_threshold + 1e-12:
            return True
        if abs(h_val - self.entropy_threshold) <= 1e-12 and self.tie_fraction > 0.0:
            u_draw = hmac_uniform_float64(self.seed_key, str(item_key), "c2_tie")
            return u_draw < self.tie_fraction
        return False


@dataclass(frozen=True)
class RecencyWindowControl:
    """Control C3: Sliding-window oldest-first truncation matching an exact token budget."""

    utility_fn: Callable[[ContextBlock, str], float] | None = None

    def prune(
        self,
        blocks: Sequence[ContextBlock],
        token_budget: int,
        slot_probabilities: Mapping[str, Mapping[str, float]] | None = None,
    ) -> ContextPruneResult:
        """Retain pinned blocks plus most-recent blocks (right-to-left) up to ``token_budget``."""
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
            raise SchemaContractViolationError(
                f"token_budget must be a positive integer, got {token_budget!r}"
            )
        if not blocks:
            return ContextPruneResult(
                decisions={},
                tokens_before=0,
                tokens_after=0,
                token_reduction_ratio=0.0,
                expected_utility=0.0,
                rendered_blocks=[],
            )

        tokens_before = sum(int(blk.full_tokens) for blk in blocks)
        decisions: dict[str, str] = {blk.id: "drop" for blk in blocks}
        used_tokens = 0

        for blk in blocks:
            if blk.pinned:
                if used_tokens + blk.full_tokens <= token_budget:
                    decisions[blk.id] = "full"
                    used_tokens += blk.full_tokens
                else:
                    decisions[blk.id] = "stub"
                    used_tokens += blk.stub_tokens

        if used_tokens > token_budget:
            raise SchemaContractViolationError(
                f"pinned context blocks require {used_tokens} tokens, exceeding budget {token_budget}"
            )

        for blk in reversed(blocks):
            if blk.pinned:
                continue
            if used_tokens + blk.full_tokens <= token_budget:
                decisions[blk.id] = "full"
                used_tokens += blk.full_tokens

        expected_utility = 0.0
        rendered_blocks: list[str] = []
        for blk in blocks:
            chosen = decisions[blk.id]
            if chosen == "full":
                rendered_blocks.append(blk.full_text or blk.id)
            elif chosen == "stub":
                rendered_blocks.append(blk.stub_text or blk.id)

            if self.utility_fn is not None:
                expected_utility += float(self.utility_fn(blk, chosen))
            elif slot_probabilities is not None and blk.id in slot_probabilities:
                dist = slot_probabilities[blk.id]
                p_full = float(dist.get("full", 0.0))
                p_stub = float(dist.get("stub", 0.0))
                if chosen == "full":
                    expected_utility += p_full + 0.5 * p_stub
                elif chosen == "stub":
                    expected_utility += 0.65 * p_stub + 0.35 * p_full

        reduction = (
            max(0.0, float(tokens_before - used_tokens) / float(tokens_before))
            if tokens_before > 0
            else 0.0
        )
        return ContextPruneResult(
            decisions=decisions,
            tokens_before=tokens_before,
            tokens_after=used_tokens,
            token_reduction_ratio=reduction,
            expected_utility=expected_utility,
            rendered_blocks=rendered_blocks,
        )


@dataclass(frozen=True)
class UniformFrameStrideControl:
    """Control C4: Uniform equispaced video frame striding matching an exact k-frame budget."""

    def select_frames(
        self,
        candidates: Sequence[SpatioTemporalFrameCandidate],
        max_frames: int = 8,
    ) -> list[SpatioTemporalFrameCandidate]:
        """Sample ``max_frames`` equispaced frames ordered chronologically without saliency cues."""
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise SchemaContractViolationError(
                f"max_frames must be an integer >= 1, got {max_frames!r}"
            )
        if not candidates:
            return []
        chronological = sorted(
            candidates, key=lambda frame: (frame.timestamp_sec, frame.frame_index)
        )
        total_frames = len(chronological)
        if total_frames <= max_frames:
            return list(chronological)
        if max_frames == 1:
            return [chronological[total_frames // 2]]

        selected_indices: list[int] = []
        used: set[int] = set()
        for slot_idx in range(max_frames):
            raw_pos = int(round(float(slot_idx) * float(total_frames - 1) / float(max_frames - 1)))
            pos = min(total_frames - 1, max(0, raw_pos))
            while pos in used and pos + 1 < total_frames:
                pos += 1
            while pos in used and pos - 1 >= 0:
                pos -= 1
            used.add(pos)
            selected_indices.append(pos)
        selected_indices.sort()
        return [chronological[idx] for idx in selected_indices]


@dataclass(frozen=True)
class PlaceboInterventionControl:
    """Control C5: Random loop-verification trigger matching NEVC's empirical trigger rate."""

    target_trigger_rate: float
    seed_key: str = "mdm-control-c5-placebo-nevc"

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_trigger_rate, bool)
            or not isinstance(self.target_trigger_rate, (int, float))
            or not math.isfinite(float(self.target_trigger_rate))
            or not (0.0 <= float(self.target_trigger_rate) <= 1.0)
        ):
            raise SchemaContractViolationError(
                f"target_trigger_rate must lie in [0, 1], got {self.target_trigger_rate!r}"
            )
        if not isinstance(self.seed_key, str) or not self.seed_key.strip():
            raise SchemaContractViolationError("seed_key must be a non-empty string")

    def should_intervene(self, episode_id: str, turn_index: int) -> bool:
        """Return deterministic Bernoulli(target_trigger_rate) intervention decision."""
        key = f"{episode_id}:t{int(turn_index)}"
        return hmac_uniform_float64(self.seed_key, key, "c5_intervene") < float(
            self.target_trigger_rate
        )

    def select_exact_quota(self, turn_keys: Sequence[str]) -> dict[str, bool]:
        """Select exact round(N * target_trigger_rate) turn interventions via HMAC rank order."""
        unique_keys = [str(key) for key in turn_keys]
        if len(set(unique_keys)) != len(unique_keys):
            raise SchemaContractViolationError("turn_keys must be unique for exact quota matching")
        n_total = len(unique_keys)
        if n_total == 0:
            return {}
        quota = min(
            n_total,
            max(0, int(round(float(n_total) * float(self.target_trigger_rate)))),
        )
        ranked = sorted(
            unique_keys,
            key=lambda key: (hmac_uniform_float64(self.seed_key, key, "c5_quota"), key),
        )
        chosen = set(ranked[:quota])
        return {key: (key in chosen) for key in unique_keys}

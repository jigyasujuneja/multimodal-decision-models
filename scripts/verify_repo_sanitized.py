#!/usr/bin/env python3
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

"""Verify repository sanitization against internal hosts, paths, keys, and codenames."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
from typing import Sequence

EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

# SHA-256 digests of lowercase internal codename tokens so this script itself
# contains zero literal restricted identifiers.
BANNED_TOKEN_SHA256: frozenset[str] = frozenset(
    {
        "e7f90b0310363d697e83711832881d3aabd37e3e8c25efe3883ed8de0c4c29fb",
        "eeda9c8d5884b58cb81f0e0478eccb9b656d07c25db802c8febad3bd46cbad7c",
        "ffbda6810ca417b9bb23af61b90adb6a38446188707798cc15c9e54629ba2047",
        "0bad769abf3c8331cf7f926a616417ad189672ae2e7248382064df7ca380d3b1",
        "b0397da4f608acbfa6963687d0e2eace84f0bd095fae34e97af02f78dd20011e",
        "e213661b27230291b4f559cf387ae1fd612af2560a0456648233fad891b49947",
        "e93ed16e76ef29da72cfc4e6b7a6b43cc4aebba41675c299c361997086f40ee3",
        "b3d9fd1b37ea401b47d98a9c14c16c847b1e34d2fed9cc790c5ed7d36123ef0d",
    }
)

WORD_SPLIT_RE = re.compile(r"[a-z0-9_]+")
SUBTOKEN_SPLIT_RE = re.compile(r"[a-z0-9]+")


def _build_blocked_line_regex() -> re.Pattern[str]:
    """Construct case-insensitive regex for internal hosts, paths, and credentials."""
    fragments = [
        "googlers" + r"\.com",
        "uplink" + r"\.goog",
        "artifact" + "-foundry",
        "corp" + r"\.google\.com",
        "decision" + "-lab",
        "/usr/local/" + "google",
        r"\b" + "EX" + "P-" + r"\d+\b",
        r"AIza[0-9A-Za-z_-]{35}",
        r"ya29\.[0-9A-Za-z_-]{20,}",
        "BEGIN" + r" (?:RSA |EC )?PRIVATE KEY",
    ]
    return re.compile("(" + "|".join(fragments) + ")", re.IGNORECASE)


def scan_repository(repo_root: Path) -> list[str]:
    """Scan all UTF-8 files under ``repo_root`` for hygiene or codename violations.

    Args:
        repo_root: Root directory of the repository to audit.

    Returns:
        Sorted list of ``path:line: reason`` violation messages (empty when clean).
    """
    blocked_re = _build_blocked_line_regex()
    self_path = Path(__file__).resolve()
    ci_workflow_path = (repo_root / ".github" / "workflows" / "ci.yml").resolve()
    violations: list[str] = []

    for file_path in sorted(repo_root.rglob("*")):
        if any(part in EXCLUDED_DIR_NAMES for part in file_path.parts) or file_path.is_dir():
            continue
        resolved = file_path.resolve()
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        rel_str = str(file_path.relative_to(repo_root))
        for line_no, line in enumerate(text.splitlines(), start=1):
            if resolved not in (self_path, ci_workflow_path):
                match = blocked_re.search(line)
                if match is not None:
                    violations.append(
                        f"{rel_str}:{line_no}: blocked pattern match ({match.group(0)})"
                    )
            lowered = line.lower()
            candidates: set[str] = set(WORD_SPLIT_RE.findall(lowered))
            candidates.update(SUBTOKEN_SPLIT_RE.findall(lowered))
            for token in candidates:
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                if digest in BANNED_TOKEN_SHA256:
                    violations.append(
                        f"{rel_str}:{line_no}: restricted internal token ({token!r})"
                    )
                    break

    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning exit code 0 when sanitized and 1 on any violation.

    Args:
        argv: Optional sequence of CLI arguments (first positional arg overrides repo root).

    Returns:
        Zero on clean scan, 1 when violations are detected.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    repo_root = Path(args[0]).resolve() if args else Path(__file__).resolve().parents[1]
    violations = scan_repository(repo_root)
    if violations:
        print(f"Sanitization audit FAILED with {len(violations)} violation(s):", file=sys.stderr)
        for entry in violations:
            print(f"  - {entry}", file=sys.stderr)
        return 1
    print("Sanitization audit PASSED: 0 internal host, path, credential, or codename leaks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

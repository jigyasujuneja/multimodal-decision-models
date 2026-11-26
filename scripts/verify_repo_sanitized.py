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

from collections.abc import Sequence
import hashlib
from pathlib import Path
import re
import sys

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
        "9918f0455cefecd78b99a6d8daf7da74732ccd50fdbafda860ecc0f050f2e725",
        "bbf997173d125ed48a1c6b9888e4f594245381b679cc44e651f964d18940d092",
        "a38c2d7bfc423b198f0e188e4d988bc1b8e8c107a760f9f66a2fc7e9b5b4ede5",
        "03fe7af632b170c39ca385e5f085b9f07ace99ce1edb6c9c4b541ad320b57d2b",
        "ebd8de38e4a70bc8b0fa11982ebe5c38d12b528bd3d0081c57f50412dcfe792b",
        "598f7a741a1e3a05654d346033571fda567af6dc2bf099b34b930171519d995f",
        "03c7909a40d39f970672583e1ef91e4ff68a733007faa80cd754bee272fb682f",
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
        "googleplex" + r"\.com",
        "google" + "3",
        "resource" + "key=",
        r"(?<![A-Za-z0-9_.:/-])" + "go/" + r"[a-z0-9_-]{2,}",
        "decision" + "-lab",
        "/usr/local/" + "google",
        r"\b" + "EX" + "P-" + r"\d+\b",
        r"\b" + "sso" + r"_client\b",
        "Google" + " corp",
        "saturn" + "-moe",
        "dgemma" + r"\.aaie\.cloud",
        r"\b" + "gev" + r"ctl\b",
        "gdm" + "_upgrades",
        "gold" + "fish",
        "dgem" + ":vertex",
        "beyond" + "-decision",
        "beyond" + "-api",
        "typesafe" + "_adapter",
        "jev" + ":openjev",
        "mdm" + ":cascade",
        "speak" + "-like-tl",
        "judgement" + "-doc",
        "golden-eval-" + "judge-audit",
        "anti" + "-slop",
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
    violations: list[str] = []
    scanned_count = 0

    for file_path in sorted(repo_root.rglob("*")):
        if file_path.is_dir():
            continue
        rel_path = file_path.relative_to(repo_root)
        if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
            continue
        resolved = file_path.resolve()
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned_count += 1

        rel_str = str(rel_path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if resolved != self_path:
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

    if scanned_count == 0:
        violations.append(f"{repo_root}:0: sanitization scan inspected 0 files")

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

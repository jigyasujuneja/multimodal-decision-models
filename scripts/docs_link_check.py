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

"""Validate relative Markdown file links and #fragment heading anchors across README.md and docs/."""

from __future__ import annotations

import pathlib
import re
import sys
import urllib.parse

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def github_heading_slug(raw_heading: str) -> str:
    """Convert a Markdown heading title into its GitHub-flavored anchor slug."""
    cleaned = re.sub(r"`([^`]*)`", r"\1", raw_heading)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.strip().lower()
    slug_chars: list[str] = []
    for char in cleaned:
        if char.isalnum() or char in ("-", "_"):
            slug_chars.append(char)
        elif char.isspace():
            slug_chars.append("-")
    return "".join(slug_chars).strip("-")


def collect_heading_anchors(markdown_path: pathlib.Path) -> set[str]:
    """Extract all GitHub-compatible heading slugs (including duplicate `-1` suffixes) from a Markdown file."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        base_slug = github_heading_slug(match.group(2))
        if not base_slug:
            continue
        seen = counts.get(base_slug, 0)
        counts[base_slug] = seen + 1
        final_slug = base_slug if seen == 0 else f"{base_slug}-{seen}"
        anchors.add(final_slug)
    return anchors


def strip_inline_code_spans(line: str) -> str:
    """Remove inline backtick code spans before scanning a Markdown line for links."""
    return re.sub(r"`[^`]*`", "", line)


def check_markdown_links(repo_root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Return human-readable error strings for broken relative paths or missing #fragment anchors."""
    markdown_files = [repo_root / "README.md"]
    if (repo_root / "CHANGELOG.md").exists():
        markdown_files.append(repo_root / "CHANGELOG.md")
    docs_dir = repo_root / "docs"
    if docs_dir.is_dir():
        markdown_files.extend(sorted(docs_dir.rglob("*.md")))

    anchor_cache: dict[pathlib.Path, set[str]] = {}
    errors: list[str] = []

    for md_file in markdown_files:
        if not md_file.exists():
            continue
        in_fence = False
        for line_num, raw_line in enumerate(md_file.read_text(encoding="utf-8").splitlines(), 1):
            if FENCE_RE.match(raw_line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            scan_line = strip_inline_code_spans(raw_line)
            for match in INLINE_LINK_RE.finditer(scan_line):
                raw_target = match.group(1).strip()
                if not raw_target or raw_target.startswith(("http://", "https://", "mailto:", "tel:")):
                    continue
                if raw_target.startswith("/"):
                    errors.append(
                        f"{md_file.relative_to(repo_root)}:{line_num}: absolute root link not portable -> {raw_target}"
                    )
                    continue

                path_part, sep, fragment = raw_target.partition("#")
                decoded_path = urllib.parse.unquote(path_part)
                target_path = (md_file.parent / decoded_path).resolve() if decoded_path else md_file.resolve()

                if not target_path.exists():
                    errors.append(
                        f"{md_file.relative_to(repo_root)}:{line_num}: broken relative link -> {raw_target}"
                    )
                    continue

                if sep and fragment and target_path.is_file() and target_path.suffix.lower() == ".md":
                    if target_path not in anchor_cache:
                        anchor_cache[target_path] = collect_heading_anchors(target_path)
                    decoded_fragment = urllib.parse.unquote(fragment).lower()
                    if decoded_fragment not in anchor_cache[target_path]:
                        errors.append(
                            f"{md_file.relative_to(repo_root)}:{line_num}: missing heading anchor #{fragment} in {target_path.relative_to(repo_root)}"
                        )
    return errors


def main() -> int:
    errors = check_markdown_links(REPO_ROOT)
    if errors:
        for item in errors:
            print(item)
        print(f"FAILED: {len(errors)} broken relative link or anchor issue(s).")
        return 1
    print("PASSED: 0 broken relative links or heading anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

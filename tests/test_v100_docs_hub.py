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

"""Verification tests for the v1.0.0 progressive developer documentation hub."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_DOCS_HUB_GUIDES: tuple[str, ...] = (
    "docs/index.md",
    "docs/quickstart-and-paradigms.md",
    "docs/calibration-and-conformal.md",
    "docs/wide-catalog-routing.md",
    "docs/multimodal-compilers.md",
    "docs/agentic-shared-control.md",
    "docs/distillation-and-tuning.md",
    "docs/production-deployment.md",
)

PRESERVED_RESEARCH_DOCS: tuple[str, ...] = (
    "docs/research_paper.md",
    "docs/research_summary.md",
    "docs/architecture.md",
    "docs/practical_use_cases.md",
)


class V100DocumentationHubTest(unittest.TestCase):
    """Ensure all progressive docs hub pages, research docs, and relative anchors stay intact."""

    def test_all_docs_hub_pages_exist_with_single_h1(self) -> None:
        for rel_path in EXPECTED_DOCS_HUB_GUIDES + PRESERVED_RESEARCH_DOCS:
            full_path = REPO_ROOT / rel_path
            self.assertTrue(full_path.is_file(), f"Missing documentation file: {rel_path}")
            content = full_path.read_text(encoding="utf-8")
            self.assertGreater(len(content), 800, f"Documentation file unexpectedly short: {rel_path}")
            if rel_path in EXPECTED_DOCS_HUB_GUIDES:
                in_fence = False
                h1_lines: list[str] = []
                for line in content.splitlines():
                    if line.strip().startswith("```"):
                        in_fence = not in_fence
                        continue
                    if not in_fence and line.startswith("# "):
                        h1_lines.append(line)
                self.assertEqual(len(h1_lines), 1, f"Expected exactly 1 H1 in {rel_path}, got {h1_lines}")

    def test_readme_changelog_and_link_checker_pass(self) -> None:
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/index.md", readme_text)
        self.assertIn("v1.0.0", readme_text)

        changelog_text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [1.0.0]", changelog_text)

        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "docs_link_check.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"docs_link_check.py failed:\n{completed.stdout}\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

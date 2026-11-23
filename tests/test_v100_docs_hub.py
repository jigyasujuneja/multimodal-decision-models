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
import re
import subprocess
import sys
import tempfile
import unittest

import multimodal_decision_models as mdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FENCED_BLOCK_RE = re.compile(r"```(python|go)\n(.*?)```", re.DOTALL)

EXPECTED_DOCS_HUB_GUIDES: tuple[str, ...] = (
    "docs/index.md",
    "docs/quickstart-and-paradigms.md",
    "docs/calibration-and-conformal.md",
    "docs/wide-catalog-routing.md",
    "docs/multimodal-compilers.md",
    "docs/agentic-shared-control.md",
    "docs/distillation-and-tuning.md",
    "docs/production-deployment.md",
    "docs/architectural-boundaries.md",
    "docs/security-and-eval-science.md",
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
        self.assertIn("v1.1.0", readme_text)
        self.assertIn("docs/architectural-boundaries.md", readme_text)
        self.assertIn("docs/security-and-eval-science.md", readme_text)

        changelog_text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## [1.1.0]", changelog_text)
        self.assertIn("## [1.0.0]", changelog_text)

        prod_deploy_text = (REPO_ROOT / "docs" / "production-deployment.md").read_text(encoding="utf-8")
        self.assertIn(
            f"version `{mdm.mcp_server.MCP_SERVER_VERSION}`",
            prod_deploy_text,
        )

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

    def test_all_fenced_python_and_go_snippets_execute_and_compile(self) -> None:
        synthetic_png = mdm.encode_png(
            mdm.ImageBuffer(width=16, height=16, rgb=bytes([120, 140, 180]) * (16 * 16))
        )
        python_blocks_executed = 0
        go_blocks_compiled = 0

        for rel_path in EXPECTED_DOCS_HUB_GUIDES:
            content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            exec_globals: dict[str, object] = {
                "__name__": "__doc_snippet__",
                "validation_probabilities": [
                    {"approve": 0.85, "reject": 0.15},
                    {"approve": 0.20, "reject": 0.80},
                    {"approve": 0.65, "reject": 0.35},
                ],
                "validation_gold_labels": ["approve", "reject", "approve"],
                "test_probabilities": [
                    {"approve": 0.80, "reject": 0.20},
                    {"approve": 0.25, "reject": 0.75},
                ],
                "test_gold_labels": ["approve", "reject"],
                "heldout_slot_probs": [
                    {"s1": {"a": 0.9, "b": 0.1}, "s2": {"x": 0.88, "y": 0.12}},
                    {"s1": {"a": 0.15, "b": 0.85}, "s2": {"x": 0.12, "y": 0.88}},
                ],
                "heldout_slot_labels": [
                    {"s1": "a", "s2": "x"},
                    {"s1": "b", "s2": "y"},
                ],
                "live_slot_probs": {"s1": {"a": 0.91, "b": 0.09}, "s2": {"x": 0.90, "y": 0.10}},
                "raw_query_png_bytes": synthetic_png,
                "raw_candidate_png_list": [synthetic_png, synthetic_png],
            }
            for block_idx, match in enumerate(_FENCED_BLOCK_RE.finditer(content)):
                lang, snippet = match.group(1), match.group(2)
                if lang == "python":
                    exec(compile(snippet, f"{rel_path}#block-{block_idx}", "exec"), exec_globals)
                    python_blocks_executed += 1
                elif lang == "go":
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        go_src = pathlib.Path(tmp_dir) / "snippet_main.go"
                        go_bin = pathlib.Path(tmp_dir) / "snippet_bin"
                        go_src.write_text(snippet, encoding="utf-8")
                        build_proc = subprocess.run(
                            ["go", "build", "-o", str(go_bin), str(go_src)],
                            cwd=str(REPO_ROOT),
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(
                            build_proc.returncode,
                            0,
                            f"Go snippet failed in {rel_path}#block-{block_idx}:\n{build_proc.stderr}",
                        )
                        go_blocks_compiled += 1

        self.assertGreaterEqual(python_blocks_executed, 19)
        self.assertGreaterEqual(go_blocks_compiled, 1)


if __name__ == "__main__":
    unittest.main()

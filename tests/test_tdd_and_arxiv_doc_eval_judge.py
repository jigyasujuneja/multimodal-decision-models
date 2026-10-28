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

"""Unit and End-to-End Document Suite Tests for the 7-Pillar, 14-Gate Binary (1/0) TDD & arXiv Evaluator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks import evaluate_tdd_and_arxiv_docs as doc_eval

REPO_ROOT = Path(__file__).resolve().parent.parent

END_TO_END_REPO_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "research_paper.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "practical_use_cases.md",
    REPO_ROOT / "docs" / "research_summary.md",
    REPO_ROOT / "benchmarks" / "results" / "multimodel_benchmark_report.md",
    REPO_ROOT / "benchmarks" / "results" / "mdm_bench_report.md",
    REPO_ROOT / "benchmarks" / "results" / "real_public_bench_report.md",
    REPO_ROOT / "benchmarks" / "results" / "frontier_genai_v070_report.md",
    REPO_ROOT / "benchmarks" / "results" / "audio_video_v080_report.md",
]


class TestTddAndArxivDocEvalJudge(unittest.TestCase):
    """Verify that all 10 end-to-end repository docs score 1 (14/14 gates) and negative cases score 0."""

    def test_all_end_to_end_repo_docs_score_binary_1_across_all_14_gates(self) -> None:
        for doc_path in END_TO_END_REPO_DOCS:
            self.assertTrue(doc_path.is_file(), f"Missing doc: {doc_path}")
            res = doc_eval.evaluate_document(doc_path, run_llm_judge=False)
            failed = {
                k: v["details"]
                for k, v in res["gates"].items()
                if v["score"] == 0
            }
            self.assertEqual(
                res["overall_binary_score"],
                1,
                f"Document {doc_path.name} failed binary 1/0 bar ({res['gates_passed']}/{res['gates_total']}): {failed}",
            )
            self.assertEqual(res["gates_passed"], 14)

    def test_negative_cases_trigger_binary_0_across_pillars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_doc = Path(tmpdir) / "bad_slop_tdd.md"
            bad_doc.write_text(
                "# Why does our legacy L8 GDM system delve into a rich tapestry?\n\n"
                "User Request: Step 1.1: We utilize and leverage a pivotal, groundbreaking paradigm shift, "
                "highlighting our holistic synergy — a true testament — across Claude-3.7-Sonnet-Dirichlet-M5.\n\n"
                "TODO: fix [broken link](missing_file.md) and citation [?] with unbalanced math $x + y.\n\n"
                "+-------+-------+\n"
                "| Box 1 | Box 2 |\n"
                "+-------+-------+\n",
                encoding="utf-8",
            )
            res = doc_eval.evaluate_document(bad_doc, run_llm_judge=False)
            self.assertEqual(res["overall_binary_score"], 0)
            gates = res["gates"]
            self.assertEqual(gates["G1_taste_terminology_and_anonymity"]["score"], 0)
            self.assertEqual(gates["G3_writing_zero_claudeisms_and_ai_slop"]["score"], 0)
            self.assertEqual(gates["G4_writing_punctuation_and_active_voice"]["score"], 0)
            self.assertEqual(gates["G6_factuality_model_and_metric_consistency"]["score"], 0)
            self.assertEqual(gates["G7_groundedness_claim_to_proof_and_zero_placeholders"]["score"], 0)
            self.assertEqual(gates["G8_groundedness_citations_and_baselines"]["score"], 0)
            self.assertEqual(gates["G10_structure_heading_and_paragraph_pacing"]["score"], 0)
            self.assertEqual(gates["G11_formatting_tables_and_math_hygiene"]["score"], 0)
            self.assertEqual(gates["G12_formatting_publication_visuals_and_zero_ascii_art"]["score"], 0)
            self.assertEqual(gates["G13_polish_three_persona_llm_judge"]["score"], 0)
            self.assertEqual(gates["G14_polish_goldfish_self_containment_and_firewall"]["score"], 0)


if __name__ == "__main__":
    unittest.main()

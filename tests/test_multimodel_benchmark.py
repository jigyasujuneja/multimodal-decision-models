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

"""Deterministic unit tests for benchmarks/run_multimodel_benchmark.py."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "run_multimodel_benchmark.py"
BENCH_DATA_DIR = REPO_ROOT / "benchmarks" / "data"
BENCH_RESULTS_JSON = REPO_ROOT / "benchmarks" / "results" / "multimodel_benchmark_results.json"
BENCH_REPORT_MD = REPO_ROOT / "benchmarks" / "results" / "multimodel_benchmark_report.md"


def _load_benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_multimodel_benchmark", BENCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMultiModelBenchmarkSuite(unittest.TestCase):
    """Behavior and hygiene tests for the deterministic multi-model benchmark suite."""

    def test_cli_help_exits_zero_and_ast_uses_no_synthetic_rng(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-B", str(BENCH_SCRIPT), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--data-dir", proc.stdout)

        src = BENCH_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "random":
                    self.assertNotIn(
                        node.func.attr,
                        {"random", "uniform", "gauss", "normalvariate", "betavariate"},
                    )

    def test_receipts_and_results_satisfy_simplex_conformal_monotonicity_and_hygiene(self) -> None:
        self.assertTrue(BENCH_RESULTS_JSON.is_file(), f"Missing {BENCH_RESULTS_JSON}")
        self.assertTrue(BENCH_REPORT_MD.is_file(), f"Missing {BENCH_REPORT_MD}")

        mod = _load_benchmark_module()
        json_text = BENCH_RESULTS_JSON.read_text(encoding="utf-8")
        md_text = BENCH_REPORT_MD.read_text(encoding="utf-8")
        mod.verify_artifact_hygiene_and_antislop(json_text, md_text)

        payload = json.loads(json_text)
        self.assertGreaterEqual(payload["distinct_model_count"], 7)
        self.assertGreaterEqual(len(payload["models"]), 14)

        for arm in payload["models"]:
            self.assertGreaterEqual(arm["accuracy"], 0.0)
            self.assertLessEqual(arm["accuracy"], 1.0)
            self.assertGreaterEqual(arm["brier"], 0.0)
            self.assertLessEqual(arm["brier"], 2.0)
            self.assertGreaterEqual(
                arm["conformal_coverage_005"] + 1e-9,
                arm["conformal_coverage_010"],
                f"Conformal monotonicity violated for {arm['model']}",
            )
            self.assertGreaterEqual(
                arm["conformal_mean_set_size_005"] + 1e-9,
                arm["conformal_mean_set_size_010"],
            )
            ci_lo, ci_hi = arm["bootstrap_ci95"]["brier"]
            self.assertLessEqual(ci_lo, ci_hi)

    def test_deterministic_reexecution_matches_committed_outputs_byte_for_byte(self) -> None:
        mod = _load_benchmark_module()
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "out.json"
            out_md = Path(td) / "out.md"
            mod.run_benchmark(BENCH_DATA_DIR, out_json, out_md)
            self.assertEqual(
                out_json.read_text(encoding="utf-8"),
                BENCH_RESULTS_JSON.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                out_md.read_text(encoding="utf-8"),
                BENCH_REPORT_MD.read_text(encoding="utf-8"),
            )

    def test_load_jsonl_rows_rejects_missing_or_empty_files(self) -> None:
        mod = _load_benchmark_module()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.jsonl"
            with self.assertRaises(FileNotFoundError):
                mod.load_jsonl_rows(missing)
            empty = Path(td) / "empty.jsonl"
            empty.write_text("\n  \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                mod.load_jsonl_rows(empty)

    def test_pareto_frontier_and_artifact_self_audit_guards_boundary_cases(self) -> None:
        mod = _load_benchmark_module()
        arms = [
            {"arm_name": "fast_accurate", "brier": 0.15, "mean_latency_ms": 200.0},
            {"arm_name": "dominated_slow", "brier": 0.22, "mean_latency_ms": 450.0},
            {"arm_name": "ultra_fast", "brier": 0.30, "mean_latency_ms": 90.0},
        ]
        frontier = mod.mark_pareto_frontier(arms)
        self.assertEqual(set(frontier), {"fast_accurate", "ultra_fast"})
        self.assertFalse(arms[1]["pareto_optimal"])

        with self.assertRaises(AssertionError):
            mod.verify_artifact_hygiene_and_antislop('{"x": NaN}', "# Clean\n")
        with self.assertRaises(AssertionError):
            mod.verify_artifact_hygiene_and_antislop('{"x": 1}', "# Bad \u2014 dash\n")
        with self.assertRaises(AssertionError):
            mod.verify_artifact_hygiene_and_antislop('{"x": 1}', "# Title\n---\nBody\n")


if __name__ == "__main__":
    unittest.main()


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

"""Offline structural, security, and brace-balance validator for deploy/terraform."""

from __future__ import annotations

import pathlib
import re
import unittest

TERRAFORM_DIR = pathlib.Path(__file__).resolve().parent

REQUIRED_TF_FILES: tuple[str, ...] = (
    "versions.tf",
    "variables.tf",
    "storage_and_registry.tf",
    "iam.tf",
    "cloudrun_orchestrator.tf",
    "vertex_endpoint.tf",
    "monitoring.tf",
    "outputs.tf",
)

EXPECTED_RESOURCE_TYPES: frozenset[str] = frozenset(
    {
        "google_artifact_registry_repository",
        "google_storage_bucket",
        "google_service_account",
        "google_storage_bucket_iam_member",
        "google_project_iam_member",
        "google_secret_manager_secret_iam_member",
        "google_cloud_run_v2_service",
        "google_cloud_run_v2_service_iam_member",
        "google_vertex_ai_model",
        "google_vertex_ai_endpoint",
        "google_logging_metric",
        "google_monitoring_dashboard",
    }
)


def _strip_hcl_strings_and_comments(hcl_text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in hcl_text.splitlines():
        in_string = False
        escaped = False
        chars: list[str] = []
        idx = 0
        while idx < len(raw_line):
            ch = raw_line[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                idx += 1
                continue
            if ch == "#":
                break
            if ch == "/" and idx + 1 < len(raw_line) and raw_line[idx + 1] == "/":
                break
            if ch == '"':
                in_string = True
                idx += 1
                continue
            chars.append(ch)
            idx += 1
        cleaned_lines.append("".join(chars))
    return "\n".join(cleaned_lines)


class TerraformModuleValidationTest(unittest.TestCase):
    """Validate HCL syntax invariants and least-privilege guardrails without cloud credentials."""

    def test_required_files_and_balanced_delimiters(self) -> None:
        for filename in REQUIRED_TF_FILES:
            tf_path = TERRAFORM_DIR / filename
            self.assertTrue(tf_path.is_file(), f"Missing Terraform file: {filename}")
            stripped = _strip_hcl_strings_and_comments(tf_path.read_text(encoding="utf-8"))
            self.assertEqual(
                stripped.count("{"),
                stripped.count("}"),
                f"Unbalanced curly braces in {filename}",
            )
            self.assertEqual(
                stripped.count("["),
                stripped.count("]"),
                f"Unbalanced square brackets in {filename}",
            )
            self.assertEqual(
                stripped.count("("),
                stripped.count(")"),
                f"Unbalanced parentheses in {filename}",
            )

    def test_resource_types_and_security_guardrails(self) -> None:
        combined = "\n".join(
            (TERRAFORM_DIR / name).read_text(encoding="utf-8") for name in REQUIRED_TF_FILES
        )
        discovered_resources = set(re.findall(r'resource\s+"([a-z0-9_]+)"\s+"[a-z0-9_]+"', combined))
        self.assertEqual(discovered_resources, EXPECTED_RESOURCE_TYPES)

        variables_text = (TERRAFORM_DIR / "variables.tf").read_text(encoding="utf-8")
        self.assertIn("@sha256:[0-9a-f]{64}", variables_text)
        self.assertIn("allUsers", variables_text)
        self.assertIn("allAuthenticatedUsers", variables_text)

        bucket_text = (TERRAFORM_DIR / "storage_and_registry.tf").read_text(encoding="utf-8")
        self.assertIn('public_access_prevention    = "enforced"', bucket_text)
        self.assertIn("uniform_bucket_level_access = true", bucket_text)

        monitoring_text = (TERRAFORM_DIR / "monitoring.tf").read_text(encoding="utf-8")
        logging_metrics = re.findall(
            r'resource\s+"google_logging_metric"\s+"([a-z0-9_]+)"',
            monitoring_text,
        )
        self.assertEqual(len(logging_metrics), 6)

        readme_text = (TERRAFORM_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("Live in Terraform", readme_text)
        self.assertIn("In-Process", readme_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

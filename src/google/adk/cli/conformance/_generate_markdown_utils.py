# Copyright 2026 Google LLC
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

"""Utilities for generating Markdown reports for conformance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING

import click

from ... import version

if TYPE_CHECKING:
  from .cli_test import _ConformanceTestSummary


def generate_markdown_report(
    summary: _ConformanceTestSummary, report_dir: Optional[str]
) -> None:
  """Generates a Markdown report of the test results."""
  report_name = f"python_{'_'.join(version.__version__.split('.'))}_report.md"
  if not report_dir:
    report_path = Path(report_name)
  else:
    report_path = Path(report_dir) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)

  with open(report_path, "w") as f:
    f.write("# ADK Python Conformance Test Report\n\n")

    # Summary
    f.write("## Summary\n\n")
    f.write(f"- **ADK Version**: {version.__version__}\n")
    f.write(f"- **Total Tests**: {summary.total_tests}\n")
    f.write(f"- **Passed**: {summary.passed_tests}\n")
    f.write(f"- **Failed**: {summary.failed_tests}\n")
    f.write(f"- **Success Rate**: {summary.success_rate:.1f}%\n\n")

    # Table
    f.write("## Test Results\n\n")
    f.write("| Status | Category | Test Name | Description |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")

    for result in summary.results:
      status_icon = "✅ PASS" if result.success else "❌ FAIL"
      description = (
          result.description.replace("\n", " ") if result.description else ""
      )
      f.write(
          f"| {status_icon} | {result.category} | {result.name} |"
          f" {description} |\n"
      )

    f.write("\n")

    # Failed Tests Details
    if summary.failed_tests > 0:
      f.write("## Failed Tests Details\n\n")
      for result in summary.results:
        if not result.success:
          f.write(f"### {result.category}/{result.name}\n\n")
          if result.description:
            f.write(f"**Description**: {result.description}\n\n")
          f.write("**Error**:\n")
          f.write("```\n")
          f.write(f"{result.error_message}\n")
          f.write("```\n\n")

  click.secho(f"\nReport generated at: {report_path.resolve()}", fg="blue")

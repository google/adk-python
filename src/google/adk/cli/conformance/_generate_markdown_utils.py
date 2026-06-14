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
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
  from .cli_test import _ConformanceTestSummary


def generate_markdown_report(
    version_data: dict[str, Any],
    summaries: list[_ConformanceTestSummary],
    report_dir: Optional[str],
) -> None:
  """Generates a Markdown report of the test results."""
  server_version = version_data.get("version", "Unknown")
  language = version_data.get("language", "Unknown")
  language_version = version_data.get("language_version", "Unknown")

  report_name = f"python_{'_'.join(server_version.split('.'))}_report.md"
  if not report_dir:
    report_path = Path(report_name)
  else:
    report_path = Path(report_dir) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)

  # Collect all test results
  test_results = {}
  test_descriptions = {}
  streaming_modes = []

  for summary in summaries:
    mode_name = (
        str(summary.streaming_mode.value)
        if summary.streaming_mode.value is not None
        else "none"
    )
    streaming_modes.append(mode_name)
    for result in summary.results:
      key = (result.category, result.name)
      if key not in test_results:
        test_results[key] = {}
      test_results[key][mode_name] = result
      if result.description:
        test_descriptions[key] = result.description

  streaming_modes.sort()

  with open(report_path, "w") as f:
    f.write("# ADK Python Conformance Test Report\n\n")
    f.write("## Summary\n\n")
    f.write(f"- **ADK Version**: {server_version}\n")
    f.write(f"- **Language**: {language} {language_version}\n\n")

    f.write(
        "| Streaming Mode | Total Tests | Passed | Failed | Success Rate |\n"
    )
    f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    for summary in summaries:
      mode_name = (
          str(summary.streaming_mode.value)
          if summary.streaming_mode.value is not None
          else "none"
      )
      f.write(
          f"| {mode_name} | {summary.total_tests} |"
          f" {summary.passed_tests} | {summary.failed_tests} |"
          f" {summary.success_rate:.1f}% |\n"
      )
    f.write("\n")

    # Table
    f.write("## Test Results\n\n")
    headers = ["Category", "Test Name", "Description"] + streaming_modes
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join([":---"] * len(headers)) + " |\n")

    sorted_keys = sorted(test_results.keys())
    for category, name in sorted_keys:
      description = test_descriptions.get((category, name), "").replace(
          "\n", " "
      )
      row = [category, name, description]
      for mode in streaming_modes:
        result = test_results[(category, name)].get(mode)
        if result:
          status_icon = "✅ PASS" if result.success else "❌ FAIL"
        else:
          status_icon = "N/A"
        row.append(status_icon)
      f.write("| " + " | ".join(row) + " |\n")

    f.write("\n")

    # Failed Tests Details
    has_failures = any(s.failed_tests > 0 for s in summaries)
    if has_failures:
      f.write("## Failed Tests Details\n\n")
      for summary in summaries:
        if summary.failed_tests > 0:
          mode_name = (
              str(summary.streaming_mode.value)
              if summary.streaming_mode.value is not None
              else "none"
          )
          for result in summary.results:
            if not result.success:
              f.write(f"### {result.category}/{result.name} ({mode_name})\n\n")
              if result.description:
                f.write(f"**Description**: {result.description}\n\n")
              f.write("**Error**:\n")
              f.write("```\n")
              f.write(f"{result.error_message}\n")
              f.write("```\n\n")

  click.secho(f"\nReport generated at: {report_path.resolve()}", fg="blue")

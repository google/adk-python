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

"""Tests for the conformance Markdown report writer."""

from __future__ import annotations

from google.adk.agents.run_config import StreamingMode
from google.adk.cli.conformance._generate_markdown_utils import generate_markdown_report
from google.adk.cli.conformance.cli_test import _ConformanceTestSummary
from google.adk.cli.conformance.cli_test import _TestResult

_VERSION_DATA = {
    'version': '1.2.3',
    'language': 'python',
    'language_version': '3.11.0',
}


def _summary(streaming_mode, results):
  passed = sum(1 for r in results if r.success)
  return _ConformanceTestSummary(
      total_tests=len(results),
      passed_tests=passed,
      failed_tests=len(results) - passed,
      results=results,
      streaming_mode=streaming_mode,
  )


def _report_text(tmp_path, version_data, summaries):
  generate_markdown_report(version_data, summaries, str(tmp_path))
  written = list(tmp_path.glob('*.md'))
  assert len(written) == 1, written
  return written[0], written[0].read_text()


def test_generate_markdown_report_names_the_file_after_the_server_version(
    tmp_path,
):
  path, _ = _report_text(
      tmp_path,
      _VERSION_DATA,
      [_summary(StreamingMode.NONE, [_TestResult('c', 'n', True)])],
  )

  # Dots in the version become underscores so the name is a single token.
  assert path.name == 'python_1_2_3_report.md'


def test_generate_markdown_report_creates_a_missing_report_directory(tmp_path):
  target = tmp_path / 'nested' / 'reports'

  generate_markdown_report(
      _VERSION_DATA,
      [_summary(StreamingMode.NONE, [_TestResult('c', 'n', True)])],
      str(target),
  )

  assert (target / 'python_1_2_3_report.md').exists()


def test_generate_markdown_report_falls_back_to_unknown_version_fields(
    tmp_path,
):
  path, text = _report_text(
      tmp_path,
      {},
      [_summary(StreamingMode.NONE, [_TestResult('c', 'n', True)])],
  )

  assert path.name == 'python_Unknown_report.md'
  assert '- **ADK Version**: Unknown' in text
  assert '- **Language**: Unknown Unknown' in text


def test_generate_markdown_report_summarizes_counts_per_streaming_mode(
    tmp_path,
):
  none_results = [
      _TestResult('cat', 'a', True),
      _TestResult('cat', 'b', False, error_message='boom'),
      _TestResult('cat', 'c', True),
      _TestResult('cat', 'd', True),
  ]
  sse_results = [_TestResult('cat', 'a', True), _TestResult('cat', 'b', True)]

  _, text = _report_text(
      tmp_path,
      _VERSION_DATA,
      [
          _summary(StreamingMode.NONE, none_results),
          _summary(StreamingMode.SSE, sse_results),
      ],
  )

  # StreamingMode.NONE has a value of None, which the report renders as "none".
  assert '| none | 4 | 3 | 1 | 75.0% |' in text
  assert '| sse | 2 | 2 | 0 | 100.0% |' in text


def test_generate_markdown_report_puts_each_streaming_mode_in_its_own_column(
    tmp_path,
):
  _, text = _report_text(
      tmp_path,
      _VERSION_DATA,
      [
          _summary(
              StreamingMode.SSE,
              [_TestResult('cat', 'only_sse', True, description='desc')],
          ),
          _summary(
              StreamingMode.NONE,
              [_TestResult('cat', 'both', False, error_message='bad')],
          ),
      ],
  )

  # Mode columns are sorted, so "none" precedes "sse" regardless of the order
  # the summaries were supplied in.
  assert '| Category | Test Name | Description | none | sse |' in text
  # A test only run under one mode is N/A under the other.
  assert '| cat | only_sse | desc | N/A | ✅ PASS |' in text
  assert '| cat | both |  | ❌ FAIL | N/A |' in text


def test_generate_markdown_report_flattens_newlines_in_descriptions(tmp_path):
  _, text = _report_text(
      tmp_path,
      _VERSION_DATA,
      [
          _summary(
              StreamingMode.NONE,
              [_TestResult('cat', 'n', True, description='line one\nline two')],
          )
      ],
  )

  # A raw newline would break the Markdown table row.
  assert '| cat | n | line one line two | ✅ PASS |' in text


def test_generate_markdown_report_details_only_failures(tmp_path):
  _, text = _report_text(
      tmp_path,
      _VERSION_DATA,
      [
          _summary(
              StreamingMode.NONE,
              [
                  _TestResult('cat', 'good', True, description='fine'),
                  _TestResult(
                      'cat',
                      'bad',
                      False,
                      error_message='event 0 mismatch',
                      description='why it matters',
                  ),
              ],
          )
      ],
  )

  assert '## Failed Tests Details' in text
  assert '### cat/bad (none)' in text
  assert '**Description**: why it matters' in text
  assert 'event 0 mismatch' in text
  assert '### cat/good' not in text


def test_generate_markdown_report_omits_failure_section_when_all_pass(tmp_path):
  _, text = _report_text(
      tmp_path,
      _VERSION_DATA,
      [_summary(StreamingMode.NONE, [_TestResult('cat', 'good', True)])],
  )

  assert '## Failed Tests Details' not in text

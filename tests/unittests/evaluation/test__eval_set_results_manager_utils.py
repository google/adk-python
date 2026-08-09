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

from __future__ import annotations

import json

from google.adk.evaluation._eval_set_results_manager_utils import _sanitize_eval_set_result_name
from google.adk.evaluation._eval_set_results_manager_utils import create_eval_set_result
from google.adk.evaluation._eval_set_results_manager_utils import parse_eval_set_result_json
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_result import EvalSetResult
import pytest


def _build_eval_case_result(
    eval_id: str = "eval_1", session_id: str = "session_1"
) -> EvalCaseResult:
  """Builds a minimal but valid EvalCaseResult for testing."""
  return EvalCaseResult(
      eval_set_id="eval_set_1",
      eval_id=eval_id,
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id=session_id,
  )


class TestSanitizeEvalSetResultName:

  def test_replaces_forward_slash_with_underscore(self):
    assert _sanitize_eval_set_result_name("app/eval_set") == "app_eval_set"

  def test_replaces_all_forward_slashes(self):
    assert _sanitize_eval_set_result_name("a/b/c/d") == "a_b_c_d"

  def test_name_without_slash_is_unchanged(self):
    assert _sanitize_eval_set_result_name("app_eval_set_123") == (
        "app_eval_set_123"
    )

  def test_empty_name_is_unchanged(self):
    assert _sanitize_eval_set_result_name("") == ""


class TestCreateEvalSetResult:

  def test_creates_eval_set_result_with_expected_fields(self):
    eval_case_results = [_build_eval_case_result()]

    result = create_eval_set_result(
        app_name="my_app",
        eval_set_id="my_eval_set",
        eval_case_results=eval_case_results,
    )

    assert isinstance(result, EvalSetResult)
    assert result.eval_set_id == "my_eval_set"
    assert result.eval_case_results == eval_case_results

  def test_result_id_encodes_app_eval_set_and_timestamp(self):
    result = create_eval_set_result(
        app_name="my_app",
        eval_set_id="my_eval_set",
        eval_case_results=[],
    )

    # The id is "{app_name}_{eval_set_id}_{timestamp}" and the timestamp is
    # stored verbatim as the creation_timestamp.
    assert result.eval_set_result_id == (
        f"my_app_my_eval_set_{result.creation_timestamp}"
    )

  def test_result_name_is_sanitized(self):
    # A "/" in the id (here via the app name) must not survive into the name,
    # since the name is used to derive a filesystem-safe identifier.
    result = create_eval_set_result(
        app_name="my/app",
        eval_set_id="my_eval_set",
        eval_case_results=[],
    )

    assert "/" not in result.eval_set_result_name
    assert result.eval_set_result_name == (
        result.eval_set_result_id.replace("/", "_")
    )

  def test_creates_result_with_empty_eval_case_results(self):
    result = create_eval_set_result(
        app_name="my_app",
        eval_set_id="my_eval_set",
        eval_case_results=[],
    )

    assert result.eval_case_results == []


class TestParseEvalSetResultJson:

  def _build_eval_set_result(self) -> EvalSetResult:
    return EvalSetResult(
        eval_set_result_id="my_app_my_eval_set_123.0",
        eval_set_result_name="my_app_my_eval_set_123.0",
        eval_set_id="my_eval_set",
        eval_case_results=[_build_eval_case_result()],
        creation_timestamp=123.0,
    )

  def test_parses_standard_json_string(self):
    original = self._build_eval_set_result()

    parsed = parse_eval_set_result_json(original.model_dump_json())

    assert parsed == original

  def test_parses_json_bytes(self):
    original = self._build_eval_set_result()

    parsed = parse_eval_set_result_json(original.model_dump_json().encode())

    assert parsed == original

  def test_parses_camel_case_aliased_json(self):
    original = self._build_eval_set_result()

    parsed = parse_eval_set_result_json(original.model_dump_json(by_alias=True))

    assert parsed == original

  def test_parses_legacy_double_encoded_json(self):
    # Legacy result files stored the object as a JSON-encoded string, i.e. the
    # outer JSON is a string whose value is itself the inner JSON object.
    original = self._build_eval_set_result()
    double_encoded = json.dumps(original.model_dump_json())

    parsed = parse_eval_set_result_json(double_encoded)

    assert parsed == original

  def test_raises_on_json_object_missing_required_fields(self):
    with pytest.raises(Exception):
      parse_eval_set_result_json('{"unexpected_field": "value"}')

  def test_raises_on_non_json_input(self):
    with pytest.raises(Exception):
      parse_eval_set_result_json("not valid json at all {")

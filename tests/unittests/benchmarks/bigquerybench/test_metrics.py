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

"""Tests for BigQueryBench metrics including LLM-as-judge adherence."""

import json
from unittest import mock

from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.eval_set import EvalSet
from google.genai import types as genai_types
import pytest

from benchmarks.bigquerybench.metrics import instruction_adherence_score
from benchmarks.bigquerybench.metrics import tool_args_score
from benchmarks.bigquerybench.metrics import tool_invocation_score


def _make_content(text: str) -> genai_types.Content:
  return genai_types.Content(parts=[genai_types.Part(text=text)], role="user")


def _make_invocation(
    tool_calls: list[dict],
    user_text: str = "test",
    final_text: str = "",
) -> Invocation:
  tool_uses = []
  for tc in tool_calls:
    tool_uses.append(
        genai_types.FunctionCall(name=tc["name"], args=tc.get("args", {}))
    )
  inv = Invocation(
      invocation_id="inv-test",
      user_content=_make_content(user_text),
      intermediate_data=IntermediateData(tool_uses=tool_uses),
  )
  if final_text:
    inv.final_response = genai_types.Content(
        parts=[genai_types.Part(text=final_text)], role="model"
    )
  return inv


# ── tool_invocation_score tests ────────────────────────────────────


class TestToolInvocationScore:

  def test_all_expected_tools_present(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {"name": "load_skill_resource", "args": {"skill_name": "x"}},
        ])
    ]
    expected = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {"name": "load_skill_resource", "args": {"skill_name": "x"}},
        ])
    ]
    result = tool_invocation_score(metric, actual, expected)
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED

  def test_missing_expected_tool(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
        ])
    ]
    expected = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {"name": "run_skill_script", "args": {"skill_name": "x"}},
        ])
    ]
    result = tool_invocation_score(metric, actual, expected)
    assert result.overall_score == 0.5
    assert result.overall_eval_status == EvalStatus.FAILED

  def test_extra_tools_are_ok(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {"name": "list_skills"},
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {"name": "execute_sql", "args": {"project_id": "x"}},
        ])
    ]
    expected = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
        ])
    ]
    result = tool_invocation_score(metric, actual, expected)
    assert result.overall_score == 1.0

  def test_empty_expected(self):
    metric = EvalMetric(metric_name="test")
    result = tool_invocation_score(metric, [], None)
    assert result.overall_score == 1.0


# ── tool_args_score tests ─────────────────────────────────────────


class TestToolArgsScore:

  def test_all_args_match(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {
                "name": "load_skill_resource",
                "args": {
                    "skill_name": "bq-sql-analyst",
                    "path": "references/public-datasets.md",
                },
            },
        ])
    ]
    expected = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
            {
                "name": "load_skill_resource",
                "args": {
                    "skill_name": "bq-sql-analyst",
                    "path": "references/public-datasets.md",
                },
            },
        ])
    ]
    result = tool_args_score(metric, actual, expected)
    assert result.overall_score == 1.0

  def test_wrong_skill_name(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "wrong-skill"}},
        ])
    ]
    expected = [
        _make_invocation([
            {"name": "load_skill", "args": {"name": "bq-sql-analyst"}},
        ])
    ]
    result = tool_args_score(metric, actual, expected)
    assert result.overall_score == 0.0

  def test_wrong_path(self):
    metric = EvalMetric(metric_name="test")
    actual = [
        _make_invocation([
            {
                "name": "load_skill_resource",
                "args": {
                    "skill_name": "bq-sql-analyst",
                    "path": "references/wrong.md",
                },
            },
        ])
    ]
    expected = [
        _make_invocation([
            {
                "name": "load_skill_resource",
                "args": {
                    "skill_name": "bq-sql-analyst",
                    "path": "references/public-datasets.md",
                },
            },
        ])
    ]
    result = tool_args_score(metric, actual, expected)
    # skill_name matches (1/2), path doesn't (0/2) -> 1/2 = 0.5
    assert result.overall_score == 0.5

  def test_no_key_args_in_expected(self):
    metric = EvalMetric(metric_name="test")
    actual = [_make_invocation([{"name": "list_skills"}])]
    expected = [_make_invocation([{"name": "list_skills"}])]
    result = tool_args_score(metric, actual, expected)
    assert result.overall_score == 1.0


# ── instruction_adherence_score tests ─────────────────────────────


def _make_rubric(rubric_id: str, text: str) -> Rubric:
  return Rubric(
      rubric_id=rubric_id,
      rubric_content=RubricContent(text_property=text),
  )


class TestInstructionAdherenceScore:

  @pytest.mark.asyncio
  async def test_no_rubrics_returns_pass(self):
    actual = [_make_invocation([{"name": "list_skills"}])]
    result = await instruction_adherence_score(actual, None)
    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED

  @pytest.mark.asyncio
  async def test_all_rubrics_pass(self):
    rubrics = [
        _make_rubric("r1", "The agent listed skills."),
        _make_rubric("r2", "The response mentions bq-sql-analyst."),
    ]
    actual = [
        _make_invocation(
            [{"name": "list_skills"}],
            user_text="What skills are available?",
            final_text="Available skills: bq-sql-analyst.",
        )
    ]

    mock_response = mock.MagicMock()
    mock_response.text = (
        "Property: The agent listed skills.\n"
        "Rationale: The agent called list_skills.\n"
        "Verdict: yes\n\n"
        "Property: The response mentions bq-sql-analyst.\n"
        "Rationale: The response explicitly names bq-sql-analyst.\n"
        "Verdict: yes\n"
    )

    mock_models = mock.AsyncMock()
    mock_models.generate_content_async.return_value = mock_response
    mock_client = mock.MagicMock()
    mock_client.models = mock_models

    with mock.patch("google.genai.Client", return_value=mock_client):
      result = await instruction_adherence_score(actual, rubrics)

    assert result.overall_score == 1.0
    assert result.overall_eval_status == EvalStatus.PASSED

  @pytest.mark.asyncio
  async def test_some_rubrics_fail(self):
    rubrics = [
        _make_rubric("r1", "The agent loaded the skill."),
        _make_rubric("r2", "The result has a table."),
    ]
    actual = [
        _make_invocation(
            [{"name": "load_skill", "args": {"name": "bq-sql-analyst"}}],
            final_text="Skill loaded.",
        )
    ]

    mock_response = mock.MagicMock()
    mock_response.text = (
        "Property: The agent loaded the skill.\n"
        "Rationale: load_skill was called.\n"
        "Verdict: yes\n\n"
        "Property: The result has a table.\n"
        "Rationale: No table in the response.\n"
        "Verdict: no\n"
    )

    mock_models = mock.AsyncMock()
    mock_models.generate_content_async.return_value = mock_response
    mock_client = mock.MagicMock()
    mock_client.models = mock_models

    with mock.patch("google.genai.Client", return_value=mock_client):
      result = await instruction_adherence_score(actual, rubrics)

    assert result.overall_score == 0.5
    assert result.overall_eval_status == EvalStatus.FAILED

  @pytest.mark.asyncio
  async def test_llm_call_failure(self):
    rubrics = [_make_rubric("r1", "Some assertion.")]
    actual = [_make_invocation([{"name": "list_skills"}])]

    with mock.patch("google.genai.Client", side_effect=Exception("API error")):
      result = await instruction_adherence_score(actual, rubrics)

    assert result.overall_score == 0.0
    assert result.overall_eval_status == EvalStatus.FAILED


# ── eval JSON validation ──────────────────────────────────────────


class TestEvalSetValidation:

  def test_eval_json_parses(self):
    import pathlib

    eval_path = (
        pathlib.Path(__file__).parent.parent.parent.parent.parent
        / "benchmarks"
        / "bigquerybench"
        / "eval_sets"
        / "bigquerybench_eval.json"
    )
    with open(eval_path) as f:
      data = json.load(f)
    es = EvalSet.model_validate(data)
    assert len(es.eval_cases) == 5

  def test_all_cases_have_rubrics(self):
    import pathlib

    eval_path = (
        pathlib.Path(__file__).parent.parent.parent.parent.parent
        / "benchmarks"
        / "bigquerybench"
        / "eval_sets"
        / "bigquerybench_eval.json"
    )
    with open(eval_path) as f:
      data = json.load(f)
    es = EvalSet.model_validate(data)
    for case in es.eval_cases:
      assert case.rubrics is not None, f"{case.eval_id} missing rubrics"
      assert len(case.rubrics) >= 2, f"{case.eval_id} should have >= 2 rubrics"

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

"""Tests for AgentEvaluator."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.evaluation import agent_evaluator as agent_evaluator_module
from google.adk.evaluation.agent_evaluator import _EvalMetricResultWithInvocation
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.agent_evaluator import load_json
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import LiveModelConfig
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.simulation.user_simulator_provider import UserSimulatorProvider
from google.genai import types as genai_types
import pandas as pd
import pytest

_NON_ASCII_TEXT = "😀 你好 café"
_real_open = builtins.open


def _make_eval_set() -> EvalSet:
  return EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[EvalCase(eval_id="case1", conversation=[])],
  )


async def _empty_async_gen(*args, **kwargs):
  """An async generator that yields nothing (mocks perform_inference/evaluate)."""
  return
  yield  # pragma: no cover - makes this a generator.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "live_model_config, expected_use_live",
    [
        (LiveModelConfig(timeout_seconds=600), True),
        (None, False),
    ],
)
async def test_get_eval_results_by_eval_id_threads_live_model_config(
    live_model_config, expected_use_live, mocker
):
  """`live_model_config` is forwarded to the InferenceRequest's InferenceConfig."""
  mock_service = mocker.MagicMock()
  mock_service.perform_inference = mocker.MagicMock(
      side_effect=_empty_async_gen
  )
  mock_service.evaluate = mocker.MagicMock(side_effect=_empty_async_gen)
  mocker.patch(
      "google.adk.evaluation.local_eval_service.LocalEvalService",
      return_value=mock_service,
  )

  await AgentEvaluator._get_eval_results_by_eval_id(
      agent_for_eval=mocker.MagicMock(),
      eval_set=_make_eval_set(),
      eval_metrics=[],
      num_runs=1,
      user_simulator_provider=UserSimulatorProvider(),
      app_name="test_app",
      live_model_config=live_model_config,
  )

  # A single inference request should be issued carrying the live flag.
  mock_service.perform_inference.assert_called_once()
  inference_request = mock_service.perform_inference.call_args.kwargs[
      "inference_request"
  ]
  assert inference_request.inference_config.use_live is expected_use_live
  if live_model_config:
    assert inference_request.inference_config.live_timeout_seconds == 600


@pytest.mark.asyncio
async def test_evaluate_eval_set_threads_artifact_service(mocker):
  """The artifact_service passed to evaluate_eval_set reaches LocalEvalService."""
  my_service = InMemoryArtifactService()

  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=mocker.AsyncMock(return_value=(mocker.MagicMock(), None)),
  )

  # LocalEvalService is imported lazily inside _get_eval_results_by_eval_id, so
  # the patch target is its defining module.
  mock_local_eval_service_cls = mocker.patch(
      "google.adk.evaluation.local_eval_service.LocalEvalService"
  )

  async def _empty(*args, **kwargs):
    return
    yield  # Makes this an (empty) async generator.

  instance = mock_local_eval_service_cls.return_value
  instance.perform_inference = _empty
  instance.evaluate = _empty

  await AgentEvaluator.evaluate_eval_set(
      agent_module="my.agent.module",
      eval_set=EvalSet(eval_set_id="es1", eval_cases=[]),
      eval_config=EvalConfig(),
      num_runs=1,
      artifact_service=my_service,
  )

  assert (
      mock_local_eval_service_cls.call_args.kwargs["artifact_service"]
      is my_service
  )


async def _mock_evaluate_eval_set(mocker, eval_case_result: EvalCaseResult):
  """Runs evaluate_eval_set against an eval service yielding the given result."""
  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=mocker.AsyncMock(return_value=(mocker.MagicMock(), None)),
  )
  mock_local_eval_service_cls = mocker.patch(
      "google.adk.evaluation.local_eval_service.LocalEvalService"
  )

  async def _one_result(*args, **kwargs):
    yield eval_case_result

  instance = mock_local_eval_service_cls.return_value
  instance.perform_inference = _empty_async_gen
  instance.evaluate = _one_result

  await AgentEvaluator.evaluate_eval_set(
      agent_module="my.agent.module",
      eval_set=_make_eval_set(),
      eval_config=EvalConfig(),
      num_runs=1,
      print_detailed_results=False,
  )


@pytest.mark.asyncio
async def test_evaluate_eval_set_fails_when_inference_crashed(mocker):
  """A FAILED eval case with no metric results is still reported as a failure.

  This is the shape recorded when inferencing raised: the verdict lives only in
  `final_eval_status`, with no per-invocation metric results to re-derive it
  from.
  """
  crashed_result = EvalCaseResult(
      eval_set_id="test_eval_set",
      eval_id="case1",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id="",
  )

  with pytest.raises(AssertionError, match="case1 for my.agent.module Failed"):
    await _mock_evaluate_eval_set(mocker, crashed_result)


@pytest.mark.asyncio
async def test_evaluate_eval_set_keeps_metric_detail_for_failed_metric(mocker):
  """A metric that scored below threshold still reports the metric detail."""
  failed_metric_result = EvalCaseResult(
      eval_set_id="test_eval_set",
      eval_id="case1",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=Invocation(
                  user_content=_content("What is 2 + 2?"),
                  final_response=_content("5"),
              ),
              expected_invocation=Invocation(
                  user_content=_content("What is 2 + 2?"),
                  final_response=_content("4"),
              ),
              eval_metric_results=[
                  EvalMetricResult(
                      metric_name="response_match_score",
                      threshold=0.8,
                      score=0.1,
                      eval_status=EvalStatus.FAILED,
                  )
              ],
          )
      ],
      session_id="",
  )

  with pytest.raises(AssertionError) as exc_info:
    await _mock_evaluate_eval_set(mocker, failed_metric_result)

  message = str(exc_info.value)
  assert "response_match_score for my.agent.module Failed" in message
  assert "Expected 0.8, but got 0.1" in message
  # The metric detail accounts for the failure; nothing extra is reported.
  assert "no metric results" not in message


@pytest.mark.asyncio
async def test_evaluate_eval_set_passes_when_metrics_pass(mocker):
  """A passing eval case is not turned into a failure."""
  passing_result = EvalCaseResult(
      eval_set_id="test_eval_set",
      eval_id="case1",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=Invocation(
                  user_content=_content("What is 2 + 2?"),
                  final_response=_content("4"),
              ),
              expected_invocation=Invocation(
                  user_content=_content("What is 2 + 2?"),
                  final_response=_content("4"),
              ),
              eval_metric_results=[
                  EvalMetricResult(
                      metric_name="response_match_score",
                      threshold=0.8,
                      score=1.0,
                      eval_status=EvalStatus.PASSED,
                  )
              ],
          )
      ],
      session_id="",
  )

  await _mock_evaluate_eval_set(mocker, passing_result)


class TestGetAgentForEval:
  """Resolution of the wrapping App alongside the agent to evaluate."""

  @pytest.mark.asyncio
  async def test_resolves_app_when_module_exposes_one(self, mocker):
    """When the module's `agent` exposes an `app`, it is returned too."""
    root_agent = BaseAgent(name="root_agent")
    app = App(name="my_app", root_agent=root_agent)
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app=app)
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is app

  @pytest.mark.asyncio
  async def test_returns_none_app_when_module_has_no_app(self, mocker):
    """When only `root_agent` is exposed, app is None."""
    root_agent = BaseAgent(name="root_agent")
    fake_module = SimpleNamespace(agent=SimpleNamespace(root_agent=root_agent))
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is None

  @pytest.mark.asyncio
  async def test_ignores_app_attribute_that_is_not_an_app(self, mocker):
    """A non-App `app` attribute is ignored and app resolves to None."""
    root_agent = BaseAgent(name="root_agent")
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app="not-an-app")
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module"
    )

    assert resolved_agent is root_agent
    assert resolved_app is None

  @pytest.mark.asyncio
  async def test_surfaces_app_even_when_selecting_sub_agent(self, mocker):
    """A sub-agent is returned for eval, but the wrapping App is still surfaced."""
    sub_agent = BaseAgent(name="sub_agent")
    root_agent = BaseAgent(name="root_agent", sub_agents=[sub_agent])
    app = App(name="my_app", root_agent=root_agent)
    fake_module = SimpleNamespace(
        agent=SimpleNamespace(root_agent=root_agent, app=app)
    )
    mocker.patch("importlib.import_module", return_value=fake_module)

    resolved_agent, resolved_app = await AgentEvaluator._get_agent_for_eval(
        module_name="some.module", agent_name="sub_agent"
    )

    assert resolved_agent is sub_agent
    assert resolved_app is app


class TestGetEvalResultsByEvalId:
  """The pytest-gate path forwards the App into LocalEvalService."""

  @staticmethod
  def _empty_async_gen_factory():
    async def _agen(*args, **kwargs):
      return
      yield  # pragma: no cover - marks this as an async generator

    return _agen

  @pytest.mark.asyncio
  async def test_app_is_forwarded_to_local_eval_service(self, mocker):
    """`_get_eval_results_by_eval_id` passes `app=` into LocalEvalService."""
    root_agent = BaseAgent(name="root_agent")
    app = App(name="my_app", root_agent=root_agent)

    mock_service_cls = mocker.patch(
        "google.adk.evaluation.local_eval_service.LocalEvalService"
    )
    mock_service = mock_service_cls.return_value
    mock_service.perform_inference = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )
    mock_service.evaluate = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )

    await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=root_agent,
        eval_set=EvalSet(eval_set_id="set-1", eval_cases=[]),
        eval_metrics=[],
        num_runs=1,
        user_simulator_provider=UserSimulatorProvider(),
        app_name="test_app",
        app=app,
    )

    assert mock_service_cls.call_args.kwargs["app"] is app

  @pytest.mark.asyncio
  async def test_none_app_is_forwarded_by_default(self, mocker):
    """When no App is provided, LocalEvalService receives app=None."""
    root_agent = BaseAgent(name="root_agent")

    mock_service_cls = mocker.patch(
        "google.adk.evaluation.local_eval_service.LocalEvalService"
    )
    mock_service = mock_service_cls.return_value
    mock_service.perform_inference = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )
    mock_service.evaluate = mocker.MagicMock(
        side_effect=self._empty_async_gen_factory()
    )

    await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=root_agent,
        eval_set=EvalSet(eval_set_id="set-1", eval_cases=[]),
        eval_metrics=[],
        num_runs=1,
        user_simulator_provider=UserSimulatorProvider(),
        app_name="test_app",
    )

    assert mock_service_cls.call_args.kwargs["app"] is None


def _content(text: str) -> genai_types.Content:
  return genai_types.Content(parts=[genai_types.Part(text=text)])


def _make_result_with_invocation(
    metric_name: str,
    score: float,
    threshold: float,
    eval_status: EvalStatus,
    prompt: str,
    expected_response: str,
    actual_response: str,
) -> _EvalMetricResultWithInvocation:
  return _EvalMetricResultWithInvocation(
      actual_invocation=Invocation(
          user_content=_content(prompt),
          final_response=_content(actual_response),
      ),
      expected_invocation=Invocation(
          user_content=_content(prompt),
          final_response=_content(expected_response),
      ),
      eval_metric_result=EvalMetricResult(
          metric_name=metric_name,
          threshold=threshold,
          score=score,
          eval_status=eval_status,
      ),
  )


def test_get_results_as_rows_flattens_metrics_and_invocations():
  eval_metric_results = {
      "response_match_score": [
          _make_result_with_invocation(
              metric_name="response_match_score",
              score=1.0,
              threshold=0.8,
              eval_status=EvalStatus.PASSED,
              prompt="What is 2 + 2?",
              expected_response="4",
              actual_response="4",
          ),
          _make_result_with_invocation(
              metric_name="response_match_score",
              score=0.0,
              threshold=0.8,
              eval_status=EvalStatus.FAILED,
              prompt="Capital of France?",
              expected_response="Paris",
              actual_response="London",
          ),
      ],
  }

  rows = AgentEvaluator._get_results_as_rows(
      eval_set_id="my_eval_set",
      eval_id="my_eval_case",
      eval_metric_results=eval_metric_results,
  )

  assert len(rows) == 2
  first = rows[0]
  assert first["eval_set_id"] == "my_eval_set"
  assert first["eval_id"] == "my_eval_case"
  assert first["metric_name"] == "response_match_score"
  assert first["threshold"] == 0.8
  assert first["score"] == 1.0
  assert first["eval_status"] == "PASSED"
  assert first["prompt"] == "What is 2 + 2?"
  assert first["expected_response"] == "4"
  assert first["actual_response"] == "4"

  # Failing invocation should still be captured.
  assert rows[1]["eval_status"] == "FAILED"
  assert rows[1]["actual_response"] == "London"


def test_get_results_as_rows_handles_missing_expected_invocation():
  result = _EvalMetricResultWithInvocation(
      actual_invocation=Invocation(
          user_content=_content("hi"),
          final_response=_content("hello"),
      ),
      expected_invocation=None,
      eval_metric_result=EvalMetricResult(
          metric_name="safety_v1",
          threshold=0.5,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  )

  rows = AgentEvaluator._get_results_as_rows(
      eval_set_id="s",
      eval_id="c",
      eval_metric_results={"safety_v1": [result]},
  )

  assert len(rows) == 1
  assert rows[0]["prompt"] == "hi"
  assert rows[0]["expected_response"] == ""
  assert rows[0]["actual_response"] == "hello"


def test_write_results_to_csv_writes_expected_file(tmp_path):
  rows = [
      {
          "eval_set_id": "s",
          "eval_id": "c",
          "metric_name": "response_match_score",
          "threshold": 0.8,
          "score": 1.0,
          "eval_status": "PASSED",
          "prompt": "What is 2 + 2?",
          "expected_response": "4",
          "actual_response": "4",
          "expected_tool_calls": "",
          "actual_tool_calls": "",
      },
  ]
  output_file = os.path.join(str(tmp_path), "nested", "eval_results.csv")

  AgentEvaluator._write_results_to_csv(rows=rows, output_file=output_file)

  # The nested directory should have been created.
  assert os.path.isfile(output_file)

  df = pd.read_csv(output_file)
  assert list(df.columns) == list(rows[0].keys())
  assert len(df) == 1
  assert df.iloc[0]["metric_name"] == "response_match_score"
  assert df.iloc[0]["eval_status"] == "PASSED"
  assert df.iloc[0]["score"] == 1.0


def test_write_results_to_csv_appends_without_duplicate_header(tmp_path):
  output_file = os.path.join(str(tmp_path), "eval_results.csv")

  def _row(eval_id: str, score: float, status: str) -> dict:
    return {
        "eval_set_id": "s",
        "eval_id": eval_id,
        "metric_name": "response_match_score",
        "threshold": 0.8,
        "score": score,
        "eval_status": status,
        "prompt": "p",
        "expected_response": "e",
        "actual_response": "a",
        "expected_tool_calls": "",
        "actual_tool_calls": "",
    }

  AgentEvaluator._write_results_to_csv(
      rows=[_row("case_1", 1.0, "PASSED")], output_file=output_file
  )
  AgentEvaluator._write_results_to_csv(
      rows=[_row("case_2", 0.0, "FAILED")], output_file=output_file
  )

  df = pd.read_csv(output_file)
  # Two appends should accumulate two rows, with the header written only once.
  assert len(df) == 2
  assert sorted(df["eval_id"].tolist()) == ["case_1", "case_2"]
  assert "eval_id" not in df["eval_id"].tolist()


# -----------------------------------------------------------------------------
# `find_config_for_test_file` -- resolves `test_config.json` from the *folder of
# the test file*, falling back to the built-in default criteria.
# -----------------------------------------------------------------------------


def test_find_config_for_test_file_reads_config_from_test_file_folder(tmp_path):
  """The config is read from `<dir of test file>/test_config.json`."""
  agent_dir = tmp_path / "agent"
  agent_dir.mkdir()
  (agent_dir / "test_config.json").write_text(
      json.dumps({"criteria": {"response_match_score": 0.25}})
  )
  # A decoy in the parent folder must be ignored -- resolution is scoped to the
  # test file's own folder.
  (tmp_path / "test_config.json").write_text(
      json.dumps({"criteria": {"response_match_score": 0.99}})
  )

  eval_config = AgentEvaluator.find_config_for_test_file(
      str(agent_dir / "simple.test.json")
  )

  assert eval_config.criteria == {"response_match_score": 0.25}


def test_find_config_for_test_file_without_config_returns_defaults(tmp_path):
  """With no `test_config.json` alongside, the documented defaults apply."""
  eval_config = AgentEvaluator.find_config_for_test_file(
      str(tmp_path / "simple.test.json")
  )

  assert eval_config.criteria == {
      "tool_trajectory_avg_score": 1.0,
      "response_match_score": 0.8,
  }


# -----------------------------------------------------------------------------
# `migrate_eval_data_to_new_schema` -- converts a pre-EvalSet test file into an
# `EvalSet` json file.
# -----------------------------------------------------------------------------


_OLD_FORMAT_DATA = [{
    "query": "Roll a 6 sided dice",
    "expected_tool_use": [
        {"tool_name": "roll_die", "tool_input": {"sides": 6}}
    ],
    "reference": "I rolled a 4.",
}]


def _write_old_format_file(folder, name="simple.test.json"):
  old_file = folder / name
  old_file.write_text(json.dumps(_OLD_FORMAT_DATA))
  return old_file


@pytest.mark.parametrize(
    "old_file, new_file",
    [("", "new.evalset.json"), ("old.test.json", "")],
)
def test_migrate_eval_data_to_new_schema_empty_path_raises(old_file, new_file):
  """Both file paths are required; an empty one is rejected up front."""
  with pytest.raises(
      ValueError, match="One of old_eval_data_file or new_eval_data_file"
  ):
    AgentEvaluator.migrate_eval_data_to_new_schema(old_file, new_file)


def test_migrate_eval_data_to_new_schema_converts_old_format(tmp_path):
  """Old-format rows become `Invocation`s on a readable `EvalSet` file."""
  old_file = _write_old_format_file(tmp_path)
  new_file = tmp_path / "migrated.evalset.json"

  AgentEvaluator.migrate_eval_data_to_new_schema(str(old_file), str(new_file))

  eval_set = EvalSet.model_validate_json(new_file.read_text())
  assert len(eval_set.eval_cases) == 1
  eval_case = eval_set.eval_cases[0]
  # The old file path is carried through as the eval case id.
  assert eval_case.eval_id == str(old_file)
  assert len(eval_case.conversation) == 1

  invocation = eval_case.conversation[0]
  assert invocation.user_content.parts[0].text == "Roll a 6 sided dice"
  assert invocation.final_response.parts[0].text == "I rolled a 4."
  tool_uses = invocation.intermediate_data.tool_uses
  assert [(t.name, t.args) for t in tool_uses] == [("roll_die", {"sides": 6})]
  # No initial session file was supplied, so no session is pinned.
  assert eval_case.session_input is None


def test_migrate_eval_data_to_new_schema_carries_initial_session(tmp_path):
  """`initial_session_file` becomes the eval case's `session_input`."""
  old_file = _write_old_format_file(tmp_path)
  session_file = tmp_path / "initial.session.json"
  session_file.write_text(
      json.dumps({
          "app_name": "dice_app",
          "user_id": "user_1",
          "state": {"rolls": 2},
      })
  )
  new_file = tmp_path / "migrated.evalset.json"

  AgentEvaluator.migrate_eval_data_to_new_schema(
      str(old_file), str(new_file), str(session_file)
  )

  session_input = (
      EvalSet.model_validate_json(new_file.read_text())
      .eval_cases[0]
      .session_input
  )
  assert session_input.app_name == "dice_app"
  assert session_input.user_id == "user_1"
  assert session_input.state == {"rolls": 2}


def test_migrate_eval_data_to_new_schema_validates_against_old_folder_config(
    tmp_path,
):
  """Criteria are validated using the config next to the *old* data file."""
  old_dir = tmp_path / "old"
  old_dir.mkdir()
  old_file = _write_old_format_file(old_dir)
  # `not_a_metric` is not an allowed criterion, so validation must reject it.
  # This only happens if the config is resolved from `old_dir`.
  (old_dir / "test_config.json").write_text(
      json.dumps({"criteria": {"not_a_metric": 1.0}})
  )

  with pytest.raises(ValueError, match="Invalid criteria key: not_a_metric"):
    AgentEvaluator.migrate_eval_data_to_new_schema(
        str(old_file), str(tmp_path / "migrated.evalset.json")
    )


def test_migrate_eval_data_to_new_schema_missing_reference_rejected(tmp_path):
  """Default criteria require a `reference` column on every row."""
  old_file = tmp_path / "simple.test.json"
  old_file.write_text(
      json.dumps([{"query": "hi", "expected_tool_use": []}]),
  )

  with pytest.raises(ValueError, match="response_match_score"):
    AgentEvaluator.migrate_eval_data_to_new_schema(
        str(old_file), str(tmp_path / "migrated.evalset.json")
    )


@pytest.mark.asyncio
async def test_evaluate_eval_set_forwards_results_manager_and_app_name(mocker):
  """Results manager and resolved app_name are handed to the eval service
  (LocalEvalService), which owns persistence."""
  eval_set = SimpleNamespace(
      eval_set_id="eval_set_1",
      eval_cases=[SimpleNamespace(eval_id="case_a")],
  )

  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=AsyncMock(return_value=(mocker.Mock(), None)),
  )
  mocker.patch(
      "google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config",
      return_value=[],
  )
  get_results_mock = mocker.patch.object(
      AgentEvaluator,
      "_get_eval_results_by_eval_id",
      new=AsyncMock(return_value={}),
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  await AgentEvaluator.evaluate_eval_set(
      agent_module="my.pkg.search_agent",
      eval_set=eval_set,
      eval_config=EvalConfig(criteria={}),
      app_name="custom_app",
      eval_set_results_manager=manager,
      print_detailed_results=False,
  )

  get_results_mock.assert_awaited_once()
  kwargs = get_results_mock.await_args.kwargs
  assert kwargs["app_name"] == "custom_app"
  assert kwargs["eval_set_results_manager"] is manager


@pytest.mark.asyncio
async def test_evaluate_eval_set_persists_before_assert_failure(mocker):
  """Persistence runs inside _get_eval_results_by_eval_id, before the failure
  assertion, so failed eval runs still leave artifacts."""
  eval_set = SimpleNamespace(
      eval_set_id="eval_set_1",
      eval_cases=[SimpleNamespace(eval_id="case_a")],
  )
  eval_result = mocker.Mock(name="eval_result")

  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=AsyncMock(return_value=(mocker.Mock(), None)),
  )
  mocker.patch(
      "google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config",
      return_value=[],
  )
  get_results_mock = mocker.patch.object(
      AgentEvaluator,
      "_get_eval_results_by_eval_id",
      new=AsyncMock(return_value={"case_a": [eval_result]}),
  )
  mocker.patch.object(
      AgentEvaluator,
      "_get_eval_metric_results_with_invocation",
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      "_process_metrics_and_get_failures",
      return_value=["failed"],
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  with pytest.raises(AssertionError):
    await AgentEvaluator.evaluate_eval_set(
        agent_module="pkg.search_agent",
        eval_set=eval_set,
        eval_config=EvalConfig(criteria={}),
        app_name="search_agent",
        eval_set_results_manager=manager,
        print_detailed_results=False,
    )

  get_results_mock.assert_awaited_once()
  assert (
      get_results_mock.await_args.kwargs["eval_set_results_manager"] is manager
  )


@pytest.mark.asyncio
async def test_evaluate_eval_set_requires_app_name_when_manager_given(mocker):
  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)
  with pytest.raises(ValueError, match="app_name is required"):
    await AgentEvaluator.evaluate_eval_set(
        agent_module="pkg.search_agent",
        eval_set=SimpleNamespace(
            eval_set_id="eval_set_1",
            eval_cases=[SimpleNamespace(eval_id="case_a")],
        ),
        eval_config=EvalConfig(criteria={}),
        eval_set_results_manager=manager,
        print_detailed_results=False,
    )


@pytest.mark.asyncio
async def test_evaluate_requires_app_name_when_manager_given(mocker):
  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)
  with pytest.raises(ValueError, match="app_name is required"):
    await AgentEvaluator.evaluate(
        agent_module="pkg.search_agent",
        eval_dataset_file_path_or_dir="some.test.json",
        eval_set_results_manager=manager,
    )


@pytest.mark.asyncio
async def test_evaluate_passes_results_manager_and_app_name(mocker, tmp_path):
  test_dir = tmp_path / "evals"
  nested_dir = test_dir / "nested"
  nested_dir.mkdir(parents=True)

  test_file_1 = test_dir / "a.test.json"
  test_file_2 = nested_dir / "b.test.json"
  test_file_1.write_text("[]", encoding="utf-8")
  test_file_2.write_text("[]", encoding="utf-8")

  eval_config = EvalConfig(criteria={})
  eval_set = SimpleNamespace(eval_set_id="eval_set_1")

  mocker.patch.object(
      AgentEvaluator, "find_config_for_test_file", return_value=eval_config
  )
  mocker.patch.object(
      AgentEvaluator,
      "_load_eval_set_from_file",
      return_value=eval_set,
  )
  evaluate_eval_set_mock = mocker.patch.object(
      AgentEvaluator,
      "evaluate_eval_set",
      new=AsyncMock(),
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  await AgentEvaluator.evaluate(
      agent_module="pkg.search_agent",
      eval_dataset_file_path_or_dir=str(test_dir),
      app_name="custom_app",
      eval_set_results_manager=manager,
      print_detailed_results=False,
  )

  assert evaluate_eval_set_mock.await_count == 2
  for await_call in evaluate_eval_set_mock.await_args_list:
    assert await_call.kwargs["app_name"] == "custom_app"
    assert await_call.kwargs["eval_set_results_manager"] is manager

  called_paths = {
      Path(call.args[0])
      for call in AgentEvaluator.find_config_for_test_file.call_args_list
  }
  assert called_paths == {test_file_1, test_file_2}


@pytest.mark.asyncio
async def test_evaluate_eval_set_keeps_positional_print_detailed_results(
    mocker,
):
  eval_set = SimpleNamespace(
      eval_set_id="eval_set_1",
      eval_cases=[SimpleNamespace(eval_id="case_a")],
  )
  eval_result = mocker.Mock(name="eval_result")

  mocker.patch.object(
      AgentEvaluator,
      "_get_agent_for_eval",
      new=AsyncMock(return_value=(mocker.Mock(), None)),
  )
  mocker.patch(
      "google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config",
      return_value=[],
  )
  mocker.patch.object(
      AgentEvaluator,
      "_get_eval_results_by_eval_id",
      new=AsyncMock(return_value={"case_a": [eval_result]}),
  )
  mocker.patch.object(
      AgentEvaluator,
      "_get_eval_metric_results_with_invocation",
      return_value={},
  )
  process_mock = mocker.patch.object(
      AgentEvaluator,
      "_process_metrics_and_get_failures",
      return_value=[],
  )

  await AgentEvaluator.evaluate_eval_set(
      "pkg.search_agent",
      eval_set,
      None,
      EvalConfig(criteria={}),
      1,
      None,
      False,
  )

  assert process_mock.call_args.kwargs["print_detailed_results"] is False


@pytest.mark.asyncio
async def test_evaluate_keeps_positional_initial_session_file_and_print_flag(
    mocker,
):
  initial_session_mock = mocker.patch.object(
      AgentEvaluator,
      "_get_initial_session",
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      "find_config_for_test_file",
      return_value=EvalConfig(criteria={}),
  )
  mocker.patch.object(
      AgentEvaluator,
      "_load_eval_set_from_file",
      return_value=SimpleNamespace(eval_set_id="eval_set_1"),
  )
  evaluate_eval_set_mock = mocker.patch.object(
      AgentEvaluator,
      "evaluate_eval_set",
      new=AsyncMock(),
  )

  await AgentEvaluator.evaluate(
      "pkg.search_agent",
      "some.test.json",
      1,
      None,
      "initial.session.json",
      False,
  )

  initial_session_mock.assert_called_once_with("initial.session.json")
  evaluate_eval_set_mock.assert_awaited_once()
  assert (
      evaluate_eval_set_mock.await_args.kwargs["print_detailed_results"]
      is False
  )


def _non_utf8_default_open(file, mode="r", *args, **kwargs):
  """Emulates a platform whose default text encoding is not UTF-8.

  On such platforms (for example Windows, where the default is cp1252),
  `open()` calls that omit `encoding=` inherit that non-UTF-8 default. This
  wrapper reproduces that behaviour on any platform by falling back to ASCII
  when a text-mode open does not specify an encoding, so a missing
  `encoding="utf-8"` argument raises instead of silently depending on the
  host locale.
  """
  if "b" not in mode and "encoding" not in kwargs:
    kwargs["encoding"] = "ascii"
  return _real_open(file, mode, *args, **kwargs)


def test_load_json_reads_non_ascii_with_non_utf8_default(tmp_path, mocker):
  """`load_json` must decode eval data as UTF-8 regardless of platform locale."""
  file_path = tmp_path / "eval.json"
  file_path.write_text(
      json.dumps([{"query": _NON_ASCII_TEXT}], ensure_ascii=False),
      encoding="utf-8",
  )

  mocker.patch.object(
      agent_evaluator_module, "open", _non_utf8_default_open, create=True
  )

  assert load_json(str(file_path)) == [{"query": _NON_ASCII_TEXT}]


def test_get_initial_session_reads_non_ascii_with_non_utf8_default(
    tmp_path, mocker
):
  """`_get_initial_session` must decode the session file as UTF-8."""
  session_file = tmp_path / "initial_session.json"
  session_file.write_text(
      json.dumps({"state": {"city": _NON_ASCII_TEXT}}, ensure_ascii=False),
      encoding="utf-8",
  )

  mocker.patch.object(
      agent_evaluator_module, "open", _non_utf8_default_open, create=True
  )

  initial_session = AgentEvaluator._get_initial_session(str(session_file))

  assert initial_session == {"state": {"city": _NON_ASCII_TEXT}}


def test_migrate_eval_data_round_trips_non_ascii_with_non_utf8_default(
    tmp_path, mocker
):
  """Migration must read the old file and write the new file as UTF-8.

  This exercises both the read (`load_json`) and the write
  (`model_dump_json`) of eval data, which must stay UTF-8 consistent so that
  datasets containing non-ASCII characters survive migration on any platform.
  """
  old_eval_data_file = tmp_path / "old_format.test.json"
  old_eval_data_file.write_text(
      json.dumps(
          [{
              "query": _NON_ASCII_TEXT,
              "reference": _NON_ASCII_TEXT,
              "expected_tool_use": [],
          }],
          ensure_ascii=False,
      ),
      encoding="utf-8",
  )
  new_eval_data_file = tmp_path / "new_format.json"

  mocker.patch.object(
      agent_evaluator_module, "open", _non_utf8_default_open, create=True
  )

  AgentEvaluator.migrate_eval_data_to_new_schema(
      str(old_eval_data_file), str(new_eval_data_file)
  )

  migrated = json.loads(new_eval_data_file.read_text(encoding="utf-8"))
  assert _NON_ASCII_TEXT in json.dumps(migrated, ensure_ascii=False)


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-v"]))

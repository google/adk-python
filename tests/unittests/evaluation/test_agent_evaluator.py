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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
import pytest


@pytest.mark.asyncio
async def test_evaluate_eval_set_saves_results_with_explicit_app_name(mocker):
  eval_set = SimpleNamespace(
      eval_set_id='eval_set_1',
      eval_cases=[
          SimpleNamespace(eval_id='case_b'),
          SimpleNamespace(eval_id='case_a'),
      ],
  )
  result_a = mocker.Mock(name='result_a')
  result_b = mocker.Mock(name='result_b')
  result_b_2 = mocker.Mock(name='result_b_2')
  eval_results_by_eval_id = {
      'case_a': [result_a],
      'case_b': [result_b, result_b_2],
  }

  mocker.patch.object(
      AgentEvaluator,
      '_get_agent_for_eval',
      new=AsyncMock(return_value=mocker.Mock()),
  )
  mocker.patch(
      'google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config',
      return_value=[],
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_results_by_eval_id',
      new=AsyncMock(return_value=eval_results_by_eval_id),
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_metric_results_with_invocation',
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      '_process_metrics_and_get_failures',
      return_value=[],
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  await AgentEvaluator.evaluate_eval_set(
      agent_module='my.pkg.search_agent',
      eval_set=eval_set,
      eval_config=EvalConfig(criteria={}),
      app_name='custom_app',
      eval_set_results_manager=manager,
      print_detailed_results=False,
  )

  assert manager.save_eval_set_result.call_count == 3
  assert manager.save_eval_set_result.call_args_list == [
      mocker.call(
          app_name='custom_app',
          eval_set_id='eval_set_1',
          eval_case_results=[result_b],
      ),
      mocker.call(
          app_name='custom_app',
          eval_set_id='eval_set_1',
          eval_case_results=[result_b_2],
      ),
      mocker.call(
          app_name='custom_app',
          eval_set_id='eval_set_1',
          eval_case_results=[result_a],
      ),
  ]


@pytest.mark.asyncio
async def test_evaluate_eval_set_uses_derived_app_name(mocker):
  eval_set = SimpleNamespace(
      eval_set_id='eval_set_1',
      eval_cases=[SimpleNamespace(eval_id='case_a')],
  )
  eval_result = mocker.Mock(name='eval_result')

  mocker.patch.object(
      AgentEvaluator,
      '_get_agent_for_eval',
      new=AsyncMock(return_value=mocker.Mock()),
  )
  mocker.patch(
      'google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config',
      return_value=[],
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_results_by_eval_id',
      new=AsyncMock(return_value={'case_a': [eval_result]}),
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_metric_results_with_invocation',
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      '_process_metrics_and_get_failures',
      return_value=[],
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  await AgentEvaluator.evaluate_eval_set(
      agent_module='pkg.search_agent.agent',
      eval_set=eval_set,
      eval_config=EvalConfig(criteria={}),
      eval_set_results_manager=manager,
      print_detailed_results=False,
  )

  manager.save_eval_set_result.assert_called_once_with(
      app_name='search_agent',
      eval_set_id='eval_set_1',
      eval_case_results=[eval_result],
  )


@pytest.mark.asyncio
async def test_evaluate_eval_set_saves_before_assert_failure(mocker):
  eval_set = SimpleNamespace(
      eval_set_id='eval_set_1',
      eval_cases=[SimpleNamespace(eval_id='case_a')],
  )
  eval_result = mocker.Mock(name='eval_result')

  mocker.patch.object(
      AgentEvaluator,
      '_get_agent_for_eval',
      new=AsyncMock(return_value=mocker.Mock()),
  )
  mocker.patch(
      'google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config',
      return_value=[],
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_results_by_eval_id',
      new=AsyncMock(return_value={'case_a': [eval_result]}),
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_metric_results_with_invocation',
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      '_process_metrics_and_get_failures',
      return_value=['failed'],
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  with pytest.raises(AssertionError):
    await AgentEvaluator.evaluate_eval_set(
        agent_module='pkg.search_agent',
        eval_set=eval_set,
        eval_config=EvalConfig(criteria={}),
        eval_set_results_manager=manager,
        print_detailed_results=False,
    )

  manager.save_eval_set_result.assert_called_once_with(
      app_name='search_agent',
      eval_set_id='eval_set_1',
      eval_case_results=[eval_result],
  )


@pytest.mark.asyncio
async def test_evaluate_passes_results_manager_and_app_name(mocker, tmp_path):
  test_dir = tmp_path / 'evals'
  nested_dir = test_dir / 'nested'
  nested_dir.mkdir(parents=True)

  test_file_1 = test_dir / 'a.test.json'
  test_file_2 = nested_dir / 'b.test.json'
  test_file_1.write_text('[]', encoding='utf-8')
  test_file_2.write_text('[]', encoding='utf-8')

  eval_config = EvalConfig(criteria={})
  eval_set = SimpleNamespace(eval_set_id='eval_set_1')

  mocker.patch.object(
      AgentEvaluator, 'find_config_for_test_file', return_value=eval_config
  )
  mocker.patch.object(
      AgentEvaluator,
      '_load_eval_set_from_file',
      return_value=eval_set,
  )
  evaluate_eval_set_mock = mocker.patch.object(
      AgentEvaluator,
      'evaluate_eval_set',
      new=AsyncMock(),
  )

  manager = mocker.create_autospec(EvalSetResultsManager, instance=True)

  await AgentEvaluator.evaluate(
      agent_module='pkg.search_agent',
      eval_dataset_file_path_or_dir=str(test_dir),
      app_name='custom_app',
      eval_set_results_manager=manager,
      print_detailed_results=False,
  )

  assert evaluate_eval_set_mock.await_count == 2
  for await_call in evaluate_eval_set_mock.await_args_list:
    assert await_call.kwargs['app_name'] == 'custom_app'
    assert await_call.kwargs['eval_set_results_manager'] is manager

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
      eval_set_id='eval_set_1',
      eval_cases=[SimpleNamespace(eval_id='case_a')],
  )
  eval_result = mocker.Mock(name='eval_result')

  mocker.patch.object(
      AgentEvaluator,
      '_get_agent_for_eval',
      new=AsyncMock(return_value=mocker.Mock()),
  )
  mocker.patch(
      'google.adk.evaluation.agent_evaluator.get_eval_metrics_from_config',
      return_value=[],
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_results_by_eval_id',
      new=AsyncMock(return_value={'case_a': [eval_result]}),
  )
  mocker.patch.object(
      AgentEvaluator,
      '_get_eval_metric_results_with_invocation',
      return_value={},
  )
  process_mock = mocker.patch.object(
      AgentEvaluator,
      '_process_metrics_and_get_failures',
      return_value=[],
  )

  await AgentEvaluator.evaluate_eval_set(
      'pkg.search_agent',
      eval_set,
      None,
      EvalConfig(criteria={}),
      1,
      None,
      False,
  )

  assert process_mock.call_args.kwargs['print_detailed_results'] is False


@pytest.mark.asyncio
async def test_evaluate_keeps_positional_initial_session_file_and_print_flag(
    mocker,
):
  initial_session_mock = mocker.patch.object(
      AgentEvaluator,
      '_get_initial_session',
      return_value={},
  )
  mocker.patch.object(
      AgentEvaluator,
      'find_config_for_test_file',
      return_value=EvalConfig(criteria={}),
  )
  mocker.patch.object(
      AgentEvaluator,
      '_load_eval_set_from_file',
      return_value=SimpleNamespace(eval_set_id='eval_set_1'),
  )
  evaluate_eval_set_mock = mocker.patch.object(
      AgentEvaluator,
      'evaluate_eval_set',
      new=AsyncMock(),
  )

  await AgentEvaluator.evaluate(
      'pkg.search_agent',
      'some.test.json',
      1,
      None,
      'initial.session.json',
      False,
  )

  initial_session_mock.assert_called_once_with('initial.session.json')
  evaluate_eval_set_mock.assert_awaited_once()
  assert (
      evaluate_eval_set_mock.await_args.kwargs['print_detailed_results']
      is False
  )

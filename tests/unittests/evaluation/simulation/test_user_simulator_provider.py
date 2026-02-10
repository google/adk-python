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

from google.adk.evaluation import conversation_scenarios
from google.adk.evaluation import eval_case
from google.adk.evaluation.simulation import user_simulator_provider
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.genai import types
import pytest

_TEST_CONVERSATION = [
    eval_case.Invocation(
        invocation_id='inv1',
        user_content=types.Content(parts=[types.Part(text='Hello!')]),
    ),
]

_TEST_CONVERSATION_SCENARIO = conversation_scenarios.ConversationScenario(
    starting_prompt='Hello!', conversation_plan='test plan'
)


class TestUserSimulatorProvider:
  """Test cases for the UserSimulatorProvider."""

  def test_provide_static_user_simulator(self):
    """Tests the case when a StaticUserSimulator should be provided."""
    provider = user_simulator_provider.UserSimulatorProvider()
    test_eval_case = eval_case.EvalCase(
        eval_id='test_eval_id',
        conversation=_TEST_CONVERSATION,
    )
    simulator = provider.provide(test_eval_case)
    assert isinstance(simulator, StaticUserSimulator)
    assert simulator.static_conversation == _TEST_CONVERSATION

  def test_provide_llm_backed_user_simulator(self, mocker):
    """Tests the case when a LlmBackedUserSimulator should be provided."""
    mock_llm_registry = mocker.patch(
        'google.adk.evaluation.simulation.llm_backed_user_simulator.LLMRegistry',
        autospec=True,
    )
    mock_llm_registry.return_value.resolve.return_value = mocker.Mock()
    # Test case 1: No config in provider.
    provider = user_simulator_provider.UserSimulatorProvider()
    test_eval_case = eval_case.EvalCase(
        eval_id='test_eval_id',
        conversation_scenario=_TEST_CONVERSATION_SCENARIO,
    )
    simulator = provider.provide(test_eval_case)
    assert isinstance(simulator, LlmBackedUserSimulator)
    assert simulator._conversation_scenario == _TEST_CONVERSATION_SCENARIO

    # Test case 2: Config in provider.
    llm_config = LlmBackedUserSimulatorConfig(
        model='test_model',
    )
    provider = user_simulator_provider.UserSimulatorProvider(
        user_simulator_config=llm_config
    )
    simulator = provider.provide(test_eval_case)
    assert isinstance(simulator, LlmBackedUserSimulator)
    assert simulator._conversation_scenario == _TEST_CONVERSATION_SCENARIO
    assert simulator._config.model == 'test_model'

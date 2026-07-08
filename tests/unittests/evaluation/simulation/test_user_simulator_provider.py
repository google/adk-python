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
from google.adk.evaluation.simulation import user_simulator as user_simulator_module
from google.adk.evaluation.simulation import user_simulator_provider
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.simulation.llm_backed_user_simulator import LlmBackedUserSimulatorConfig
from google.adk.evaluation.simulation.static_user_simulator import StaticUserSimulator
from google.adk.evaluation.simulation.user_simulator import BaseUserSimulatorConfig
from google.genai import types
from pydantic import Field
import pytest
from typing_extensions import Literal

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

  # ---------------------------------------------------------------------------
  # Backward-compat + discriminator + registry
  # ---------------------------------------------------------------------------

  def test_init_accepts_bare_base_config_but_provide_raises(self):
    """A bare `BaseUserSimulatorConfig` is a valid *instance* of the base,

    so `__init__` accepts it -- but the base has no registered simulator,
    so `provide()` should raise a clear error pointing the caller at the
    fix. Callers wanting the default should pass `None` instead.
    """
    provider = user_simulator_provider.UserSimulatorProvider(
        user_simulator_config=BaseUserSimulatorConfig()
    )
    # __init__ stores the bare base as-is; no silent up-conversion.
    assert type(provider._user_simulator_config) is BaseUserSimulatorConfig

    test_eval_case = eval_case.EvalCase(
        eval_id='test_eval_id',
        conversation_scenario=_TEST_CONVERSATION_SCENARIO,
    )
    with pytest.raises(
        ValueError,
        match=(
            r'No UserSimulator registered for config type'
            r' `BaseUserSimulatorConfig`'
        ),
    ):
      provider.provide(test_eval_case)

  def test_init_rejects_non_config_argument(self):
    """Passing something that isn't a `BaseUserSimulatorConfig` should raise

    a clear ValueError.
    """
    with pytest.raises(
        ValueError,
        match=r'Expect config of type `.*BaseUserSimulatorConfig.*`\.',
    ):
      user_simulator_provider.UserSimulatorProvider(
          user_simulator_config='not a config'  # type: ignore[arg-type]
      )

  # NOTE: The "both / neither of conversation, conversation_scenario"
  # checks in `provide()` are defensive; `EvalCase` itself enforces the same
  # invariant at construction time via a model_validator, so those branches
  # in the provider are effectively unreachable and don't warrant a unit
  # test at this layer.

  def test_base_config_type_defaults_to_none(self):
    """The base `BaseUserSimulatorConfig.type` must default to `None` -- the

    base class must not hard-code a specific subclass's discriminator value.
    Concrete subclasses supply their own `Literal[...]` default.
    """
    base = BaseUserSimulatorConfig()
    assert base.type is None

  def test_llm_backed_config_has_locked_type_literal(self):
    """The `type` discriminator on `LlmBackedUserSimulatorConfig` must be a

    Literal locked to `"llm_backed"`, so future subclasses can dispatch
    correctly via pydantic's discriminated union.
    """
    config = LlmBackedUserSimulatorConfig()
    assert config.type == 'llm_backed'
    # Attempting to construct with a different `type` value must fail
    # validation (Literal constraint).
    with pytest.raises(Exception):
      LlmBackedUserSimulatorConfig(type='something_else')

  def test_llm_backed_user_simulator_registered_by_provider_module(self):
    """Importing `user_simulator_provider` must wire the built-in

    `LlmBackedUserSimulator` into the shared registry. This is the "batteries
    included" contract callers rely on: they can `UserSimulatorProvider()`
    without ever touching `register_user_simulator(...)`. If the registration
    line at the top of the provider module is removed, this test catches it
    immediately -- otherwise dispatch would silently fall through to the
    "unregistered config type" error path.
    """
    assert (
        user_simulator_module._SIMULATOR_BY_CONFIG_TYPE.get(
            LlmBackedUserSimulatorConfig
        )
        is LlmBackedUserSimulator
    )

  def test_provide_raises_for_unregistered_config_type(self, mocker):
    """If the caller supplies a config subclass that no one has registered,

    provide() must raise a clear error naming the offending type.
    """
    mocker.patch(
        'google.adk.evaluation.simulation.llm_backed_user_simulator.LLMRegistry',
        autospec=True,
    )

    class _UnregisteredConfig(BaseUserSimulatorConfig):
      type: Literal['unregistered'] = Field(default='unregistered')

    provider = user_simulator_provider.UserSimulatorProvider(
        user_simulator_config=_UnregisteredConfig()
    )
    test_eval_case = eval_case.EvalCase(
        eval_id='test_eval_id',
        conversation_scenario=_TEST_CONVERSATION_SCENARIO,
    )
    with pytest.raises(
        ValueError,
        match=(
            r'No UserSimulator registered for config type'
            r' `_UnregisteredConfig`'
        ),
    ):
      provider.provide(test_eval_case)

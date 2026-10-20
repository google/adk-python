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

"""Unit tests for AgentMode and agent mode type aliases."""

from __future__ import annotations

from google.adk.agents._managed_agent import ManagedAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.utils._agent_mode import AgentMode
from google.adk.utils._agent_mode import DELEGATED_TASK_MODES
from google.adk.workflow._llm_agent_wrapper import _effective_llm_agent_mode
from pydantic import ValidationError
import pytest


def test_agent_mode_values_match_plain_strings():
  """AgentMode members compare and format equal to their plain string values."""
  assert AgentMode.CHAT == 'chat'
  assert AgentMode.TASK == 'task'
  assert AgentMode.SINGLE_TURN == 'single_turn'
  assert str(AgentMode.CHAT) == 'chat'
  assert str(AgentMode.TASK) == 'task'
  assert str(AgentMode.SINGLE_TURN) == 'single_turn'
  assert isinstance(AgentMode.CHAT, str)
  assert isinstance(AgentMode.TASK, str)
  assert isinstance(AgentMode.SINGLE_TURN, str)


def test_delegated_task_modes_membership():
  """DELEGATED_TASK_MODES contains task and single_turn, excluding chat."""
  assert AgentMode.TASK in DELEGATED_TASK_MODES
  assert AgentMode.SINGLE_TURN in DELEGATED_TASK_MODES
  assert 'task' in DELEGATED_TASK_MODES
  assert 'single_turn' in DELEGATED_TASK_MODES
  assert AgentMode.CHAT not in DELEGATED_TASK_MODES
  assert 'chat' not in DELEGATED_TASK_MODES


@pytest.mark.parametrize(
    ('mode_input', 'expected'),
    [
        ('chat', AgentMode.CHAT),
        ('task', AgentMode.TASK),
        ('single_turn', AgentMode.SINGLE_TURN),
        (AgentMode.CHAT, AgentMode.CHAT),
        (AgentMode.TASK, AgentMode.TASK),
        (AgentMode.SINGLE_TURN, AgentMode.SINGLE_TURN),
    ],
)
def test_llm_agent_accepts_enum_and_string_modes(mode_input, expected):
  """LlmAgent.mode accepts both plain strings and AgentMode enum members."""
  agent = LlmAgent(name='worker', mode=mode_input)

  assert agent.mode == expected
  assert agent.mode == expected.value
  assert type(agent.mode) is str  # pylint: disable=unidiomatic-typecheck


def test_llm_agent_sub_agent_default_mode_stores_plain_string():
  """Sub-agents with mode=None default to plain string 'chat'."""
  sub = LlmAgent(name='sub')
  _ = LlmAgent(name='parent', sub_agents=[sub])

  assert sub.mode == 'chat'
  assert type(sub.mode) is str  # pylint: disable=unidiomatic-typecheck


def test_effective_llm_agent_mode_returns_plain_string():
  """_effective_llm_agent_mode returns plain str so model_copy stores str."""
  standalone = LlmAgent(name='standalone')
  sub = LlmAgent(name='sub')
  parent = LlmAgent(name='parent', sub_agents=[sub])

  standalone_mode = _effective_llm_agent_mode(standalone)
  parent_mode = _effective_llm_agent_mode(parent)

  assert standalone_mode == 'single_turn'
  assert type(standalone_mode) is str  # pylint: disable=unidiomatic-typecheck
  assert parent_mode == 'chat'
  assert type(parent_mode) is str  # pylint: disable=unidiomatic-typecheck


def test_llm_agent_rejects_invalid_mode():
  """LlmAgent.mode rejects unsupported mode strings."""
  with pytest.raises(ValidationError):
    LlmAgent(name='worker', mode='invalid_mode')  # type: ignore[arg-type]


def test_managed_agent_accepts_single_turn_enum_and_rejects_other_modes():
  """ManagedAgent.mode accepts AgentMode.SINGLE_TURN and normalizes to str."""
  agent = ManagedAgent(
      name='managed',
      agent_id='agents/test',
      mode=AgentMode.SINGLE_TURN,
  )
  assert agent.mode == AgentMode.SINGLE_TURN
  assert agent.mode == 'single_turn'
  assert type(agent.mode) is str  # pylint: disable=unidiomatic-typecheck

  with pytest.raises(ValidationError):
    ManagedAgent(
        name='managed',
        agent_id='agents/test',
        mode=AgentMode.CHAT,  # type: ignore[arg-type]
    )

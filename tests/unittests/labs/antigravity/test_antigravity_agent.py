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

"""Tests for AntigravityAgent.

Verifies the root-only construction constraint that keeps the agent usable only
as a standalone root agent while the SDK supports local mode only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.labs.antigravity import _antigravity_agent
from google.adk.labs.antigravity._antigravity_agent import AntigravityAgent
from google.antigravity import LocalAgentConfig
import pytest


def _make_config(**kwargs) -> LocalAgentConfig:
  """Returns a minimal real LocalAgentConfig for the wrapped SDK agent."""
  return LocalAgentConfig(system_instructions='test', **kwargs)


def test_standalone_agent_is_allowed():
  """An AntigravityAgent with no parent and no sub-agents constructs cleanly."""
  agent = AntigravityAgent(name='agy', config=_make_config())

  assert agent.parent_agent is None
  assert agent.sub_agents == []


def test_giving_sub_agents_is_rejected():
  """Constructing with sub-agents raises a temporary root-only error."""
  child = BaseAgent(name='child')

  with pytest.raises(ValueError, match='standalone root agent'):
    AntigravityAgent(name='agy', config=_make_config(), sub_agents=[child])


def test_using_as_sub_agent_is_rejected():
  """Adopting the agent under a parent raises a temporary root-only error."""
  agy = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='standalone root agent'):
    BaseAgent(name='parent', sub_agents=[agy])


@pytest.mark.asyncio
async def test_run_without_save_dir_raises():
  """Running without config.save_dir raises, since trajectories need a folder."""
  agent = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='requires config.save_dir'):
    async for _ in agent._run_async_impl(MagicMock()):
      pass


@pytest.mark.asyncio
async def test_resumed_replayed_steps_are_skipped(tmp_path):
  """On resume, steps at or below the resume index are not re-emitted."""
  from google.antigravity import types as sdk_types

  def _step(step_index: int, text: str):
    step = MagicMock()
    step.step_index = step_index
    step.source = sdk_types.StepSource.MODEL
    step.type = sdk_types.StepType.TEXT_RESPONSE
    step.status = sdk_types.StepStatus.DONE
    step.is_complete_response = True
    step.content = text
    step.tool_calls = []
    return step

  # The harness replays steps 0-1 (prior turn) then emits step 2 (this turn).
  async def _receive_steps():
    yield _step(0, 'old-1')
    yield _step(1, 'old-2')
    yield _step(2, 'new')

  conversation = MagicMock()
  conversation.send = AsyncMock()
  conversation.receive_steps = _receive_steps
  active_agent = MagicMock()
  active_agent.conversation = conversation
  active_agent.conversation_id = 'sess_456_agy'
  active_agent.__aenter__ = AsyncMock(return_value=active_agent)
  active_agent.__aexit__ = AsyncMock(return_value=None)

  # A prior trajectory + resume index in save_dir triggers resume at index 1.
  save_dir = tmp_path
  (save_dir / 'traj-sess_456_agy').write_bytes(b'data')
  (save_dir / 'traj-sess_456_agy.resume').write_text('1')
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(save_dir))
  )

  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  ctx.session.id = 'sess_456'
  ctx.user_content = None
  ctx.run_config = None

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [event async for event in agent._run_async_impl(ctx)]

  texts = [e.content.parts[0].text for e in events]
  assert texts == ['new']

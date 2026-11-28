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

"""Tests for the description an AgentTool declares for an agent that reads it lazily.

`RemoteA2aAgent` only learns its description from the remote agent's card, so
`agent.description` is empty when the model request is built. ADK resolves it
through `_get_transfer_description` when the agent is a `transfer_to_agent`
target; these tests cover the same for an agent exposed as a tool (`AgentTool`,
or the delegation tool of a task-mode sub-agent).
"""

from unittest.mock import AsyncMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.agent_tool import _TaskAgentTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
import pytest

from .. import testing_utils

_TASK_SUFFIX = (
    '\nIMPORTANT: This tool delegates execution to a specialized agent. Do NOT'
    ' call this tool in parallel with any other tools.'
)


class _LazyDescriptionAgent(BaseAgent):
  """An agent whose description is only available asynchronously."""

  resolved: str = 'resolved description'
  fails: bool = False

  async def _get_transfer_description(self, ctx) -> str:
    if self.fails:
      raise RuntimeError('the card cannot be reached')
    return self.resolved


async def _declare(tool: AgentTool, invocation_context=None):
  """Runs `process_llm_request` and returns the declaration it added."""
  if invocation_context is None:
    root = Agent(
        name='root', model=testing_utils.MockModel.create(responses=[])
    )
    invocation_context = await testing_utils.create_invocation_context(root)
  llm_request = LlmRequest()
  await tool.process_llm_request(
      tool_context=ToolContext(invocation_context), llm_request=llm_request
  )
  declarations = llm_request.config.tools[0].function_declarations
  assert len(declarations) == 1
  return declarations[0]


@pytest.mark.asyncio
async def test_agent_tool_declares_the_lazily_resolved_description():
  agent = _LazyDescriptionAgent(name='lazy')
  assert agent.description == ''

  declaration = await _declare(AgentTool(agent))

  assert declaration.name == 'lazy'
  assert declaration.description == 'resolved description'
  # The agent itself is not changed: the description is per request.
  assert agent.description == ''


@pytest.mark.asyncio
async def test_task_tool_declares_the_resolved_description_before_its_suffix():
  agent = _LazyDescriptionAgent(name='lazy')

  declaration = await _declare(_TaskAgentTool(agent))

  assert declaration.description == f'resolved description{_TASK_SUFFIX}'
  assert agent.description == ''


@pytest.mark.asyncio
async def test_task_tool_without_a_description_keeps_only_its_suffix():
  agent = _LazyDescriptionAgent(name='lazy', fails=True)

  declaration = await _declare(_TaskAgentTool(agent))

  assert declaration.description == _TASK_SUFFIX.strip()


@pytest.mark.asyncio
async def test_agent_tool_falls_back_to_the_local_description_on_failure():
  agent = _LazyDescriptionAgent(
      name='lazy', description='local description', fails=True
  )

  declaration = await _declare(AgentTool(agent))

  assert declaration.description == 'local description'


@pytest.mark.asyncio
async def test_agent_tool_leaves_an_ordinary_agent_unchanged():
  agent = Agent(
      name='plain',
      description='plain description',
      model=testing_utils.MockModel.create(responses=[]),
  )
  assert not hasattr(agent, '_get_transfer_description')

  declaration = await _declare(AgentTool(agent))

  assert declaration.description == 'plain description'


@pytest.mark.asyncio
async def test_description_is_resolved_once_per_invocation():
  agent = _LazyDescriptionAgent(name='lazy')
  hook = AsyncMock(return_value='resolved description')
  object.__setattr__(agent, '_get_transfer_description', hook)
  root = Agent(name='root', model=testing_utils.MockModel.create(responses=[]))
  invocation_context = await testing_utils.create_invocation_context(root)

  first = await _declare(AgentTool(agent), invocation_context)
  second = await _declare(_TaskAgentTool(agent), invocation_context)

  assert first.description == 'resolved description'
  assert second.description == f'resolved description{_TASK_SUFFIX}'
  assert hook.await_count == 1

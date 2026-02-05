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

"""Behavioral tests for agent transfer system instructions.

These tests verify the behavior of the agent transfer system by calling
the request processor and checking the resulting system instructions not just
implementation.
"""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.flows.llm_flows import agent_transfer
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.runners import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

from ... import testing_utils


async def create_test_invocation_context(agent: Agent) -> InvocationContext:
  """Helper to create constructed InvocationContext."""
  session_service = InMemorySessionService()
  memory_service = InMemoryMemoryService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )

  return InvocationContext(
      artifact_service=InMemoryArtifactService(),
      session_service=session_service,
      memory_service=memory_service,
      plugin_manager=PluginManager(plugins=[]),
      invocation_id='test_invocation_id',
      agent=agent,
      session=session,
      user_content=types.Content(
          role='user', parts=[types.Part.from_text(text='test')]
      ),
      run_config=RunConfig(),
  )


@pytest.mark.asyncio
async def test_agent_transfer_includes_sorted_agent_names_in_system_instructions():
  """Test that agent transfer adds NOTE with sorted agent names to system instructions."""
  mockModel = testing_utils.MockModel.create(responses=[])

  # Create agents with names that will test alphabetical sorting
  z_agent = Agent(name='z_agent', model=mockModel, description='Last agent')
  a_agent = Agent(name='a_agent', model=mockModel, description='First agent')
  m_agent = Agent(name='m_agent', model=mockModel, description='Middle agent')
  peer_agent = Agent(
      name='peer_agent', model=mockModel, description='Peer agent'
  )

  # Create parent agent with a peer agent
  parent_agent = Agent(
      name='parent_agent',
      model=mockModel,
      sub_agents=[peer_agent],
      description='Parent agent',
  )

  # Create main agent with sub-agents and parent (intentionally unsorted order)
  main_agent = Agent(
      name='main_agent',
      model=mockModel,
      sub_agents=[z_agent, a_agent, m_agent],  # Unsorted input
      parent_agent=parent_agent,
      description='Main coordinating agent',
  )

  # Create test context and LLM request
  invocation_context = await create_test_invocation_context(main_agent)
  llm_request = LlmRequest()

  # Call the actual agent transfer request processor (this behavior we're testing)
  async for _ in agent_transfer.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Check on the behavior: verify system instructions contain sorted agent names
  instructions = llm_request.config.system_instruction

  # The NOTE should contain agents in alphabetical order: sub-agents + parent + peers
  expected_content = """\

You have a list of other agents to transfer to:


Agent name: z_agent
Agent description: Last agent


Agent name: a_agent
Agent description: First agent


Agent name: m_agent
Agent description: Middle agent


Agent name: parent_agent
Agent description: Parent agent


Agent name: peer_agent
Agent description: Peer agent


If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `transfer_to_agent` function are
`a_agent`, `m_agent`, `parent_agent`, `peer_agent`, `z_agent`.

If neither you nor the other agents are best for the question, transfer to your parent agent parent_agent."""

  assert expected_content in instructions


@pytest.mark.asyncio
async def test_agent_transfer_system_instructions_without_parent():
  """Test system instructions when agent has no parent."""
  mockModel = testing_utils.MockModel.create(responses=[])

  # Create agents without parent
  sub_agent_1 = Agent(
      name='agent1', model=mockModel, description='First sub-agent'
  )
  sub_agent_2 = Agent(
      name='agent2', model=mockModel, description='Second sub-agent'
  )

  main_agent = Agent(
      name='main_agent',
      model=mockModel,
      sub_agents=[sub_agent_1, sub_agent_2],
      # No parent_agent
      description='Main agent without parent',
  )

  # Create test context and LLM request
  invocation_context = await create_test_invocation_context(main_agent)
  llm_request = LlmRequest()

  # Call the agent transfer request processor
  async for _ in agent_transfer.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Assert behavior: should only include sub-agents in NOTE, no parent
  instructions = llm_request.config.system_instruction

  # Direct multiline string assertion showing the exact expected content
  expected_content = """\

You have a list of other agents to transfer to:


Agent name: agent1
Agent description: First sub-agent


Agent name: agent2
Agent description: Second sub-agent


If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `transfer_to_agent` function are
`agent1`, `agent2`."""

  assert expected_content in instructions


@pytest.mark.asyncio
async def test_agent_transfer_simplified_parent_instructions():
  """Test that parent agent instructions are simplified and not verbose."""
  mockModel = testing_utils.MockModel.create(responses=[])

  # Create agent with parent
  sub_agent = Agent(name='sub_agent', model=mockModel, description='Sub agent')
  parent_agent = Agent(
      name='parent_agent', model=mockModel, description='Parent agent'
  )

  main_agent = Agent(
      name='main_agent',
      model=mockModel,
      sub_agents=[sub_agent],
      parent_agent=parent_agent,
      description='Main agent with parent',
  )

  # Create test context and LLM request
  invocation_context = await create_test_invocation_context(main_agent)
  llm_request = LlmRequest()

  # Call the agent transfer request processor
  async for _ in agent_transfer.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Assert behavior: parent instructions should be simplified
  instructions = llm_request.config.system_instruction

  # Direct multiline string assertion showing the exact expected content
  expected_content = """\

You have a list of other agents to transfer to:


Agent name: sub_agent
Agent description: Sub agent


Agent name: parent_agent
Agent description: Parent agent


If you are the best to answer the question according to your description,
you can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the question to that
agent. When transferring, do not generate any text other than the function
call.

**NOTE**: the only available agents for `transfer_to_agent` function are
`parent_agent`, `sub_agent`.

If neither you nor the other agents are best for the question, transfer to your parent agent parent_agent."""

  assert expected_content in instructions


@pytest.mark.asyncio
async def test_agent_transfer_no_instructions_when_no_transfer_targets():
  """Test that no instructions are added when there are no transfer targets."""
  mockModel = testing_utils.MockModel.create(responses=[])

  # Create agent with no sub-agents and no parent
  main_agent = Agent(
      name='main_agent',
      model=mockModel,
      # No sub_agents, no parent_agent
      description='Isolated agent',
  )

  # Create test context and LLM request
  invocation_context = await create_test_invocation_context(main_agent)
  llm_request = LlmRequest()
  original_system_instruction = llm_request.config.system_instruction

  # Call the agent transfer request processor
  async for _ in agent_transfer.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Assert behavior: no instructions should be added
  assert llm_request.config.system_instruction == original_system_instruction

  instructions = llm_request.config.system_instruction or ''
  assert '**NOTE**:' not in instructions
  assert 'transfer_to_agent' not in instructions

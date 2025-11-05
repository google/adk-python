# Copyright 2025 Google LLC
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

"""Tests for basic LLM request processor."""

from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from pydantic import BaseModel
from pydantic import Field
import pytest


class OutputSchema(BaseModel):
  """Test schema for output."""

  name: str = Field(description='A name')
  value: int = Field(description='A value')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Result: {query}'


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


class TestBasicLlmRequestProcessor:
  """Test class for _BasicLlmRequestProcessor."""

  @pytest.mark.asyncio
  async def test_sets_output_schema_when_no_tools(self):
    """Test that processor sets output_schema when agent has no tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        output_schema=OutputSchema,
        tools=[],  # No tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set response_schema since agent has no tools
    assert llm_request.config.response_schema == OutputSchema
    assert llm_request.config.response_mime_type == 'application/json'

  @pytest.mark.asyncio
  async def test_skips_output_schema_when_tools_present(self, mocker):
    """Test that processor skips output_schema when agent has tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        output_schema=OutputSchema,
        tools=[FunctionTool(func=dummy_tool)],  # Has tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    can_use_output_schema_with_tools = mocker.patch(
        'google.adk.flows.llm_flows.basic.can_use_output_schema_with_tools',
        mock.MagicMock(return_value=False),
    )

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should NOT have set response_schema since agent has tools
    assert llm_request.config.response_schema is None
    assert llm_request.config.response_mime_type != 'application/json'

    # Should have checked if output schema can be used with tools
    can_use_output_schema_with_tools.assert_called_once_with(agent.model)

  @pytest.mark.asyncio
  async def test_sets_output_schema_when_tools_present(self, mocker):
    """Test that processor skips output_schema when agent has tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        output_schema=OutputSchema,
        tools=[FunctionTool(func=dummy_tool)],  # Has tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    can_use_output_schema_with_tools = mocker.patch(
        'google.adk.flows.llm_flows.basic.can_use_output_schema_with_tools',
        mock.MagicMock(return_value=True),
    )

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set response_schema since output schema can be used with tools
    assert llm_request.config.response_schema == OutputSchema
    assert llm_request.config.response_mime_type == 'application/json'

    # Should have checked if output schema can be used with tools
    can_use_output_schema_with_tools.assert_called_once_with(agent.model)

  @pytest.mark.asyncio
  async def test_no_output_schema_no_tools(self):
    """Test that processor works normally when agent has no output_schema or tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        # No output_schema, no tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should not have set anything
    assert llm_request.config.response_schema is None
    assert llm_request.config.response_mime_type != 'application/json'

  @pytest.mark.asyncio
  async def test_sets_model_name(self):
    """Test that processor sets the model name correctly."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set the model name
    assert llm_request.model == 'gemini-1.5-flash'

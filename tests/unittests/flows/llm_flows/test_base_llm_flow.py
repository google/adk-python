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

"""Unit tests for BaseLlmFlow toolset integration."""

from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types
import pytest

from ... import testing_utils


class BaseLlmFlowForTesting(BaseLlmFlow):
  """Test implementation of BaseLlmFlow for testing purposes."""

  pass


@pytest.mark.asyncio
async def test_preprocess_calls_toolset_process_llm_request():
  """Test that _preprocess_async calls process_llm_request on toolsets."""

  # Create a mock toolset that tracks if process_llm_request was called
  class _MockToolset(BaseToolset):

    def __init__(self):
      super().__init__()
      self.process_llm_request_called = False
      self.process_llm_request = AsyncMock(side_effect=self._track_call)

    async def _track_call(self, **kwargs):
      self.process_llm_request_called = True

    async def get_tools(self, readonly_context=None):
      return []

    async def close(self):
      pass

  mock_toolset = _MockToolset()

  # Create a mock model that returns a simple response
  mock_response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='Test response')]
      ),
      partial=False,
  )

  mock_model = testing_utils.MockModel.create(responses=[mock_response])

  # Create agent with the mock toolset
  agent = Agent(name='test_agent', model=mock_model, tools=[mock_toolset])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()

  # Call _preprocess_async
  llm_request = LlmRequest()
  events = []
  async for event in flow._preprocess_async(invocation_context, llm_request):
    events.append(event)

  # Verify that process_llm_request was called on the toolset
  assert mock_toolset.process_llm_request_called


@pytest.mark.asyncio
async def test_preprocess_handles_mixed_tools_and_toolsets():
  """Test that _preprocess_async properly handles both tools and toolsets."""
  from google.adk.tools.base_tool import BaseTool
  from google.adk.tools.function_tool import FunctionTool

  # Create a mock tool
  class _MockTool(BaseTool):

    def __init__(self):
      super().__init__(name='mock_tool', description='Mock tool')
      self.process_llm_request_called = False
      self.process_llm_request = AsyncMock(side_effect=self._track_call)

    async def _track_call(self, **kwargs):
      self.process_llm_request_called = True

    async def call(self, **kwargs):
      return 'mock result'

  # Create a mock toolset
  class _MockToolset(BaseToolset):

    def __init__(self):
      super().__init__()
      self.process_llm_request_called = False
      self.process_llm_request = AsyncMock(side_effect=self._track_call)

    async def _track_call(self, **kwargs):
      self.process_llm_request_called = True

    async def get_tools(self, readonly_context=None):
      return []

    async def close(self):
      pass

  def _test_function():
    """Test function tool."""
    return 'function result'

  mock_tool = _MockTool()
  mock_toolset = _MockToolset()

  # Create agent with mixed tools and toolsets
  agent = Agent(
      name='test_agent', tools=[mock_tool, _test_function, mock_toolset]
  )

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  flow = BaseLlmFlowForTesting()

  # Call _preprocess_async
  llm_request = LlmRequest()
  events = []
  async for event in flow._preprocess_async(invocation_context, llm_request):
    events.append(event)

  # Verify that process_llm_request was called on both tools and toolsets
  assert mock_tool.process_llm_request_called
  assert mock_toolset.process_llm_request_called

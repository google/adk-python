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

"""Unit tests for BaseToolset."""

from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext
import pytest


class _TestingToolset(BaseToolset):
  """A test implementation of BaseToolset."""

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    return []

  async def close(self) -> None:
    pass


@pytest.mark.asyncio
async def test_process_llm_request_default_implementation():
  """Test that the default process_llm_request implementation does nothing."""
  toolset = _TestingToolset()

  # Create test objects
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context)
  llm_request = LlmRequest()

  # The default implementation should not modify the request
  original_request = LlmRequest.model_validate(llm_request.model_dump())

  await toolset.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Verify the request was not modified
  assert llm_request.model_dump() == original_request.model_dump()


@pytest.mark.asyncio
async def test_process_llm_request_can_be_overridden():
  """Test that process_llm_request can be overridden by subclasses."""

  class _CustomToolset(_TestingToolset):

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
      # Add some custom processing
      if not llm_request.contents:
        llm_request.contents = []
      llm_request.contents.append('Custom processing applied')

  toolset = _CustomToolset()

  # Create test objects
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  tool_context = ToolContext(invocation_context)
  llm_request = LlmRequest()

  await toolset.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # Verify the custom processing was applied
  assert llm_request.contents == ['Custom processing applied']

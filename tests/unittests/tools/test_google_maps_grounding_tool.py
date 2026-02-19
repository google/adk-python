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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


async def _create_tool_context() -> ToolContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return ToolContext(invocation_context=invocation_context)


class TestGoogleMapsGroundingTool:
  """Tests for GoogleMapsGroundingTool."""

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_2_model(self):
    tool = GoogleMapsGroundingTool()
    tool_context = await _create_tool_context()
    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_maps is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_non_gemini_model_raises_error(self):
    tool = GoogleMapsGroundingTool()
    tool_context = await _create_tool_context()
    llm_request = LlmRequest(
        model='claude-3-sonnet', config=types.GenerateContentConfig()
    )

    with pytest.raises(
        ValueError,
        match='Google maps tool is not supported for model claude-3-sonnet',
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_non_gemini_and_disabled_check(
      self, monkeypatch
  ):
    monkeypatch.setenv('ADK_DISABLE_GEMINI_MODEL_ID_CHECK', 'true')
    tool = GoogleMapsGroundingTool()
    tool_context = await _create_tool_context()
    llm_request = LlmRequest(
        model='internal-model-v1', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_maps is not None

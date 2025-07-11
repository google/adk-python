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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.enterprise_search_tool import EnterpriseWebSearchTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


async def _create_tool_context() -> ToolContext:
  """Creates a ToolContext for testing."""
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
  return ToolContext(invocation_context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-2.5-flash',
        'projects/test-project/locations/global/publishers/google/models/gemini-2.5-flash',
    ],
)
async def test_process_llm_request_success_with_gemini_models(model_name):
  tool = EnterpriseWebSearchTool()
  llm_request = LlmRequest(
      model=model_name, config=types.GenerateContentConfig()
  )
  tool_context = await _create_tool_context()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert (
      llm_request.config.tools[0].enterprise_web_search
      == types.EnterpriseWebSearch()
  )


@pytest.mark.asyncio
async def test_process_llm_request_failure_with_non_gemini_models():
  tool = EnterpriseWebSearchTool()
  llm_request = LlmRequest(model='gpt-4o', config=types.GenerateContentConfig())
  tool_context = await _create_tool_context()

  with pytest.raises(ValueError) as exc_info:
    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
  assert 'is not supported for model' in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_llm_request_failure_with_multiple_tools_gemini_1_models():
  tool = EnterpriseWebSearchTool()
  llm_request = LlmRequest(
      model='gemini-1.5-flash',
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(google_search=types.GoogleSearch()),
          ]
      ),
  )
  tool_context = await _create_tool_context()

  with pytest.raises(ValueError) as exc_info:
    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
  assert 'can not be used with other tools in Gemini 1.x.' in str(
      exc_info.value
  )

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

from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


class _TestingTool(BaseTool):

  def __init__(
      self,
      declaration: Optional[types.FunctionDeclaration] = None,
  ):
    super().__init__(name='test_tool', description='test_description')
    self.declaration = declaration

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return self.declaration


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
  return ToolContext(invocation_context)


@pytest.mark.asyncio
async def test_process_llm_request_no_declaration():
  tool = _TestingTool()
  tool_context = await _create_tool_context()
  llm_request = LlmRequest()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert llm_request.config == types.GenerateContentConfig()


@pytest.mark.asyncio
async def test_process_llm_request_with_declaration():
  declaration = types.FunctionDeclaration(
      name='test_tool',
      description='test_description',
      parameters=types.Schema(
          type=types.Type.STRING,
          title='param_1',
      ),
  )
  tool = _TestingTool(declaration)
  llm_request = LlmRequest()
  tool_context = await _create_tool_context()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  assert llm_request.config.tools[0].function_declarations == [declaration]


@pytest.mark.asyncio
async def test_process_llm_request_with_builtin_tool():
  declaration = types.FunctionDeclaration(
      name='test_tool',
      description='test_description',
      parameters=types.Schema(
          type=types.Type.STRING,
          title='param_1',
      ),
  )
  tool = _TestingTool(declaration)
  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          tools=[types.Tool(google_search=types.GoogleSearch())]
      )
  )
  tool_context = await _create_tool_context()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # function_declaration is added to another types.Tool without builtin tool.
  assert llm_request.config.tools[1].function_declarations == [declaration]


@pytest.mark.asyncio
async def test_process_llm_request_with_builtin_tool_and_another_declaration():
  declaration = types.FunctionDeclaration(
      name='test_tool',
      description='test_description',
      parameters=types.Schema(
          type=types.Type.STRING,
          title='param_1',
      ),
  )
  tool = _TestingTool(declaration)
  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(google_search=types.GoogleSearch()),
              types.Tool(function_declarations=[types.FunctionDeclaration()]),
          ]
      )
  )
  tool_context = await _create_tool_context()

  await tool.process_llm_request(
      tool_context=tool_context, llm_request=llm_request
  )

  # function_declaration is added to existing types.Tool with function_declaration.
  assert llm_request.config.tools[1].function_declarations[1] == declaration


def test_defers_response_flag():
  """Tests that _defers_response defaults to False and can be set by subclasses."""

  class SimpleTool(BaseTool):

    async def run_async(self, **kwargs):
      pass

  t = SimpleTool(name='test', description='desc')
  assert t._defers_response is False

  t2 = SimpleTool(name='test2', description='desc')
  t2._defers_response = True
  assert t2._defers_response is True


def _sample_handler(value: str) -> str:
  return value


def test_from_config_resolves_parametrized_callable():
  """A ``Callable[..., ...]`` arg is resolved, not skipped as unsupported."""
  from google.adk.tools.tool_configs import ToolArgsConfig

  class CallbackTool(BaseTool):

    def __init__(self, handler: Callable[[str], str]):
      super().__init__(name='callback_tool', description='desc')
      self.handler = handler

    async def run_async(self, **kwargs):
      pass

  config = ToolArgsConfig(handler=f'{__name__}._sample_handler')
  tool = CallbackTool.from_config(config, '')

  assert tool.handler is _sample_handler


def test_from_config_skips_list_of_non_class():
  """Non-class ``list`` element types are skipped, not an issubclass error."""
  from google.adk.tools.tool_configs import ToolArgsConfig

  class ListTool(BaseTool):

    def __init__(
        self,
        unions: Optional[list[Union[int, str]]] = None,
        anys: Optional[list[Any]] = None,
    ):
      super().__init__(name='list_tool', description='desc')
      self.unions = unions
      self.anys = anys

    async def run_async(self, **kwargs):
      pass

  config = ToolArgsConfig(unions=[1, 'two'], anys=[1, 'two'])
  tool = ListTool.from_config(config, '')

  assert tool.unions is None
  assert tool.anys is None


def test_response_scheduling_defaults_to_none():
  """response_scheduling defaults to None, preserving existing behavior."""

  class SimpleTool(BaseTool):

    async def run_async(self, **kwargs):
      pass

  t = SimpleTool(name='test', description='desc')
  assert t.response_scheduling is None

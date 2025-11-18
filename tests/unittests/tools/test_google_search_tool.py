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

"""Tests for GoogleSearchTool."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.google_search_tool import GoogleSearchTool
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


class TestGoogleSearchTool:
  """Test the GoogleSearchTool class."""

  def test_init(self):
    """Test initialization of GoogleSearchTool."""
    tool = GoogleSearchTool()
    assert tool.name == 'google_search'
    assert tool.description == 'google_search'

  def test_google_search_singleton(self):
    """Test that google_search is a singleton instance."""
    assert isinstance(google_search, GoogleSearchTool)
    assert google_search.name == 'google_search'

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_1_model(self):
    """Test processing LLM request with Gemini 1.x model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-1.5-flash', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search_retrieval is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_gemini_1_model(self):
    """Test processing LLM request with path-based Gemini 1.x model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='projects/265104255505/locations/us-central1/publishers/google/models/gemini-1.5-flash-001',
        config=types.GenerateContentConfig(),
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search_retrieval is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_1_0_model(self):
    """Test processing LLM request with Gemini 1.0 model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-1.0-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search_retrieval is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_2_model(self):
    """Test processing LLM request with Gemini 2.x model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-2.0-flash', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_gemini_2_model(self):
    """Test processing LLM request with path-based Gemini 2.x model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001',
        config=types.GenerateContentConfig(),
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_2_5_model(self):
    """Test processing LLM request with Gemini 2.5 model."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_1_model_and_existing_tools_raises_error(
      self,
  ):
    """Test that Gemini 1.x model with existing tools raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    existing_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name='test_function', description='test')
        ]
    )

    llm_request = LlmRequest(
        model='gemini-1.5-flash',
        config=types.GenerateContentConfig(tools=[existing_tool]),
    )

    with pytest.raises(
        ValueError,
        match=(
            'Google search tool cannot be used with other tools in Gemini 1.x'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_gemini_1_model_and_existing_tools_raises_error(
      self,
  ):
    """Test that path-based Gemini 1.x model with existing tools raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    existing_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name='test_function', description='test')
        ]
    )

    llm_request = LlmRequest(
        model='projects/265104255505/locations/us-central1/publishers/google/models/gemini-1.5-pro-preview',
        config=types.GenerateContentConfig(tools=[existing_tool]),
    )

    with pytest.raises(
        ValueError,
        match=(
            'Google search tool cannot be used with other tools in Gemini 1.x'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_2_model_and_existing_tools_succeeds(
      self,
  ):
    """Test that Gemini 2.x model with existing tools succeeds."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    existing_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name='test_function', description='test')
        ]
    )

    llm_request = LlmRequest(
        model='gemini-2.0-flash',
        config=types.GenerateContentConfig(tools=[existing_tool]),
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 2
    assert llm_request.config.tools[0] == existing_tool
    assert llm_request.config.tools[1].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_non_gemini_model_raises_error(self):
    """Test that non-Gemini model raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='claude-3-sonnet', config=types.GenerateContentConfig()
    )

    with pytest.raises(
        ValueError,
        match='Google search tool is not supported for model claude-3-sonnet',
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_non_gemini_model_raises_error(
      self,
  ):
    """Test that path-based non-Gemini model raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    non_gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/claude-3-sonnet'
    llm_request = LlmRequest(
        model=non_gemini_path, config=types.GenerateContentConfig()
    )

    with pytest.raises(
        ValueError,
        match=(
            f'Google search tool is not supported for model {non_gemini_path}'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_none_model_raises_error(self):
    """Test that None model raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(model=None, config=types.GenerateContentConfig())

    with pytest.raises(
        ValueError, match='Google search tool is not supported for model None'
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_empty_model_raises_error(self):
    """Test that empty model raises ValueError."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(model='', config=types.GenerateContentConfig())

    with pytest.raises(
        ValueError, match='Google search tool is not supported for model '
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_no_config(self):
    """Test processing LLM request with None config."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(model='gemini-2.0-flash')

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config is not None
    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_with_none_tools(self):
    """Test processing LLM request with None tools."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-2.0-flash', config=types.GenerateContentConfig(tools=None)
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    assert llm_request.config.tools[0].google_search is not None

  @pytest.mark.asyncio
  async def test_process_llm_request_edge_cases(self):
    """Test edge cases for model name validation."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    # Test with model names that contain gemini but don't start with it
    edge_cases = [
        'my-gemini-1.5-model',
        'custom-gemini-2.0-flash',
        'projects/265104255505/locations/us-central1/publishers/gemini/models/claude-3-sonnet',
    ]

    for model in edge_cases:
      llm_request = LlmRequest(
          model=model, config=types.GenerateContentConfig()
      )

      with pytest.raises(
          ValueError,
          match=f'Google search tool is not supported for model {model}',
      ):
        await tool.process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

  @pytest.mark.asyncio
  async def test_process_llm_request_gemini_version_specifics(self):
    """Test specific Gemini version behaviors."""
    tool = GoogleSearchTool()
    tool_context = await _create_tool_context()

    # Test various Gemini versions
    gemini_1_models = [
        'gemini-1.0-pro',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-1.9-experimental',
    ]

    gemini_2_models = [
        'gemini-2.0-flash',
        'gemini-2.0-pro',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
    ]

    # Test Gemini 1.x models use google_search_retrieval
    for model in gemini_1_models:
      llm_request = LlmRequest(
          model=model, config=types.GenerateContentConfig()
      )

      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

      assert llm_request.config.tools is not None
      assert len(llm_request.config.tools) == 1
      assert llm_request.config.tools[0].google_search_retrieval is not None
      assert llm_request.config.tools[0].google_search is None

    # Test Gemini 2.x models use google_search
    for model in gemini_2_models:
      llm_request = LlmRequest(
          model=model, config=types.GenerateContentConfig()
      )

      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

      assert llm_request.config.tools is not None
      assert len(llm_request.config.tools) == 1
      assert llm_request.config.tools[0].google_search is not None
      assert llm_request.config.tools[0].google_search_retrieval is None

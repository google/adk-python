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

import logging

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.adk.utils.model_name_utils import extract_model_name
from google.adk.utils.model_name_utils import is_gemini_1_model
from google.adk.utils.model_name_utils import is_gemini_model
from google.genai import types
import pytest

VERTEX_SEARCH_TOOL_LOGGER_NAME = (
    'google_adk.google.adk.tools.vertex_ai_search_tool'
)


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


class TestVertexAiSearchToolHelperFunctions:
  """Test the helper functions for model name extraction and validation."""

  def test_extract_model_name_simple_model(self):
    """Test extraction of simple model names."""
    assert extract_model_name('gemini-2.5-pro') == 'gemini-2.5-pro'
    assert extract_model_name('gemini-1.5-flash') == 'gemini-1.5-flash'
    assert extract_model_name('gemini-1.0-pro') == 'gemini-1.0-pro'
    assert extract_model_name('claude-3-sonnet') == 'claude-3-sonnet'

  def test_extract_model_name_path_based_model(self):
    """Test extraction of path-based model names."""
    path_model = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert extract_model_name(path_model) == 'gemini-2.0-flash-001'

    path_model_2 = 'projects/12345/locations/us-east1/publishers/google/models/gemini-1.5-pro-preview'
    assert extract_model_name(path_model_2) == 'gemini-1.5-pro-preview'

  def test_extract_model_name_invalid_path(self):
    """Test that invalid path formats return the original string."""
    invalid_path = 'projects/invalid/path/format'
    assert extract_model_name(invalid_path) == invalid_path

  def test_is_gemini_model_simple_names(self):
    """Test Gemini model detection with simple model names."""
    assert is_gemini_model('gemini-2.5-pro') is True
    assert is_gemini_model('gemini-1.5-flash') is True
    assert is_gemini_model('gemini-1.0-pro') is True
    assert is_gemini_model('claude-3-sonnet') is False
    assert is_gemini_model('gpt-4') is False
    assert is_gemini_model('gemini') is False  # Must have dash after gemini

  def test_is_gemini_model_path_based_names(self):
    """Test Gemini model detection with path-based model names."""
    gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert is_gemini_model(gemini_path) is True

    non_gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/claude-3-sonnet'
    assert is_gemini_model(non_gemini_path) is False

  def test_is_gemini_1_model_simple_names(self):
    """Test Gemini 1.x model detection with simple model names."""
    assert is_gemini_1_model('gemini-1.5-flash') is True
    assert is_gemini_1_model('gemini-1.0-pro') is True
    assert is_gemini_1_model('gemini-1.5-pro-preview') is True
    assert is_gemini_1_model('gemini-2.0-flash') is False
    assert is_gemini_1_model('gemini-2.5-pro') is False
    assert is_gemini_1_model('gemini-10.0-pro') is False  # Only 1.x versions
    assert is_gemini_1_model('claude-3-sonnet') is False

  def test_is_gemini_1_model_path_based_names(self):
    """Test Gemini 1.x model detection with path-based model names."""
    gemini_1_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-1.5-flash-001'
    assert is_gemini_1_model(gemini_1_path) is True

    gemini_2_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert is_gemini_1_model(gemini_2_path) is False

  def test_edge_cases(self):
    """Test edge cases for model name validation."""
    # Test with empty string
    assert is_gemini_model('') is False
    assert is_gemini_1_model('') is False

    # Test with model names containing gemini but not starting with it
    assert is_gemini_model('my-gemini-model') is False
    assert is_gemini_1_model('my-gemini-1.5-model') is False

    # Test with model names that have gemini in the middle of the path
    tricky_path = 'projects/265104255505/locations/us-central1/publishers/gemini/models/claude-3-sonnet'
    assert is_gemini_model(tricky_path) is False


class TestVertexAiSearchTool:
  """Test the VertexAiSearchTool class."""

  def test_init_with_data_store_id(self):
    """Test initialization with data store ID."""
    tool = VertexAiSearchTool(data_store_id='test_data_store')
    assert tool.data_store_id == 'test_data_store'
    assert tool.search_engine_id is None
    assert tool.data_store_specs is None

  def test_init_with_search_engine_id(self):
    """Test initialization with search engine ID."""
    tool = VertexAiSearchTool(search_engine_id='test_search_engine')
    assert tool.search_engine_id == 'test_search_engine'
    assert tool.data_store_id is None
    assert tool.data_store_specs is None

  def test_init_with_engine_and_specs(self):
    """Test initialization with search engine ID and specs."""
    specs = [
        types.VertexAISearchDataStoreSpec(
            dataStore=(
                'projects/p/locations/l/collections/c/dataStores/spec_store'
            )
        )
    ]
    engine_id = (
        'projects/p/locations/l/collections/c/engines/test_search_engine'
    )
    tool = VertexAiSearchTool(
        search_engine_id=engine_id,
        data_store_specs=specs,
    )
    assert tool.search_engine_id == engine_id
    assert tool.data_store_id is None
    assert tool.data_store_specs == specs

  def test_init_with_neither_raises_error(self):
    """Test that initialization without either ID raises ValueError."""
    with pytest.raises(
        ValueError,
        match='Either data_store_id or search_engine_id must be specified',
    ):
      VertexAiSearchTool()

  def test_init_with_both_raises_error(self):
    """Test that initialization with both IDs raises ValueError."""
    with pytest.raises(
        ValueError,
        match='Either data_store_id or search_engine_id must be specified',
    ):
      VertexAiSearchTool(
          data_store_id='test_data_store', search_engine_id='test_search_engine'
      )

  def test_init_with_specs_but_no_engine_raises_error(self):
    """Test that specs without engine ID raises ValueError."""
    specs = [
        types.VertexAISearchDataStoreSpec(
            dataStore=(
                'projects/p/locations/l/collections/c/dataStores/spec_store'
            )
        )
    ]
    with pytest.raises(
        ValueError,
        match=(
            'search_engine_id must be specified if data_store_specs is'
            ' specified'
        ),
    ):
      VertexAiSearchTool(
          data_store_id='test_data_store', data_store_specs=specs
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_simple_gemini_model(self, caplog):
    """Test processing LLM request with simple Gemini model name."""
    caplog.set_level(logging.DEBUG, logger=VERTEX_SEARCH_TOOL_LOGGER_NAME)

    tool = VertexAiSearchTool(
        data_store_id='test_data_store', filter='f', max_results=5
    )
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    retrieval_tool = llm_request.config.tools[0]
    assert retrieval_tool.retrieval is not None
    assert retrieval_tool.retrieval.vertex_ai_search is not None
    assert (
        retrieval_tool.retrieval.vertex_ai_search.datastore == 'test_data_store'
    )
    assert retrieval_tool.retrieval.vertex_ai_search.engine is None
    assert retrieval_tool.retrieval.vertex_ai_search.filter == 'f'
    assert retrieval_tool.retrieval.vertex_ai_search.max_results == 5

    # Verify debug log
    debug_records = [
        r
        for r in caplog.records
        if 'Adding Vertex AI Search tool config' in r.message
    ]
    assert len(debug_records) == 1
    log_message = debug_records[0].getMessage()
    assert 'datastore=test_data_store' in log_message
    assert 'engine=None' in log_message
    assert 'filter=f' in log_message
    assert 'max_results=5' in log_message
    assert 'data_store_specs=None' in log_message

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_gemini_model(self, caplog):
    """Test processing LLM request with path-based Gemini model name."""
    caplog.set_level(logging.DEBUG, logger=VERTEX_SEARCH_TOOL_LOGGER_NAME)

    specs = [
        types.VertexAISearchDataStoreSpec(
            dataStore=(
                'projects/p/locations/l/collections/c/dataStores/spec_store'
            )
        )
    ]
    engine_id = 'projects/p/locations/l/collections/c/engines/test_engine'
    tool = VertexAiSearchTool(
        search_engine_id=engine_id,
        data_store_specs=specs,
        filter='f2',
        max_results=10,
    )
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model=(
            'projects/265104255505/locations/us-central1/publishers/'
            'google/models/gemini-2.0-flash-001'
        ),
        config=types.GenerateContentConfig(),
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    retrieval_tool = llm_request.config.tools[0]
    assert retrieval_tool.retrieval is not None
    assert retrieval_tool.retrieval.vertex_ai_search is not None
    assert retrieval_tool.retrieval.vertex_ai_search.datastore is None
    assert retrieval_tool.retrieval.vertex_ai_search.engine == engine_id
    assert retrieval_tool.retrieval.vertex_ai_search.filter == 'f2'
    assert retrieval_tool.retrieval.vertex_ai_search.max_results == 10
    assert retrieval_tool.retrieval.vertex_ai_search.data_store_specs == specs

    # Verify debug log
    debug_records = [
        r
        for r in caplog.records
        if 'Adding Vertex AI Search tool config' in r.message
    ]
    assert len(debug_records) == 1
    log_message = debug_records[0].getMessage()
    assert 'datastore=None' in log_message
    assert f'engine={engine_id}' in log_message
    assert 'filter=f2' in log_message
    assert 'max_results=10' in log_message
    assert 'data_store_specs=1 spec(s): [spec_store]' in log_message

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_1_and_other_tools_raises_error(
      self,
  ):
    """Test that Gemini 1.x with other tools raises ValueError."""
    tool = VertexAiSearchTool(data_store_id='test_data_store')
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
            'Vertex AI search tool cannot be used with other tools in'
            ' Gemini 1.x'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_gemini_1_and_other_tools_raises_error(
      self,
  ):
    """Test that path-based Gemini 1.x with other tools raises ValueError."""
    tool = VertexAiSearchTool(data_store_id='test_data_store')
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
            'Vertex AI search tool cannot be used with other tools in'
            ' Gemini 1.x'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_non_gemini_model_raises_error(self):
    """Test that non-Gemini model raises ValueError."""
    tool = VertexAiSearchTool(data_store_id='test_data_store')
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='claude-3-sonnet', config=types.GenerateContentConfig()
    )

    with pytest.raises(
        ValueError,
        match=(
            'Vertex AI search tool is not supported for model claude-3-sonnet'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_path_based_non_gemini_model_raises_error(
      self,
  ):
    """Test that path-based non-Gemini model raises ValueError."""
    tool = VertexAiSearchTool(data_store_id='test_data_store')
    tool_context = await _create_tool_context()

    non_gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/claude-3-sonnet'
    llm_request = LlmRequest(
        model=non_gemini_path, config=types.GenerateContentConfig()
    )

    with pytest.raises(
        ValueError,
        match=(
            'Vertex AI search tool is not supported for model'
            f' {non_gemini_path}'
        ),
    ):
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @pytest.mark.asyncio
  async def test_process_llm_request_with_gemini_2_and_other_tools_succeeds(
      self, caplog
  ):
    """Test that Gemini 2.x with other tools succeeds."""
    caplog.set_level(logging.DEBUG, logger=VERTEX_SEARCH_TOOL_LOGGER_NAME)

    tool = VertexAiSearchTool(data_store_id='test_data_store')
    tool_context = await _create_tool_context()

    existing_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(name='test_function', description='test')
        ]
    )

    llm_request = LlmRequest(
        model='gemini-2.5-pro',
        config=types.GenerateContentConfig(tools=[existing_tool]),
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    # Should have both the existing tool and the new vertex AI search tool
    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 2
    assert llm_request.config.tools[0] == existing_tool
    retrieval_tool = llm_request.config.tools[1]
    assert retrieval_tool.retrieval is not None
    assert retrieval_tool.retrieval.vertex_ai_search is not None
    assert (
        retrieval_tool.retrieval.vertex_ai_search.datastore == 'test_data_store'
    )

    # Verify debug log
    debug_records = [
        r
        for r in caplog.records
        if 'Adding Vertex AI Search tool config' in r.message
    ]
    assert len(debug_records) == 1
    log_message = debug_records[0].getMessage()
    assert 'datastore=test_data_store' in log_message
    assert 'engine=None' in log_message
    assert 'filter=None' in log_message
    assert 'max_results=None' in log_message
    assert 'data_store_specs=None' in log_message

  @pytest.mark.asyncio
  async def test_subclass_with_dynamic_filter(self):
    """Test subclassing to provide dynamic filter based on context."""

    class DynamicFilterSearchTool(VertexAiSearchTool):
      """Custom search tool with dynamic filter."""

      def _build_vertex_ai_search_config(self, ctx):
        user_id = ctx.state.get('user_id', 'default_user')
        return types.VertexAISearch(
            datastore=self.data_store_id,
            engine=self.search_engine_id,
            filter=f"user_id = '{user_id}'",
            max_results=self.max_results,
        )

    tool = DynamicFilterSearchTool(data_store_id='test_data_store')
    tool_context = await _create_tool_context()
    tool_context.state['user_id'] = 'test_user_123'

    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    assert llm_request.config.tools is not None
    assert len(llm_request.config.tools) == 1
    retrieval_tool = llm_request.config.tools[0]
    assert retrieval_tool.retrieval is not None
    assert retrieval_tool.retrieval.vertex_ai_search is not None
    # Verify the filter was dynamically set
    assert (
        retrieval_tool.retrieval.vertex_ai_search.filter
        == "user_id = 'test_user_123'"
    )

  @pytest.mark.asyncio
  async def test_subclass_with_dynamic_max_results(self):
    """Test subclassing to provide dynamic max_results based on context."""

    class DynamicMaxResultsSearchTool(VertexAiSearchTool):
      """Custom search tool with dynamic max_results."""

      def _build_vertex_ai_search_config(self, ctx):
        # Use a larger max_results for premium users
        is_premium = ctx.state.get('is_premium', False)
        dynamic_max_results = 20 if is_premium else 5
        return types.VertexAISearch(
            datastore=self.data_store_id,
            engine=self.search_engine_id,
            filter=self.filter,
            max_results=dynamic_max_results,
        )

    tool = DynamicMaxResultsSearchTool(
        data_store_id='test_data_store', max_results=10
    )
    tool_context = await _create_tool_context()
    tool_context.state['is_premium'] = True

    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    retrieval_tool = llm_request.config.tools[0]
    # Verify max_results was dynamically set to premium value
    assert retrieval_tool.retrieval.vertex_ai_search.max_results == 20

  @pytest.mark.asyncio
  async def test_subclass_receives_readonly_context(self):
    """Test that subclass receives the context correctly."""
    received_contexts = []

    class ContextCapturingSearchTool(VertexAiSearchTool):
      """Custom search tool that captures the context."""

      def _build_vertex_ai_search_config(self, ctx):
        received_contexts.append(ctx)
        return types.VertexAISearch(
            datastore=self.data_store_id,
            engine=self.search_engine_id,
            filter=self.filter,
            max_results=self.max_results,
        )

    tool = ContextCapturingSearchTool(data_store_id='test_data_store')
    tool_context = await _create_tool_context()

    llm_request = LlmRequest(
        model='gemini-2.5-pro', config=types.GenerateContentConfig()
    )

    await tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    # Verify the context was passed to _build_vertex_ai_search_config
    assert len(received_contexts) == 1
    assert received_contexts[0] is tool_context

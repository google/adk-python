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

"""Unit tests for BaseLlmFlow toolset integration."""

from unittest import mock
from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _handle_after_model_callback
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types
import pytest

from ... import testing_utils

google_search = GoogleSearchTool(bypass_multi_tools_limit=True)


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


# TODO(b/448114567): Remove the following test_preprocess_with_google_search
# tests once the workaround is no longer needed.
@pytest.mark.asyncio
async def test_preprocess_with_google_search_only():
  """Test _preprocess_async with only the google_search tool."""
  agent = Agent(name='test_agent', model='gemini-pro', tools=[google_search])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  flow = BaseLlmFlowForTesting()
  llm_request = LlmRequest(model='gemini-pro')
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  assert len(llm_request.config.tools) == 1
  assert llm_request.config.tools[0].google_search is not None


@pytest.mark.asyncio
async def test_preprocess_with_google_search_workaround():
  """Test _preprocess_async with google_search and another tool."""

  def _my_tool(sides: int) -> int:
    """A simple tool."""
    return sides

  agent = Agent(
      name='test_agent', model='gemini-pro', tools=[_my_tool, google_search]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  flow = BaseLlmFlowForTesting()
  llm_request = LlmRequest(model='gemini-pro')
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  assert len(llm_request.config.tools) == 1
  declarations = llm_request.config.tools[0].function_declarations
  assert len(declarations) == 2
  assert {d.name for d in declarations} == {'_my_tool', 'google_search_agent'}


@pytest.mark.asyncio
async def test_preprocess_calls_convert_tool_union_to_tools():
  """Test that _preprocess_async calls _convert_tool_union_to_tools."""

  class _MockTool:
    process_llm_request = AsyncMock()

  mock_tool_instance = _MockTool()

  def _my_tool(sides: int) -> int:
    """A simple tool."""
    return sides

  with mock.patch(
      'google.adk.agents.llm_agent._convert_tool_union_to_tools',
      new_callable=AsyncMock,
  ) as mock_convert:
    mock_convert.return_value = [mock_tool_instance]

    model = Gemini(model='gemini-2')
    agent = Agent(
        name='test_agent', model=model, tools=[_my_tool, google_search]
    )
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content='test message'
    )
    flow = BaseLlmFlowForTesting()
    llm_request = LlmRequest(model='gemini-2')

    async for _ in flow._preprocess_async(invocation_context, llm_request):
      pass

    mock_convert.assert_called_with(
        google_search,
        mock.ANY,  # ReadonlyContext(invocation_context)
        model,
        True,  # multiple_tools
    )


# TODO(b/448114567): Remove the following
# test_handle_after_model_callback_grounding tests once the workaround
# is no longer needed.
def dummy_tool():
  pass


@pytest.mark.parametrize(
    'tools, state_metadata, expect_metadata',
    [
        ([], None, False),
        ([google_search, dummy_tool], {'foo': 'bar'}, True),
        ([dummy_tool], {'foo': 'bar'}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        'no_search_no_grounding',
        'with_search_with_grounding',
        'no_search_with_grounding',
        'with_search_no_grounding',
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_no_callbacks(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there are no callbacks."""
  agent = Agent(name='test_agent', tools=tools)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  if state_metadata:
    invocation_context.session.state['temp:_adk_grounding_metadata'] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text='response')])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )
  flow = BaseLlmFlowForTesting()

  result = await _handle_after_model_callback(
      invocation_context, llm_response, event
  )

  if expect_metadata:
    llm_response.grounding_metadata = state_metadata
    assert result == llm_response
  else:
    assert result is None


@pytest.mark.parametrize(
    'tools, state_metadata, expect_metadata',
    [
        ([], None, False),
        ([google_search, dummy_tool], {'foo': 'bar'}, True),
        ([dummy_tool], {'foo': 'bar'}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        'no_search_no_grounding',
        'with_search_with_grounding',
        'no_search_with_grounding',
        'with_search_no_grounding',
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_callback_override(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there is a callback override."""
  agent_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text='agent')])
  )
  agent_callback = AsyncMock(return_value=agent_response)

  agent = Agent(
      name='test_agent', tools=tools, after_model_callback=[agent_callback]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  if state_metadata:
    invocation_context.session.state['temp:_adk_grounding_metadata'] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text='response')])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )
  flow = BaseLlmFlowForTesting()

  result = await _handle_after_model_callback(
      invocation_context, llm_response, event
  )

  if expect_metadata:
    agent_response.grounding_metadata = state_metadata

  assert result == agent_response
  agent_callback.assert_called_once()


@pytest.mark.parametrize(
    'tools, state_metadata, expect_metadata',
    [
        ([], None, False),
        ([google_search, dummy_tool], {'foo': 'bar'}, True),
        ([dummy_tool], {'foo': 'bar'}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        'no_search_no_grounding',
        'with_search_with_grounding',
        'no_search_with_grounding',
        'with_search_no_grounding',
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_plugin_override(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there is a plugin override."""
  plugin_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text='plugin')])
  )

  class _MockPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name='mock_plugin')

    after_model_callback = AsyncMock(return_value=plugin_response)

  plugin = _MockPlugin()
  agent = Agent(name='test_agent', tools=tools)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  if state_metadata:
    invocation_context.session.state['temp:_adk_grounding_metadata'] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text='response')])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )
  flow = BaseLlmFlowForTesting()

  result = await _handle_after_model_callback(
      invocation_context, llm_response, event
  )

  if expect_metadata:
    plugin_response.grounding_metadata = state_metadata

  assert result == plugin_response
  plugin.after_model_callback.assert_called_once()


@pytest.mark.asyncio
async def test_handle_after_model_callback_caches_canonical_tools():
  """Test that canonical_tools is only called once per invocation_context."""
  canonical_tools_call_count = 0

  async def mock_canonical_tools(self, readonly_context=None):
    nonlocal canonical_tools_call_count
    canonical_tools_call_count += 1
    from google.adk.tools.base_tool import BaseTool

    class MockGoogleSearchTool(BaseTool):

      def __init__(self):
        super().__init__(name='google_search_agent', description='Mock search')

      async def call(self, **kwargs):
        return 'mock result'

    return [MockGoogleSearchTool()]

  agent = Agent(name='test_agent', tools=[google_search, dummy_tool])

  with mock.patch.object(
      type(agent), 'canonical_tools', new=mock_canonical_tools
  ):
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent
    )

    assert invocation_context.canonical_tools_cache is None

    invocation_context.session.state['temp:_adk_grounding_metadata'] = {
        'foo': 'bar'
    }

    llm_response = LlmResponse(
        content=types.Content(parts=[types.Part.from_text(text='response')])
    )
    event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=agent.name,
    )
    flow = BaseLlmFlowForTesting()

    # Call _handle_after_model_callback multiple times with the same context
    result1 = await _handle_after_model_callback(
        invocation_context, llm_response, event
    )
    result2 = await _handle_after_model_callback(
        invocation_context, llm_response, event
    )
    result3 = await _handle_after_model_callback(
        invocation_context, llm_response, event
    )

    assert canonical_tools_call_count == 1, (
        'canonical_tools should be called once, but was called '
        f'{canonical_tools_call_count} times'
    )

    assert invocation_context.canonical_tools_cache is not None
    assert len(invocation_context.canonical_tools_cache) == 1
    assert (
        invocation_context.canonical_tools_cache[0].name
        == 'google_search_agent'
    )

    assert result1.grounding_metadata == {'foo': 'bar'}
    assert result2.grounding_metadata == {'foo': 'bar'}
    assert result3.grounding_metadata == {'foo': 'bar'}


# ---------------------------------------------------------------------------
# Tests for _finalize_model_response_event function-call ID consistency
# ---------------------------------------------------------------------------

from google.adk.flows.llm_flows.base_llm_flow import _finalize_model_response_event


def _make_fc_response(fc_name: str, fc_id: str | None = None, partial: bool = False) -> LlmResponse:
    """Helper: build an LlmResponse with a single function call."""
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=fc_name,
                        args={'x': 1},
                        id=fc_id,
                    )
                )
            ],
        ),
        partial=partial,
    )


def test_finalize_model_response_event_consistent_fc_id_across_partial_and_final():
    """Function call IDs must be identical in partial and final SSE events.

    Regression test for https://github.com/google/adk-python/issues/4609.
    When SSE streaming is active, _finalize_model_response_event is called
    once for the partial event and once for the final event, both sharing the
    same model_response_event object.  The assigned adk-* ID must be the same
    in both calls.
    """
    llm_request = LlmRequest()
    llm_request.tools_dict = {}
    base_event = Event(
        invocation_id='inv1',
        author='agent',
    )

    # First call: partial streaming event (function call has no ID from LLM)
    partial_response = _make_fc_response('my_tool', fc_id=None, partial=True)
    partial_finalized = _finalize_model_response_event(
        llm_request, partial_response, base_event
    )
    partial_fc_id = partial_finalized.get_function_calls()[0].id
    assert partial_fc_id is not None
    assert partial_fc_id.startswith('adk-')

    # Second call: final (non-partial) event for the same function call
    final_response = _make_fc_response('my_tool', fc_id=None, partial=False)
    final_finalized = _finalize_model_response_event(
        llm_request, final_response, base_event
    )
    final_fc_id = final_finalized.get_function_calls()[0].id

    assert final_fc_id == partial_fc_id, (
        f'Function call ID changed between partial ({partial_fc_id!r}) and '
        f'final ({final_fc_id!r}) SSE events — HITL workflows will break.'
    )


def test_finalize_model_response_event_preserves_llm_assigned_id():
    """If the LLM already assigned an ID, it must be preserved as-is."""
    llm_request = LlmRequest()
    llm_request.tools_dict = {}
    base_event = Event(invocation_id='inv1', author='agent')

    response = _make_fc_response('my_tool', fc_id='llm-assigned-id')
    finalized = _finalize_model_response_event(llm_request, response, base_event)
    assert finalized.get_function_calls()[0].id == 'llm-assigned-id'

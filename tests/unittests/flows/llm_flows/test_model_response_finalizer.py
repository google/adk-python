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

"""Unit tests for _model_response_finalizer module."""

from unittest import mock
from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import _model_response_finalizer
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types
import pytest

from ... import testing_utils

google_search = GoogleSearchTool(bypass_multi_tools_limit=True)


def dummy_tool():
  pass


# --- Tests for finalize_model_response_event ---


def test_finalize_model_response_event_merges_llm_response_fields():
  llm_request = LlmRequest()
  content = types.Content(parts=[types.Part.from_text(text="hello world")])
  usage = types.GenerateContentResponseUsageMetadata(total_token_count=42)
  llm_response = LlmResponse(content=content, usage_metadata=usage)
  event = Event(id="e-1", invocation_id="inv-1", author="model_agent")

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  assert finalized.content == content
  assert finalized.usage_metadata == usage
  assert finalized.id == "e-1"
  assert finalized.invocation_id == "inv-1"


def test_finalize_model_response_event_populates_function_call_ids():
  from google.adk.tools.function_tool import FunctionTool

  tool = FunctionTool(func=dummy_tool)
  fc = types.FunctionCall(name="dummy_tool", args={"x": 1})
  content = types.Content(parts=[types.Part(function_call=fc)])
  llm_response = LlmResponse(content=content)
  llm_request = LlmRequest(tools_dict={"dummy_tool": tool})
  event = Event(id="e-2", invocation_id="inv-1", author="model_agent")

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  function_calls = finalized.get_function_calls()
  assert len(function_calls) == 1
  assert function_calls[0].name == "dummy_tool"
  # Client function call ID was populated
  assert function_calls[0].id is not None


# --- Tests for handle_before_model_callback ---


@pytest.mark.asyncio
async def test_handle_before_model_callback_none_by_default():
  agent = Agent(name="test_agent", tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  result = await _model_response_finalizer.handle_before_model_callback(
      ctx, llm_request, event
  )
  assert result is None


@pytest.mark.asyncio
async def test_handle_before_model_callback_agent_override():
  expected_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="override")])
  )
  callback = AsyncMock(return_value=expected_response)
  agent = Agent(name="test_agent", tools=[], before_model_callback=[callback])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  result = await _model_response_finalizer.handle_before_model_callback(
      ctx, llm_request, event
  )
  assert result == expected_response
  callback.assert_called_once()


@pytest.mark.asyncio
async def test_handle_before_model_callback_plugin_override():
  plugin_response = LlmResponse(
      content=types.Content(
          parts=[types.Part.from_text(text="plugin_override")]
      )
  )

  class _Plugin(BasePlugin):

    def __init__(self):
      super().__init__(name="p1")

    before_model_callback = AsyncMock(return_value=plugin_response)

  plugin = _Plugin()
  agent = Agent(name="test_agent", tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  result = await _model_response_finalizer.handle_before_model_callback(
      ctx, llm_request, event
  )
  assert result == plugin_response
  plugin.before_model_callback.assert_called_once()


# --- Tests for handle_after_model_callback and Grounding Metadata ---


@pytest.mark.parametrize(
    "tools, state_metadata, expect_metadata",
    [
        ([], None, False),
        ([google_search, dummy_tool], {"foo": "bar"}, True),
        ([dummy_tool], {"foo": "bar"}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        "no_search_no_grounding",
        "with_search_with_grounding",
        "no_search_with_grounding",
        "with_search_no_grounding",
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_no_callbacks(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there are no callbacks."""
  agent = Agent(name="test_agent", tools=tools)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  if state_metadata:
    invocation_context.session.state["temp:_adk_grounding_metadata"] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="response")])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )

  result = await _model_response_finalizer.handle_after_model_callback(
      invocation_context, llm_response, event
  )

  if expect_metadata:
    llm_response.grounding_metadata = state_metadata
    assert result == llm_response
  else:
    assert result is None


@pytest.mark.parametrize(
    "tools, state_metadata, expect_metadata",
    [
        ([], None, False),
        ([google_search, dummy_tool], {"foo": "bar"}, True),
        ([dummy_tool], {"foo": "bar"}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        "no_search_no_grounding",
        "with_search_with_grounding",
        "no_search_with_grounding",
        "with_search_no_grounding",
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_callback_override(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there is a callback override."""
  agent_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="agent")])
  )
  agent_callback = AsyncMock(return_value=agent_response)

  agent = Agent(
      name="test_agent", tools=tools, after_model_callback=[agent_callback]
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  if state_metadata:
    invocation_context.session.state["temp:_adk_grounding_metadata"] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="response")])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )

  result = await _model_response_finalizer.handle_after_model_callback(
      invocation_context, llm_response, event
  )

  if expect_metadata:
    agent_response.grounding_metadata = state_metadata

  assert result == agent_response
  agent_callback.assert_called_once()


@pytest.mark.parametrize(
    "tools, state_metadata, expect_metadata",
    [
        ([], None, False),
        ([google_search, dummy_tool], {"foo": "bar"}, True),
        ([dummy_tool], {"foo": "bar"}, False),
        ([google_search, dummy_tool], None, False),
    ],
    ids=[
        "no_search_no_grounding",
        "with_search_with_grounding",
        "no_search_with_grounding",
        "with_search_no_grounding",
    ],
)
@pytest.mark.asyncio
async def test_handle_after_model_callback_grounding_with_plugin_override(
    tools, state_metadata, expect_metadata
):
  """Test handling grounding metadata when there is a plugin override."""
  plugin_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="plugin")])
  )

  class _MockPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="mock_plugin")

    after_model_callback = AsyncMock(return_value=plugin_response)

  plugin = _MockPlugin()
  agent = Agent(name="test_agent", tools=tools)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )
  if state_metadata:
    invocation_context.session.state["temp:_adk_grounding_metadata"] = (
        state_metadata
    )

  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="response")])
  )
  event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
  )

  result = await _model_response_finalizer.handle_after_model_callback(
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
        super().__init__(name="google_search_agent", description="Mock search")
        self.propagate_grounding_metadata = True

      async def call(self, **kwargs):
        return "mock result"

    return [MockGoogleSearchTool()]

  agent = Agent(name="test_agent", tools=[google_search, dummy_tool])

  with mock.patch.object(
      type(agent), "canonical_tools", new=mock_canonical_tools
  ):
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent
    )

    assert invocation_context.canonical_tools_cache is None

    invocation_context.session.state["temp:_adk_grounding_metadata"] = {
        "foo": "bar"
    }

    llm_response = LlmResponse(
        content=types.Content(parts=[types.Part.from_text(text="response")])
    )
    event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=agent.name,
    )

    result1 = await _model_response_finalizer.handle_after_model_callback(
        invocation_context, llm_response, event
    )
    result2 = await _model_response_finalizer.handle_after_model_callback(
        invocation_context, llm_response, event
    )
    result3 = await _model_response_finalizer.handle_after_model_callback(
        invocation_context, llm_response, event
    )

    assert canonical_tools_call_count == 1, (
        "canonical_tools should be called once, but was called "
        f"{canonical_tools_call_count} times"
    )

    assert invocation_context.canonical_tools_cache is not None
    assert len(invocation_context.canonical_tools_cache) == 1
    assert (
        invocation_context.canonical_tools_cache[0].name
        == "google_search_agent"
    )

    assert result1.grounding_metadata == {"foo": "bar"}
    assert result2.grounding_metadata == {"foo": "bar"}
    assert result3.grounding_metadata == {"foo": "bar"}


# --- Tests for run_and_handle_error ---


@pytest.mark.asyncio
async def test_run_and_handle_error_yields_normally():
  resp = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="ok")])
  )

  async def mock_generator():
    yield resp

  agent = Agent(name="test_agent", tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  results = []
  async for item in _model_response_finalizer.run_and_handle_error(
      mock_generator(), ctx, llm_request, event
  ):
    results.append(item)

  assert results == [resp]


@pytest.mark.asyncio
async def test_run_and_handle_error_recovers_via_callback():
  recovery_response = LlmResponse(
      content=types.Content(parts=[types.Part.from_text(text="recovered")])
  )
  on_error = AsyncMock(return_value=recovery_response)

  async def failing_generator():
    if False:
      yield
    raise ValueError("LLM generation failed")

  agent = Agent(name="test_agent", tools=[], on_model_error_callback=[on_error])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  results = []
  async for item in _model_response_finalizer.run_and_handle_error(
      failing_generator(), ctx, llm_request, event
  ):
    results.append(item)

  assert results == [recovery_response]
  on_error.assert_called_once()


@pytest.mark.asyncio
async def test_run_and_handle_error_reraises_when_unhandled():
  async def failing_generator():
    if False:
      yield
    raise RuntimeError("Unrecoverable error")

  agent = Agent(name="test_agent", tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  event = Event(invocation_id=ctx.invocation_id, author=agent.name)
  llm_request = LlmRequest()

  with pytest.raises(RuntimeError, match="Unrecoverable error"):
    async for _ in _model_response_finalizer.run_and_handle_error(
        failing_generator(), ctx, llm_request, event
    ):
      pass

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

import copy
from unittest import mock
from unittest.mock import AsyncMock

from google.adk.agents.llm_agent import Agent
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import LIVE_USAGE_KEY
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.events.event import Event
from google.adk.events.event import TOOL_DISPLAY_NAMES_METADATA_KEY
from google.adk.flows.llm_flows.core import _finalizer as _model_response_finalizer
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types
from pydantic import ConfigDict
import pytest

from .... import testing_utils

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
  assert finalized.custom_metadata is None


class _TitledTool(BaseTool):
  """Minimal BaseTool test double with an optional display title."""

  def __init__(self, *, name: str, title: str | None = None):
    super().__init__(name=name, description=f"Test tool {name}", title=title)


def test_finalize_model_response_event_populates_tool_display_names():
  """Called titled tools populate adk:tool_display_names metadata."""
  prefixed_tool = _TitledTool(
      name="kb_Internal-KB-Search",
      title="  @Internal-KB Search  ",
  )
  plain_tool = FunctionTool(func=dummy_tool)

  content = types.Content(
      parts=[
          types.Part.from_function_call(
              name="kb_Internal-KB-Search", args={"q": "policy"}
          ),
          types.Part.from_function_call(name="dummy_tool", args={}),
      ]
  )
  llm_response = LlmResponse(content=content)
  llm_request = LlmRequest(
      tools_dict={
          "kb_Internal-KB-Search": prefixed_tool,
          "dummy_tool": plain_tool,
      }
  )
  event = Event(id="e-3", invocation_id="inv-1", author="model_agent")

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  assert TOOL_DISPLAY_NAMES_METADATA_KEY == "adk:tool_display_names"
  assert finalized.custom_metadata == {
      TOOL_DISPLAY_NAMES_METADATA_KEY: {
          "kb_Internal-KB-Search": "@Internal-KB Search",
      }
  }


def test_finalize_merges_tool_display_names_without_mutating_inputs():
  """Real tool titles override response entries without mutating input dicts."""
  titled_tool = _TitledTool(
      name="Internal-KB-Search",
      title="@Internal-KB Search",
  )
  base_metadata = {"run_label": "prod"}
  response_metadata = {
      "trace_id": "abc",
      TOOL_DISPLAY_NAMES_METADATA_KEY: {
          "prepopulated_tool": "Prepopulated Title",
          "Internal-KB-Search": "Spoofed Response Title",
      },
  }
  llm_response = LlmResponse(
      content=types.Content(
          parts=[
              types.Part.from_function_call(
                  name="Internal-KB-Search", args={"q": "travel policy"}
              )
          ]
      ),
      custom_metadata=response_metadata,
  )
  llm_request = LlmRequest(tools_dict={"Internal-KB-Search": titled_tool})
  event = Event(
      id="e-4",
      invocation_id="inv-1",
      author="model_agent",
      custom_metadata=base_metadata,
  )

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  assert finalized.custom_metadata == {
      "trace_id": "abc",
      TOOL_DISPLAY_NAMES_METADATA_KEY: {
          "prepopulated_tool": "Prepopulated Title",
          "Internal-KB-Search": "@Internal-KB Search",
      },
  }
  assert base_metadata == {"run_label": "prod"}
  assert response_metadata == {
      "trace_id": "abc",
      TOOL_DISPLAY_NAMES_METADATA_KEY: {
          "prepopulated_tool": "Prepopulated Title",
          "Internal-KB-Search": "Spoofed Response Title",
      },
  }


def test_finalize_ignores_mocked_tools_without_str_title():
  """Mocked BaseTool instances do not inject mock titles."""
  mocked_tool = mock.MagicMock(spec=BaseTool)
  mocked_tool.is_long_running = False
  llm_response = LlmResponse(
      content=types.Content(
          parts=[
              types.Part.from_function_call(name="mocked_tool", args={}),
          ]
      )
  )
  llm_request = LlmRequest(tools_dict={"mocked_tool": mocked_tool})
  event = Event(id="e-5", invocation_id="inv-1", author="model_agent")

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  assert finalized.custom_metadata is None
  assert finalized.model_dump(mode="json")["custom_metadata"] is None


def test_finalize_overwrites_non_mapping_existing_tool_display_names():
  """Non-mapping existing adk:tool_display_names entries are replaced safely."""
  titled_tool = _TitledTool(
      name="Internal-KB-Search",
      title="@Internal-KB Search",
  )
  llm_response = LlmResponse(
      content=types.Content(
          parts=[
              types.Part.from_function_call(
                  name="Internal-KB-Search", args={"q": "policy"}
              )
          ]
      ),
      custom_metadata={
          "trace_id": "abc",
          TOOL_DISPLAY_NAMES_METADATA_KEY: "invalid-non-dict",
      },
  )
  llm_request = LlmRequest(tools_dict={"Internal-KB-Search": titled_tool})
  event = Event(id="e-6", invocation_id="inv-1", author="model_agent")

  finalized = _model_response_finalizer.finalize_model_response_event(
      llm_request=llm_request,
      llm_response=llm_response,
      model_response_event=event,
  )

  assert finalized.custom_metadata == {
      "trace_id": "abc",
      TOOL_DISPLAY_NAMES_METADATA_KEY: {
          "Internal-KB-Search": "@Internal-KB Search",
      },
  }


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


async def test_after_model_preserves_same_object_edits():
  """Metadata restoration must not undo deliberate edits on the same response."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  response = LlmResponse(
      custom_metadata={LIVE_USAGE_KEY: marker},
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=15
      ),
  )

  def callback(callback_context, llm_response):
    del callback_context
    llm_response.usage_metadata = None
    return llm_response

  agent = Agent(name="test_agent", after_model_callback=callback)
  context = await testing_utils.create_invocation_context(agent=agent)
  result = await _model_response_finalizer.handle_after_model_callback(
      context, response, Event(author=agent.name)
  )
  assert result is response
  assert response.usage_metadata is None
  assert response.custom_metadata == {LIVE_USAGE_KEY: marker}


@pytest.mark.parametrize("plugin_callback", [False, True])
async def test_after_model_does_not_contaminate_reused_replacement(
    plugin_callback,
):
  """A cached response must not carry Live identity into a later unary call."""
  replacement = LlmResponse(
      content=types.Content(parts=[types.Part(text="cached")]),
      custom_metadata={"user_tag": "cached"},
  )
  saved_replacement = replacement.model_dump()
  callback = AsyncMock(return_value=replacement)
  agent = Agent(
      name="test_agent",
      after_model_callback=None if plugin_callback else callback,
  )
  plugins = []
  if plugin_callback:
    plugin = BasePlugin(name="cached-response")
    plugin.after_model_callback = callback
    plugins.append(plugin)
  context = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins
  )
  marker = {"connection_id": "connection-1", "sequence": 1}
  live = LlmResponse(
      custom_metadata={LIVE_USAGE_KEY: marker},
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=15
      ),
  )
  unary = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=100
      ),
  )

  live_result = await _model_response_finalizer.handle_after_model_callback(
      context, live, Event(author=agent.name)
  )
  after_live = replacement.model_dump()
  unary_result = await _model_response_finalizer.handle_after_model_callback(
      context, unary, Event(author=agent.name)
  )

  assert after_live == saved_replacement
  assert replacement.model_dump() == saved_replacement
  assert live_result is not replacement
  assert live_result.custom_metadata == {
      "user_tag": "cached",
      LIVE_USAGE_KEY: marker,
  }
  assert live_result.usage_metadata.total_token_count == 15
  assert unary_result is not replacement
  assert unary_result.custom_metadata == {"user_tag": "cached"}
  assert unary_result.usage_metadata.total_token_count == 100


@pytest.mark.parametrize(
    "field_name",
    [
        "usage_metadata",
        "partial",
        "turn_complete",
        "model_version",
        "live_session_id",
    ],
)
async def test_after_model_replacement_respects_fields_cleared_by_plugin(
    field_name,
):
  """A later replacement inherits the current fields, not an old snapshot."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  original = LlmResponse(
      custom_metadata={LIVE_USAGE_KEY: marker},
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=15
      ),
      partial=True,
      turn_complete=False,
      model_version="live-model",
      live_session_id="provider-session",
  )

  async def clear_field(*, callback_context, llm_response):
    del callback_context
    setattr(llm_response, field_name, None)

  plugin = BasePlugin(name="field-editor")
  plugin.after_model_callback = clear_field
  replacement = LlmResponse(
      content=types.Content(parts=[types.Part(text="replacement")])
  )
  agent = Agent(
      name="test_agent",
      after_model_callback=AsyncMock(return_value=replacement),
  )
  context = await testing_utils.create_invocation_context(
      agent=agent, plugins=[plugin]
  )

  result = await _model_response_finalizer.handle_after_model_callback(
      context, original, Event(author=agent.name)
  )

  assert getattr(result, field_name) is None
  assert result.custom_metadata == {LIVE_USAGE_KEY: marker}
  assert replacement.custom_metadata is None


@pytest.mark.parametrize("edit", ["clear", "nested"])
async def test_after_model_canonical_callbacks_each_see_owned_metadata(edit):
  """Each agent callback sees original identity and earlier public edits."""
  marker = {"connection_id": "connection-1", "sequence": 1}
  original = LlmResponse(custom_metadata={LIVE_USAGE_KEY: dict(marker)})
  observed = []

  def edit_metadata(callback_context, llm_response):
    del callback_context
    if edit == "clear":
      llm_response.custom_metadata.clear()
    else:
      llm_response.custom_metadata[LIVE_USAGE_KEY]["sequence"] = 99
    llm_response.custom_metadata["user_tag"] = "voice"

  def observe_metadata(callback_context, llm_response):
    del callback_context
    observed.append(copy.deepcopy(llm_response.custom_metadata))

  agent = Agent(
      name="test_agent",
      after_model_callback=[edit_metadata, observe_metadata],
  )
  context = await testing_utils.create_invocation_context(agent=agent)

  result = await _model_response_finalizer.handle_after_model_callback(
      context, original, Event(author=agent.name)
  )

  expected = {LIVE_USAGE_KEY: marker, "user_tag": "voice"}
  assert observed == [expected]
  assert result is None
  assert original.custom_metadata == expected


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
async def test_after_model_keeps_refreshed_plugin_metadata(frozen, replacement):
  """Canonical callbacks keep plugin stamps while restoring model provenance."""

  class Response(LlmResponse):
    model_config = ConfigDict(frozen=frozen)

  marker = {"connection_id": "connection-1", "sequence": 1}
  plugin_key = INTERNAL_METADATA_PREFIX + "plugin_owned"
  original = Response(
      custom_metadata={LIVE_USAGE_KEY: dict(marker), plugin_key: "old"}
  )
  observed = []
  cached_metadata = {"user_tag": "cached", plugin_key: "new"}
  cached_response = LlmResponse(custom_metadata=cached_metadata)

  def edit(callback_context, llm_response):
    if frozen:
      llm_response.custom_metadata.clear()
      llm_response.custom_metadata.update(cached_metadata)
    else:
      llm_response.custom_metadata = cached_metadata

  def observe(callback_context, llm_response):
    observed.append(copy.deepcopy(llm_response.custom_metadata))
    return cached_response if replacement else None

  agent = Agent(name="agent", after_model_callback=[edit, observe])
  context = await testing_utils.create_invocation_context(agent=agent)
  result = await _model_response_finalizer.handle_after_model_callback(
      context, original, Event(author=agent.name)
  )

  expected = {LIVE_USAGE_KEY: marker, **cached_metadata}
  assert observed == [expected]
  assert original.custom_metadata == expected
  assert LIVE_USAGE_KEY not in cached_metadata
  assert cached_response.custom_metadata == cached_metadata
  if replacement:
    assert result.custom_metadata == expected


@pytest.mark.parametrize("kind", ["copy", "fresh", "inplace", "nested"])
@pytest.mark.parametrize("plugin_callback", [False, True])
async def test_after_model_preserves_owned_metadata(kind, plugin_callback):
  """Callback edits preserve internal identity and the callback's public tags."""
  marker = {"connection_id": "connection-1", "sequence": 2}
  original = LlmResponse(
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=15
      ),
      model_version="live-model",
      live_session_id="provider-session",
      custom_metadata={LIVE_USAGE_KEY: dict(marker), RESTORED_EVENT_KEY: True},
  )

  def callback(callback_context, llm_response):
    del callback_context
    metadata = {
        "user_tag": "voice",
        RESTORED_EVENT_KEY: False,
        INTERNAL_METADATA_PREFIX + "callback_owned": "kept",
    }
    if kind == "copy":
      return llm_response.model_copy(update={"custom_metadata": metadata})
    if kind == "fresh":
      return Event(author="test_agent", custom_metadata=metadata)
    if kind == "nested":
      llm_response.custom_metadata[LIVE_USAGE_KEY]["sequence"] = 99
      llm_response.custom_metadata["user_tag"] = "voice"
      llm_response.custom_metadata[
          INTERNAL_METADATA_PREFIX + "callback_owned"
      ] = "kept"
    else:
      llm_response.custom_metadata = metadata

  agent = Agent(
      name="test_agent",
      after_model_callback=None if plugin_callback else callback,
  )
  plugins = []
  if plugin_callback:
    plugin = BasePlugin(name="metadata-editor")
    plugin.after_model_callback = AsyncMock(side_effect=callback)
    plugins = [plugin]
  context = await testing_utils.create_invocation_context(
      agent=agent, plugins=plugins
  )
  result = await _model_response_finalizer.handle_after_model_callback(
      context, original, Event(author=agent.name)
  )
  response = result if result is not None else original
  assert response.custom_metadata == {
      LIVE_USAGE_KEY: marker,
      RESTORED_EVENT_KEY: True,
      "user_tag": "voice",
      INTERNAL_METADATA_PREFIX + "callback_owned": "kept",
  }
  assert response.model_version == "live-model"
  assert response.live_session_id == "provider-session"
  assert response.usage_metadata.total_token_count == 15


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
  assert result.grounding_metadata == (
      state_metadata if expect_metadata else None
  )
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
  assert result.grounding_metadata == (
      state_metadata if expect_metadata else None
  )
  plugin.after_model_callback.assert_called_once()


@pytest.mark.asyncio
async def test_handle_after_model_callback_caches_canonical_tools():
  """Test that canonical_tools is only called once per invocation_context."""
  canonical_tools_call_count = 0

  async def mock_canonical_tools(self, readonly_context=None):
    nonlocal canonical_tools_call_count
    canonical_tools_call_count += 1
    from google.adk.tools.base_tool import BaseTool

    class MockResearchTool(BaseTool):

      def __init__(self):
        super().__init__(name="research_agent", description="Mock research")
        self.propagate_grounding_metadata = True

      async def call(self, **kwargs):
        return "mock result"

    return [MockResearchTool()]

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
    assert invocation_context.canonical_tools_cache[0].name == "research_agent"

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


# --- Tests for has_meaningful_content ---


def test_has_meaningful_content_none_response():
  assert not _model_response_finalizer.has_meaningful_content(None)


def test_has_meaningful_content_none_content():
  resp = LlmResponse(content=None)
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_empty_parts():
  resp = LlmResponse(content=types.Content(role="model", parts=[]))
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_thought_only():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part(text="Thinking about this...", thought=True)],
      )
  )
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_multiple_thoughts_only():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(text="Step 1...", thought=True),
              types.Part(text="Step 2...", thought=True),
          ],
      )
  )
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_whitespace_only():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_text(text="   \n\t  ")],
      )
  )
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_empty_string():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_text(text="")],
      )
  )
  assert not _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_valid_text():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_text(text="Hello world")],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_thought_and_valid_text():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(text="Thinking...", thought=True),
              types.Part.from_text(text="Final answer."),
          ],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_function_call():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_function_call(name="search", args={})],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_function_response():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_function_response(name="search", response={})],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_executable_code():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  executable_code=types.ExecutableCode(
                      code="print(1)", language=types.Language.PYTHON
                  )
              )
          ],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_code_execution_result():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  code_execution_result=types.CodeExecutionResult(
                      outcome=types.Outcome.OUTCOME_OK, output="1"
                  )
              )
          ],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_inline_data():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  inline_data=types.Blob(data=b"data", mime_type="image/png")
              )
          ],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)


def test_has_meaningful_content_file_data():
  resp = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  file_data=types.FileData(
                      file_uri="gs://bucket/file", mime_type="application/pdf"
                  )
              )
          ],
      )
  )
  assert _model_response_finalizer.has_meaningful_content(resp)

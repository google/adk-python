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

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.multimodal_tool_results_plugin import _CURRENT_TURN_PARTS_ID
from google.adk.plugins.multimodal_tool_results_plugin import _SESSION_UPDATED_KEY
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.adk.plugins.multimodal_tool_results_plugin import PARTS_RETURNED_BY_TOOLS_ID
from google.adk.plugins.multimodal_tool_results_plugin import SESSION_PARTS_RETURNED_BY_TOOLS_ID
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


@pytest.fixture
def plugin() -> MultimodalToolResultsPlugin:
  """Create a default plugin instance for testing."""
  return MultimodalToolResultsPlugin()


@pytest.fixture
def mock_tool() -> BaseTool:
  """Create a mock tool for testing."""
  return Mock(spec=BaseTool)


@pytest.fixture
async def tool_context() -> ToolContext:
  """Create a mock tool context."""
  return ToolContext(
      invocation_context=await testing_utils.create_invocation_context(
          agent=Mock(spec=BaseAgent)
      )
  )


@pytest.mark.asyncio
async def test_tool_returning_parts_are_added_to_llm_request(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Test that parts returned by a tool are present in the llm_request later."""
  parts = [types.Part(text="part1"), types.Part(text="part2")]

  result = await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  assert result == None
  assert PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  assert tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] == parts

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == parts


@pytest.mark.asyncio
async def test_tool_returning_non_list_of_parts_is_unchanged(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Test where tool returning non list of parts, has this result unchanged."""
  original_result = {"some": "data"}

  result = await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=original_result,
  )

  assert result == original_result
  assert PARTS_RETURNED_BY_TOOLS_ID not in tool_context.state

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(
      contents=[types.Content(parts=[types.Part(text="original")])]
  )
  original_parts = list(llm_request.contents[-1].parts)

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == original_parts


@pytest.mark.asyncio
async def test_empty_contents_leaves_saved_parts_pending(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Test that an empty request is a no-op and the parts stay for later."""
  parts = [types.Part(text="part1")]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents == []
  assert tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] == parts


@pytest.mark.asyncio
async def test_session_retention_reattaches_parts_across_turns():
  """Verify that saved parts survive a real turn boundary in session retention mode.

  This must go through two separate runner.run_async() calls against a real
  session service: the saved parts are stored under a "temp:"-prefixed key by
  default, and that prefix is stripped by BaseSessionService before an event is
  persisted, so a test that only calls before_model_callback twice on the same
  in-memory State object (without an intervening append_event()) cannot detect
  whether the parts actually survive a real turn boundary.
  """
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/document.pdf", mime_type="application/pdf"
      )
  )
  another_file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/another_document.pdf",
          mime_type="application/pdf",
      )
  )

  def get_document() -> types.Part:
    return file_part

  def get_another_document() -> types.Part:
    return another_file_part

  mock_model = testing_utils.MockModel.create(
      responses=[
          # Turn 1
          types.Part.from_function_call(name="get_document", args={}),
          "Here is a summary of the document.",
          # Turn 2
          types.Part.from_function_call(name="get_another_document", args={}),
          "Here is a summary of the second document.",
      ]
  )
  agent = Agent(
      name="root_agent",
      model=mock_model,
      tools=[get_document, get_another_document],
  )
  runner = testing_utils.InMemoryRunner(
      agent, plugins=[MultimodalToolResultsPlugin(retention="session")]
  )

  # Turn 1: triggers the tool call.
  await runner.run_async("Please fetch the document")
  # Turn 2: a NEW invocation, sharing the same session as turn 1.
  # This turn also triggers a tool call.
  await runner.run_async("Please fetch another document")

  assert len(mock_model.requests) == 4
  # Turn 1's first request precedes the tool call: nothing attached yet.
  assert file_part not in mock_model.requests[0].contents[-1].parts
  # Turn 1's second request: parts attached within the same invocation.
  assert file_part in mock_model.requests[1].contents[-1].parts

  # Turn 2's first request: Turn 1 parts must still be attached.
  assert file_part in mock_model.requests[2].contents[-1].parts
  assert another_file_part not in mock_model.requests[2].contents[-1].parts

  # Turn 2's second request: should have another_file_part attached,
  # and file_part should have been replaced (not accumulated).
  assert another_file_part in mock_model.requests[3].contents[-1].parts
  assert file_part not in mock_model.requests[3].contents[-1].parts


@pytest.mark.asyncio
async def test_multiple_tools_returning_parts_are_accumulated(
    plugin: MultimodalToolResultsPlugin,
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Test that parts from multiple tool calls are accumulated."""
  parts1 = [types.Part(text="part1")]
  parts2 = [types.Part(text="part2")]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts1,
  )

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts2,
  )

  assert PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  assert tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] == parts1 + parts2

  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert llm_request.contents[-1].parts == parts1 + parts2


@pytest.mark.asyncio
async def test_session_retention_serializes_parts_in_state(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify that session retention stores serialized parts (dicts) in state."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  parts = [types.Part(text="part1"), types.Part(text="part2")]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  assert SESSION_PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  stored = tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID]
  assert all(isinstance(p, dict) for p in stored)
  assert stored == [p.model_dump(mode="json") for p in parts]


@pytest.mark.asyncio
async def test_session_retention_replaces_parts_on_new_invocation(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify that session retention replaces parts from previous turn on new tool call."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  parts_turn1 = [types.Part(text="part1")]
  parts_turn2 = [types.Part(text="part2")]

  # Simulate Turn 1
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts_turn1,
  )
  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      p.model_dump(mode="json") for p in parts_turn1
  ]

  # Simulate end of Turn 1 by stripping temp keys.
  for key in [_SESSION_UPDATED_KEY, _CURRENT_TURN_PARTS_ID]:
    if key in tool_context.state._value:
      del tool_context.state._value[key]
    if key in tool_context.state._delta:
      del tool_context.state._delta[key]

  # Simulate Turn 2
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts_turn2,
  )
  # It should replace parts_turn1 with parts_turn2, not accumulate.
  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      p.model_dump(mode="json") for p in parts_turn2
  ]


@pytest.mark.asyncio
async def test_session_retention_accumulates_parts_within_same_invocation(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify that session retention accumulates parts from multiple tool calls in same turn."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  parts1 = [types.Part(text="part1")]
  parts2 = [types.Part(text="part2")]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts1,
  )
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts2,
  )

  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      p.model_dump(mode="json") for p in parts1 + parts2
  ]


@pytest.mark.asyncio
async def test_session_retention_skips_binary_parts(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify that session retention skips binary parts (inline_data) and keeps them as temp."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  binary_part = types.Part.from_bytes(
      data=b"fake image data", mime_type="image/png"
  )
  parts = [binary_part]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  # Should not be in session state
  assert SESSION_PARTS_RETURNED_BY_TOOLS_ID not in tool_context.state

  # Should be in temp state
  assert _CURRENT_TURN_PARTS_ID in tool_context.state
  assert tool_context.state[_CURRENT_TURN_PARTS_ID] == parts

  # Verify before_model_callback attaches it
  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  restored_part = llm_request.contents[-1].parts[0]
  assert restored_part == binary_part
  # Temp state should be cleared
  assert tool_context.state[_CURRENT_TURN_PARTS_ID] == []


@pytest.mark.asyncio
async def test_session_retention_mixed_parts(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify that session retention splits mixed parts correctly."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/doc.pdf", mime_type="application/pdf"
      )
  )
  binary_part = types.Part.from_bytes(
      data=b"fake image data", mime_type="image/png"
  )
  parts = [file_part, binary_part]

  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=parts,
  )

  # file_part should be serialized in session state
  assert SESSION_PARTS_RETURNED_BY_TOOLS_ID in tool_context.state
  session_stored = tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID]
  assert len(session_stored) == 1
  assert session_stored[0] == file_part.model_dump(mode="json")

  # all parts should be in current turn state (preserving order)
  assert _CURRENT_TURN_PARTS_ID in tool_context.state
  temp_stored = tool_context.state[_CURRENT_TURN_PARTS_ID]
  assert len(temp_stored) == 2
  assert temp_stored == parts

  # Verify before_model_callback
  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  restored_parts = llm_request.contents[-1].parts
  assert len(restored_parts) == 2
  # Order: original order preserved
  assert restored_parts == [file_part, binary_part]

  # Temp state should be cleared, session state remains
  assert tool_context.state[_CURRENT_TURN_PARTS_ID] == []
  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      file_part.model_dump(mode="json")
  ]


@pytest.mark.asyncio
async def test_session_retention_retains_across_turns_when_intermediate_turn_only_returns_binary_parts(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify session parts are not lost if an intermediate turn only returns binary parts."""
  plugin = MultimodalToolResultsPlugin(retention="session")
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/doc.pdf", mime_type="application/pdf"
      )
  )
  binary_part = types.Part.from_bytes(
      data=b"fake image data", mime_type="image/png"
  )

  # Turn 1: returns file_part (session part)
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=[file_part],
  )
  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      file_part.model_dump(mode="json")
  ]

  # Simulate end of Turn 1 by stripping temp keys.
  for key in [_SESSION_UPDATED_KEY, _CURRENT_TURN_PARTS_ID]:
    if key in tool_context.state._value:
      del tool_context.state._value[key]
    if key in tool_context.state._delta:
      del tool_context.state._delta[key]

  # Turn 2: returns ONLY binary_part (inline_data)
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=[binary_part],
  )

  # file_part should STILL be in session state (not cleared by the binary-only turn)
  assert tool_context.state[SESSION_PARTS_RETURNED_BY_TOOLS_ID] == [
      file_part.model_dump(mode="json")
  ]
  # binary_part should be in temp state
  assert tool_context.state[_CURRENT_TURN_PARTS_ID] == [binary_part]


def test_invalid_retention_raises_value_error():
  with pytest.raises(ValueError, match="retention must be"):
    MultimodalToolResultsPlugin(retention="invalid")  # type: ignore[arg-type] # Testing runtime validation


@pytest.mark.asyncio
async def test_session_retention_retains_parts_on_subsequent_model_calls_in_same_turn(
    mock_tool: BaseTool,
    tool_context: ToolContext,
):
  """Verify session parts are retained on later model calls in the same turn.

  Even if a subsequent tool call in that turn does not return any parts.
  """
  plugin = MultimodalToolResultsPlugin(retention="session")
  file_part = types.Part(
      file_data=types.FileData(
          file_uri="gs://bucket/doc.pdf", mime_type="application/pdf"
      )
  )

  # Tool 1: returns file_part (session part)
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result=[file_part],
  )

  # Model call 1: should attach file_part
  callback_context = Mock(spec=CallbackContext)
  callback_context.state = tool_context.state
  llm_request_1 = LlmRequest(contents=[types.Content(parts=[])])

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request_1
  )
  assert llm_request_1.contents[-1].parts == [file_part]
  # CURRENT_TURN_PARTS_ID should be cleared now
  assert tool_context.state[_CURRENT_TURN_PARTS_ID] == []

  # Tool 2: returns a plain dict (no parts)
  await plugin.after_tool_callback(
      tool=mock_tool,
      tool_args={},
      tool_context=tool_context,
      result={"status": "ok"},
  )

  # Model call 2: should STILL attach file_part
  llm_request_2 = LlmRequest(contents=[types.Content(parts=[])])
  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request_2
  )
  assert llm_request_2.contents[-1].parts == [file_part]

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

"""Tests for ToolSpecMockStrategy."""

from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.environment_simulation.strategies import tool_spec_mock_strategy
from google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy import ToolSpecMockStrategy
from google.adk.tools.environment_simulation.tool_connection_map import StatefulParameter
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.genai import types
import pytest


def _make_strategy(response_chunks: List[str]) -> ToolSpecMockStrategy:
  """Builds a strategy whose LLM streams back ``response_chunks``."""

  async def fake_generate_content_async(request):
    for chunk in response_chunks:
      yield LlmResponse(
          content=types.Content(role="model", parts=[types.Part(text=chunk)])
      )

  mock_llm = MagicMock()
  mock_llm.generate_content_async = fake_generate_content_async

  with patch.object(
      tool_spec_mock_strategy, "LLMRegistry", autospec=True
  ) as mock_registry:
    mock_registry.return_value.resolve.return_value = MagicMock(
        return_value=mock_llm
    )
    return ToolSpecMockStrategy(
        llm_name="fake-model",
        llm_config=types.GenerateContentConfig(),
    )


def _make_tool(name: str, declared: bool = True) -> MagicMock:
  tool = MagicMock()
  tool.name = name
  tool.description = f"{name} description"
  tool._get_declaration.return_value = (
      types.FunctionDeclaration(name=name) if declared else None
  )
  return tool


def _connection_map(
    parameter_name: str, creating: List[str], consuming: List[str]
) -> ToolConnectionMap:
  return ToolConnectionMap(
      stateful_parameters=[
          StatefulParameter(
              parameter_name=parameter_name,
              creating_tools=creating,
              consuming_tools=consuming,
          )
      ]
  )


async def _mock(
    strategy: ToolSpecMockStrategy,
    tool: MagicMock,
    state_store: Dict[str, Any],
    connection_map: ToolConnectionMap = None,
    args: Dict[str, Any] = None,
) -> Dict[str, Any]:
  return await strategy.mock(
      tool=tool,
      args=args if args is not None else {},
      tool_context=None,
      tool_connection_map=connection_map,
      state_store=state_store,
  )


@pytest.mark.asyncio
async def test_tool_without_declaration_is_reported_as_an_error():
  """Without a schema there is nothing to mock against, so no LLM call."""
  strategy = _make_strategy(['{"ok": true}'])

  result = await _mock(strategy, _make_tool("t", declared=False), {})

  assert result == {
      "status": "error",
      "error_message": "Could not get tool declaration.",
  }


@pytest.mark.asyncio
async def test_fenced_json_response_is_unwrapped():
  """Models often wrap JSON in a markdown fence; the fence is not data."""
  strategy = _make_strategy(['```json\n{"ticket_id": "T-1"}\n```'])

  result = await _mock(strategy, _make_tool("create_ticket"), {})

  assert result == {"ticket_id": "T-1"}


@pytest.mark.asyncio
async def test_streamed_chunks_are_concatenated_before_parsing():
  """A response split across stream events is still one JSON document."""
  strategy = _make_strategy(['{"ticket', '_id": "T-2"}'])

  result = await _mock(strategy, _make_tool("create_ticket"), {})

  assert result == {"ticket_id": "T-2"}


@pytest.mark.asyncio
async def test_unparseable_response_is_returned_as_an_error_with_raw_output():
  """The caller needs the raw text to debug why the model went off-format."""
  strategy = _make_strategy(["sorry, I cannot do that"])

  result = await _mock(strategy, _make_tool("create_ticket"), {})

  assert result == {
      "status": "error",
      "error_message": "Failed to generate valid JSON mock response.",
      "llm_output": "sorry, I cannot do that",
  }


@pytest.mark.asyncio
async def test_creating_tool_records_the_new_entity_in_the_state_store():
  """A tool that creates an id must leave it behind for consuming tools."""
  strategy = _make_strategy(['{"ticket_id": "T-3", "status": "open"}'])
  state_store = {}

  result = await _mock(
      strategy,
      _make_tool("create_ticket"),
      state_store,
      _connection_map("ticket_id", ["create_ticket"], ["get_ticket"]),
  )

  assert state_store == {"ticket_id": {"T-3": result}}


@pytest.mark.asyncio
async def test_state_store_entry_is_keyed_by_a_nested_parameter_value():
  """The id is looked up anywhere in the response, not just at the top level."""
  strategy = _make_strategy(['{"data": {"ticket_id": "T-4"}}'])
  state_store = {}

  result = await _mock(
      strategy,
      _make_tool("create_ticket"),
      state_store,
      _connection_map("ticket_id", ["create_ticket"], []),
  )

  assert state_store == {"ticket_id": {"T-4": result}}


@pytest.mark.asyncio
async def test_consuming_tool_does_not_write_to_the_state_store():
  """Only creating tools own state; a reader must not invent entries."""
  strategy = _make_strategy(['{"ticket_id": "T-5"}'])
  state_store = {}

  await _mock(
      strategy,
      _make_tool("get_ticket"),
      state_store,
      _connection_map("ticket_id", ["create_ticket"], ["get_ticket"]),
  )

  assert state_store == {}


@pytest.mark.asyncio
async def test_existing_state_entries_are_kept_when_a_new_one_is_added():
  """Creating a second entity must not drop the first one."""
  strategy = _make_strategy(['{"ticket_id": "T-7"}'])
  state_store = {"ticket_id": {"T-6": {"ticket_id": "T-6"}}}

  result = await _mock(
      strategy,
      _make_tool("create_ticket"),
      state_store,
      _connection_map("ticket_id", ["create_ticket"], []),
  )

  assert state_store["ticket_id"]["T-6"] == {"ticket_id": "T-6"}
  assert state_store["ticket_id"]["T-7"] == result


@pytest.mark.asyncio
async def test_missing_parameter_in_response_leaves_state_untouched():
  """Nothing to key the entry by, so no half-formed entry is written."""
  strategy = _make_strategy(['{"status": "open"}'])
  state_store = {}

  await _mock(
      strategy,
      _make_tool("create_ticket"),
      state_store,
      _connection_map("ticket_id", ["create_ticket"], []),
  )

  assert state_store == {}


@pytest.mark.asyncio
async def test_no_connection_map_means_no_state_tracking():
  strategy = _make_strategy(['{"ticket_id": "T-8"}'])
  state_store = {}

  result = await _mock(strategy, _make_tool("create_ticket"), state_store)

  assert result == {"ticket_id": "T-8"}
  assert state_store == {}

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

from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.events.event import Event
from google.adk.plugins.plugin_manager import PluginManager
from google.genai import types
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
import pytest


@pytest.mark.parametrize(
    "checkpointer_value, events_list, expected_messages",
    [
        (
            MagicMock(),
            [
                Event(
                    invocation_id="test_invocation_id",
                    author="user",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="test prompt")],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="root_agent",
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="(some delegation)")],
                    ),
                ),
            ],
            [
                SystemMessage(content="test system prompt"),
                HumanMessage(content="test prompt"),
            ],
        ),
        (
            None,
            [
                Event(
                    invocation_id="test_invocation_id",
                    author="user",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="user prompt 1")],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="root_agent",
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text="root agent response")
                        ],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="weather_agent",
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text="weather agent response")
                        ],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="user",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="user prompt 2")],
                    ),
                ),
            ],
            [
                SystemMessage(content="test system prompt"),
                HumanMessage(content="user prompt 1"),
                AIMessage(content="weather agent response"),
                HumanMessage(content="user prompt 2"),
            ],
        ),
        (
            MagicMock(),
            [
                Event(
                    invocation_id="test_invocation_id",
                    author="user",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="user prompt 1")],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="root_agent",
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text="root agent response")
                        ],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="weather_agent",
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text="weather agent response")
                        ],
                    ),
                ),
                Event(
                    invocation_id="test_invocation_id",
                    author="user",
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="user prompt 2")],
                    ),
                ),
            ],
            [
                SystemMessage(content="test system prompt"),
                HumanMessage(content="user prompt 2"),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_langgraph_agent(
    checkpointer_value, events_list, expected_messages
):
  mock_graph = MagicMock(spec=CompiledGraph)
  mock_graph_state = MagicMock()
  mock_graph_state.values = {}
  mock_graph.get_state.return_value = mock_graph_state

  mock_graph.checkpointer = checkpointer_value
  mock_graph.invoke.return_value = {
      "messages": [AIMessage(content="test response")]
  }

  mock_parent_context = MagicMock(spec=InvocationContext)
  mock_session = MagicMock()
  mock_parent_context.session = mock_session
  mock_parent_context.branch = "parent_agent"
  mock_parent_context.end_invocation = False
  mock_session.events = events_list
  mock_parent_context.invocation_id = "test_invocation_id"
  mock_parent_context.model_copy.return_value = mock_parent_context
  mock_parent_context.plugin_manager = PluginManager(plugins=[])

  weather_agent = LangGraphAgent(
      name="weather_agent",
      description="A agent that answers weather questions",
      instruction="test system prompt",
      graph=mock_graph,
  )

  result_event = None
  async for event in weather_agent.run_async(mock_parent_context):
    result_event = event

  assert result_event.author == "weather_agent"
  assert result_event.content.parts[0].text == "test response"

  mock_graph.invoke.assert_called_once()
  mock_graph.invoke.assert_called_with(
      {"messages": expected_messages},
      {"configurable": {"thread_id": mock_session.id}},
  )

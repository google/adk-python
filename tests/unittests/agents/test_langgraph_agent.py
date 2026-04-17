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

from unittest.mock import MagicMock

import pytest

# Skip all tests in this module if LangGraph dependencies are not available
LANGGRAPH_AVAILABLE = True
try:
  from google.adk.agents.invocation_context import InvocationContext
  from google.adk.agents.langgraph_agent import LangGraphAgent
  from google.adk.events.event import Event
  from google.adk.plugins.plugin_manager import PluginManager
  from google.genai import types
  from langchain_core.messages import AIMessage
  from langchain_core.messages import HumanMessage
  from langchain_core.messages import SystemMessage
  from langgraph.graph.graph import CompiledGraph
except ImportError:
  LANGGRAPH_AVAILABLE = False

  # IMPORTANT: Dummy classes are REQUIRED in this file but NOT in A2A test files.
  # Here's why this file is different from A2A test files:
  #
  # 1. MODULE-LEVEL USAGE IN DECORATORS:
  #    This file uses @pytest.mark.parametrize decorator with complex nested structures
  #    that directly reference imported types like Event(), types.Content(), types.Part.from_text().
  #    These decorator expressions are evaluated during MODULE COMPILATION TIME,
  #    not during test execution time.
  #
  # 2. A2A TEST FILES PATTERN:
  #    Most A2A test files only use imported types within test method bodies:
  #    - Inside test functions: def test_something(): Message(...)
  #    - These are evaluated during TEST EXECUTION TIME when tests are skipped
  #    - No NameError occurs because skipped tests don't execute their bodies
  #
  # 3. WHAT HAPPENS WITHOUT DUMMIES:
  #    If we remove dummy classes from this file:
  #    - Python tries to compile the @pytest.mark.parametrize decorator
  #    - It encounters Event(...), types.Content(...), etc.
  #    - These names are undefined → NameError during module compilation
  #    - Test collection fails before pytest.mark.skipif can even run
  #
  # 4. WHY DUMMIES WORK:
  #    - DummyTypes() can be called like Event() → returns DummyTypes instance
  #    - DummyTypes.__getattr__ handles types.Content → returns DummyTypes instance
  #    - DummyTypes.__call__ handles types.Part.from_text() → returns DummyTypes instance
  #    - The parametrize decorator gets dummy objects instead of real ones
  #    - Tests still get skipped due to pytestmark, so dummies never actually run
  #
  # 5. EXCEPTION CASES IN A2A FILES:
  #    A few A2A files DID need dummies initially because they had:
  #    - Type annotations: def create_helper(x: str) -> Message
  #    - But we removed those type annotations to eliminate the need for dummies
  #
  # This file cannot avoid dummies because the parametrize decorator usage
  # is fundamental to the test structure and cannot be easily refactored.

  class DummyTypes:

    def __getattr__(self, name):
      return DummyTypes()

    def __call__(self, *args, **kwargs):
      return DummyTypes()

  InvocationContext = DummyTypes()
  LangGraphAgent = DummyTypes()
  Event = DummyTypes()
  PluginManager = DummyTypes()
  types = (
      DummyTypes()
  )  # Must support chained calls like types.Content(), types.Part.from_text()
  AIMessage = DummyTypes()
  HumanMessage = DummyTypes()
  SystemMessage = DummyTypes()
  CompiledGraph = DummyTypes()

pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE, reason="LangGraph dependencies not available"
)


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

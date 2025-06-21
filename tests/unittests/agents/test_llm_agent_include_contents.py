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

"""Unit tests for LlmAgent include_contents field behavior."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.genai import types
import pytest

from .. import testing_utils


@pytest.mark.asyncio
async def test_include_contents_default_behavior():
  """Test that include_contents='default' preserves conversation history including tool interactions."""

  def simple_tool(message: str) -> dict:
    return {"result": f"Tool processed: {message}"}

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),  # First turn: tool call
          "First response based on tool result",  # First turn: final response
          types.Part.from_function_call(
              name="simple_tool", args={"message": "second"}
          ),  # Second turn: tool call
          "Second response based on tool result",  # Second turn: final response
      ]
  )

  agent = LlmAgent(
      name="test_agent",
      model=mock_model,
      include_contents="default",
      instruction="You are a helpful assistant",
      tools=[simple_tool],
  )

  runner = testing_utils.InMemoryRunner(agent)

  # First turn with tool usage
  runner.run("First message")

  # Second turn with tool usage
  runner.run("Second message")

  # Examine what was sent to LLM in each request:

  # First turn, first request: should only have first user message
  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "First message")
  ]

  # First turn, second request: should have user message + tool call + tool response
  assert testing_utils.simplify_contents(mock_model.requests[1].contents) == [
      ("user", "First message"),
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool", response={"result": "Tool processed: first"}
          ),
      ),
  ]

  # Second turn, first request: should have FULL conversation history
  assert testing_utils.simplify_contents(mock_model.requests[2].contents) == [
      ("user", "First message"),  # Previous user message
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool", response={"result": "Tool processed: first"}
          ),
      ),
      (
          "model",
          "First response based on tool result",
      ),
      ("user", "Second message"),
  ]

  # Second turn, second request: should have full history + current tool
  # interaction
  assert testing_utils.simplify_contents(mock_model.requests[3].contents) == [
      ("user", "First message"),  # Previous conversation
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool", response={"result": "Tool processed: first"}
          ),
      ),
      ("model", "First response based on tool result"),
      ("user", "Second message"),
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "second"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool", response={"result": "Tool processed: second"}
          ),
      ),
  ]


@pytest.mark.asyncio
async def test_include_contents_none_behavior():
  """Test that include_contents='none' should exclude conversation history but include current input."""

  def simple_tool(message: str) -> dict:
    return {"result": f"Tool processed: {message}"}

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),  # First turn: tool call
          "First response based on tool result",  # First turn: final response
          "Just a simple response",  # Second turn: simple response (no tools)
      ]
  )

  agent = LlmAgent(
      name="test_agent",
      model=mock_model,
      include_contents="none",
      instruction="You are a helpful assistant",
      tools=[simple_tool],
  )

  runner = testing_utils.InMemoryRunner(agent)

  # First turn with tool usage
  runner.run("First message")

  # Second turn with just text (to test history exclusion clearly)
  runner.run("Second message")

  # Verify expected behavior:

  # First turn, first request: should have current user input
  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "First message")  # Current input should be included
  ]

  # First turn, second request: should have current tool interaction
  assert testing_utils.simplify_contents(mock_model.requests[1].contents) == [
      ("user", "First message"),  # Current turn input
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool", response={"result": "Tool processed: first"}
          ),
      ),
  ]

  # Second turn: should have ONLY current input, NO history from first turn
  assert testing_utils.simplify_contents(mock_model.requests[2].contents) == [
      ("user", "Second message")  # ONLY current input, no previous conversation
  ]

  # Instructions and tools should be preserved
  assert (
      "You are a helpful assistant"
      in mock_model.requests[0].config.system_instruction
  )
  assert len(mock_model.requests[0].config.tools) > 0


@pytest.mark.asyncio
async def test_include_contents_none_sequential_agents():
  """Test include_contents='none' in a sequential agent scenario.

  This test verifies the behavior when using include_contents='none' in a
  sequential agent setup. It ensures that:
  1. User provides input to Agent1
  2. Agent1 processes and responds
  3. Agent2 receives Agent1's output as its current turn context
  4. Agent2 with include_contents='none' should only see Agent1's message,
     not the original user input (which is considered conversation history)
  """

  agent1_model = testing_utils.MockModel.create(
      responses=["Agent1 processed your request and found: XYZ"]
  )

  agent1 = LlmAgent(
      name="agent1",
      model=agent1_model,
      instruction="You are Agent1, analyze the user request",
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Agent2 final response based on Agent1 analysis"]
  )

  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_contents="none",
      instruction=(
          "You are Agent2, provide final response based on previous analysis"
      ),
  )

  sequential_agent = SequentialAgent(
      name="sequential_test_agent", sub_agents=[agent1, agent2]
  )

  runner = testing_utils.InMemoryRunner(sequential_agent)

  original_request = "Original user request that should not be seen by Agent2"
  events = runner.run(original_request)

  assert len(events) == 2
  assert events[0].author == "agent1"
  assert events[1].author == "agent2"

  agent1_contents = testing_utils.simplify_contents(
      agent1_model.requests[0].contents
  )
  assert ("user", original_request) in agent1_contents

  agent2_contents = testing_utils.simplify_contents(
      agent2_model.requests[0].contents
  )

  assert not any(
      original_request in str(role_content)
      for _, role_content in agent2_contents
  ), "Agent2 with include_contents='none' should not see conversation history"

  assert any(
      "Agent1 processed your request" in str(role_content)
      for _, role_content in agent2_contents
  ), "Agent2 should see Agent1's output as current turn context"

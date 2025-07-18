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
          ),
          "First response",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "second"}
          ),
          "Second response",
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
  runner.run("First message")
  runner.run("Second message")

  # First turn requests
  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "First message")
  ]

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

  # Second turn should include full conversation history
  assert testing_utils.simplify_contents(mock_model.requests[2].contents) == [
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
      ("model", "First response"),
      ("user", "Second message"),
  ]

  # Second turn with tool should include full history + current tool interaction
  assert testing_utils.simplify_contents(mock_model.requests[3].contents) == [
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
      ("model", "First response"),
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
  """Test that include_contents='none' excludes conversation history but includes current input."""

  def simple_tool(message: str) -> dict:
    return {"result": f"Tool processed: {message}"}

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="simple_tool", args={"message": "first"}
          ),
          "First response",
          "Second response",
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
  runner.run("First message")
  runner.run("Second message")

  # First turn behavior
  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "First message")
  ]

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

  # Second turn should only have current input, no history
  assert testing_utils.simplify_contents(mock_model.requests[2].contents) == [
      ("user", "Second message")
  ]

  # System instruction and tools should be preserved
  assert (
      "You are a helpful assistant"
      in mock_model.requests[0].config.system_instruction
  )
  assert len(mock_model.requests[0].config.tools) > 0


@pytest.mark.asyncio
async def test_include_contents_none_sequential_agents():
  """Test include_contents='none' with sequential agents."""

  agent1_model = testing_utils.MockModel.create(
      responses=["Agent1 response: XYZ"]
  )
  agent1 = LlmAgent(
      name="agent1",
      model=agent1_model,
      instruction="You are Agent1",
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Agent2 final response"]
  )
  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_contents="none",
      instruction="You are Agent2",
  )

  sequential_agent = SequentialAgent(
      name="sequential_test_agent", sub_agents=[agent1, agent2]
  )

  runner = testing_utils.InMemoryRunner(sequential_agent)
  events = runner.run("Original user request")

  assert len(events) == 2
  assert events[0].author == "agent1"
  assert events[1].author == "agent2"

  # Agent1 sees original user request
  agent1_contents = testing_utils.simplify_contents(
      agent1_model.requests[0].contents
  )
  assert ("user", "Original user request") in agent1_contents

  # Agent2 with include_contents='none' should not see original request
  agent2_contents = testing_utils.simplify_contents(
      agent2_model.requests[0].contents
  )

  assert not any(
      "Original user request" in str(content) for _, content in agent2_contents
  )
  assert any(
      "Agent1 response" in str(content) for _, content in agent2_contents
  )

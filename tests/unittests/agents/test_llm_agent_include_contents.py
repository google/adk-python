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

"""Unit tests for LlmAgent include_contents field behavior."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
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


def test_model_input_context_is_sent_to_model_without_persisting_to_session():
  mock_model = testing_utils.MockModel.create(responses=["Answer"])
  agent = LlmAgent(name="test_agent", model=mock_model)
  runner = testing_utils.InMemoryRunner(agent)
  session = runner.session

  list(
      runner.runner.run(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content("Question"),
          run_config=RunConfig(
              model_input_context=[
                  types.UserContent("Relevant context for this turn")
              ]
          ),
      )
  )

  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "Relevant context for this turn"),
      ("user", "Question"),
  ]
  assert testing_utils.simplify_events(runner.session.events) == [
      ("user", "Question"),
      ("test_agent", "Answer"),
  ]


def test_model_input_context_stays_before_user_message_after_tool_call():
  def simple_tool(message: str) -> dict:
    return {"result": f"Tool processed: {message}"}

  mock_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(
              name="simple_tool", args={"message": "payload"}
          ),
          "Answer",
      ]
  )
  agent = LlmAgent(name="test_agent", model=mock_model, tools=[simple_tool])
  runner = testing_utils.InMemoryRunner(agent)
  session = runner.session

  list(
      runner.runner.run(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content("Question"),
          run_config=RunConfig(
              model_input_context=[
                  types.UserContent("Relevant context for this turn")
              ]
          ),
      )
  )

  assert testing_utils.simplify_contents(mock_model.requests[0].contents) == [
      ("user", "Relevant context for this turn"),
      ("user", "Question"),
  ]
  assert testing_utils.simplify_contents(mock_model.requests[1].contents) == [
      ("user", "Relevant context for this turn"),
      ("user", "Question"),
      (
          "model",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "payload"}
          ),
      ),
      (
          "user",
          types.Part.from_function_response(
              name="simple_tool",
              response={"result": "Tool processed: payload"},
          ),
      ),
  ]
  assert testing_utils.simplify_events(runner.session.events) == [
      ("user", "Question"),
      (
          "test_agent",
          types.Part.from_function_call(
              name="simple_tool", args={"message": "payload"}
          ),
      ),
      (
          "test_agent",
          types.Part.from_function_response(
              name="simple_tool",
              response={"result": "Tool processed: payload"},
          ),
      ),
      ("test_agent", "Answer"),
  ]


def test_model_input_context_with_include_contents_none_sub_agent():
  agent1_model = testing_utils.MockModel.create(
      responses=["Agent1 response: XYZ"]
  )
  agent1 = LlmAgent(name="agent1", model=agent1_model)

  agent2_model = testing_utils.MockModel.create(
      responses=["Agent2 final response"]
  )
  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_contents="none",
  )
  sequential_agent = SequentialAgent(
      name="sequential_test_agent", sub_agents=[agent1, agent2]
  )
  runner = testing_utils.InMemoryRunner(sequential_agent)
  session = runner.session

  list(
      runner.runner.run(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content("Original user request"),
          run_config=RunConfig(
              model_input_context=[
                  types.UserContent("Relevant context for this turn")
              ]
          ),
      )
  )

  assert testing_utils.simplify_contents(agent1_model.requests[0].contents) == [
      ("user", "Relevant context for this turn"),
      ("user", "Original user request"),
  ]
  assert testing_utils.simplify_contents(agent2_model.requests[0].contents) == [
      ("user", "Relevant context for this turn"),
      (
          "user",
          [
              types.Part(text="For context:"),
              types.Part(text="[agent1] said: Agent1 response: XYZ"),
          ],
      ),
  ]


def test_model_input_context_without_user_message_is_prepended_before_history():
  mock_model = testing_utils.MockModel.create(
      responses=["First answer", "Second answer"]
  )
  agent = LlmAgent(name="test_agent", model=mock_model)
  runner = testing_utils.InMemoryRunner(agent)
  session = runner.session

  list(
      runner.runner.run(
          user_id=session.user_id,
          session_id=session.id,
          new_message=testing_utils.get_user_content("First question"),
      )
  )
  # No new_message, so the invocation has no user content to anchor before
  # (e.g. live mode or a re-run over existing history). The transient context
  # falls back to the front of the request, before all prior history.
  list(
      runner.runner.run(
          user_id=session.user_id,
          session_id=session.id,
          new_message=None,
          run_config=RunConfig(
              model_input_context=[
                  types.UserContent("Relevant context for this turn")
              ]
          ),
      )
  )

  assert testing_utils.simplify_contents(mock_model.requests[1].contents) == [
      ("user", "Relevant context for this turn"),
      ("user", "First question"),
      ("model", "First answer"),
  ]
  assert testing_utils.simplify_events(runner.session.events) == [
      ("user", "First question"),
      ("test_agent", "First answer"),
      ("test_agent", "Second answer"),
  ]


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

  simplified_events = [event for event in events if event.content]
  assert len(simplified_events) == 2
  assert "Agent1 response" in str(simplified_events[0].content)
  assert "Agent2 final response" in str(simplified_events[1].content)

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

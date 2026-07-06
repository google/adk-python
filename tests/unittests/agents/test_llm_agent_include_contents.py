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

"""Unit tests for LlmAgent include_contents and include_sources field behavior."""

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


# ---------------------------------------------------------------------------
# include_sources: field validation
# ---------------------------------------------------------------------------


def test_include_sources_empty_list_raises():
  """include_sources=[] must raise ValueError — use None to disable filtering."""
  with pytest.raises(ValueError, match="include_sources=\\[\\]"):
    LlmAgent(
        name="agent",
        model="gemini-2.5-flash",
        include_sources=[],
    )


def test_include_sources_none_is_accepted():
  """include_sources=None (default) must not raise."""
  agent = LlmAgent(name="agent", model="gemini-2.5-flash", include_sources=None)
  assert agent.include_sources is None


# ---------------------------------------------------------------------------
# include_sources: integration — user-only in sequential pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_sources_user_only_drops_upstream_agent_entries():
  """Downstream agent with include_sources=['user'] receives only the human user message."""
  agent1_model = testing_utils.MockModel.create(
      responses=["Upstream agent reply"]
  )
  agent1 = LlmAgent(
      name="upstream",
      model=agent1_model,
      instruction="You are upstream",
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Downstream response"]
  )
  agent2 = LlmAgent(
      name="downstream",
      model=agent2_model,
      include_sources=["user"],
      instruction="You are downstream",
  )

  sequential = SequentialAgent(name="pipeline", sub_agents=[agent1, agent2])
  runner = testing_utils.InMemoryRunner(sequential)
  runner.run("Original user request")

  agent2_contents = testing_utils.simplify_contents(
      agent2_model.requests[0].contents
  )

  # User message must be present
  assert any("Original user request" in str(c) for _, c in agent2_contents)
  # Upstream agent's narrative entry must be absent
  assert not any("Upstream agent reply" in str(c) for _, c in agent2_contents)
  assert not any("For context:" in str(c) for _, c in agent2_contents)


# ---------------------------------------------------------------------------
# include_sources: composing with include_contents='default' — multi-turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_sources_user_self_drops_upstream_across_turns():
  """include_sources=['user','self'] + include_contents='default' (full history):
  downstream agent sees all user messages and its own prior turns, but no
  narrative entries from the upstream agent across multiple invocations.
  """
  agent1_model = testing_utils.MockModel.create(
      responses=["Turn1 upstream reply", "Turn2 upstream reply"]
  )
  agent1 = LlmAgent(
      name="upstream",
      model=agent1_model,
      instruction="You are upstream",
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Turn1 downstream", "Turn2 downstream"]
  )
  agent2 = LlmAgent(
      name="downstream",
      model=agent2_model,
      include_sources=["user", "self"],
      instruction="You are downstream",
  )

  sequential = SequentialAgent(name="pipeline", sub_agents=[agent1, agent2])
  runner = testing_utils.InMemoryRunner(sequential)
  runner.run("Turn 1 user message")
  runner.run("Turn 2 user message")

  # Second invocation of downstream agent — should see user messages + own
  # prior turn, but not upstream's narrative entries.
  agent2_second_contents = testing_utils.simplify_contents(
      agent2_model.requests[1].contents
  )

  # User messages must be present
  assert any("Turn 1 user message" in str(c) for _, c in agent2_second_contents)
  assert any("Turn 2 user message" in str(c) for _, c in agent2_second_contents)
  # Upstream agent's narrative entries must be absent
  assert not any(
      "upstream reply" in str(c).lower() for _, c in agent2_second_contents
  )
  assert not any("For context:" in str(c) for _, c in agent2_second_contents)


# ---------------------------------------------------------------------------
# include_contents='default' + include_sources combinations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_contents_default_with_source_filter_user():
  """include_contents='default' + include_sources=['user'] keeps only user messages across full history."""
  agent1_model = testing_utils.MockModel.create(
      responses=["Agent1 result turn1", "Agent1 result turn2"]
  )
  agent1 = LlmAgent(
      name="agent1", model=agent1_model, instruction="You are agent1"
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Turn1 response", "Turn2 response"]
  )
  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_sources=["user"],
      instruction="You are agent2",
  )

  runner = testing_utils.InMemoryRunner(
      SequentialAgent(name="pipeline", sub_agents=[agent1, agent2])
  )
  runner.run("First user message")
  runner.run("Second user message")

  # Second invocation: full history, but only user messages kept
  agent2_second_contents = testing_utils.simplify_contents(
      agent2_model.requests[1].contents
  )
  assert any("First user message" in str(c) for _, c in agent2_second_contents)
  assert any("Second user message" in str(c) for _, c in agent2_second_contents)
  assert not any("Agent1 result" in str(c) for _, c in agent2_second_contents)
  assert not any("For context:" in str(c) for _, c in agent2_second_contents)


# ---------------------------------------------------------------------------
# include_contents='current'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_contents_current_sees_user_and_all_upstream_agents():
  """include_contents='current' anchors at user message — all sibling agents visible."""
  agent1_model = testing_utils.MockModel.create(responses=["Agent1 result"])
  agent1 = LlmAgent(
      name="agent1", model=agent1_model, instruction="You are agent1"
  )

  agent2_model = testing_utils.MockModel.create(responses=["Agent2 result"])
  agent2 = LlmAgent(
      name="agent2", model=agent2_model, instruction="You are agent2"
  )

  agent3_model = testing_utils.MockModel.create(responses=["Agent3 done"])
  agent3 = LlmAgent(
      name="agent3",
      model=agent3_model,
      include_contents="current",
      instruction="You are agent3",
  )

  runner = testing_utils.InMemoryRunner(
      SequentialAgent(name="pipeline", sub_agents=[agent1, agent2, agent3])
  )
  runner.run("Original user request")

  agent3_contents = testing_utils.simplify_contents(
      agent3_model.requests[0].contents
  )

  # User message must be present
  assert any("Original user request" in str(c) for _, c in agent3_contents)
  # Both upstream agents' narrative entries must be present
  assert any("Agent1 result" in str(c) for _, c in agent3_contents)
  assert any("Agent2 result" in str(c) for _, c in agent3_contents)


@pytest.mark.asyncio
async def test_include_contents_current_with_source_filter_user_not_empty():
  """include_contents='current' + include_sources=['user'] → user message, not empty.

  Contrast with include_contents='none' + include_sources=['user'] which
  produces empty context when the last event is a peer agent's output.
  """
  agent1_model = testing_utils.MockModel.create(responses=["Agent1 result"])
  agent1 = LlmAgent(
      name="agent1", model=agent1_model, instruction="You are agent1"
  )

  agent2_model = testing_utils.MockModel.create(responses=["Agent2 done"])
  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_contents="current",
      include_sources=["user"],
      instruction="You are agent2",
  )

  runner = testing_utils.InMemoryRunner(
      SequentialAgent(name="pipeline", sub_agents=[agent1, agent2])
  )
  runner.run("Hello from user")

  agent2_contents = testing_utils.simplify_contents(
      agent2_model.requests[0].contents
  )

  # User message must be present and result must not be empty
  assert len(agent2_contents) > 0
  assert any("Hello from user" in str(c) for _, c in agent2_contents)
  # Upstream agent narrative must be filtered out
  assert not any("Agent1 result" in str(c) for _, c in agent2_contents)
  assert not any("For context:" in str(c) for _, c in agent2_contents)


@pytest.mark.asyncio
async def test_include_contents_current_with_source_filter_user_and_self():
  """include_contents='current' + include_sources=['user', 'self'] keeps user + own turns only."""
  agent1_model = testing_utils.MockModel.create(
      responses=["Agent1 result turn1", "Agent1 result turn2"]
  )
  agent1 = LlmAgent(
      name="agent1", model=agent1_model, instruction="You are agent1"
  )

  agent2_model = testing_utils.MockModel.create(
      responses=["Agent2 first turn", "Agent2 second turn"]
  )
  agent2 = LlmAgent(
      name="agent2",
      model=agent2_model,
      include_contents="current",
      include_sources=["user", "self"],
      instruction="You are agent2",
  )

  runner = testing_utils.InMemoryRunner(
      SequentialAgent(name="pipeline", sub_agents=[agent1, agent2])
  )
  runner.run("First user message")
  runner.run("Second user message")

  # Second invocation: current window starts at 'Second user message'.
  # Agent2's own prior turn from invocation 1 is outside that window — naturally absent.
  # What we verify: user present, upstream agent filtered by include_sources.
  agent2_second_contents = testing_utils.simplify_contents(
      agent2_model.requests[1].contents
  )
  assert any("Second user message" in str(c) for _, c in agent2_second_contents)
  assert not any("Agent1 result" in str(c) for _, c in agent2_second_contents)
  assert not any("For context:" in str(c) for _, c in agent2_second_contents)


def test_include_contents_none_with_include_sources_warns():
  """include_contents='none' + include_sources triggers a UserWarning."""
  import warnings as _warnings

  with _warnings.catch_warnings(record=True) as w:
    _warnings.simplefilter("always")
    LlmAgent(
        name="agent",
        model="gemini-2.5-flash",
        include_contents="none",
        include_sources=["user"],
    )
  assert len(w) == 1
  assert issubclass(w[0].category, UserWarning)
  assert "include_contents='current'" in str(w[0].message)


def test_include_contents_none_with_agent_name_in_sources_still_warns():
  """Warning fires even with a concrete agent name — still risky at runtime."""
  import warnings as _warnings

  with _warnings.catch_warnings(record=True) as w:
    _warnings.simplefilter("always")
    LlmAgent(
        name="agent",
        model="gemini-2.5-flash",
        include_contents="none",
        include_sources=["user", "upstream_agent"],
    )
  assert len(w) == 1
  assert issubclass(w[0].category, UserWarning)

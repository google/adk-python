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

"""Tests for tool call de-duplication (Issue #3940)."""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.genai import types
import pytest

from ... import testing_utils


def _function_call(name: str, args: dict) -> types.Part:
  return types.Part.from_function_call(name=name, args=args)


@pytest.mark.asyncio
async def test_dedupe_identical_tool_calls_across_steps():
  """Identical tool calls should execute once and reuse the cached result."""
  responses = [
      _function_call("test_tool", {"x": 1}),
      _function_call("test_tool", {"x": 1}),
      "done",
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> dict:
    nonlocal call_count
    call_count += 1
    return {"result": call_count}

  agent = Agent(name="root_agent", model=mock_model, tools=[test_tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  run_config = RunConfig(dedupe_tool_calls=True)
  events = []
  async for event in runner.runner.run_async(
      user_id=runner.session.user_id,
      session_id=runner.session.id,
      new_message=testing_utils.get_user_content("run"),
      run_config=run_config,
  ):
    events.append(event)
  simplified = testing_utils.simplify_events(events)

  # Tool should execute exactly once even though the model calls it twice.
  assert call_count == 1

  # Both tool responses should contain the same cached payload.
  tool_responses = [
      content
      for _, content in simplified
      if isinstance(content, types.Part) and content.function_response
  ]
  assert len(tool_responses) == 2
  assert tool_responses[0].function_response.response == {"result": 1}
  assert tool_responses[1].function_response.response == {"result": 1}


def test_dedupe_identical_tool_calls_within_one_step():
  """Identical tool calls within the same step should execute once."""
  responses = [
      [
          _function_call("test_tool", {"x": 1}),
          _function_call("test_tool", {"x": 1}),
      ],
      "done",
  ]
  mock_model = testing_utils.MockModel.create(responses=responses)

  call_count = 0

  def test_tool(x: int) -> dict:
    nonlocal call_count
    call_count += 1
    return {"result": call_count}

  agent = Agent(name="root_agent", model=mock_model, tools=[test_tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  run_config = RunConfig(dedupe_tool_calls=True)
  events = list(
      runner.runner.run(
          user_id=runner.session.user_id,
          session_id=runner.session.id,
          new_message=testing_utils.get_user_content("run"),
          run_config=run_config,
      )
  )
  simplified = testing_utils.simplify_events(events)

  assert call_count == 1

  # The merged tool response event contains 2 function_response parts.
  merged_parts = [
      content
      for _, content in simplified
      if isinstance(content, list)
      and all(isinstance(p, types.Part) for p in content)
      and any(p.function_response for p in content)
  ]
  assert len(merged_parts) == 1
  function_responses = [
      p.function_response.response
      for p in merged_parts[0]
      if p.function_response is not None
  ]
  assert function_responses == [{"result": 1}, {"result": 1}]


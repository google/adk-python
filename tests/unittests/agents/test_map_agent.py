import json
import re
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents import LoopAgent
from google.adk.agents import MapAgent
from google.adk.agents import ParallelAgent
from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ..testing_utils import MockModel
from ..testing_utils import ModelContent
from ..testing_utils import TestInMemoryRunner


class OneTwoThreeModel(MockModel):
  """Maps an input of 'i' to output of "['i', 'i+1', 'i+2']", e.g. '5' -> "['5', '6', '7']" """

  responses: list[LlmResponse] = []

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    agent_input: str | None = (
        (llm_request.contents[-1] or types.Content()).parts or [types.Part()]
    )[-1].text
    assert agent_input is not None
    agent_input = re.sub(r"\[\w+\] said: ", "", agent_input)
    assert agent_input.isnumeric()
    res = json.dumps([str(int(agent_input) + i) for i in range(3)])
    yield LlmResponse(content=ModelContent([types.Part(text=res)]))


def extract_event_text(events: list[Event], agent_prefix: str) -> list[str]:
  filtered_events = [e for e in events if e.author.startswith(agent_prefix)]
  sorted_events = sorted(
      filtered_events,
      key=lambda e: (
          e.author,
          ((e.content or types.Content()).parts or [types.Part()])[0].text
          or "",
      ),
  )
  contents = [e.content or types.Content() for e in sorted_events]
  return [(c.parts or [types.Part()])[0].text or "" for c in contents]


@pytest.mark.asyncio
async def test_map_agent_empty_input():
  def delete_events(callback_context: CallbackContext) -> None:
    callback_context._invocation_context.session.events.clear()

  map = MapAgent(
      name="map_agent",
      sub_agents=[
          LlmAgent(
              name="test", model=MockModel.create([], error=RuntimeError())
          )
      ],
      before_agent_callback=delete_events,
  )

  runner = TestInMemoryRunner(map)
  await runner.run_async_with_new_session("")


@pytest.mark.asyncio
async def test_map_agent_text_input():
  map = MapAgent(
      name="map_agent",
      sub_agents=[LlmAgent(name="mock_agent", model=OneTwoThreeModel())],
  )

  runner = TestInMemoryRunner(map)

  n_runs = 100

  input_data = json.dumps([str(i) for i in range(n_runs)])
  expected_output = [
      json.dumps([str(j) for j in range(i, i + 3)]) for i in range(n_runs)
  ]

  events = await runner.run_async_with_new_session(input_data)
  res = extract_event_text(events, "mock_agent")

  assert res == expected_output


@pytest.mark.asyncio
async def test_map_agent_with_loop_agent_parent():
  map_agent = MapAgent(
      name="map_agent",
      sub_agents=[LlmAgent(name="mock_agent", model=OneTwoThreeModel())],
  )

  loop_agent = LoopAgent(
      name="test_loop",
      sub_agents=[map_agent],
      max_iterations=2,
  )

  runner = TestInMemoryRunner(loop_agent)

  input_data = json.dumps(["0"])
  expected_output = [json.dumps(["0", "1", "2"])] + [
      json.dumps([str(j) for j in range(i, i + 3)]) for i in range(3)
  ]

  events = await runner.run_async_with_new_session(input_data)
  res = extract_event_text(events, "mock_agent")
  assert res == expected_output


@pytest.mark.parametrize("SubagentClass", [ParallelAgent, SequentialAgent])
@pytest.mark.asyncio
async def test_map_agent_with_sequential_or_parallel_agent(SubagentClass):
  """test map agent with a parallel / sequential sub-agent whose sub-agents don't communicate"""

  # A lone parallel agent wrapper hides mock_1's output from its 'cousin' mock_2
  mock1 = ParallelAgent(
      name="seq_1",
      sub_agents=[LlmAgent(name="mock_1", model=OneTwoThreeModel())],
  )
  mock2 = LlmAgent(name="mock_2", model=OneTwoThreeModel())

  subagent = SubagentClass(
      name="subagent",
      sub_agents=[mock1, mock2],
  )

  map = MapAgent(
      name="map_agent",
      sub_agents=[subagent],
  )

  runner = TestInMemoryRunner(map)

  input_data = json.dumps(["0", "1"])
  expected_output = [
      json.dumps([str(j) for j in range(i, i + 3)]) for i in [0, 1, 0, 1]
  ]

  events = await runner.run_async_with_new_session(input_data)
  res = extract_event_text(events, "mock_")
  assert res == expected_output


@pytest.mark.asyncio
async def test_map_agent_with_map_agent():
  mock_leaf = LlmAgent(name="nested_mock", model=OneTwoThreeModel())

  inner_map = MapAgent(
      name="inner_map",
      sub_agents=[mock_leaf],
  )

  outer_map = MapAgent(
      name="outer_map",
      sub_agents=[inner_map],
  )

  runner = TestInMemoryRunner(outer_map)

  input_data = json.dumps(
      [json.dumps([str(i), str(i + 1)]) for i in [10, 20, 30]]
  )
  expected_output = [
      json.dumps([str(j) for j in range(i, i + 3)])
      for i in [10, 11, 20, 21, 30, 31]
  ]

  events = await runner.run_async_with_new_session(input_data)

  res = extract_event_text(events, "nested_mock")
  assert res == expected_output


@pytest.mark.asyncio
async def test_map_agent_tree():
  inner_map = MapAgent(
      name="map_inner",
      sub_agents=[LlmAgent(name="mock_1", model=OneTwoThreeModel())],
  )

  main_loop = LoopAgent(
      name="main_sequential",
      sub_agents=[
          LlmAgent(name="mock_0", model=OneTwoThreeModel()),
          inner_map,
      ],
      max_iterations=1,
  )

  outer_map = MapAgent(
      name="map_outer",
      sub_agents=[main_loop],
  )

  runner = TestInMemoryRunner(outer_map)

  input_data = json.dumps(["0", "1"])
  expected_output = [
      json.dumps([str(j) for j in range(i, i + 3)])
      for i in [0, 1, 0, 1, 2, 1, 2, 3]
  ]

  events = await runner.run_async_with_new_session(input_data)
  res = extract_event_text(events, "mock_")

  assert res == expected_output

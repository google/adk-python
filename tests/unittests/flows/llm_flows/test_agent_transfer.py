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

from google.adk.agents.llm_agent import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.exit_loop_tool import exit_loop
from google.genai.types import Part

from ... import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name='transfer_to_agent', args={'agent_name': agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name='transfer_to_agent', response={'result': None}
)


def test_auto_to_auto():
  response = [
      transfer_call_part('sub_agent_1'),
      'response1',
      'response2',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (auto)
  sub_agent_1 = Agent(name='sub_agent_1', model=mockModel)
  root_agent = Agent(
      name='root_agent',
      model=mockModel,
      sub_agents=[sub_agent_1],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the transfer.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      ('sub_agent_1', 'response1'),
  ]

  # sub_agent_1 should still be the current agent.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('sub_agent_1', 'response2'),
  ]


def test_auto_to_single():
  response = [
      transfer_call_part('sub_agent_1'),
      'response1',
      'response2',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (single)
  sub_agent_1 = Agent(
      name='sub_agent_1',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  root_agent = Agent(
      name='root_agent', model=mockModel, sub_agents=[sub_agent_1]
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the responses.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      ('sub_agent_1', 'response1'),
  ]

  # root_agent should still be the current agent, because sub_agent_1 is single.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('root_agent', 'response2'),
  ]


def test_auto_to_auto_to_single():
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 transfers to sub_agent_1_1.
      transfer_call_part('sub_agent_1_1'),
      'response1',
      'response2',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (auto) - sub_agent_1_1 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = Agent(
      name='sub_agent_1', model=mockModel, sub_agents=[sub_agent_1_1]
  )
  root_agent = Agent(
      name='root_agent', model=mockModel, sub_agents=[sub_agent_1]
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the responses.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
      ('sub_agent_1', TRANSFER_RESPONSE_PART),
      ('sub_agent_1_1', 'response1'),
  ]

  # sub_agent_1 should still be the current agent. sub_agent_1_1 is single so it should
  # not be the current agent, otherwise the conversation will be tied to
  # sub_agent_1_1 forever.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('sub_agent_1', 'response2'),
  ]


def test_auto_to_sequential():
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 responds directly instead of transferring.
      'response1',
      'response2',
      'response3',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (sequential) - sub_agent_1_1 (single)
  #                                   \ sub_agent_1_2 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = SequentialAgent(
      name='sub_agent_1',
      sub_agents=[sub_agent_1_1, sub_agent_1_2],
  )
  root_agent = Agent(
      name='root_agent',
      model=mockModel,
      sub_agents=[sub_agent_1],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the transfer.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      ('sub_agent_1_1', 'response1'),
      ('sub_agent_1_2', 'response2'),
  ]

  # root_agent should still be the current agent because sub_agent_1 is sequential.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('root_agent', 'response3'),
  ]


def test_auto_to_sequential_to_auto():
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 responds directly instead of transferring.
      'response1',
      transfer_call_part('sub_agent_1_2_1'),
      'response2',
      'response3',
      'response4',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (seq) - sub_agent_1_1 (single)
  #                            \ sub_agent_1_2 (auto) - sub_agent_1_2_1 (auto)
  #                            \ sub_agent_1_3 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2_1 = Agent(name='sub_agent_1_2_1', model=mockModel)
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mockModel,
      sub_agents=[sub_agent_1_2_1],
  )
  sub_agent_1_3 = Agent(
      name='sub_agent_1_3',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = SequentialAgent(
      name='sub_agent_1',
      sub_agents=[sub_agent_1_1, sub_agent_1_2, sub_agent_1_3],
  )
  root_agent = Agent(
      name='root_agent',
      model=mockModel,
      sub_agents=[sub_agent_1],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the transfer.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      ('sub_agent_1_1', 'response1'),
      ('sub_agent_1_2', transfer_call_part('sub_agent_1_2_1')),
      ('sub_agent_1_2', TRANSFER_RESPONSE_PART),
      ('sub_agent_1_2_1', 'response2'),
      ('sub_agent_1_3', 'response3'),
  ]

  # root_agent should still be the current agent because sub_agent_1 is sequential.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('root_agent', 'response4'),
  ]


def test_auto_to_loop():
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 responds directly instead of transferring.
      'response1',
      'response2',
      'response3',
      Part.from_function_call(name='exit_loop', args={}),
      'response4',
      'response5',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (loop) - sub_agent_1_1 (single)
  #                             \ sub_agent_1_2 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mockModel,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
      tools=[exit_loop],
  )
  sub_agent_1 = LoopAgent(
      name='sub_agent_1',
      sub_agents=[sub_agent_1_1, sub_agent_1_2],
  )
  root_agent = Agent(
      name='root_agent',
      model=mockModel,
      sub_agents=[sub_agent_1],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  # Asserts the transfer.
  assert testing_utils.simplify_events(runner.run('test1')) == [
      # Transfers to sub_agent_1.
      ('root_agent', transfer_call_part('sub_agent_1')),
      ('root_agent', TRANSFER_RESPONSE_PART),
      # Loops.
      ('sub_agent_1_1', 'response1'),
      ('sub_agent_1_2', 'response2'),
      ('sub_agent_1_1', 'response3'),
      # Exits.
      ('sub_agent_1_2', Part.from_function_call(name='exit_loop', args={})),
      (
          'sub_agent_1_2',
          Part.from_function_response(
              name='exit_loop', response={'result': None}
          ),
      ),
  ]

  # root_agent should still be the current agent because sub_agent_1 is loop.
  assert testing_utils.simplify_events(runner.run('test2')) == [
      ('root_agent', 'response4'),
  ]

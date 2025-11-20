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
from google.adk.agents.loop_agent import LoopAgentState
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.tools.exit_loop_tool import exit_loop
from google.genai.types import Part
import pytest

from ... import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name='transfer_to_agent', args={'agent_name': agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name='transfer_to_agent', response={'result': None}
)

END_OF_AGENT = testing_utils.END_OF_AGENT


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_auto(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      'response1',
      'response2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (auto)
  sub_agent_1 = Agent(name='sub_agent_1', model=mock_model)
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1],
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
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
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', 'response1'),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('sub_agent_1', 'response2'),
        ('sub_agent_1', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_single(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      'response1',
      'response2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (single)
  sub_agent_1 = Agent(
      name='sub_agent_1',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  root_agent = Agent(
      name='root_agent', model=mock_model, sub_agents=[sub_agent_1]
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
    # Asserts the responses.
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', 'response1'),
    ]

    # root_agent should still be the current agent, because sub_agent_1 is
    # single.
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', 'response1'),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response2'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_auto_to_single(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 transfers to sub_agent_1_1.
      transfer_call_part('sub_agent_1_1'),
      'response1',
      'response2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (auto) - sub_agent_1_1 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = Agent(
      name='sub_agent_1', model=mock_model, sub_agents=[sub_agent_1_1]
  )
  root_agent = Agent(
      name='root_agent', model=mock_model, sub_agents=[sub_agent_1]
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
    # Asserts the responses.
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1_1', 'response1'),
    ]

    # sub_agent_1 should still be the current agent. sub_agent_1_1 is single so
    # it should not be the current agent; otherwise, the conversation will be
    # tied to sub_agent_1_1 forever.
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('sub_agent_1', 'response2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_1', END_OF_AGENT),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('sub_agent_1', 'response2'),
        ('sub_agent_1', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_sequential(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 responds directly instead of transferring.
      'response1',
      'response2',
      'response3',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (sequential) - sub_agent_1_1 (single)
  #                                   \ sub_agent_1_2 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = SequentialAgent(
      name='sub_agent_1',
      sub_agents=[sub_agent_1_1, sub_agent_1_2],
  )
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1],
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
    # Asserts the transfer.
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_2', 'response2'),
    ]

    # root_agent should still be the current agent because sub_agent_1 is
    # sequential.
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response3'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        (
            'sub_agent_1',
            SequentialAgentState(current_sub_agent='sub_agent_1_1').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_1', END_OF_AGENT),
        (
            'sub_agent_1',
            SequentialAgentState(current_sub_agent='sub_agent_1_2').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_2', 'response2'),
        ('sub_agent_1_2', END_OF_AGENT),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response3'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_sequential_to_auto(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      # sub_agent_1 responds directly instead of transferring.
      'response1',
      transfer_call_part('sub_agent_1_2_1'),
      'response2',
      'response3',
      'response4',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (seq) - sub_agent_1_1 (single)
  #                            \ sub_agent_1_2 (auto) - sub_agent_1_2_1 (auto)
  #                            \ sub_agent_1_3 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2_1 = Agent(name='sub_agent_1_2_1', model=mock_model)
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mock_model,
      sub_agents=[sub_agent_1_2_1],
  )
  sub_agent_1_3 = Agent(
      name='sub_agent_1_3',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1 = SequentialAgent(
      name='sub_agent_1',
      sub_agents=[sub_agent_1_1, sub_agent_1_2, sub_agent_1_3],
  )
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1],
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
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

    # root_agent should still be the current agent because sub_agent_1 is
    # sequential.
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response4'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        (
            'sub_agent_1',
            SequentialAgentState(current_sub_agent='sub_agent_1_1').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_1', END_OF_AGENT),
        (
            'sub_agent_1',
            SequentialAgentState(current_sub_agent='sub_agent_1_2').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_2', transfer_call_part('sub_agent_1_2_1')),
        ('sub_agent_1_2', TRANSFER_RESPONSE_PART),
        ('sub_agent_1_2_1', 'response2'),
        ('sub_agent_1_2_1', END_OF_AGENT),
        ('sub_agent_1_2', END_OF_AGENT),
        (
            'sub_agent_1',
            SequentialAgentState(current_sub_agent='sub_agent_1_3').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_3', 'response3'),
        ('sub_agent_1_3', END_OF_AGENT),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response4'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_loop(is_resumable: bool):
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
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (loop) - sub_agent_1_1 (single)
  #                             \ sub_agent_1_2 (single)
  sub_agent_1_1 = Agent(
      name='sub_agent_1_1',
      model=mock_model,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )
  sub_agent_1_2 = Agent(
      name='sub_agent_1_2',
      model=mock_model,
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
      model=mock_model,
      sub_agents=[sub_agent_1],
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
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
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        # Transfers to sub_agent_1.
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        # Loops.
        (
            'sub_agent_1',
            LoopAgentState(current_sub_agent='sub_agent_1_1').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_1', END_OF_AGENT),
        (
            'sub_agent_1',
            LoopAgentState(current_sub_agent='sub_agent_1_2').model_dump(
                mode='json'
            ),
        ),
        ('sub_agent_1_2', 'response2'),
        ('sub_agent_1_2', END_OF_AGENT),
        (
            'sub_agent_1',
            LoopAgentState(
                current_sub_agent='sub_agent_1_1', times_looped=1
            ).model_dump(mode='json'),
        ),
        ('sub_agent_1_1', 'response3'),
        ('sub_agent_1_1', END_OF_AGENT),
        (
            'sub_agent_1',
            LoopAgentState(
                current_sub_agent='sub_agent_1_2', times_looped=1
            ).model_dump(mode='json'),
        ),
        # Exits.
        ('sub_agent_1_2', Part.from_function_call(name='exit_loop', args={})),
        (
            'sub_agent_1_2',
            Part.from_function_response(
                name='exit_loop', response={'result': None}
            ),
        ),
        ('sub_agent_1_2', END_OF_AGENT),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', END_OF_AGENT),
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response4'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_auto_to_auto_to_auto_forms_transfer_loop(is_resumable: bool):
  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_2'),
      transfer_call_part('root_agent'),
      'response from root',
      'response 2 from root',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)
  # root (auto) - sub_agent_1 (auto) - sub_agent_2 (auto) - root (auto)
  sub_agent_1 = Agent(name='sub_agent_1', model=mock_model)
  sub_agent_2 = Agent(name='sub_agent_2', model=mock_model)
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1, sub_agent_2],
  )
  app = App(
      name='test_app',
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=is_resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  if not is_resumable:
    # Asserts the transfer.
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_2', transfer_call_part('root_agent')),
        ('sub_agent_2', TRANSFER_RESPONSE_PART),
        ('root_agent', 'response from root'),
    ]

    # root_agent should be the current agent.
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response 2 from root'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_2', transfer_call_part('root_agent')),
        ('sub_agent_2', TRANSFER_RESPONSE_PART),
        ('root_agent', 'response from root'),
        (
            'root_agent',
            END_OF_AGENT,
        ),  # First time root_agent marked as ended.
        ('sub_agent_2', END_OF_AGENT),
        ('sub_agent_1', END_OF_AGENT),
        (
            'root_agent',
            END_OF_AGENT,
        ),  # Second time root_agent marked as ended.
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response 2 from root'),
        ('root_agent', END_OF_AGENT),
    ]

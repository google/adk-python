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

"""Integration tests for multi-agent dynamic transfers in ADK 2.0."""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.loop_agent import LoopAgentState
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.sequential_agent import SequentialAgentState
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.exit_loop_tool import exit_loop
from google.genai.types import Part
import pytest

from tests.unittests import testing_utils


def transfer_call_part(agent_name: str) -> Part:
  return Part.from_function_call(
      name='transfer_to_agent', args={'agent_name': agent_name}
  )


TRANSFER_RESPONSE_PART = Part.from_function_response(
    name='transfer_to_agent', response={'result': None}
)

END_OF_AGENT = testing_utils.END_OF_AGENT


@pytest.mark.parametrize('is_resumable', [True, False])
def test_transfer_parent_to_child(is_resumable: bool):
  """Verify direct Parent -> Child dynamic transfer and conversational resumption."""
  # Arrange
  response = [
      transfer_call_part('sub_agent_1'),
      'response1',
      'response2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

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

  # Act & Assert: Turn 1
  if not is_resumable:
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', 'response1'),
    ]

    # Turn 2: Conversation continues at sub_agent_1
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('sub_agent_1', 'response2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', 'response1'),
        ('sub_agent_1', END_OF_AGENT),
    ]

    # Turn 2: Resumed session continues at sub_agent_1
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
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', 'response1'),
        ('sub_agent_1', END_OF_AGENT),
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
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', END_OF_AGENT),
        ('sub_agent_1_1', 'response1'),
        ('sub_agent_1_1', END_OF_AGENT),
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
        ('root_agent', END_OF_AGENT),
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
        ('root_agent', END_OF_AGENT),
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
        ('root_agent', END_OF_AGENT),
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
    ]
    # Same session, different invocation.
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response4'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_transfer_child_to_sibling(is_resumable: bool):
  """Verify Child A -> Sibling B peer dynamic transfer and conversational resumption."""
  # Arrange
  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_2'),
      'response1',
      'response2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

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

  # Act & Assert: Turn 1
  if not is_resumable:
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_2', 'response1'),
    ]

    # Turn 2: Conversation continues at sibling sub_agent_2
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('sub_agent_2', 'response2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', END_OF_AGENT),
        ('sub_agent_2', 'response1'),
        ('sub_agent_2', END_OF_AGENT),
    ]

    # Turn 2: Resumed session continues at sub_agent_2
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('sub_agent_2', 'response2'),
        ('sub_agent_2', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_transfer_child_to_parent(is_resumable: bool):
  """Verify Child -> Parent dynamic climbing transfer and conversational resumption."""
  # Arrange
  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('root_agent'),
      'response_root',
      'response_root_2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

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

  # Act & Assert: Turn 1
  if not is_resumable:
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('root_agent')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('root_agent', 'response_root'),
    ]

    # Turn 2: Conversation continues back at root_agent coordinator
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response_root_2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', transfer_call_part('root_agent')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', END_OF_AGENT),
        ('root_agent', 'response_root'),
        ('root_agent', END_OF_AGENT),
    ]

    # Turn 2: Resumed session continues at root_agent
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response_root_2'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.parametrize('is_resumable', [True, False])
def test_transfer_child_to_grandchild(is_resumable: bool):
  """Verify deep 3-layer Child -> Grandchild nested dynamic transfers."""
  # Arrange
  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_1_1'),
      'response_grandchild',
      'response_grandchild_2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  sub_agent_1_1 = Agent(name='sub_agent_1_1', model=mock_model)
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

  # Act & Assert: Turn 1
  if not is_resumable:
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1_1', 'response_grandchild'),
    ]

    # Turn 2: Conversation continues at grandchild
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('sub_agent_1_1', 'response_grandchild_2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', transfer_call_part('sub_agent_1_1')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', END_OF_AGENT),
        ('sub_agent_1_1', 'response_grandchild'),
        ('sub_agent_1_1', END_OF_AGENT),
    ]

    # Turn 2: Resumed session continues at sub_agent_1_1
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('sub_agent_1_1', 'response_grandchild_2'),
        ('sub_agent_1_1', END_OF_AGENT),
    ]


@pytest.mark.asyncio
async def test_transfer_to_self_raises_error():
  """Verify that an agent trying to transfer to itself raises a ValueError."""
  # Arrange

  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_1'),  # Transfer to self
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  sub_agent_1 = Agent(name='sub_agent_1', model=mock_model)
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1],
  )

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test_app', user_id='test_user', session_id='test_session'
  )
  runner = Runner(
      app_name='test_app',
      agent=root_agent,
      session_service=session_service,
  )

  msg = testing_utils.types.Content(
      role='user', parts=[testing_utils.types.Part(text='start')]
  )

  # Act & Assert
  with pytest.raises(ValueError, match='cannot transfer to itself'):
    async for _ in runner.run_async(
        user_id='test_user',
        session_id='test_session',
        new_message=msg,
    ):
      pass


@pytest.mark.asyncio
async def test_transfer_to_unrelated_agent_raises_error():
  """Verify that an agent transferring to an unrelated agent raises a ValueError."""
  # Arrange

  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_1_1'),
      transfer_call_part(
          'sub_agent_2'
      ),  # Structurally unrelated to sub_agent_1_1!
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

  sub_agent_1_1 = Agent(name='sub_agent_1_1', model=mock_model)
  sub_agent_1 = Agent(
      name='sub_agent_1',
      model=mock_model,
      sub_agents=[sub_agent_1_1],
  )
  sub_agent_2 = Agent(name='sub_agent_2', model=mock_model)
  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      sub_agents=[sub_agent_1, sub_agent_2],
  )

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test_app', user_id='test_user', session_id='test_session'
  )
  runner = Runner(
      app_name='test_app',
      agent=root_agent,
      session_service=session_service,
  )

  msg = testing_utils.types.Content(
      role='user', parts=[testing_utils.types.Part(text='start')]
  )

  # Act & Assert
  with pytest.raises(
      ValueError,
      match=(
          "Cannot transfer from 'sub_agent_1_1' to unrelated agent"
          " 'sub_agent_2'"
      ),
  ):
    async for _ in runner.run_async(
        user_id='test_user',
        session_id='test_session',
        new_message=msg,
    ):
      pass


@pytest.mark.parametrize('is_resumable', [True, False])
def test_transfer_cyclic_loop(is_resumable: bool):
  """Verify multi-stage cyclic loop transfer (Root -> SubA -> SubB -> Root) and resumption."""
  # Arrange
  response = [
      transfer_call_part('sub_agent_1'),
      transfer_call_part('sub_agent_2'),
      transfer_call_part('root_agent'),
      'response_from_root',
      'response_from_root_2',
  ]
  mock_model = testing_utils.MockModel.create(responses=response)

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

  # Act & Assert: Turn 1
  if not is_resumable:
    assert testing_utils.simplify_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_2', transfer_call_part('root_agent')),
        ('sub_agent_2', TRANSFER_RESPONSE_PART),
        ('root_agent', 'response_from_root'),
    ]

    # Turn 2: Conversation continues at root coordinator
    assert testing_utils.simplify_events(runner.run('test2')) == [
        ('root_agent', 'response_from_root_2'),
    ]
  else:
    assert testing_utils.simplify_resumable_app_events(runner.run('test1')) == [
        ('root_agent', transfer_call_part('sub_agent_1')),
        ('root_agent', TRANSFER_RESPONSE_PART),
        ('root_agent', END_OF_AGENT),
        ('sub_agent_1', transfer_call_part('sub_agent_2')),
        ('sub_agent_1', TRANSFER_RESPONSE_PART),
        ('sub_agent_1', END_OF_AGENT),
        ('sub_agent_2', transfer_call_part('root_agent')),
        ('sub_agent_2', TRANSFER_RESPONSE_PART),
        ('sub_agent_2', END_OF_AGENT),
        ('root_agent', 'response_from_root'),
        ('root_agent', END_OF_AGENT),
    ]

    # Turn 2: Resumed session continues at root_agent
    assert testing_utils.simplify_resumable_app_events(runner.run('test2')) == [
        ('root_agent', 'response_from_root_2'),
        ('root_agent', END_OF_AGENT),
    ]


@pytest.mark.asyncio
async def test_three_level_nested_dynamic_node_transfer():
  """Verify parent relationship climbing in 3-level deep nested dynamic nodes.

  Setup:
    - root_agent with sub_agents=[mid_agent].
    - mid_agent with sub_agents=[leaf_agent, target_agent].
  Act:
    - Run root_agent. Model responses trigger transfers:
      Root -> Mid -> Leaf -> Target.
  Assert:
    - Events verify the full transfer chain and target response.
  """
  # Arrange
  target_agent = Agent(name='target_agent')
  leaf_agent = Agent(name='leaf_agent')
  mid_agent = Agent(name='mid_agent', sub_agents=[leaf_agent, target_agent])
  root_agent = Agent(name='root_agent', sub_agents=[mid_agent])

  root_agent.model = testing_utils.MockModel.create(
      responses=[transfer_call_part('mid_agent')]
  )
  mid_agent.model = testing_utils.MockModel.create(
      responses=[transfer_call_part('leaf_agent')]
  )
  leaf_agent.model = testing_utils.MockModel.create(
      responses=[transfer_call_part('target_agent')]
  )
  target_agent.model = testing_utils.MockModel.create(
      responses=['hello from target']
  )

  app = App(name='test_app', root_agent=root_agent)
  runner = testing_utils.InMemoryRunner(app=app)

  # Act
  events = runner.run('go')
  simple_events = testing_utils.simplify_events(events)

  # Assert
  assert ('root_agent', transfer_call_part('mid_agent')) in simple_events
  assert ('mid_agent', transfer_call_part('leaf_agent')) in simple_events
  assert ('leaf_agent', transfer_call_part('target_agent')) in simple_events
  assert ('target_agent', 'hello from target') in simple_events


@pytest.mark.asyncio
async def test_agent_transfer_hitl_resume_rehydration():
  """Verify B's rehydration after transfer A -> B -> LRO confirmation on turn 2.

  Setup:
    - agent_a with sub_agents=[agent_b].
    - agent_b with a LongRunningFunctionTool.
  Act:
    - Turn 1: Run agent_a. Model transfers to agent_b, which calls LRO.
    - Turn 2: Resume with LRO function response.
  Assert:
    - Turn 1: Yields LRO interrupt.
    - Turn 2: agent_b successfully rehydrates and completes.
  """
  from google.adk.tools.long_running_tool import LongRunningFunctionTool

  # Arrange
  def confirm_tool() -> None:
    return None

  lro_tool = LongRunningFunctionTool(confirm_tool)
  agent_b = Agent(name='agent_b', tools=[lro_tool])
  agent_a = Agent(name='agent_a', sub_agents=[agent_b], tools=[lro_tool])

  agent_a.model = testing_utils.MockModel.create(
      responses=[transfer_call_part('agent_b')]
  )

  LRO_ID = 'adk-test-lro-123'
  agent_b.model = testing_utils.MockModel.create(
      responses=[
          testing_utils.types.Part(
              function_call=testing_utils.types.FunctionCall(
                  name='confirm_tool', args={}, id=LRO_ID
              )
          ),
          'B task finished',
      ]
  )

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test_app', user_id='test_user', session_id='test_session'
  )
  runner = Runner(
      app_name='test_app',
      agent=agent_a,
      session_service=session_service,
  )

  # Act: Turn 1
  events1 = [
      e
      async for e in runner.run_async(
          user_id='test_user',
          session_id='test_session',
          new_message=testing_utils.types.Content(
              role='user', parts=[testing_utils.types.Part(text='start task')]
          ),
      )
  ]

  # Assert: Turn 1
  assert any(e.long_running_tool_ids for e in events1)

  # Arrange: Turn 2
  lro_id = None
  for e in events1:
    if e.content and e.content.parts:
      for p in e.content.parts:
        if (
            p.function_call
            and p.function_call.name == 'confirm_tool'
            and p.function_call.id
        ):
          lro_id = p.function_call.id
          break
      if lro_id:
        break

  if not lro_id:
    for e in events1:
      if e.long_running_tool_ids:
        lro_id = list(e.long_running_tool_ids)[0]
        break

  assert lro_id is not None
  invocation_id = events1[0].invocation_id

  confirm_response = testing_utils.types.Content(
      role='user',
      parts=[
          testing_utils.types.Part(
              function_response=testing_utils.types.FunctionResponse(
                  id=lro_id,
                  name='confirm_tool',
                  response={'result': 'done'},
              )
          )
      ],
  )

  # Act: Turn 2
  events2 = [
      e
      async for e in runner.run_async(
          user_id='test_user',
          session_id='test_session',
          invocation_id=invocation_id,
          new_message=confirm_response,
      )
  ]

  # Assert: Turn 2
  simplified2 = testing_utils.simplify_resumable_app_events(events2)
  assert ('agent_b', 'B task finished') in simplified2


@pytest.mark.asyncio
async def test_llm_agent_transfer_inside_custom_node():
  """Verify transfer from an LlmAgent called inside a custom dynamic node.

  Setup:
    - root_agent with sub_agents=[inner_agent, target_agent] to establish static sibling relation.
    - A custom @node that calls inner_agent via ctx.run_node.
    - A Workflow that executes the custom node.
  Act:
    - Run the workflow. inner_agent is executed and triggers transfer to target_agent.
  Assert:
    - The transfer is correctly resolved as SIBLING and target_agent executes.
  """
  from google.adk.workflow import node
  from google.adk.workflow import START
  from google.adk.workflow import Workflow

  # Arrange
  target_agent = Agent(name='target_agent')
  inner_agent = Agent(name='inner_agent')
  # Establish static hierarchy for transfer resolution
  root_agent = Agent(name='root_agent', sub_agents=[inner_agent, target_agent])

  inner_agent.model = testing_utils.MockModel.create(
      responses=[transfer_call_part('target_agent')]
  )
  target_agent.model = testing_utils.MockModel.create(
      responses=['hello from target']
  )

  @node(rerun_on_resume=True)
  async def custom_node(*, ctx, node_input):
    # Call llm agent inside custom node
    result = await ctx.run_node(inner_agent, node_input='go')
    yield f'custom: {result}'

  wf = Workflow(name='wf', edges=[(START, custom_node)])

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test_app', user_id='test_user', session_id='test_session'
  )
  runner = Runner(
      app_name='test_app',
      node=wf,
      session_service=session_service,
  )

  # Act
  events = [
      e
      async for e in runner.run_async(
          user_id='test_user',
          session_id='test_session',
          new_message=testing_utils.types.Content(
              role='user', parts=[testing_utils.types.Part(text='start')]
          ),
      )
  ]

  simplified = testing_utils.simplify_events(events)

  # Assert
  assert ('inner_agent', transfer_call_part('target_agent')) in simplified
  assert ('target_agent', 'hello from target') in simplified

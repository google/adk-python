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

"""Tests for AntigravityAgent.

Verifies the root-only construction constraint that keeps the agent usable only
as a standalone root agent while the SDK supports local mode only, and the node
plumbing ``_run_impl`` adds on top of ``BaseAgent``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.labs.antigravity import _antigravity_agent
from google.adk.labs.antigravity._antigravity_agent import AntigravityAgent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._node_runner import NodeRunner
from google.antigravity import LocalAgentConfig
from google.antigravity import types as sdk_types
from google.genai import types as genai_types
import pytest


def _make_config(**kwargs) -> LocalAgentConfig:
  """Returns a minimal real LocalAgentConfig for the wrapped SDK agent."""
  return LocalAgentConfig(system_instructions='test', **kwargs)


async def _invocation_context(agent, user_text='the original message'):
  """Builds a REAL InvocationContext rooted at `agent`."""
  session_service = InMemorySessionService()
  return InvocationContext(
      session_service=session_service,
      invocation_id='inv_1',
      agent=agent,
      session=await session_service.create_session(
          app_name='test_app', user_id='test_user'
      ),
      user_content=genai_types.Content(
          role='user', parts=[genai_types.Part.from_text(text=user_text)]
      ),
      run_config=RunConfig(),
  )


async def _node_ctx(*, agent, user_text='the original message'):
  """A mock node Context wrapping a REAL InvocationContext.

  Args:
    agent: The agent the invocation is rooted at.
    user_text: The original end-user message, i.e. what a dropped node_input
      would silently fall back to.

  Returns:
    A MagicMock node Context whose get_invocation_context() is real.
  """
  ctx = MagicMock()
  ctx.get_invocation_context.return_value = await _invocation_context(
      agent, user_text=user_text
  )
  ctx.node_path = 'root/agy'
  return ctx


async def _run_via_node_runner(agent, node_input):
  """Runs `agent` through a real NodeRunner.

  This is the path _SingleTurnAgentTool takes, so it exercises the event
  enrichment and output tracking a bare _run_impl call cannot see.

  Args:
    agent: The agent to run as the node.
    node_input: The parent's composed request.

  Returns:
    (child_ctx, enqueued_events). The events are post-enrichment, i.e. exactly
    what NodeRunner would append to the session.
  """
  inner = await _invocation_context(agent)
  enqueued = []

  async def _enqueue(event):
    enqueued.append(event)

  # No Runner drains the queue here, so the real _enqueue_event would raise.
  object.__setattr__(inner, '_enqueue_event', AsyncMock(side_effect=_enqueue))

  parent_ctx = Context(invocation_context=inner, node_path='')
  child_ctx = await NodeRunner(node=agent, parent_ctx=parent_ctx).run(
      node_input
  )
  return child_ctx, enqueued


def _event(author='agy', partial=False, parts=None):
  """Builds an ADK Event; `parts=None` means the event carries no content."""
  return Event(
      invocation_id='inv_1',
      author=author,
      partial=partial,
      content=(
          None
          if parts is None
          else genai_types.Content(role='model', parts=parts)
      ),
  )


_TEXT_PART = genai_types.Part.from_text(text='answer')
_THOUGHT_PART = genai_types.Part(text='thinking out loud', thought=True)
_CALL_PART = genai_types.Part(
    function_call=genai_types.FunctionCall(name='run_command', args={})
)
_RESPONSE_PART = genai_types.Part(
    function_response=genai_types.FunctionResponse(
        name='run_command', response={'result': 'ok'}
    )
)


@pytest.mark.parametrize(
    'event,expected',
    [
        pytest.param(_event(parts=[_TEXT_PART]), 'answer', id='text'),
        pytest.param(
            _event(parts=[_TEXT_PART, _TEXT_PART]),
            'answeranswer',
            id='text_parts_concatenated',
        ),
        pytest.param(
            _event(parts=[_THOUGHT_PART, _TEXT_PART]),
            'answer',
            id='thought_dropped_text_kept',
        ),
        pytest.param(
            _event(partial=True, parts=[_TEXT_PART]), None, id='partial'
        ),
        pytest.param(_event(parts=[_THOUGHT_PART]), None, id='thought_only'),
        pytest.param(_event(parts=[_CALL_PART]), None, id='function_call_only'),
        pytest.param(
            _event(author='run_command', parts=[_RESPONSE_PART]),
            None,
            id='function_response_from_tool',
        ),
        pytest.param(
            _event(author='some_other_agent', parts=[_TEXT_PART]),
            None,
            id='wrong_author',
        ),
        pytest.param(_event(parts=[]), None, id='empty_parts'),
        pytest.param(_event(parts=None), None, id='no_content'),
    ],
)
def test_final_model_text_filters(event, expected):
  """Only this agent's own, complete, user-visible text becomes node output.

  Notably `partial`: in SSE mode a trajectory can end on a streaming chunk,
  which would otherwise surface as a truncated answer.
  """
  assert _antigravity_agent._final_model_text(event, 'agy') == expected


def test_standalone_agent_is_allowed():
  """An AntigravityAgent with no parent and no sub-agents constructs cleanly."""
  agent = AntigravityAgent(name='agy', config=_make_config())

  assert agent.parent_agent is None
  assert agent.sub_agents == []


def test_giving_sub_agents_is_rejected():
  """Constructing with sub-agents raises a temporary root-only error."""
  child = BaseAgent(name='child')

  with pytest.raises(ValueError, match='standalone root agent'):
    AntigravityAgent(name='agy', config=_make_config(), sub_agents=[child])


def test_using_as_sub_agent_is_rejected():
  """Adopting the agent under a parent raises a temporary root-only error."""
  agy = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='standalone root agent'):
    BaseAgent(name='parent', sub_agents=[agy])


@pytest.mark.asyncio
async def test_run_without_save_dir_raises():
  """Running without config.save_dir raises, since trajectories need a folder."""
  agent = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='requires config.save_dir'):
    async for _ in agent._run_async_impl(MagicMock()):
      pass


def _text_step(step_index: int, text: str):
  """Builds a stub SDK Step carrying one complete model text response.

  Args:
    step_index: The harness step index, which drives resume skipping.
    text: The model text the step carries.

  Returns:
    A step that converts to a single complete text event authored by the agent.
  """
  step = MagicMock()
  step.step_index = step_index
  step.source = sdk_types.StepSource.MODEL
  step.type = sdk_types.StepType.TEXT_RESPONSE
  step.status = sdk_types.StepStatus.DONE
  step.is_complete_response = True
  step.content = text
  step.tool_calls = []
  return step


def _fake_active_agent(receive_steps, conversation_id='conv-1'):
  """Builds a stand-in for the SDK ``Agent`` that `_run_async_impl` enters.

  Args:
    receive_steps: A zero-arg async generator function yielding the steps of the
      simulated trajectory.
    conversation_id: The id the harness reports back. Only matters when the test
      cares about trajectory file naming.

  Returns:
    A MagicMock usable as an async context manager, whose
    ``conversation.send`` is an AsyncMock the test can assert against.
  """
  conversation = MagicMock()
  conversation.send = AsyncMock()
  conversation.receive_steps = receive_steps
  active_agent = MagicMock()
  active_agent.conversation = conversation
  active_agent.conversation_id = conversation_id
  active_agent.__aenter__ = AsyncMock(return_value=active_agent)
  active_agent.__aexit__ = AsyncMock(return_value=None)
  return active_agent


@pytest.mark.asyncio
async def test_resumed_replayed_steps_are_skipped(tmp_path):
  """On resume, steps at or below the resume index are not re-emitted."""

  # The harness replays steps 0-1 (prior turn) then emits step 2 (this turn).
  async def _receive_steps():
    yield _text_step(0, 'old-1')
    yield _text_step(1, 'old-2')
    yield _text_step(2, 'new')

  conversation_id = _antigravity_agent._derive_conversation_id(
      'sess_456', 'agy'
  )
  active_agent = _fake_active_agent(
      _receive_steps, conversation_id=conversation_id
  )

  # A prior trajectory + resume index in save_dir triggers resume at index 1.
  save_dir = tmp_path
  (save_dir / f'traj-{conversation_id}').write_bytes(b'data')
  (save_dir / f'traj-{conversation_id}.resume').write_text('1')
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(save_dir))
  )

  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  ctx.session.id = 'sess_456'
  ctx.user_content = None
  ctx.run_config = None

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [event async for event in agent._run_async_impl(ctx)]

  texts = [e.content.parts[0].text for e in events]
  assert texts == ['new']


@pytest.mark.asyncio
async def test_node_input_becomes_the_prompt(tmp_path):
  """The parent's composed request wins over the original user message.

  Without the _run_impl override the SDK silently receives ctx.user_content:
  a plausible-looking wrong prompt rather than an exception.
  """

  async def _receive_steps():
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(tmp_path))
  )
  ctx = await _node_ctx(
      user_text='hi, can you help me with bug 42?', agent=agent
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    async for _ in agent._run_impl(ctx=ctx, node_input='Fix bug 42.'):
      pass

  active_agent.conversation.send.assert_awaited_once_with('Fix bug 42.')


@pytest.mark.asyncio
async def test_last_complete_response_becomes_node_output(tmp_path):
  """Output is the final model text, not the first.

  A trajectory emits one complete response per model turn, so promoting the
  first would return the model's opening remark.
  """

  async def _receive_steps():
    yield _text_step(0, 'Let me look at the file.')
    yield _text_step(1, 'Done: patch sent for review.')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(tmp_path))
  )
  ctx = await _node_ctx(agent=agent)

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [e async for e in agent._run_impl(ctx=ctx, node_input='go')]

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Done: patch sent for review.']


def _tool_response_step(step_index: int, name: str):
  """Builds a real SDK Step for a completed tool execution.

  The converter authors the resulting event with the tool name.

  Args:
    step_index: The harness step index.
    name: The tool name, which becomes the event author.

  Returns:
    An SDK Step that converts to a single function-response event.
  """
  return sdk_types.Step(
      step_index=step_index,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.SYSTEM,
      status=sdk_types.StepStatus.DONE,
      content='ok',
      tool_calls=[sdk_types.ToolCall(name=name, args={}, id=f'c{step_index}')],
  )


@pytest.mark.asyncio
async def test_output_reaches_the_parent_through_node_runner(tmp_path):
  """End-to-end: the parent reads the answer off ctx.output, correctly authored.

  The run ends on a tool step so that NodeRunner's author enrichment, which
  would otherwise attribute the output event to 'run_command', is exercised.
  """

  async def _receive_steps():
    yield _text_step(0, 'Done: patch sent for review.')
    yield _tool_response_step(1, 'run_command')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(tmp_path))
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    child_ctx, enqueued = await _run_via_node_runner(agent, 'go')

  assert child_ctx.output == 'Done: patch sent for review.'
  output_events = [e for e in enqueued if e.output is not None]
  assert [e.author for e in output_events] == ['agy']


@pytest.mark.asyncio
async def test_text_less_run_outputs_empty_string_not_none(tmp_path):
  """A completed run with no model text must not hand the parent None.

  Reachable when a trajectory ends on tool calls with no closing summary;
  None would put `{"result": null}` in front of the parent's model.
  """

  async def _receive_steps():
    yield _tool_response_step(0, 'run_command')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(tmp_path))
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    child_ctx, _ = await _run_via_node_runner(agent, 'go')

  assert child_ctx.output == ''


@pytest.mark.asyncio
async def test_node_input_none_is_a_no_op(tmp_path):
  """A classic agent-tree run still reads ctx.user_content."""

  async def _receive_steps():
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(save_dir=str(tmp_path))
  )
  ctx = await _node_ctx(user_text='the original message', agent=agent)

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    async for _ in agent._run_impl(ctx=ctx, node_input=None):
      pass

  active_agent.conversation.send.assert_awaited_once_with(
      'the original message'
  )

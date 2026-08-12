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
from pydantic import ValidationError
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
  """Constructing with sub-agents raises, naming the sub_agents guard.

  The match string is specific to that guard: matching text shared with the
  parent guard would pass on the wrong error.
  """
  child = BaseAgent(name='child')

  with pytest.raises(ValueError, match='cannot be given sub_agents'):
    AntigravityAgent(name='agy', config=_make_config(), sub_agents=[child])


def test_using_as_sub_agent_is_rejected():
  """Adopting the agent under a parent without mode='single_turn' raises."""
  agy = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='may only be a sub-agent'):
    BaseAgent(name='parent', sub_agents=[agy])


def test_single_turn_agent_can_be_a_sub_agent():
  """mode='single_turn' lifts the root-only restriction on adoption.

  The parent composes an isolated request, so no ADK session history reaches
  the harness and its conversation does not outlive the call.
  """
  agy = AntigravityAgent(name='agy', config=_make_config(), mode='single_turn')

  parent = BaseAgent(name='parent', sub_agents=[agy])

  assert agy.parent_agent is parent


def test_single_turn_agent_still_cannot_have_sub_agents():
  """Children stay blocked in every mode: the SDK runs its own agent loop.

  Unlike adoption, this restriction is independent of how the agent is
  invoked -- the harness would never dispatch to an ADK child either way.
  """
  child = BaseAgent(name='child')

  with pytest.raises(ValueError, match='cannot be given sub_agents'):
    AntigravityAgent(
        name='agy',
        config=_make_config(),
        mode='single_turn',
        sub_agents=[child],
    )


def test_single_turn_agent_is_wrapped_as_a_parent_tool():
  """LlmAgent wraps a non-LlmAgent sub-agent that declares mode='single_turn'.

  The wrapping in LlmAgent.model_post_init is duck-typed on `mode`, so if it
  breaks, every other test here still passes.
  """
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.tools.agent_tool import _SingleTurnAgentTool

  coder = AntigravityAgent(
      name='antigravity_coder',
      description='Writes code.',
      config=_make_config(),
      mode='single_turn',
  )

  parent = LlmAgent(
      name='triager', model='gemini-2.5-flash', sub_agents=[coder]
  )

  assert any(
      isinstance(t, _SingleTurnAgentTool) and t.agent is coder
      for t in parent.tools
  )


def test_single_turn_agent_is_not_a_transfer_target():
  """The parent must never hand the conversation over by LLM transfer.

  Being called as an inline tool is the whole safety argument for allowing a
  parent. The exclusion is duck-typed on `mode` in
  flows/llm_flows/agent_transfer.py, which this file knows nothing about.
  """
  from google.adk.agents.llm_agent import LlmAgent
  from google.adk.flows.llm_flows.agent_transfer import _get_transfer_targets

  coder = AntigravityAgent(
      name='antigravity_coder',
      description='Writes code.',
      config=_make_config(),
      mode='single_turn',
  )

  parent = LlmAgent(
      name='triager', model='gemini-2.5-flash', sub_agents=[coder]
  )

  assert coder not in _get_transfer_targets(parent)


def test_mode_cannot_be_reassigned_after_construction():
  """`mode` is frozen: the adoption guard only gets to run once.

  Clearing `mode` after adoption would leave the agent adopted while
  _run_async_impl went back to session-keyed resumption.
  """
  from google.adk.agents.llm_agent import LlmAgent

  agy = AntigravityAgent(name='agy', config=_make_config(), mode='single_turn')
  parent = LlmAgent(name='triager', model='gemini-2.5-flash', sub_agents=[agy])

  with pytest.raises(ValidationError, match='frozen'):
    agy.mode = None

  assert agy.mode == 'single_turn'
  assert agy.parent_agent is parent


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


def _mock_run_ctx(session_id='sess_456'):
  """A minimal InvocationContext stand-in for _run_async_impl.

  Args:
    session_id: The ADK session id the conversation id is derived from.

  Returns:
    A MagicMock usable as the ctx argument to _run_async_impl.
  """
  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  ctx.session.id = session_id
  ctx.user_content = None
  ctx.run_config = None
  return ctx


@pytest.mark.asyncio
async def test_resumed_replayed_steps_are_skipped(tmp_path):
  """On resume, steps at or below the resume index are not re-emitted.

  Also pins the new resume index being persisted: without that write the next
  turn would replay everything this turn emitted.
  """

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

  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [event async for event in agent._run_async_impl(ctx)]

  texts = [e.content.parts[0].text for e in events]
  assert texts == ['new']
  # Step 2 was the highest index emitted, so the next turn resumes from it.
  assert (save_dir / f'traj-{conversation_id}.resume').read_text() == '2'


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
      name='agy',
      config=_make_config(save_dir=str(tmp_path)),
      mode='single_turn',
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
      name='agy',
      config=_make_config(save_dir=str(tmp_path)),
      mode='single_turn',
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
      name='agy',
      config=_make_config(save_dir=str(tmp_path)),
      mode='single_turn',
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
      name='agy',
      config=_make_config(save_dir=str(tmp_path)),
      mode='single_turn',
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    child_ctx, _ = await _run_via_node_runner(agent, 'go')

  assert child_ctx.output == ''


def test_chat_mode_is_rejected():
  """Only 'single_turn' is accepted; the Literal is deliberately narrow.

  AntigravityAgent is not an LlmAgent, so LlmAgent's other modes ('chat',
  'task') have no meaning here.
  """
  with pytest.raises(ValidationError, match='single_turn'):
    AntigravityAgent(name='agy', config=_make_config(), mode='chat')


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


@pytest.mark.asyncio
async def test_single_turn_calls_do_not_resume_or_persist(tmp_path):
  """Single-turn calls are isolated and leave no trajectory behind.

  The planted trajectory is what an earlier call in the same ADK session would
  have left; picking it up would make call two silently resume call one.
  """

  async def _receive_steps():
    yield _text_step(0, 'first')
    yield _text_step(1, 'second')

  conversation_id = _antigravity_agent._derive_conversation_id(
      'sess_456', 'agy'
  )
  # A harness id distinct from the derived one: if they matched,
  # rename_trajectory would early-return and a stray rename be invisible.
  active_agent = _fake_active_agent(
      _receive_steps, conversation_id='harness-random'
  )

  # What an earlier single-turn call in this same ADK session would have left.
  (tmp_path / f'traj-{conversation_id}').write_bytes(b'data')
  (tmp_path / f'traj-{conversation_id}.resume').write_text('0')
  # What this call's harness would have written under its own random id.
  (tmp_path / 'traj-harness-random').write_bytes(b'harness')
  agent = AntigravityAgent(
      name='agy',
      config=_make_config(save_dir=str(tmp_path)),
      mode='single_turn',
  )

  ctx = _mock_run_ctx()

  handed_configs = []

  def _capture_config(config):
    handed_configs.append(config)
    return active_agent

  with patch.object(_antigravity_agent, 'Agent', _capture_config):
    events = [event async for event in agent._run_async_impl(ctx)]

  # The harness was handed no id, so it cannot replay the planted trajectory.
  assert [c.conversation_id for c in handed_configs] == [None]
  # Nothing was skipped as an already-emitted replay.
  assert [e.content.parts[0].text for e in events] == ['first', 'second']
  # The earlier call's resume index was left exactly as it was found.
  assert (tmp_path / f'traj-{conversation_id}.resume').read_text() == '0'
  # save_dir as a whole is untouched: no rename onto the derived id, and no
  # bookkeeping file added.
  assert {p.name for p in tmp_path.iterdir()} == {
      f'traj-{conversation_id}',
      f'traj-{conversation_id}.resume',
      'traj-harness-random',
  }


@pytest.mark.asyncio
async def test_single_turn_run_without_save_dir_is_allowed():
  """save_dir is only needed to resume, and single-turn never resumes."""

  async def _receive_steps():
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )

  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert [e.content.parts[0].text for e in events] == ['done']

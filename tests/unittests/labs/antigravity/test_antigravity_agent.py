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

# The tests drive the agent's protected seams directly: `_run_async_impl` and
# the `_*_cls` injection hooks.
# pylint: disable=protected-access
# Test names carry the intent; a docstring is added only where one cannot.
# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Callable
from typing import Literal
from typing import Sequence
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.labs.antigravity import _antigravity_agent
from google.adk.labs.antigravity import _tool_result_capture
from google.adk.labs.antigravity._antigravity_agent import AntigravityAgent
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._node_runner import NodeRunner
from google.antigravity import AgentConfig
from google.antigravity import LocalAgentConfig
from google.antigravity import LocalOpenAIAgentConfig
from google.antigravity import types as sdk_types
from google.antigravity.connections.local.local_connection_config import BaseLocalAgentConfig
from google.antigravity.hooks import hooks as sdk_hooks
from google.genai import types as genai_types
from pydantic import JsonValue
from pydantic import ValidationError
import pytest


def _make_config(**kwargs: object) -> LocalAgentConfig:
  """A minimal real LocalAgentConfig for the wrapped Antigravity agent."""
  return LocalAgentConfig(system_instructions='test', **kwargs)


class _StubChild(BaseAgent):
  """A runnable ADK child agent."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(invocation_id=ctx.invocation_id, author=self.name)


def _user_tool(query: str) -> str:
  return query


async def _invocation_context(
    agent: BaseAgent, user_text: str = 'the original message'
) -> InvocationContext:
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


async def _node_ctx(
    *, agent: BaseAgent, user_text: str = 'the original message'
) -> MagicMock:
  """A mock node Context whose get_invocation_context() is real."""
  ctx = MagicMock()
  ctx.get_invocation_context.return_value = await _invocation_context(
      agent, user_text=user_text
  )
  ctx.node_path = 'root/agy'
  return ctx


async def _run_via_node_runner(
    agent: BaseAgent, node_input: object
) -> tuple[Context, list[Event]]:
  """Runs `agent` as _SingleTurnAgentTool does: through a real NodeRunner."""
  inner = await _invocation_context(agent)
  enqueued: list[Event] = []

  async def _enqueue(event: Event) -> None:
    enqueued.append(event)

  # No Runner drains the queue here, so the real _enqueue_event would raise.
  object.__setattr__(inner, '_enqueue_event', AsyncMock(side_effect=_enqueue))

  parent_ctx = Context(invocation_context=inner, node_path='')
  child_ctx = await NodeRunner(node=agent, parent_ctx=parent_ctx).run(
      node_input
  )
  # Returned events are post-enrichment, which a bare _run_impl cannot see.
  return child_ctx, enqueued


def test_standalone_agent_is_allowed():
  agent = AntigravityAgent(name='agy', config=_make_config())

  assert agent.parent_agent is None
  assert agent.sub_agents == []


def test_sub_agents_are_allowed():
  child = _StubChild(name='reviewer', description='Reviews a diff.')

  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )

  assert agent.sub_agents == [child]


def test_a_sub_agent_without_a_description_is_rejected():
  child = _StubChild(name='reviewer')

  with pytest.raises(ValueError, match='description'):
    AntigravityAgent(name='coder', config=_make_config(), sub_agents=[child])


def test_two_sub_agents_with_the_same_name_are_rejected():
  # `BaseAgent.validate_sub_agents_unique_names` only warns, and the SDK's
  # later "Tool 'reviewer' is already registered." names no agent.
  first = _StubChild(name='reviewer', description='Reviews a diff.')
  second = _StubChild(name='reviewer', description='Reviews a design.')

  with pytest.raises(ValueError, match='share the name'):
    AntigravityAgent(
        name='coder', config=_make_config(), sub_agents=[first, second]
    )


def test_a_sub_agent_named_like_a_config_tool_is_rejected():
  # The child joins the same `config.tools` as `_user_tool` (whose name is
  # `_user_tool`), so the two would collide in the harness's one-tool-per-name
  # namespace.
  child = _StubChild(name='_user_tool', description='Shadows a tool.')

  with pytest.raises(ValueError, match='collides with a tool'):
    AntigravityAgent(
        name='coder',
        config=_make_config(tools=[_user_tool]),
        sub_agents=[child],
    )


def test_using_as_sub_agent_is_rejected():
  agy = AntigravityAgent(name='agy', config=_make_config())

  with pytest.raises(ValueError, match='may only be an ADK sub-agent'):
    BaseAgent(name='parent', sub_agents=[agy])


def test_single_turn_agent_can_be_a_sub_agent():
  agy = AntigravityAgent(name='agy', config=_make_config(), mode='single_turn')

  parent = BaseAgent(name='parent', sub_agents=[agy])

  assert agy.parent_agent is parent


def test_no_sub_agents_leaves_the_config_tools_alone():
  config = _make_config(tools=[_user_tool])
  agent = AntigravityAgent(name='agy', config=config)

  built = agent._build_sdk_config()

  # The tool object itself stays identical: `AgentConfig.model_copy`
  # re-shallow-copies `tools` after the deep copy.
  assert built is not config
  assert built.tools is not config.tools
  assert built.tools == [_user_tool]
  assert config.tools == [_user_tool]


def test_single_turn_agent_is_wrapped_as_a_parent_tool():
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

  # The wrapping is duck-typed on `mode` in LlmAgent.model_post_init.
  assert any(
      isinstance(t, _SingleTurnAgentTool) and t.agent is coder
      for t in parent.tools
  )


def test_single_turn_agent_is_not_a_transfer_target():
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

  # The exclusion is duck-typed on `mode` in flows/llm_flows/agent_transfer.py.
  assert coder not in _get_transfer_targets(parent)


def test_mode_cannot_be_reassigned_after_construction():
  """`mode` is frozen: the adoption guard only gets to run once."""
  from google.adk.agents.llm_agent import LlmAgent

  agy = AntigravityAgent(name='agy', config=_make_config(), mode='single_turn')
  parent = LlmAgent(name='triager', model='gemini-2.5-flash', sub_agents=[agy])

  with pytest.raises(ValidationError, match='frozen'):
    agy.mode = None

  assert agy.mode == 'single_turn'
  assert agy.parent_agent is parent


def _text_step(step_index: int, text: str) -> MagicMock:
  """A stub Antigravity SDK Step with one complete model text response."""
  step = MagicMock()
  step.step_index = step_index
  step.source = sdk_types.StepSource.MODEL
  step.type = sdk_types.StepType.TEXT_RESPONSE
  step.status = sdk_types.StepStatus.DONE
  step.is_complete_response = True
  step.content = text
  step.tool_calls = []
  return step


async def _steps_once() -> AsyncIterator[MagicMock]:
  """Yields the single step of a minimal, complete trajectory."""
  yield _text_step(0, 'done')


_StepStream = Callable[[], AsyncIterator[sdk_types.Step | MagicMock]]


def _fake_active_agent(
    receive_steps: _StepStream,
    conversation_id: str = '',
    history: Sequence[sdk_types.Step | MagicMock] = (),
) -> MagicMock:
  """A stand-in for the Antigravity SDK ``Agent`` `_run_async_impl` enters."""
  conversation = MagicMock()
  conversation.send = AsyncMock()
  conversation.receive_steps = receive_steps
  conversation.history = list(history)
  active_agent = MagicMock()
  active_agent.conversation = conversation
  # Explicit because a MagicMock would auto-create a truthy non-string here.
  active_agent.conversation_id = conversation_id
  active_agent.__aenter__ = AsyncMock(return_value=active_agent)
  active_agent.__aexit__ = AsyncMock(return_value=None)
  return active_agent


def _mock_run_ctx(
    session_id: str = 'sess_456', state: dict[str, str | None] | None = None
) -> MagicMock:
  """A minimal InvocationContext stand-in for _run_async_impl."""
  ctx = MagicMock()
  ctx.invocation_id = 'inv_1'
  ctx.branch = 'main'
  ctx.app_name = 'test_app'
  ctx.user_id = 'test_user'
  ctx.session.id = session_id
  # A real dict, not the auto-created MagicMock: `MagicMock().get(k)` returns a
  # truthy MagicMock, which would make every turn look like a resume.
  ctx.session.state = {} if state is None else state
  ctx.user_content = None
  ctx.run_config = None
  return ctx


def test_sdk_agent_cls_defaults_to_the_sdk_agent():
  agent = AntigravityAgent(name='agy', config=_make_config())

  assert agent._sdk_agent_cls is _antigravity_agent.Agent


@pytest.mark.asyncio
async def test_subclass_overrides_the_sdk_agent_class():
  """Guards the seam against being inlined back to the module global."""

  async def _receive_steps() -> AsyncIterator[MagicMock]:
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)

  def _refuse(config: AgentConfig) -> MagicMock:
    raise AssertionError('the module global Agent was used')

  class _Swapped(AntigravityAgent):

    @property
    def _sdk_agent_cls(self) -> Callable[[AgentConfig], MagicMock]:
      return lambda config: active_agent

  agent = _Swapped(name='agy', config=_make_config(), mode='single_turn')

  with patch.object(_antigravity_agent, 'Agent', _refuse):
    events = [event async for event in agent._run_async_impl(_mock_run_ctx())]

  assert [event.content.parts[0].text for event in events] == ['done']


@pytest.mark.asyncio
async def test_each_turn_of_one_session_builds_a_new_sdk_agent():
  built: list[MagicMock] = []

  def _build(config: AgentConfig) -> MagicMock:
    del config
    active_agent = _fake_active_agent(_steps_once)
    built.append(active_agent)
    return active_agent

  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(ctx):
      pass
    async for _ in agent._run_async_impl(ctx):
      pass

  assert len(built) == 2
  assert [a.conversation.send.await_count for a in built] == [1, 1]
  assert [a.__aexit__.await_count for a in built] == [1, 1]


@pytest.mark.asyncio
async def test_single_turn_builds_a_new_sdk_agent_per_call():
  built: list[MagicMock] = []

  def _build(config: AgentConfig) -> MagicMock:
    del config
    active_agent = _fake_active_agent(_steps_once)
    built.append(active_agent)
    return active_agent

  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(ctx):
      pass
    async for _ in agent._run_async_impl(ctx):
      pass

  assert len(built) == 2
  assert [a.conversation.send.await_count for a in built] == [1, 1]
  assert [a.__aexit__.await_count for a in built] == [1, 1]


_CID = 'a' * 36
_OTHER_CID = 'b' * 36


def _deltas(events: Sequence[Event]) -> list[dict[str, str | None]]:
  """The state deltas carried by these events, in order."""
  return [e.actions.state_delta for e in events if e.actions.state_delta]


def _capture_builder(
    active_agent: MagicMock,
) -> tuple[Callable[[AgentConfig], MagicMock], list[AgentConfig]]:
  """A fake `Agent` factory recording the config it was handed."""
  configs: list[AgentConfig] = []

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return active_agent

  return _build, configs


@pytest.mark.asyncio
async def test_turn_one_records_the_runtime_assigned_id():
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert _deltas(events) == [{'_antigravity_conversation_id_agy': _CID}]


@pytest.mark.asyncio
async def test_turn_two_passes_the_stored_id_back():
  """And asks for CREATE_OR_RESUME, the only mode that survives a lost store."""
  active_agent = _fake_active_agent(
      _steps_once, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  build, configs = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={'_antigravity_conversation_id_agy': _CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    async for _ in agent._run_async_impl(ctx):
      pass

  assert configs[0].conversation_id == _CID
  assert (
      configs[0].session_continuation_mode
      == sdk_types.SessionContinuationMode.CREATE_OR_RESUME
  )


@pytest.mark.asyncio
async def test_an_unchanged_id_is_not_rewritten():
  active_agent = _fake_active_agent(
      _steps_once, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={'_antigravity_conversation_id_agy': _CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert not _deltas(events)


@pytest.mark.asyncio
async def test_a_turn_with_no_steps_records_nothing():

  async def _no_steps() -> AsyncIterator[MagicMock]:
    return
    yield  # pragma: no cover - makes this an async generator

  active_agent = _fake_active_agent(_no_steps, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  # Its history is legitimately empty, so recording the id would make the next
  # turn's resume check report a spurious silent drop.
  assert not _deltas(events)


@pytest.mark.asyncio
async def test_a_content_less_turn_with_history_still_records_the_id():
  # Mirrors external report PR #6765 (cl/966049341): a compaction turn's steps
  # carry no user-visible content, so the turn yields zero ADK events, yet the
  # conversation has history and an id and must still be recorded -- otherwise
  # the next turn starts fresh and orphans this conversation.

  async def _no_steps() -> AsyncIterator[MagicMock]:
    return
    yield  # pragma: no cover - makes this an async generator

  active_agent = _fake_active_agent(
      _no_steps, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert _deltas(events) == [{_STATE_KEY: _CID}]


@pytest.mark.asyncio
async def test_a_content_less_turn_without_history_records_nothing():
  # The other side of the guard: zero events AND no history is a genuinely
  # empty turn, so recording the id would make the next resume look dropped.

  async def _no_steps() -> AsyncIterator[MagicMock]:
    return
    yield  # pragma: no cover - makes this an async generator

  active_agent = _fake_active_agent(_no_steps, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert not _deltas(events)


@pytest.mark.asyncio
async def test_an_id_published_after_the_first_step_is_still_recorded():
  active_agent = _fake_active_agent(_steps_once, conversation_id='')

  # ``event_processor.py:547`` records the trajectory id only when it is
  # non-empty, so a first step can leave ``conversation_id`` blank.
  async def _receive_steps() -> AsyncIterator[MagicMock]:
    yield _text_step(0, 'thinking')
    # The runtime publishes the id only now, one step late.
    active_agent.conversation_id = _CID
    yield _text_step(1, 'done')

  active_agent.conversation.receive_steps = _receive_steps
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert _deltas(events) == [{'_antigravity_conversation_id_agy': _CID}]


@pytest.mark.asyncio
async def test_single_turn_neither_reads_nor_writes_the_id():
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, configs = _capture_builder(active_agent)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )
  ctx = _mock_run_ctx(state={'_antigravity_conversation_id_agy': _OTHER_CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    events = [event async for event in agent._run_async_impl(ctx)]

  assert configs[0].conversation_id is None
  assert not _deltas(events)


@pytest.mark.asyncio
async def test_two_agents_in_one_session_use_separate_keys():
  """Otherwise the second agent resumes the first agent's conversation."""
  ctx = _mock_run_ctx()
  keys = []

  for name, cid in (('first', _CID), ('second', _OTHER_CID)):
    active_agent = _fake_active_agent(_steps_once, conversation_id=cid)
    build, _ = _capture_builder(active_agent)
    agent = AntigravityAgent(name=name, config=_make_config())
    with patch.object(_antigravity_agent, 'Agent', build):
      events = [event async for event in agent._run_async_impl(ctx)]
    for delta in _deltas(events):
      keys.extend(delta)

  assert keys == [
      '_antigravity_conversation_id_first',
      '_antigravity_conversation_id_second',
  ]


def test_a_local_config_without_save_dir_warns(caplog):
  """The one case that loses history with no error and no log of its own."""
  with caplog.at_level(logging.WARNING, logger='google_adk'):
    AntigravityAgent(name='agy', config=_make_config())

  assert 'save_dir' in caplog.text


def test_a_local_config_with_save_dir_is_silent(caplog, tmp_path):
  with caplog.at_level(logging.WARNING, logger='google_adk'):
    AntigravityAgent(name='agy', config=_make_config(save_dir=str(tmp_path)))

  assert 'save_dir' not in caplog.text


def test_single_turn_without_save_dir_is_silent(caplog):
  with caplog.at_level(logging.WARNING, logger='google_adk'):
    AntigravityAgent(name='agy', config=_make_config(), mode='single_turn')

  assert 'save_dir' not in caplog.text


def test_a_non_local_config_is_silent(caplog):

  class _Remote(AntigravityAgent):

    @property
    def _local_config_cls(self) -> type[BaseAgent]:
      return _StubChild  # nothing the config could ever be an instance of

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    _Remote(name='agy', config=_make_config())

  assert 'save_dir' not in caplog.text


def test_local_config_cls_defaults_to_the_sdk_local_config_base():
  agent = AntigravityAgent(name='agy', config=_make_config())

  # The base, not ``LocalAgentConfig``: ``_get_or_create_save_dir`` is defined
  # on ``BaseLocalAgentConfig``, so every subclass has the same failure mode.
  assert agent._local_config_cls is BaseLocalAgentConfig


def test_another_local_config_subclass_also_warns(caplog):
  # A real sibling subclass: a fake one would pass even against a seam still
  # narrowed to ``LocalAgentConfig``.
  config = LocalOpenAIAgentConfig(model='llama3')
  assert config.save_dir is None
  assert isinstance(config, BaseLocalAgentConfig)
  assert not isinstance(config, LocalAgentConfig)

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    AntigravityAgent(name='agy', config=config)

  assert 'save_dir' in caplog.text


_STATE_KEY = '_antigravity_conversation_id_agy'


async def _drain(
    agen: AsyncIterator[Event],
) -> tuple[list[Event], BaseException | None]:
  """Collects events until the generator raises, returning both."""
  events: list[Event] = []
  try:
    async for event in agen:
      events.append(event)
  except BaseException as exc:  # pylint: disable=broad-except
    return events, exc
  return events, None


class _ConnectRefusedError(Exception):
  """A distinct connect failure, so the re-raise can be pinned by identity."""


@pytest.mark.asyncio
async def test_connect_failure_after_resume_reraises_and_keeps_the_id():
  # A connect failure keeps the stored id: CREATE_OR_RESUME will resume-or-
  # recreate the conversation on the next turn, so a transient connect error
  # must not orphan a conversation that is probably still alive.
  # A distinct type pins that this very object reaches the caller unwrapped.
  failure = _ConnectRefusedError('conversation not found')

  def _refuse(config: AgentConfig) -> MagicMock:
    del config
    raise failure

  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  with patch.object(_antigravity_agent, 'Agent', _refuse):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert exc is failure
  # No clear/state-delta event: the stored id is left intact.
  assert not _deltas(events)
  assert not events


def _hanging_connect_agent() -> tuple[MagicMock, asyncio.Event]:
  """An `Agent` stand-in whose ``__aenter__`` never returns."""
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)

  async def _hang() -> None:
    # Set once the connect has been reached, i.e. once it is safe to cancel.
    entered.set()
    await asyncio.Event().wait()  # never set

  entered = asyncio.Event()
  active_agent.__aenter__ = AsyncMock(side_effect=_hang)
  return active_agent, entered


async def _cancel_during_connect(
    agent: AntigravityAgent,
    ctx: MagicMock,
    active_agent: MagicMock,
    entered: asyncio.Event,
) -> tuple[asyncio.Task[None], list[Event]]:
  """Runs one turn, cancels it while it is suspended inside the connect."""
  build, _ = _capture_builder(active_agent)
  events: list[Event] = []

  async def _drive() -> None:
    async for event in agent._run_async_impl(ctx):
      events.append(event)

  with patch.object(_antigravity_agent, 'Agent', build):
    task = asyncio.create_task(_drive())
    await asyncio.wait_for(entered.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  return task, events


@pytest.mark.asyncio
async def test_cancelling_a_connect_keeps_the_stored_id():
  """A cancelled connect says the caller left, not that the conversation did."""
  active_agent, entered = _hanging_connect_agent()
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  task, events = await _cancel_during_connect(agent, ctx, active_agent, entered)

  # A connect failure never clears the stored id, so no state delta is emitted
  # and the cancellation propagates unchanged.
  assert not _deltas(events)
  assert task.cancelled()


@pytest.mark.asyncio
async def test_a_cancelled_connect_still_closes_the_sdk_agent():
  active_agent, entered = _hanging_connect_agent()
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  await _cancel_during_connect(agent, ctx, active_agent, entered)

  # The Antigravity SDK's ``__aenter__`` cleans up under ``except Exception``
  # (``agent.py:136-139``), so a ``CancelledError`` escapes it uncleaned.
  assert active_agent.__aexit__.await_count == 1


@pytest.mark.asyncio
async def test_abandoning_the_generator_mid_stream_does_not_error():
  """Pins the ``try: <await> / except: yield`` shape when a consumer leaves."""
  # ``aclose`` throws ``GeneratorExit`` in at whichever ``yield`` the consumer
  # stopped on; a ``try`` widened to cover a ``yield`` would answer it with
  # another ``yield``: ``RuntimeError: async generator ignored GeneratorExit``.
  # The consumer abandons the generator mid-turn.
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  active_agent = _fake_active_agent(
      _steps_once, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  _build, _ = _capture_builder(active_agent)

  seen = []
  with patch.object(_antigravity_agent, 'Agent', _build):
    async with contextlib.aclosing(agent._run_async_impl(ctx)) as agen:
      async for event in agen:
        seen.append(event)
        break

  assert len(seen) == 1


class _HarnessError(Exception):
  """A distinct harness failure, so the re-raise can be pinned by identity."""


@pytest.mark.asyncio
async def test_normal_completion_inside_an_outer_except_reports_no_error():
  # A turn that completes normally must close the SDK agent with no exception,
  # even while an unrelated exception is being handled higher in the same task.
  # A bare ``finally: __aexit__(*sys.exc_info())`` reported that outer exception
  # to the SDK; ``async with`` reports the true unwinding instead.
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    try:
      raise KeyError('unrelated')
    except KeyError:
      async for _ in agent._run_async_impl(ctx):
        pass

  active_agent.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_resume_index_survives_a_harness_error_mid_turn():
  # Mirrors external report PR #6765 case 2: the harness dies mid-turn, once a
  # conversation with history and an id exists but before any event recorded it.
  # The block after the loop is skipped on the error path, so the id would be
  # lost and the next turn would orphan the conversation without this.
  failure = _HarnessError('harness died')
  active_agent = _fake_active_agent(
      _steps_once, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  # The harness raises before the turn yields its first event.
  active_agent.conversation.send = AsyncMock(side_effect=failure)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert exc is failure
  assert _deltas(events) == [{_STATE_KEY: _CID}]
  # The SDK agent is still closed on the error path.
  assert active_agent.__aexit__.await_count == 1


@pytest.mark.asyncio
async def test_connect_failure_without_a_stored_id_is_untouched():

  def _refuse(config: AgentConfig) -> MagicMock:
    del config
    raise RuntimeError('boom')

  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', _refuse):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert isinstance(exc, RuntimeError)
  assert not _deltas(events)


@pytest.mark.asyncio
async def test_a_local_resume_with_no_history_fails_the_turn():
  """Local forgets silently; an empty history is the only signal there is."""
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert isinstance(exc, RuntimeError)
  assert _deltas(events) == [{_STATE_KEY: None}]
  # The turn is abandoned, not half-run.
  assert active_agent.conversation.send.await_count == 0
  assert active_agent.__aexit__.await_count == 1


@pytest.mark.asyncio
async def test_a_local_resume_with_history_proceeds():
  active_agent = _fake_active_agent(
      _steps_once, conversation_id=_CID, history=[_text_step(0, 'earlier')]
  )
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert exc is None
  assert [e.content.parts[0].text for e in events if e.content] == ['done']


@pytest.mark.asyncio
async def test_a_non_local_resume_with_no_history_proceeds():
  # Only a local connection populates ``Conversation.history``; applying the
  # check elsewhere would fail every resumed turn those connections run.
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)

  class _Remote(AntigravityAgent):

    @property
    def _local_config_cls(self) -> type[BaseAgent]:
      return _StubChild  # nothing the config could ever be an instance of

  agent = _Remote(name='agy', config=_make_config())
  ctx = _mock_run_ctx(state={_STATE_KEY: _CID})

  with patch.object(_antigravity_agent, 'Agent', build):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert exc is None
  assert [e.content.parts[0].text for e in events if e.content] == ['done']


@pytest.mark.asyncio
async def test_a_fresh_local_conversation_with_no_history_proceeds():
  # Checking the history without also checking that a resume was asked for
  # would fail every first turn.
  active_agent = _fake_active_agent(_steps_once, conversation_id=_CID)
  build, _ = _capture_builder(active_agent)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', build):
    events, exc = await _drain(agent._run_async_impl(ctx))

  assert exc is None
  assert _deltas(events) == [{_STATE_KEY: _CID}]


@pytest.mark.asyncio
async def test_sub_agents_reach_the_sdk_config_as_tools():
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  configs: list[AgentConfig] = []

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return _fake_active_agent(_steps_once)

  agent = AntigravityAgent(
      name='coder', config=_make_config(tools=[_user_tool]), sub_agents=[child]
  )

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(_mock_run_ctx()):
      pass

  # Ordering is deliberate: the caller's own tools keep their positions and
  # identities, children are appended after them.
  assert [tool.__name__ for tool in configs[0].tools] == [
      '_user_tool',
      'reviewer',
  ]
  assert configs[0].tools[0] is _user_tool
  assert configs[0].tools[1].__doc__ == 'Reviews a diff.'


def test_a_child_appended_after_construction_is_still_checked():
  # `sub_agents` is a plain list field with no `validate_assignment`, so
  # `append` never reaches `model_post_init`; the guard has to re-run later.
  agent = AntigravityAgent(name='coder', config=_make_config())

  agent.sub_agents.append(_StubChild(name='reviewer'))

  with pytest.raises(ValueError, match='description'):
    agent._build_sdk_config()


def test_a_duplicate_appended_after_construction_is_still_checked():
  first = _StubChild(name='reviewer', description='Reviews a diff.')
  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[first]
  )

  agent.sub_agents.append(_StubChild(name='reviewer', description='Also.'))

  with pytest.raises(ValueError, match='share the name'):
    agent._build_sdk_config()


@pytest.mark.asyncio
async def test_a_single_turn_agent_bridges_its_children_too():
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  configs: list[AgentConfig] = []

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return _fake_active_agent(_steps_once)

  agent = AntigravityAgent(
      name='coder',
      config=_make_config(),
      mode='single_turn',
      sub_agents=[child],
  )
  ctx = _mock_run_ctx()

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(ctx):
      pass
    async for _ in agent._run_async_impl(ctx):
      pass

  assert len(configs) == 2
  assert [[tool.__name__ for tool in c.tools] for c in configs] == [
      ['reviewer'],
      ['reviewer'],
  ]


@pytest.mark.asyncio
async def test_save_dir_is_no_longer_required():
  config = _make_config()
  assert config.save_dir is None
  agent = AntigravityAgent(name='agy', config=config)

  with patch.object(
      _antigravity_agent, 'Agent', lambda cfg: _fake_active_agent(_steps_once)
  ):
    events = [event async for event in agent._run_async_impl(_mock_run_ctx())]

  assert [e.content.parts[0].text for e in events] == ['done']


@pytest.mark.asyncio
async def test_node_input_becomes_the_prompt():

  async def _receive_steps() -> AsyncIterator[MagicMock]:
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )
  ctx = await _node_ctx(
      user_text='hi, can you help me with bug 42?', agent=agent
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    async for _ in agent._run_impl(ctx=ctx, node_input='Fix bug 42.'):
      pass

  # Without the _run_impl override the Antigravity SDK silently receives
  # ctx.user_content:
  # a plausible-looking wrong prompt rather than an exception.
  active_agent.conversation.send.assert_awaited_once_with('Fix bug 42.')


@pytest.mark.asyncio
async def test_last_complete_response_becomes_node_output():

  async def _receive_steps() -> AsyncIterator[MagicMock]:
    yield _text_step(0, 'Let me look at the file.')
    yield _text_step(1, 'Done: patch sent for review.')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )
  ctx = await _node_ctx(agent=agent)

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    events = [e async for e in agent._run_impl(ctx=ctx, node_input='go')]

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Done: patch sent for review.']


def _tool_response_step(step_index: int, name: str) -> sdk_types.Step:
  """A real Antigravity SDK Step for a completed tool call by ``name``."""
  return sdk_types.Step(
      step_index=step_index,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.SYSTEM,
      status=sdk_types.StepStatus.DONE,
      content='ok',
      tool_calls=[sdk_types.ToolCall(name=name, args={}, id=f'c{step_index}')],
  )


@pytest.mark.asyncio
async def test_output_reaches_the_parent_through_node_runner():

  async def _receive_steps() -> AsyncIterator[sdk_types.Step | MagicMock]:
    yield _text_step(0, 'Done: patch sent for review.')
    # Ending on a tool step exercises NodeRunner's author enrichment, which
    # would otherwise attribute the output event to 'run_command'.
    yield _tool_response_step(1, 'run_command')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    child_ctx, enqueued = await _run_via_node_runner(agent, 'go')

  assert child_ctx.output == 'Done: patch sent for review.'
  output_events = [e for e in enqueued if e.output is not None]
  assert [e.author for e in output_events] == ['agy']


@pytest.mark.asyncio
async def test_text_less_run_outputs_empty_string_not_none():

  async def _receive_steps() -> AsyncIterator[sdk_types.Step]:
    # Reachable when a trajectory ends on tool calls with no closing summary.
    yield _tool_response_step(0, 'run_command')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(
      name='agy', config=_make_config(), mode='single_turn'
  )

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    child_ctx, _ = await _run_via_node_runner(agent, 'go')

  # None would put `{"result": null}` in front of the parent's model.
  assert child_ctx.output == ''


def test_chat_mode_is_rejected():
  # AntigravityAgent is not an LlmAgent, so LlmAgent's other modes ('chat',
  # 'task') have no meaning here.
  with pytest.raises(ValidationError, match='single_turn'):
    AntigravityAgent(name='agy', config=_make_config(), mode='chat')


def _client_tool_active_step() -> sdk_types.Step:
  return sdk_types.Step(
      id='call_3',
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      target=sdk_types.StepTarget.ENVIRONMENT,
      status=sdk_types.StepStatus.ACTIVE,
      tool_calls=[
          sdk_types.ToolCall(
              name='reviewer', args={'request': 'go'}, id='call_3'
          )
      ],
  )


def _client_tool_done_step() -> sdk_types.Step:
  return sdk_types.Step(
      step_index=1,
      type=sdk_types.StepType.TOOL_CALL,
      source=sdk_types.StepSource.MODEL,
      target=sdk_types.StepTarget.ENVIRONMENT,
      status=sdk_types.StepStatus.DONE,
      content='Calling custom tool "reviewer"',
      tool_calls=[],
  )


class _ToolFailure(RuntimeError):
  """Stands in for the SDK's ``ToolExecutionError``, which is not exported."""

  def __init__(self, message: str, tool_name: str, call_id: str | None = None):
    super().__init__(message)
    self.tool_name = tool_name
    self.call_id = call_id


def _client_tool_conversation(
    *,
    hook_arrives: Literal['before_done', 'after_stream'],
    outcome: Literal['success', 'failure'] = 'success',
    built: list[AgentConfig] | None = None,
) -> Callable[[AgentConfig], MagicMock]:
  """An `Agent` stand-in replaying one client-tool call, firing the hook itself."""
  # Hook dispatch is backgrounded (event_processor.py:483-487), so the outcome
  # races the terminal step; `hook_arrives` replays both orderings.

  def _build(config: AgentConfig) -> MagicMock:
    if built is not None:
      built.append(config)
    hooks = list(config.hooks)

    async def _deliver_result() -> None:
      if outcome == 'failure':
        failure = _ToolFailure(
            'child agent exploded', 'reviewer', call_id='call_3'
        )
        for hook in hooks:
          if isinstance(hook, sdk_hooks.OnToolErrorHook):
            await hook.run(None, failure)
        return
      result = sdk_types.ToolResult(
          name='reviewer', id='call_3', result='{"result": "Looks good."}'
      )
      for hook in hooks:
        if isinstance(hook, sdk_hooks.PostToolCallHook):
          await hook.run(None, result)

    async def _receive_steps() -> AsyncIterator[sdk_types.Step]:
      yield _client_tool_active_step()
      if hook_arrives == 'before_done':
        await _deliver_result()
      yield _client_tool_done_step()
      if hook_arrives == 'after_stream':
        await _deliver_result()

    return _fake_active_agent(_receive_steps)

  return _build


def _function_responses(
    events: Sequence[Event],
) -> list[tuple[str | None, str | None, dict[str, JsonValue] | None]]:
  """Returns (name, id, response) for each function-response part emitted."""
  found = []
  for event in events:
    for part in event.content.parts if event.content else []:
      if part.function_response:
        found.append((
            part.function_response.name,
            part.function_response.id,
            part.function_response.response,
        ))
  return found


def test_tool_result_capture_cls_defaults_to_the_sdk_binding():
  agent = AntigravityAgent(name='agy', config=_make_config())

  assert (
      agent._tool_result_capture_cls is _tool_result_capture.ToolResultCapture
  )


@pytest.mark.asyncio
async def test_subclass_overrides_the_tool_result_capture_class():
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  configs: list[AgentConfig] = []

  class _OtherCapture(_tool_result_capture.ToolResultCapture):
    """Stands in for a binding against a different copy of the SDK."""

  class _Swapped(AntigravityAgent):

    @property
    def _tool_result_capture_cls(self) -> type[_OtherCapture]:
      return _OtherCapture

  agent = _Swapped(
      name='coder',
      config=_make_config(),
      sub_agents=[child],
      mode='single_turn',
  )

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return _fake_active_agent(_steps_once)

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(_mock_run_ctx()):
      pass

  assert [type(hook) for hook in configs[0].hooks] == [
      _OtherCapture,
      _tool_result_capture.ToolErrorCapture,
  ]


@pytest.mark.asyncio
async def test_no_capture_hook_is_registered_without_sub_agents():
  # Registering a post_tool_call hook flips LIFECYCLE_HOOK_POST_TOOL on, which
  # costs a blocking round trip per successful tool call.
  configs: list[AgentConfig] = []

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return _fake_active_agent(_steps_once)

  agent = AntigravityAgent(name='agy', config=_make_config())

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(_mock_run_ctx()):
      pass

  assert not configs[0].hooks


@pytest.mark.asyncio
async def test_both_capture_hooks_are_registered_when_there_are_sub_agents():
  # Both halves: the harness routes a tool to exactly one of them, a success to
  # `post_tool_call` and a failure to `on_tool_error`.
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  configs: list[AgentConfig] = []

  def _build(config: AgentConfig) -> MagicMock:
    configs.append(config)
    return _fake_active_agent(_steps_once)

  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )

  with patch.object(_antigravity_agent, 'Agent', _build):
    async for _ in agent._run_async_impl(_mock_run_ctx()):
      pass

  assert [type(hook) for hook in configs[0].hooks] == [
      _tool_result_capture.ToolResultCapture,
      _tool_result_capture.ToolErrorCapture,
  ]


@pytest.mark.asyncio
async def test_the_capture_keeps_its_identity_through_the_config_deep_copy():
  # `AgentConfig.model_copy` re-shallow-copies `hooks` after the base deep
  # copy, which is why the capture needs no `__deepcopy__`.
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )
  capture = agent._tool_result_capture_cls()

  built = agent._build_sdk_config(capture)

  assert built.hooks[0] is capture
  assert isinstance(built.hooks[1], _tool_result_capture.ToolErrorCapture)
  # Both hooks feed one buffer: a failure recorded through the error hook is
  # readable off the success capture.
  await built.hooks[1].run(None, _ToolFailure('boom', 'reviewer', call_id='c1'))
  assert [call_id for call_id, _ in capture.take({'c1'})] == ['c1']


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_arrives', ['before_done', 'after_stream'])
async def test_a_client_tool_call_is_answered_whichever_order_it_arrives_in(
    hook_arrives: Literal['before_done', 'after_stream'],
):
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )

  with patch.object(
      _antigravity_agent,
      'Agent',
      _client_tool_conversation(hook_arrives=hook_arrives),
  ):
    events = [e async for e in agent._run_async_impl(_mock_run_ctx())]

  calls = []
  for e in events:
    for part in e.content.parts if e.content else []:
      if part.function_call:
        calls.append(part.function_call.id)
  assert calls == ['call_3']
  assert _function_responses(events) == [
      ('reviewer', 'call_3', {'result': 'Looks good.'})
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_arrives', ['before_done', 'after_stream'])
async def test_a_failed_client_tool_is_answered_with_an_error_response(
    hook_arrives: Literal['before_done', 'after_stream'],
):
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )

  with patch.object(
      _antigravity_agent,
      'Agent',
      _client_tool_conversation(hook_arrives=hook_arrives, outcome='failure'),
  ):
    events = [e async for e in agent._run_async_impl(_mock_run_ctx())]

  assert _function_responses(events) == [
      ('reviewer', 'call_3', {'error': 'child agent exploded'})
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize('hook_arrives', ['before_done', 'after_stream'])
async def test_a_single_turn_agent_answers_its_client_tool_calls_too(
    hook_arrives: Literal['before_done', 'after_stream'],
):
  # mode='single_turn' bypasses the session-keyed path: it enters and exits its
  # own SDK agent, reaching the capture by a different route.
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  agent = AntigravityAgent(
      name='coder',
      config=_make_config(),
      sub_agents=[child],
      mode='single_turn',
  )

  with patch.object(
      _antigravity_agent,
      'Agent',
      _client_tool_conversation(hook_arrives=hook_arrives),
  ):
    events = [e async for e in agent._run_async_impl(_mock_run_ctx())]

  assert _function_responses(events) == [
      ('reviewer', 'call_3', {'result': 'Looks good.'})
  ]


@pytest.mark.asyncio
async def test_each_conversation_gets_its_own_capture():
  # Call ids are only unique within a conversation, so a shared buffer would
  # drain session A's result against session B's identically-numbered call.
  child = _StubChild(name='reviewer', description='Reviews a diff.')
  built: list[AgentConfig] = []
  agent = AntigravityAgent(
      name='coder', config=_make_config(), sub_agents=[child]
  )

  with patch.object(
      _antigravity_agent,
      'Agent',
      _client_tool_conversation(hook_arrives='before_done', built=built),
  ):
    async for _ in agent._run_async_impl(_mock_run_ctx(session_id='s1')):
      pass
    async for _ in agent._run_async_impl(_mock_run_ctx(session_id='s2')):
      pass

  first, second = built[0].hooks[-1], built[1].hooks[-1]
  assert first is not second


@pytest.mark.asyncio
async def test_node_input_none_is_a_no_op():
  """A classic agent-tree run still reads ctx.user_content."""

  async def _receive_steps() -> AsyncIterator[MagicMock]:
    yield _text_step(0, 'done')

  active_agent = _fake_active_agent(_receive_steps)
  agent = AntigravityAgent(name='agy', config=_make_config())
  ctx = await _node_ctx(user_text='the original message', agent=agent)

  with patch.object(_antigravity_agent, 'Agent', return_value=active_agent):
    async for _ in agent._run_impl(ctx=ctx, node_input=None):
      pass

  active_agent.conversation.send.assert_awaited_once_with(
      'the original message'
  )

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

import asyncio
import time
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.platform.thread import create_thread
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai.types import Content
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


class TestInvocationContext:
  """Test suite for InvocationContext."""

  @pytest.fixture
  def mock_events(self):
    """Create mock events for testing."""
    event1 = Mock(spec=Event)
    event1.invocation_id = 'inv_1'
    event1.branch = 'agent_1'

    event2 = Mock(spec=Event)
    event2.invocation_id = 'inv_1'
    event2.branch = 'agent_2'

    event3 = Mock(spec=Event)
    event3.invocation_id = 'inv_2'
    event3.branch = 'agent_1'

    event4 = Mock(spec=Event)
    event4.invocation_id = 'inv_2'
    event4.branch = 'agent_2'

    return [event1, event2, event3, event4]

  @pytest.fixture
  def mock_invocation_context(self, mock_events):
    """Create a mock invocation context for testing."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        branch='agent_1',
        session=Mock(spec=Session, events=mock_events),
    )
    return ctx

  def test_get_events_returns_all_events_by_default(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events returns all events when no filters are applied."""
    events = mock_invocation_context._get_events()
    assert events == mock_events

  def test_get_events_filters_by_current_invocation(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current invocation."""
    event1, event2, _, _ = mock_events
    events = mock_invocation_context._get_events(current_invocation=True)
    assert events == [event1, event2]

  def test_get_events_filters_by_current_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current branch."""
    event1, _, event3, _ = mock_events
    events = mock_invocation_context._get_events(current_branch=True)
    assert events == [event1, event3]

  def test_get_events_filters_by_invocation_and_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events filters by invocation and branch."""
    event1, _, _, _ = mock_events
    events = mock_invocation_context._get_events(
        current_invocation=True,
        current_branch=True,
    )
    assert events == [event1]

  def test_get_events_with_no_events_in_session(self, mock_invocation_context):
    """Tests get_events when the session has no events."""
    mock_invocation_context.session.events = []
    events = mock_invocation_context._get_events()
    assert not events

  def test_get_events_with_no_matching_events(self, mock_invocation_context):
    """Tests get_events when no events match the filters."""
    mock_invocation_context.invocation_id = 'inv_3'
    mock_invocation_context.branch = 'branch_C'

    # Filter by invocation
    events = mock_invocation_context._get_events(current_invocation=True)
    assert not events

    # Filter by branch
    events = mock_invocation_context._get_events(current_branch=True)
    assert not events

    # Filter by both
    events = mock_invocation_context._get_events(
        current_invocation=True,
        current_branch=True,
    )
    assert not events

  def test_abort_without_external_signal_trips_internal_signal(self):
    """Calling abort() without an explicit signal sets is_aborted and trips internal signal."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
    )
    assert ctx.is_aborted is False
    assert ctx._abort_signal.is_set() is False

    ctx.abort()

    assert ctx.is_aborted is True
    assert ctx._abort_signal.is_set() is True

  async def test_abort_same_tick_immediacy(self):
    """Calling abort() on the event loop thread sets is_aborted and trips signal on the same tick."""
    abort_signal = asyncio.Event()
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
    )
    ctx._attach_abort_signal(abort_signal)
    assert ctx.is_aborted is False
    assert abort_signal.is_set() is False

    ctx.abort()

    assert ctx.is_aborted is True
    assert abort_signal.is_set() is True

  async def test_abort_cross_thread_wakeup(self):
    """Calling abort() from a worker thread wakes a coroutine awaiting the attached signal."""
    abort_signal = asyncio.Event()
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
    )
    ctx._attach_abort_signal(abort_signal)

    woke = asyncio.Event()

    async def parked_waiter():
      await abort_signal.wait()
      woke.set()

    waiter_task = asyncio.create_task(parked_waiter())
    await asyncio.sleep(0.05)

    def foreign_worker():
      time.sleep(0.05)
      ctx.abort()

    worker_thread = create_thread(target=foreign_worker)
    worker_thread.start()

    await asyncio.wait_for(woke.wait(), timeout=2.0)
    worker_thread.join()
    await waiter_task

    assert ctx.is_aborted is True
    assert abort_signal.is_set() is True

  async def test_abort_initialized_outside_loop_cross_thread_wakeup(self):
    """Attaching an abort signal inside a running loop enables cross-thread wakeup even if context was created synchronously."""
    loop_holder = []

    def create_ctx_sync():
      loop_holder.append(
          InvocationContext(
              session_service=Mock(spec=BaseSessionService),
              agent=Mock(spec=BaseAgent),
              invocation_id='inv_outside_loop',
              session=Mock(spec=Session, events=[]),
          )
      )

    init_thread = create_thread(target=create_ctx_sync)
    init_thread.start()
    init_thread.join()
    ctx = loop_holder[0]

    abort_signal = asyncio.Event()
    ctx._attach_abort_signal(abort_signal)

    woke = asyncio.Event()

    async def parked_waiter():
      await abort_signal.wait()
      woke.set()

    waiter_task = asyncio.create_task(parked_waiter())
    await asyncio.sleep(0.05)

    def foreign_worker():
      time.sleep(0.05)
      ctx.abort()

    worker_thread = create_thread(target=foreign_worker)
    worker_thread.start()

    await asyncio.wait_for(woke.wait(), timeout=2.0)
    worker_thread.join()
    await waiter_task

    assert ctx.is_aborted is True
    assert abort_signal.is_set() is True

  async def test_abort_signal_captures_loop_and_wakes_on_cross_thread_abort(
      self,
  ):
    """Awaiting _abort_signal.wait() captures the running loop and wakes when abort() is called from another thread."""
    ctx_holder = []

    def create_ctx_sync():
      ctx_holder.append(
          InvocationContext(
              session_service=Mock(spec=BaseSessionService),
              agent=Mock(spec=BaseAgent),
              invocation_id='inv_abort_signal_wait',
              session=Mock(spec=Session, events=[]),
          )
      )

    init_thread = create_thread(target=create_ctx_sync)
    init_thread.start()
    init_thread.join()
    ctx = ctx_holder[0]
    assert ctx._abort_state.loop is None

    woke = asyncio.Event()

    async def parked_waiter():
      await ctx._abort_signal.wait()
      woke.set()

    waiter_task = asyncio.create_task(parked_waiter())
    await asyncio.sleep(0.05)

    def foreign_worker():
      time.sleep(0.05)
      ctx.abort()

    worker_thread = create_thread(target=foreign_worker)
    worker_thread.start()

    await asyncio.wait_for(woke.wait(), timeout=2.0)
    worker_thread.join()
    await waiter_task

    assert ctx.is_aborted is True

  def test_abort_with_closed_loop_falls_back_to_direct_set(self):
    """Calling abort() after its event loop closes still sets is_aborted without raising RuntimeError."""
    loop = asyncio.new_event_loop()

    async def init_on_loop():
      ctx = InvocationContext(
          session_service=Mock(spec=BaseSessionService),
          agent=Mock(spec=BaseAgent),
          invocation_id='inv_closed_loop',
          session=Mock(spec=Session, events=[]),
      )
      ctx._attach_abort_signal(asyncio.Event())
      return ctx

    ctx = loop.run_until_complete(init_on_loop())
    loop.close()

    ctx.abort()

    assert ctx.is_aborted is True
    assert ctx._abort_signal.is_set() is True

  def test_abort_signal_not_in_model_fields(self):
    """The abort signal is private and excluded from Pydantic model fields."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_priv_attr',
        session=Mock(spec=Session, events=[]),
    )
    assert 'abort_signal' not in InvocationContext.model_fields
    assert '_abort_signal' not in InvocationContext.model_fields
    assert not hasattr(ctx, 'abort_signal')
    assert isinstance(ctx._abort_signal, asyncio.Event)
    assert ctx._abort_signal.is_set() is False

  def test_attach_abort_signal(self):
    """Attaching a caller-owned abort signal shares it across model_copy() instances."""
    custom_signal = asyncio.Event()
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_signal',
        session=Mock(spec=Session, events=[]),
    )
    ctx._attach_abort_signal(custom_signal)

    assert ctx._abort_signal is custom_signal
    assert ctx.model_copy()._abort_signal is custom_signal

  def test_abort_signal_propagates_across_model_copy(self):
    """Calling abort() on a copied context immediately sets is_aborted on the parent context."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_parent',
        session=Mock(spec=Session, events=[]),
    )
    copied = ctx.model_copy()
    assert ctx.is_aborted is False
    assert copied.is_aborted is False

    copied.abort()

    assert copied.is_aborted is True
    assert ctx.is_aborted is True
    assert ctx._abort_signal.is_set() is True

  def test_abort_state_deepcopy_shares_instance(self):
    """Deepcopying InvocationContext preserves shared _abort_state instance."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_deep',
        session=Mock(spec=Session, events=[]),
    )
    copied = ctx.model_copy(deep=True)
    assert copied._abort_state is ctx._abort_state

    copied.abort()
    assert copied.is_aborted is True
    assert ctx.is_aborted is True

  async def test_model_copy_deep_with_running_async_generator(self):
    """Calling model_copy(deep=True) inside a running event loop with active async generators does not raise."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_deep_running_loop',
        session=Mock(spec=Session, events=[]),
    )

    async def sample_generator():
      yield 1

    gen = sample_generator()
    try:
      copied = ctx.model_copy(deep=True)
      assert copied._abort_state is ctx._abort_state
    finally:
      await gen.aclose()

  async def test_abort_signal_on_model_copy_wakes_when_parent_aborts_from_thread(
      self,
  ):
    """A coroutine awaiting _abort_signal.wait() on a copied context wakes when the parent aborts from a worker thread."""
    ctx_holder = []

    def create_ctx_sync():
      ctx_holder.append(
          InvocationContext(
              session_service=Mock(spec=BaseSessionService),
              agent=Mock(spec=BaseAgent),
              invocation_id='inv_parent',
              session=Mock(spec=Session, events=[]),
          )
      )

    init_thread = create_thread(target=create_ctx_sync)
    init_thread.start()
    init_thread.join()
    ctx = ctx_holder[0]
    copied = ctx.model_copy()

    woke = asyncio.Event()

    async def parked_waiter():
      await copied._abort_signal.wait()
      woke.set()

    waiter_task = asyncio.create_task(parked_waiter())
    await asyncio.sleep(0.05)

    def foreign_worker():
      time.sleep(0.05)
      ctx.abort()

    worker_thread = create_thread(target=foreign_worker)
    worker_thread.start()

    await asyncio.wait_for(woke.wait(), timeout=2.0)
    worker_thread.join()
    await waiter_task

    assert copied.is_aborted is True
    assert ctx.is_aborted is True

  def test_abort_foreign_thread_sets_is_aborted_immediately(self):
    """Calling abort() from a worker thread sets is_aborted synchronously before the loop processes callbacks."""
    loop = Mock(spec=asyncio.AbstractEventLoop)
    loop.call_soon_threadsafe = Mock()

    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_foreign',
        session=Mock(spec=Session, events=[]),
    )
    ctx._abort_state.loop = loop

    worker_exc = None

    def foreign_worker():
      nonlocal worker_exc
      try:
        ctx.abort()
        assert ctx.is_aborted is True
      except BaseException as e:
        worker_exc = e

    worker = create_thread(target=foreign_worker)
    worker.start()
    worker.join()

    if worker_exc is not None:
      raise worker_exc

    assert ctx.is_aborted is True
    loop.call_soon_threadsafe.assert_called_once()


class TestInvocationContextInitialization:
  """Test suite for InvocationContext initialization."""

  def test_custom_metadata_propagation(self):
    """Tests that custom_metadata from RunConfig is propagated to InvocationContext."""
    run_cfg = RunConfig(custom_metadata={'test_key': 'test_value'})
    inv_ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
        run_config=run_cfg,
    )
    # Access private attribute to verify
    assert inv_ctx._custom_metadata == {'test_key': 'test_value'}

  def test_custom_metadata_default_empty(self):
    """Tests that _custom_metadata is empty by default when no RunConfig is provided."""
    inv_ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
    )
    assert inv_ctx._custom_metadata == {}

  def test_custom_metadata_empty_run_config(self):
    """Tests that _custom_metadata is empty when RunConfig has no custom_metadata."""
    run_cfg = RunConfig()
    inv_ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
        run_config=run_cfg,
    )
    assert inv_ctx._custom_metadata == {}


class TestInvocationContextWithAppResumablity:
  """Test suite for InvocationContext regarding app resumability."""

  @pytest.fixture
  def long_running_function_call(self) -> FunctionCall:
    """A long running function call."""
    return FunctionCall(
        id='tool_call_id_1',
        name='long_running_function_call',
        args={},
    )

  @pytest.fixture
  def event_to_pause(self, long_running_function_call) -> Event:
    """An event with a long running function call."""
    return Event(
        invocation_id='inv_1',
        author='agent',
        content=testing_utils.ModelContent(
            [Part(function_call=long_running_function_call)]
        ),
        long_running_tool_ids=[long_running_function_call.id],
    )

  def _create_test_invocation_context(
      self, resumability_config: ResumabilityConfig | None = None
  ) -> InvocationContext:
    """Create a mock invocation context for testing."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
        resumability_config=resumability_config,
    )
    return ctx

  def test_should_pause_invocation_with_resumable_app(self, event_to_pause):
    """Tests should_pause_invocation with a resumable app."""
    mock_invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )

    assert mock_invocation_context.should_pause_invocation(event_to_pause)

  def test_should_pause_invocation_with_non_resumable_app(self, event_to_pause):
    """Tests should_pause_invocation pauses even without resumability."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )

    assert invocation_context.should_pause_invocation(event_to_pause)

  def test_should_not_pause_invocation_with_no_long_running_tool_ids(
      self, event_to_pause
  ):
    """Tests should_pause_invocation with no long running tools."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    nonpausable_event = event_to_pause.model_copy(
        update={'long_running_tool_ids': []}
    )

    assert not invocation_context.should_pause_invocation(nonpausable_event)

  def test_should_not_pause_invocation_with_no_function_calls(
      self, event_to_pause
  ):
    """Tests should_pause_invocation with a non-model event."""
    mock_invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    nonpausable_event = event_to_pause.model_copy(
        update={'content': testing_utils.UserContent('test text part')}
    )

    assert not mock_invocation_context.should_pause_invocation(
        nonpausable_event
    )

  def test_should_not_pause_when_user_resumes_in_sub_branch(
      self, event_to_pause, long_running_function_call
  ):
    """We do not pause the invocation if a subsequent user event belongs to a sub-branch."""
    # Arrange
    mock_invocation_context = self._create_test_invocation_context()
    user_event = Event(
        invocation_id='inv_1',
        author='user',
        branch=f'agent@{long_running_function_call.id}.child',
    )
    mock_invocation_context.session.events = [event_to_pause, user_event]

    # Act
    should_pause = mock_invocation_context.should_pause_invocation(
        event_to_pause
    )

    # Assert
    assert not should_pause

  def test_should_not_pause_when_user_resumes_in_deeply_nested_sub_branch(
      self, event_to_pause, long_running_function_call
  ):
    """We do not pause if the user resumes in a deeply nested sub-branch containing the tool call."""
    # Arrange
    mock_invocation_context = self._create_test_invocation_context()
    user_event = Event(
        invocation_id='inv_1',
        author='user',
        branch=f'parent@other.child@{long_running_function_call.id}.grandchild',
    )
    mock_invocation_context.session.events = [event_to_pause, user_event]

    # Act
    should_pause = mock_invocation_context.should_pause_invocation(
        event_to_pause
    )

    # Assert
    assert not should_pause

  def test_should_pause_when_user_resumes_in_different_branch(
      self, event_to_pause
  ):
    """We still pause the invocation if the subsequent user event belongs to a different branch."""
    # Arrange
    mock_invocation_context = self._create_test_invocation_context()
    user_event = Event(
        invocation_id='inv_1',
        author='user',
        branch='parent@different_id.child',
    )
    mock_invocation_context.session.events = [event_to_pause, user_event]

    # Act
    should_pause = mock_invocation_context.should_pause_invocation(
        event_to_pause
    )

    # Assert
    assert should_pause

  def test_is_resumable_true(self):
    """Tests that is_resumable is True when resumability is enabled."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    assert invocation_context.is_resumable

  def test_is_resumable_false(self):
    """Tests that is_resumable is False when resumability is disabled."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )
    assert not invocation_context.is_resumable

  def test_is_resumable_no_config(self):
    """Tests that is_resumable is False when no resumability config is set."""
    invocation_context = self._create_test_invocation_context(None)
    assert not invocation_context.is_resumable

  def test_populate_invocation_agent_states_not_resumable(self):
    """Tests that populate_invocation_agent_states does nothing if not resumable."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=True, agent_state=None),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_populate_invocation_agent_states_end_of_agent(self):
    """Tests that populate_invocation_agent_states handles end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=True, agent_state=None),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert invocation_context.end_of_agents == {'agent1': True}

  def test_populate_invocation_agent_states_with_agent_state(self):
    """Tests that populate_invocation_agent_states handles agent_state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(
            end_of_agent=False,
            agent_state=BaseAgentState().model_dump(mode='json'),
        ),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert invocation_context.agent_states == {'agent1': {}}
    assert invocation_context.end_of_agents == {'agent1': False}

  def test_populate_invocation_agent_states_with_agent_state_and_end_of_agent(
      self,
  ):
    """Tests that populate_invocation_agent_states handles agent_state and end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(
            end_of_agent=True,
            agent_state=BaseAgentState().model_dump(mode='json'),
        ),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    # When both agent_state and end_of_agent are set, agent_state should be
    # cleared, as end_of_agent is of a higher priority.
    assert not invocation_context.agent_states
    assert invocation_context.end_of_agents == {'agent1': True}

  def test_populate_invocation_agent_states_with_content_no_state(self):
    """Tests that populate_invocation_agent_states creates default state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=False, agent_state=None),
        content=Content(role='model', parts=[Part(text='hi')]),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert invocation_context.agent_states == {
        'agent1': BaseAgentState().model_dump(mode='json')
    }
    assert invocation_context.end_of_agents == {'agent1': False}

  def test_populate_invocation_agent_states_user_message_event(self):
    """Tests that populate_invocation_agent_states ignores user message events for default state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='user',
        actions=EventActions(end_of_agent=False, agent_state=None),
        content=Content(role='user', parts=[Part(text='hi')]),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_populate_invocation_agent_states_no_content(self):
    """Tests that populate_invocation_agent_states ignores events with no content if no state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=None, agent_state=None),
        content=None,
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_set_agent_state_with_end_of_agent_true(self):
    """Tests that set_agent_state clears agent_state and sets end_of_agent to True."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent_states['agent1'] = {}
    invocation_context.end_of_agents['agent1'] = False

    # Set state with end_of_agent=True, which should clear the existing
    # agent_state.
    invocation_context.set_agent_state('agent1', end_of_agent=True)
    assert 'agent1' not in invocation_context.agent_states
    assert invocation_context.end_of_agents['agent1']

  def test_set_agent_state_with_agent_state(self):
    """Tests that set_agent_state sets agent_state and sets end_of_agent to False."""
    agent_state = BaseAgentState()
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.end_of_agents['agent1'] = True

    # Set state with agent_state=agent_state, which should set the agent_state
    # and reset the end_of_agent flag to False.
    invocation_context.set_agent_state('agent1', agent_state=agent_state)
    assert invocation_context.agent_states['agent1'] == agent_state.model_dump(
        mode='json'
    )
    assert invocation_context.end_of_agents['agent1'] is False

  def test_reset_agent_state(self):
    """Tests that set_agent_state clears agent_state and end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent_states['agent1'] = {}
    invocation_context.end_of_agents['agent1'] = True

    # Reset state, which should clear the agent_state and end_of_agent flag.
    invocation_context.set_agent_state('agent1')
    assert 'agent1' not in invocation_context.agent_states
    assert 'agent1' not in invocation_context.end_of_agents

  def test_reset_sub_agent_states(self):
    """Tests that reset_sub_agent_states resets sub-agent states."""
    sub_sub_agent_1 = BaseAgent(name='sub_sub_agent_1')
    sub_agent_1 = BaseAgent(name='sub_agent_1', sub_agents=[sub_sub_agent_1])
    sub_agent_2 = BaseAgent(name='sub_agent_2')
    root_agent = BaseAgent(
        name='root_agent', sub_agents=[sub_agent_1, sub_agent_2]
    )

    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent = root_agent
    invocation_context.set_agent_state(
        'sub_agent_1', agent_state=BaseAgentState()
    )
    invocation_context.set_agent_state('sub_agent_2', end_of_agent=True)
    invocation_context.set_agent_state(
        'sub_sub_agent_1', agent_state=BaseAgentState()
    )

    assert 'sub_agent_1' in invocation_context.agent_states
    assert 'sub_agent_2' in invocation_context.end_of_agents
    assert 'sub_sub_agent_1' in invocation_context.agent_states

    invocation_context.reset_sub_agent_states('root_agent')

    assert 'sub_agent_1' not in invocation_context.agent_states
    assert 'sub_agent_1' not in invocation_context.end_of_agents
    assert 'sub_agent_2' not in invocation_context.agent_states
    assert 'sub_agent_2' not in invocation_context.end_of_agents
    assert 'sub_sub_agent_1' not in invocation_context.agent_states
    assert 'sub_sub_agent_1' not in invocation_context.end_of_agents


class TestIncrementLlmCallCount:
  """Test suite for InvocationContext.increment_llm_call_count."""

  def _context(self, run_config=None):
    kwargs = {} if run_config is None else {'run_config': run_config}
    return InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session, events=[]),
        **kwargs,
    )

  def test_allows_exactly_max_llm_calls_then_raises(self):
    """The limit is the number of calls allowed, not the count before it."""
    ctx = self._context(RunConfig(max_llm_calls=2))

    ctx.increment_llm_call_count()
    ctx.increment_llm_call_count()

    with pytest.raises(LlmCallsLimitExceededError, match='limit of `2`'):
      ctx.increment_llm_call_count()

  def test_keeps_raising_once_the_limit_is_passed(self):
    """The limit latches: a caller cannot swallow one error and carry on."""
    ctx = self._context(RunConfig(max_llm_calls=1))
    ctx.increment_llm_call_count()

    with pytest.raises(LlmCallsLimitExceededError):
      ctx.increment_llm_call_count()
    with pytest.raises(LlmCallsLimitExceededError):
      ctx.increment_llm_call_count()

  @pytest.mark.parametrize('max_llm_calls', [0, -1])
  def test_non_positive_limit_is_not_enforced(self, max_llm_calls: int):
    """A non-positive limit documents 'no enforcement', not 'no calls'."""
    ctx = self._context(RunConfig(max_llm_calls=max_llm_calls))

    for _ in range(5):
      ctx.increment_llm_call_count()

  def test_without_run_config_the_limit_is_not_enforced(self):
    """run_config is optional, so counting must tolerate its absence."""
    ctx = self._context()
    assert ctx.run_config is None

    for _ in range(5):
      ctx.increment_llm_call_count()

  def test_count_is_per_invocation_context(self):
    """Two invocations must not share a budget."""
    first = self._context(RunConfig(max_llm_calls=1))
    second = self._context(RunConfig(max_llm_calls=1))

    first.increment_llm_call_count()
    second.increment_llm_call_count()

    with pytest.raises(LlmCallsLimitExceededError):
      second.increment_llm_call_count()


def _ctx_on_branch(branch, events):
  """An InvocationContext on `branch` over a session holding `events`."""
  return InvocationContext(
      session_service=Mock(spec=BaseSessionService),
      agent=Mock(spec=BaseAgent),
      invocation_id='inv_1',
      branch=branch,
      session=Mock(spec=Session, events=events),
  )


def test_get_events_current_branch_includes_user_event_on_sub_branch():
  """A user event from a descendant sub-branch belongs to this subtree."""
  user_on_child = Event(
      invocation_id='inv_1', author='user', branch='agent_1.child'
  )
  ctx = _ctx_on_branch('agent_1', [user_on_child])

  assert ctx._get_events(current_branch=True) == [user_on_child]


def test_get_events_current_branch_excludes_agent_event_on_sub_branch():
  """A non-user event from a descendant sub-branch is not returned.

  This is asymmetric with the user case above on purpose: widening it would
  hand every caller a descendant's internal events. Pinned here so the
  asymmetry is a stated contract rather than an accident.
  """
  agent_on_child = Event(
      invocation_id='inv_1', author='some_agent', branch='agent_1.child'
  )
  ctx = _ctx_on_branch('agent_1', [agent_on_child])

  assert ctx._get_events(current_branch=True) == []


def test_get_events_current_branch_excludes_sibling_branch():
  """A sibling branch is never part of this subtree."""
  user_on_sibling = Event(
      invocation_id='inv_1', author='user', branch='agent_2'
  )
  ctx = _ctx_on_branch('agent_1', [user_on_sibling])

  assert ctx._get_events(current_branch=True) == []


def test_get_events_empty_branch_does_not_match_every_branched_event():
  """An empty branch must not behave like "match everything".

  An empty string is a real branch value in the workflow code, and a bare
  descendant test would treat every branched event as its descendant.
  """
  user_on_branch = Event(invocation_id='inv_1', author='user', branch='agent_1')
  ctx = _ctx_on_branch('', [user_on_branch])

  assert ctx._get_events(current_branch=True) == []


def _call_event(branch, call_id):
  """A non-user event issuing function call `call_id` on `branch`."""
  return Event(
      invocation_id='inv_1',
      author='some_agent',
      branch=branch,
      content=Content(
          parts=[Part(function_call=FunctionCall(id=call_id, name='t'))]
      ),
  )


def _user_response_event(branch, call_id):
  """A user event answering function call `call_id` on `branch`."""
  return Event(
      invocation_id='inv_1',
      author='user',
      branch=branch,
      content=Content(
          parts=[
              Part(
                  function_response=FunctionResponse(
                      id=call_id, name='t', response={}
                  )
              )
          ]
      ),
  )


def test_get_events_current_branch_keeps_user_response_to_a_call_here():
  """A reply answering a call issued in this subtree is returned."""
  call_here = _call_event('agent_1.child', 'fc_1')
  reply = _user_response_event('agent_1', 'fc_1')
  ctx = _ctx_on_branch('agent_1', [call_here, reply])

  assert ctx._get_events(current_branch=True) == [reply]


def test_get_events_current_branch_drops_user_response_to_a_call_elsewhere():
  """Sitting on this branch is not enough for a reply to a foreign call.

  The function-response gate is the only difference from the test above, so a
  reply that answers a parallel tree's call is dropped even though its own
  branch matches exactly.
  """
  call_elsewhere = _call_event('agent_2', 'fc_1')
  reply = _user_response_event('agent_1', 'fc_1')
  ctx = _ctx_on_branch('agent_1', [call_elsewhere, reply])

  assert ctx._get_events(current_branch=True) == []


def test_get_events_current_branch_drops_user_response_to_a_lookalike_call():
  """A branch that merely shares a prefix does not count as a sub-branch.

  `agent_10` starts with `agent_1`, so a plain prefix test would read the call
  as issued in this subtree and let the reply through.
  """
  call_on_lookalike = _call_event('agent_10', 'fc_1')
  reply = _user_response_event('agent_1', 'fc_1')
  ctx = _ctx_on_branch('agent_1', [call_on_lookalike, reply])

  assert ctx._get_events(current_branch=True) == []


def test_get_events_without_a_branch_matches_every_user_event():
  """A context with no branch sees user events wherever they sit.

  Non-user events stay on the strict rule, so this also pins the asymmetry:
  the agent event beside it is not returned.
  """
  user_elsewhere = Event(
      invocation_id='inv_1', author='user', branch='agent_2.child'
  )
  agent_elsewhere = Event(
      invocation_id='inv_1', author='some_agent', branch='agent_2.child'
  )
  ctx = _ctx_on_branch(None, [user_elsewhere, agent_elsewhere])

  assert ctx._get_events(current_branch=True) == [user_elsewhere]


def test_get_events_current_branch_filters_each_response_independently():
  """Several replies on one branch are each judged against their own call.

  The set of calls issued in this subtree is the same for every event, so it
  is built once per call rather than rescanned per reply. This pins that the
  shared set still discriminates: the reply to a foreign call is dropped while
  the replies around it are kept.
  """
  call_here = _call_event('agent_1', 'fc_here')
  call_on_child = _call_event('agent_1.child', 'fc_child')
  call_elsewhere = _call_event('agent_2', 'fc_far')
  reply_here = _user_response_event('agent_1', 'fc_here')
  reply_far = _user_response_event('agent_1', 'fc_far')
  reply_child = _user_response_event('agent_1', 'fc_child')
  ctx = _ctx_on_branch(
      'agent_1',
      [
          call_here,
          call_on_child,
          call_elsewhere,
          reply_here,
          reply_far,
          reply_child,
      ],
  )

  # `call_here` sits on exactly this branch, so the non-user rule returns it
  # too; the two sub-branch calls do not.
  assert ctx._get_events(current_branch=True) == [
      call_here,
      reply_here,
      reply_child,
  ]

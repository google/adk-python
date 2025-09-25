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

"""Testings for the BaseAgent."""

from enum import Enum
from functools import partial
from typing import AsyncGenerator
from typing import List
from typing import Optional
from typing import Union
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
import pytest_mock
from typing_extensions import override

from .. import testing_utils


def _before_agent_callback_noop(callback_context: CallbackContext) -> None:
  pass


async def _async_before_agent_callback_noop(
    callback_context: CallbackContext,
) -> None:
  pass


def _before_agent_callback_bypass_agent(
    callback_context: CallbackContext,
) -> types.Content:
  return types.Content(parts=[types.Part(text='agent run is bypassed.')])


async def _async_before_agent_callback_bypass_agent(
    callback_context: CallbackContext,
) -> types.Content:
  return types.Content(parts=[types.Part(text='agent run is bypassed.')])


def _after_agent_callback_noop(callback_context: CallbackContext) -> None:
  pass


async def _async_after_agent_callback_noop(
    callback_context: CallbackContext,
) -> None:
  pass


def _after_agent_callback_append_agent_reply(
    callback_context: CallbackContext,
) -> types.Content:
  return types.Content(
      parts=[types.Part(text='Agent reply from after agent callback.')]
  )


async def _async_after_agent_callback_append_agent_reply(
    callback_context: CallbackContext,
) -> types.Content:
  return types.Content(
      parts=[types.Part(text='Agent reply from after agent callback.')]
  )


class MockPlugin(BasePlugin):
  before_agent_text = 'before_agent_text from MockPlugin'
  after_agent_text = 'after_agent_text from MockPlugin'

  def __init__(self, name='mock_plugin'):
    self.name = name
    self.enable_before_agent_callback = False
    self.enable_after_agent_callback = False

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    if not self.enable_before_agent_callback:
      return None
    return types.Content(parts=[types.Part(text=self.before_agent_text)])

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    if not self.enable_after_agent_callback:
      return None
    return types.Content(parts=[types.Part(text=self.after_agent_text)])


@pytest.fixture
def mock_plugin():
  return MockPlugin()


class _IncompleteAgent(BaseAgent):
  pass


class _TestingAgent(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
        content=types.Content(parts=[types.Part(text='Hello, world!')]),
    )

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        branch=ctx.branch,
        content=types.Content(parts=[types.Part(text='Hello, live!')]),
    )


async def _create_parent_invocation_context(
    test_name: str,
    agent: BaseAgent,
    branch: Optional[str] = None,
    plugins: list[BasePlugin] = [],
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id=f'{test_name}_invocation_id',
      branch=branch,
      agent=agent,
      session=session,
      session_service=session_service,
      plugin_manager=PluginManager(plugins=plugins),
  )


def test_invalid_agent_name():
  with pytest.raises(ValueError):
    _ = _TestingAgent(name='not an identifier')


@pytest.mark.asyncio
async def test_run_async(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, world!'


@pytest.mark.asyncio
async def test_run_async_with_branch(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent, branch='parent_branch'
  )

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, world!'
  assert events[0].branch == 'parent_branch'


@pytest.mark.asyncio
async def test_run_async_before_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
) -> Union[types.Content, None]:
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_before_agent_callback_noop,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_run_async_impl = mocker.spy(agent, BaseAgent._run_async_impl.__name__)
  spy_before_agent_callback = mocker.spy(agent, 'before_agent_callback')

  # Act
  _ = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_before_agent_callback.assert_called_once()
  _, kwargs = spy_before_agent_callback.call_args
  assert 'callback_context' in kwargs
  assert isinstance(kwargs['callback_context'], CallbackContext)

  spy_run_async_impl.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_before_agent_callback_use_plugin(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
    mock_plugin: MockPlugin,
):
  """Test that the before agent callback uses the plugin response if both plugin callback and canonical agent callbacks are present."""
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_before_agent_callback_bypass_agent,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent, plugins=[mock_plugin]
  )
  mock_plugin.enable_before_agent_callback = True
  spy_run_async_impl = mocker.spy(agent, BaseAgent._run_async_impl.__name__)
  spy_before_agent_callback = mocker.spy(agent, 'before_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_before_agent_callback.assert_not_called()
  spy_run_async_impl.assert_not_called()

  assert len(events) == 1
  assert events[0].content.parts[0].text == MockPlugin.before_agent_text


@pytest.mark.asyncio
async def test_run_async_with_async_before_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
) -> Union[types.Content, None]:
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_async_before_agent_callback_noop,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_run_async_impl = mocker.spy(agent, BaseAgent._run_async_impl.__name__)
  spy_before_agent_callback = mocker.spy(agent, 'before_agent_callback')

  # Act
  _ = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_before_agent_callback.assert_called_once()
  _, kwargs = spy_before_agent_callback.call_args
  assert 'callback_context' in kwargs
  assert isinstance(kwargs['callback_context'], CallbackContext)

  spy_run_async_impl.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_before_agent_callback_bypass_agent(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_before_agent_callback_bypass_agent,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_run_async_impl = mocker.spy(agent, BaseAgent._run_async_impl.__name__)
  spy_before_agent_callback = mocker.spy(agent, 'before_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_before_agent_callback.assert_called_once()
  spy_run_async_impl.assert_not_called()

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'agent run is bypassed.'


@pytest.mark.asyncio
async def test_run_async_with_async_before_agent_callback_bypass_agent(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_async_before_agent_callback_bypass_agent,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_run_async_impl = mocker.spy(agent, BaseAgent._run_async_impl.__name__)
  spy_before_agent_callback = mocker.spy(agent, 'before_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_before_agent_callback.assert_called_once()
  spy_run_async_impl.assert_not_called()

  assert len(events) == 1
  assert events[0].content.parts[0].text == 'agent run is bypassed.'


class CallbackType(Enum):
  SYNC = 1
  ASYNC = 2


async def mock_async_agent_cb_side_effect(
    callback_context: CallbackContext,
    ret_value=None,
):
  if ret_value:
    return types.Content(parts=[types.Part(text=ret_value)])
  return None


def mock_sync_agent_cb_side_effect(
    callback_context: CallbackContext,
    ret_value=None,
):
  if ret_value:
    return types.Content(parts=[types.Part(text=ret_value)])
  return None


BEFORE_AGENT_CALLBACK_PARAMS = [
    pytest.param(
        [
            (None, CallbackType.SYNC),
            ('callback_2_response', CallbackType.ASYNC),
            ('callback_3_response', CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        ['callback_2_response'],
        [1, 1, 0, 0],
        id='middle_async_callback_returns',
    ),
    pytest.param(
        [
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        ['Hello, world!'],
        [1, 1, 1, 1],
        id='all_callbacks_return_none',
    ),
    pytest.param(
        [
            ('callback_1_response', CallbackType.SYNC),
            ('callback_2_response', CallbackType.ASYNC),
        ],
        ['callback_1_response'],
        [1, 0],
        id='first_sync_callback_returns',
    ),
]

AFTER_AGENT_CALLBACK_PARAMS = [
    pytest.param(
        [
            (None, CallbackType.SYNC),
            ('callback_2_response', CallbackType.ASYNC),
            ('callback_3_response', CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        ['Hello, world!', 'callback_2_response'],
        [1, 1, 0, 0],
        id='middle_async_callback_returns',
    ),
    pytest.param(
        [
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        ['Hello, world!'],
        [1, 1, 1, 1],
        id='all_callbacks_return_none',
    ),
    pytest.param(
        [
            ('callback_1_response', CallbackType.SYNC),
            ('callback_2_response', CallbackType.ASYNC),
        ],
        ['Hello, world!', 'callback_1_response'],
        [1, 0],
        id='first_sync_callback_returns',
    ),
]


@pytest.mark.parametrize(
    'callbacks, expected_responses, expected_calls',
    BEFORE_AGENT_CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_before_agent_callbacks_chain(
    callbacks: List[tuple[str, int]],
    expected_responses: List[str],
    expected_calls: List[int],
    request: pytest.FixtureRequest,
):
  mock_cbs = []
  for response, callback_type in callbacks:

    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_agent_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_agent_cb_side_effect, ret_value=response
          )
      )
    mock_cbs.append(mock_cb)

  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=[mock_cb for mock_cb in mock_cbs],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  result = [e async for e in agent.run_async(parent_ctx)]
  assert testing_utils.simplify_events(result) == [
      (f'{request.function.__name__}_test_agent', response)
      for response in expected_responses
  ]

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)


@pytest.mark.parametrize(
    'callbacks, expected_responses, expected_calls',
    AFTER_AGENT_CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_after_agent_callbacks_chain(
    callbacks: List[tuple[str, int]],
    expected_responses: List[str],
    expected_calls: List[int],
    request: pytest.FixtureRequest,
):
  mock_cbs = []
  for response, callback_type in callbacks:

    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_agent_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_agent_cb_side_effect, ret_value=response
          )
      )
    mock_cbs.append(mock_cb)

  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=[mock_cb for mock_cb in mock_cbs],
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  result = [e async for e in agent.run_async(parent_ctx)]
  assert testing_utils.simplify_events(result) == [
      (f'{request.function.__name__}_test_agent', response)
      for response in expected_responses
  ]

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)


@pytest.mark.asyncio
async def test_run_async_after_agent_callback_use_plugin(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
    mock_plugin: MockPlugin,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_after_agent_callback_noop,
  )
  mock_plugin.enable_after_agent_callback = True
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent, plugins=[mock_plugin]
  )
  spy_after_agent_callback = mocker.spy(agent, 'after_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_after_agent_callback.assert_not_called()
  # The first event is regular model response, the second event is
  # after_agent_callback response.
  assert len(events) == 2
  assert events[1].content.parts[0].text == mock_plugin.after_agent_text


@pytest.mark.asyncio
async def test_run_async_after_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_after_agent_callback_noop,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_after_agent_callback = mocker.spy(agent, 'after_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_after_agent_callback.assert_called_once()
  _, kwargs = spy_after_agent_callback.call_args
  assert 'callback_context' in kwargs
  assert isinstance(kwargs['callback_context'], CallbackContext)
  assert len(events) == 1


@pytest.mark.asyncio
async def test_run_async_with_async_after_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_async_after_agent_callback_noop,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )
  spy_after_agent_callback = mocker.spy(agent, 'after_agent_callback')

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  spy_after_agent_callback.assert_called_once()
  _, kwargs = spy_after_agent_callback.call_args
  assert 'callback_context' in kwargs
  assert isinstance(kwargs['callback_context'], CallbackContext)
  assert len(events) == 1


@pytest.mark.asyncio
async def test_run_async_after_agent_callback_append_reply(
    request: pytest.FixtureRequest,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_after_agent_callback_append_agent_reply,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  assert len(events) == 2
  assert events[1].author == agent.name
  assert (
      events[1].content.parts[0].text
      == 'Agent reply from after agent callback.'
  )


@pytest.mark.asyncio
async def test_run_async_with_async_after_agent_callback_append_reply(
    request: pytest.FixtureRequest,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_async_after_agent_callback_append_agent_reply,
  )
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  # Act
  events = [e async for e in agent.run_async(parent_ctx)]

  # Assert
  assert len(events) == 2
  assert events[1].author == agent.name
  assert (
      events[1].content.parts[0].text
      == 'Agent reply from after agent callback.'
  )


@pytest.mark.asyncio
async def test_run_async_incomplete_agent(request: pytest.FixtureRequest):
  agent = _IncompleteAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  with pytest.raises(NotImplementedError):
    [e async for e in agent.run_async(parent_ctx)]


@pytest.mark.asyncio
async def test_run_live(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  events = [e async for e in agent.run_live(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, live!'


@pytest.mark.asyncio
async def test_run_live_with_branch(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent, branch='parent_branch'
  )

  events = [e async for e in agent.run_live(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, live!'
  assert events[0].branch == 'parent_branch'


@pytest.mark.asyncio
async def test_run_live_incomplete_agent(request: pytest.FixtureRequest):
  agent = _IncompleteAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = await _create_parent_invocation_context(
      request.function.__name__, agent
  )

  with pytest.raises(NotImplementedError):
    [e async for e in agent.run_live(parent_ctx)]


@pytest.mark.asyncio
async def test_create_agent_state_event(request: pytest.FixtureRequest):
  # Arrange
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  ctx = await _create_parent_invocation_context(
      request.function.__name__, agent, branch='test_branch'
  )
  state = BaseAgentState()

  # Act
  event = agent._create_agent_state_event(ctx, state=state)

  # Assert
  assert event.invocation_id == ctx.invocation_id
  assert event.author == agent.name
  assert event.branch == 'test_branch'
  assert event.actions is not None
  assert event.actions.agent_state is not None
  assert event.actions.agent_state == state.model_dump(mode='json')
  assert not event.actions.end_of_agent

  # Act
  event = agent._create_agent_state_event(ctx, end_of_agent=True)

  # Assert
  assert event.invocation_id == ctx.invocation_id
  assert event.author == agent.name
  assert event.branch == 'test_branch'
  assert event.actions is not None
  assert event.actions.end_of_agent
  assert event.actions.agent_state is None


def test_set_parent_agent_for_sub_agents(request: pytest.FixtureRequest):
  sub_agents: list[BaseAgent] = [
      _TestingAgent(name=f'{request.function.__name__}_sub_agent_1'),
      _TestingAgent(name=f'{request.function.__name__}_sub_agent_2'),
  ]
  parent = _TestingAgent(
      name=f'{request.function.__name__}_parent',
      sub_agents=sub_agents,
  )

  for sub_agent in sub_agents:
    assert sub_agent.parent_agent == parent


def test_find_agent(request: pytest.FixtureRequest):
  grand_sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_1'
  )
  grand_sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_2'
  )
  sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_1',
      sub_agents=[grand_sub_agent_1],
  )
  sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_2',
      sub_agents=[grand_sub_agent_2],
  )
  parent = _TestingAgent(
      name=f'{request.function.__name__}_parent',
      sub_agents=[sub_agent_1, sub_agent_2],
  )

  assert parent.find_agent(parent.name) == parent
  assert parent.find_agent(sub_agent_1.name) == sub_agent_1
  assert parent.find_agent(sub_agent_2.name) == sub_agent_2
  assert parent.find_agent(grand_sub_agent_1.name) == grand_sub_agent_1
  assert parent.find_agent(grand_sub_agent_2.name) == grand_sub_agent_2
  assert sub_agent_1.find_agent(grand_sub_agent_1.name) == grand_sub_agent_1
  assert sub_agent_1.find_agent(grand_sub_agent_2.name) is None
  assert sub_agent_2.find_agent(grand_sub_agent_1.name) is None
  assert sub_agent_2.find_agent(sub_agent_2.name) == sub_agent_2
  assert parent.find_agent('not_exist') is None


def test_find_sub_agent(request: pytest.FixtureRequest):
  grand_sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_1'
  )
  grand_sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_2'
  )
  sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_1',
      sub_agents=[grand_sub_agent_1],
  )
  sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_2',
      sub_agents=[grand_sub_agent_2],
  )
  parent = _TestingAgent(
      name=f'{request.function.__name__}_parent',
      sub_agents=[sub_agent_1, sub_agent_2],
  )

  assert parent.find_sub_agent(sub_agent_1.name) == sub_agent_1
  assert parent.find_sub_agent(sub_agent_2.name) == sub_agent_2
  assert parent.find_sub_agent(grand_sub_agent_1.name) == grand_sub_agent_1
  assert parent.find_sub_agent(grand_sub_agent_2.name) == grand_sub_agent_2
  assert sub_agent_1.find_sub_agent(grand_sub_agent_1.name) == grand_sub_agent_1
  assert sub_agent_1.find_sub_agent(grand_sub_agent_2.name) is None
  assert sub_agent_2.find_sub_agent(grand_sub_agent_1.name) is None
  assert sub_agent_2.find_sub_agent(grand_sub_agent_2.name) == grand_sub_agent_2
  assert parent.find_sub_agent(parent.name) is None
  assert parent.find_sub_agent('not_exist') is None


def test_root_agent(request: pytest.FixtureRequest):
  grand_sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_1'
  )
  grand_sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}__grand_sub_agent_2'
  )
  sub_agent_1 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_1',
      sub_agents=[grand_sub_agent_1],
  )
  sub_agent_2 = _TestingAgent(
      name=f'{request.function.__name__}_sub_agent_2',
      sub_agents=[grand_sub_agent_2],
  )
  parent = _TestingAgent(
      name=f'{request.function.__name__}_parent',
      sub_agents=[sub_agent_1, sub_agent_2],
  )

  assert parent.root_agent == parent
  assert sub_agent_1.root_agent == parent
  assert sub_agent_2.root_agent == parent
  assert grand_sub_agent_1.root_agent == parent
  assert grand_sub_agent_2.root_agent == parent


def test_set_parent_agent_for_sub_agent_twice(
    request: pytest.FixtureRequest,
):
  sub_agent = _TestingAgent(name=f'{request.function.__name__}_sub_agent')
  _ = _TestingAgent(
      name=f'{request.function.__name__}_parent_1',
      sub_agents=[sub_agent],
  )
  with pytest.raises(ValueError):
    _ = _TestingAgent(
        name=f'{request.function.__name__}_parent_2',
        sub_agents=[sub_agent],
    )


if __name__ == '__main__':
  pytest.main([__file__])


class _TestAgentState(BaseAgentState):
  test_field: str = ''


@pytest.mark.asyncio
async def test_load_agent_state_no_resume():
  agent = BaseAgent(name='test_agent')
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  ctx = InvocationContext(
      invocation_id='test_invocation',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  default_state = _TestAgentState(test_field='default')

  state, is_resuming = agent._load_agent_state(
      ctx, _TestAgentState, default_state
  )

  assert not is_resuming
  assert state == default_state


@pytest.mark.asyncio
async def test_load_agent_state_with_resume():
  agent = BaseAgent(name='test_agent')
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  ctx = InvocationContext(
      invocation_id='test_invocation',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  persisted_state = _TestAgentState(test_field='resumed')
  ctx.agent_states[agent.name] = persisted_state.model_dump(mode='json')

  state, is_resuming = agent._load_agent_state(
      ctx, _TestAgentState, _TestAgentState()
  )

  assert is_resuming
  assert state == persisted_state

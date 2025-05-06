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

from typing import AsyncGenerator
from typing import Optional
from typing import Union

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest
import pytest_mock
from typing_extensions import override


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


def _create_parent_invocation_context(
    test_name: str, agent: BaseAgent, branch: Optional[str] = None
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id=f'{test_name}_invocation_id',
      branch=branch,
      agent=agent,
      session=session,
      session_service=session_service,
  )


def test_invalid_agent_name():
  with pytest.raises(ValueError):
    _ = _TestingAgent(name='not an identifier')


@pytest.mark.asyncio
async def test_run_async(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent
  )

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, world!'


@pytest.mark.asyncio
async def test_run_async_with_branch(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent, branch='parent_branch'
  )

  events = [e async for e in agent.run_async(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, world!'
  assert events[0].branch.endswith(agent.name)


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
  parent_ctx = _create_parent_invocation_context(
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
async def test_run_async_with_async_before_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
) -> Union[types.Content, None]:
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      before_agent_callback=_async_before_agent_callback_noop,
  )
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
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
async def test_run_async_after_agent_callback_noop(
    request: pytest.FixtureRequest,
    mocker: pytest_mock.MockerFixture,
):
  # Arrange
  agent = _TestingAgent(
      name=f'{request.function.__name__}_test_agent',
      after_agent_callback=_after_agent_callback_noop,
  )
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
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
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent
  )

  with pytest.raises(NotImplementedError):
    [e async for e in agent.run_async(parent_ctx)]


@pytest.mark.asyncio
async def test_run_live(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent
  )

  events = [e async for e in agent.run_live(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, live!'


@pytest.mark.asyncio
async def test_run_live_with_branch(request: pytest.FixtureRequest):
  agent = _TestingAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent, branch='parent_branch'
  )

  events = [e async for e in agent.run_live(parent_ctx)]

  assert len(events) == 1
  assert events[0].author == agent.name
  assert events[0].content.parts[0].text == 'Hello, live!'
  assert events[0].branch.endswith(agent.name)


@pytest.mark.asyncio
async def test_run_live_incomplete_agent(request: pytest.FixtureRequest):
  agent = _IncompleteAgent(name=f'{request.function.__name__}_test_agent')
  parent_ctx = _create_parent_invocation_context(
      request.function.__name__, agent
  )

  with pytest.raises(NotImplementedError):
    [e async for e in agent.run_live(parent_ctx)]


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

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

"""Tests for runner.edit_message_async."""

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest


class _EchoAgent(BaseAgent):
  """Yields the user prompt prefixed with echo:."""

  def __init__(self, name: str):
    super().__init__(name=name, sub_agents=[])

  async def _run_async_impl(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    text = ''
    user_content = invocation_context.user_content
    if user_content and user_content.parts:
      text = user_content.parts[0].text or ''
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=self.name,
        content=types.Content(
            role='model', parts=[types.Part.from_text(text=f'echo:{text}')]
        ),
    )


def _user_text(content: types.Content | None) -> str:
  if not content or not content.parts:
    return ''
  return content.parts[0].text or ''


async def _run_turn(
    runner: InMemoryRunner, user_id: str, session_id: str, text: str
) -> None:
  async for _ in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=types.Content(
          role='user', parts=[types.Part.from_text(text=text)]
      ),
  ):
    pass


class TestRunnerEditMessage:
  """Tests for runner.edit_message_async."""

  runner: InMemoryRunner

  def setup_method(self):
    self.runner = InMemoryRunner(agent=_EchoAgent(name='echo_agent'))

  async def test_edit_message_rewinds_and_regenerates_from_that_turn(self):
    """Editing a prior user turn rewinds later history and reruns the prompt."""
    runner = self.runner
    user_id = 'test_user'
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id
    )
    session_id = session.id

    await _run_turn(runner, user_id, session_id, 'first')
    await _run_turn(runner, user_id, session_id, 'second')

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    user_events = [event for event in session.events if event.author == 'user']
    assert len(user_events) == 2
    first_invocation_id = user_events[0].invocation_id

    regenerated = [
        event
        async for event in runner.edit_message_async(
            user_id=user_id,
            session_id=session_id,
            invocation_id=first_invocation_id,
            new_message=types.Content(
                role='user', parts=[types.Part.from_text(text='edited')]
            ),
        )
    ]
    assert regenerated
    assert any(
        event.author == 'echo_agent'
        and _user_text(event.content) == 'echo:edited'
        for event in regenerated
    )

    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    assert any(
        event.actions.rewind_before_invocation_id == first_invocation_id
        for event in session.events
    )
    user_texts = [
        _user_text(event.content)
        for event in session.events
        if event.author == 'user'
    ]
    assert user_texts[-1] == 'edited'

  async def test_edit_message_rejects_unknown_invocation(self):
    """Editing a missing invocation raises ValueError."""
    runner = self.runner
    user_id = 'test_user'
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id
    )

    with pytest.raises(ValueError, match='No user message found'):
      async for _ in runner.edit_message_async(
          user_id=user_id,
          session_id=session.id,
          invocation_id='missing-invocation',
          new_message=types.Content(
              role='user', parts=[types.Part.from_text(text='edited')]
          ),
      ):
        pass

  async def test_edit_message_rejects_agent_only_invocation(self):
    """Editing an invocation that has no user message raises ValueError."""
    runner = self.runner
    user_id = 'test_user'
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id
    )
    await runner.session_service.append_event(
        session=session,
        event=Event(
            invocation_id='agent-only',
            author='echo_agent',
            content=types.Content(
                role='model', parts=[types.Part.from_text(text='no user')]
            ),
        ),
    )

    with pytest.raises(ValueError, match='No user message found'):
      async for _ in runner.edit_message_async(
          user_id=user_id,
          session_id=session.id,
          invocation_id='agent-only',
          new_message=types.Content(
              role='user', parts=[types.Part.from_text(text='edited')]
          ),
      ):
        pass

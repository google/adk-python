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

"""Tests for ContextVar error handling during cancellation."""

import asyncio
import logging
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry.tracing import tracer
from google.genai import types
import pytest
from typing_extensions import override


class MinimalAgent(BaseAgent):

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text='Hello')]),
    )


class FailingCM:

  def __enter__(self):
    return mock.Mock()

  def __exit__(self, exc_type, exc_val, exc_tb):
    raise ValueError('Mocked ContextVar error')


@pytest.mark.asyncio
async def test_run_async_handles_context_var_error(
    caplog: pytest.LogCaptureFixture,
):
  agent = MinimalAgent(name='test_agent')

  with mock.patch.object(
      tracer, 'start_as_current_span', return_value=FailingCM()
  ):

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name='test', user_id='user'
    )
    ctx = InvocationContext(
        invocation_id='inv_id',
        agent=agent,
        session=session,
        session_service=session_service,
    )

    events = []
    with caplog.at_level(logging.WARNING):
      async for event in agent.run_async(ctx):
        events.append(event)

    assert len(events) == 1
    assert events[0].content.parts[0].text == 'Hello'

    assert any(
        'Failed to detach context during generator cleanup' in record.message
        for record in caplog.records
    )

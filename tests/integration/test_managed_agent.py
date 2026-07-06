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

"""Live integration tests for ManagedAgent.

Assumes tests/integration/.env is present (auto-loaded by conftest.py) and that
auth (ADC) is configured. Run explicitly:

  pytest tests/integration/test_managed_agent.py -v -s
"""

from __future__ import annotations

from google.adk.agents import ManagedAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools import google_search
from google.adk.utils.context_utils import Aclosing
from google.genai import types
import pytest

_AGENT_ID = 'antigravity-preview-05-2026'


async def _run_turn(runner, session, text: str) -> list:
  events = []
  async with Aclosing(
      runner.run_async(
          user_id='test_user',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part.from_text(text=text)]
          ),
      )
  ) as agen:
    async for event in agen:
      events.append(event)
  return events


def _joined_text(events) -> str:
  return ' '.join(
      part.text
      for e in events
      if e.content and e.content.parts
      for part in e.content.parts
      if part.text
  )


@pytest.mark.asyncio
async def test_google_search_project_hail_mary():
  agent = ManagedAgent(
      name='managed_search_agent',
      agent_id=_AGENT_ID,
      environment={'type': 'remote'},
      tools=[google_search],
  )
  session_service = InMemorySessionService()
  runner = Runner(
      app_name='managed_agent_it',
      agent=agent,
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name='managed_agent_it', user_id='test_user'
  )

  events = await _run_turn(
      runner, session, 'Who plays Rocky in the movie Project Hail Mary?'
  )

  answer = _joined_text(events)
  print('\n=== ManagedAgent answer ===\n', answer)
  assert (
      'james ortiz' in answer.lower()
  ), f'expected the grounded answer to contain "James Ortiz"; got: {answer!r}'

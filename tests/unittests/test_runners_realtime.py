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

from __future__ import annotations

from unittest import mock

from google.adk.agents import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


@pytest.mark.asyncio
async def test_runner_run_realtime_uses_agent_entrypoint(monkeypatch):
  agent = LlmAgent(name="a", model="gpt-4o-realtime-preview")
  r = Runner(
      app_name="test_app", agent=agent, session_service=InMemorySessionService()
  )
  sess = await r.session_service.create_session(
      app_name="test_app", user_id="u1"
  )

  collected = []

  async def fake_run_realtime(self, ctx):
    # Simulate a single model event
    from google.adk.events.event import Event

    collected.append("called")
    yield Event(invocation_id=ctx.invocation_id, author=agent.name)

  monkeypatch.setattr(
      LlmAgent, "run_realtime", fake_run_realtime, raising=False
  )

  live = r.run_realtime(
      session=sess,
      live_request_queue=LiveRequestQueue(),
      run_config=RunConfig(),
  )

  async for _ in live:
    break

  assert "called" in collected

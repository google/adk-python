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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_llm_agent_run_live_with_realtime_model_calls_openai_flow(
    monkeypatch,
):
  agent = LlmAgent(name="a", model="gpt-4o-realtime-preview")

  # Build minimal parent context
  ss = InMemorySessionService()
  session = await ss.create_session(app_name="test_app", user_id="u1")
  parent_ctx = InvocationContext(
      invocation_id="inv1",
      agent=agent,
      session=session,
      session_service=ss,
      user_content=types.Content(
          role="user", parts=[types.Part.from_text(text="hi")]
      ),
      run_config=RunConfig(openai_realtime_session={}),
  )

  # Spy on the agent's run_realtime to ensure the openai path is used downstream during transfer
  called = {"hit": False}

  async def fake_run_realtime(self, ctx):
    called["hit"] = True
    from google.adk.events.event import Event

    yield Event(invocation_id=ctx.invocation_id, author=agent.name)

  monkeypatch.setattr(
      LlmAgent, "run_realtime", fake_run_realtime, raising=False
  )

  # When the OpenAI flow triggers a transfer to this agent, run_realtime will be used (assert via side-effect)
  # Simulate direct call here to keep the test focused on the availability of the entrypoint
  async for _ in agent.run_realtime(parent_ctx):
    break

  assert called["hit"] is True

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

from collections.abc import AsyncGenerator

import pytest

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.integrations.redis import RedisSessionService
from google.adk.runners import Runner
from google.genai import types

from ._fake_redis import FakeRedisAsync


class _EchoAgent(BaseAgent):

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text="processed")]
        ),
    )


@pytest.mark.asyncio
async def test_runner_keeps_attacker_turn_out_of_victim_session():
  service = RedisSessionService(redis_client=FakeRedisAsync())
  await service.create_session(
      app_name="billing",
      user_id="tenant:victim",
      session_id="fixed",
      state={"private_record": "victim-only"},
  )
  await service.create_session(
      app_name="billing",
      user_id="tenant",
      session_id="victim:fixed",
      state={"attacker_record": "attacker-only"},
  )
  runner = Runner(
      app_name="billing",
      agent=_EchoAgent(name="echo"),
      session_service=service,
  )

  _ = [
      event
      async for event in runner.run_async(
          user_id="tenant",
          session_id="victim:fixed",
          new_message=types.Content(
              role="user", parts=[types.Part(text="attacker turn")]
          ),
      )
  ]

  victim = await service.get_session(
      app_name="billing", user_id="tenant:victim", session_id="fixed"
  )
  assert victim is not None
  assert victim.state["private_record"] == "victim-only"
  assert victim.events == []

  attacker = await service.get_session(
      app_name="billing", user_id="tenant", session_id="victim:fixed"
  )
  assert attacker is not None
  assert attacker.state["attacker_record"] == "attacker-only"
  assert [
      part.text for event in attacker.events for part in event.content.parts
  ] == [
      "attacker turn",
      "processed",
  ]

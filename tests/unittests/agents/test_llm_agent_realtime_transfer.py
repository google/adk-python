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

from contextlib import asynccontextmanager
from unittest import mock

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.openai_llm_flow import OpenAILlmFlow
from google.adk.flows.llm_flows.openai_llm_flow import OpenAutoFlow
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from .. import testing_utils


@pytest.mark.asyncio
async def test_realtime_subagent_transfer_path(monkeypatch):
  # Parent and child both realtime models
  parent = LlmAgent(name="parent", model="gpt-4o-realtime-preview")
  child = LlmAgent(name="child", model="gpt-4o-realtime-preview")
  parent.sub_agents = [child]

  # Patch OpenAI flow to simulate a function_response transfer from parent
  flow = OpenAutoFlow()
  fc = types.FunctionCall(
      name="transfer_to_agent", args={"agent_name": "child"}
  )
  events = [
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part(function_call=fc)]
          )
      )
  ]

  mock_llm = mock.Mock()

  class _Conn:

    async def receive(self):
      for e in events:
        yield e

    async def send_history(self, *a, **k):
      return

    async def send_content(self, *a, **k):
      return

    async def send_realtime(self, *a, **k):
      return

    async def close(self):
      return

  @asynccontextmanager
  async def fake_connect(_req):
    yield _Conn()

  mock_llm.connect = fake_connect
  monkeypatch.setattr(flow, "_BaseLlmFlow__get_llm", lambda ic: mock_llm)

  # Build a real invocation context to satisfy preprocessors
  ic = await testing_utils.create_invocation_context(
      agent=parent,
      user_content="",
      run_config=RunConfig(openai_realtime_session={}),
  )
  ic.live_request_queue = LiveRequestQueue()

  # Run and ensure no exceptions through transfer path
  events_out = []
  agen = flow.run_live(ic)  # type: ignore
  async for ev in agen:
    events_out.append(ev)

  assert any(e.get_function_responses() for e in events_out)

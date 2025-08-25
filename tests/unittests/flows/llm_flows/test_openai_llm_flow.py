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

import asyncio
from contextlib import asynccontextmanager
from unittest import mock

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.openai_llm_flow import OpenAILlmFlow
from google.adk.flows.llm_flows.openai_llm_flow import OpenAutoFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class _MockRealtimeConn:

  def __init__(self, responses: list[LlmResponse]):
    self._responses = responses
    self.closed = False

  async def send_history(self, history):
    return

  async def send_content(self, content):
    return

  async def send_realtime(self, blob):
    return

  async def receive(self):
    for r in self._responses:
      yield r

  async def close(self):
    self.closed = True


@pytest.mark.asyncio
async def test_openai_flow_emits_and_transfers_realtime(monkeypatch):
  # Build a parent agent with a realtime sub-agent
  parent = LlmAgent(name="parent", model="gpt-4o-realtime-preview")
  child = LlmAgent(name="child", model="gpt-4o-realtime-preview")
  parent.sub_agents = [child]

  # Simulate a function_call; the flow will handle tools and emit a function_response event
  fc = types.FunctionCall(
      name="transfer_to_agent", args={"agent_name": "child"}
  )
  llm_responses = [
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part(function_call=fc)]
          )
      ),
  ]

  # Provide a real async context manager for connect()
  mock_llm = mock.Mock()

  @asynccontextmanager
  async def fake_connect(_req):
    yield _MockRealtimeConn(llm_responses)

  mock_llm.connect = fake_connect

  # Patch flow to return the mock llm
  flow = OpenAutoFlow()
  monkeypatch.setattr(flow, "_BaseLlmFlow__get_llm", lambda ic: mock_llm)

  # Prepare invocation context with a real session to satisfy preprocessors
  q = LiveRequestQueue()
  ic = await testing_utils.create_invocation_context(
      agent=parent,
      user_content="",
      run_config=RunConfig(openai_realtime_session={}),
  )
  ic.live_request_queue = q

  # Execute and collect events (ensure no exception on transfer path)
  events = []
  agen = flow.run_live(ic)  # type: ignore
  async for ev in agen:
    events.append(ev)

  # We should have emitted at least the tool response event
  assert any(e.get_function_responses() for e in events)

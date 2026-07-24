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

"""Integration tests for LlmAgent output_key under streaming with tool calls."""

from __future__ import annotations

from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


def _text(text: str) -> types.Part:
  return types.Part.from_text(text=text)


def _call(name: str) -> types.Part:
  return types.Part(function_call=types.FunctionCall(name=name, args={}))


def _response(name: str) -> types.Part:
  return types.Part(
      function_response=types.FunctionResponse(name=name, response={"ok": True})
  )


def _event(
    parts: list[types.Part], *, role: str = "model", partial: bool = False
) -> Event:
  return Event(
      invocation_id="inv",
      author="agent",
      content=types.Content(role=role, parts=parts),
      actions=EventActions(),
      partial=partial,
  )


@pytest.mark.asyncio
async def test_run_async_accumulates_text_around_tool_calls():
  """Regression test for issue #5590.

  Under StreamingMode.SSE with tools, an LlmAgent emits text in several
  non-partial events: some carry text only, others carry text alongside a
  function_call. Event.is_final_response() returns False for any event
  with a function_call or function_response part, so the text on those
  events was historically dropped from output_key — only the final
  tool-free event's text was saved. Reporters measured ~60-70% loss.

  Drive _run_async_impl with a stubbed _llm_flow that yields the canned
  event sequence the streaming flow produces, merge each event's
  state_delta into session state via session_service.append_event, and
  assert that the user-visible session.state[output_key] contains every
  non-partial text segment the agent emitted, in order.
  """
  canned = [
      _event([_text("Intro one. ")], partial=True),
      _event([_text("Intro two.")], partial=True),
      _event([_text("Intro one. Intro two."), _call("t")]),
      _event([_response("t")], role="user"),
      _event([_text("Progress.")], partial=True),
      _event([_text("Progress."), _call("t")]),
      _event([_response("t")], role="user"),
      _event([_text("Conclusion one. ")], partial=True),
      _event([_text("Conclusion two.")], partial=True),
      _event([_text("Conclusion one. Conclusion two.")]),
  ]

  class _FakeFlow:

    async def run_async(
        self, _ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
      for event in canned:
        yield event

  agent = LlmAgent(name="agent", output_key="final_output")
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="t", user_id="u", session_id="s"
  )
  ctx = InvocationContext(
      invocation_id="inv",
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )

  with mock.patch.object(
      type(agent),
      "_llm_flow",
      new_callable=mock.PropertyMock,
      return_value=_FakeFlow(),
  ):
    async for event in agent._run_async_impl(ctx):
      await session_service.append_event(session, event)

  assert session.state["final_output"] == (
      "Intro one. Intro two.Progress.Conclusion one. Conclusion two."
  )

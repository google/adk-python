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

"""End-to-end test: Runner + event compaction.

Exercises the full ``runner.run_async`` path with a mock model, an in-memory
session service, and token-threshold event compaction.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

from .. import testing_utils


def _function_call_event(timestamp, invocation_id, call_id):
  return Event(
      timestamp=timestamp,
      invocation_id=invocation_id,
      author="agent",
      content=Content(
          role="model",
          parts=[
              Part(
                  function_call=types.FunctionCall(
                      id=call_id, name="tool", args={}
                  )
              )
          ],
      ),
  )


def _function_response_event(timestamp, invocation_id, call_id, tokens=None):
  usage = (
      types.GenerateContentResponseUsageMetadata(prompt_token_count=tokens)
      if tokens is not None
      else None
  )
  return Event(
      timestamp=timestamp,
      invocation_id=invocation_id,
      author="user",
      content=Content(
          role="user",
          parts=[
              Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name="tool", response={"result": "ok"}
                  )
              )
          ],
      ),
      usage_metadata=usage,
  )


@pytest.mark.asyncio
async def test_runner_compaction_does_not_break_execution():
  """Compaction must not orphan a function response, or the runner breaks.

  Mocks two tool calls with distinct ids: ``call-1`` is finished (it has a
  function_response) while ``call-2`` is still pending (no response). Because
  ``call-2`` sits between ``call-1``'s call and its response,
  compaction could summarize ``call-1``'s function_call while leaving its
  function_response behind -- an orphan with no matching call.

  Runs the full ``runner.run_async`` path with token-threshold compaction and
  checks that compaction works properly: it must keep every call together
  with its response. Otherwise, the prompt assembly will raise ``ValueError``
  ("No function call event found ...") and ``run_async`` will crash before
  the model is ever reached.
  """
  agent_model = testing_utils.MockModel.create(responses=["final answer"])
  agent = Agent(name="agent", model=agent_model)
  app = App(
      name="test_app",
      root_agent=agent,
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=10_000,
          overlap_size=0,
          token_threshold=1_000,
          event_retention_size=0,
          summarizer=LlmEventSummarizer(
              llm=testing_utils.MockModel.create(responses=["summary"])
          ),
      ),
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )
  events = [
      Event(
          timestamp=1.0,
          invocation_id="inv1",
          author="user",
          content=Content(role="user", parts=[Part(text="hello")]),
      ),
      _function_call_event(2.0, "inv2", "call-1"),
      _function_call_event(3.0, "inv3", "call-2"),  # stays pending
      # Last event and carries a high token count to trigger token compaction.
      _function_response_event(4.0, "inv3", "call-1", tokens=100_000),
  ]
  for event in events:
    await session_service.append_event(session=session, event=event)

  runner = Runner(app=app, session_service=session_service)

  # No new_message: just (re)process the existing session.
  # If we allow compaction to orphan the function response,
  # this will raise ValueError before the model is reached.
  produced = [
      event
      async for event in runner.run_async(
          user_id="u1", session_id="s1", new_message=None
      )
  ]

  # We got past request assembly and actually called the model.
  assert (
      agent_model.requests
  ), "model was never called; compaction orphaned the response"
  assert produced

  # call-1's function_call survived in the assembled prompt (not compacted).
  prompt_call_ids = []
  for content in agent_model.requests[-1].contents:
    for part in content.parts or []:
      if part.function_call is not None:
        prompt_call_ids.append(part.function_call.id)
  assert "call-1" in prompt_call_ids


@pytest.mark.asyncio
async def test_runner_appends_sliding_window_compaction_event():
  """The runner loop, not compaction, persists the sliding-window event.

  Compaction now yields its event and the runner is the single append site.
  With a sliding-window config (no token threshold), the full ``run_async``
  path must leave a compaction event persisted in the session -- proving the
  runner performed the append the compaction function no longer does.
  """
  agent_model = testing_utils.MockModel.create(responses=["final answer"])
  agent = Agent(name="agent", model=agent_model)
  app = App(
      name="test_app",
      root_agent=agent,
      events_compaction_config=EventsCompactionConfig(
          compaction_interval=2,
          overlap_size=0,
          summarizer=LlmEventSummarizer(
              llm=testing_utils.MockModel.create(responses=["summary"])
          ),
      ),
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )
  events = [
      Event(
          timestamp=1.0,
          invocation_id="inv1",
          author="user",
          content=Content(role="user", parts=[Part(text="hello")]),
      ),
      Event(
          timestamp=2.0,
          invocation_id="inv2",
          author="user",
          content=Content(role="user", parts=[Part(text="world")]),
      ),
  ]
  for event in events:
    await session_service.append_event(session=session, event=event)

  runner = Runner(app=app, session_service=session_service)
  async for _ in runner.run_async(
      user_id="u1", session_id="s1", new_message=None
  ):
    pass

  refreshed = await session_service.get_session(
      app_name="test_app", user_id="u1", session_id="s1"
  )
  compaction_events = [
      event for event in refreshed.events if event.actions.compaction
  ]
  assert (
      compaction_events
  ), "runner did not append the sliding-window compaction event"

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

"""Unit tests for google.adk.flows.llm_flows._content_compaction."""

from __future__ import annotations

from pathlib import Path

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context import _contents
from google.adk.flows.llm_flows.context._compaction import _process_compaction_events
from google.adk.flows.llm_flows.context._compaction import _recover_compacted_function_calls
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types
import pytest

from .... import testing_utils


def _long_running_call_event() -> Event:
  return Event(
      invocation_id="inv2",
      author="model",
      timestamp=2.0,
      long_running_tool_ids={"lr-1"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="lr-1", name="lr_tool", args={}
                  )
              )
          ],
      ),
  )


def _long_running_response_event(response: dict[str, str]) -> Event:
  return Event(
      invocation_id="inv2",
      author="user",
      timestamp=4.0,
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="lr-1", name="lr_tool", response=response
                  )
              )
          ],
      ),
  )


def test_process_compaction_events_basic():
  """Tests that events covered by compaction summary are filtered out."""
  compaction = EventCompaction(
      start_timestamp=1.0,
      end_timestamp=3.0,
      compacted_content=types.Content(
          role="model", parts=[types.Part(text="summary")]
      ),
  )
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          timestamp=1.0,
          content=types.UserContent("hello"),
      ),
      Event(
          invocation_id="inv1",
          author="model",
          timestamp=2.0,
          content=types.ModelContent("hi"),
      ),
      Event(
          invocation_id="compacted",
          author="model",
          timestamp=3.0,
          content=compaction.compacted_content,
          actions=EventActions(compaction=compaction),
      ),
      Event(
          invocation_id="inv2",
          author="user",
          timestamp=4.0,
          content=types.UserContent("next step"),
      ),
  ]

  result = _process_compaction_events(events, agent_name="model")

  assert len(result) == 2
  assert result[0].content.parts[0].text == "summary"
  assert result[1].timestamp == 4.0


def test_recover_compacted_function_calls_reinjects_missing_call():
  """A response whose call was compacted gets its call re-injected before it."""
  summary_event = Event(
      invocation_id="compacted",
      author="model",
      timestamp=3.0,
      content=types.Content(role="model", parts=[types.Part(text="summary")]),
  )
  call_event = _long_running_call_event()
  resume_response = _long_running_response_event({"result": "done"})

  effective = [summary_event, resume_response]
  source = [call_event, resume_response]

  result = _recover_compacted_function_calls(effective, source)

  assert result == [summary_event, call_event, resume_response]


def test_recover_compacted_function_calls_noop_when_call_present():
  """No change when every response already has its call in the list."""
  call_event = _long_running_call_event()
  resume_response = _long_running_response_event({"result": "done"})
  effective = [call_event, resume_response]

  result = _recover_compacted_function_calls(effective, effective)

  assert result is effective


def test_recover_compacted_function_calls_uses_latest_sibling_response():
  """A recovered sibling contributes its real result, not a stale placeholder."""

  def _response_event(
      call_id: str, response: dict[str, str], timestamp: float
  ) -> Event:
    return Event(
        invocation_id="inv2",
        author="user",
        timestamp=timestamp,
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id, name="lr_tool", response=response
                    )
                )
            ],
        ),
    )

  parallel_call = Event(
      invocation_id="inv2",
      author="model",
      timestamp=2.0,
      long_running_tool_ids={"lr-1", "lr-2"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="lr-1", name="lr_tool_1", args={}
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id="lr-2", name="lr_tool_2", args={}
                  )
              ),
          ],
      ),
  )
  lr2_placeholder = _response_event("lr-2", {"status": "pending"}, 3.0)
  lr2_result = _response_event("lr-2", {"result": "done-2"}, 4.0)
  summary_event = Event(
      invocation_id="compacted",
      author="model",
      timestamp=5.0,
      content=types.Content(role="model", parts=[types.Part(text="summary")]),
  )
  lr1_result = _response_event("lr-1", {"result": "done-1"}, 7.0)

  effective = [summary_event, lr1_result]
  source = [parallel_call, lr2_placeholder, lr2_result, lr1_result]

  result = _recover_compacted_function_calls(effective, source)

  assert result == [summary_event, parallel_call, lr2_result, lr1_result]
  assert result[2].get_function_responses()[0].response == {"result": "done-2"}


def _call(call_id: str, item: str, timestamp: float, invocation: str) -> Event:
  return Event(
      invocation_id=invocation,
      author="agent",
      timestamp=timestamp,
      long_running_tool_ids={call_id},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name="lookup", args={"item": item}
                  ),
                  thought_signature=b"call-signature",
              )
          ],
      ),
  )


def _response(
    call_id: str, value: str, timestamp: float, invocation: str
) -> Event:
  return Event(
      invocation_id=invocation,
      author="user",
      timestamp=timestamp,
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name="lookup", response={"result": value}
                  )
              )
          ],
      ),
  )


def _summary(start: float, end: float) -> Event:
  return Event(
      invocation_id="summary",
      author="agent",
      timestamp=end + 0.5,
      actions=EventActions(
          compaction=EventCompaction(
              start_timestamp=start,
              end_timestamp=end,
              compacted_content=types.ModelContent("summary"),
          )
      ),
  )


@pytest.mark.parametrize("reuse_id", [False, True])
@pytest.mark.parametrize("compaction_start", [None, 1.0, 3.0])
def test_resumed_response_keeps_its_call_after_compaction(
    reuse_id: bool, compaction_start: float | None
) -> None:
  """A reused call id does not pair the resumed result with an older call."""
  old_id = "shared" if reuse_id else "old"
  events = [
      _call(old_id, "old-item", 1.0, "old-invocation"),
      _response(old_id, "old-result", 2.0, "old-invocation"),
      _call("shared", "current-item", 3.0, "current-invocation"),
      _response("shared", "pending", 4.0, "current-invocation"),
  ]
  if compaction_start is not None:
    events.append(_summary(compaction_start, 4.0))
  events.append(
      _response("shared", "current-result", 6.0, "current-invocation")
  )
  original = [event.model_dump() for event in events]

  result = _contents._get_contents(
      None, events, agent_name="agent", preserve_function_call_ids=True
  )

  call_parts = [
      part
      for content in result
      for part in content.parts or []
      if part.function_call
  ]
  responses = [
      part.function_response
      for content in result
      for part in content.parts or []
      if part.function_response
  ]
  assert call_parts[-1].function_call is not None
  assert call_parts[-1].function_call.args == {"item": "current-item"}
  assert call_parts[-1].thought_signature == b"call-signature"
  assert responses[-1].response == {"result": "current-result"}
  assert [event.model_dump() for event in events] == original


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_runner_resumes_compacted_call_with_reused_id(
    backend: str, tmp_path: Path
) -> None:
  """Persisted resume history sends the current call's args to the model."""
  service = (
      InMemorySessionService()
      if backend == "memory"
      else SqliteSessionService(str(tmp_path / "session.db"))
  )
  executed = []

  def lookup(item: str) -> None:
    """Starts a lookup that will be answered later."""
    executed.append(item)

  model = testing_utils.MockModel.create(
      responses=[
          types.Part(
              function_call=types.FunctionCall(
                  id="shared", name="lookup", args={"item": "old-item"}
              )
          ),
          types.Part.from_text(text="old-ready"),
          types.Part(
              function_call=types.FunctionCall(
                  id="shared", name="lookup", args={"item": "current-item"}
              )
          ),
          types.Part.from_text(text="resumed"),
      ]
  )
  agent = LlmAgent(
      name="agent", model=model, tools=[LongRunningFunctionTool(lookup)]
  )
  runner = Runner(
      app=App(
          name="test_app",
          root_agent=agent,
          resumability_config=ResumabilityConfig(is_resumable=True),
      ),
      session_service=service,
  )
  try:
    session = await service.create_session(app_name="test_app", user_id="user")
    # Generate the history through real tool execution and a prior resume.
    for message in [
        types.UserContent("Start the old lookup"),
        _response("shared", "old-result", 2.0, "old-invocation").content,
        types.UserContent("Start the current lookup"),
    ]:
      async for _ in runner.run_async(
          user_id="user", session_id=session.id, new_message=message
      ):
        pass
    loaded_session = await service.get_session(
        app_name="test_app", user_id="user", session_id=session.id
    )
    assert loaded_session is not None
    session = loaded_session
    timestamps = [event.timestamp for event in session.events]
    await service.append_event(
        session=session,
        event=Event(
            author="agent",
            invocation_id="summary",
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=min(timestamps),
                    end_timestamp=max(timestamps),
                    compacted_content=types.ModelContent("summary"),
                )
            ),
        ),
    )
    before = [event.model_dump() for event in session.events]

    resumed = [
        event
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=_response(
                "shared", "current-result", 6.0, "current-invocation"
            ).content,
        )
    ]

    assert executed == ["old-item", "current-item"]
    assert len(model.requests) == 4
    parts = [
        p
        for content in model.requests[-1].contents
        for p in content.parts or []
    ]
    calls = [p.function_call for p in parts if p.function_call]
    responses = [p.function_response for p in parts if p.function_response]
    assert calls[-1].args == {"item": "current-item"}
    assert responses[-1].response == {"result": "current-result"}
    assert any(
        p.text == "resumed"
        for e in resumed
        for p in (e.content.parts if e.content else []) or []
    )
    saved = await service.get_session(
        app_name="test_app", user_id="user", session_id=session.id
    )
    assert saved is not None
    assert [
        event.model_dump() for event in saved.events[: len(before)]
    ] == before
  finally:
    await runner.close()
    if isinstance(service, SqliteSessionService):
      await service.close()


@pytest.mark.parametrize("reuse_sibling_id", [False, True])
def test_recovered_parallel_call_keeps_its_own_sibling_result(
    reuse_sibling_id: bool,
) -> None:
  """A compacted sibling's result must not come from a later unrelated call."""
  parallel_call = _call("pending", "pending-item", 1.0, "invocation")
  assert parallel_call.content and parallel_call.content.parts is not None
  parallel_call.content.parts.append(
      types.Part(
          function_call=types.FunctionCall(
              id="sibling", name="lookup", args={"item": "old-sibling-item"}
          )
      )
  )
  newer_id = "sibling" if reuse_sibling_id else "other"
  events = [
      parallel_call,
      _response("pending", "pending", 2.0, "invocation"),
      _response("sibling", "old-sibling-result", 3.0, "invocation"),
      _call(newer_id, "new-item", 4.0, "invocation"),
      _response(newer_id, "new-result", 5.0, "invocation"),
      _summary(1.0, 5.0),
      _response("pending", "done", 7.0, "invocation"),
  ]
  original = [event.model_dump() for event in events]

  result = _contents._get_contents(
      None, events, agent_name="agent", preserve_function_call_ids=True
  )

  responses = {
      part.function_response.id: part.function_response.response
      for content in result
      for part in content.parts or []
      if part.function_response
  }
  assert responses == {
      "pending": {"result": "done"},
      "sibling": {"result": "old-sibling-result"},
  }
  call_parts = [
      part
      for content in result
      for part in content.parts or []
      if part.function_call
  ]
  assert [
      part.function_call.args for part in call_parts if part.function_call
  ] == [
      {"item": "pending-item"},
      {"item": "old-sibling-item"},
  ]
  assert call_parts[0].thought_signature == b"call-signature"
  assert [event.model_dump() for event in events] == original

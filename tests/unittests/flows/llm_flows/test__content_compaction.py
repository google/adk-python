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

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows._content_compaction import _process_compaction_events
from google.adk.flows.llm_flows._content_compaction import _recover_compacted_function_calls
from google.genai import types


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

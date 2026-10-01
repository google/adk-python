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

from typing import Any
from typing import Optional

from google.adk.events._abort_events import _build_abort_events
from google.adk.events._abort_events import _is_abort_event
from google.adk.events.event import Event
from google.genai import types

_INVOCATION_ID = "inv_1"
_ABORT_MESSAGE = "Invocation was aborted by client."


def _fc_event(
    author: str,
    *call_ids: Optional[str],
    invocation_id: str = _INVOCATION_ID,
    **kwargs: Any,
) -> Event:
  return Event(
      invocation_id=invocation_id,
      author=author,
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name=f"tool_{call_id}", args={}
                  )
              )
              for call_id in call_ids
          ],
      ),
      **kwargs,
  )


def _fr_event(call_id: str, invocation_id: str = _INVOCATION_ID) -> Event:
  return Event(
      invocation_id=invocation_id,
      author="agent",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name=f"tool_{call_id}", response={}
                  )
              )
          ],
      ),
  )


def _build(events: list[Event], branch: Optional[str] = None) -> list[Event]:
  return _build_abort_events(
      events,
      invocation_id=_INVOCATION_ID,
      root_agent_name="root_agent",
      branch=branch,
  )


def _response_ids(events: list[Event]) -> list[list[str]]:
  return [[fr.id for fr in e.get_function_responses()] for e in events]


def test_dangling_call_is_sealed_with_error_response():
  """A dangling call gets an error response from its author, in the user role."""
  (event,) = _build([_fc_event("agent", "call_1")])

  assert event.invocation_id == _INVOCATION_ID
  assert event.author == "agent"
  assert event.content.role == "user"
  assert event.error_code == "INVOCATION_ABORTED"
  assert event.error_message == _ABORT_MESSAGE
  (fr,) = event.get_function_responses()
  assert (fr.id, fr.name) == ("call_1", "tool_call_1")
  assert fr.response == {"error": _ABORT_MESSAGE}


def test_answered_and_id_less_calls_are_skipped():
  """Calls that already have a response, or have no id to pair with, are left alone."""
  events = [_fc_event("agent", "call_1", "call_2", None), _fr_event("call_1")]

  assert _response_ids(_build(events)) == [["call_2"]]


def test_long_running_calls_are_sealed():
  """Pending long-running calls are sealed too, since their invocation is gone."""
  events = [_fc_event("agent", "call_1", long_running_tool_ids={"call_1"})]

  assert _response_ids(_build(events)) == [["call_1"]]


def test_only_events_of_the_invocation_are_considered():
  """Dangling calls from other invocations are not sealed."""
  events = [
      _fc_event("agent", "old_call", invocation_id="inv_0"),
      _fc_event("agent", "call_1"),
  ]

  assert _response_ids(_build(events)) == [["call_1"]]


def test_calls_are_grouped_by_author_branch_and_isolation_scope():
  """Each (author, branch, isolation_scope) gets its own response event."""
  events = [
      _fc_event("researcher", "call_a", branch="root.researcher"),
      _fc_event("researcher", "call_b", branch="root.researcher"),
      _fc_event("coder", "call_c", branch="root.coder"),
      _fc_event(
          "coder", "call_d", branch="root.coder", isolation_scope="task_1"
      ),
  ]

  result = _build(events)

  assert [(e.author, e.branch, e.isolation_scope) for e in result] == [
      ("researcher", "root.researcher", None),
      ("coder", "root.coder", None),
      ("coder", "root.coder", "task_1"),
  ]
  assert _response_ids(result) == [["call_a", "call_b"], ["call_c"], ["call_d"]]


def test_missing_branch_falls_back_to_invocation_branch():
  """A call event without a branch is answered on the invocation branch."""
  (event,) = _build([_fc_event("agent", "call_1")], branch="root")

  assert (event.author, event.branch) == ("agent", "root")


def test_root_agent_event_is_returned_when_nothing_is_dangling():
  """With nothing to answer, a single content-less event from the root agent is returned."""
  events = [_fc_event("agent", "call_1"), _fr_event("call_1")]

  (event,) = _build(events, branch="root")

  assert event.author == "root_agent"
  assert event.branch == "root"
  assert event.content is None
  assert event.error_code == "INVOCATION_ABORTED"
  assert event.error_message == _ABORT_MESSAGE


def test_built_events_are_recognized_as_abort_events():
  """Both kinds of built events carry the abort marker."""
  built = _build([]) + _build([_fc_event("agent", "call_1")])

  assert all(_is_abort_event(e) for e in built)


def test_other_error_events_are_not_abort_events():
  """Events with a different error code, or none, are not abort events."""
  assert not _is_abort_event(_fc_event("agent", "call_1"))
  assert not _is_abort_event(
      Event(author="agent", error_code="SAFETY", error_message="blocked")
  )


def test_paused_task_reply_at_end_is_not_sealed():
  """When the last event is a task agent's reply waiting for the user, nothing is sealed."""
  events = [
      _fc_event("coordinator", "call_1"),
      Event(
          invocation_id=_INVOCATION_ID,
          author="task_worker",
          isolation_scope="call_1",
          content=types.Content(
              role="model", parts=[types.Part(text="Which city?")]
          ),
      ),
  ]

  assert _build(events) == []

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

"""Unit tests for _tool_call_rearranger helper module."""

from __future__ import annotations

from typing import Any

from google.adk.events.event import Event
from google.adk.flows.llm_flows import _tool_call_rearranger
from google.adk.flows.llm_flows._tool_call_rearranger import drop_orphaned_function_calls
from google.adk.flows.llm_flows._tool_call_rearranger import drop_orphaned_function_responses
from google.adk.flows.llm_flows._tool_call_rearranger import merge_function_response_events
from google.adk.flows.llm_flows._tool_call_rearranger import rearrange_events_for_async_function_responses_in_history
from google.adk.flows.llm_flows._tool_call_rearranger import rearrange_events_for_latest_function_response
from google.genai import types
import pytest


def _call_event(call_id: str, name: str = "tool") -> Event:
  return Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name=name, args={}
                  )
              )
          ],
      ),
  )


def _resp_event(
    call_id: str | None, name: str = "tool", result: Any = "ok"
) -> Event:
  resp = result if isinstance(result, dict) else {"result": result}
  return Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name=name, response=resp
                  )
              )
          ],
      ),
  )


def test_drop_orphaned_responses_prunes_unpaired_and_preserves_valid():
  """Unpaired function response IDs are pruned while matched and ID-less responses survive."""
  call = _call_event("c1", "lookup")
  valid_resp = _resp_event("c1", "lookup", "found")
  no_id_resp = _resp_event(None, "legacy", "ok")
  orphan_resp = _resp_event("orphan_99", "ghost", "fail")
  events = [call, valid_resp, no_id_resp, orphan_resp]

  result = drop_orphaned_function_responses(events)

  assert result == [call, valid_resp, no_id_resp]


def test_drop_orphaned_responses_removes_event_when_all_parts_orphaned():
  """An event whose parts are all orphaned function responses is omitted completely."""
  call = _call_event("c1")
  orphan_event = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="o1", name="t1", response={}
                  )
              ),
              types.Part(
                  function_response=types.FunctionResponse(
                      id="o2", name="t2", response={}
                  )
              ),
          ],
      ),
  )
  events = [call, orphan_event]

  result = drop_orphaned_function_responses(events)

  assert result == [call]


def test_drop_orphaned_calls_prunes_unpaired_and_preserves_valid():
  """Unpaired function call IDs are pruned while matched and ID-less calls survive."""
  valid_call = _call_event("c1", "lookup")
  valid_resp = _resp_event("c1", "lookup", "found")
  no_id_call = Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[types.Part(function_call=types.FunctionCall(name="legacy"))],
      ),
  )
  orphan_call = _call_event("orphan_99", "ghost")
  events = [valid_call, valid_resp, no_id_call, orphan_call]

  result = drop_orphaned_function_calls(events)

  assert result == [valid_call, valid_resp, no_id_call]


def test_drop_orphaned_calls_removes_event_when_all_parts_orphaned():
  """An event whose parts are all orphaned function calls is omitted completely."""
  user_turn = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[types.Part(text="hello")],
      ),
  )
  orphan_event = Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(id="o1", name="t1", args={})
              ),
              types.Part(
                  function_call=types.FunctionCall(id="o2", name="t2", args={})
              ),
          ],
      ),
  )
  events = [user_turn, orphan_event]

  result = drop_orphaned_function_calls(events)

  assert result == [user_turn]


def test_drop_orphaned_calls_preserves_non_call_parts_in_mixed_event():
  """Non-call parts like text are preserved when an orphaned call is dropped."""
  model_event = Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(text="Checking inventory..."),
              types.Part(
                  function_call=types.FunctionCall(
                      id="c1", name="check_stock", args={}
                  )
              ),
          ],
      ),
  )
  user_event = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[types.Part(text="cancel")],
      ),
  )
  events = [model_event, user_event]

  result = drop_orphaned_function_calls(events)

  assert len(result) == 2
  assert result[0].content.parts == [types.Part(text="Checking inventory...")]
  assert result[0].get_function_calls() == []
  assert result[1] == user_event


def test_drop_orphaned_calls_prunes_unanswered_in_parallel_tool_calls():
  """Unanswered call in a parallel batch is dropped while answered call survives."""
  model_event = Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="c1", name="tool_1", args={}
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id="c2", name="tool_2", args={}
                  )
              ),
          ],
      ),
  )
  resp_1 = _resp_event("c1", "tool_1", "ok")
  events = [model_event, resp_1]

  result = drop_orphaned_function_calls(events)

  assert len(result) == 2
  calls = result[0].get_function_calls()
  assert len(calls) == 1
  assert calls[0].id == "c1"
  assert result[1] == resp_1


def test_drop_orphaned_calls_prevents_unclosed_tool_use_in_anthropic_conversion():
  """Pruned events converted for Anthropic contain no unclosed tool_use blocks."""
  from google.adk.models.anthropic_llm import content_to_message_param

  orphan_call = _call_event("fc_interrupted", "slow_tool")
  user_turn = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[types.Part(text="interrupted, do something else")],
      ),
  )

  pruned_events = drop_orphaned_function_calls([orphan_call, user_turn])

  messages = [
      content_to_message_param(e.content) for e in pruned_events if e.content
  ]
  assert len(messages) == 1
  assert messages[0]["role"] == "user"


def test_drop_orphaned_calls_preserves_pending_long_running_tool():
  """Calls marked in long_running_tool_ids are preserved even without response."""
  lr_call = Event(
      author="test_agent",
      long_running_tool_ids={"lr_1"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="lr_1", name="long_tool", args={}
                  )
              )
          ],
      ),
  )
  orphan_call = _call_event("orphan_99", "ghost")
  events = [lr_call, orphan_call]

  result = drop_orphaned_function_calls(events)

  assert result == [lr_call]


def test_drop_orphaned_calls_preserves_pending_auth_and_confirmation_calls():
  """Pending auth and confirmation calls with long_running_tool_ids are preserved."""
  auth_call = Event(
      author="test_agent",
      long_running_tool_ids={"auth_1"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="auth_1", name="request_euc", args={}
                  )
              )
          ],
      ),
  )
  confirm_call = Event(
      author="test_agent",
      long_running_tool_ids={"confirm_1"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="confirm_1", name="request_confirmation", args={}
                  )
              )
          ],
      ),
  )
  user_message = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[types.Part(text="interleaved message")],
      ),
  )
  events = [auth_call, confirm_call, user_message]

  result = drop_orphaned_function_calls(events)

  assert result == [auth_call, confirm_call, user_message]


def test_merge_function_response_events_updates_existing_and_appends_distinct():
  """Later responses for the same ID replace earlier parts; new IDs and text are appended."""
  event1 = _resp_event("c1", "t1", {"status": "pending"})
  event2 = Event(
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="c1", name="t1", response={"status": "done"}
                  )
              ),
              types.Part(
                  function_response=types.FunctionResponse(
                      id="c2", name="t2", response={"result": "ok"}
                  )
              ),
              types.Part(text="done note"),
          ],
      ),
  )

  merged = merge_function_response_events([event1, event2])

  responses = merged.get_function_responses()
  assert len(responses) == 2
  assert responses[0].response == {"status": "done"}
  assert responses[1].id == "c2"
  assert merged.content.parts[-1].text == "done note"


def test_merge_function_response_events_empty_input_raises_value_error():
  """Merging an empty event list or an event without parts raises ValueError."""
  with pytest.raises(ValueError, match="At least one function_response"):
    merge_function_response_events([])

  empty_part_event = Event(author="user", content=types.Content(parts=[]))
  with pytest.raises(ValueError, match="at least one function_response part"):
    merge_function_response_events([empty_part_event])


def test_rearrange_latest_response_moves_to_call_and_prunes_intervening():
  """Intervening turns are removed and intermediate responses merged next to the call."""
  call = _call_event("c1", "job")
  step1 = _resp_event("c1", "job", {"step": 1})
  intervening_msg = Event(
      author="user", content=types.UserContent("any updates?")
  )
  step2 = _resp_event("c1", "job", {"step": 2, "status": "finished"})
  events = [call, step1, intervening_msg, step2]

  result = rearrange_events_for_latest_function_response(events)

  assert len(result) == 2
  assert result[0] == call
  assert result[1].get_function_responses()[0].response == {
      "step": 2,
      "status": "finished",
  }


def test_rearrange_latest_response_missing_matching_call_raises_value_error():
  """A trailing response with no matching preceding call raises ValueError."""
  events = [
      Event(author="user", content=types.UserContent("hello")),
      _resp_event("missing_call_id"),
  ]

  with pytest.raises(ValueError, match="No function call event found"):
    rearrange_events_for_latest_function_response(events)


def test_rearrange_history_reused_id_across_tools_pairs_correctly():
  """Reused call IDs across different tools pair each tool with its own response."""
  events = [
      _call_event("call_807", "site_posture"),
      _resp_event("call_807", "site_posture", "site"),
      _call_event("call_807", "fleet_summary"),
      _resp_event("call_807", "fleet_summary", "fleet"),
  ]

  result = rearrange_events_for_async_function_responses_in_history(events)

  assert len(result) == 4
  assert result[0].get_function_calls()[0].name == "site_posture"
  assert result[1].get_function_responses()[0].name == "site_posture"
  assert result[2].get_function_calls()[0].name == "fleet_summary"
  assert result[3].get_function_responses()[0].name == "fleet_summary"


def test_rearrange_history_reused_id_same_tool_pairs_each_call():
  """Reused call IDs for the same tool pair each call with its respective response."""
  events = [
      _call_event("call_42", "lookup"),
      _resp_event("call_42", "lookup", "first"),
      _call_event("call_42", "lookup"),
      _resp_event("call_42", "lookup", "second"),
  ]

  result = rearrange_events_for_async_function_responses_in_history(events)

  assert len(result) == 4
  assert result[1].get_function_responses()[0].response == {"result": "first"}
  assert result[3].get_function_responses()[0].response == {"result": "second"}


def test_rearrange_history_reused_id_keeps_last_progress_update():
  """A tool reporting progress multiple times retains its last update before a new call."""
  events = [
      _call_event("call_7", "watch"),
      _resp_event("call_7", "watch", "progress"),
      _resp_event("call_7", "watch", "done"),
      _call_event("call_7", "watch"),
      _resp_event("call_7", "watch", "second_call"),
  ]

  result = rearrange_events_for_async_function_responses_in_history(events)

  assert len(result) == 4
  assert result[1].get_function_responses()[0].response == {"result": "done"}
  assert result[3].get_function_responses()[0].response == {
      "result": "second_call"
  }


def test_rearrange_history_async_parallel_responses_merged_next_to_call():
  """Parallel async responses arriving in separate events are merged next to their call."""
  parallel_call = Event(
      author="test_agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="c1", name="tool_a", args={}
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id="c2", name="tool_b", args={}
                  )
              ),
          ],
      ),
  )
  resp_c1 = _resp_event("c1", "tool_a", "res_a")
  intervening_user = Event(
      author="user", content=types.UserContent("any update?")
  )
  resp_c2 = _resp_event("c2", "tool_b", "res_b")
  events = [parallel_call, resp_c1, intervening_user, resp_c2]

  result = rearrange_events_for_async_function_responses_in_history(events)

  assert len(result) == 3
  assert result[0] == parallel_call
  merged_responses = result[1].get_function_responses()
  assert len(merged_responses) == 2
  assert {r.id for r in merged_responses} == {"c1", "c2"}
  assert result[2] == intervening_user


def test_backward_compatibility_aliases_exported():
  """Private leading-underscore aliases are exported for backward compatibility."""
  assert (
      _tool_call_rearranger._drop_orphaned_function_responses
      is drop_orphaned_function_responses
  )
  assert (
      _tool_call_rearranger._merge_function_response_events
      is merge_function_response_events
  )
  assert (
      _tool_call_rearranger._rearrange_events_for_async_function_responses_in_history
      is rearrange_events_for_async_function_responses_in_history
  )
  assert (
      _tool_call_rearranger._rearrange_events_for_latest_function_response
      is rearrange_events_for_latest_function_response
  )

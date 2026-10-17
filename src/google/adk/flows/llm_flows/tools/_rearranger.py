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

"""Tool call and response rearrangement logic for LLM request building."""

from __future__ import annotations

from bisect import bisect_left
import logging

from google.genai import types

from ....events.event import Event

logger = logging.getLogger('google_adk.' + __name__)


def merge_function_response_events(
    function_response_events: list[Event],
) -> Event:
  """Merges a list of function_response events into one event.

  The key goal is to ensure:
  1. function_call and function_response are always of the same number.
  2. The function_call and function_response are consecutively in the content.

  Args:
    function_response_events: A list of function_response events.
      NOTE: function_response_events must fulfill these requirements: 1. The
        list is in increasing order of timestamp; 2. the first event is the
        initial function_response event; 3. all later events should contain at
        least one function_response part that related to the function_call
        event.
      Caveat: This implementation doesn't support when a parallel function_call
        event contains async function_call of the same name.

  Returns:
    A merged event, that is
      1. All later function_response will replace function_response part in
          the initial function_response event.
      2. All non-function_response parts will be appended to the part list of
          the initial function_response event.
  """
  if not function_response_events:
    raise ValueError('At least one function_response event is required.')

  merged_event = function_response_events[0].model_copy(deep=True)
  merged_content = merged_event.content
  if merged_content is None or not merged_content.parts:
    raise ValueError('There should be at least one function_response part.')
  parts_in_merged_event = merged_content.parts

  # Function-response IDs are optional for legacy and long-running tools.  A
  # missing ID is therefore a valid correlation key, matching the historical
  # runtime behavior (with the same documented limitation for parallel calls
  # that cannot otherwise be distinguished).
  part_indices_in_merged_event: dict[str | None, int] = {}
  for idx, part in enumerate(parts_in_merged_event):
    if part.function_response:
      function_call_id = part.function_response.id
      part_indices_in_merged_event[function_call_id] = idx

  for event in function_response_events[1:]:
    event_content = event.content
    if event_content is None or not event_content.parts:
      raise ValueError('There should be at least one function_response part.')

    for part in event_content.parts:
      if part.function_response:
        function_call_id = part.function_response.id
        if function_call_id in part_indices_in_merged_event:
          parts_in_merged_event[
              part_indices_in_merged_event[function_call_id]
          ] = part
        else:
          parts_in_merged_event.append(part)
          part_indices_in_merged_event[function_call_id] = (
              len(parts_in_merged_event) - 1
          )

      else:
        parts_in_merged_event.append(part)

  return merged_event


def _is_stale_text_reply(event: Event, invocation_id: str) -> bool:
  return bool(
      event.invocation_id == invocation_id
      and event.author != 'user'
      and event.content
      and event.content.role == 'model'
      and not event.get_function_calls()
      and not (event.actions and event.actions.compaction)
      and any(part.text for part in event.content.parts or [])
  )


def rearrange_events_for_async_function_responses_in_history(
    events: list[Event],
) -> list[Event]:
  """Rearrange async function responses and their model replies in history."""
  # A model may hand out the same function call id more than once in a session,
  # so an id on its own does not identify a single call. Each response is
  # attributed to the newest call that precedes it and carries the same id, and
  # a call then takes the last response attributed to it. Taking the last one
  # keeps the closing update of a long-running tool, which reports progress
  # several times under one id, while attributing first stops a reused id from
  # handing a call the response that belongs to a different call.
  call_event_indices_by_id: dict[str | None, list[int]] = {}
  for i, event in enumerate(events):
    if event.get_function_responses():
      continue
    for function_call in event.get_function_calls():
      call_event_indices_by_id.setdefault(function_call.id, []).append(i)

  response_event_indices_by_call: dict[tuple[str | None, int], list[int]] = {}
  history_has_function_responses = False
  for i, event in enumerate(events):
    for function_response in event.get_function_responses():
      history_has_function_responses = True
      call_event_indices = call_event_indices_by_id.get(function_response.id)
      if not call_event_indices:
        continue
      # Indices are collected in ascending order, so the call that owns this
      # response is the one just before it. A response preceding every call
      # that carries its id keeps the first, as it did before ids could repeat.
      preceding_calls = bisect_left(call_event_indices, i)
      owning_call_event_index = call_event_indices[max(preceding_calls - 1, 0)]
      call_key = (function_response.id, owning_call_event_index)
      response_event_indices_by_call.setdefault(call_key, []).append(i)

  if not history_has_function_responses:
    return events

  # Drop intermediate model text replies between consecutive updates of the
  # same tool call within the same invocation.
  # Caveat: Positional; drops any intervening text replies during parallel calls.
  stale_model_event_indices: set[int] = set()
  for response_event_indices in response_event_indices_by_call.values():
    for cur_idx, next_idx in zip(
        response_event_indices, response_event_indices[1:]
    ):
      if events[cur_idx].invocation_id != events[next_idx].invocation_id:
        continue
      if events[cur_idx].author != events[next_idx].author:
        continue
      if events[cur_idx].actions and (
          events[cur_idx].actions.requested_tool_confirmations
          or events[cur_idx].actions.requested_auth_configs
      ):
        continue
      run = range(cur_idx + 1, next_idx)
      if run and all(
          _is_stale_text_reply(events[i], events[cur_idx].invocation_id)
          for i in run
      ):
        stale_model_event_indices.update(run)

  result_events: list[Event] = []
  for i, event in enumerate(events):
    if i in stale_model_event_indices:
      continue
    if event.get_function_responses():
      # function_response should be handled together with function_call below.
      continue
    elif event.get_function_calls():

      function_response_events_indices = set()
      for function_call in event.get_function_calls():
        response_indices = response_event_indices_by_call.get(
            (function_call.id, i)
        )
        if response_indices:
          function_response_events_indices.add(response_indices[-1])
      result_events.append(event)
      if not function_response_events_indices:
        continue
      if len(function_response_events_indices) == 1:
        result_events.append(
            events[next(iter(function_response_events_indices))]
        )
      else:  # Merge all async function_response as one response event
        result_events.append(
            merge_function_response_events(
                [events[i] for i in sorted(function_response_events_indices)]
            )
        )
      continue
    else:
      result_events.append(event)

  return result_events


def drop_orphaned_function_responses(
    events: list[Event],
) -> list[Event]:
  """Drops function_response parts that have no matching function_call.

  An orphan can reach this point when the producer of the call is gone, for
  example a session edited by hand or a history stitched together from more
  than one source. Left in place, the same orphan behaves differently
  depending on where it sits: mid-history it is quietly discarded, while as
  the trailing event it aborts the whole request. Pruning it here makes the
  outcome the same wherever it appears, and keeps unpaired results from being
  forwarded to providers that reject them.

  Responses without a preceding matching function_call are dropped as orphans.

  Args:
    events: The events being assembled into request contents.

  Returns:
    The events with orphaned function_response parts removed.
  """
  seen_call_ids: set[str] = set()
  seen_idless_call_names: set[str] = set()
  orphaned_ids: list[str] = []
  result_events: list[Event] = []
  for event in events:
    parts = event.content.parts if event.content else None
    if parts and event.get_function_responses():
      kept_parts: list[types.Part] = []
      for part in parts:
        response = part.function_response
        if response:
          is_matched = (
              response.id in seen_call_ids
              if response.id
              else (
                  bool(response.name)
                  and response.name in seen_idless_call_names
              )
          )
          if not is_matched:
            orphaned_ids.append(response.id or '<missing-id>')
            continue
        kept_parts.append(part)

      if kept_parts:
        if len(kept_parts) != len(parts):
          event = event.model_copy(deep=True)
          if event.content:
            event.content.parts = kept_parts
        result_events.append(event)
    else:
      result_events.append(event)

    for fc in event.get_function_calls():
      if fc.id:
        seen_call_ids.add(fc.id)
      elif fc.name:
        seen_idless_call_names.add(fc.name)

  if orphaned_ids:
    logger.warning(
        'Dropping function responses with no matching function call: %s',
        orphaned_ids,
    )

  return result_events


def _collect_function_response_ids(events: list[Event]) -> set[str]:
  """Returns the ids of every function response recorded in ``events``."""
  response_ids: set[str] = set()
  for event in events:
    for function_response in event.get_function_responses():
      if function_response.id:
        response_ids.add(function_response.id)
  return response_ids


def _collect_long_running_tool_ids(events: list[Event]) -> set[str]:
  """Returns the ids of all long-running tool calls marked in ``events``."""
  ids: set[str] = set()
  for event in events:
    if event.long_running_tool_ids:
      ids.update(event.long_running_tool_ids)
  return ids


def drop_orphaned_function_calls(
    events: list[Event],
) -> list[Event]:
  """Drops function_call parts that have no matching function_response.

  When a turn is interrupted (e.g. user abort, process restart, or follow-up
  user input prior to tool execution), unanswered function calls are pruned
  so downstream providers (Anthropic, OpenAI) do not reject the conversation
  history with HTTP 400 errors.

  Calls without an id are left alone: ids are stripped on the way out for
  some model families, so a missing id does not imply a missing response.

  Pending long-running tool calls (including auth and confirmation requests)
  marked in ``event.long_running_tool_ids`` are also left alone because they
  legitimately emit no response until resumed.

  Args:
    events: The events being assembled into request contents.

  Returns:
    The events with orphaned function_call parts removed.
  """
  response_ids = _collect_function_response_ids(events)
  long_running_ids = _collect_long_running_tool_ids(events)

  orphaned_ids: list[str] = []
  result_events: list[Event] = []
  for event in events:
    parts = event.content.parts if event.content else None
    if not parts or not event.get_function_calls():
      result_events.append(event)
      continue

    kept_parts: list[types.Part] = []
    for part in parts:
      call = part.function_call
      if (
          call
          and call.id
          and call.id not in response_ids
          and call.id not in long_running_ids
      ):
        orphaned_ids.append(call.id)
        continue
      kept_parts.append(part)

    if not kept_parts:
      continue
    if len(kept_parts) != len(parts):
      event = event.model_copy(deep=True)
      if event.content:
        event.content.parts = kept_parts
    result_events.append(event)

  if orphaned_ids:
    logger.warning(
        'Dropping function calls with no matching function response: %s',
        orphaned_ids,
    )

  return result_events


def _find_owning_call_event_index(
    history_events: list[Event],
    response: types.FunctionResponse,
) -> int:
  for idx in range(len(history_events) - 1, -1, -1):
    if any(
        (bool(response.id) and c.id == response.id)
        or (
            not response.id
            and not c.id
            and bool(response.name)
            and c.name == response.name
        )
        for c in history_events[idx].get_function_calls()
    ):
      return idx
  return -1


def rearrange_events_for_latest_function_response(
    events: list[Event],
) -> list[Event]:
  """Rearrange the events for the latest function_response.

  If the latest function_response is for an async function_call, all events
  between the initial function_call and the latest function_response will be
  removed.

  If the latest event carries function responses with no matching function
  call in history (an orphaned FR), those responses are dropped and history
  is rearranged from the remaining events.

  Args:
    events: A list of events.

  Returns:
    A list of events with the latest function_response rearranged.
  """
  events = drop_orphaned_function_responses(events)
  if len(events) < 2 or not events[-1].get_function_responses():
    return events

  trailing = events[-1]
  parts = trailing.content.parts if trailing.content else None
  if not parts:
    return events

  history_events = events[:-1]
  parts_by_call_idx: dict[int, list[types.Part]] = {}
  non_fr_parts: list[types.Part] = []
  for part in parts:
    if part.function_response:
      call_idx = _find_owning_call_event_index(
          history_events, part.function_response
      )
      if call_idx != -1:
        parts_by_call_idx.setdefault(call_idx, []).append(part)
    else:
      non_fr_parts.append(part)

  if not parts_by_call_idx:
    return events

  latest_call_idx = max(parts_by_call_idx.keys())
  parts_by_call_idx[latest_call_idx].extend(non_fr_parts)

  if latest_call_idx == len(events) - 2 and len(parts_by_call_idx) == 1:
    return events

  def _merged_response_for_call(call_idx: int) -> Event:
    split_event = trailing.model_copy(deep=True)
    if split_event.content:
      split_event.content.parts = parts_by_call_idx[call_idx]
    intermediate: list[Event] = []
    for ev_idx in range(call_idx + 1, len(events) - 1):
      ev = events[ev_idx]
      ev_parts = ev.content.parts if ev.content else None
      if not ev_parts or not ev.get_function_responses():
        continue
      is_call_event = bool(ev.get_function_calls())
      matched_parts = [
          p
          for p in ev_parts
          if (
              _find_owning_call_event_index(
                  events[:ev_idx], p.function_response
              )
              == call_idx
              if p.function_response
              else not is_call_event
          )
      ]
      if any(p.function_response for p in matched_parts):
        if len(matched_parts) != len(ev_parts):
          ev = ev.model_copy(deep=True)
          if ev.content:
            ev.content.parts = matched_parts
        intermediate.append(ev)
    all_resps = intermediate + [split_event]
    return (
        merge_function_response_events(all_resps)
        if len(all_resps) > 1
        else all_resps[0]
    )

  result_events: list[Event] = []
  for idx in range(latest_call_idx + 1):
    ev = events[idx]
    ev_parts = ev.content.parts if ev.content else None
    if ev_parts and ev.get_function_responses():
      kept_parts = [
          p
          for p in ev_parts
          if not p.function_response
          or _find_owning_call_event_index(events[:idx], p.function_response)
          not in parts_by_call_idx
      ]
      if not any(p.function_response or p.function_call for p in kept_parts):
        continue
      if len(kept_parts) != len(ev_parts):
        ev = ev.model_copy(deep=True)
        if ev.content:
          ev.content.parts = kept_parts
    result_events.append(ev)
    if idx in parts_by_call_idx:
      result_events.append(_merged_response_for_call(idx))

  return result_events


# Backward compatibility aliases
_merge_function_response_events = merge_function_response_events
_rearrange_events_for_async_function_responses_in_history = (
    rearrange_events_for_async_function_responses_in_history
)
_drop_orphaned_function_responses = drop_orphaned_function_responses
_rearrange_events_for_latest_function_response = (
    rearrange_events_for_latest_function_response
)

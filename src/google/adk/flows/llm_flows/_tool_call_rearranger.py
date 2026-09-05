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

from ...events.event import Event
from .functions import _collect_function_call_ids

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


def rearrange_events_for_async_function_responses_in_history(
    events: list[Event],
) -> list[Event]:
  """Rearrange the async function_response events in the history."""
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

  response_event_index_by_call: dict[tuple[str | None, int], int] = {}
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
      response_event_index_by_call[
          (function_response.id, owning_call_event_index)
      ] = i

  if not history_has_function_responses:
    return events

  result_events: list[Event] = []
  for i, event in enumerate(events):
    if event.get_function_responses():
      # function_response should be handled together with function_call below.
      continue
    elif event.get_function_calls():

      function_response_events_indices = set()
      for function_call in event.get_function_calls():
        response_event_index = response_event_index_by_call.get(
            (function_call.id, i)
        )
        if response_event_index is not None:
          function_response_events_indices.add(response_event_index)
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

  Responses without an id are left alone: ids are stripped on the way out for
  some model families, so a missing id does not imply a missing call.

  Args:
    events: The events being assembled into request contents.

  Returns:
    The events with orphaned function_response parts removed.
  """
  call_ids = _collect_function_call_ids(events)

  orphaned_ids: list[str] = []
  result_events: list[Event] = []
  for event in events:
    parts = event.content.parts if event.content else None
    if not parts or not event.get_function_responses():
      result_events.append(event)
      continue

    kept_parts: list[types.Part] = []
    for part in parts:
      response = part.function_response
      if response and response.id and response.id not in call_ids:
        orphaned_ids.append(response.id)
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
        'Dropping function responses with no matching function call: %s',
        orphaned_ids,
    )

  return result_events


def rearrange_events_for_latest_function_response(
    events: list[Event],
) -> list[Event]:
  """Rearrange the events for the latest function_response.

  If the latest function_response is for an async function_call, all events
  between the initial function_call and the latest function_response will be
  removed.

  Args:
    events: A list of events.

  Returns:
    A list of events with the latest function_response rearranged.
  """
  if len(events) < 2:
    # No need to process, since there is no function_call.
    return events

  function_responses = events[-1].get_function_responses()
  if not function_responses:
    # No need to process, since the latest event is not function_response.
    return events

  function_responses_ids = set()
  for function_response in function_responses:
    function_responses_ids.add(function_response.id)

  function_calls = events[-2].get_function_calls()

  if function_calls:
    for function_call in function_calls:
      # The latest function_response is already matched
      if function_call.id in function_responses_ids:
        return events

  function_call_event_idx = -1
  # look for corresponding function call event reversely
  for idx in range(len(events) - 2, -1, -1):
    event = events[idx]
    function_calls = event.get_function_calls()
    if function_calls:
      for function_call in function_calls:
        if function_call.id in function_responses_ids:
          function_call_event_idx = idx
          function_call_ids = {
              function_call.id for function_call in function_calls
          }
          # last response event should only contain the responses for the
          # function calls in the same function call event
          if not function_responses_ids.issubset(function_call_ids):
            raise ValueError(
                'Last response event should only contain the responses for the'
                ' function calls in the same function call event. Function'
                f' call ids found : {function_call_ids}, function response'
                f' ids provided: {function_responses_ids}'
            )
          # collect all function responses from the function call event to
          # the last response event
          function_responses_ids = function_call_ids
          break

  if function_call_event_idx == -1:
    logger.debug(
        'No function call event found for function responses ids: %s in'
        ' event list: %s',
        function_responses_ids,
        events,
    )
    raise ValueError(
        'No function call event found for function responses ids:'
        f' {function_responses_ids}'
    )

  # collect all function response between last function response event
  # and function call event

  function_response_events: list[Event] = []
  for idx in range(function_call_event_idx + 1, len(events) - 1):
    event = events[idx]
    function_responses = event.get_function_responses()
    if function_responses and any([
        function_response.id in function_responses_ids
        for function_response in function_responses
    ]):
      function_response_events.append(event)
  function_response_events.append(events[-1])

  result_events = events[: function_call_event_idx + 1]
  result_events.append(merge_function_response_events(function_response_events))

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

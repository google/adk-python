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


# Stands in for a result that was never recorded. It names no cause, because
# there is none to name: the turn may have been interrupted, or the process may
# have died between the two events.
_MISSING_FUNCTION_RESULT = 'No response available for this function call.'

# Stands in for a call ADK is deliberately holding open: a long-running tool, a
# human approval, or a request for user input. No function response exists for
# these until the answer arrives, so the call is pending rather than lost. The
# distinction changes what the model does next: told a tool returned nothing, it
# reissues the call or proceeds without it; told the call is still awaiting a
# response, it can wait.
_PENDING_FUNCTION_RESULT = (
    'This call is awaiting a response and has not completed yet.'
)


def pending_call_ids(events: list[Event]) -> set[str]:
  """Returns the ids of the calls ADK is holding open.

  ``long_running_tool_ids`` lives on the event and does not survive the
  conversion to contents, so it is read here.

  Args:
    events: The events being assembled into request contents.

  Returns:
    The ids of the calls that are awaiting a response.
  """
  pending: set[str] = set()
  for event in events:
    if event.long_running_tool_ids:
      pending.update(event.long_running_tool_ids)
  return pending


def _unanswered_calls(
    calls: list[types.FunctionCall],
    answered_ids: list[str | None],
) -> list[types.FunctionCall]:
  """Returns the calls that ``answered_ids`` does not account for.

  Ids are consumed one at a time rather than matched through a set, so a turn
  that carries several calls without an id (Gemini omits them, and ADK strips
  its own ``adk-`` ids before the request is built) does not have a single
  response silently answer all of them.

  Args:
    calls: The function calls in one content.
    answered_ids: The ids of the function responses that follow it, in order.

  Returns:
    The calls with no response of their own.
  """
  remaining = list(answered_ids)
  unanswered: list[types.FunctionCall] = []
  for call in calls:
    if call.id in remaining:
      remaining.remove(call.id)
      continue
    unanswered.append(call)
  return unanswered


def pair_unanswered_function_calls(
    contents: list[types.Content],
    pending_ids: set[str],
) -> list[types.Content]:
  """Gives every function call a function response in the next content.

  A call and its result are two separate session events. When a turn ends
  between them (a restart, an OOM kill, a disconnect, a cancellation), the
  session keeps a ``function_call`` that no ``function_response`` answers. That
  history is replayed on every later turn, and a provider that requires strict
  pairing then rejects the whole conversation: Anthropic answers ``tool_use ids
  were found without tool_result blocks immediately after``, and the session
  stays unusable until it is deleted.

  The repair runs on the request rather than the stored events, so recorded
  history stays intact and a session that is already broken heals on its next
  turn without a migration. It also sits above the session service, so it
  applies to every store.

  Pairing is checked positionally, against the immediately following content,
  because that is the invariant the provider enforces. A conversation whose
  calls are all answered is returned unchanged, and any conversation this does
  change is one the provider would have rejected.

  Args:
    contents: The contents built for the request. Not mutated in place, except
      that a placeholder is appended to the ``parts`` list of a content that
      already answers part of a turn.
    pending_ids: The ids of the calls ADK is holding open.

  Returns:
    The contents, with a placeholder response for every unanswered call.
  """
  paired: list[types.Content] = []
  synthesized = 0
  for index, content in enumerate(contents):
    paired.append(content)

    calls = [p.function_call for p in content.parts or [] if p.function_call]
    if not calls:
      continue

    following = contents[index + 1] if index + 1 < len(contents) else None
    answered_ids = [
        p.function_response.id
        for p in (following.parts or [] if following else [])
        if p.function_response
    ]
    unanswered = _unanswered_calls(calls, answered_ids)
    if not unanswered:
      continue
    synthesized += len(unanswered)

    parts = [
        types.Part(
            function_response=types.FunctionResponse(
                id=call.id,
                name=call.name,
                response={
                    'result': (
                        _PENDING_FUNCTION_RESULT
                        if call.id and call.id in pending_ids
                        else _MISSING_FUNCTION_RESULT
                    )
                },
            )
        )
        for call in unanswered
    ]

    # Join a turn that already answers this call event, so the results stay in
    # one message; otherwise the results need a turn of their own, before
    # whatever currently follows.
    if following is not None and answered_ids:
      following.parts = list(following.parts or []) + parts
    else:
      paired.append(types.Content(role='user', parts=parts))

  if synthesized:
    logger.info(
        'Paired %d unanswered function call(s) before the model request',
        synthesized,
    )
  return paired


# Backward compatibility aliases
_merge_function_response_events = merge_function_response_events
_rearrange_events_for_async_function_responses_in_history = (
    rearrange_events_for_async_function_responses_in_history
)
_drop_orphaned_function_responses = drop_orphaned_function_responses
_rearrange_events_for_latest_function_response = (
    rearrange_events_for_latest_function_response
)
_pending_call_ids = pending_call_ids
_pair_unanswered_function_calls = pair_unanswered_function_calls

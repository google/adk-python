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

"""Compaction request processor and event processing helpers for LLM flows."""

from __future__ import annotations

from bisect import bisect_right
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ....apps.compaction import _get_compaction_ranges
from ....apps.compaction import _has_token_threshold_config
from ....apps.compaction import _run_compaction_for_token_threshold_config
from ....events.event import Event
from .._base_llm_processor import BaseLlmRequestProcessor
from ..core._utils import require_agent

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext
  from ....models.llm_request import LlmRequest


class CompactionRequestProcessor(BaseLlmRequestProcessor):
  """Compacts session events before contents are prepared for model calls."""

  name = 'compaction'

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del llm_request
    config = invocation_context.events_compaction_config
    if not _has_token_threshold_config(config):
      return
      yield  # Required for AsyncGenerator.

    agent = require_agent(invocation_context)
    token_compacted = await _run_compaction_for_token_threshold_config(
        config=config,
        session=invocation_context.session,
        session_service=invocation_context.session_service,
        agent=agent,
        agent_name=agent.name,
        current_branch=invocation_context.branch,
        isolation_scope=invocation_context.isolation_scope,
    )
    if token_compacted:
      invocation_context.token_compaction_checked = True
    return
    yield  # Required for AsyncGenerator.


request_processor = CompactionRequestProcessor()


def _process_compaction_events(
    events: list[Event], agent_name: str = ''
) -> list[Event]:
  """Processes events by applying compaction.

  Identifies compacted ranges and filters out events that are covered by
  compaction summaries.

  Args:
    events: A list of events to process.
    agent_name: The name of the agent the history is being assembled for. The
      materialized summary is attributed to it so the agent reads its own
      compacted history as its own prior turns.

  Returns:
    A list of events with compaction applied.
  """
  # Example:
  # [event_1(ts=1), event_2(ts=2), compaction_1(1-2), event_3(ts=4),
  #  compaction_2(2-4), event_4(ts=6)].
  #
  # Overlaps are resolved by keeping only non-subsumed compaction summaries.
  # A summary event is materialized at its compaction end timestamp, and raw
  # events inside any kept compaction range are filtered out.
  compaction_infos = _get_compaction_ranges(events)
  retained_indexes = {index for index, _, _ in compaction_infos}
  # Union coverage for raw-event lookup, while keeping every non-subsumed
  # summary separately. Partially overlapping summaries must both survive.
  coverage: list[tuple[float, float]] = []
  for start, end in sorted(
      (start, end) for _, start, end in compaction_infos if start <= end
  ):
    if coverage and start <= coverage[-1][1]:
      coverage[-1] = (coverage[-1][0], max(coverage[-1][1], end))
    else:
      coverage.append((start, end))
  coverage_starts = [start for start, _ in coverage]
  processed_items: list[tuple[float, int, Event]] = []

  for i, event in enumerate(events):
    if event.actions and event.actions.compaction:
      if i not in retained_indexes:
        continue
      compaction = event.actions.compaction
      if (
          compaction.start_timestamp is None
          or compaction.end_timestamp is None
          or compaction.compacted_content is None
      ):
        continue
      processed_items.append((
          compaction.end_timestamp,
          i,
          Event(
              timestamp=compaction.end_timestamp,
              author=agent_name or 'model',
              content=compaction.compacted_content,
              branch=event.branch,
              isolation_scope=event.isolation_scope,
              invocation_id=event.invocation_id,
              actions=event.actions,
          ),
      ))

  def _is_timestamp_compacted(ts: float) -> bool:
    index = bisect_right(coverage_starts, ts) - 1
    return index >= 0 and ts <= coverage[index][1]

  for i, event in enumerate(events):
    if event.actions and event.actions.compaction:
      continue
    if _is_timestamp_compacted(event.timestamp):
      continue
    processed_items.append((event.timestamp, i, event))

  # Keep chronological order and a stable tie-breaker for equal timestamps.
  processed_items.sort(key=lambda item: (item[0], item[1]))
  return [event for _, _, event in processed_items]


def _recover_compacted_function_calls(
    events: list[Event],
    source_events: list[Event],
) -> list[Event]:
  """Re-injects function-call events that compaction removed.

  Compaction can summarize away a function_call while a matching
  function_response survives outside the compacted range. The clearest case
  is a long-running tool call: the call is compacted along with its
  intermediate placeholder response, then the real result arrives on resume
  (a later event not covered by the summary). That surviving response would
  be orphaned, which breaks call/response pairing during prompt assembly (it
  raises in `_rearrange_events_for_latest_function_response`).

  For each response whose call is no longer present, this restores the
  original call event from `source_events` (the pre-compaction list),
  inserting it immediately before the first surviving response that
  references it. The whole call event is re-injected verbatim (rather than
  trimmed to the resumed call) so parallel-call thought signatures, which only
  the first part carries, are preserved. Any sibling responses that compaction
  removed are re-injected too, so a sibling is not surfaced as a phantom
  pending call.

  Call ids can be reused. Ownership is resolved in source order before any
  events are restored: a response belongs to the latest preceding call with
  its id, even when another call with that id survived compaction.

  Args:
    events: The post-compaction events being assembled into request contents.
      Must contain the same Event objects as `source_events`; ownership is
      matched by identity, not by value.
    source_events: The pre-compaction events to recover missing calls from.

  Returns:
    `events` with any recoverable missing function-call events (and their
    compacted sibling responses) re-injected; the original list is returned
    unchanged when nothing needs recovery.
  """
  if not any(event.get_function_responses() for event in events):
    return events

  present_events = {id(event) for event in events}
  latest_call_by_id: dict[str, Event] = {}
  call_by_response: dict[tuple[int, str], Event] = {}
  response_by_call: dict[tuple[int, str], Event] = {}
  surviving_responses: set[tuple[int, str]] = set()
  for event in source_events:
    for function_call in event.get_function_calls():
      if function_call.id:
        latest_call_by_id[function_call.id] = event
    for function_response in event.get_function_responses():
      response_id = function_response.id
      if not response_id:
        continue
      call_event = latest_call_by_id.get(response_id)
      if call_event is None:
        continue
      call_by_response[id(event), response_id] = call_event
      call_key = (id(call_event), response_id)
      if id(event) in present_events:
        surviving_responses.add(call_key)
      # Keep the latest update for this occurrence, not a later call that
      # happens to reuse the id. Equal timestamps retain source order.
      existing = response_by_call.get(call_key)
      if existing is None or event.timestamp >= existing.timestamp:
        response_by_call[call_key] = event

  result: list[Event] = []
  reinjected_calls: set[int] = set()
  reinjected_responses: set[int] = set()
  for event in events:
    for function_response in event.get_function_responses():
      function_response_id = function_response.id
      if not function_response_id:
        continue
      call_event = call_by_response.get((id(event), function_response_id))
      if (
          call_event is None
          or id(call_event) in present_events
          or id(call_event) in reinjected_calls
      ):
        continue
      result.append(call_event)
      reinjected_calls.add(id(call_event))
      # Recover sibling responses that compaction removed so a parallel sibling
      # is not left looking like a pending call.
      for sibling_call in call_event.get_function_calls():
        if not sibling_call.id:
          continue
        sibling_key = (id(call_event), sibling_call.id)
        if sibling_key not in surviving_responses:
          sibling_response = response_by_call.get(sibling_key)
          if (
              sibling_response is not None
              and id(sibling_response) not in reinjected_responses
          ):
            result.append(sibling_response)
            reinjected_responses.add(id(sibling_response))
    result.append(event)
  return result if reinjected_calls else events


__all__ = [
    'CompactionRequestProcessor',
    '_process_compaction_events',
    '_recover_compacted_function_calls',
    'request_processor',
]

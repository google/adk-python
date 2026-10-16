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

  Args:
    events: The post-compaction events being assembled into request contents.
    source_events: The pre-compaction events to recover missing calls from.

  Returns:
    `events` with any recoverable missing function-call events (and their
    compacted sibling responses) re-injected; the original list is returned
    unchanged when nothing needs recovery.
  """
  call_ids_present: set[str] = set()
  response_ids_present: set[str] = set()
  for event in events:
    for function_call in event.get_function_calls():
      if function_call.id:
        call_ids_present.add(function_call.id)
    for function_response in event.get_function_responses():
      if function_response.id:
        response_ids_present.add(function_response.id)

  orphaned_ids = {
      response_id
      for response_id in response_ids_present
      if response_id not in call_ids_present
  }
  if not orphaned_ids:
    return events

  call_event_by_id: dict[str, Event] = {}
  for event in source_events:
    for function_call in event.get_function_calls():
      if function_call.id in orphaned_ids:
        call_event_by_id.setdefault(function_call.id, event)

  if not call_event_by_id:
    return events

  # Keep the highest-timestamp response per id so a sibling that completed
  # before being compacted contributes its real result, not its stale
  # placeholder; ties fall back to source order.
  response_event_by_id: dict[str, Event] = {}
  for event in source_events:
    for function_response in event.get_function_responses():
      if not function_response.id:
        continue
      existing = response_event_by_id.get(function_response.id)
      if existing is None or event.timestamp >= existing.timestamp:
        response_event_by_id[function_response.id] = event

  result: list[Event] = []
  reinjected_ids: set[str] = set()
  for event in events:
    for function_response in event.get_function_responses():
      function_response_id = function_response.id
      if not function_response_id:
        continue
      call_event = call_event_by_id.get(function_response_id)
      if call_event is None or function_response_id in reinjected_ids:
        continue
      result.append(call_event)
      sibling_ids = [
          function_call.id
          for function_call in call_event.get_function_calls()
          if function_call.id
      ]
      reinjected_ids.update(sibling_ids)
      # Recover sibling responses that compaction removed so a parallel sibling
      # is not left looking like a pending call.
      for sibling_id in sibling_ids:
        if sibling_id not in response_ids_present:
          sibling_response = response_event_by_id.get(sibling_id)
          if sibling_response is not None:
            result.append(sibling_response)
    result.append(event)
  return result


__all__ = [
    'CompactionRequestProcessor',
    '_process_compaction_events',
    '_recover_compacted_function_calls',
    'request_processor',
]

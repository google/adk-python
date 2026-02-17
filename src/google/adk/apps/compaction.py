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

from __future__ import annotations

import logging

from ..events.event import Event
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from .app import App
from .llm_event_summarizer import LlmEventSummarizer

logger = logging.getLogger('google_adk.' + __name__)


def _count_text_chars_in_event(event: Event) -> int:
  """Returns the number of text characters in an event's content."""
  total_chars = 0
  if event.content and event.content.parts:
    for part in event.content.parts:
      if part.text:
        total_chars += len(part.text)
  return total_chars


def _is_compaction_subsumed(
    *,
    start_timestamp: float,
    end_timestamp: float,
    event_index: int,
    compactions: list[tuple[int, float, float, Event]],
) -> bool:
  """Returns True if a compaction range is fully contained by another.

  If two compactions have identical ranges, the earlier event is treated as
  subsumed by the later event.
  """
  for other_index, other_start, other_end, _ in compactions:
    if other_index == event_index:
      continue
    if other_start <= start_timestamp and other_end >= end_timestamp:
      if (
          other_start < start_timestamp
          or other_end > end_timestamp
          or other_index > event_index
      ):
        return True
  return False


def _estimate_prompt_token_count(events: list[Event]) -> int | None:
  """Returns an approximate prompt token count from session events.

  This estimate is compaction-aware: it counts compaction summaries and only
  counts raw events that would remain visible after applying compaction ranges.
  """
  compactions: list[tuple[int, float, float, Event]] = []
  for i, event in enumerate(events):
    if not (event.actions and event.actions.compaction):
      continue
    compaction = event.actions.compaction
    if (
        compaction.start_timestamp is None
        or compaction.end_timestamp is None
        or compaction.compacted_content is None
    ):
      continue
    compactions.append((
        i,
        compaction.start_timestamp,
        compaction.end_timestamp,
        Event(
            timestamp=compaction.end_timestamp,
            author='model',
            content=compaction.compacted_content,
            branch=event.branch,
            invocation_id=event.invocation_id,
            actions=event.actions,
        ),
    ))

  effective_compactions = [
      (i, start, end, summary_event)
      for i, start, end, summary_event in compactions
      if not _is_compaction_subsumed(
          start_timestamp=start,
          end_timestamp=end,
          event_index=i,
          compactions=compactions,
      )
  ]
  compaction_ranges = [
      (start, end) for _, start, end, _ in effective_compactions
  ]

  def _is_timestamp_compacted(ts: float) -> bool:
    for start_ts, end_ts in compaction_ranges:
      if start_ts <= ts <= end_ts:
        return True
    return False

  total_chars = 0
  for _, _, _, summary_event in effective_compactions:
    total_chars += _count_text_chars_in_event(summary_event)

  for event in events:
    if event.actions and event.actions.compaction:
      continue
    if _is_timestamp_compacted(event.timestamp):
      continue
    total_chars += _count_text_chars_in_event(event)

  if total_chars <= 0:
    return None

  # Rough estimate: 4 characters per token.
  return total_chars // 4


def _latest_prompt_token_count(events: list[Event]) -> int | None:
  """Returns the most recently observed prompt token count, if available."""
  for event in reversed(events):
    if (
        event.usage_metadata
        and event.usage_metadata.prompt_token_count is not None
    ):
      return event.usage_metadata.prompt_token_count
  return _estimate_prompt_token_count(events)


def _latest_compaction_event(events: list[Event]) -> Event | None:
  """Returns the compaction event with the greatest covered end timestamp."""
  latest_event = None
  latest_end = 0.0
  for event in events:
    if (
        event.actions
        and event.actions.compaction
        and event.actions.compaction.end_timestamp is not None
    ):
      end_ts = event.actions.compaction.end_timestamp
      if end_ts is not None and end_ts >= latest_end:
        latest_end = end_ts
        latest_event = event
  return latest_event


def _latest_compaction_end_timestamp(events: list[Event]) -> float:
  """Returns the end timestamp of the most recent compaction event."""
  latest_event = _latest_compaction_event(events)
  if not latest_event or not latest_event.actions.compaction:
    return 0.0
  if latest_event.actions.compaction.end_timestamp is None:
    return 0.0
  return latest_event.actions.compaction.end_timestamp


async def _run_compaction_for_token_threshold(
    app: App, session: Session, session_service: BaseSessionService
):
  """Runs post-invocation compaction based on a token threshold.

  If triggered, this compacts older raw events and keeps the last
  `event_retention_size` raw events un-compacted.
  """
  config = app.events_compaction_config
  if not config:
    return False
  if config.token_threshold is None or config.event_retention_size is None:
    return False

  prompt_token_count = _latest_prompt_token_count(session.events)
  if prompt_token_count is None or prompt_token_count < config.token_threshold:
    return False

  latest_compaction_event = _latest_compaction_event(session.events)
  last_compacted_end_timestamp = 0.0
  if (
      latest_compaction_event
      and latest_compaction_event.actions
      and latest_compaction_event.actions.compaction
      and latest_compaction_event.actions.compaction.end_timestamp is not None
  ):
    last_compacted_end_timestamp = (
        latest_compaction_event.actions.compaction.end_timestamp
    )
  candidate_events = [
      e
      for e in session.events
      if not (e.actions and e.actions.compaction)
      and e.timestamp > last_compacted_end_timestamp
  ]

  if len(candidate_events) <= config.event_retention_size:
    return False

  if config.event_retention_size == 0:
    events_to_compact = candidate_events
  else:
    events_to_compact = candidate_events[: -config.event_retention_size]
  if not events_to_compact:
    return False

  # Rolling summary: if a previous compaction exists, seed the next summary with
  # the previous compaction summary content so new compactions can subsume older
  # ones while still keeping `event_retention_size` raw events visible.
  if (
      latest_compaction_event
      and latest_compaction_event.actions
      and latest_compaction_event.actions.compaction
      and latest_compaction_event.actions.compaction.start_timestamp is not None
      and latest_compaction_event.actions.compaction.compacted_content
      is not None
  ):
    seed_event = Event(
        timestamp=latest_compaction_event.actions.compaction.start_timestamp,
        author='model',
        content=latest_compaction_event.actions.compaction.compacted_content,
        branch=latest_compaction_event.branch,
        invocation_id=Event.new_id(),
    )
    events_to_compact = [seed_event] + events_to_compact

  if not config.summarizer:
    config.summarizer = LlmEventSummarizer(llm=app.root_agent.canonical_model)

  compaction_event = await config.summarizer.maybe_summarize_events(
      events=events_to_compact
  )
  if compaction_event:
    await session_service.append_event(session=session, event=compaction_event)
    logger.debug('Token-threshold event compactor finished.')
    return True
  return False


async def _run_compaction_for_sliding_window(
    app: App, session: Session, session_service: BaseSessionService
):
  """Runs compaction for SlidingWindowCompactor.

  This method implements the sliding window compaction logic. It determines
  if enough new invocations have occurred since the last compaction based on
  `compaction_invocation_threshold`. If so, it selects a range of events to
  compact based on `overlap_size`, and calls `maybe_compact_events` on the
  compactor.

  The compaction process is controlled by two parameters:
  1.  `compaction_invocation_threshold`: The number of *new* user-initiated
  invocations that, once fully
      represented in the session's events, will trigger a compaction.
  2.  `overlap_size`: The number of preceding invocations to include from the
  end of the last
      compacted range. This creates an overlap between consecutive compacted
      summaries,
      maintaining context.

  The compactor is called after an agent has finished processing a turn and all
  its events
  have been added to the session. It checks if a new compaction is needed.

  When a compaction is triggered:
  -   The compactor identifies the range of `invocation_id`s to be summarized.
  -   This range starts `overlap_size` invocations before the beginning of the
      new block of `compaction_invocation_threshold` invocations and ends
      with the last
      invocation
      in the current block.
  -   A `CompactedEvent` is created, summarizing all events within this
  determined
      `invocation_id` range. This `CompactedEvent` is then appended to the
      session.

  Here is an example with `compaction_invocation_threshold = 2` and
  `overlap_size = 1`:
  Let's assume events are added for `invocation_id`s 1, 2, 3, and 4 in order.

  1.  **After `invocation_id` 2 events are added:**
      -   The session now contains events for invocations 1 and 2. This
      fulfills the `compaction_invocation_threshold = 2` criteria.
      -   Since this is the first compaction, the range starts from the
      beginning.
      -   A `CompactedEvent` is generated, summarizing events within
      `invocation_id` range [1, 2].
      -   The session now contains: `[
          E(inv=1, role=user), E(inv=1, role=model),
          E(inv=2, role=user), E(inv=2, role=model),
          CompactedEvent(inv=[1, 2])]`.

  2.  **After `invocation_id` 3 events are added:**
      -   No compaction happens yet, because only 1 new invocation (`inv=3`)
      has been completed since the last compaction, and
      `compaction_invocation_threshold` is 2.

  3.  **After `invocation_id` 4 events are added:**
      -   The session now contains new events for invocations 3 and 4, again
      fulfilling `compaction_invocation_threshold = 2`.
      -   The last `CompactedEvent` covered up to `invocation_id` 2. With
      `overlap_size = 1`, the new compaction range
          will start one invocation before the new block (inv 3), which is
          `invocation_id` 2.
      -   The new compaction range is from `invocation_id` 2 to 4.
      -   A new `CompactedEvent` is generated, summarizing events within
      `invocation_id` range [2, 4].
      -   The session now contains: `[
          E(inv=1, role=user), E(inv=1, role=model),
          E(inv=2, role=user), E(inv=2, role=model),
          CompactedEvent(inv=[1, 2]),
          E(inv=3, role=user), E(inv=3, role=model),
          E(inv=4, role=user), E(inv=4, role=model),
          CompactedEvent(inv=[2, 4])]`.


  Args:
    app: The application instance.
    session: The session containing events to compact.
    session_service: The session service for appending events.
  """
  events = session.events
  if not events:
    return None

  # Prefer token-threshold compaction if configured and triggered.
  if (
      app.events_compaction_config
      and app.events_compaction_config.token_threshold is not None
  ):
    token_compacted = await _run_compaction_for_token_threshold(
        app, session, session_service
    )
    if token_compacted:
      return None

  # Find the last compaction event and its range.
  last_compacted_end_timestamp = 0.0
  for event in reversed(events):
    if (
        event.actions
        and event.actions.compaction
        and event.actions.compaction.end_timestamp
    ):
      last_compacted_end_timestamp = event.actions.compaction.end_timestamp
      break

  # Get unique invocation IDs and their latest timestamps.
  invocation_latest_timestamps = {}
  for event in events:
    # Only consider non-compaction events for unique invocation IDs.
    if event.invocation_id and not (event.actions and event.actions.compaction):
      invocation_latest_timestamps[event.invocation_id] = max(
          invocation_latest_timestamps.get(event.invocation_id, 0.0),
          event.timestamp,
      )

  unique_invocation_ids = list(invocation_latest_timestamps.keys())

  # Determine which invocations are new since the last compaction.
  new_invocation_ids = [
      inv_id
      for inv_id in unique_invocation_ids
      if invocation_latest_timestamps[inv_id] > last_compacted_end_timestamp
  ]

  if len(new_invocation_ids) < app.events_compaction_config.compaction_interval:
    return None  # Not enough new invocations to trigger compaction.

  # Determine the range of invocations to compact.
  # The end of the compaction range is the last of the new invocations.
  end_inv_id = new_invocation_ids[-1]

  # The start of the compaction range is overlap_size invocations before
  # the first of the new invocations.
  first_new_inv_id = new_invocation_ids[0]
  first_new_inv_idx = unique_invocation_ids.index(first_new_inv_id)

  start_idx = max(
      0, first_new_inv_idx - app.events_compaction_config.overlap_size
  )
  start_inv_id = unique_invocation_ids[start_idx]

  # Find the index of the last event with end_inv_id.
  last_event_idx = -1
  for i in range(len(events) - 1, -1, -1):
    if events[i].invocation_id == end_inv_id:
      last_event_idx = i
      break

  events_to_compact = []
  # Trim events_to_compact to include all events up to and including the
  # last event of end_inv_id.
  if last_event_idx != -1:
    # Find the index of the first event of start_inv_id in events.
    first_event_start_inv_idx = -1
    for i, event in enumerate(events):
      if event.invocation_id == start_inv_id:
        first_event_start_inv_idx = i
        break
    if first_event_start_inv_idx != -1:
      events_to_compact = events[first_event_start_inv_idx : last_event_idx + 1]
      # Filter out any existing compaction events from the list.
      events_to_compact = [
          e
          for e in events_to_compact
          if not (e.actions and e.actions.compaction)
      ]

  if not events_to_compact:
    return None

  if not app.events_compaction_config.summarizer:
    app.events_compaction_config.summarizer = LlmEventSummarizer(
        llm=app.root_agent.canonical_model
    )

  compaction_event = (
      await app.events_compaction_config.summarizer.maybe_summarize_events(
          events=events_to_compact
      )
  )
  if compaction_event:
    await session_service.append_event(session=session, event=compaction_event)
  logger.debug('Event compactor finished.')

# Copyright 2025 Google LLC
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

from google.adk.apps.app import App
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


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
      -   The session now contains: `[E(inv=1, role=user), E(inv=1, role=model),
      E(inv=2, role=user), E(inv=2, role=model), E(inv=2, role=user),
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
      -   The session now contains: `[E(inv=1, role=user), E(inv=1, role=model),
      E(inv=2, role=user), E(inv=2, role=model), E(inv=2, role=user),
      CompactedEvent(inv=[1, 2]), E(inv=3, role=user), E(inv=3, role=model),
      E(inv=4, role=user), E(inv=4, role=model), CompactedEvent(inv=[2, 4])]`.


  Args:
    app: The application instance.
    session: The session containing events to compact.
    session_service: The session service for appending events.
  """
  events = session.events
  if not events:
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

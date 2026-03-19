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

from google.genai import types

from ..agents.base_agent import BaseAgent
from ..events.event import Event
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from .app import App
from .app import EventsCompactionConfig
from .llm_event_summarizer import LlmEventSummarizer

logger = logging.getLogger('google_adk.' + __name__)


def _count_text_chars_in_content(content: types.Content | None) -> int:
  """Returns the number of text characters in a content object."""
  total_chars = 0
  if content and content.parts:
    for part in content.parts:
      if part.text:
        total_chars += len(part.text)
  return total_chars


def _valid_compactions(
    events: list[Event],
) -> list[tuple[int, float, float, Event]]:
  """Returns compaction events with fully-defined compaction ranges."""
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
        event,
    ))
  return compactions


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


def _estimate_prompt_token_count(
    *,
    events: list[Event],
    current_branch: str | None,
    agent_name: str,
) -> int | None:
  """Returns an approximate prompt token count from session events.

  This estimate mirrors the effective content-building path used by the
  contents request processor.
  """
  # Deferred import: contents depends on agents.invocation_context which
  # imports from apps, so a top-level import would create a circular dependency.
  from ..flows.llm_flows import contents

  effective_contents = contents._get_contents(
      current_branch=current_branch,
      events=events,
      agent_name=agent_name,
  )
  total_chars = 0
  for content in effective_contents:
    total_chars += _count_text_chars_in_content(content)

  if total_chars <= 0:
    return None

  # Rough estimate: 4 characters per token.
  return total_chars // 4


def _latest_prompt_token_count(
    events: list[Event],
    *,
    current_branch: str | None = None,
    agent_name: str = '',
) -> int | None:
  """Returns the most recently observed prompt token count, if available."""
  for event in reversed(events):
    if (
        event.usage_metadata
        and event.usage_metadata.prompt_token_count is not None
    ):
      return event.usage_metadata.prompt_token_count
  return _estimate_prompt_token_count(
      events=events,
      current_branch=current_branch,
      agent_name=agent_name,
  )


def _latest_compaction_event(events: list[Event]) -> Event | None:
  """Returns the latest non-subsumed compaction event by stream order."""
  compactions = _valid_compactions(events)
  latest_event = None
  latest_index = -1
  for event_index, start_ts, end_ts, event in compactions:
    if _is_compaction_subsumed(
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        event_index=event_index,
        compactions=compactions,
    ):
      continue
    if event_index > latest_index:
      latest_index = event_index
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


def _has_token_threshold_config(config: EventsCompactionConfig | None) -> bool:
  """Returns whether token-threshold compaction is fully configured."""
  return bool(
      config
      and config.token_threshold is not None
      and config.event_retention_size is not None
  )


def _has_sliding_window_config(config: EventsCompactionConfig | None) -> bool:
  """Returns whether sliding-window compaction is fully configured."""
  return bool(
      config
      and config.compaction_interval is not None
      and config.overlap_size is not None
  )


def _ensure_compaction_summarizer(
    *, config: EventsCompactionConfig, agent: BaseAgent
) -> None:
  """Ensures compaction config has a summarizer initialized."""
  if config.summarizer is not None:
    return

  from ..agents.llm_agent import LlmAgent

  if not isinstance(agent, LlmAgent):
    raise ValueError(
        'No LlmAgent model available for event compaction summarizer.'
    )
  config.summarizer = LlmEventSummarizer(llm=agent.canonical_model)


def _events_to_compact_for_token_threshold(
    *,
    events: list[Event],
    event_retention_size: int,
) -> list[Event]:
  """Collects token-threshold compaction candidates with rolling-summary seed.

  If a previous compaction exists, include its summary as the first event so
  the next summary can supersede it.
  """
  latest_compaction_event = _latest_compaction_event(events)
  last_compacted_end_timestamp = _latest_compaction_end_timestamp(events)

  candidate_events = [
      event
      for event in events
      if not (event.actions and event.actions.compaction)
      and event.timestamp > last_compacted_end_timestamp
  ]
  if len(candidate_events) <= event_retention_size:
    return []

  if event_retention_size == 0:
    events_to_compact = candidate_events
  else:
    split_index = _safe_token_compaction_split_index(
        candidate_events=candidate_events,
        event_retention_size=event_retention_size,
    )
    events_to_compact = candidate_events[:split_index]
  if not events_to_compact:
    return []

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
    return [seed_event] + events_to_compact

  return events_to_compact


def _event_function_call_ids(event: Event) -> set[str]:
  """Returns function call ids found in an event."""
  function_call_ids: set[str] = set()
  for function_call in event.get_function_calls():
    if function_call.id:
      function_call_ids.add(function_call.id)
  return function_call_ids


def _event_function_response_ids(event: Event) -> set[str]:
  """Returns function response ids found in an event."""
  function_response_ids: set[str] = set()
  for function_response in event.get_function_responses():
    if function_response.id:
      function_response_ids.add(function_response.id)
  return function_response_ids


def _safe_token_compaction_split_index(
    *,
    candidate_events: list[Event],
    event_retention_size: int,
) -> int:
  """Returns a split index that avoids orphaning retained tool responses.

  Retained events (tail of candidate events) may contain function responses.
  If their matching function call events are in the compacted prefix, contents
  assembly can fail. This method shifts the split earlier so matching function
  call events are retained together with their responses.

  Iterates backwards through candidate_events once, maintaining a running set
  of unmatched response IDs. The latest valid split point where no unmatched
  responses remain is returned.
  """
  initial_split = len(candidate_events) - event_retention_size
  if initial_split <= 0:
    return 0

  unmatched_response_ids: set[str] = set()
  best_split = 0

  for i in range(len(candidate_events) - 1, -1, -1):
    event = candidate_events[i]
    unmatched_response_ids.update(_event_function_response_ids(event))
    call_ids = _event_function_call_ids(event)
    unmatched_response_ids -= call_ids

    if not unmatched_response_ids and i <= initial_split:
      best_split = i
      break

  return best_split


async def _run_compaction_for_token_threshold_config(
    *,
    config: EventsCompactionConfig | None,
    session: Session,
    session_service: BaseSessionService,
    agent: BaseAgent,
    agent_name: str = '',
    current_branch: str | None = None,
) -> bool:
  """Runs token-threshold compaction for a provided compaction config."""
  if not _has_token_threshold_config(config):
    return False
  if config is None:
    return False

  if config.token_threshold is None or config.event_retention_size is None:
    return False

  prompt_token_count = _latest_prompt_token_count(
      session.events,
      current_branch=current_branch,
      agent_name=agent_name,
  )
  if prompt_token_count is None or prompt_token_count < config.token_threshold:
    return False

  events_to_compact = _events_to_compact_for_token_threshold(
      events=session.events,
      event_retention_size=config.event_retention_size,
  )
  if not events_to_compact:
    return False

  _ensure_compaction_summarizer(config=config, agent=agent)
  if config.summarizer is None:
    return False

  compaction_event = await config.summarizer.maybe_summarize_events(
      events=events_to_compact
  )
  if compaction_event:
    await session_service.append_event(session=session, event=compaction_event)
    logger.debug('Token-threshold event compactor finished.')
    return True
  return False


async def _run_compaction_for_token_threshold(
    app: App, session: Session, session_service: BaseSessionService
):
  """Runs post-invocation compaction based on a token threshold.

  If triggered, this compacts older raw events and keeps the last
  `event_retention_size` raw events un-compacted.
  """
  return await _run_compaction_for_token_threshold_config(
      config=app.events_compaction_config,
      session=session,
      session_service=session_service,
      agent=app.root_agent,
      agent_name='',
      current_branch=None,
  )


async def _run_compaction_for_sliding_window(
    app: App,
    session: Session,
    session_service: BaseSessionService,
    *,
    skip_token_compaction: bool = False,
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
    skip_token_compaction: Whether to skip token-threshold compaction.
  """
  events = session.events
  if not events:
    return None

  config = app.events_compaction_config
  if config is None:
    return None

  # Prefer token-threshold compaction if configured and triggered.
  if not skip_token_compaction and _has_token_threshold_config(config):
    token_compacted = await _run_compaction_for_token_threshold(
        app, session, session_service
    )
    if token_compacted:
      return None

  if not _has_sliding_window_config(config):
    return None

  if config.compaction_interval is None or config.overlap_size is None:
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

  if len(new_invocation_ids) < config.compaction_interval:
    return None  # Not enough new invocations to trigger compaction.

  # Determine the range of invocations to compact.
  # The end of the compaction range is the last of the new invocations.
  end_inv_id = new_invocation_ids[-1]

  # The start of the compaction range is overlap_size invocations before
  # the first of the new invocations.
  first_new_inv_id = new_invocation_ids[0]
  first_new_inv_idx = unique_invocation_ids.index(first_new_inv_id)

  start_idx = max(0, first_new_inv_idx - config.overlap_size)
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

  _ensure_compaction_summarizer(config=config, agent=app.root_agent)
  if config.summarizer is None:
    return None

  compaction_event = await config.summarizer.maybe_summarize_events(
      events=events_to_compact
  )
  if compaction_event:
    await session_service.append_event(session=session, event=compaction_event)
  logger.debug('Event compactor finished.')

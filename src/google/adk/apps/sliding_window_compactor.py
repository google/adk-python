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

from typing import Optional

from google.genai import types
from google.genai.types import Content
from google.genai.types import Part

from ..events.event import Event
from ..events.event_actions import EventActions
from ..events.event_actions import EventCompaction
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from .base_events_compactor import BaseEventsCompactor


class SlidingWindowCompactor(BaseEventsCompactor):
  """A summarizer for Sliding Window Compaction logic in Runner.

  This compactor works with ADK runner to provide sliding window compaction.
  The runner uses `compaction_invocation_threshold` and `overlap_size`
  configured in `EventsCompactionConfig` on the `App` to determine when to
  trigger compaction and which events to compact. This class performs
  summarization of events passed by the runner.

  The compaction process is controlled by two parameters read by the Runner from
  `EventsCompactionConfig`:
  1.  `compaction_invocation_threshold`: The number of *new* user-initiated
  invocations that, once fully
      represented in the session's events, will trigger a compaction.
  2.  `overlap_size`: The number of preceding invocations to include from the
  end of the last
      compacted range. This creates an overlap between consecutive compacted
      summaries,
      maintaining context.

  When `Runner` determines compaction is needed based on
  `compaction_invocation_threshold`,
  it selects a range of events based on `overlap_size` and passes them to
  `maybe_compact_events` for summarization into a `CompactedEvent`.
  This `CompactedEvent` is then appended to the session by the `Runner`.
  """

  _DEFAULT_PROMPT_TEMPLATE = (
      'The following is a conversation history between a user and an AI'
      ' agent. Please summarize the conversation, focusing on key'
      ' information and decisions made, as well as any unresolved'
      ' questions or tasks. The summary should be concise and capture the'
      ' essence of the interaction. Each event is prefixed with the'
      ' author.\\n\\n{conversation_history}'
  )

  def __init__(
      self,
      llm: BaseLlm,
      prompt_template: Optional[str] = None,
  ):
    """Initializes the SlidingWindowCompactor.

    Args:
        llm: The LLM used for summarization.
        prompt_template: An optional template string for the summarization
          prompt. If not provided, a default template will be used. The template
          should contain a '{conversation_history}' placeholder.
    """
    self._llm = llm
    self._prompt_template = prompt_template or self._DEFAULT_PROMPT_TEMPLATE

  def _format_events_for_prompt(self, events: list[Event]) -> str:
    """Formats a list of events into a string for the LLM prompt."""
    formatted_history = []
    for event in events:
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            formatted_history.append(f'{event.author}: {part.text}')
    return '\\n'.join(formatted_history)

  async def maybe_compact_events(
      self, *, events: list[Event]
  ) -> Optional[Event]:
    """Compacts given events and returns the compacted content.

    Args:
      events: A list of events to compact.

    Returns:
      The new compacted event, or None if no compaction is needed.
    """
    if not events:
      return None

    conversation_history = self._format_events_for_prompt(events)
    prompt = self._prompt_template.format(
        conversation_history=conversation_history
    )

    llm_request = LlmRequest(
        model=self._llm.model,
        contents=[Content(role='user', parts=[Part(text=prompt)])],
    )
    summary_content = None
    async for llm_response in self._llm.generate_content_async(
        llm_request, stream=False
    ):
      if llm_response.content:
        summary_content = llm_response.content
        break

    if summary_content is None:
      return None

    # Ensure the compacted content has the role 'user'
    summary_content.role = 'user'

    start_timestamp = events[0].timestamp
    end_timestamp = events[-1].timestamp

    compaction = EventCompaction(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        compacted_content=summary_content,
    )

    actions = EventActions(compaction=compaction)

    return Event(
        author='user',
        actions=actions,
        invocation_id=Event.new_id(),
    )

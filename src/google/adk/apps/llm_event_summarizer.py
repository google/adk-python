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

from typing import Optional

from google.genai import types
from google.genai.types import Content
from google.genai.types import Part

from ..apps.base_events_summarizer import BaseEventsSummarizer
from ..events.event import Event
from ..events.event_actions import EventActions
from ..events.event_actions import EventCompaction
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest


class LlmEventSummarizer(BaseEventsSummarizer):
  """An LLM-based event summarizer for sliding window compaction.

  This class is responsible for summarizing a provided list of events into a
  single compacted event. It is designed to be used as part of a sliding window
  compaction process.

  The actual logic for determining *when* to trigger compaction and *which*
  events form the sliding window (based on parameters like
  `compaction_invocation_threshold` and `overlap_size` from
  `EventsCompactionConfig`) is handled by an external component, such as an ADK
  "Runner". This compactor focuses solely on generating a summary of the events
  it receives.

  When `maybe_compact_events` is called with a list of events, this class
  formats the events, generates a summary using an LLM, and returns a new
  `Event` containing the summary within an `EventCompaction`.
  """

  _DEFAULT_PROMPT_TEMPLATE = (
      'The following is a conversation history between a user and an AI agent.'
      ' It may or may not start from a compacted history. Please identify and'
      ' reiterate the user request, summarize the context so far, focusing on'
      ' key decisions made and information obtained, as well as any unresolved'
      ' questions or tasks. The summary should be concise and capture the'
      ' essence of the interaction.\n\n{conversation_history}'
  )

  # Tool call args and responses can be large (e.g. search results). Cap how
  # much of each is rendered so compaction does not inflate the very context
  # it exists to shrink.
  _MAX_TOOL_CONTENT_CHARS = 2000

  def __init__(
      self,
      llm: BaseLlm,
      prompt_template: Optional[str] = None,
  ):
    """Initializes the LlmEventSummarizer.

    Args:
        llm: The LLM used for summarization.
        prompt_template: An optional template string for the summarization
          prompt. If not provided, a default template will be used. The template
          should contain a '{conversation_history}' placeholder.
    """
    self._llm = llm
    self._prompt_template = prompt_template or self._DEFAULT_PROMPT_TEMPLATE

  def _format_events_for_prompt(self, events: list[Event]) -> str:
    """Formats events into prompt text, including thoughts and tool calls.

    Thoughts carry the agent's analysis of tool responses, and tool calls and
    responses carry the evidence retrieved so far, so all three are included.
    Thoughts emitted by a compaction event are skipped so a prior summary's
    reasoning does not leak into the next summary.
    """
    formatted_history = []
    for event in events:
      if not (event.content and event.content.parts):
        continue
      is_compaction = bool(event.actions and event.actions.compaction)
      for part in event.content.parts:
        if part.thought and part.text:
          if not is_compaction:
            formatted_history.append(f'{event.author} (thought): {part.text}')
        elif part.text:
          formatted_history.append(f'{event.author}: {part.text}')
        if part.function_call:
          args = self._truncate(str(part.function_call.args))
          formatted_history.append(
              f'{event.author} called tool: {part.function_call.name}({args})'
          )
        if part.function_response:
          response = self._truncate(str(part.function_response.response))
          formatted_history.append(
              f'Tool response from {part.function_response.name}: {response}'
          )
    return '\n'.join(formatted_history)

  def _truncate(self, text: str) -> str:
    """Caps `text` at the tool-content limit, marking dropped characters."""
    limit = self._MAX_TOOL_CONTENT_CHARS
    if len(text) <= limit:
      return text
    return f'{text[:limit]}... [truncated {len(text) - limit} chars]'

  async def maybe_summarize_events(
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
    summary_usage_metadata = None
    async for llm_response in self._llm.generate_content_async(
        llm_request, stream=False
    ):
      if llm_response.content:
        summary_content = llm_response.content
        summary_usage_metadata = llm_response.usage_metadata
        break

    if summary_content is None:
      return None

    # Ensure the compacted content has the role 'model'
    summary_content.role = 'model'

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
        usage_metadata=summary_usage_metadata,
    )

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

import copy
from typing import AsyncGenerator
from typing import Optional

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import remove_client_function_call_id
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from .functions import REQUEST_EUC_FUNCTION_CALL_NAME


class _ContentLlmRequestProcessor(BaseLlmRequestProcessor):
  """Builds the contents for the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if agent.include_contents == 'default':
      # Include full conversation history
      llm_request.contents = _get_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
      )
    else:
      # Include current turn context only (no conversation history)
      llm_request.contents = _get_current_turn_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
      )

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _ContentLlmRequestProcessor()


def _rearrange_events_for_async_function_responses_in_history(
    events: list[Event],
) -> list[Event]:
  """Rearrange the async function_response events in the history."""

  function_call_id_to_response_events_index: dict[str, list[Event]] = {}
  for i, event in enumerate(events):
    function_responses = event.get_function_responses()
    if function_responses:
      for function_response in function_responses:
        function_call_id = function_response.id
        function_call_id_to_response_events_index[function_call_id] = i

  result_events: list[Event] = []
  for event in events:
    if event.get_function_responses():
      # function_response should be handled together with function_call below.
      continue
    elif event.get_function_calls():

      function_response_events_indices = set()
      for function_call in event.get_function_calls():
        function_call_id = function_call.id
        if function_call_id in function_call_id_to_response_events_index:
          function_response_events_indices.add(
              function_call_id_to_response_events_index[function_call_id]
          )
      result_events.append(event)
      if not function_response_events_indices:
        continue
      if len(function_response_events_indices) == 1:
        result_events.append(
            events[next(iter(function_response_events_indices))]
        )
      else:  # Merge all async function_response as one response event
        result_events.append(
            _merge_function_response_events(
                [events[i] for i in sorted(function_response_events_indices)]
            )
        )
      continue
    else:
      result_events.append(event)

  return result_events


def _rearrange_events_for_latest_function_response(
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
  if not events:
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
  result_events.append(
      _merge_function_response_events(function_response_events)
  )

  return result_events


def _contains_empty_content(event: Event) -> bool:
  """Check if an event should be skipped due to missing or empty content.

  This can happen to the evnets that only changed session state.
  When both content and transcriptions are empty, the event will be considered as empty.

  Args:
    event: The event to check.

  Returns:
    True if the event should be skipped, False otherwise.
  """
  return (
      not event.content
      or not event.content.role
      or not event.content.parts
      or event.content.parts[0].text == ''
  ) and (not event.output_transcription and not event.input_transcription)


def _get_contents(
    current_branch: Optional[str], events: list[Event], agent_name: str = ''
) -> list[types.Content]:
  """Get the contents for the LLM request.

  Applies filtering, rearrangement, and content processing to events.

  Args:
    current_branch: The current branch of the agent.
    events: Events to process.
    agent_name: The name of the agent.

  Returns:
    A list of processed contents.
  """
  accumulated_input_transcription = ''
  accumulated_output_transcription = ''

  # Parse the events, leaving the contents and the function calls and
  # responses from the current agent.
  raw_filtered_events = []
  for event in events:
    if _contains_empty_content(event):
      continue
    if not _is_event_belongs_to_branch(current_branch, event):
      # Skip events not belong to current branch.
      continue
    if _is_auth_event(event):
      # Skip auth events.
      continue
    if _is_request_confirmation_event(event):
      # Skip request confirmation events.
      continue

    raw_filtered_events.append(event)

  filtered_events = []
  # aggregate transcription events
  for i in range(len(raw_filtered_events)):
    event = raw_filtered_events[i]
    if not event.content:
      # Convert transcription into normal event
      if event.input_transcription and event.input_transcription.text:
        accumulated_input_transcription += event.input_transcription.text
        if (
            i != len(raw_filtered_events) - 1
            and raw_filtered_events[i + 1].input_transcription
            and raw_filtered_events[i + 1].input_transcription.text
        ):
          continue
        event = event.model_copy(deep=True)
        event.input_transcription = None
        event.content = types.Content(
            role='user',
            parts=[types.Part(text=accumulated_input_transcription)],
        )
        accumulated_input_transcription = ''
      elif event.output_transcription and event.output_transcription.text:
        accumulated_output_transcription += event.output_transcription.text
        if (
            i != len(raw_filtered_events) - 1
            and raw_filtered_events[i + 1].output_transcription
            and raw_filtered_events[i + 1].output_transcription.text
        ):
          continue
        event = event.model_copy(deep=True)
        event.output_transcription = None
        event.content = types.Content(
            role='model',
            parts=[types.Part(text=accumulated_output_transcription)],
        )
        accumulated_output_transcription = ''

    if _is_other_agent_reply(agent_name, event):
      if converted_event := _present_other_agent_message(event):
        filtered_events.append(converted_event)
    else:
      filtered_events.append(event)

  # Rearrange events for proper function call/response pairing
  result_events = _rearrange_events_for_latest_function_response(
      filtered_events
  )
  result_events = _rearrange_events_for_async_function_responses_in_history(
      result_events
  )

  # Convert events to contents
  contents = []
  for event in result_events:
    content = copy.deepcopy(event.content)
    remove_client_function_call_id(content)
    contents.append(content)
  return contents


def _get_current_turn_contents(
    current_branch: Optional[str], events: list[Event], agent_name: str = ''
) -> list[types.Content]:
  """Get contents for the current turn only (no conversation history).

  When include_contents='none', we want to include:
  - The current user input
  - Tool calls and responses from the current turn
  But exclude conversation history from previous turns.

  In multi-agent scenarios, the "current turn" for an agent starts from an
  actual user or from another agent.

  Args:
    current_branch: The current branch of the agent.
    events: A list of all session events.
    agent_name: The name of the agent.

  Returns:
    A list of contents for the current turn only, preserving context needed
    for proper tool execution while excluding conversation history.
  """
  # Find the latest event that starts the current turn and process from there
  for i in range(len(events) - 1, -1, -1):
    event = events[i]
    if event.author == 'user' or _is_other_agent_reply(agent_name, event):
      return _get_contents(current_branch, events[i:], agent_name)

  return []


def _is_other_agent_reply(current_agent_name: str, event: Event) -> bool:
  """Whether the event is a reply from another agent."""
  return bool(
      current_agent_name
      and event.author != current_agent_name
      and event.author != 'user'
  )


def _present_other_agent_message(event: Event) -> Optional[Event]:
  """Presents another agent's message as user context for the current agent.

  Reformats the event with role='user' and adds '[agent_name] said:' prefix
  to provide context without confusion about authorship.

  Args:
    event: The event from another agent to present as context.

  Returns:
    Event reformatted as user-role context with agent attribution, or None
    if no meaningful content remains after filtering.
  """
  if not event.content or not event.content.parts:
    return event

  content = types.Content()
  content.role = 'user'
  content.parts = [types.Part(text='For context:')]
  for part in event.content.parts:
    if part.thought:
      # Exclude thoughts from the context.
      continue
    elif part.text:
      content.parts.append(
          types.Part(text=f'[{event.author}] said: {part.text}')
      )
    elif part.function_call:
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] called tool `{part.function_call.name}`'
                  f' with parameters: {part.function_call.args}'
              )
          )
      )
    elif part.function_response:
      # Otherwise, create a new text part.
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] `{part.function_response.name}` tool'
                  f' returned result: {part.function_response.response}'
              )
          )
      )
    # Fallback to the original part for non-text and non-functionCall parts.
    else:
      content.parts.append(part)

  # If no meaningful parts were added (only "For context:" remains), return None
  if len(content.parts) == 1:
    return None

  return Event(
      timestamp=event.timestamp,
      author='user',
      content=content,
      branch=event.branch,
  )


def _merge_function_response_events(
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
  parts_in_merged_event: list[types.Part] = merged_event.content.parts  # type: ignore

  if not parts_in_merged_event:
    raise ValueError('There should be at least one function_response part.')

  part_indices_in_merged_event: dict[str, int] = {}
  for idx, part in enumerate(parts_in_merged_event):
    if part.function_response:
      function_call_id: str = part.function_response.id  # type: ignore
      part_indices_in_merged_event[function_call_id] = idx

  for event in function_response_events[1:]:
    if not event.content.parts:
      raise ValueError('There should be at least one function_response part.')

    for part in event.content.parts:
      if part.function_response:
        function_call_id: str = part.function_response.id  # type: ignore
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


def _is_event_belongs_to_branch(
    invocation_branch: Optional[str], event: Event
) -> bool:
  """Check if an event belongs to the current branch.

  This is for event context segration between agents. E.g. agent A shouldn't
  see output of agent B.
  """
  if not invocation_branch or not event.branch:
    return True
  return invocation_branch.startswith(event.branch)


def _is_function_call_event(event: Event, function_name: str) -> bool:
  """Checks if an event is a function call/response for a given function name."""
  if not event.content or not event.content.parts:
    return False
  for part in event.content.parts:
    if part.function_call and part.function_call.name == function_name:
      return True
    if part.function_response and part.function_response.name == function_name:
      return True
  return False


def _is_auth_event(event: Event) -> bool:
  """Checks if the event is an authentication event."""
  return _is_function_call_event(event, REQUEST_EUC_FUNCTION_CALL_NAME)


def _is_request_confirmation_event(event: Event) -> bool:
  """Checks if the event is a request confirmation event."""
  return _is_function_call_event(event, REQUEST_CONFIRMATION_FUNCTION_CALL_NAME)


def _is_live_model_audio_event(event: Event) -> bool:
  """Check if the event is an audio event produced by live/bidi models

  There are two possible cases:
  content=Content(
    parts=[
      Part(
        file_data=FileData(
          file_uri='artifact://live_bidi_streaming_multi_agent/user/cccf0b8b-4a30-449a-890e-e8b8deb661a1/_adk_live/adk_live_audio_storage_input_audio_1756092402277.pcm#1',
          mime_type='audio/pcm'
        )
      ),
    ],
    role='user'
  )
  content=Content(
    parts=[
      Part(
        inline_data=Blob(
          data=b'\x01\x00\x00...',
          mime_type='audio/pcm'
        )
      ),
    ],
    role='model'
  ) grounding_metadata=None partial=None turn_complete=None finish_reason=None error_code=None error_message=None ...
  """
  if not event.content:
    return False
  if not event.content.parts:
    return False
  # If it's audio data, then one event only has one part of audio.
  for part in event.content.parts:
    if part.inline_data and part.inline_data.mime_type == 'audio/pcm':
      return True
    if part.file_data and part.file_data.mime_type == 'audio/pcm':
      return True
  return False

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
from typing import Generator
from typing import Optional

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import remove_client_function_call_id
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

    if agent.include_contents != 'none':
      llm_request.contents = _get_contents(
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
    # No need to process, since the latest event is not fuction_response.
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
          break
        if function_call_event_idx != -1:
          # in case the last response event only have part of the responses
          # for the function calls in the function call event
          for function_call in function_calls:
            function_responses_ids.add(function_call.id)
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


def _get_contents(
    current_branch: Optional[str], events: list[Event], agent_name: str = ''
) -> list[types.Content]:
  """Get the contents for the LLM request.

  Args:
    current_branch: The current branch of the agent.
    events: A list of events.
    agent_name: The name of the agent.

  Returns:
    A list of contents.
  """
  filtered_events = []
  # Parse the events, leaving the contents and the function calls and
  # responses from the current agent.
  for event in events:
    if (
        not event.content
        or not event.content.role
        or not event.content.parts
        or event.content.parts[0].text == ''
    ):
      # Skip events without content, or generated neither by user nor by model
      # or has empty text.
      # E.g. events purely for mutating session states.
      continue
    if not _is_event_belongs_to_branch(current_branch, event):
      # Skip events not belong to current branch.
      continue
    if _is_auth_event(event):
      # skip auth event
      continue
    filtered_events.append(
        _convert_foreign_event(event)
        if _is_other_agent_reply(agent_name, event)
        else event
    )

  result_events = _rearrange_events_for_latest_function_response(
      filtered_events
  )
  result_events = _rearrange_events_for_async_function_responses_in_history(
      result_events
  )
  contents = []
  for event in result_events:
    content = copy.deepcopy(event.content)
    remove_client_function_call_id(content)
    contents.append(content)
  return contents


def _is_other_agent_reply(current_agent_name: str, event: Event) -> bool:
  """Whether the event is a reply from another agent."""
  return bool(
      current_agent_name
      and event.author != current_agent_name
      and event.author != 'user'
  )


def _convert_foreign_event(event: Event) -> Event:
  """Converts an event authored by another agent as a user-content event.

  This is to provide another agent's output as context to the current agent, so
  that current agent can continue to respond, such as summarizing previous
  agent's reply, etc.

  Args:
    event: The event to convert.

  Returns:
    The converted event.

  """
  if not event.content or not event.content.parts:
    return event

  content = types.Content()
  content.role = 'user'
  content.parts = [types.Part(text='For context:')]
  for part in event.content.parts:
    if part.text:
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
        event. (Note, 3. may not be true when aync function return some
        intermediate response, there could also be some intermediate model
        response event without any function_response and such event will be
        ignored.)
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
  """Event belongs to a branch, when event.branch is prefix of the invocation branch."""
  if not invocation_branch or not event.branch:
    return True
  return invocation_branch.startswith(event.branch)


def _is_auth_event(event: Event) -> bool:
  if not event.content.parts:
    return False
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
    if (
        part.function_response
        and part.function_response.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
  return False

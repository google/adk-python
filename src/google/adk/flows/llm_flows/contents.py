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
from typing import AsyncGenerator, Generator, Optional

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import remove_client_function_call_id
from .functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ...agents.content_config import ContentConfig, SummarizationConfig
import asyncio
import logging

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

    config = agent.canonical_content_config
    if config.enabled:
      llm_request.contents = await _get_contents(
          config,
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
          invocation_context.session.state,
          agent
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
    if (
        function_responses
        and function_responses[0].id in function_responses_ids
    ):
      function_response_events.append(event)
  function_response_events.append(events[-1])

  result_events = events[: function_call_event_idx + 1]
  result_events.append(
      _merge_function_response_events(function_response_events)
  )

  return result_events


async def summarize_events_with_llm(
    agent: "LlmAgent",
    events: list[Event],
    summarization_config: Optional[SummarizationConfig] = None,
    summary_template: Optional[str] = None,
    model_name_for_request: Optional[str] = None,
) -> str:
    """
    Summarize a list of events using the agent's LLM or a model specified in SummarizationConfig.
    The model name MUST be provided via model_name_for_request for the LlmRequest.
    Returns the summary string.
    agent: LlmAgent (type as string to avoid circular import)
    """
    logger = logging.getLogger(__name__)

    if not events:
        return ""
    if not model_name_for_request:
        logger.error("Model name for request is required for summarization LLM call.")
        return "[Summary unavailable: Missing model name]"

    event_texts = []
    for event_item in events:
        if event_item.content and event_item.content.parts:
            for part in event_item.content.parts:
                if part.text:
                    event_texts.append(f"[{event_item.author}] {part.text}")
    history_text = "\n".join(event_texts)

    # Build the instruction
    instruction_text = "Summarize the following conversation history as concisely and informatively as possible."
    if summarization_config and summarization_config.instruction:
        instruction_text = summarization_config.instruction

    # Build the LLM request content for summarization
    from google.genai import types as GoogleGenAITypes
    combined_prompt_text = f"\nConversation History:\\n{history_text}"
    summarization_contents = [GoogleGenAITypes.Content(role="user", parts=[GoogleGenAITypes.Part(text=combined_prompt_text)])]

    summarization_llm_obj = agent.canonical_model
    # Determine which model object to call based on config, fallback to agent's
    # Note: We use model_name_for_request (string) IN the LlmRequest below.
    if summarization_config and summarization_config.model:
        if summarization_config.model != model_name_for_request:
             # This case should ideally not happen if logic in _get_contents is correct,
             # but log if config model and requested model name differ.
             logger.warning(f"Summarization config model '{summarization_config.model}' differs from requested model name '{model_name_for_request}'. Using object for '{summarization_config.model}'.")
        try:
            from ...models.registry import LLMRegistry
            summarization_llm_obj = LLMRegistry.new_llm(summarization_config.model)
            logger.info(f"Using custom model object '{summarization_config.model}' for summarization call.")
        except Exception as e:
            logger.warning(f"Failed to instantiate custom summarization model object '{summarization_config.model}', falling back to agent's default model object. Error: {e}")
            summarization_llm_obj = agent.canonical_model
    else:
         logger.info(f"Using agent's default model object for summarization call (model name for request: '{model_name_for_request}').")
         summarization_llm_obj = agent.canonical_model

    # Prepare generation_config
    final_generation_config = GoogleGenAITypes.GenerateContentConfig(
        system_instruction=instruction_text,
        temperature=0.1,
        max_output_tokens=1000
    )

    # Construct LlmRequest and Call the LLM by passing the request object
    try:
        from ...models.llm_request import LlmRequest
        llm_summary_request = LlmRequest(
            model=model_name_for_request,
            contents=summarization_contents,
            config=final_generation_config, 
            tools=[],                            
        )
        
        logger.info(f"Sending summarization request object via {summarization_llm_obj.__class__.__name__}: {llm_summary_request.model_dump(exclude_none=True)}")

        response_parts = []
        # Call generate_content_async passing the LlmRequest object positionally
        async for response_chunk in summarization_llm_obj.generate_content_async(llm_summary_request):
            if response_chunk and response_chunk.content and response_chunk.content.parts:
                for part in response_chunk.content.parts:
                    if part.text:
                        response_parts.append(part.text)
        
        summary = "".join(response_parts)
        
        if summary.strip():
            if summary_template:
                return summary_template.format(summary=summary)
            return summary
        
        logger.warning("Summarization produced an empty or whitespace-only result.")
        return "[Summary produced no content]"

    except Exception as e:
        logger.error(f"Summarization LLM call failed: {e}", exc_info=True)
        return "[Summary unavailable due to error]"


async def _get_contents(
    config: ContentConfig,
    current_branch: Optional[str],
    events: list[Event],
    agent_name: str = '',
    session_state: Optional[dict] = None,
    agent: "LlmAgent" = None,
) -> list[types.Content]:
  """Get the contents for the LLM request, using ContentConfig-driven pipeline."""
  logger = logging.getLogger(__name__)
  
  initial_filtered_events = []
  for event in events:
    if (
        not event.content
        or not event.content.role
        or not event.content.parts
        or (hasattr(event.content.parts[0], 'text') and not event.content.parts[0].text)
    ): continue # Skip empty/invalid events
    if not _is_event_belongs_to_branch(current_branch, event): continue
    if _is_auth_event(event): continue
    if config.include_authors and event.author not in config.include_authors: continue
    if config.exclude_authors and event.author in config.exclude_authors: continue
    initial_filtered_events.append(event)
  if config.max_events is not None and len(initial_filtered_events) > config.max_events:
    initial_filtered_events = initial_filtered_events[-config.max_events:]

  # Isolate events for 'always_include_last_n' (Y)
  Y = config.always_include_last_n or 0
  events_to_always_include_in_full: list[Event] = []
  events_for_further_processing: list[Event] = [] # Declare here

  if Y > 0 and len(initial_filtered_events) >= Y:
      events_to_always_include_in_full = initial_filtered_events[-Y:]
      events_for_further_processing = initial_filtered_events[:-Y]
  elif Y > 0: # Less total events than Y, all are "always_include"
      events_to_always_include_in_full = initial_filtered_events[:] # Use a copy
      events_for_further_processing = [] # Explicitly empty
  else: # Y is 0 or None
      events_for_further_processing = initial_filtered_events[:] # Use a copy
      # events_to_always_include_in_full remains empty as initialized

  # Apply summarization_window (N) to the remaining events (events_for_further_processing)
  N = config.summarization_window
  events_from_window_to_consider: list[Event] = events_for_further_processing

  if N is not None and len(events_for_further_processing) > N:
      events_from_window_to_consider = events_for_further_processing[-N:]
  # If N is None or larger/equal, all events_for_further_processing are considered.

  processed_summary_event: Optional[Event] = None
  final_events_before_always_include: list[Event] = [] 

  if config.summarize and agent and events_from_window_to_consider:
      logger.info(f"Summarizing {len(events_from_window_to_consider)} events (from window before last {Y})...")
      
      summarization_model_name_str: Optional[str] = None
      if config.summarization_config and config.summarization_config.model:
          summarization_model_name_str = config.summarization_config.model
      elif isinstance(agent.model, str) and agent.model:
            summarization_model_name_str = agent.model
      elif hasattr(agent.canonical_model, 'model_name'):
            try: summarization_model_name_str = agent.canonical_model.model_name
            except: pass # Keep as None if attribute exists but fails to retrieve
      elif hasattr(agent.canonical_model, '_model_id'): # Gemini specific, attempt if model_name wasn't found
            try: summarization_model_name_str = agent.canonical_model._model_id
            except: pass # Keep as None

      if summarization_model_name_str:
          summary_text = await summarize_events_with_llm(
              agent=agent,
              events=events_from_window_to_consider,
              summarization_config=config.summarization_config,
              summary_template=config.summary_template,
              model_name_for_request=summarization_model_name_str
          )
          if summary_text.strip() and not summary_text.startswith("[Summary unavailable") and not summary_text.startswith("[Summary produced no content"):
              summary_part = types.Part(text=summary_text)
              content_for_summary_event = types.Content(role='user', parts=[summary_part])
              processed_summary_event = Event(
                  author='system_summary', 
                  content=content_for_summary_event, 
                  branch=current_branch,
                  id=Event.new_id()
              )
              final_events_before_always_include.append(processed_summary_event)
              logger.debug(f"Created summary event.")
          else:
              logger.warning(f"Summarization did not produce valid text: {summary_text}. Including original events from window.")
              final_events_before_always_include.extend(events_from_window_to_consider)
      else:
          logger.error("Could not determine model name for summarization. Skipping summarization and including original events from window.")
          final_events_before_always_include.extend(events_from_window_to_consider)
  else: 
      final_events_before_always_include.extend(events_from_window_to_consider)

  # State/context injection (if configured)
  context_event: Optional[Event] = None
  if config.context_from_state and session_state:
    context_values = {k: session_state.get(k, "") for k in config.context_from_state}
    context_str = ""
    if config.state_template:
      try: context_str = config.state_template.format(context=context_values)
      except Exception as e: 
          logger.warning(f"Failed state template format: {e}. Defaulting."); 
          context_str = "\n".join(f"{k}: {v}" for k, v in context_values.items())
    else: context_str = "\n".join(f"{k}: {v}" for k, v in context_values.items())
    
    if context_str.strip():
      context_part = types.Part(text=context_str)
      content_for_context_event = types.Content(role='user', parts=[context_part])
      context_event = Event(
          author='system_context',
          content=content_for_context_event,
          branch=current_branch,
          id=Event.new_id()
      )
      logger.debug("Created context injection event.")
      
  # Recombination
  final_event_list: list[Event] = []
  if context_event:
      final_event_list.append(context_event)

  final_event_list.extend(final_events_before_always_include)
  final_event_list.extend(events_to_always_include_in_full)

  # Convert foreign events (if enabled)
  if config.convert_foreign_events:
    converted_final_event_list: list[Event] = []
    for event_item in final_event_list:
      if _is_other_agent_reply(agent_name, event_item):
        converted_final_event_list.append(_convert_foreign_event(event_item))
      else:
        converted_final_event_list.append(event_item)
    final_event_list = converted_final_event_list # Update list with converted events

  # Final processing: Rearrangement and Conversion to types.Content
  final_model_contents: list[types.Content] = []
  if final_event_list:
      rearranged_events = _rearrange_events_for_latest_function_response(final_event_list)
      rearranged_events = _rearrange_events_for_async_function_responses_in_history(rearranged_events)
      
      for event_item in rearranged_events:
        if event_item.content:
            content_copy = copy.deepcopy(event_item.content)
            remove_client_function_call_id(content_copy)
            final_model_contents.append(content_copy)
      logger.debug(f"Final processed content count: {len(final_model_contents)}")
  else:
       logger.debug("No events generated for the final content list.")
       
  return final_model_contents


def _is_other_agent_reply(current_agent_name: str, event: Event) -> bool:
  """Whether the event is a reply from another agent."""
  if event.author in ['system_context', 'system_summary']: # Ignore system-generated events
      return False
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

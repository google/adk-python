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
from ...agents.content_config import ContentConfig, SummarizationConfig

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
    if agent.include_contents != 'none':
      llm_request.contents = await _get_contents(
          config,
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
          session_state=invocation_context.session.state,
          current_invocation_id=invocation_context.invocation_id
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


def _normalize_limits(
    max_events: Optional[int], 
    always_include_last_n: Optional[int], 
    total: int
) -> tuple[Optional[int], int]:
    """
    Normalize the values of max_events and always_include_last_n for robust slicing logic.

    - None means "no limit" for max_events.
    - always_include_last_n is never greater than max_events (if both are set).
    - always_include_last_n is never greater than the total number of contents.
    - Negative or zero values are treated as no limit (for max_events) or zero (for always_include_last_n).

    Args:
        max_events: Optional maximum number of events to include (None means unlimited).
        always_include_last_n: Optional number of most recent events to always include in full (None or 0 means none).
        total: The total number of available content items.

    Returns:
        A tuple (max_events, always_include_last_n) with normalized, safe values for downstream logic.
    """
    if max_events is not None and max_events <= 0:
        max_events = None
    if always_include_last_n is None or always_include_last_n <= 0:
        always_include_last_n = 0
    if max_events is not None and always_include_last_n > max_events:
        always_include_last_n = max_events
    if always_include_last_n > total:
        always_include_last_n = total
    return max_events, always_include_last_n


def _apply_max_events_with_always_include(
    contents: list[types.Content], 
    max_events: Optional[int], 
    always_include_last_n: Optional[int]
) -> list[types.Content]:
    """
    Apply max_events and always_include_last_n slicing to the content list in a consistent, robust way.

    - If max_events is set, returns at most max_events items from the end of the list.
    - If always_include_last_n is set and greater than max_events, always returns that many from the end.
    - If neither is set, returns the full list.

    This ensures that the always_include_last_n constraint takes precedence if it is more restrictive than max_events.

    Args:
        contents: The list of content items to slice.
        max_events: Optional maximum number of events to include.
        always_include_last_n: Optional number of most recent events to always include in full.

    Returns:
        The sliced list of content items, respecting the constraints.
    """
    total = len(contents)
    max_events, always_include_last_n = _normalize_limits(max_events, always_include_last_n, total)
    if max_events is not None:
        if always_include_last_n > max_events:
            return contents[-always_include_last_n:]
        return contents[-max_events:]
    if always_include_last_n > 0:
        return contents[-always_include_last_n:]
    return contents


def _get_contents_to_summarize(contents: list[types.Content], always_include_last_n: int) -> list[types.Content]:
    """
    Returns the contents that should be summarized (all except the always-include tail).

    Args:
        contents: The full list of content items.
        always_include_last_n: The number of most recent items to always include in full.

    Returns:
        The list of content items to be summarized (all except the last N).
    """
    if always_include_last_n < len(contents):
        return contents[:-always_include_last_n]
    return []


def _get_always_include_contents(contents: list[types.Content], always_include_last_n: int) -> list[types.Content]:
    """
    Returns the last always_include_last_n contents, or the full list if N is zero.

    Args:
        contents: The full list of content items.
        always_include_last_n: The number of most recent items to always include in full.

    Returns:
        The list of content items to always include (the last N, or all if N is zero).
    """
    if always_include_last_n > 0:
        return contents[-always_include_last_n:]
    return contents


def _prepare_final_contents(contents: list[types.Content], always_include_last_n: int) -> tuple[list[types.Content], list[types.Content]]:
    """
    Groups the logic for splitting contents into those to summarize and those to always include.

    Args:
        contents: The full list of content items.
        always_include_last_n: The number of most recent items to always include in full.

    Returns:
        A tuple (to_summarize, always_include) where:
            - to_summarize: list of content items to be summarized
            - always_include: list of content items to always include in full
    """
    to_summarize = _get_contents_to_summarize(contents, always_include_last_n)
    always_include = _get_always_include_contents(contents, always_include_last_n)
    return to_summarize, always_include


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


async def _summarize_contents_with_llm(
    events: list[Event],
    summarization_config: Optional[SummarizationConfig] = None,
    summary_template: Optional[str] = None,
    model_name_for_request: Optional[str] = None,
) -> str:
    """
    Summarize a list of events using the agent's LLM or a model specified in SummarizationConfig.
    The model name MUST be provided via model_name_for_request for the LlmRequest.
    Returns the summary string.
    """
     #in future permit to use any agent
     #Type str to avoid circular import with LlmAgent
    agent: str =  "LlmAgent",
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


def _get_state_context_content(
    state: dict,
    context_keys: Optional[list[str]],
    state_template: Optional[str] = None
) -> Optional[types.Content]:
    """
    Extracts and formats session state as a Content object for LLM context injection.

    Args:
        state: The full session state dictionary.
        context_keys: List of keys to extract from the state.
        state_template: Optional template string for formatting.

    Returns:
        A types.Content object with the formatted state, or None if no keys found.
    """
    if not context_keys:
        return None
    context_data = {k: state.get(k) for k in context_keys if k in state}
    if not context_data:
        return None
    template = state_template or "Session Information:\n{context}"
    try:
        context_str = "\n".join(f"{k}: {v}" for k, v in context_data.items())
        formatted = template.format(context=context_str)
    except Exception:
        # Fallback to simple key-value listing
        formatted = f"Session Information:\n" + "\n".join(f"{k}: {v}" for k, v in context_data.items())
    from google.genai import types as GoogleGenAITypes
    return GoogleGenAITypes.Content(role='user', parts=[GoogleGenAITypes.Part(text=formatted)])


def _is_event_from_agent_flow(event, current_agent_name, current_branch, current_invocation_id):
    """
    Returns True if the event is from the current agent's direct flow (same agent, branch is prefix, and invocation_id matches).
    """
    return (
        event.author == current_agent_name
        and event.branch
        and current_branch
        and current_branch.startswith(event.branch)
        and getattr(event, 'invocation_id', '') == current_invocation_id
    )


def _should_exclude_event(event: Event, config: ContentConfig, current_agent_name: str, current_branch: str, current_invocation_id: str):
    """
    Returns True if the event should be excluded based on exclude_authors, unless it is from the agent's own flow.
    """
    if config.exclude_authors and event.author in config.exclude_authors:
        if not _is_event_from_agent_flow(event, current_agent_name, current_branch, current_invocation_id):
            return True
    return False


def _should_include_event(event: Event, config: ContentConfig, current_agent_name: str, current_branch: str, current_invocation_id: str):
    """
    Returns True if the event should be included based on include_authors, or if it is from the agent's own flow.
    """
    if config.include_authors:
        if event.author not in config.include_authors:
            if not _is_event_from_agent_flow(event, current_agent_name, current_branch, current_invocation_id):
                return False
    return True


async def _get_contents(
    config: ContentConfig,
    current_branch: Optional[str], 
    events: list[Event], 
    agent_name: str = '',     
    session_state: Optional[dict] = None,  # new argument for state injection
    current_invocation_id: Optional[str] = None,  # new argument for invocation_id
) -> list[types.Content]:
    """
    Get the contents for the LLM request, applying all filtering, slicing, summarization, and state context injection logic.

    This function:
    - Optionally injects formatted session state as the first content item.
    - Filters events to only those relevant for the current branch and agent.
    - Converts foreign agent events to user-readable context if needed.
    - Applies max_events and always_include_last_n constraints in a robust, consistent way.
    - Splits the result into contents to summarize and those to always include in full.
    - Optionally summarizes older history if configured.

    Args:
        config: The ContentConfig object with all content window settings.
        current_branch: The current branch of the agent (for event filtering).
        events: The full list of session events.
        agent_name: The name of the current agent (for foreign event conversion).
        session_state: The session state dictionary (for context injection).
        current_invocation_id: The current invocation ID (for invocation-specific filtering).

    Returns:
        The final list of content items to send to the LLM, with state, summarization, and always-include logic applied.
    """

    # A. If config.enabled is False, return only state context (if any) or an empty list
    if not config.enabled:
        state_content = None
        if config.context_from_state and session_state is not None:
            state_content = _get_state_context_content(
                session_state,
                config.context_from_state,
                config.state_template
            )
        return [state_content] if state_content else []

    # 1. Filter and process events as before
    filtered_events = []
    for event in events:
        if (
            not event.content
            or not event.content.role
            or not event.content.parts
            or event.content.parts[0].text == ''
        ):
            continue
        if not _is_event_belongs_to_branch(current_branch, event):
            continue
        if _is_auth_event(event):
            continue
        # Author filtering logic
        if _should_exclude_event(event, config, agent_name, current_branch, current_invocation_id):
            continue
        if not _should_include_event(event, config, agent_name, current_branch, current_invocation_id):
            continue
        # B. Respect convert_foreign_events
        if _is_other_agent_reply(agent_name, event):
            if config.convert_foreign_events:
                filtered_events.append(_convert_foreign_event(event))
            else:
                filtered_events.append(event)
        else:
            filtered_events.append(event)

    # 2. Rearrange events for function call/response consistency
    result_events = _rearrange_events_for_latest_function_response(filtered_events)

    result_events = _rearrange_events_for_async_function_responses_in_history(result_events)

    contents = []
    for event in result_events:
        content = copy.deepcopy(event.content)
        remove_client_function_call_id(content)
        contents.append(content)
    
    # 3. Optionally inject state context as the first content item
    state_content = None
    if config.context_from_state and session_state is not None:
        state_content = _get_state_context_content(
            session_state,
            config.context_from_state,
            config.state_template
        )

    # 4. Apply max_events and always_include_last_n constraints
    max_events = config.max_events
    n_always = config.always_include_last_n
    contents = _apply_max_events_with_always_include(contents, max_events, n_always)

    # 5. Normalize n_always for downstream splitting
    _, n_always = _normalize_limits(max_events, n_always, len(contents))
    contents_to_summarize, always_include_contents = _prepare_final_contents(contents, n_always)

    # C. Implement use of summarization_window
    summarization_window = config.summarization_window
    if summarization_window is not None and summarization_window > 0:
        if len(contents_to_summarize) > summarization_window:
            contents_to_summarize = contents_to_summarize[-summarization_window:]

    summarized_contents = []
    if config.summarize and n_always > 0 and contents_to_summarize:
        summary = await _summarize_contents_with_llm(
            contents_to_summarize,
            config.summarization_config,
            config.summary_template,
            agent_name
        )
        if summary:
            if isinstance(summary, list):
                summarized_contents = summary
            else:
                from google.genai import types as GoogleGenAITypes
                summarized_contents = [GoogleGenAITypes.Content(role="user", parts=[GoogleGenAITypes.Part(text=summary)])]

    # 6. Compose the final content list: [state] + [summary] + [always_include]
    filtered_contents = []
    if state_content:
        filtered_contents.append(state_content)
    if summarized_contents:
        filtered_contents.extend(summarized_contents)
    filtered_contents.extend(always_include_contents)
    return filtered_contents
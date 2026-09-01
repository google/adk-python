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

import copy
import functools
import logging
from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events._branch_path import _BranchPath
from ...events._rewind_events import _apply_rewinds
from ...events.event import Event
from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from ._content_compaction import _process_compaction_events
from ._content_compaction import _recover_compacted_function_calls
from ._fencing import _is_other_agent_reply
from ._fencing import _present_other_agent_message
from ._invocation_utils import as_llm_agent
from ._tool_call_rearranger import _drop_orphaned_function_responses
from ._tool_call_rearranger import _rearrange_events_for_async_function_responses_in_history
from ._tool_call_rearranger import _rearrange_events_for_latest_function_response
from .functions import AF_FUNCTION_CALL_ID_PREFIX
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from .functions import REQUEST_EUC_FUNCTION_CALL_NAME

logger = logging.getLogger('google_adk.' + __name__)


@functools.cache
def _id_pairing_model_types() -> tuple[type[BaseLlm], ...]:
  """Returns the installed model types that pair tool calls with results by id.

  Each provider is optional, so an absent one is simply left out. The result is
  memoized because Python does not cache a failed import: without this, an
  install that has none of these packages re-runs three doomed imports on every
  LLM request. Installing a provider into a running interpreter therefore has no
  effect until restart, which is already true of a successful import.
  """
  model_types: list[type[BaseLlm]] = []
  try:
    from ...models.anthropic_llm import AnthropicLlm

    model_types.append(AnthropicLlm)
  except (ImportError, OSError):
    pass
  try:
    from ...models.lite_llm import LiteLlm

    model_types.append(LiteLlm)
  except (ImportError, OSError):
    pass
  try:
    from ...labs.openai import OpenAIResponsesLlm

    model_types.append(OpenAIResponsesLlm)
  except (ImportError, OSError):
    pass
  return tuple(model_types)


class _ContentLlmRequestProcessor(BaseLlmRequestProcessor):
  """Builds the contents for the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...models.google_llm import Gemini

    agent = as_llm_agent(invocation_context)
    preserve_function_call_ids = False
    if hasattr(agent, 'canonical_model'):
      canonical_model = agent.canonical_model
      if (
          isinstance(canonical_model, Gemini)
          and canonical_model.use_interactions_api
      ):
        preserve_function_call_ids = True
      else:
        # Anthropic and LiteLLM-backed providers (e.g. OpenAI) pair tool
        # calls with their results by id, so `adk-*` fallback ids must
        # survive replay.
        if isinstance(canonical_model, _id_pairing_model_types()):
          preserve_function_call_ids = True

    # Preserve all contents that were added by instruction processor
    # (since llm_request.contents will be completely reassigned below)
    instruction_related_contents = llm_request.contents
    run_config = invocation_context.run_config
    include_thoughts_from_other_agents = (
        run_config.include_thoughts_from_other_agents
        if run_config is not None
        else False
    )

    is_single_turn = getattr(agent, 'mode', None) == 'single_turn'
    if (
        agent.include_contents == 'default'
        and not llm_request.previous_interaction_id
    ):
      # Include full conversation history
      llm_request.contents = _get_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
          preserve_function_call_ids=preserve_function_call_ids,
          isolation_scope=invocation_context.isolation_scope,
          is_single_turn=is_single_turn,
          user_content=invocation_context.user_content,
          include_thoughts_from_other_agents=include_thoughts_from_other_agents,
      )
    else:
      # Include current turn context only (no conversation history). Stateful
      # Interactions requests already retain earlier turns server-side.
      llm_request.contents = _get_current_turn_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
          preserve_function_call_ids=preserve_function_call_ids,
          isolation_scope=invocation_context.isolation_scope,
          is_single_turn=is_single_turn,
          user_content=invocation_context.user_content,
          include_thoughts_from_other_agents=False,
      )

    if run_config is not None and run_config.model_input_context:
      _add_model_input_context_to_user_content(
          invocation_context,
          llm_request,
          copy.deepcopy(run_config.model_input_context),
      )

    # Add instruction-related contents to proper position in conversation
    await _add_instructions_to_user_content(
        invocation_context, llm_request, instruction_related_contents
    )

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _ContentLlmRequestProcessor()


def _is_part_invisible(
    p: types.Part, *, include_thoughts: bool = False
) -> bool:
  """Returns whether a part is invisible for LLM context.

  A part is invisible if:
  - It has no meaningful content (text, inline_data, file_data, function_call,
    function_response, tool_call, tool_response, executable_code, or
    code_execution_result), OR
  - It is marked as a thought AND does not contain function_call,
    function_response, tool_call, tool_response or thought_signature

  Function calls and responses are never invisible, even if marked as thought,
  because they represent actions that need to be executed or results that need
  to be processed.

  A part carrying a thought signature is never invisible either. The signature
  is opaque state the model expects back verbatim, and it commonly arrives on
  a part that holds nothing else, which would otherwise read as empty.

  Server-side tool calls and their responses are never invisible either. The
  model runs those tools itself and the caller is required to echo the parts
  back on the next request; dropping them makes the model redo the work it
  already did, or fail because a call has no matching response.

  Args:
    p: The part to check.
  """
  # Function calls and responses are never invisible, even if marked as thought
  if p.function_call or p.function_response:
    return False

  # A thought signature is opaque state the model hands back for us to return
  # verbatim on the next request. It routinely arrives on a part with no other
  # content at all, so it has to be checked before the emptiness test below.
  if p.thought_signature:
    return False

  # Server-side tool calls/responses must be echoed back to the model.
  if p.tool_call or p.tool_response:
    return False

  return (p.thought and not include_thoughts) or not (
      p.text
      or p.inline_data
      or p.file_data
      or p.executable_code
      or p.code_execution_result
  )


def _contains_empty_content(
    event: Event, *, include_thoughts: bool = False
) -> bool:
  """Check if an event should be skipped due to missing or empty content.

  This can happen to the events that only changed session state.
  When both content and transcriptions are empty, the event will be considered
  as empty. The content is considered empty if none of its parts contain text,
  inline data, file data, function call, function response, server-side tool
  call, server-side tool response, executable code, or code execution result.
  Parts with only thoughts are also considered empty.

  Args:
    event: The event to check.

  Returns:
    True if the event should be skipped, False otherwise.
  """
  if event.actions and event.actions.compaction:
    return False

  return (
      not event.content
      or not event.content.role
      or not event.content.parts
      or all(
          _is_part_invisible(p, include_thoughts=include_thoughts)
          for p in event.content.parts
      )
  ) and (not event.output_transcription and not event.input_transcription)


_SINGLE_TURN_NUDGE = (
    'Important: You will not receive any user replies or clarifications.'
    ' Complete the task using only the information provided above.'
)


def _build_task_input_user_content(
    all_events: list[Event],
    isolation_scope: str,
    is_single_turn: bool = False,
    user_content: types.Content | None = None,
) -> types.Content | None:
  """Find the originating task-delegation FC and convert its args to user content.

  A task agent runs under ``isolation_scope=<fc_id>``, where ``fc_id``
  matches the function_call.id that delegated to it.  The FC itself
  lives on a parent event (typically the chat coordinator's), so it
  is filtered out of the task agent's content by the isolation_scope
  filter.  This helper rebuilds it as a user-role text content so the
  task agent's LLM sees its task as the first turn.

  When no matching FC is found (workflow-node task case — task agent
  dispatched directly by a Workflow, not via FC delegation), falls
  back to ``user_content`` (set on the InvocationContext by the
  wrapper to ``to_user_content(node_input)``).

  When ``is_single_turn`` is True, appends a second text part nudging
  the LLM that no further user replies will arrive — single-turn
  agents must complete the task from the input alone.

  Returns None if neither source yields content.
  """
  for event in all_events:
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      fc = part.function_call
      if fc and fc.id == isolation_scope and fc.args:
        # Render args as JSON string — same shape an LLM would emit.
        try:
          import json as _json

          text = _json.dumps(dict(fc.args), ensure_ascii=False)
        except (TypeError, ValueError):
          text = str(fc.args)
        parts = [types.Part(text=text)]
        if is_single_turn:
          parts.append(types.Part(text=_SINGLE_TURN_NUDGE))
        return types.Content(role='user', parts=parts)

  # Fallback: workflow-node task with no originating FC.  Use the
  # node_input that the wrapper stamped onto ``ic.user_content``.
  if user_content and user_content.parts:
    parts = list(user_content.parts)
    if is_single_turn:
      parts.append(types.Part(text=_SINGLE_TURN_NUDGE))
    return types.Content(role='user', parts=parts)
  return None


def _should_include_event_in_context(
    current_branch: str | None,
    event: Event,
    isolation_scope: str | None = None,
    *,
    include_thoughts: bool = False,
) -> bool:
  """Determines if an event should be included in the LLM context.

  This filters out events that are considered empty (e.g., no text, function
  calls, or transcriptions), do not belong to the current agent's branch, or
  are internal events like authentication or confirmation requests.

  Events are scoped via ``isolation_scope``: an event is visible to an
  agent only when their ``isolation_scope`` values match exactly. A chat
  coordinator (unscoped, ``isolation_scope=None``) sees only unscoped
  events; a task or single_turn agent (scoped under the originating
  function-call id) sees only its own scoped events.

  Args:
    current_branch: The current branch of the agent.
    event: The event to filter.
    isolation_scope: The agent's isolation_scope. None means unscoped.

  Returns:
    True if the event should be included in the context, False otherwise.
  """
  ev_iso = getattr(event, 'isolation_scope', None)
  if ev_iso != isolation_scope:
    return False
  return not (
      _contains_empty_content(event, include_thoughts=include_thoughts)
      or not _is_event_belongs_to_branch(current_branch, event)
      or _is_adk_framework_event(event)
      or _is_auth_event(event)
      or _is_request_confirmation_event(event)
  )


def _copy_content_for_request(
    content: types.Content,
    *,
    strip_client_function_call_ids: bool,
) -> types.Content:
  """Returns a session-isolated copy of ``content`` for an LLM request.

  ``Content`` and every ``Part`` are shallow-copied so downstream request
  processors (nl_planning, code_execution) can mutate them without corrupting
  session events; payloads are shared by reference to avoid the deep recursion
  that the previous ``deepcopy`` paid on every request.

  Because the copy is shallow, nested fields (e.g. ``function_call.args``,
  ``inline_data.data``) are shared with the session events. Downstream
  processors must therefore only replace ``Part`` objects or set top-level
  ``Part`` fields; mutating a nested field in place would corrupt session
  history.

  Args:
    content: The (session-owned) content to copy. Not mutated.
    strip_client_function_call_ids: Whether to remove ``adk-`` prefixed function
      call/response ids (mirrors ``remove_client_function_call_id``).

  Returns:
    An isolated ``Content`` safe to attach to an ``LlmRequest``.
  """
  new_content = content.model_copy()
  parts = content.parts
  if not parts:
    return new_content

  new_parts = []
  for part in parts:
    new_part = part.model_copy()
    if strip_client_function_call_ids:
      fc = new_part.function_call
      if fc and fc.id and fc.id.startswith(AF_FUNCTION_CALL_ID_PREFIX):
        new_part.function_call = fc.model_copy(update={'id': None})
      fr = new_part.function_response
      if fr and fr.id and fr.id.startswith(AF_FUNCTION_CALL_ID_PREFIX):
        new_part.function_response = fr.model_copy(update={'id': None})
    new_parts.append(new_part)
  new_content.parts = new_parts
  return new_content


def _get_contents(
    current_branch: str | None,
    events: list[Event],
    agent_name: str = '',
    *,
    preserve_function_call_ids: bool = False,
    isolation_scope: str | None = None,
    is_single_turn: bool = False,
    user_content: types.Content | None = None,
    include_thoughts_from_other_agents: bool = False,
) -> list[types.Content]:
  """Get the contents for the LLM request.

  Applies filtering, rearrangement, and content processing to events.

  Args:
    current_branch: The current branch of the agent.
    events: Events to process.
    agent_name: The name of the agent.
    preserve_function_call_ids: Whether to preserve function call ids.
    isolation_scope: scope tag — when set, restricts events
      to those with matching ``event.isolation_scope`` (or unscoped).
    user_content: Fallback first user turn for task agents whose
      originating delegation FC is not in session (workflow-node
      task case).
    include_thoughts_from_other_agents: Whether to include thought parts from
      other agents when presenting their messages as user context.

  Returns:
    A list of processed contents.
  """
  accumulated_input_transcription = ''
  accumulated_output_transcription = ''

  # Filter out events that are annulled by a rewind, so the rewound history is
  # never sent to the LLM. This is the same rewind logic the context compactor
  # applies, keeping the two consistent (see google.adk.events._rewind_events).
  rewind_filtered_events = _apply_rewinds(events)

  # Parse the events, leaving the contents and the function calls and
  # responses from the current agent.
  raw_filtered_events = [
      e
      for e in rewind_filtered_events
      if _should_include_event_in_context(
          current_branch,
          e,
          isolation_scope=isolation_scope,
          include_thoughts=(
              include_thoughts_from_other_agents
              and _is_other_agent_reply(agent_name, e)
          ),
      )
  ]

  has_compaction_events = any(
      e.actions and e.actions.compaction for e in raw_filtered_events
  )

  if has_compaction_events:
    events_to_process = _process_compaction_events(
        raw_filtered_events, agent_name
    )
    # Compaction may have removed a function_call whose response survives
    # (e.g. a long-running call resumed after it was compacted); restore it so
    # the call/response pairing is intact.
    events_to_process = _recover_compacted_function_calls(
        events_to_process, raw_filtered_events
    )
  else:
    events_to_process = raw_filtered_events

  # Build mapping of function call IDs to their authors
  fc_author_by_id: dict[str, str] = {}
  for e in events_to_process:
    if e.content and e.content.parts:
      for part in e.content.parts:
        if part.function_call:
          function_call_id = part.function_call.id
          if function_call_id:
            fc_author_by_id[function_call_id] = e.author

  filtered_events = []
  # aggregate transcription events
  for i in range(len(events_to_process)):
    event = events_to_process[i]
    if not event.content:
      # Convert transcription into normal event
      if event.input_transcription and event.input_transcription.text:
        accumulated_input_transcription += event.input_transcription.text
        next_input_transcription = (
            events_to_process[i + 1].input_transcription
            if i != len(events_to_process) - 1
            else None
        )
        if next_input_transcription and next_input_transcription.text:
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
        next_output_transcription = (
            events_to_process[i + 1].output_transcription
            if i != len(events_to_process) - 1
            else None
        )
        if next_output_transcription and next_output_transcription.text:
          continue
        event = event.model_copy(deep=True)
        event.output_transcription = None
        event.content = types.Content(
            role='model',
            parts=[types.Part(text=accumulated_output_transcription)],
        )
        accumulated_output_transcription = ''

    is_other_reply = _is_other_agent_reply(agent_name, event)

    # Check if it's a FunctionResponse for another agent
    if not is_other_reply and event.content:
      for part in event.content.parts or []:
        if part.function_response:
          resp_id = part.function_response.id
          call_author = fc_author_by_id.get(resp_id) if resp_id else None
          if (
              call_author
              and call_author != agent_name
              and call_author != 'user'
          ):
            is_other_reply = True
            break

    if is_other_reply:
      if converted_event := _present_other_agent_message(
          event, include_thoughts=include_thoughts_from_other_agents
      ):
        filtered_events.append(converted_event)
    else:
      filtered_events.append(event)

  # Rearrange events for proper function call/response pairing
  filtered_events = _drop_orphaned_function_responses(filtered_events)
  result_events = _rearrange_events_for_latest_function_response(
      filtered_events
  )
  result_events = _rearrange_events_for_async_function_responses_in_history(
      result_events
  )

  # Convert events to contents
  contents = []
  for event in result_events:
    if event.content:
      contents.append(
          _copy_content_for_request(
              event.content,
              strip_client_function_call_ids=not preserve_function_call_ids,
          )
      )

  # for scoped agents (task / single_turn), prepend a
  # synthetic user-role content built from the originating FC's args.
  # The FC lives in an UNSCOPED parent event (e.g., the coordinator's
  # task-delegation FC), which the strict isolation filter just
  # excluded — so we re-derive it directly from the full session
  # events here.  This becomes the agent's first turn: "your task is
  # X" instead of starting cold from system instruction only.
  if isolation_scope is not None:
    leading = _build_task_input_user_content(
        events,
        isolation_scope,
        is_single_turn=is_single_turn,
        user_content=user_content,
    )
    if leading is not None:
      contents.insert(0, leading)

  return contents


def _get_current_turn_contents(
    current_branch: str | None,
    events: list[Event],
    agent_name: str = '',
    *,
    preserve_function_call_ids: bool = False,
    is_single_turn: bool = False,
    isolation_scope: str | None = None,
    user_content: types.Content | None = None,
    include_thoughts_from_other_agents: bool = False,
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
    preserve_function_call_ids: Whether to preserve function call ids.
    include_thoughts_from_other_agents: Whether to include thought parts from
      other agents when presenting their messages as user context.

  Returns:
    A list of contents for the current turn only, preserving context needed
    for proper tool execution while excluding conversation history.
  """
  # Find the latest event that starts the current turn and process from there.
  # A posted-back tool result is not a turn start, and the slice must reach
  # back far enough to include the call it answers: the conversation can carry
  # on while a long-running tool is pending, so an ordinary user turn can sit
  # between the two, and anchoring there would leave the result orphaned.
  unmatched_response_ids: set[str] = set()
  for i in range(len(events) - 1, -1, -1):
    event = events[i]
    unmatched_response_ids -= {
        function_call.id
        for function_call in event.get_function_calls()
        if function_call.id
    }
    is_submitted_result = _is_submitted_tool_result(event)
    if is_submitted_result:
      unmatched_response_ids.update(
          function_response.id
          for function_response in event.get_function_responses()
          if function_response.id
      )
    if (
        not unmatched_response_ids
        and _should_include_event_in_context(
            current_branch,
            event,
            isolation_scope=isolation_scope,
            include_thoughts=(
                include_thoughts_from_other_agents
                and _is_other_agent_reply(agent_name, event)
            ),
        )
        and (event.author == 'user' or _is_other_agent_reply(agent_name, event))
        and not _is_direct_transfer(event)
        and not is_submitted_result
    ):
      return _get_contents(
          current_branch,
          events[i:],
          agent_name,
          preserve_function_call_ids=preserve_function_call_ids,
          isolation_scope=isolation_scope,
          is_single_turn=is_single_turn,
          user_content=user_content,
          include_thoughts_from_other_agents=include_thoughts_from_other_agents,
      )

  return []


def _is_event_belongs_to_branch(
    invocation_branch: str | None, event: Event
) -> bool:
  """Check if an event belongs to the current branch.

  This is for event context segregation between agents. E.g. agent A shouldn't
  see output of agent B.
  """
  if not invocation_branch or not event.branch:
    return True

  inv_path = _BranchPath.from_string(invocation_branch)
  evt_path = _BranchPath.from_string(event.branch)
  return inv_path == evt_path or inv_path.is_descendant_of(evt_path)


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


def _is_adk_framework_event(event: Event) -> bool:
  """Checks if the event is an ADK framework event."""
  return _is_function_call_event(event, 'adk_framework')


def _is_submitted_tool_result(event: Event) -> bool:
  """Whether the event is a tool result the caller posted back.

  A long-running tool is finished by calling the runner again with the result
  as the new message, which is stored as a user-authored ``function_response``
  event. It answers a call the model made earlier, so it continues the turn
  rather than starting one. Anchoring on it would cut the history above the
  matching ``function_call``, and the response left with nothing to pair with
  is then pruned as an orphan, so the request would carry no contents at all.
  """
  return event.author == 'user' and bool(event.get_function_responses())


def _is_direct_transfer(event: Event) -> bool:
  """Whether the event is a direct ``transfer_to_agent`` event.

  When ``include_contents='none'`` and control is handed to a sub-agent via
  ``transfer_to_agent``, the trailing transfer events (the function call and
  its response) must not be treated as the start of the current turn.
  Otherwise the sub-agent's turn would anchor on the parent's transfer event
  and drop the latest user input. Skipping these events lets the turn anchor
  on the real user input (or a non-transfer model request) instead, while the
  transfer events are still included as context.
  """
  return bool(
      event.actions.transfer_to_agent
      or (
          event.content
          and event.content.parts
          and any(
              p.function_call and p.function_call.name == 'transfer_to_agent'
              for p in event.content.parts
          )
      )
  )


def _is_live_model_media_event_with_inline_data(event: Event) -> bool:
  """Check if the event is a live/bidi media event (audio, video, image) with inline data.

  There are two possible cases and we only care about the second case:
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
          mime_type='audio/pcm;rate=24000'
        )
      ),
    ],
    role='model'
  ) grounding_metadata=None partial=None turn_complete=None finish_reason=None
  error_code=None error_message=None...
  """
  if not event.content or not event.content.parts:
    return False
  for part in event.content.parts:
    if part.inline_data and part.inline_data.mime_type:
      mime = part.inline_data.mime_type.lower()
      if (
          mime.startswith('audio/')
          or mime.startswith('video/')
          or mime.startswith('image/')
      ):
        return True
  return False


def _add_model_input_context_to_user_content(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_input_context: list[types.Content],
) -> None:
  """Insert transient model input context before the invocation user content."""
  if not model_input_context:
    return

  insert_index = 0
  user_content = invocation_context.user_content
  if user_content:
    for i in range(len(llm_request.contents) - 1, -1, -1):
      if llm_request.contents[i] == user_content:
        insert_index = i
        break

  llm_request.contents[insert_index:insert_index] = model_input_context


async def _add_instructions_to_user_content(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    instruction_contents: list[types.Content],
) -> None:
  """Insert instruction-related contents at proper position in conversation.

  This function inserts instruction-related contents (passed as parameter) at
  the
  proper position in the conversation flow, specifically before the last
  continuous
  batch of user content to maintain conversation context.

  Args:
    invocation_context: The invocation context
    llm_request: The LLM request to modify
    instruction_contents: List of instruction-related contents to insert
  """
  if not instruction_contents:
    return
  llm_request._insert_transient_user_content(  # pylint: disable=protected-access
      instruction_contents
  )

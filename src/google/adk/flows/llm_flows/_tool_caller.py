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

"""Single tool preparation, invocation, execution, and response event construction."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
import contextvars
import copy
import dataclasses
import inspect
import json
import logging
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
import weakref

from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.genai import types

from . import _tool_error_handler
from ...agents.active_streaming_tool import ActiveStreamingTool
from ...events.event import Event
from ...live.live_request_queue import LiveRequestQueue
from ...telemetry import _instrumentation
from ...tools.base_tool import BaseTool
from ...tools.function_tool import _use_sync_callable_runner
from ...tools.function_tool import FunctionTool
from ...tools.tool_confirmation import ToolConfirmation
from ...tools.tool_context import ToolContext
from ...utils._callback_pipeline import _run_callbacks
from ...utils._callback_pipeline import _stop_on_non_none
from ...utils.context_utils import Aclosing
from ._invocation_utils import require_agent_name as _require_agent_name

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)

# Thread pool executors for running tools in background threads, keyed by the
# event loop they serve and then by max_workers. A pool dedicated to tools keeps
# blocking tools from blocking the event loop in Live API mode without competing
# with the loop's own default executor. Each pool is shut down once its loop is
# gone, so its idle threads do not survive the loop.
_TOOL_THREAD_POOLS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[int, ThreadPoolExecutor]
] = weakref.WeakKeyDictionary()
# Loops on other threads reach this registry concurrently.
_TOOL_THREAD_POOL_LOCK = threading.Lock()

# The deepest container whose entries are searched for media: the value a tool
# returns, and one container inside it. Searching further would mean walking a
# tool's own data structures on every call, whether or not it ever returns
# media, and the bound also stops a self-referential result being walked
# forever.
_MAX_MEDIA_CONTAINER_DEPTH = 1

_MESSAGE_EVENT_FIELDS = frozenset({'content', 'id', 'timestamp'})


def _is_live_request_queue_annotation(param: inspect.Parameter) -> bool:
  """Check whether a parameter is annotated as LiveRequestQueue.

  Handles both the class itself and the string form produced by
  ``from __future__ import annotations``.
  """
  ann = param.annotation
  return ann is LiveRequestQueue or (
      isinstance(ann, str) and ann == 'LiveRequestQueue'
  )


def _normalize_tool_result(function_result: object) -> dict[str, Any]:
  """Normalizes a dynamic tool result to the documented callback shape."""
  if isinstance(function_result, dict):
    return cast(dict[str, Any], function_result)
  return {'result': function_result}


def _as_callback_result(function_result: object) -> dict[str, Any]:
  """Passes a tool result through to the after-tool callback contract.

  The contract is declared as a dict, but a tool may return any value and
  callbacks have always received it unchanged; normalizing here would alter
  what every plugin and after_tool_callback observes.
  """
  return cast(dict[str, Any], function_result)


def _get_tool_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
  """Gets or creates the running loop's thread pool executor for tool execution.

  The pool is only used for tool calls, so a blocking tool cannot starve work
  the loop itself submits to its default executor, such as name resolution.

  Args:
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    A ThreadPoolExecutor with the specified max_workers, shut down when the
    event loop that created it is collected.
  """
  loop = asyncio.get_running_loop()
  with _TOOL_THREAD_POOL_LOCK:
    pools = _TOOL_THREAD_POOLS.setdefault(loop, {})
    pool = pools.get(max_workers)
    if pool is None:
      pool = ThreadPoolExecutor(
          max_workers=max_workers, thread_name_prefix='adk_tool_executor'
      )
      pools[max_workers] = pool
      weakref.finalize(loop, pool.shutdown, wait=False)
    return pool


def _is_sync_tool(tool: BaseTool) -> bool:
  """Checks if a tool's underlying function is synchronous."""
  if not hasattr(tool, 'func'):
    return False
  func = tool.func
  return not (
      inspect.iscoroutinefunction(func)
      or inspect.isasyncgenfunction(func)
      or (
          hasattr(func, '__call__')
          and inspect.iscoroutinefunction(func.__call__)
      )
  )


async def _call_tool_in_thread_pool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    max_workers: int = 4,
) -> object:
  """Runs a tool in a thread pool to avoid blocking the event loop.

  The complete ``BaseTool.run_async`` contract is preserved. For synchronous
  ``FunctionTool`` callables, tool-owned validation, authentication, and
  confirmation stay on the caller loop while only synchronous callables enter
  the pool. Other tools run their complete async contract in a worker loop.

  Note: Due to Python's GIL, this does NOT help with pure Python CPU-bound code.
  Thread pool only helps when the GIL is released (blocking I/O, C extensions).

  Args:
    tool: The tool to execute.
    args: Arguments to pass to the tool.
    tool_context: The tool context.
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    The result of running the tool.
  """
  loop = asyncio.get_running_loop()
  executor = _get_tool_thread_pool(max_workers)

  if _is_sync_tool(tool) and isinstance(tool, FunctionTool):

    async def run_sync_callable(
        target: Callable[..., Any], call_args: dict[str, Any]
    ) -> Any:
      call_context = contextvars.copy_context()

      def invoke() -> Any:
        with _use_sync_callable_runner(None):
          return target(**call_args)

      return await loop.run_in_executor(
          executor,
          lambda: call_context.run(invoke),
      )

    with _use_sync_callable_runner(run_sync_callable):
      return await tool.run_async(args=args, tool_context=tool_context)

  ctx = contextvars.copy_context()

  def run_tool_in_new_loop() -> Any:
    return asyncio.run(tool.run_async(args=args, tool_context=tool_context))

  return await loop.run_in_executor(
      executor, lambda: ctx.run(run_tool_in_new_loop)
  )


def _get_tool(
    function_call: types.FunctionCall, tools_dict: dict[str, BaseTool]
) -> BaseTool:
  """Returns the tool corresponding to the function call."""
  tool_name = function_call.name
  if tool_name is None or tool_name not in tools_dict:
    available = list(tools_dict.keys())
    error_msg = (
        f"Tool '{tool_name}' not found.\nAvailable tools:"
        f" {', '.join(available)}\n\nPossible causes:\n  1. LLM hallucinated"
        ' the function name - review agent instruction clarity\n  2. Tool not'
        ' registered - verify agent.tools list\n  3. Name mismatch - check for'
        ' typos\n\nSuggested fixes:\n  - Review agent instruction to ensure'
        ' tool usage is clear\n  - Verify tool is included in agent.tools'
        ' list\n  - Check for typos in function name'
    )
    raise ValueError(error_msg)

  return tools_dict[tool_name]


def _create_tool_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> ToolContext:
  """Creates a ToolContext object."""
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id=function_call.id,
      tool_confirmation=tool_confirmation,
  )


def _get_tool_and_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> tuple[BaseTool, ToolContext]:
  """Returns the tool and tool context corresponding to the function call."""
  tool = _get_tool(function_call, tools_dict)
  tool_context = _create_tool_context(
      invocation_context,
      function_call,
      tool_confirmation,
  )
  return (tool, tool_context)


def _try_decode_computer_use_image(
    tool: BaseTool,
    function_result: dict[str, object],
) -> Optional[list[types.FunctionResponsePart]]:
  """Decodes the image from the function result for a computer use tool.

  Args:
    tool: The tool that produced the function result.
    function_result: The dictionary containing the function's result. This
      dictionary may be modified in-place to remove the 'image' key if an image
      is successfully decoded.

  Returns:
    A list containing a `types.FunctionResponsePart` with the decoded image
    data, or None if no image was found or decoding failed.
  """
  if not isinstance(tool, ComputerUseTool):
    return None

  image = function_result.get('image')
  if not isinstance(image, dict):
    return None
  image_data_encoded = image.get('data')
  mime_type = image.get('mimetype')
  if not isinstance(image_data_encoded, (str, bytes)) or not isinstance(
      mime_type, str
  ):
    return None

  try:
    image_data = base64.b64decode(image_data_encoded)
    part = types.FunctionResponsePart.from_bytes(
        data=image_data, mime_type=mime_type
    )
    del function_result['image']
    return [part]
  except (binascii.Error, ValueError):
    logger.exception('Failed to decode image from computer use tool')
    return None


def _as_function_response_part(
    value: object,
) -> Optional[types.FunctionResponsePart]:
  """Converts a tool-returned part into a function response part.

  Returns None when the value is not a part carrying usable media.
  """
  if not isinstance(value, types.Part):
    return None
  blob = value.inline_data
  if blob is not None and blob.data is not None and blob.mime_type:
    return types.FunctionResponsePart.from_bytes(
        data=blob.data, mime_type=blob.mime_type
    )
  file = value.file_data
  if file is not None and file.file_uri and file.mime_type:
    return types.FunctionResponsePart.from_uri(
        file_uri=file.file_uri, mime_type=file.mime_type
    )
  return None


def _extract_media_from_entry(
    value: object,
    parts: list[types.FunctionResponsePart],
    depth: int,
) -> tuple[bool, object]:
  """Removes media from one entry of a tool result.

  Any parts found are appended to ``parts``. Only dicts, lists and tuples are
  descended into, so an arbitrary object a tool returns is left alone.

  Returns:
    Whether the entry should be kept, and what is left of it. An entry that
    was media, or a container left empty once its media was taken out, is not
    kept.
  """
  part = _as_function_response_part(value)
  if part is not None:
    parts.append(part)
    return False, None
  if depth >= _MAX_MEDIA_CONTAINER_DEPTH or not isinstance(
      value, (dict, list, tuple)
  ):
    return True, value
  remaining, nested_parts = _extract_multimodal_parts(value, depth + 1)
  if not nested_parts:
    return True, value
  parts.extend(nested_parts)
  return bool(remaining), remaining


def _extract_multimodal_parts(
    function_result: object,
    depth: int = 0,
) -> tuple[object, Optional[list[types.FunctionResponsePart]]]:
  """Moves media in a tool result into function response parts.

  A tool result is otherwise required to be JSON-serializable, which leaves no
  way to hand back media except by encoding it into a string the model reads
  as text. A tool that produces an image, audio clip or document returns a
  part holding the raw bytes or a uri instead, on its own or among the entries
  of a returned container, which may itself hold a container of parts.

  Returns:
    The result with the media removed, and the extracted parts. The parts are
    None when the result carries no media, in which case the result is
    returned unchanged.
  """
  single_part = _as_function_response_part(function_result)
  if single_part is not None:
    return {}, [single_part]

  parts: list[types.FunctionResponsePart] = []
  remaining: object
  if isinstance(function_result, dict):
    kept_items = {}
    for key, value in function_result.items():
      keep, kept = _extract_media_from_entry(value, parts, depth)
      if keep:
        kept_items[key] = kept
    remaining = kept_items
  elif isinstance(function_result, (list, tuple)):
    kept_values = []
    for value in function_result:
      keep, kept = _extract_media_from_entry(value, parts, depth)
      if keep:
        kept_values.append(kept)
    remaining = kept_values
  else:
    return function_result, None

  if not parts:
    return function_result, None
  return remaining or {}, parts


async def _call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> object:
  """Calls the tool."""
  result: object = await tool.run_async(args=args, tool_context=tool_context)
  return result


def _build_function_response_content(
    tool: BaseTool,
    function_result: object,
    function_call_id: Optional[str],
    function_response_parts: Optional[list[types.FunctionResponsePart]] = None,
) -> types.Content:
  """Builds the content carrying a tool result as a FunctionResponse."""
  # A streaming tool that wants a different Live scheduling mode for one
  # particular chunk hands back a FunctionResponse holding that chunk's
  # payload and mode. Only those two fields are read: `id` and `name` have to
  # address the function call being answered, which a tool cannot know, so ADK
  # keeps owning them. Unwrapped before the extraction below so that media in
  # the payload is still reachable.
  scheduling_override = None
  if isinstance(function_result, types.FunctionResponse):
    scheduling_override = function_result.scheduling
    function_result = function_result.response

  if function_response_parts is None:
    function_result, function_response_parts = _extract_multimodal_parts(
        function_result
    )

  # Specs requires the result to be a dict.
  if not isinstance(function_result, dict):
    function_result = {'result': function_result}

  part_function_response = types.Part.from_function_response(
      name=tool.name,
      response=function_result,
      parts=function_response_parts,
  )
  function_response = part_function_response.function_response
  if function_response is None:
    raise RuntimeError('Function response part was not created.')
  function_response.id = function_call_id
  # A scheduling asked for on this one result wins over the tool-wide default,
  # which is the fallback for every result that does not name one.
  effective_scheduling = (
      scheduling_override
      if scheduling_override is not None
      else tool.response_scheduling
  )
  if effective_scheduling is not None:
    function_response.scheduling = effective_scheduling

  return types.Content(role='user', parts=[part_function_response])


def _build_response_event(
    tool: BaseTool,
    function_result: object,
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> Event:
  """Builds a function response Event from tool results and context."""
  # Capture the raw result for display purposes before any normalization.
  display_result = function_result
  # Media has to come out before the result is coerced to a dict, so that a
  # media part returned on its own or inside a list is still reachable.
  remaining_result, function_response_parts = _extract_multimodal_parts(
      function_result
  )
  # The callback and FunctionResponse contracts require a string-keyed dict.
  function_result = _normalize_tool_result(remaining_result)

  if function_response_parts is None and isinstance(tool, ComputerUseTool):
    function_response_parts = _try_decode_computer_use_image(
        tool, function_result
    )

  content = _build_function_response_content(
      tool,
      function_result,
      tool_context.function_call_id,
      function_response_parts,
  )

  # When summarization is skipped, ensure a displayable text part is added so
  # the tool's output is not lost in UIs that don't render function responses.
  # Control-flow tools (e.g. exit_loop) are also skipped to avoid emitting a
  # noisy "null" text part.
  has_displayable_result = (
      display_result is not None
      and display_result != {'result': None}
      and display_result != ''
  )
  if (
      tool_context.actions.skip_summarization
      and 'error' not in function_result
      and has_displayable_result
  ):
    # Imported lazily: AgentTool is only needed on the skip-summarization
    # path, so it is not worth pulling into every functions.py import.
    from ...tools.agent_tool import AgentTool

    # This is scoped to AgentTool deliberately: other tools (e.g. UI/widget-
    # rendering tools) set skip_summarization precisely because their function
    # response is an internal acknowledgement that must NOT be surfaced as
    # visible text. AgentTool subclasses can still return None (e.g.
    # _SingleTurnAgentTool delegating to run_node), hence the
    # has_displayable_result guard above.
    if isinstance(tool, AgentTool):
      if isinstance(display_result, str):
        result_text = display_result
      else:
        result_text = json.dumps(
            display_result, ensure_ascii=False, default=str
        )
      if content.parts is None:
        raise RuntimeError('Function response content must contain parts.')
      content.parts.append(types.Part.from_text(text=result_text))

  # Builds the function response event.
  return Event(
      invocation_id=invocation_context.invocation_id,
      author=_require_agent_name(invocation_context),
      content=content,
      actions=tool_context.actions,
      branch=invocation_context.branch,
  )


def _message_content_for_user(
    event: Event, *, tool: BaseTool
) -> Optional[types.Content]:
  """Returns the content to deliver, or None if the event has no message.

  Only the ``content`` field is considered for delivery. All other fields are
  ignored. The role is set to "user", overriding any other value.

  Args:
    event: The event the tool yielded.
    tool: The tool that yielded it, named in the warning.

  Returns:
    The content to send to the user, or None if there is nothing to send.
  """
  problem = None
  if not event.content:
    problem = 'it has no content, so there is nothing to deliver'
  # Load-bearing beside exclude_defaults: a field with a custom serializer
  # skips the default comparison, so ``long_running_tool_ids`` reports as
  # set on every event. This reads the raw value instead.
  # Only the presence of a field is read, so a mistyped value is not worth
  # a warning of its own.
  elif event.model_dump(
      exclude=set(_MESSAGE_EVENT_FIELDS),
      exclude_defaults=True,
      exclude_none=True,
      warnings=False,
  ):
    problem = 'it sets fields beyond the message, which are ignored'

  if problem:
    logger.warning(
        'Streaming tool `%s` yielded an Event that is not a purely'
        ' user-facing message: %s. To send a message, use Event(message=...)',
        tool.name,
        problem,
    )
  if not event.content:
    return None
  return event.content.model_copy(deep=True, update={'role': 'user'})


async def _emit_streaming_tool_event(
    event: Event,
    *,
    tool: BaseTool,
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> None:
  """Streams an Event yielded by a streaming tool to the user.

  Args:
    event: The event the tool yielded.
    tool: The tool that yielded it, named in the branch and in any warning.
    tool_context: The context of the call, for its function call id.
    invocation_context: The invocation to enqueue on.
  """
  content = _message_content_for_user(event, tool=tool)
  if content is None:
    return
  # Built fresh rather than copied, so the delivered event carries the message
  # and nothing else, and each delivery gets its own id and timestamp: a tool
  # may hold one Event and yield it twice, and the session orders events and
  # decides what compaction has already summarized by timestamp.
  await invocation_context._enqueue_event(
      Event(
          content=content,
          author=_require_agent_name(invocation_context),
          invocation_id=invocation_context.invocation_id,
          branch=(
              f'{tool.name}@{tool_context.function_call_id}'
              if tool_context.function_call_id
              else tool.name
          ),
      )
  )


@dataclasses.dataclass
class _PreparedFunctionCall:
  """One function call taken as far as its before-tool callbacks.

  Attributes:
    function_call: The call the model made.
    tool: The tool it names, or a placeholder tool when the name is unknown.
    tool_context: The context the tool and all of its callbacks share.
    function_args: The deep copy of the call arguments handed to the tool.
    contextvars_snapshot: The `contextvars` context as the before-tool callbacks
      left it. The execute phase runs the tool in this context, so a contextvar
      one of those callbacks set is still set when the tool reads it.
    override_response: A response that already answers the call, so the tool
      does not run: either what a before-tool callback returned, or the
      not-found payload for a tool name the model invented. None means the tool
      still has to run.
    tool_lookup_error: The lookup failure, when the tool name was unknown.
    is_tool_lookup_failure: Whether `override_response` answers a failed lookup
      rather than coming from a before-tool callback. Such a response skips the
      after-tool callbacks, which describe a tool that never ran.
  """

  function_call: types.FunctionCall
  tool: BaseTool
  tool_context: ToolContext
  function_args: dict[str, Any]
  contextvars_snapshot: contextvars.Context
  override_response: Optional[object] = None
  tool_lookup_error: Optional[Exception] = None
  is_tool_lookup_failure: bool = False


async def _prepare_single(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> _PreparedFunctionCall:
  """Resolves one call's tool and runs its before-tool callbacks.

  This is steps 1 and 2 of the tool pipeline, plus the handling of a tool name
  that resolves to nothing. Nothing here runs the tool.
  """
  # Do not use "args" as the variable name, because it is a reserved keyword
  # in python debugger.
  # Make a deep copy to avoid being modified.
  function_args = (
      copy.deepcopy(function_call.args) if function_call.args else {}
  )
  tool_context = _create_tool_context(
      invocation_context, function_call, tool_confirmation
  )

  tool_lookup_error: Exception | None = None
  try:
    tool = _get_tool(function_call, tools_dict)
  except ValueError as tool_error:
    tool = BaseTool(
        name=function_call.name or '<unnamed>', description='Tool not found'
    )
    # Defer error handling until the before-tool callbacks have run, so that
    # one of them can still answer the call.
    tool_lookup_error = tool_error

  # Step 1: Check if plugin before_tool_callback overrides the function
  # response.
  override_response: object | None = (
      await invocation_context.plugin_manager.run_before_tool_callback(
          tool=tool, tool_args=function_args, tool_context=tool_context
      )
  )

  # Step 2: If no overrides are provided from the plugins, further run the
  # canonical callback.
  if override_response is None:
    override_response = await _run_callbacks(
        agent.canonical_before_tool_callbacks,  # type: ignore[arg-type]
        _stop_on_non_none,
        tool=tool,
        args=function_args,
        tool_context=tool_context,
    )

  # Handle tool lookup failure if before-tool callbacks did not override the
  # response.
  is_tool_lookup_failure = False
  if override_response is None and tool_lookup_error is not None:
    is_tool_lookup_failure = True
    override_response = await _tool_error_handler.run_on_tool_error_callbacks(
        invocation_context=invocation_context,
        agent=agent,
        tool=tool,
        tool_args=function_args,
        tool_context=tool_context,
        error=tool_lookup_error,
    )
    if override_response is None:
      logger.warning('%s', tool_lookup_error)
      override_response = _tool_error_handler.build_tool_not_found_response(
          tool.name, tools_dict
      )

  return _PreparedFunctionCall(
      function_call=function_call,
      tool=tool,
      tool_context=tool_context,
      function_args=function_args,
      # Taken here, at the end of the prepare phase, so it carries whatever the
      # before-tool callbacks just set.
      contextvars_snapshot=contextvars.copy_context(),
      override_response=override_response,
      tool_lookup_error=tool_lookup_error,
      is_tool_lookup_failure=is_tool_lookup_failure,
  )


async def _execute_single_prepared_call(
    invocation_context: InvocationContext,
    prepared_call: _PreparedFunctionCall,
    agent: LlmAgent,
    *,
    tool_runner: Callable[[], Awaitable[Any]],
) -> Optional[Event]:
  """Runs one prepared function call and builds its response event.

  Args:
    invocation_context: The invocation context.
    prepared_call: The prepared function call holding tool, args, context, etc.
    agent: The agent owning the call.
    tool_runner: An async callable that invokes the tool logic when no
      before-tool callback overrides the response.

  Returns:
    The built function response Event, or None if response is deferred/omitted.
  """
  tool = prepared_call.tool
  tool_context = prepared_call.tool_context
  function_args = prepared_call.function_args
  function_response = prepared_call.override_response
  detected_error_type: Optional[str] = None

  async def _run_with_trace() -> Event | None:
    """Executes the tool with full lifecycle management and telemetry.

    This function orchestrates the rest of the tool execution pipeline,
    including:
    1. Executing the actual tool logic.
    2. Running plugin and canonical after-tool callbacks.
    3. Detecting error types for telemetry.
    4. Building the final FunctionResponse Event to be returned.
    """
    nonlocal function_response, detected_error_type

    # A response the prepare phase built for a tool that was never found
    # answers the call as it is: the after-tool callbacks describe a tool run
    # that did not happen.
    if prepared_call.is_tool_lookup_failure:
      detected_error_type = type(prepared_call.tool_lookup_error).__name__
      return _build_response_event(
          tool, function_response, tool_context, invocation_context
      )

    # Step 3: No before-tool callback answered the call, so proceed calling
    # the tool normally.
    if function_response is None:
      try:
        function_response = await tool_runner()
      except Exception as tool_error:
        error_response = await _tool_error_handler.run_on_tool_error_callbacks(
            invocation_context=invocation_context,
            agent=agent,
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            error=tool_error,
        )
        if error_response is not None:
          function_response = error_response
        else:
          raise tool_error

    # Step 4: Check if plugin after_tool_callback overrides the function
    # response.
    callback_tool_response = _as_callback_result(function_response)
    altered_function_response = (
        await invocation_context.plugin_manager.run_after_tool_callback(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            result=callback_tool_response,
        )
    )

    # Step 5: If no overrides are provided from the plugins, further run the
    # canonical after_tool_callbacks.
    if altered_function_response is None:
      altered_function_response = await _run_callbacks(
          agent.canonical_after_tool_callbacks,  # type: ignore[arg-type]
          _stop_on_non_none,
          tool=tool,
          args=function_args,
          tool_context=tool_context,
          tool_response=callback_tool_response,
      )

    # Step 6: If alternative response exists from after_tool_callback, use it
    # instead of the original function response.
    if altered_function_response is not None:
      function_response = altered_function_response

    if (
        tool.is_long_running or tool._defers_response
    ) and not function_response:
      # The tool either runs long (FR will arrive later via session
      # injection) or defers its response by design (e.g., the LlmAgent
      # wrapper for task delegation synthesizes the FR after the
      # sub-agent completes).  Either way, skip the auto-FR build when
      # the tool returned nothing.  Truthiness is deliberate here, unlike
      # the callback chains above: the real FR still arrives later, so an
      # empty dict must not answer the call early.
      return None

    detected_error_type = _tool_error_handler.detect_error_type_for_telemetry(
        tool, tool_context, function_response
    )

    # Note: State deltas are not applied here - they are collected in
    # tool_context.actions.state_delta and applied later when the session
    # service processes the events
    return _build_response_event(
        tool, function_response, tool_context, invocation_context
    )

  async with _instrumentation.record_tool_execution(
      tool, agent, function_args, invocation_context=invocation_context
  ) as tel_ctx:
    tel_ctx.function_response_event = await _run_with_trace()
    tel_ctx.error_type = detected_error_type
    return tel_ctx.function_response_event


async def _execute_single_prepared_call_async(
    invocation_context: InvocationContext,
    prepared_call: _PreparedFunctionCall,
    agent: LlmAgent,
) -> Optional[Event]:
  """Runs one prepared function call and builds its response event.

  This is steps 3 to 6 of the tool pipeline: run the tool unless the prepare
  phase already answered the call, run the after-tool callbacks, and turn the
  result into an event. State modifications stay thread safe because each call
  owns its own ToolContext.
  """
  return await _execute_single_prepared_call(
      invocation_context,
      prepared_call,
      agent,
      tool_runner=lambda: _call_tool_async(
          prepared_call.tool,
          args=prepared_call.function_args,
          tool_context=prepared_call.tool_context,
      ),
  )


async def _execute_single_prepared_call_live(
    invocation_context: InvocationContext,
    prepared_call: _PreparedFunctionCall,
    agent: LlmAgent,
    active_tools_lock: asyncio.Lock,
) -> Optional[Event]:
  """Runs one prepared function call in live mode.

  This is the live counterpart of `_execute_single_prepared_call_async`: steps
  3 to 6 of the tool pipeline, with the tool call itself going through
  `_process_function_live_helper`.
  """
  return await _execute_single_prepared_call(
      invocation_context,
      prepared_call,
      agent,
      tool_runner=lambda: _process_function_live_helper(
          prepared_call.tool,
          prepared_call.tool_context,
          prepared_call.function_call,
          prepared_call.function_args,
          invocation_context,
          active_tools_lock,
      ),
  )


async def _process_function_live_helper(
    tool: BaseTool,
    tool_context: ToolContext,
    function_call: types.FunctionCall,
    function_args: dict[str, Any],
    invocation_context: InvocationContext,
    active_tools_lock: asyncio.Lock,
) -> object:
  """Handles dispatching of live tool calls (stop_streaming, generator tools, thread pool)."""
  function_response: object = None

  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    if not isinstance(function_name, str):
      raise ValueError('stop_streaming requires a string function_name.')
    # Thread-safe access to active_streaming_tools
    async with active_tools_lock:
      active_tasks = invocation_context.active_streaming_tools
      active_task = (
          active_tasks[function_name].task
          if active_tasks and function_name in active_tasks
          else None
      )
      task = active_task if active_task and not active_task.done() else None

    if task:
      task.cancel()
      try:
        # Wait for the task to be cancelled
        await asyncio.wait_for(task, timeout=1.0)
      except (asyncio.CancelledError, asyncio.TimeoutError):
        # Log the specific condition
        if task.cancelled():
          logging.info('Task %s was cancelled successfully', function_name)
        elif task.done():
          logging.info('Task %s completed during cancellation', function_name)
        else:
          logging.warning(
              'Task %s might still be running after cancellation timeout',
              function_name,
          )
          function_response = {
              'status': f'The task is not cancelled yet for {function_name}.'
          }
      if not function_response:
        # Clean up the reference under lock
        async with active_tools_lock:
          if (
              invocation_context.active_streaming_tools
              and function_name in invocation_context.active_streaming_tools
          ):
            invocation_context.active_streaming_tools[function_name].task = None
            invocation_context.active_streaming_tools[function_name].stream = (
                None
            )

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif hasattr(tool, 'func') and inspect.isasyncgenfunction(
      cast('FunctionTool', tool).func
  ):
    # for streaming tool use case
    # we require the function to be an async generator function
    streaming_tool = cast('FunctionTool', tool)

    async def run_tool_and_update_queue(
        tool: FunctionTool,
        function_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
      live_request_queue = invocation_context.live_request_queue
      if live_request_queue is None:
        raise RuntimeError('Streaming tools require a live request queue.')
      try:
        res = await _call_tool_async(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
        )
        if inspect.isasyncgen(res):
          async with Aclosing(res) as agen:
            async for result in agen:
              if isinstance(result, Event):
                await _emit_streaming_tool_event(
                    result,
                    tool=tool,
                    tool_context=tool_context,
                    invocation_context=invocation_context,
                )
                continue

              updated_content = _build_function_response_content(
                  tool, result, tool_context.function_call_id
              )
              live_request_queue.send_content(updated_content, partial=True)
        else:
          # `res` is a single terminal payload (e.g. the error dict returned
          # when confirmation is required/rejected or a mandatory argument is
          # missing), not a chunk of a stream.
          # TODO: for the confirmation-required case, hold the call pending
          # (as long-running tools do) instead of relaying the error. Relaying
          # it closes the call id with the model, so a later approval would
          # have to send a second response reusing that same id.
          updated_content = _build_function_response_content(
              tool, res, tool_context.function_call_id
          )
          live_request_queue.send_content(updated_content, partial=False)
      except asyncio.CancelledError:
        raise
      except Exception:
        # The model already got a `pending` response for this call, so it waits
        # for a follow-up FunctionResponse. Swallowing the exception here would
        # leave the live session hanging, so report the failure to the model.
        # The exception text is deliberately not forwarded to the model: it can
        # carry internal detail that is irrelevant to it. It is logged instead.
        logger.exception('Error executing streaming tool %s.', tool.name)
        error_content = _build_function_response_content(
            tool,
            {
                'error': (
                    f'Invoking `{tool.name}()` failed with an internal error.'
                )
            },
            tool_context.function_call_id,
        )
        live_request_queue.send_content(error_content, partial=False)

    # TODO: resolve `require_confirmation` before spawning the task. The
    # confirmation request is recorded on `tool_context.actions` by the
    # background task while the caller builds the response event, and nothing
    # orders the two, so the request can be missing from the emitted event.
    task = asyncio.create_task(
        run_tool_and_update_queue(streaming_tool, function_args, tool_context)
    )

    async with active_tools_lock:
      if invocation_context.active_streaming_tools is None:
        invocation_context.active_streaming_tools = {}
      if tool.name in invocation_context.active_streaming_tools:
        invocation_context.active_streaming_tools[tool.name].task = task
      else:
        # Register the streaming tool lazily when the model calls it.
        invocation_context.active_streaming_tools[tool.name] = (
            ActiveStreamingTool(task=task)
        )
        logger.debug('Lazily registered streaming tool: %s', tool.name)

      # For input-streaming tools (those with `input_stream:
      # LiveRequestQueue`), create a dedicated LiveRequestQueue so
      # _send_to_model starts duplicating data to it. This also
      # handles re-invocation after stop_streaming reset .stream
      # to None.
      sig = inspect.signature(streaming_tool.func)
      if (
          'input_stream' in sig.parameters
          and _is_live_request_queue_annotation(sig.parameters['input_stream'])
      ):
        invocation_context.active_streaming_tools[tool.name].stream = (
            LiveRequestQueue()
        )

    # Immediately return a pending response.
    # This is required by current live model.
    function_response = {
        'status': (
            'The function is running asynchronously and the results are'
            ' pending.'
        )
    }
  else:
    # Check if we should run tools in thread pool to avoid blocking event loop
    run_config = invocation_context.run_config
    if run_config is None:
      raise RuntimeError('Live function execution requires a run config.')
    thread_pool_config = run_config.tool_thread_pool_config
    if thread_pool_config is not None:
      function_response = await _call_tool_in_thread_pool(
          tool,
          args=function_args,
          tool_context=tool_context,
          max_workers=thread_pool_config.max_workers,
      )
    else:
      function_response = await _call_tool_async(
          tool, args=function_args, tool_context=tool_context
      )
  return function_response

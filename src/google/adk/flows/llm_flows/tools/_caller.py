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

import base64
import binascii
from collections.abc import Awaitable
import contextlib
import contextvars
import copy
import dataclasses
import json
import logging
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.genai import types

from . import _error_handler as _tool_error_handler
from ....events.event import Event
from ....telemetry import _instrumentation
from ....tools._confirmation_utils import apply_confirmation_gate
from ....tools.base_tool import BaseTool
from ....tools.function_tool import FunctionTool
from ....tools.tool_confirmation import ToolConfirmation
from ....tools.tool_context import ToolContext
from ....utils._callback_pipeline import _run_callbacks
from ....utils._callback_pipeline import _stop_on_non_none
from ..core._utils import require_agent_name as _require_agent_name
from ._thread_pool import _get_tool_thread_pool
from ._thread_pool import _is_sync_tool
from ._thread_pool import _use_executor_for_sync_callables

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext
  from ....agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)

# The deepest container whose entries are searched for media: the value a tool
# returns, and one container inside it. Searching further would mean walking a
# tool's own data structures on every call, whether or not it ever returns
# media, and the bound also stops a self-referential result being walked
# forever.
_MAX_MEDIA_CONTAINER_DEPTH = 1


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
    # Imported lazily: AgentTool and NodeTool are only needed on the
    # skip-summarization path, so they are not worth pulling into every
    # functions.py import.
    from ....tools._node_tool import NodeTool
    from ....tools.agent_tool import AgentTool

    # This is scoped to AgentTool and NodeTool deliberately: other tools (e.g.
    # UI/widget-rendering tools) set skip_summarization precisely because their
    # function response is an internal acknowledgement that must NOT be surfaced
    # as visible text. AgentTool subclasses can still return None (e.g.
    # _SingleTurnAgentTool delegating to run_node), hence the
    # has_displayable_result guard above.
    if isinstance(tool, (AgentTool, NodeTool)):
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


@dataclasses.dataclass
class _PreparedFunctionCall:
  """One function call resolved to the tool it names.

  Attributes:
    function_call: The call the model made.
    tool: The tool it names, or a placeholder tool when the name is unknown.
    tool_context: The context the tool and all of its callbacks share.
    function_args: The deep copy of the call arguments handed to the tool.
    contextvars_snapshot: The `contextvars` context the prepare phase ran in.
      The execute phase runs in a copy of it.
    tools_dict: The tools the call was resolved against, listed back to the
      model when it names one that does not exist.
    tool_lookup_error: The lookup failure, when the tool name was unknown.
  """

  function_call: types.FunctionCall
  tool: BaseTool
  tool_context: ToolContext
  function_args: dict[str, Any]
  contextvars_snapshot: contextvars.Context
  tools_dict: dict[str, BaseTool] = dataclasses.field(default_factory=dict)
  tool_lookup_error: Optional[Exception] = None


async def _prepare_single(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> _PreparedFunctionCall:
  """Resolves one call to the tool it names and the context it will run in.

  No callback and no tool runs here. The before-tool callbacks belong to the
  execute phase, next to the after-tool callbacks they pair with. Running them
  here runs every call's before-tool callback before any tool runs, so a turn
  in which nothing awaits no longer completes one call before starting the
  next. Calls whose callbacks or tools do await still interleave, and each one
  keeps its own state on its own `ToolContext`.
  """
  del agent  # The callbacks it owns run in the execute phase.
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

  return _PreparedFunctionCall(
      function_call=function_call,
      tool=tool,
      tool_context=tool_context,
      function_args=function_args,
      contextvars_snapshot=contextvars.copy_context(),
      tools_dict=tools_dict,
      tool_lookup_error=tool_lookup_error,
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
  function_response: object | None = None
  detected_error_type: Optional[str] = None
  response_source: _instrumentation.ToolResponseSource | None = None

  async def _run_with_trace() -> Event | None:
    """Executes the tool with full lifecycle management and telemetry.

    This function orchestrates the tool execution pipeline, including:
    1. Running plugin and canonical before-tool callbacks.
    2. Executing the actual tool logic.
    3. Running plugin and canonical after-tool callbacks.
    4. Detecting error types for telemetry.
    5. Building the final FunctionResponse Event to be returned.
    """
    nonlocal function_response, detected_error_type, response_source

    # Step 1: Check if plugin before_tool_callback overrides the function
    # response.
    function_response = (
        await invocation_context.plugin_manager.run_before_tool_callback(
            tool=tool, tool_args=function_args, tool_context=tool_context
        )
    )

    # Step 2: If no overrides are provided from the plugins, further run the
    # canonical callback.
    if function_response is None:
      function_response = await _run_callbacks(
          agent.canonical_before_tool_callbacks,  # type: ignore[arg-type]
          _stop_on_non_none,
          tool=tool,
          args=function_args,
          tool_context=tool_context,
      )
    if function_response is not None:
      response_source = 'before_tool_callback'

    # A tool name that resolved to nothing is answered once the before-tool
    # callbacks have had their chance to answer it themselves. The after-tool
    # callbacks are skipped: they describe a tool run that did not happen.
    if (
        function_response is None
        and prepared_call.tool_lookup_error is not None
    ):
      detected_error_type = type(prepared_call.tool_lookup_error).__name__
      function_response = await _tool_error_handler.run_on_tool_error_callbacks(
          invocation_context=invocation_context,
          agent=agent,
          tool=tool,
          tool_args=function_args,
          tool_context=tool_context,
          error=prepared_call.tool_lookup_error,
      )
      if function_response is None:
        logger.warning('%s', prepared_call.tool_lookup_error)
        function_response = _tool_error_handler.build_tool_not_found_response(
            tool.name, prepared_call.tools_dict
        )
      else:
        response_source = 'on_tool_error_callback'
      return _build_response_event(
          tool, function_response, tool_context, invocation_context
      )

    # Step 3: No before-tool callback answered the call, so proceed calling
    # the tool normally. A tool that requires confirmation is answered by the
    # gate instead, so the gate holds for every tool rather than only the ones
    # that check it themselves, and a gate that raises is handled like a tool
    # that raises.
    if function_response is None:
      try:
        function_response = await apply_confirmation_gate(
            tool, function_args, tool_context
        )
        if function_response is None:
          function_response = await tool_runner()
        elif tool_context.tool_confirmation is None:
          # The pause is not a tool result for the model to summarize; without
          # this the flow re-invokes the model, which calls the tool again.
          tool_context.actions.skip_summarization = True
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
          response_source = 'on_tool_error_callback'
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
      if altered_function_response is not callback_tool_response:
        response_source = 'after_tool_callback'
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
    tel_ctx.response_source = response_source
    return tel_ctx.function_response_event


async def _call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> object:
  """Calls the tool."""
  result: object = await tool.run_async(args=args, tool_context=tool_context)
  return result


async def _execute_single_prepared_call_async(
    invocation_context: InvocationContext,
    prepared_call: _PreparedFunctionCall,
    agent: LlmAgent,
) -> Optional[Event]:
  """Runs one prepared function call and builds its response event.

  This is steps 1 to 6 of the tool pipeline: run the before-tool callbacks, run
  the tool unless one of them answered the call, run the after-tool callbacks,
  and turn the result into an event. State modifications stay thread safe
  because each call owns its own ToolContext.

  With `RunConfig.tool_thread_pool_config` set, a synchronous `FunctionTool`
  calls its function on the tool thread pool, and a `NodeTool` runs synchronous
  `FunctionNode` callables (including those inside a `Workflow`) there. Every
  other tool, async function tools included, runs on the event loop as it does
  without the config.
  """
  tool = prepared_call.tool
  run_config = invocation_context.run_config
  thread_pool_config = (
      run_config.tool_thread_pool_config if run_config else None
  )

  async def call_tool() -> object:
    from ....tools._node_tool import NodeTool

    sync_callables: contextlib.AbstractContextManager[None] = (
        contextlib.nullcontext()
    )
    if (
        thread_pool_config is not None
        and _is_sync_tool(tool)
        and isinstance(tool, (FunctionTool, NodeTool))
    ):
      sync_callables = _use_executor_for_sync_callables(
          _get_tool_thread_pool(thread_pool_config.max_workers)
      )
    with sync_callables:
      return await _call_tool_async(
          tool,
          args=prepared_call.function_args,
          tool_context=prepared_call.tool_context,
      )

  return await _execute_single_prepared_call(
      invocation_context, prepared_call, agent, tool_runner=call_tool
  )

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

"""Translates Antigravity SDK trajectory steps into ADK events.

Kept separate from the agent wrapper so the mapping rules stay readable and
independently testable.

Scope: model text (final and, in SSE streaming mode, partial thinking/text
deltas), function calls, function responses, and ``final_model_text`` for
reading an event's text back out.

A client-side tool's result never reaches the trajectory; it arrives through
the post-tool-call hook instead (see ``_tool_result_capture``), which is why
``drain_tool_results`` is also called once at the end of a turn.

TODO: Surface SYSTEM_MESSAGE steps (emitted on turn cancellation) as ADK
events; they are currently dropped.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from google.antigravity import types as sdk_types
from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import JsonValue

from ...events.event import Event

if TYPE_CHECKING:
  from . import _tool_result_capture
  from ...agents.invocation_context import InvocationContext


def _build_tool_call_id(
    step: sdk_types.Step, call: sdk_types.ToolCall, index: int = 0
) -> str:
  """A stable tool-call id, synthesized when the Antigravity SDK omits one."""
  if isinstance(call.id, str) and call.id:
    return str(call.id)
  if call.args:
    args_key = json.dumps(call.args, sort_keys=True, default=str)
    return f'{step.step_index}-{call.name}-{args_key}'
  return f'{step.step_index}-{index}-{call.name}'


def _partial_event(
    ctx: InvocationContext, author: str, part: genai_types.Part
) -> Event:
  """Builds a partial model event carrying a single streamed delta part."""
  return Event(
      invocation_id=ctx.invocation_id,
      author=author,
      branch=ctx.branch,
      content=genai_types.Content(role='model', parts=[part]),
      partial=True,
  )


def _convert_partial_deltas(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
) -> list[Event]:
  """Converts a model step's incremental deltas into partial events.

  Only called in SSE streaming mode. ``thinking_delta`` and ``content_delta``
  are independent (a step may carry either or both); thinking is emitted first,
  matching the Antigravity SDK's own chunk ordering.
  """
  if step.source != sdk_types.StepSource.MODEL:
    return []

  events = []
  if step.thinking_delta:
    events.append(
        _partial_event(
            ctx,
            author,
            genai_types.Part(text=step.thinking_delta, thought=True),
        )
    )
  if step.content_delta:
    events.append(
        _partial_event(
            ctx, author, genai_types.Part.from_text(text=step.content_delta)
        )
    )
  return events


def _step_custom_metadata(
    step: sdk_types.Step,
) -> dict[str, JsonValue] | None:
  """Returns trajectory hierarchy metadata when present on the step."""
  trajectory_id = (
      step.trajectory_id if isinstance(step.trajectory_id, str) else ''
  )
  parent_trajectory_id = (
      step.parent_trajectory_id
      if isinstance(step.parent_trajectory_id, str)
      else ''
  )
  depth = step.depth if isinstance(step.depth, int) and step.depth > 0 else 0
  if not trajectory_id and not parent_trajectory_id and depth <= 0:
    return None
  step_index = step.step_index if isinstance(step.step_index, int) else 0
  metadata: dict[str, JsonValue] = {
      'trajectory_id': trajectory_id,
      'step_index': step_index,
      'depth': depth,
  }
  if parent_trajectory_id:
    metadata['parent_trajectory_id'] = parent_trajectory_id
  return metadata


def _unemitted_thinking(
    thinking: str,
    thought_key: str,
    seen_thought_steps: dict[str, str] | None,
) -> str:
  """Returns the portion of ``thinking`` not yet emitted for ``thought_key``."""
  if seen_thought_steps is None:
    return thinking
  emitted = seen_thought_steps.get(thought_key, '')
  if thinking == emitted:
    return ''
  if emitted and thinking.startswith(emitted):
    return thinking[len(emitted) :]
  return thinking


def _record_emitted_thinking(
    thinking: str,
    thought_key: str,
    seen_thought_steps: dict[str, str] | None,
) -> None:
  """Records cumulative ``thinking`` as emitted for ``thought_key``."""
  if seen_thought_steps is None:
    return
  seen_thought_steps[thought_key] = thinking


def _consume_step_thought_part(
    step: sdk_types.Step,
    seen_thought_steps: dict[str, str] | None,
    *,
    trajectory_key: str = '',
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
) -> genai_types.Part | None:
  """Returns a thought Part if the step has unseen thinking text, else None."""
  thinking = step.thinking
  if not isinstance(thinking, str) or not thinking:
    return None
  effective_trajectory = trajectory_key or (
      step.trajectory_id if isinstance(step.trajectory_id, str) else ''
  )
  thought_key = f'{effective_trajectory}:{step.step_index}'
  unemitted = _unemitted_thinking(thinking, thought_key, seen_thought_steps)
  if not unemitted:
    return None
  _record_emitted_thinking(thinking, thought_key, seen_thought_steps)
  if latest_thoughts is not None:
    latest_thoughts.pop(effective_trajectory, None)
  return genai_types.Part(text=unemitted, thought=True)


def _consume_cached_thought_part(
    trajectory_key: str,
    seen_thought_steps: dict[str, str] | None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None,
) -> genai_types.Part | None:
  """Pops and returns the cached thought Part for ``trajectory_key``."""
  if latest_thoughts is None:
    return None
  cached = latest_thoughts.pop(trajectory_key, None)
  if cached is None:
    return None
  cached_key, cached_part = cached
  cached_text = cached_part.text or ''
  prior = (
      seen_thought_steps.get(cached_key, '')
      if seen_thought_steps is not None
      else ''
  )
  full_thinking = f'{prior}{cached_text}'
  unemitted = _unemitted_thinking(full_thinking, cached_key, seen_thought_steps)
  if not unemitted:
    return None
  _record_emitted_thinking(full_thinking, cached_key, seen_thought_steps)
  return genai_types.Part(text=unemitted, thought=True)


def _should_flush_pending_calls(step: sdk_types.Step) -> bool:
  """Returns True when buffered ACTIVE function calls must be flushed."""
  return (
      (
          step.type == sdk_types.StepType.TOOL_CALL
          and step.status
          in (sdk_types.StepStatus.DONE, sdk_types.StepStatus.ERROR)
      )
      or bool(step.is_complete_response)
      or (
          step.source == sdk_types.StepSource.MODEL
          and bool(step.tool_calls)
          and step.status != sdk_types.StepStatus.ACTIVE
      )
  )


def _prepend_thought_to_pending(
    pending_function_calls: list[Event],
    thought_part: genai_types.Part,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> None:
  """Prepends ``thought_part`` to the first pending call, replacing or inserting."""
  first_content = pending_function_calls[0].content
  if first_content and first_content.parts is not None:
    for idx, part in enumerate(first_content.parts):
      if bool(part.thought):
        first_content.parts[idx] = thought_part
        break
    else:
      first_content.parts.insert(0, thought_part)
  if custom_metadata is not None:
    pending_function_calls[0].custom_metadata = custom_metadata


def _flush_pending_function_calls(
    step: sdk_types.Step,
    *,
    trajectory_key: str,
    pending_function_calls: list[Event],
    seen_tool_calls: set[str] | None = None,
    seen_thought_steps: dict[str, str] | None = None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
    step_meta: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Flushes buffered function calls, attaching any cached thought Part."""
  cached_part = _consume_step_thought_part(
      step,
      seen_thought_steps,
      trajectory_key=trajectory_key,
      latest_thoughts=latest_thoughts,
  )
  if cached_part is None:
    cached_part = _consume_cached_thought_part(
        trajectory_key, seen_thought_steps, latest_thoughts
    )
  if cached_part is not None:
    _prepend_thought_to_pending(pending_function_calls, cached_part)
  if step_meta is not None and step.type == sdk_types.StepType.TOOL_CALL:
    for pending_event in pending_function_calls:
      pending_event.custom_metadata = step_meta
  flushed = list(pending_function_calls)
  pending_function_calls.clear()
  if seen_tool_calls is not None:
    for event in flushed:
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.function_call and part.function_call.id:
            seen_tool_calls.add(part.function_call.id)
  return flushed


def _convert_model_thought_or_flush_pending(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_tool_calls: set[str] | None = None,
    pending_function_calls: list[Event] | None = None,
    seen_thought_steps: dict[str, str] | None = None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Pairs completed model thinking with pending function calls or emits it."""
  if (
      pending_function_calls is None
      and seen_thought_steps is None
      and latest_thoughts is None
  ):
    return []
  step_meta = (
      custom_metadata
      if custom_metadata is not None
      else _step_custom_metadata(step)
  )
  trajectory_key = (
      str(step_meta.get('trajectory_id') or '')
      if step_meta
      else (step.trajectory_id if isinstance(step.trajectory_id, str) else '')
  )
  has_unseen_tool_calls = bool(
      step.source == sdk_types.StepSource.MODEL
      and step.tool_calls
      and (
          seen_tool_calls is None
          or any(
              _build_tool_call_id(step, call, idx) not in seen_tool_calls
              for idx, call in enumerate(step.tool_calls)
          )
      )
  )
  flushed_prior: list[Event] = []
  if (
      pending_function_calls
      and (step.is_complete_response or has_unseen_tool_calls)
      and _should_flush_pending_calls(step)
  ):
    flushed_prior = _flush_pending_function_calls(
        step,
        trajectory_key=trajectory_key,
        pending_function_calls=pending_function_calls,
        seen_tool_calls=seen_tool_calls,
        seen_thought_steps=seen_thought_steps,
        latest_thoughts=latest_thoughts,
        step_meta=step_meta,
    )

  if (
      latest_thoughts is not None
      and step.source == sdk_types.StepSource.MODEL
      and isinstance(step.thinking, str)
      and step.thinking
  ):
    thought_key = f'{trajectory_key}:{step.step_index}'
    unemitted = _unemitted_thinking(
        step.thinking, thought_key, seen_thought_steps
    )
    if unemitted:
      latest_thoughts[trajectory_key] = (
          thought_key,
          genai_types.Part(text=unemitted, thought=True),
      )

  is_completed_thought = (
      step.source == sdk_types.StepSource.MODEL
      and step.status == sdk_types.StepStatus.DONE
      and not (step.is_complete_response and step.content)
      and not has_unseen_tool_calls
  )
  if is_completed_thought:
    thought_part = _consume_step_thought_part(
        step,
        seen_thought_steps,
        trajectory_key=trajectory_key,
        latest_thoughts=latest_thoughts,
    )
    if thought_part is not None:
      if pending_function_calls:
        _prepend_thought_to_pending(
            pending_function_calls, thought_part, step_meta
        )
        flushed = list(pending_function_calls)
        pending_function_calls.clear()
        if seen_tool_calls is not None:
          for event in flushed:
            if event.content and event.content.parts:
              for part in event.content.parts:
                if part.function_call and part.function_call.id:
                  seen_tool_calls.add(part.function_call.id)
        return [*flushed_prior, *flushed]
      return [
          *flushed_prior,
          Event(
              invocation_id=ctx.invocation_id,
              author=author,
              branch=ctx.branch,
              content=genai_types.Content(role='model', parts=[thought_part]),
              custom_metadata=step_meta,
          ),
      ]

  if pending_function_calls and _should_flush_pending_calls(step):
    return [
        *flushed_prior,
        *_flush_pending_function_calls(
            step,
            trajectory_key=trajectory_key,
            pending_function_calls=pending_function_calls,
            seen_tool_calls=seen_tool_calls,
            seen_thought_steps=seen_thought_steps,
            latest_thoughts=latest_thoughts,
            step_meta=step_meta,
        ),
    ]

  return flushed_prior


def _convert_model_text(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_thought_steps: dict[str, str] | None = None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Converts a completed model text response into one final model text event.

  The Antigravity SDK re-broadcasts the cumulative ``content`` on every step
  transition as
  the response grows, so emitting on each transition would record the same
  message many times. We emit only when ``is_complete_response`` is set, using
  the final cumulative ``content``. Partial streaming is handled separately by
  ``_convert_partial_deltas``.
  """
  is_model_text = step.source == sdk_types.StepSource.MODEL and step.type in (
      sdk_types.StepType.TEXT_RESPONSE,
      sdk_types.StepType.UNKNOWN,
  )
  if not is_model_text or not step.is_complete_response or not step.content:
    return []

  step_meta = (
      custom_metadata
      if custom_metadata is not None
      else _step_custom_metadata(step)
  )
  trajectory_key = (
      str(step_meta.get('trajectory_id') or '')
      if step_meta
      else (step.trajectory_id if isinstance(step.trajectory_id, str) else '')
  )
  thought_part = _consume_step_thought_part(
      step,
      seen_thought_steps,
      trajectory_key=trajectory_key,
      latest_thoughts=latest_thoughts,
  )
  if thought_part is None:
    thought_part = _consume_cached_thought_part(
        trajectory_key, seen_thought_steps, latest_thoughts
    )
  parts = [thought_part] if thought_part is not None else []
  parts.append(genai_types.Part.from_text(text=step.content))

  return [
      Event(
          invocation_id=ctx.invocation_id,
          author=author,
          branch=ctx.branch,
          content=genai_types.Content(role='model', parts=parts),
          custom_metadata=step_meta,
      )
  ]


def _pending_call_ids(pending_function_calls: list[Event] | None) -> set[str]:
  """Returns the tool call IDs currently buffered in pending_function_calls."""
  if not pending_function_calls:
    return set()
  call_ids: set[str] = set()
  for event in pending_function_calls:
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.function_call and part.function_call.id:
          call_ids.add(part.function_call.id)
  return call_ids


def _convert_function_calls(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_tool_calls: set[str],
    pending_function_calls: list[Event] | None = None,
    seen_thought_steps: dict[str, str] | None = None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Converts model-issued tool calls into model function-call events."""
  if step.source != sdk_types.StepSource.MODEL or not step.tool_calls:
    return []

  pending_ids = _pending_call_ids(pending_function_calls)
  unseen_calls = [
      (call, _build_tool_call_id(step, call, idx))
      for idx, call in enumerate(step.tool_calls)
      if _build_tool_call_id(step, call, idx) not in seen_tool_calls
      and _build_tool_call_id(step, call, idx) not in pending_ids
  ]
  if not unseen_calls:
    return []

  step_meta = (
      custom_metadata
      if custom_metadata is not None
      else _step_custom_metadata(step)
  )
  trajectory_key = (
      str(step_meta.get('trajectory_id') or '')
      if step_meta
      else (step.trajectory_id if isinstance(step.trajectory_id, str) else '')
  )
  has_thinking = bool(step.thinking) or (
      latest_thoughts is not None and trajectory_key in latest_thoughts
  )
  should_buffer_active = (
      pending_function_calls is not None
      and step.status == sdk_types.StepStatus.ACTIVE
      and has_thinking
  )
  has_thought_state = (
      pending_function_calls is not None
      or seen_thought_steps is not None
      or latest_thoughts is not None
  )
  thought_part: genai_types.Part | None = None
  if has_thought_state and not should_buffer_active:
    thought_part = _consume_step_thought_part(
        step,
        seen_thought_steps,
        trajectory_key=trajectory_key,
        latest_thoughts=latest_thoughts,
    )
    if thought_part is None:
      thought_part = _consume_cached_thought_part(
          trajectory_key, seen_thought_steps, latest_thoughts
      )
  events: list[Event] = []
  for call, call_id in unseen_calls:
    parts = [thought_part] if (thought_part is not None and not events) else []
    parts.append(
        genai_types.Part(
            function_call=genai_types.FunctionCall(
                name=call.name,
                args=call.args,
                id=call_id,
            )
        )
    )
    events.append(
        Event(
            invocation_id=ctx.invocation_id,
            author=author,
            branch=ctx.branch,
            content=genai_types.Content(role='model', parts=parts),
            custom_metadata=step_meta,
        )
    )
  if events and should_buffer_active and pending_function_calls is not None:
    pending_function_calls.extend(events)
    return []

  for _, call_id in unseen_calls:
    seen_tool_calls.add(call_id)
  return events


def _function_response_event(
    *,
    ctx: InvocationContext,
    name: str,
    call_id: str,
    response: dict[str, JsonValue],
    custom_metadata: dict[str, JsonValue] | None = None,
) -> Event:
  """Builds the ADK event recording one tool's answer to one call."""
  return Event(
      invocation_id=ctx.invocation_id,
      # Author is the tool name so session history attributes the response to
      # the tool, mirroring ADK's own function-response events.
      author=name,
      branch=ctx.branch,
      content=genai_types.Content(
          role='user',
          parts=[
              genai_types.Part(
                  function_response=genai_types.FunctionResponse(
                      name=name,
                      id=call_id,
                      response=response,
                  )
              )
          ],
      ),
      custom_metadata=custom_metadata,
  )


_BUILTIN_OUTPUT_KEYS: dict[str, tuple[str, ...]] = {
    'run_command': ('combined_output', 'exit_code'),
    'find_file': ('output',),
    'list_dir': ('results',),
    'search_dir': ('num_results',),
    'search_web': ('summary',),
    'read_url_content': ('title', 'summary', 'content_path'),
    'generate_image': ('image_name', 'aspect_ratio', 'output_path'),
    'edit_file': ('diff_block',),
}


def _unwrap_result_payload(
    value: JsonValue | BaseModel, *, dict_only: bool = False
) -> dict[str, JsonValue]:
  """Unwraps a tool result value or JSON string into a FunctionResponse dict."""
  unwrapped: JsonValue
  if isinstance(value, BaseModel):
    unwrapped = value.model_dump(mode='json')
  else:
    unwrapped = value
  for pass_index in range(2):
    if not isinstance(unwrapped, str):
      break
    try:
      parsed = json.loads(unwrapped)
    except ValueError:
      break
    if (dict_only or pass_index > 0) and not isinstance(parsed, dict):
      break
    unwrapped = parsed
  if isinstance(unwrapped, dict):
    return unwrapped
  return {'result': 'success' if unwrapped is None else unwrapped}


def _buffered_result_payload(
    result: _tool_result_capture.ToolResult,
) -> dict[str, JsonValue]:
  """Returns the ``FunctionResponse.response`` dict for one captured result."""
  if result.error:
    return {'error': result.error}

  # The harness hands back a client tool's value as the JSON string
  # ``json.dumps(tool_result_to_dict(...))``, so it usually needs unwrapping.
  # Built-in tools pass a Pydantic model instance (e.g. ``RunCommandResult``).
  return _unwrap_result_payload(result.result)


def _extract_builtin_step_output(
    step: sdk_types.Step,
    call: sdk_types.ToolCall,
) -> dict[str, JsonValue]:
  """Extracts the completed output payload from a built-in tool step."""
  output_keys = _BUILTIN_OUTPUT_KEYS.get(call.name, ())
  if output_keys and isinstance(call.args, dict):
    extracted = {key: call.args[key] for key in output_keys if key in call.args}
    if extracted:
      return extracted
  # Built-in tool step.content is free-form text (e.g. 'true' or '123');
  # only unwrap JSON objects so scalar strings stay wrapped in {'result':}.
  return _unwrap_result_payload(step.content or None, dict_only=True)


def _convert_function_responses(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    tool_results: _tool_result_capture.ToolResultBuffer | None = None,
    pending_function_calls: list[Event] | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Converts completed tool-execution steps into function-response events."""
  is_tool_response = (
      step.type == sdk_types.StepType.TOOL_CALL
      and step.status
      in (
          sdk_types.StepStatus.DONE,
          sdk_types.StepStatus.ERROR,
      )
  )
  if not is_tool_response:
    return []

  step_meta = (
      custom_metadata
      if custom_metadata is not None
      else _step_custom_metadata(step)
  )
  # A client-side tool: the Antigravity SDK blanks its ``tool_calls`` and
  # ``Step`` has no
  # field for a result, so the step names nothing and holds nothing.
  if not step.tool_calls:
    if pending_function_calls:
      return []
    return drain_tool_results(
        ctx=ctx,
        seen_tool_calls=seen_tool_calls,
        seen_tool_results=seen_tool_results,
        tool_results=tool_results,
        custom_metadata=step_meta,
    )

  events: list[Event] = []
  for idx, call in enumerate(step.tool_calls):
    call_id = _build_tool_call_id(step, call, idx)
    if call_id in seen_tool_results:
      continue
    if pending_function_calls and call_id not in seen_tool_calls:
      continue
    captured = tool_results.take({call_id}) if tool_results is not None else []
    response: dict[str, JsonValue]
    if captured:
      response = _buffered_result_payload(captured[0][1])
    elif step.status == sdk_types.StepStatus.ERROR:
      response = {
          'error': (
              step.error
              or f'Tool call execution failed with status {step.status.name}.'
          )
      }
    else:
      output_keys = _BUILTIN_OUTPUT_KEYS.get(call.name, ())
      has_output_args = bool(
          output_keys
          and isinstance(call.args, dict)
          and any(key in call.args for key in output_keys)
      )
      if not has_output_args and not step.content:
        continue
      response = _extract_builtin_step_output(step, call)

    seen_tool_results.add(call_id)

    events.append(
        _function_response_event(
            ctx=ctx,
            name=call.name,
            call_id=call_id,
            response=response,
            custom_metadata=step_meta,
        )
    )
  return events


def drain_tool_results(
    *,
    ctx: InvocationContext,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    tool_results: _tool_result_capture.ToolResultBuffer | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Answers emitted calls that have a captured outcome and no response yet.

  Args:
    ctx: The active invocation context, used for event correlation fields.
    seen_tool_calls: Ids of tool calls already emitted. Read to decide what may
      be answered; not mutated.
    seen_tool_results: Ids of tool results already emitted, mutated in place to
      record the ones answered here.
    tool_results: This conversation's client-tool outcomes, or None when no
      capture hook was registered, in which case nothing is answered. Drained of
      every id this call answers.
    custom_metadata: Optional trajectory hierarchy metadata from the step that
      triggered draining.

  Returns:
    One function-response event per answered call, in the order the tools
    finished in. Empty when nothing is owed a response.
  """
  if tool_results is None:
    return []

  events: list[Event] = []
  # A response may not precede the call it answers, hence ``seen_tool_calls``.
  for call_id, result in tool_results.take(seen_tool_calls - seen_tool_results):
    seen_tool_results.add(call_id)
    events.append(
        _function_response_event(
            ctx=ctx,
            name=result.name,
            call_id=call_id,
            response=_buffered_result_payload(result),
            custom_metadata=custom_metadata,
        )
    )
  return events


def convert_step_to_events(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    tool_results: _tool_result_capture.ToolResultBuffer | None = None,
    streaming: bool = False,
    pending_function_calls: list[Event] | None = None,
    seen_thought_steps: dict[str, str] | None = None,
    latest_thoughts: dict[str, tuple[str, genai_types.Part]] | None = None,
    custom_metadata: dict[str, JsonValue] | None = None,
) -> list[Event]:
  """Translates one Antigravity ``Step`` into the ADK events it maps to.

  Args:
    step: An Antigravity SDK ``Step`` from ``conversation.receive_steps()``.
    ctx: The active invocation context, used for event correlation fields.
    author: The agent name to stamp on model-authored events.
    seen_tool_calls: Ids of tool calls already emitted, mutated in place to
      deduplicate calls repeated across step transitions.
    seen_tool_results: Ids of tool results already emitted, mutated in place to
      deduplicate results repeated across step transitions.
    tool_results: This conversation's client-tool results, or None when no
      capture hook was registered.
    streaming: When True (SSE mode), incremental thinking/text deltas are also
      emitted as ``partial=True`` events. When False, only final events are
      emitted.
    pending_function_calls: Optional buffer of ACTIVE function-call events
      awaiting their companion ``StepStatus.DONE`` planner step carrying
      ``thinking``.
    seen_thought_steps: Optional mapping of step-thinking keys to cumulative
      thinking text already emitted in this turn.
    latest_thoughts: Optional per-trajectory cache of the most recent unseen
      thinking ``Part`` from ``StepStatus.ACTIVE`` steps.
    custom_metadata: Optional resolved trajectory hierarchy metadata for the
      step; defaults to ``_step_custom_metadata(step)``.

  Returns:
    The ADK events the step maps to, in emission order. Partial deltas (if any)
    precede the final aggregated text event. May be empty for steps that carry
    no user-visible content (e.g. compaction).
  """
  # 1. Resolve step metadata and optional SSE partial deltas.
  step_meta = (
      custom_metadata
      if custom_metadata is not None
      else _step_custom_metadata(step)
  )
  partials = (
      _convert_partial_deltas(step, ctx=ctx, author=author) if streaming else []
  )

  # 2. Pair or flush pending function calls and drain completed client tools.
  had_pending = bool(pending_function_calls)
  thought_or_flushed = _convert_model_thought_or_flush_pending(
      step,
      ctx=ctx,
      author=author,
      seen_tool_calls=seen_tool_calls,
      pending_function_calls=pending_function_calls,
      seen_thought_steps=seen_thought_steps,
      latest_thoughts=latest_thoughts,
      custom_metadata=step_meta,
  )
  prior_responses: list[Event] = []
  if not pending_function_calls:
    prior_responses = drain_tool_results(
        ctx=ctx,
        seen_tool_calls=seen_tool_calls,
        seen_tool_results=seen_tool_results,
        tool_results=tool_results,
        custom_metadata=(
            step_meta if step.type == sdk_types.StepType.TOOL_CALL else None
        ),
    )
  leading_events = (
      [*thought_or_flushed, *prior_responses]
      if had_pending
      else [*prior_responses, *thought_or_flushed]
  )

  # 3. Emit partials, flushed calls/responses, and current step events.
  return [
      *partials,
      *leading_events,
      *_convert_model_text(
          step,
          ctx=ctx,
          author=author,
          seen_thought_steps=seen_thought_steps,
          latest_thoughts=latest_thoughts,
          custom_metadata=step_meta,
      ),
      *_convert_function_calls(
          step,
          ctx=ctx,
          author=author,
          seen_tool_calls=seen_tool_calls,
          pending_function_calls=pending_function_calls,
          seen_thought_steps=seen_thought_steps,
          latest_thoughts=latest_thoughts,
          custom_metadata=step_meta,
      ),
      *_convert_function_responses(
          step,
          ctx=ctx,
          seen_tool_calls=seen_tool_calls,
          seen_tool_results=seen_tool_results,
          tool_results=tool_results,
          pending_function_calls=pending_function_calls,
          custom_metadata=step_meta,
      ),
  ]


def final_model_text(event: Event, author: str | None = None) -> str | None:
  """Returns an event's user-visible model text, or None if it carries none.

  Partials and thought/function parts never count.

  Args:
    event: The event to inspect.
    author: If given, only this ADK agent's own events count. A composite ADK
      agent authors its events under its sub-agents' names, so leave this unset
      to accept whatever the ADK agent tree produced.

  Returns:
    The user-visible text, its parts joined with newlines as ``AgentTool``
    does, or None if the event carries none.
  """
  if event.partial or not event.content:
    return None
  if author is not None and event.author != author:
    return None
  parts = event.content.parts or []
  chunks = [
      part.text
      for part in parts
      if part.text
      and not part.thought
      and not part.function_call
      and not part.function_response
  ]
  return '\n'.join(chunks) if chunks else None

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
from pydantic import JsonValue

from ...events.event import Event

if TYPE_CHECKING:
  from . import _tool_result_capture
  from ...agents.invocation_context import InvocationContext


def _build_tool_call_id(step: sdk_types.Step, call: sdk_types.ToolCall) -> str:
  """Derives a stable id for a tool call, falling back when the SDK omits one."""
  return call.id or f'{step.step_index}-{call.name}'


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
  matching the SDK's own chunk ordering.
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


def _convert_model_text(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
) -> list[Event]:
  """Converts a completed model text response into one final model text event.

  The SDK re-broadcasts the cumulative ``content`` on every step transition as
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

  return [
      Event(
          invocation_id=ctx.invocation_id,
          author=author,
          branch=ctx.branch,
          content=genai_types.Content(
              role='model',
              parts=[genai_types.Part.from_text(text=step.content)],
          ),
      )
  ]


def _convert_function_calls(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_tool_calls: set[str],
) -> list[Event]:
  """Converts model-issued tool calls into model function-call events."""
  if step.source != sdk_types.StepSource.MODEL or not step.tool_calls:
    return []

  events = []
  for call in step.tool_calls:
    call_id = _build_tool_call_id(step, call)
    if call_id in seen_tool_calls:
      continue
    seen_tool_calls.add(call_id)

    events.append(
        Event(
            invocation_id=ctx.invocation_id,
            author=author,
            branch=ctx.branch,
            content=genai_types.Content(
                role='model',
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=call.name,
                            args=call.args,
                            id=call_id,
                        )
                    )
                ],
            ),
        )
    )
  return events


def _function_response_event(
    *,
    ctx: InvocationContext,
    name: str,
    call_id: str,
    response: dict[str, JsonValue],
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
  )


def _buffered_result_payload(
    result: _tool_result_capture.ToolResult,
) -> dict[str, JsonValue]:
  """Returns the ``FunctionResponse.response`` dict for one captured result."""
  if result.error:
    return {'error': result.error}

  # The harness hands back a client tool's value as the JSON string
  # ``json.dumps(tool_result_to_dict(...))``, so it usually needs unwrapping.
  value = result.result
  if isinstance(value, str):
    try:
      value = json.loads(value)
    except ValueError:
      pass  # Not JSON; the raw text is wrapped below rather than raising.
  if isinstance(value, dict):
    return value
  return {'result': 'success' if value is None else value}


def _convert_function_responses(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    tool_results: _tool_result_capture.ToolResultBuffer | None = None,
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

  # A client-side tool: the SDK blanks its ``tool_calls`` and ``Step`` has no
  # field for a result, so the step names nothing and holds nothing.
  if not step.tool_calls:
    return drain_tool_results(
        ctx=ctx,
        seen_tool_calls=seen_tool_calls,
        seen_tool_results=seen_tool_results,
        tool_results=tool_results,
    )

  # The hook fires for these tools too, but its copy is keyed by an id this
  # side never sees: the SDK gives a built-in's ``ToolCall.id`` the step id
  # ``f'{trajectory_id}:{step_index}'``, while the hook is handed the model's
  # own call id, or a SHA-256 of that step id when there is none
  # (``localharness/tool_metadata.go``, ``ResolveStepCallID``). That copy
  # therefore cannot be dropped by id here -- and need not be: the same
  # mismatch keeps it out of ``drain_tool_results``, which only takes ids in
  # ``seen_tool_calls``. The turn clears the buffer at its end.
  # ``ToolResult.step_id`` would be the key that does match -- at head the SDK
  # already sets it to that same ``f'{trajectory_id}:{step_index}'`` -- but the
  # SDK vendored here predates the field.
  events = []
  for call in step.tool_calls:
    call_id = _build_tool_call_id(step, call)
    if call_id in seen_tool_results:
      continue
    seen_tool_results.add(call_id)

    response: dict[str, JsonValue]
    if step.status == sdk_types.StepStatus.ERROR:
      response = {
          'error': (
              step.error
              or f'Tool call execution failed with status {step.status.name}.'
          )
      }
    else:
      response = {'result': step.content or 'success'}

    events.append(
        _function_response_event(
            ctx=ctx, name=call.name, call_id=call_id, response=response
        )
    )
  return events


def drain_tool_results(
    *,
    ctx: InvocationContext,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    tool_results: _tool_result_capture.ToolResultBuffer | None = None,
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

  Returns:
    One function-response event per answered call, in the order the tools
    finished in. Empty when nothing is owed a response.
  """
  if tool_results is None:
    return []

  events = []
  # A response may not precede the call it answers, hence ``seen_tool_calls``.
  for call_id, result in tool_results.take(seen_tool_calls - seen_tool_results):
    seen_tool_results.add(call_id)
    events.append(
        _function_response_event(
            ctx=ctx,
            name=result.name,
            call_id=call_id,
            response=_buffered_result_payload(result),
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

  Returns:
    The ADK events the step maps to, in emission order. Partial deltas (if any)
    precede the final aggregated text event. May be empty for steps that carry
    no user-visible content (e.g. compaction).
  """
  partials = (
      _convert_partial_deltas(step, ctx=ctx, author=author) if streaming else []
  )
  return [
      *partials,
      *_convert_model_text(step, ctx=ctx, author=author),
      *_convert_function_calls(
          step, ctx=ctx, author=author, seen_tool_calls=seen_tool_calls
      ),
      *_convert_function_responses(
          step,
          ctx=ctx,
          seen_tool_calls=seen_tool_calls,
          seen_tool_results=seen_tool_results,
          tool_results=tool_results,
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

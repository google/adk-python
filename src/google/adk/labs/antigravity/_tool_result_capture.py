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

"""Buffers the outcome of tool calls, keyed by the call they answer.

A client-side tool's outcome never reaches the trajectory: the terminal
``Step`` for such a call carries an empty ``tool_calls``, and ``Step`` has no
field for a result in the first place. The tool hooks are the one place it
does arrive, so it is captured there and held until ``_event_converter`` can
pair it with its call.

Success and failure arrive on different hooks, and exactly one of them fires
per call: the harness routes a failed tool to ``on_tool_error`` and never to
``post_tool_call`` (``localharness/hook_tool_calls_fragment.go``, "Exactly one
hook fires per step ... They are mutually exclusive"). Hence the two capture
classes below, both feeding one buffer.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Collection
from typing import Protocol
from typing import runtime_checkable

from google.antigravity.hooks import hooks as sdk_hooks
from pydantic import JsonValue

logger = logging.getLogger('google_adk.' + __name__)


class ToolResult(Protocol):
  """The fields of an SDK ``ToolResult`` that this package reads.

  A protocol rather than ``sdk_types.ToolResult`` because a structurally
  identical copy of that class also exists, and results of both types reach
  this buffer.
  """

  name: str
  id: str | None
  result: JsonValue
  error: str | None


@runtime_checkable
class ToolError(Protocol):
  """The fields of an SDK ``ToolExecutionError`` that this package reads.

  A protocol for the same reason ``ToolResult`` is one. Note the different
  field names: the SDK models a failure as a raised exception rather than as a
  result carrying an error, so nothing is shared with ``ToolResult``.
  """

  tool_name: str
  call_id: str | None
  # An SDK old enough to hand the hook a bare ``RuntimeError`` instead fails
  # this check, and such a failure is dropped: without a call id there is no
  # call to pair it with. That is the pre-existing behaviour, not a regression
  # -- the dangling function_call simply survives, as it did before this hook
  # existed.


@dataclasses.dataclass
class _FailedToolResult:
  """Stands in for the ``ToolResult`` the SDK does not send for a failed call.

  Satisfies ``ToolResult`` so a failure drains through the same path a value
  does.
  """

  name: str
  id: str | None
  result: JsonValue
  error: str | None


class ToolResultBuffer:
  """One SDK conversation's tool outcomes, keyed by ``ToolCall.id``.

  Carries the hook bodies but is deliberately not a hook itself: a
  ``HookRunner`` classifies hooks by ``isinstance`` against its own SDK's hook
  classes, and a subclass cannot remove a base class. The capture classes
  below bind this to the Antigravity SDK; another copy of it gets its own.

  Not thread-safe, and does not need to be: outcomes arrive on hook-dispatch
  tasks and are drained by the turn, both on the one event loop the
  conversation belongs to.
  """

  def __init__(self) -> None:
    # `ToolResultCapture` mixes this in ahead of an SDK hook base class, which
    # has no `__init__` of its own today but is the SDK's to change.
    super().__init__()
    self._results: dict[str, ToolResult] = {}

  def __len__(self) -> int:
    return len(self._results)

  def record(self, result: ToolResult) -> None:
    """Buffers one tool result, dropping one that cannot be correlated."""
    # ``id`` is the only thing tying a result to an emitted function call, so
    # keeping one without it risks draining it against an unrelated call.
    if not result.id:
      logger.debug(
          '[ADK] Dropping an Antigravity tool result for %s: it carries no '
          'call id to correlate it with.',
          result.name,
      )
      return
    self._results[result.id] = result

  def record_error(self, error: ToolError) -> None:
    """Buffers one failed tool call, dropping one that cannot be correlated."""
    if not error.call_id:
      logger.debug(
          '[ADK] Dropping an Antigravity tool failure for %s: it carries no '
          'call id to correlate it with.',
          error.tool_name,
      )
      return
    self._results[error.call_id] = _FailedToolResult(
        name=error.tool_name,
        id=error.call_id,
        result=None,
        error=str(error) or 'Tool call execution failed.',
    )

  def take(self, call_ids: Collection[str]) -> list[tuple[str, ToolResult]]:
    """Removes and returns any buffered results for ``call_ids``."""
    # Insertion order is arrival order, i.e. the order the tools finished in.
    return [
        (call_id, self._results.pop(call_id))
        for call_id in list(self._results)
        if call_id in call_ids
    ]

  def clear(self) -> None:
    """Forgets everything buffered."""
    self._results.clear()


class ToolResultCapture(ToolResultBuffer, sdk_hooks.PostToolCallHook):  # type: ignore[misc]
  """``ToolResultBuffer`` bound to the Antigravity SDK as its success hook.

  Each SDK copy needs its own such subclass, because a ``HookRunner``
  classifies hooks by ``isinstance`` against the hook classes of the SDK it
  came from. The `misc` ignore is that base class: the SDK ships no type
  information, so mypy sees `PostToolCallHook` as `Any` and `strict` forbids
  subclassing it.
  """

  async def run(self, context: object, data: ToolResult) -> None:
    """Records one tool result. The post-tool-call hook entry point."""
    # `context` is dictated by the SDK's `InspectHook.run`, not wanted here:
    # the correlation id is on the result itself, so there is nothing to read
    # off the context. It is accepted only to match the signature the
    # `HookRunner` calls.
    del context
    self.record(data)


class ToolErrorCapture(sdk_hooks.OnToolErrorHook):  # type: ignore[misc]
  """Feeds a failed tool call into a ``ToolResultBuffer``.

  A separate class rather than a second base on ``ToolResultCapture``:
  ``HookRunner.register_hook`` appends to every list an object matches, and
  both hook interfaces name their entry point ``run``, so one object claiming
  both would have a single ``run`` handed a ``ToolResult`` and an exception
  alike.

  Holds the buffer rather than subclassing it so the two hooks share one, and
  needs its own per-SDK subclass for the same reason ``ToolResultCapture``
  does. The `misc` ignore is likewise the untyped SDK base class.
  """

  def __init__(self, buffer: ToolResultBuffer) -> None:
    super().__init__()
    self._buffer = buffer

  async def run(self, context: object, data: Exception) -> None:
    """Records one failed tool call. The on-tool-error hook entry point."""
    # `context` is unused for the same reason as in `ToolResultCapture.run`.
    # Returning None is what leaves the harness's own error message in place:
    # this hook is an observer, and `OnToolErrorHook` only overrides the
    # message the model sees when it returns a non-empty string.
    del context
    # `data` is declared as broadly as the SDK declares it. Every failure the
    # harness routes here is a `ToolExecutionError`, but only that shape
    # carries the call id, so anything else is not correlatable.
    if not isinstance(data, ToolError):
      logger.debug(
          '[ADK] Dropping an Antigravity tool failure of type %s: it carries '
          'no tool name or call id to correlate it with.',
          type(data).__name__,
      )
      return
    self._buffer.record_error(data)

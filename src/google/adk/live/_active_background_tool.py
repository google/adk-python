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

"""Registry of the tool calls running in the background of a Live session.

Also holds the backward compatibility for the deprecated InvocationContext
fields that the registry replaces.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from typing import Any
from typing import cast
from typing import Optional
import warnings

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr

from ._active_streaming_tool import ActiveStreamingTool
from .live_request_queue import LiveRequestQueue

# Prefix the registry key of an entry added through a deprecated view. Each
# view has its own, so the same name written to both views makes two entries.
_LEGACY_STREAMING_ID_PREFIX = 'legacy_streaming:'
_LEGACY_NON_BLOCKING_ID_PREFIX = 'legacy_non_blocking:'

_ACTIVE_STREAMING_TOOLS = 'active_streaming_tools'
_ACTIVE_NON_BLOCKING_TOOL_TASKS = 'active_non_blocking_tool_tasks'


class ActiveBackgroundTool(BaseModel):
  """One tool call running in the background of a live invocation.

  Streaming tools (async generators) and non-blocking tools are both tracked
  with this model, one entry per function call.
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  tool_name: str
  """The name of the tool that was called."""

  function_call_id: str
  """The id of the function call this entry runs."""

  task: Optional[asyncio.Task[Any]]
  """The task running the tool call.

  ``None`` only after code cleared it through the deprecated
  ``InvocationContext.active_streaming_tools``.
  """

  input_stream: Optional[LiveRequestQueue] = None
  """The input stream the tool reads live requests from.

  Set only for tools that declare an ``input_stream: LiveRequestQueue``
  parameter.
  """

  is_generator: bool = False
  """Whether the tool is an async generator that yields several responses."""


class _ActiveStreamingToolHandle(ActiveStreamingTool):
  """An ActiveStreamingTool that writes its changes through to the registry."""

  _entry: ActiveBackgroundTool = PrivateAttr()

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
    if name == 'task':
      self._entry.task = value
    elif name == 'stream':
      self._entry.input_stream = value


class ActiveStreamingToolsView(MutableMapping[str, ActiveStreamingTool]):
  """The streaming tool entries of a registry, keyed by tool name.

  Backs the deprecated ``InvocationContext.active_streaming_tools``. Reads and
  writes go to the registry, so code written against the old dict keeps
  working:

  - Reading a tool returns its most recent call. Setting ``task`` or ``stream``
    on the returned value updates that call's entry.
  - Assigning a tool replaces all its entries with the given one.
  - Deleting a tool removes all its entries without cancelling their tasks.
  """

  def __init__(self, registry: dict[str, ActiveBackgroundTool]):
    self._registry = registry

  def _entries(
      self, tool_name: str | None = None
  ) -> list[ActiveBackgroundTool]:
    return [
        entry
        for entry in self._registry.values()
        if entry.is_generator
        and (tool_name is None or entry.tool_name == tool_name)
    ]

  def __getitem__(self, tool_name: str) -> ActiveStreamingTool:
    entries = self._entries(tool_name)
    if not entries:
      raise KeyError(tool_name)
    entry = entries[-1]
    handle = _ActiveStreamingToolHandle(
        task=entry.task, stream=entry.input_stream
    )
    handle._entry = entry
    return handle

  def __setitem__(self, tool_name: str, value: ActiveStreamingTool) -> None:
    for entry in self._entries(tool_name):
      del self._registry[entry.function_call_id]
    function_call_id = f'{_LEGACY_STREAMING_ID_PREFIX}{tool_name}'
    self._registry[function_call_id] = ActiveBackgroundTool(
        tool_name=tool_name,
        function_call_id=function_call_id,
        task=value.task,
        input_stream=value.stream,
        is_generator=True,
    )

  def __delitem__(self, tool_name: str) -> None:
    entries = self._entries(tool_name)
    if not entries:
      raise KeyError(tool_name)
    for entry in entries:
      del self._registry[entry.function_call_id]

  def __iter__(self) -> Iterator[str]:
    return iter(dict.fromkeys(entry.tool_name for entry in self._entries()))

  def __len__(self) -> int:
    return len({entry.tool_name for entry in self._entries()})

  def __repr__(self) -> str:
    return repr(dict(self.items()))

  def _replace(self, value: Mapping[str, ActiveStreamingTool] | None) -> None:
    """Replaces all streaming tool entries, like assigning a new dict."""
    items = list((value or {}).items())
    for entry in self._entries():
      del self._registry[entry.function_call_id]
    for tool_name, tool in items:
      self[tool_name] = tool


class ActiveNonBlockingToolTasksView(MutableMapping[str, asyncio.Task[Any]]):
  """The non-blocking tool tasks of a registry.

  Backs the deprecated ``InvocationContext.active_non_blocking_tool_tasks``,
  keyed by ``f'{tool_name}_{function_call_id}'`` as before. Reads and writes go
  to the registry. Deleting a key removes the entry without cancelling its
  task.
  """

  def __init__(self, registry: dict[str, ActiveBackgroundTool]):
    self._registry = registry

  @staticmethod
  def _key(entry: ActiveBackgroundTool) -> str:
    # An entry added through this view is registered under its key, which
    # doesn't split back into a tool name and a function call id.
    if entry.function_call_id.startswith(_LEGACY_NON_BLOCKING_ID_PREFIX):
      return entry.function_call_id[len(_LEGACY_NON_BLOCKING_ID_PREFIX) :]
    return f'{entry.tool_name}_{entry.function_call_id}'

  def _entries(self) -> list[ActiveBackgroundTool]:
    return [
        entry for entry in self._registry.values() if not entry.is_generator
    ]

  def _find(self, key: str) -> ActiveBackgroundTool:
    for entry in self._entries():
      if self._key(entry) == key:
        return entry
    raise KeyError(key)

  def __getitem__(self, key: str) -> asyncio.Task[Any]:
    # Only the streaming tools view clears a task, so a non-blocking entry
    # always has one, unless code stored None here itself.
    return cast(asyncio.Task[Any], self._find(key).task)

  def __setitem__(self, key: str, task: asyncio.Task[Any]) -> None:
    try:
      self._find(key).task = task
    except KeyError:
      function_call_id = f'{_LEGACY_NON_BLOCKING_ID_PREFIX}{key}'
      self._registry[function_call_id] = ActiveBackgroundTool(
          tool_name=key, function_call_id=function_call_id, task=task
      )

  def __delitem__(self, key: str) -> None:
    del self._registry[self._find(key).function_call_id]

  def __iter__(self) -> Iterator[str]:
    return iter([self._key(entry) for entry in self._entries()])

  def __len__(self) -> int:
    return len(self._entries())

  def __repr__(self) -> str:
    return repr(dict(self.items()))

  def _replace(self, value: Mapping[str, asyncio.Task[Any]] | None) -> None:
    """Replaces all non-blocking tool entries, like assigning a new dict."""
    items = list((value or {}).items())
    for entry in self._entries():
      del self._registry[entry.function_call_id]
    for key, task in items:
      self[key] = task


# Backward compatibility for the deprecated InvocationContext fields
# `active_streaming_tools` and `active_non_blocking_tool_tasks`, which the
# registry replaces. InvocationContext delegates its properties for them here.
#
# Each function that warns does so from one fixed call depth, which the
# `stacklevel` of its warning counts on to point at the user's code.


def _warn_deprecated(name: str, *, stacklevel: int) -> None:
  warnings.warn(
      f'`InvocationContext.{name}` is deprecated and will be removed in a'
      ' future release.',
      DeprecationWarning,
      stacklevel=stacklevel + 1,
  )


def get_active_streaming_tools(
    registry: dict[str, ActiveBackgroundTool] | None,
) -> ActiveStreamingToolsView | None:
  """Backs the getter of ``InvocationContext.active_streaming_tools``."""
  # Called from the property, which the user's code reads.
  _warn_deprecated(_ACTIVE_STREAMING_TOOLS, stacklevel=3)
  return None if registry is None else ActiveStreamingToolsView(registry)


def set_active_streaming_tools(
    registry: dict[str, ActiveBackgroundTool],
    value: Mapping[str, ActiveStreamingTool] | None,
) -> None:
  """Backs the setter of ``InvocationContext.active_streaming_tools``."""
  # Called from the setter, which pydantic's __setattr__ calls through a
  # handler on behalf of the user's code.
  _warn_deprecated(_ACTIVE_STREAMING_TOOLS, stacklevel=5)
  ActiveStreamingToolsView(registry)._replace(value)


def get_active_non_blocking_tool_tasks(
    registry: dict[str, ActiveBackgroundTool] | None,
) -> ActiveNonBlockingToolTasksView | None:
  """Backs the getter of ``InvocationContext.active_non_blocking_tool_tasks``."""
  _warn_deprecated(_ACTIVE_NON_BLOCKING_TOOL_TASKS, stacklevel=3)
  return None if registry is None else ActiveNonBlockingToolTasksView(registry)


def set_active_non_blocking_tool_tasks(
    registry: dict[str, ActiveBackgroundTool],
    value: Mapping[str, asyncio.Task[Any]] | None,
) -> None:
  """Backs the setter of ``InvocationContext.active_non_blocking_tool_tasks``."""
  _warn_deprecated(_ACTIVE_NON_BLOCKING_TOOL_TASKS, stacklevel=5)
  ActiveNonBlockingToolTasksView(registry)._replace(value)


def pop_deprecated_fields(data: Any) -> tuple[Any, dict[str, Any]]:
  """Splits the deprecated fields off InvocationContext constructor data.

  Args:
    data: The data the InvocationContext is being validated from.

  Returns:
    The data without the deprecated fields, and the deprecated fields.
  """
  if not isinstance(data, dict):
    return data, {}
  names = (_ACTIVE_STREAMING_TOOLS, _ACTIVE_NON_BLOCKING_TOOL_TASKS)
  deprecated = {name: data[name] for name in names if name in data}
  if deprecated:
    data = {k: v for k, v in data.items() if k not in deprecated}
  return data, deprecated


def apply_deprecated_fields(
    registry: dict[str, ActiveBackgroundTool], deprecated: dict[str, Any]
) -> None:
  """Registers the deprecated fields passed to the InvocationContext constructor.

  Args:
    registry: The registry of the constructed InvocationContext.
    deprecated: The fields `pop_deprecated_fields` split off.
  """
  for name, value in deprecated.items():
    # Called from the model validator, which pydantic's __init__ calls on
    # behalf of the user's code.
    _warn_deprecated(name, stacklevel=4)
    if name == _ACTIVE_STREAMING_TOOLS:
      ActiveStreamingToolsView(registry)._replace(value)
    else:
      ActiveNonBlockingToolTasksView(registry)._replace(value)

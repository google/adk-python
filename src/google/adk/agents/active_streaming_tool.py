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

import asyncio
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr

from ..live.live_request_queue import LiveRequestQueue


class ActiveStreamingTool(BaseModel):
  """Manages streaming tool related resources during invocation."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  task: Optional[asyncio.Task[Any]] = None
  """The most recently started task of this streaming tool."""

  stream: Optional[LiveRequestQueue] = None
  """The input stream associated with the most recent task."""

  _task_streams: dict[asyncio.Task[Any], LiveRequestQueue | None] = PrivateAttr(
      default_factory=dict
  )

  def _track_task(
      self,
      task: asyncio.Task[Any],
      stream: LiveRequestQueue | None = None,
  ) -> None:
    """Tracks one call and releases its resources when it completes."""
    self.task = task
    self.stream = stream
    self._task_streams[task] = stream
    task.add_done_callback(self._discard_task)

  def _active_tasks(self) -> set[asyncio.Task[Any]]:
    """Returns a snapshot of all running calls."""
    tasks = {task for task in self._task_streams if not task.done()}
    if self.task is not None and not self.task.done():
      tasks.add(self.task)
    return tasks

  def _active_streams(self) -> list[LiveRequestQueue]:
    """Returns a snapshot of input streams for all running calls."""
    streams = [
        stream
        for task, stream in self._task_streams.items()
        if not task.done() and stream is not None
    ]
    if (
        not self._task_streams
        and self.task is not None
        and self.stream is not None
    ):
      streams.append(self.stream)
    return streams

  def _discard_tasks(self, tasks: set[asyncio.Task[Any]]) -> None:
    """Discards tracked calls without affecting calls started later."""
    for task in tasks:
      self._task_streams.pop(task, None)
    if not self._task_streams:
      self.task = None
      self.stream = None
    elif self.task in tasks:
      self._set_latest_task()

  def _discard_task(self, task: asyncio.Task[Any]) -> None:
    self._task_streams.pop(task, None)
    if self.task is task:
      self._set_latest_task()

  def _set_latest_task(self) -> None:
    if self._task_streams:
      task = next(reversed(self._task_streams))
      self.task = task
      self.stream = self._task_streams[task]
    else:
      self.task = None
      self.stream = None

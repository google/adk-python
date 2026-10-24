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

from typing import Any

from a2a.server.events import Event
from a2a.types import Message
from a2a.types import TaskStatusUpdateEvent

from .. import _compat
from ..experimental import a2a_experimental


@a2a_experimental
class TaskResultAggregator:
  """Aggregates the task status updates and provides the final task state."""

  def __init__(self) -> None:
    # The state and message from the non-failure signals (auth required,
    # input required, working). A failure is recorded separately, so that
    # these keep being tracked while the task is failed and survive a failure
    # that the run recovers from.
    self._task_state = _compat.TS_WORKING
    self._task_status_message: Message | None = None
    self._failed = False
    self._failure_message: Message | None = None

  def process_event(self, event: Event) -> None:
    """Process an event from the agent run and detect signals about the task status.

    Priority of task state: - failed - auth_required - input_required - working
    """
    if isinstance(event, TaskStatusUpdateEvent):
      if event.status.state == _compat.TS_FAILED:
        self._failed = True
        self._failure_message = _compat.normalize_message(event.status.message)
      elif event.status.state == _compat.TS_AUTH_REQUIRED:
        self._task_state = _compat.TS_AUTH_REQUIRED
        self._task_status_message = _compat.normalize_message(
            event.status.message
        )
      elif (
          event.status.state == _compat.TS_INPUT_REQUIRED
          and self._task_state != _compat.TS_AUTH_REQUIRED
      ):
        self._task_state = _compat.TS_INPUT_REQUIRED
        self._task_status_message = _compat.normalize_message(
            event.status.message
        )
      # final state is already recorded and make sure the intermediate state is
      # always working because other state may terminate the event aggregation
      # in a2a request handler
      elif self._task_state == _compat.TS_WORKING:
        self._task_status_message = _compat.normalize_message(
            event.status.message
        )
      event.status.state = _compat.TS_WORKING
      # For backward compatibility with a2a v0.3 (a2a v1.0+ removes the `final`
      # attribute from TaskStatusUpdateEvent). If we mutate the intermediate
      # event to TS_WORKING, we must clear the final flag if present.
      # Otherwise, the client runner will see final=True, terminate the stream
      # prematurely, and miss the true final event (e.g. TS_FAILED) sent after
      # the loop.
      if hasattr(event, "final"):
        event.final = False

  def _clear_failure(self) -> None:
    """Forgets the recorded failure, falling back to the non-failure state.

    Used by the executor when the run recovered from every error it emitted.
    """
    self._failed = False
    self._failure_message = None

  @property
  def task_state(self) -> Any:
    if self._failed:
      return _compat.TS_FAILED
    return self._task_state

  @property
  def task_status_message(self) -> Message | None:
    if self._failed:
      return self._failure_message
    return self._task_status_message

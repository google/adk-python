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
from ...events._abort_events import _INVOCATION_ABORTED
from ..converters.utils import _get_adk_metadata_key

_ERROR_CODE_KEY = _get_adk_metadata_key("error_code")


def _is_canceled_status_event(event: TaskStatusUpdateEvent) -> bool:
  if event.status.state == _compat.TS_CANCELED:
    return True
  if event.status.state != _compat.TS_FAILED:
    return False
  msg = _compat.normalize_message(event.status.message)
  return _INVOCATION_ABORTED in (
      _compat.metadata_get(event.metadata, _ERROR_CODE_KEY),
      _compat.metadata_get(msg.metadata, _ERROR_CODE_KEY) if msg else None,
  )


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

    Priority of task state:
      - canceled - failed - auth_required - input_required - working
    """
    if isinstance(event, TaskStatusUpdateEvent):
      if _is_canceled_status_event(event):
        self._task_state = _compat.TS_CANCELED
        self._task_status_message = _compat.normalize_message(
            event.status.message
        )
      elif self._task_state == _compat.TS_CANCELED:
        pass
      elif event.status.state == _compat.TS_FAILED:
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

  def _keep_failures(self, messages: list[Message | None]) -> None:
    """Replaces the recorded failures with ``messages``, in emission order.

    Used by the executor to drop the failures the run recovered from; with no
    messages left it falls back to the non-failure state.
    """
    self._failed = bool(messages)
    self._failure_message = (
        _compat.normalize_message(messages[-1]) if messages else None
    )

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

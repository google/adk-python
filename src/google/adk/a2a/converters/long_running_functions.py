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

from datetime import datetime
from datetime import timezone
from typing import List
from typing import Set
import uuid

from a2a.server.agent_execution.context import RequestContext
from a2a.types import DataPart
from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Role
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.genai import types as genai_types

from ...events.event import Event
from ...flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from .part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from .part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from .part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
from .part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from .part_converter import A2APartToGenAIPartConverter
from .part_converter import convert_a2a_part_to_genai_part
from .utils import _get_adk_metadata_key


class LongRunningFunctions:
  """Keeps track of long running function calls and related responses."""

  def __init__(
      self, part_converter: A2APartToGenAIPartConverter | None = None
  ) -> None:
    self._parts: List[genai_types.Part] = []
    self._long_running_tool_ids: Set[str] = set()
    self._part_converter = part_converter or convert_a2a_part_to_genai_part
    self._task_state: TaskState = TaskState.input_required

  def has_long_running_function_calls(self) -> bool:
    """Returns True if there are long running function calls."""
    return bool(self._long_running_tool_ids)

  def process_event(self, event: Event) -> Event:
    """Processes parts to extract long running calls and responses.

    Returns a copy of the input event with processed parts removed from
    event.content.parts.

    Args:
      event: The ADK event containing long running tool IDs and content parts.
    """
    event = event.model_copy(deep=True)
    if not event.content or not event.content.parts:
      return event

    kept_parts = []
    for part in event.content.parts:
      should_remove = False
      if part.function_call:
        if (
            event.long_running_tool_ids
            and part.function_call.id in event.long_running_tool_ids
        ):
          if not event.partial:
            self._parts.append(part)
            self._long_running_tool_ids.add(part.function_call.id)
          should_remove = True

      elif part.function_response:
        if part.function_response.id in self._long_running_tool_ids:
          if not event.partial:
            self._parts.append(part)
          should_remove = True

      if not should_remove:
        kept_parts.append(part)

    event.content.parts = kept_parts
    return event

  def create_long_running_function_call_event(
      self,
      task_id: str,
      context_id: str,
  ) -> TaskStatusUpdateEvent:
    """Creates a task status update event for the long running function calls."""
    if not self._long_running_tool_ids:
      return None

    a2a_parts = self._return_long_running_parts()
    if not a2a_parts:
      return None

    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=TaskStatus(
            state=self._task_state,
            message=Message(
                message_id=str(uuid.uuid4()),
                role=Role.agent,
                parts=a2a_parts,
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        final=True,
    )

  def _return_long_running_parts(self) -> List[A2APart]:
    """Converts long-running parts to A2A parts."""
    if not self._long_running_tool_ids:
      return []

    output_parts = []
    for part in self._parts:
      a2a_parts = self._part_converter(part)
      if not isinstance(a2a_parts, list):
        a2a_parts = [a2a_parts] if a2a_parts else []
      for a2a_part in a2a_parts:
        self._mark_long_running_function_call(a2a_part)
        output_parts.append(a2a_part)

    return output_parts

  def _mark_long_running_function_call(self, a2a_part: A2APart) -> None:
    """Processes long-running tool metadata for an A2A part.

    Args:
      a2a_part: The A2A part to potentially mark as long-running.
    """

    if (
        isinstance(a2a_part.root, DataPart)
        and a2a_part.root.metadata
        and a2a_part.root.metadata.get(
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        )
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    ):
      a2a_part.root.metadata[
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
      ] = True
      # If the function is a request for EUC, set the task state to
      # auth_required. Otherwise, set it to input_required. Save the state of
      # the last function call, as it will be the state of the task.
      if a2a_part.root.metadata.get("name") == REQUEST_EUC_FUNCTION_CALL_NAME:
        self._task_state = TaskState.auth_required
      else:
        self._task_state = TaskState.input_required


def handle_user_input(context: RequestContext) -> TaskStatusUpdateEvent | None:
  """Processes user input events, validating function responses."""

  if (
      not context.current_task
      or not context.current_task.status
      or (
          context.current_task.status.state != TaskState.input_required
          and context.current_task.status.state != TaskState.auth_required
      )
  ):
    return None

  # If the task is in input_required or auth_required state, we expect the user
  # to provide a response for the function call. Check if the user input
  # contains a function response.
  for a2a_part in context.message.parts:
    if (
        isinstance(a2a_part.root, DataPart)
        and a2a_part.root.metadata
        and a2a_part.root.metadata.get(
            _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        )
        == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
    ):
      return None

  return TaskStatusUpdateEvent(
      task_id=context.task_id,
      context_id=context.context_id,
      status=TaskStatus(
          state=context.current_task.status.state,
          timestamp=datetime.now(timezone.utc).isoformat(),
          message=Message(
              message_id=str(uuid.uuid4()),
              role=Role.agent,
              parts=[
                  A2APart(
                      root=TextPart(
                          text=(
                              "It was not provided a function response for the"
                              " function call."
                          )
                      )
                  )
              ],
          ),
      ),
      final=True,
  )

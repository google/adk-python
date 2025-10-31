# Copyright 2025 Google LLC
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

from collections.abc import Callable
from datetime import datetime
from datetime import timezone
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import uuid

from a2a.server.events import Event as A2AEvent
from a2a.types import DataPart
from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Role
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from google.genai import types as genai_types

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..experimental import a2a_experimental
from .part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from .part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from .part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from .part_converter import A2APartToGenAIPartConverter
from .part_converter import convert_a2a_part_to_genai_part
from .part_converter import convert_genai_part_to_a2a_part
from .part_converter import GenAIPartToA2APartConverter
from .utils import _get_adk_metadata_key

# Constants

ARTIFACT_ID_SEPARATOR = "-"
DEFAULT_ERROR_MESSAGE = "An error occurred during processing"

# Logger
logger = logging.getLogger("google_adk." + __name__)


AdkEventToA2AEventsConverter = Callable[
    [
        Event,
        InvocationContext,
        Optional[str],
        Optional[str],
        GenAIPartToA2APartConverter,
    ],
    List[A2AEvent],
]
"""A callable that converts an ADK Event into a list of A2A events.

This interface allows for custom logic to map ADK's event structure to the
event structure expected by the A2A server.

Args:
    event: The source ADK Event to convert.
    invocation_context: The context of the ADK agent invocation.
    task_id: The ID of the A2A task being processed.
    context_id: The context ID from the A2A request.
    part_converter: A function to convert GenAI content parts to A2A
      parts.

Returns:
    A list of A2A events.
"""


def _serialize_metadata_value(value: Any) -> str:
  """Safely serializes metadata values to string format.

  Args:
    value: The value to serialize.

  Returns:
    String representation of the value.
  """
  if hasattr(value, "model_dump"):
    try:
      return value.model_dump(exclude_none=True, by_alias=True)
    except Exception as e:
      logger.warning("Failed to serialize metadata value: %s", e)
      return str(value)
  return str(value)


def _get_context_metadata(
    event: Event, invocation_context: InvocationContext
) -> Dict[str, str]:
  """Gets the context metadata for the event.

  Args:
    event: The ADK event to extract metadata from.
    invocation_context: The invocation context containing session information.

  Returns:
    A dictionary containing the context metadata.

  Raises:
    ValueError: If required fields are missing from event or context.
  """
  if not event:
    raise ValueError("Event cannot be None")
  if not invocation_context:
    raise ValueError("Invocation context cannot be None")

  try:
    metadata = {
        _get_adk_metadata_key("app_name"): invocation_context.app_name,
        _get_adk_metadata_key("user_id"): invocation_context.user_id,
        _get_adk_metadata_key("session_id"): invocation_context.session.id,
        _get_adk_metadata_key("invocation_id"): event.invocation_id,
        _get_adk_metadata_key("author"): event.author,
    }

    # Add optional metadata fields if present
    optional_fields = [
        ("branch", event.branch),
        ("grounding_metadata", event.grounding_metadata),
        ("custom_metadata", event.custom_metadata),
        ("usage_metadata", event.usage_metadata),
        ("error_code", event.error_code),
    ]

    for field_name, field_value in optional_fields:
      if field_value is not None:
        metadata[_get_adk_metadata_key(field_name)] = _serialize_metadata_value(
            field_value
        )

    return metadata

  except Exception as e:
    logger.error("Failed to create context metadata: %s", e)
    raise


def _create_artifact_id(
    app_name: str, user_id: str, session_id: str, filename: str, version: int
) -> str:
  """Creates a unique artifact ID.

  Args:
    app_name: The application name.
    user_id: The user ID.
    session_id: The session ID.
    filename: The artifact filename.
    version: The artifact version.

  Returns:
    A unique artifact ID string.
  """
  components = [app_name, user_id, session_id, filename, str(version)]
  return ARTIFACT_ID_SEPARATOR.join(components)


def _process_long_running_tool(a2a_part: A2APart, event: Event) -> None:
  """Processes long-running tool metadata for an A2A part.

  Args:
    a2a_part: The A2A part to potentially mark as long-running.
    event: The ADK event containing long-running tool information.
  """
  if (
      isinstance(a2a_part.root, DataPart)
      and event.long_running_tool_ids
      and a2a_part.root.metadata
      and a2a_part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
      )
      == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      and a2a_part.root.data.get("id") in event.long_running_tool_ids
  ):
    a2a_part.root.metadata[
        _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
    ] = True


def convert_a2a_task_to_event(
    a2a_task: Task,
    author: Optional[str] = None,
    invocation_context: Optional[InvocationContext] = None,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> Event:
  """Converts an A2A task to an ADK event.

  Args:
    a2a_task: The A2A task to convert. Must not be None.
    author: The author of the event. Defaults to "a2a agent" if not provided.
    invocation_context: The invocation context containing session information.
      If provided, the branch will be set from the context.
    part_converter: The function to convert A2A part to GenAI part.

  Returns:
    An ADK Event object representing the converted task.

  Raises:
    ValueError: If a2a_task is None.
    RuntimeError: If conversion of the underlying message fails.
  """
  if a2a_task is None:
    raise ValueError("A2A task cannot be None")

  try:
    # Extract message from task status or history
    message = None
    if a2a_task.artifacts:
      message = Message(
          message_id="", role=Role.agent, parts=a2a_task.artifacts[-1].parts
      )
    elif a2a_task.status and a2a_task.status.message:
      message = a2a_task.status.message
    elif a2a_task.history:
      message = a2a_task.history[-1]

    # Convert message if available
    if message:
      try:
        return convert_a2a_message_to_event(
            message, author, invocation_context, part_converter=part_converter
        )
      except Exception as e:
        logger.error("Failed to convert A2A task message to event: %s", e)
        raise RuntimeError(f"Failed to convert task message: {e}") from e

    # Create minimal event if no message is available
    return Event(
        invocation_id=(
            invocation_context.invocation_id
            if invocation_context
            else str(uuid.uuid4())
        ),
        author=author or "a2a agent",
        branch=invocation_context.branch if invocation_context else None,
    )

  except Exception as e:
    logger.error("Failed to convert A2A task to event: %s", e)
    raise


@a2a_experimental
def convert_a2a_message_to_event(
    a2a_message: Message,
    author: Optional[str] = None,
    invocation_context: Optional[InvocationContext] = None,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> Event:
  """Converts an A2A message to an ADK event.

  Args:
    a2a_message: The A2A message to convert. Must not be None.
    author: The author of the event. Defaults to "a2a agent" if not provided.
    invocation_context: The invocation context containing session information.
      If provided, the branch will be set from the context.
    part_converter: The function to convert A2A part to GenAI part.

  Returns:
    An ADK Event object with converted content and long-running tool metadata.

  Raises:
    ValueError: If a2a_message is None.
    RuntimeError: If conversion of message parts fails.
  """
  if a2a_message is None:
    raise ValueError("A2A message cannot be None")

  if not a2a_message.parts:
    logger.warning(
        "A2A message has no parts, creating event with empty content"
    )
    return Event(
        invocation_id=(
            invocation_context.invocation_id
            if invocation_context
            else str(uuid.uuid4())
        ),
        author=author or "a2a agent",
        branch=invocation_context.branch if invocation_context else None,
        content=genai_types.Content(role="model", parts=[]),
    )

  try:
    parts = []
    long_running_tool_ids = set()

    for a2a_part in a2a_message.parts:
      try:
        part = part_converter(a2a_part)
        if part is None:
          logger.warning("Failed to convert A2A part, skipping: %s", a2a_part)
          continue

        # Check for long-running tools
        if (
            a2a_part.root.metadata
            and a2a_part.root.metadata.get(
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
                )
            )
            is True
        ):
          long_running_tool_ids.add(part.function_call.id)

        parts.append(part)

      except Exception as e:
        logger.error("Failed to convert A2A part: %s, error: %s", a2a_part, e)
        # Continue processing other parts instead of failing completely
        continue

    if not parts:
      logger.warning(
          "No parts could be converted from A2A message %s", a2a_message
      )

    return Event(
        invocation_id=(
            invocation_context.invocation_id
            if invocation_context
            else str(uuid.uuid4())
        ),
        author=author or "a2a agent",
        branch=invocation_context.branch if invocation_context else None,
        long_running_tool_ids=long_running_tool_ids
        if long_running_tool_ids
        else None,
        content=genai_types.Content(
            role="model",
            parts=parts,
        ),
    )

  except Exception as e:
    logger.error("Failed to convert A2A message to event: %s", e)
    raise RuntimeError(f"Failed to convert message: {e}") from e


@a2a_experimental
def convert_event_to_a2a_message(
    event: Event,
    invocation_context: InvocationContext,
    role: Role = Role.agent,
    part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
) -> Optional[Message]:
  """Converts an ADK event to an A2A message.

  Args:
    event: The ADK event to convert.
    invocation_context: The invocation context.
    role: The role of the message.
    part_converter: The function to convert GenAI part to A2A part.

  Returns:
    An A2A Message if the event has content, None otherwise.

  Raises:
    ValueError: If required parameters are invalid.
  """
  if not event:
    raise ValueError("Event cannot be None")
  if not invocation_context:
    raise ValueError("Invocation context cannot be None")

  if not event.content or not event.content.parts:
    return None

  try:
    a2a_parts = []
    for part in event.content.parts:
      a2a_part = part_converter(part)
      if a2a_part:
        a2a_parts.append(a2a_part)
        _process_long_running_tool(a2a_part, event)

    if a2a_parts:
      return Message(message_id=str(uuid.uuid4()), role=role, parts=a2a_parts)

  except Exception as e:
    logger.error("Failed to convert event to status message: %s", e)
    raise

  return None


def _create_error_status_event(
    event: Event,
    invocation_context: InvocationContext,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> TaskStatusUpdateEvent:
  """Creates a TaskStatusUpdateEvent for error scenarios.

  Args:
    event: The ADK event containing error information.
    invocation_context: The invocation context.
    task_id: Optional task ID to use for generated events.
    context_id: Optional Context ID to use for generated events.

  Returns:
    A TaskStatusUpdateEvent with FAILED state.
  """
  error_message = getattr(event, "error_message", None) or DEFAULT_ERROR_MESSAGE

  # Get context metadata and add error code
  event_metadata = _get_context_metadata(event, invocation_context)
  if event.error_code:
    event_metadata[_get_adk_metadata_key("error_code")] = str(event.error_code)

  return TaskStatusUpdateEvent(
      task_id=task_id,
      context_id=context_id,
      metadata=event_metadata,
      status=TaskStatus(
          state=TaskState.failed,
          message=Message(
              message_id=str(uuid.uuid4()),
              role=Role.agent,
              parts=[TextPart(text=error_message)],
              metadata={
                  _get_adk_metadata_key("error_code"): str(event.error_code)
              }
              if event.error_code
              else {},
          ),
          timestamp=datetime.now(timezone.utc).isoformat(),
      ),
      final=False,
  )


def _create_status_update_event(
    message: Message,
    invocation_context: InvocationContext,
    event: Event,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> TaskStatusUpdateEvent:
  """Creates a TaskStatusUpdateEvent for running scenarios.

  Args:
    message: The A2A message to include.
    invocation_context: The invocation context.
    event: The ADK event.
    task_id: Optional task ID to use for generated events.
    context_id: Optional Context ID to use for generated events.


  Returns:
    A TaskStatusUpdateEvent with RUNNING state.
  """
  status = TaskStatus(
      state=TaskState.working,
      message=message,
      timestamp=datetime.now(timezone.utc).isoformat(),
  )

  if any(
      part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
      )
      == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      and part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
      )
      is True
      and part.root.data.get("name") == REQUEST_EUC_FUNCTION_CALL_NAME
      for part in message.parts
      if part.root.metadata
  ):
    status.state = TaskState.auth_required
  elif any(
      part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
      )
      == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      and part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
      )
      is True
      for part in message.parts
      if part.root.metadata
  ):
    status.state = TaskState.input_required

  return TaskStatusUpdateEvent(
      task_id=task_id,
      context_id=context_id,
      status=status,
      metadata=_get_context_metadata(event, invocation_context),
      final=False,
  )


@a2a_experimental
def convert_event_to_a2a_events(
    event: Event,
    invocation_context: InvocationContext,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
) -> List[A2AEvent]:
  """Converts a GenAI event to a list of A2A events.

  Args:
    event: The ADK event to convert.
    invocation_context: The invocation context.
    task_id: Optional task ID to use for generated events.
    context_id: Optional Context ID to use for generated events.
    part_converter: The function to convert GenAI part to A2A part.

  Returns:
    A list of A2A events representing the converted ADK event.

  Raises:
    ValueError: If required parameters are invalid.
  """
  if not event:
    raise ValueError("Event cannot be None")
  if not invocation_context:
    raise ValueError("Invocation context cannot be None")

  a2a_events = []

  try:

    # Handle error scenarios
    if event.error_code:
      error_event = _create_error_status_event(
          event, invocation_context, task_id, context_id
      )
      a2a_events.append(error_event)

    # Handle regular message content
    message = convert_event_to_a2a_message(
        event, invocation_context, part_converter=part_converter
    )
    if message:
      running_event = _create_status_update_event(
          message, invocation_context, event, task_id, context_id
      )
      a2a_events.append(running_event)

  except Exception as e:
    logger.error("Failed to convert event to A2A events: %s", e)
    raise

  return a2a_events

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

import datetime
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import uuid

from a2a.server.events import Event as A2AEvent
from a2a.types import Artifact
from a2a.types import DataPart
from a2a.types import Message
from a2a.types import Role
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...utils.feature_decorator import working_in_progress
from .part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from .part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from .part_converter import convert_genai_part_to_a2a_part
from .utils import _get_adk_metadata_key

# Constants

ARTIFACT_ID_SEPARATOR = "-"
DEFAULT_ERROR_MESSAGE = "An error occurred during processing"

# Logger
logger = logging.getLogger("google_adk." + __name__)


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


def _convert_artifact_to_a2a_events(
    event: Event,
    invocation_context: InvocationContext,
    filename: str,
    version: int,
) -> TaskArtifactUpdateEvent:
  """Converts a new artifact version to an A2A TaskArtifactUpdateEvent.

  Args:
    event: The ADK event containing the artifact information.
    invocation_context: The invocation context.
    filename: The name of the artifact file.
    version: The version number of the artifact.

  Returns:
    A TaskArtifactUpdateEvent representing the artifact update.

  Raises:
    ValueError: If required parameters are invalid.
    RuntimeError: If artifact loading fails.
  """
  if not filename:
    raise ValueError("Filename cannot be empty")
  if version < 0:
    raise ValueError("Version must be non-negative")

  try:
    artifact_part = invocation_context.artifact_service.load_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=filename,
        version=version,
    )

    converted_part = convert_genai_part_to_a2a_part(part=artifact_part)
    if not converted_part:
      raise RuntimeError(f"Failed to convert artifact part for {filename}")

    artifact_id = _create_artifact_id(
        invocation_context.app_name,
        invocation_context.user_id,
        invocation_context.session.id,
        filename,
        version,
    )

    return TaskArtifactUpdateEvent(
        taskId=str(uuid.uuid4()),
        append=False,
        contextId=invocation_context.session.id,
        lastChunk=True,
        artifact=Artifact(
            artifactId=artifact_id,
            name=filename,
            metadata={
                "filename": filename,
                "version": version,
            },
            parts=[converted_part],
        ),
    )
  except Exception as e:
    logger.error(
        "Failed to convert artifact for %s, version %s: %s",
        filename,
        version,
        e,
    )
    raise RuntimeError(f"Artifact conversion failed: {e}") from e


def _process_long_running_tool(a2a_part, event: Event) -> None:
  """Processes long-running tool metadata for an A2A part.

  Args:
    a2a_part: The A2A part to potentially mark as long-running.
    event: The ADK event containing long-running tool information.
  """
  if (
      isinstance(a2a_part.root, DataPart)
      and event.long_running_tool_ids
      and a2a_part.root.metadata.get(
          _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
      )
      == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      and a2a_part.root.metadata.get("id") in event.long_running_tool_ids
  ):
    a2a_part.root.metadata[_get_adk_metadata_key("is_long_running")] = True


@working_in_progress
def convert_event_to_a2a_status_message(
    event: Event, invocation_context: InvocationContext
) -> Optional[Message]:
  """Converts an ADK event to an A2A message.

  Args:
    event: The ADK event to convert.
    invocation_context: The invocation context.

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
      a2a_part = convert_genai_part_to_a2a_part(part)
      if a2a_part:
        a2a_parts.append(a2a_part)
        _process_long_running_tool(a2a_part, event)

    if a2a_parts:
      return Message(
          messageId=str(uuid.uuid4()), role=Role.agent, parts=a2a_parts
      )

  except Exception as e:
    logger.error("Failed to convert event to status message: %s", e)
    raise

  return None


def _create_error_status_event(
    event: Event, invocation_context: InvocationContext
) -> TaskStatusUpdateEvent:
  """Creates a TaskStatusUpdateEvent for error scenarios.

  Args:
    event: The ADK event containing error information.
    invocation_context: The invocation context.

  Returns:
    A TaskStatusUpdateEvent with FAILED state.
  """
  error_message = getattr(event, "error_message", None) or DEFAULT_ERROR_MESSAGE

  return TaskStatusUpdateEvent(
      taskId=str(uuid.uuid4()),
      contextId=invocation_context.session.id,
      final=False,
      metadata=_get_context_metadata(event, invocation_context),
      status=TaskStatus(
          state=TaskState.failed,
          message=Message(
              messageId=str(uuid.uuid4()),
              role=Role.agent,
              parts=[TextPart(text=error_message)],
          ),
          timestamp=datetime.datetime.now().isoformat(),
      ),
  )


def _create_running_status_event(
    message: Message, invocation_context: InvocationContext, event: Event
) -> TaskStatusUpdateEvent:
  """Creates a TaskStatusUpdateEvent for running scenarios.

  Args:
    message: The A2A message to include.
    invocation_context: The invocation context.
    event: The ADK event.

  Returns:
    A TaskStatusUpdateEvent with RUNNING state.
  """
  return TaskStatusUpdateEvent(
      taskId=str(uuid.uuid4()),
      contextId=invocation_context.session.id,
      final=False,
      status=TaskStatus(
          state=TaskState.working,
          message=message,
          timestamp=datetime.datetime.now().isoformat(),
      ),
      metadata=_get_context_metadata(event, invocation_context),
  )


@working_in_progress
def convert_event_to_a2a_events(
    event: Event, invocation_context: InvocationContext
) -> List[A2AEvent]:
  """Converts a GenAI event to a list of A2A events.

  Args:
    event: The ADK event to convert.
    invocation_context: The invocation context.

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
    # Handle artifact deltas
    if event.actions.artifact_delta:
      for filename, version in event.actions.artifact_delta.items():
        artifact_event = _convert_artifact_to_a2a_events(
            event, invocation_context, filename, version
        )
        a2a_events.append(artifact_event)

    # Handle error scenarios
    if event.error_code:
      error_event = _create_error_status_event(event, invocation_context)
      a2a_events.append(error_event)

    # Handle regular message content
    message = convert_event_to_a2a_status_message(event, invocation_context)
    if message:
      running_event = _create_running_status_event(
          message, invocation_context, event
      )
      a2a_events.append(running_event)

  except Exception as e:
    logger.error("Failed to convert event to A2A events: %s", e)
    raise

  return a2a_events

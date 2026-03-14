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

from collections.abc import Callable
from datetime import datetime
from datetime import timezone
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import uuid

from a2a.server.events import Event as A2AEvent
from a2a.types import Artifact
from a2a.types import DataPart
from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Role
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart

from ...events.event import Event
from ...flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..experimental import a2a_experimental
from .part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from .part_converter import A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
from .part_converter import A2A_DATA_PART_METADATA_TYPE_KEY
from .part_converter import convert_genai_part_to_a2a_part
from .part_converter import GenAIPartToA2APartConverter
from .utils import _get_adk_metadata_key

# Constants
DEFAULT_ERROR_MESSAGE = "An error occurred during processing"

# Logger
logger = logging.getLogger("google_adk." + __name__)

A2AUpdateEvent = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]

AdkEventToA2AEventsConverter = Callable[
    [
        Event,
        Optional[Dict[str, str]],
        Optional[str],
        Optional[str],
        GenAIPartToA2APartConverter,
    ],
    List[A2AUpdateEvent],
]
"""A callable that converts an ADK Event into a list of A2A events.

This interface allows for custom logic to map ADK's event structure to the
event structure expected by the A2A server.

Args:
    event: The source ADK Event to convert.
    agents_artifacts: State map for tracking active artifact IDs across chunks.
    task_id: The ID of the A2A task being processed.
    context_id: The context ID from the A2A request.
    part_converter: A function to convert GenAI content parts to A2A
      parts.

Returns:
    A list of A2A events.
"""


def _convert_adk_parts_to_a2a_parts(
    event: Event,
    part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
) -> Optional[List[A2APart]]:
  """Converts an ADK event to an A2A parts list.

  Args:
    event: The ADK event to convert.
    part_converter: The function to convert GenAI part to A2A part.

  Returns:
    A list of A2A parts representing the converted ADK event.

  Raises:
    ValueError: If required parameters are invalid.
  """
  if not event:
    raise ValueError("Event cannot be None")

  if not event.content or not event.content.parts:
    return []

  try:
    output_parts = []
    for part in event.content.parts:
      a2a_parts = part_converter(part)
      if not isinstance(a2a_parts, list):
        a2a_parts = [a2a_parts] if a2a_parts else []
      for a2a_part in a2a_parts:
        output_parts.append(a2a_part)

    return output_parts

  except Exception as e:
    logger.error("Failed to convert event to status message: %s", e)
    raise


def create_error_status_event(
    event: Event,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> TaskStatusUpdateEvent:
  """Creates a TaskStatusUpdateEvent for error scenarios.

  Args:
    event: The ADK event containing error information.
    task_id: Optional task ID to use for generated events.
    context_id: Optional Context ID to use for generated events.

  Returns:
    A TaskStatusUpdateEvent with FAILED state.
  """
  error_message = getattr(event, "error_message", None) or DEFAULT_ERROR_MESSAGE

  error_event = TaskStatusUpdateEvent(
      task_id=task_id,
      context_id=context_id,
      status=TaskStatus(
          state=TaskState.failed,
          message=Message(
              message_id=str(uuid.uuid4()),
              role=Role.agent,
              parts=[A2APart(root=TextPart(text=error_message))],
          ),
          timestamp=datetime.now(timezone.utc).isoformat(),
      ),
      final=True,
  )
  return _add_event_metadata(event, [error_event])[0]


@a2a_experimental
def convert_event_to_a2a_events(
    event: Event,
    agents_artifacts: Dict[str, str],
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
    part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
) -> List[A2AUpdateEvent]:
  """Converts a GenAI event to a list of A2A StatusUpdate and ArtifactUpdate events.

  Args:
    event: The ADK event to convert.
    agents_artifacts: State map for tracking active artifact IDs across chunks.
    task_id: Optional task ID to use for generated events.
    context_id: Optional Context ID to use for generated events.
    part_converter: The function to convert GenAI part to A2A part.

  Returns:
    A list of A2A update events representing the converted ADK event.

  Raises:
    ValueError: If required parameters are invalid.
  """
  if not event:
    raise ValueError("Event cannot be None")
  if agents_artifacts is None:
    raise ValueError("Agents artifacts cannot be None")

  a2a_events = []
  try:
    a2a_parts = _convert_adk_parts_to_a2a_parts(
        event, part_converter=part_converter
    )
    # Handle artifact updates for normal parts
    if a2a_parts:
      agent_name = event.author
      partial = event.partial or False

      artifact_id = agents_artifacts.get(agent_name)
      if artifact_id:
        append = partial
        if not partial:
          del agents_artifacts[agent_name]
      else:
        artifact_id = str(uuid.uuid4())
        # TODO: Clarify if new artifact id must have append=False
        append = False
        if partial:
          agents_artifacts[agent_name] = artifact_id

      a2a_events.append(
          TaskArtifactUpdateEvent(
              task_id=task_id,
              context_id=context_id,
              last_chunk=not partial,
              append=append,
              artifact=Artifact(
                  artifact_id=artifact_id,
                  parts=a2a_parts,
              ),
          )
      )

    a2a_events = _add_event_metadata(event, a2a_events)
    return a2a_events

  except Exception as e:
    logger.error("Failed to convert event to A2A events: %s", e)
    raise


def _serialize_value(value: Any) -> Optional[Any]:
  """Serializes a value and returns it if it contains meaningful content.

  Returns None if the value is empty or missing.
  """
  if value is None:
    return None

  # Handle Pydantic models
  if hasattr(value, "model_dump"):
    try:
      dumped = value.model_dump(
          exclude_none=True,
          exclude_unset=True,
          exclude_defaults=True,
          by_alias=True,
      )
      return dumped if dumped else None
    except Exception as e:
      logger.warning("Failed to serialize Pydantic model, falling back: %s", e)
      return str(value)

  return str(value)


# TODO: Clarify if this metadata needs to be translated back into the ADK event
def _add_event_metadata(
    event: Event, a2a_events: List[A2AEvent]
) -> List[A2AEvent]:
  """Gets the context metadata for the event and applies it to A2A events."""
  if not event:
    raise ValueError("Event cannot be None")

  metadata_values = {
      "invocation_id": event.invocation_id,
      "author": event.author,
      "event_id": event.id,
      "branch": event.branch,
      "citation_metadata": event.citation_metadata,
      "grounding_metadata": event.grounding_metadata,
      "custom_metadata": event.custom_metadata,
      "usage_metadata": event.usage_metadata,
      "error_code": event.error_code,
      "actions": event.actions,
  }

  metadata = {}
  for field_name, field_value in metadata_values.items():
    value = _serialize_value(field_value)
    if value is not None:
      metadata[_get_adk_metadata_key(field_name)] = value

  for a2a_event in a2a_events:
    if isinstance(a2a_event, TaskStatusUpdateEvent):
      a2a_event.status.message.metadata = metadata.copy()
    elif isinstance(a2a_event, TaskArtifactUpdateEvent):
      a2a_event.artifact.metadata = metadata.copy()

  return a2a_events

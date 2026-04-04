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
import json
import logging
from typing import Any
from typing import List
from typing import Optional
import uuid

from a2a.types import Message
from a2a.types import Part as A2APart
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent
from google.genai import types as genai_types
from pydantic import ValidationError

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...events.event_actions import EventActions
from ..experimental import a2a_experimental
from .part_converter import A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY
from .part_converter import A2APartToGenAIPartConverter
from .part_converter import convert_a2a_part_to_genai_part
from .utils import _get_adk_metadata_key

# Logger
logger = logging.getLogger("google_adk." + __name__)

A2AMessageToEventConverter = Callable[
    [
        Message,
        Optional[str],
        Optional[InvocationContext],
        A2APartToGenAIPartConverter,
    ],
    Optional[Event],
]
"""A Callable that converts an A2A Message to an ADK Event.

Args:
  Message: The A2A message to convert.
  Optional[str]: The author of the event.
  Optional[InvocationContext]: The invocation context.
  A2APartToGenAIPartConverter: The part converter function.

Returns:
  Optional[Event]: The converted ADK Event.
"""

A2ATaskToEventConverter = Callable[
    [
        Task,
        Optional[str],
        Optional[InvocationContext],
        A2APartToGenAIPartConverter,
    ],
    Optional[Event],
]
"""A Callable that converts an A2A Task to an ADK Event.

Args:
  Task: The A2A task to convert.
  Optional[str]: The author of the event.
  Optional[InvocationContext]: The invocation context.
  A2APartToGenAIPartConverter: The part converter function.

Returns:
  Optional[Event]: The converted ADK Event.
"""

A2AStatusUpdateToEventConverter = Callable[
    [
        TaskStatusUpdateEvent,
        Optional[str],
        Optional[InvocationContext],
        A2APartToGenAIPartConverter,
    ],
    Optional[Event],
]
"""A Callable that converts an A2A TaskStatusUpdateEvent to an ADK Event.

Args:
  TaskStatusUpdateEvent: The A2A status update event to convert.
  Optional[str]: The author of the event.
  Optional[InvocationContext]: The invocation context.
  A2APartToGenAIPartConverter: The part converter function.

Returns:
  Optional[Event]: The converted ADK Event.
"""

A2AArtifactUpdateToEventConverter = Callable[
    [
        TaskArtifactUpdateEvent,
        Optional[str],
        Optional[InvocationContext],
        A2APartToGenAIPartConverter,
    ],
    Optional[Event],
]
"""A Callable that converts an A2A TaskArtifactUpdateEvent to an ADK Event.

Args:
  TaskArtifactUpdateEvent: The A2A artifact update event to convert.
  Optional[str]: The author of the event.
  Optional[InvocationContext]: The invocation context.
  A2APartToGenAIPartConverter: The part converter function.

Returns:
  Optional[Event]: The converted ADK Event.
"""


def _convert_a2a_parts_to_adk_parts(
    a2a_parts: List[A2APart],
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> tuple[List[genai_types.Part], set[str]]:
  """Converts a list of A2A parts to a list of ADK parts."""
  output_parts = []
  long_running_function_ids = set()

  for a2a_part in a2a_parts:
    try:
      parts = part_converter(a2a_part)
      if not isinstance(parts, list):
        parts = [parts] if parts else []
      if not parts:
        logger.warning("Failed to convert A2A part, skipping: %s", a2a_part)
        continue

      # Check for long-running functions
      if (
          a2a_part.root.metadata
          and a2a_part.root.metadata.get(
              _get_adk_metadata_key(A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY)
          )
          is True
      ):
        for part in parts:
          if part.function_call:
            long_running_function_ids.add(part.function_call.id)

      output_parts.extend(parts)

    except Exception as e:
      logger.error("Failed to convert A2A part: %s, error: %s", a2a_part, e)
      # Continue processing other parts instead of failing completely
      continue

  if not output_parts:
    logger.warning("No parts could be converted from A2A message")

  return output_parts, long_running_function_ids


def _create_event(
    output_parts: List[genai_types.Part],
    invocation_context: Optional[InvocationContext],
    author: Optional[str],
    actions: Optional[EventActions] = None,
    long_running_function_ids: Optional[set[str]] = None,
    partial: bool = False,
) -> Optional[Event]:
  """Creates an ADK event from parts and metadata."""
  event_actions = actions or EventActions()
  if not output_parts and not event_actions.model_dump(
      exclude_none=True, exclude_defaults=True
  ):
    return None

  event = Event(
      invocation_id=(
          invocation_context.invocation_id
          if invocation_context
          else str(uuid.uuid4())
      ),
      author=author or "a2a agent",
      branch=invocation_context.branch if invocation_context else None,
      actions=event_actions,
      long_running_tool_ids=(
          long_running_function_ids if long_running_function_ids else None
      ),
      content=(
          genai_types.Content(
              role="model",
              parts=output_parts,
          )
          if output_parts
          else None
      ),
      partial=partial,
  )

  return event


def _parse_adk_metadata_value(value: Any) -> Any:
  """Parses ADK metadata values serialized through A2A."""
  if not isinstance(value, str):
    return value

  try:
    return json.loads(value)
  except json.JSONDecodeError:
    return value


def _extract_event_actions(
    metadata: Optional[dict[str, Any]],
) -> EventActions:
  """Extracts ADK event actions from A2A metadata."""
  if not metadata:
    return EventActions()

  raw_actions = metadata.get(_get_adk_metadata_key("actions"))
  if raw_actions is None:
    return EventActions()

  parsed_actions = _parse_adk_metadata_value(raw_actions)
  if not isinstance(parsed_actions, dict):
    logger.warning(
        "Ignoring invalid ADK actions metadata of type %s",
        type(parsed_actions).__name__,
    )
    return EventActions()

  try:
    return EventActions.model_validate(parsed_actions)
  except ValidationError as error:
    logger.warning("Ignoring invalid ADK actions metadata: %s", error)
    return EventActions()


def _merge_top_level_dicts(
    base: dict[str, Any], new_values: dict[str, Any]
) -> dict[str, Any]:
  """Merges dictionaries while preserving top-level overwrite semantics."""
  merged = dict(base)
  for key, value in new_values.items():
    if (
        key in merged
        and isinstance(merged[key], dict)
        and isinstance(value, dict)
    ):
      merged[key] = {**merged[key], **value}
    else:
      merged[key] = value
  return merged


def _merge_event_actions(
    existing_actions: EventActions, new_actions: EventActions
) -> EventActions:
  """Merges action metadata from multiple A2A sources."""
  merged_actions_data = _merge_top_level_dicts(
      existing_actions.model_dump(exclude_none=True, by_alias=True),
      new_actions.model_dump(exclude_none=True, by_alias=True),
  )
  return EventActions.model_validate(merged_actions_data)


@a2a_experimental
def convert_a2a_task_to_event(
    a2a_task: Task,
    author: Optional[str] = None,
    invocation_context: Optional[InvocationContext] = None,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> Optional[Event]:
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
    event_actions = EventActions()
    output_parts = []
    long_running_function_ids = set()
    if a2a_task.artifacts:
      artifact_parts = [
          part for artifact in a2a_task.artifacts for part in artifact.parts
      ]
      for artifact in a2a_task.artifacts:
        event_actions = _merge_event_actions(
            event_actions, _extract_event_actions(artifact.metadata)
        )
      output_parts, _ = _convert_a2a_parts_to_adk_parts(
          artifact_parts, part_converter
      )
    if (
        a2a_task.status.message
        and a2a_task.status.state == TaskState.input_required
    ):
      event_actions = _merge_event_actions(
          event_actions,
          _extract_event_actions(a2a_task.status.message.metadata),
      )
      parts, ids = _convert_a2a_parts_to_adk_parts(
          a2a_task.status.message.parts, part_converter
      )
      output_parts.extend(parts)
      long_running_function_ids.update(ids)

    return _create_event(
        output_parts,
        invocation_context,
        author,
        event_actions,
        long_running_function_ids,
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
) -> Optional[Event]:
  """Converts an A2A message to an ADK event.

  Args:
    a2a_message: The A2A message to convert. Must not be None.
    author: The author of the event. Defaults to "a2a agent" if not provided.
    invocation_context: The invocation context containing session information.
      If provided, the branch will be set from the context.
    part_converter: The function to convert A2A part to GenAI part.

  Returns:
    An ADK Event object with converted content and long-running function
    metadata.

  Raises:
    ValueError: If a2a_message is None.
    RuntimeError: If conversion of message parts fails.
  """
  if a2a_message is None:
    raise ValueError("A2A message cannot be None")

  try:
    output_parts, _ = _convert_a2a_parts_to_adk_parts(
        a2a_message.parts, part_converter
    )
    return _create_event(
        output_parts,
        invocation_context,
        author,
        _extract_event_actions(a2a_message.metadata),
    )

  except Exception as e:
    logger.error("Failed to convert A2A message to event: %s", e)
    raise RuntimeError(f"Failed to convert message: {e}") from e


@a2a_experimental
def convert_a2a_status_update_to_event(
    a2a_status_update: TaskStatusUpdateEvent,
    author: Optional[str] = None,
    invocation_context: Optional[InvocationContext] = None,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> Optional[Event]:
  """Converts an A2A task status update to an ADK event.

  Args:
    a2a_status_update: The A2A task status update to convert.
    author: The author of the event. Defaults to "a2a agent" if not provided.
    invocation_context: The invocation context containing session information.
    part_converter: The function to convert A2A part to GenAI part.

  Returns:
    An ADK Event object representing the converted status update.
  """
  if a2a_status_update is None:
    raise ValueError("A2A status update cannot be None")

  try:
    output_parts = []
    long_running_function_ids = set()
    event_actions = EventActions()
    if a2a_status_update.status.message:
      event_actions = _extract_event_actions(
          a2a_status_update.status.message.metadata
      )
      parts, ids = _convert_a2a_parts_to_adk_parts(
          a2a_status_update.status.message.parts, part_converter
      )
      output_parts.extend(parts)
      long_running_function_ids.update(ids)

    return _create_event(
        output_parts,
        invocation_context,
        author,
        event_actions,
        long_running_function_ids,
    )
  except Exception as e:
    logger.error("Failed to convert A2A status update to event: %s", e)
    raise RuntimeError(f"Failed to convert status update: {e}") from e


# TODO: Add support for non-ADK Artifact Updates.
@a2a_experimental
def convert_a2a_artifact_update_to_event(
    a2a_artifact_update: TaskArtifactUpdateEvent,
    author: Optional[str] = None,
    invocation_context: Optional[InvocationContext] = None,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> Optional[Event]:
  """Converts an A2A task artifact update to an ADK event.

  Args:
    a2a_artifact_update: The A2A task artifact update to convert.
    author: The author of the event. Defaults to "a2a agent" if not provided.
    invocation_context: The invocation context containing session information.
    part_converter: The function to convert A2A part to GenAI part.

  Returns:
    An ADK Event object representing the converted artifact update.
  """
  if a2a_artifact_update is None:
    raise ValueError("A2A artifact update cannot be None")

  try:
    output_parts, _ = _convert_a2a_parts_to_adk_parts(
        a2a_artifact_update.artifact.parts, part_converter
    )
    return _create_event(
        output_parts,
        invocation_context,
        author,
        _extract_event_actions(a2a_artifact_update.artifact.metadata),
        partial=not a2a_artifact_update.last_chunk,
    )
  except Exception as e:
    logger.error("Failed to convert A2A artifact update to event: %s", e)
    raise RuntimeError(f"Failed to convert artifact update: {e}") from e

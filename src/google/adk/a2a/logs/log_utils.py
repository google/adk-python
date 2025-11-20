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

"""Utility functions for structured A2A request and response logging."""

from __future__ import annotations

import json
import sys

try:
  from a2a.client import ClientEvent as A2AClientEvent
  from a2a.types import DataPart as A2ADataPart
  from a2a.types import Message as A2AMessage
  from a2a.types import Part as A2APart
  from a2a.types import Task as A2ATask
  from a2a.types import TextPart as A2ATextPart
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        "A2A requires Python 3.10 or above. Please upgrade your Python version."
    ) from e
  else:
    raise e


# Constants
_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"file": {"bytes"}}


def _is_a2a_task(obj) -> bool:
  """Check if an object is an A2A Task, with fallback for isinstance issues."""
  try:
    return isinstance(obj, A2ATask)
  except (TypeError, AttributeError):
    return type(obj).__name__ == "Task" and hasattr(obj, "status")


def _is_a2a_client_event(obj) -> bool:
  """Check if an object is an A2A Client Event (Task, UpdateEvent) tuple."""
  try:
    return isinstance(obj, tuple) and _is_a2a_task(obj[0])
  except (TypeError, AttributeError):
    return (
        hasattr(obj, "__getitem__") and len(obj) == 2 and _is_a2a_task(obj[0])
    )


def _is_a2a_message(obj) -> bool:
  """Check if an object is an A2A Message, with fallback for isinstance issues."""
  try:
    return isinstance(obj, A2AMessage)
  except (TypeError, AttributeError):
    return type(obj).__name__ == "Message" and hasattr(obj, "role")


def _is_a2a_text_part(obj) -> bool:
  """Check if an object is an A2A TextPart, with fallback for isinstance issues."""
  try:
    return isinstance(obj, A2ATextPart)
  except (TypeError, AttributeError):
    return type(obj).__name__ == "TextPart" and hasattr(obj, "text")


def _is_a2a_data_part(obj) -> bool:
  """Check if an object is an A2A DataPart, with fallback for isinstance issues."""
  try:
    return isinstance(obj, A2ADataPart)
  except (TypeError, AttributeError):
    return type(obj).__name__ == "DataPart" and hasattr(obj, "data")


def build_message_part_log(part: A2APart) -> str:
  """Builds a log representation of an A2A message part.

  Args:
    part: The A2A message part to log.

  Returns:
    A string representation of the part.
  """
  part_content = ""
  if _is_a2a_text_part(part.root):
    part_content = f"TextPart: {part.root.text[:100]}" + (
        "..." if len(part.root.text) > 100 else ""
    )
  elif _is_a2a_data_part(part.root):
    # For data parts, show the data keys but exclude large values
    data_summary = {
        k: (
            f"<{type(v).__name__}>"
            if isinstance(v, (dict, list)) and len(str(v)) > 100
            else v
        )
        for k, v in part.root.data.items()
    }
    part_content = f"DataPart: {json.dumps(data_summary, indent=2)}"
  else:
    part_content = (
        f"{type(part.root).__name__}:"
        f" {part.model_dump_json(exclude_none=True, exclude=_EXCLUDED_PART_FIELD)}"
    )

  # Add part metadata if it exists
  if hasattr(part.root, "metadata") and part.root.metadata:
    metadata_str = json.dumps(part.root.metadata, indent=2).replace(
        "\n", "\n    "
    )
    part_content += f"\n    Part Metadata: {metadata_str}"

  return part_content


def build_a2a_request_log(req: A2AMessage) -> str:
  """Builds a structured log representation of an A2A request.

  Args:
    req: The A2A SendMessageRequest to log.

  Returns:
    A formatted string representation of the request.
  """
  # Message parts logs
  message_parts_logs = []
  if req.parts:
    for i, part in enumerate(req.parts):
      part_log = build_message_part_log(part)
      # Replace any internal newlines with indented newlines to maintain formatting
      part_log_formatted = part_log.replace("\n", "\n  ")
      message_parts_logs.append(f"Part {i}: {part_log_formatted}")

  # Build message metadata section
  message_metadata_section = ""
  if req.metadata:
    message_metadata_section = f"""
  Metadata:
  {json.dumps(req.metadata, indent=2).replace(chr(10), chr(10) + '  ')}"""

  # Build optional sections
  optional_sections = []

  if req.metadata:
    optional_sections.append(
        f"""-----------------------------------------------------------
Metadata:
{json.dumps(req.metadata, indent=2)}"""
    )

  optional_sections_str = _NEW_LINE.join(optional_sections)

  return f"""
A2A Send Message Request:
-----------------------------------------------------------
Message:
  ID: {req.message_id}
  Role: {req.role}
  Task ID: {req.task_id}
  Context ID: {req.context_id}{message_metadata_section}
-----------------------------------------------------------
Message Parts:
{_NEW_LINE.join(message_parts_logs) if message_parts_logs else "No parts"}
-----------------------------------------------------------
{optional_sections_str}
-----------------------------------------------------------
"""


def build_a2a_response_log(resp: A2AClientEvent | A2AMessage) -> str:
  """Builds a structured log representation of an A2A response.

  Args:
    resp: The A2A SendMessage Response to log.

  Returns:
    A formatted string representation of the response.
  """

  # Handle success responses
  result = resp
  result_type = type(result).__name__
  if result_type == "tuple":
    result_type = "ClientEvent"

  # Build result details based on type
  result_details = []

  if _is_a2a_client_event(result):
    result = result[0]
    result_details.extend([
        f"Task ID: {result.id}",
        f"Context ID: {result.context_id}",
        f"Status State: {result.status.state}",
        f"Status Timestamp: {result.status.timestamp}",
        f"History Length: {len(result.history) if result.history else 0}",
        f"Artifacts Count: {len(result.artifacts) if result.artifacts else 0}",
    ])

    # Add task metadata if it exists
    if result.metadata:
      result_details.append("Task Metadata:")
      metadata_formatted = json.dumps(result.metadata, indent=2).replace(
          "\n", "\n  "
      )
      result_details.append(f"  {metadata_formatted}")

  elif _is_a2a_message(result):
    result_details.extend([
        f"Message ID: {result.message_id}",
        f"Role: {result.role}",
        f"Task ID: {result.task_id}",
        f"Context ID: {result.context_id}",
    ])

    # Add message parts
    if result.parts:
      result_details.append("Message Parts:")
      for i, part in enumerate(result.parts):
        part_log = build_message_part_log(part)
        # Replace any internal newlines with indented newlines to maintain formatting
        part_log_formatted = part_log.replace("\n", "\n    ")
        result_details.append(f"  Part {i}: {part_log_formatted}")

    # Add metadata if it exists
    if result.metadata:
      result_details.append("Metadata:")
      metadata_formatted = json.dumps(result.metadata, indent=2).replace(
          "\n", "\n  "
      )
      result_details.append(f"  {metadata_formatted}")

  else:
    # Handle other result types by showing their JSON representation
    if hasattr(result, "model_dump_json"):
      try:
        result_json = result.model_dump_json()
        result_details.append(f"JSON Data: {result_json}")
      except Exception:
        result_details.append("JSON Data: <unable to serialize>")

  # Build status message section
  status_message_section = "None"
  if _is_a2a_task(result) and result.status.message:
    status_parts_logs = []
    if result.status.message.parts:
      for i, part in enumerate(result.status.message.parts):
        part_log = build_message_part_log(part)
        # Replace any internal newlines with indented newlines to maintain formatting
        part_log_formatted = part_log.replace("\n", "\n  ")
        status_parts_logs.append(f"Part {i}: {part_log_formatted}")

    # Build status message metadata section
    status_metadata_section = ""
    if result.status.message.metadata:
      status_metadata_section = f"""
Metadata:
{json.dumps(result.status.message.metadata, indent=2)}"""

    status_message_section = f"""ID: {result.status.message.message_id}
Role: {result.status.message.role}
Task ID: {result.status.message.task_id}
Context ID: {result.status.message.context_id}
Message Parts:
{_NEW_LINE.join(status_parts_logs) if status_parts_logs else "No parts"}{status_metadata_section}"""

  # Build history section
  history_section = "No history"
  if _is_a2a_task(result) and result.history:
    history_logs = []
    for i, message in enumerate(result.history):
      message_parts_logs = []
      if message.parts:
        for j, part in enumerate(message.parts):
          part_log = build_message_part_log(part)
          # Replace any internal newlines with indented newlines to maintain formatting
          part_log_formatted = part_log.replace("\n", "\n    ")
          message_parts_logs.append(f"  Part {j}: {part_log_formatted}")

      # Build message metadata section
      message_metadata_section = ""
      if message.metadata:
        message_metadata_section = f"""
  Metadata:
  {json.dumps(message.metadata, indent=2).replace(chr(10), chr(10) + '  ')}"""

      history_logs.append(
          f"""Message {i + 1}:
  ID: {message.message_id}
  Role: {message.role}
  Task ID: {message.task_id}
  Context ID: {message.context_id}
  Message Parts:
{_NEW_LINE.join(message_parts_logs) if message_parts_logs else "  No parts"}{message_metadata_section}"""
      )

    history_section = _NEW_LINE.join(history_logs)

  return f"""
A2A Response:
-----------------------------------------------------------
Type: SUCCESS
Result Type: {result_type}
-----------------------------------------------------------
Result Details:
{_NEW_LINE.join(result_details)}
-----------------------------------------------------------
Status Message:
{status_message_section}
-----------------------------------------------------------
History:
{history_section}
-----------------------------------------------------------
"""

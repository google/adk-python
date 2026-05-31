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

"""Utility functions for structured A2A request and response logging."""

from __future__ import annotations

import json
import sys

try:
  from a2a.types import Message as A2AMessage
  from a2a.types import Part as A2APart
  from a2a.types import StreamResponse as A2AStreamResponse
  from google.protobuf import json_format
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        "A2A requires Python 3.10 or above. Please upgrade your Python version."
    ) from e
  else:
    raise e


# Constants
_NEW_LINE = "\n"


def _proto_metadata_to_dict(metadata) -> dict:
  """Convert proto Struct metadata to a plain Python dict."""
  try:
    return dict(metadata)
  except Exception:
    return {}


def build_message_part_log(part: A2APart) -> str:
  """Builds a log representation of an A2A message part.

  Args:
    part: The A2A message part to log.

  Returns:
    A string representation of the part.
  """
  part_content = ""
  try:
    content_type = part.WhichOneof("content")
    if content_type == "text":
      text = part.text
      part_content = f"TextPart: {text[:100]}" + ("..." if len(text) > 100 else "")
    elif content_type == "data":
      try:
        data_dict = json_format.MessageToDict(part).get("data", {})
        data_summary = {
            k: (
                f"<{type(v).__name__}>"
                if isinstance(v, (dict, list)) and len(str(v)) > 100
                else v
            )
            for k, v in data_dict.items()
        }
        part_content = f"DataPart: {json.dumps(data_summary, indent=2)}"
      except Exception:
        part_content = "DataPart: <unable to serialize>"
    elif content_type == "url":
      part_content = f"UrlPart: {part.url}"
    elif content_type == "raw":
      part_content = f"RawPart: <{len(part.raw)} bytes, media_type={part.media_type}>"
    else:
      # Unknown/empty content
      try:
        part_content = f"Part: {json_format.MessageToJson(part)}"
      except Exception:
        part_content = "Part: <unable to serialize>"
  except AttributeError:
    # Fallback for Mock objects in tests
    if hasattr(part, "root"):
      root = part.root
      part_content = f"{type(root).__name__}: {getattr(root, 'text', str(root))}"
    else:
      try:
        part_content = f"{type(part).__name__}: {part.model_dump_json(exclude_none=True)}"
      except Exception:
        part_content = f"{type(part).__name__}: <unable to serialize>"

  # Add part metadata if it exists
  metadata_dict = {}
  try:
    if part.metadata:
      metadata_dict = _proto_metadata_to_dict(part.metadata)
  except AttributeError:
    # Mock object fallback
    if hasattr(part, "root") and hasattr(part.root, "metadata") and part.root.metadata:
      metadata_dict = dict(part.root.metadata) if isinstance(part.root.metadata, dict) else {}

  if metadata_dict:
    try:
      metadata_str = json.dumps(metadata_dict, indent=2, default=str).replace("\n", "\n    ")
      part_content += f"\n    Part Metadata: {metadata_str}"
    except Exception:
      pass

  return part_content


def build_a2a_request_log(req: A2AMessage) -> str:
  """Builds a structured log representation of an A2A request.

  Args:
    req: The A2A Message request to log.

  Returns:
    A formatted string representation of the request.
  """
  # Message parts logs
  message_parts_logs = []
  try:
    parts = req.parts
    if parts:
      for i, part in enumerate(parts):
        part_log = build_message_part_log(part)
        part_log_formatted = part_log.replace("\n", "\n  ")
        message_parts_logs.append(f"Part {i}: {part_log_formatted}")
  except Exception:
    pass

  # Build message metadata section
  message_metadata_section = ""
  try:
    if req.metadata:
      meta_dict = _proto_metadata_to_dict(req.metadata)
      if meta_dict:
        message_metadata_section = f"\n  Metadata:\n  {json.dumps(meta_dict, indent=2, default=str).replace(chr(10), chr(10) + '  ')}"  # pylint: disable=line-too-long
  except Exception:
    pass

  # Optional sections
  optional_sections = []
  try:
    if req.metadata:
      meta_dict = _proto_metadata_to_dict(req.metadata)
      if meta_dict:
        optional_sections.append(
            f"-----------------------------------------------------------\nMetadata:\n{json.dumps(meta_dict, indent=2, default=str)}"
        )
  except Exception:
    pass

  optional_sections_str = _NEW_LINE.join(optional_sections)

  try:
    msg_id = req.message_id
    role = req.role
    task_id = getattr(req, "task_id", "")
    context_id = getattr(req, "context_id", "")
  except Exception:
    msg_id = role = task_id = context_id = "<unknown>"

  return f"""
A2A Send Message Request:
-----------------------------------------------------------
Message:
  ID: {msg_id}
  Role: {role}
  Task ID: {task_id}
  Context ID: {context_id}{message_metadata_section}
-----------------------------------------------------------
Message Parts:
{_NEW_LINE.join(message_parts_logs) if message_parts_logs else "No parts"}
-----------------------------------------------------------
{optional_sections_str}
-----------------------------------------------------------
"""


def build_a2a_response_log(resp) -> str:
  """Builds a structured log representation of an A2A response.

  Args:
    resp: The A2A StreamResponse or Message response to log.

  Returns:
    A formatted string representation of the response.
  """
  result_type = type(resp).__name__
  result_details = []

  # Handle tuple (legacy ClientEvent pattern) for backward compat
  if isinstance(resp, tuple):
    result_type = "ClientEvent"
    try:
      task = resp[0]
      if task:
        result_details.extend([
            f"Task ID: {task.id}",
            f"Context ID: {task.context_id}",
            f"Status State: {task.status.state}",
        ])
    except Exception:
      pass

  # Handle StreamResponse proto (check isinstance to avoid matching other
  # proto messages like A2AMessage which also have WhichOneof)
  elif isinstance(resp, A2AStreamResponse):
    try:
      payload_type = resp.WhichOneof("payload")
      result_type = f"StreamResponse({payload_type})"
      if payload_type == "task":
        task = resp.task
        result_details.extend([
            f"Task ID: {task.id}",
            f"Context ID: {task.context_id}",
            f"Status State: {task.status.state}",
        ])
      elif payload_type == "message":
        msg = resp.message
        result_details.extend([
            f"Message ID: {msg.message_id}",
            f"Role: {msg.role}",
        ])
      elif payload_type == "status_update":
        su = resp.status_update
        result_details.append(f"Task ID: {su.task_id}")
        result_details.append(f"State: {su.status.state}")
      elif payload_type == "artifact_update":
        au = resp.artifact_update
        result_details.append(f"Task ID: {au.task_id}")
        result_details.append(f"Artifact ID: {au.artifact.artifact_id}")
    except Exception:
      pass

  # Handle A2AMessage
  elif _is_a2a_message(resp):
    try:
      result_details.extend([
          f"Message ID: {resp.message_id}",
          f"Role: {resp.role}",
          f"Task ID: {getattr(resp, 'task_id', '')}",  # pylint: disable=line-too-long
          f"Context ID: {getattr(resp, 'context_id', '')}",
      ])
      if resp.parts:
        result_details.append("Message Parts:")
        for i, part in enumerate(resp.parts):
          part_log = build_message_part_log(part).replace("\n", "\n    ")
          result_details.append(f"  Part {i}: {part_log}")
    except Exception:
      pass

  else:
    # Generic fallback
    if hasattr(resp, "model_dump_json"):
      try:
        result_details.append(f"JSON Data: {resp.model_dump_json()}")
      except Exception:
        pass

  # Build status message section
  status_message_section = "None"
  try:
    if isinstance(resp, tuple) and resp[0] and resp[0].status and resp[0].status.message:
      msg = resp[0].status.message
      status_parts_logs = []
      if msg.parts:
        for i, part in enumerate(msg.parts):
          part_log = build_message_part_log(part).replace("\n", "\n  ")
          status_parts_logs.append(f"Part {i}: {part_log}")
      status_message_section = f"""ID: {msg.message_id}
Role: {msg.role}
Task ID: {getattr(msg, 'task_id', '')}
Context ID: {getattr(msg, 'context_id', '')}
Message Parts:
{_NEW_LINE.join(status_parts_logs) if status_parts_logs else "No parts"}"""
  except Exception:
    pass

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
-----------------------------------------------------------
"""


def _is_a2a_message(obj) -> bool:
  """Check if an object is an A2A Message."""
  try:
    return isinstance(obj, A2AMessage)
  except (TypeError, AttributeError):
    return type(obj).__name__ == "Message" and hasattr(obj, "role")

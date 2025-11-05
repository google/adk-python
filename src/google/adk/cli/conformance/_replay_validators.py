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

"""Validation logic for conformance test replay mode."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
from typing import Optional

from ...events.event import Event
from ...sessions.session import Session


@dataclass
class ComparisonResult:
  """Result of comparing two objects during conformance testing."""

  success: bool
  error_message: Optional[str] = None


def _generate_mismatch_message(
    context: str, actual_value: str, recorded_value: str
) -> str:
  """Generate a generic mismatch error message."""
  return (
      f"{context} mismatch - \nActual: \n{actual_value} \nRecorded:"
      f" \n{recorded_value}"
  )


def _generate_diff_message(
    context: str, actual_dict: dict, recorded_dict: dict
) -> str:
  """Generate a diff-based error message for comparison failures."""
  # Convert to pretty-printed JSON for better readability
  actual_json = json.dumps(actual_dict, indent=2, sort_keys=True)
  recorded_json = json.dumps(recorded_dict, indent=2, sort_keys=True)

  # Generate unified diff
  diff_lines = list(
      difflib.unified_diff(
          recorded_json.splitlines(keepends=True),
          actual_json.splitlines(keepends=True),
          fromfile=f"recorded {context}\n",
          tofile=f"actual {context}\n",
          lineterm="",
      )
  )

  if diff_lines:
    return f"{context} mismatch:\n" + "".join(diff_lines)
  else:
    # Fallback to generic format if diff doesn't work
    return _generate_mismatch_message(context, actual_json, recorded_json)


def compare_event(
    actual_event: Event, recorded_event: Event, index: int
) -> ComparisonResult:
  """Compare a single actual event with a recorded event."""
  # Comprehensive exclude dict for all fields that can differ between runs
  excluded_fields = {
      # Event-level fields that vary per run
      "id": True,
      "timestamp": True,
      "invocation_id": True,
      "long_running_tool_ids": True,
      # Content fields that vary per run
      "content": {
          "parts": {
              "__all__": {
                  "thought_signature": True,
                  "function_call": {"id": True},
                  "function_response": {"id": True},
              }
          }
      },
      # Action fields that vary per run
      "actions": {
          "state_delta": {
              "_adk_recordings_config": True,
              "_adk_replay_config": True,
          },
          "requested_auth_configs": True,
          "requested_tool_confirmations": True,
      },
  }

  # Compare events using model dumps with comprehensive exclude dict
  actual_dict = actual_event.model_dump(
      exclude_none=True, exclude=excluded_fields
  )
  recorded_dict = recorded_event.model_dump(
      exclude_none=True, exclude=excluded_fields
  )

  if actual_dict != recorded_dict:
    return ComparisonResult(
        success=False,
        error_message=_generate_diff_message(
            f"event {index}", actual_dict, recorded_dict
        ),
    )

  return ComparisonResult(success=True)


def compare_events(
    actual_events: list[Event], recorded_events: list[Event]
) -> ComparisonResult:
  """Compare actual events with recorded events."""
  if len(actual_events) != len(recorded_events):
    return ComparisonResult(
        success=False,
        error_message=_generate_mismatch_message(
            "Event count", str(len(actual_events)), str(len(recorded_events))
        ),
    )

  for i, (actual, recorded) in enumerate(zip(actual_events, recorded_events)):
    result = compare_event(actual, recorded, i)
    if not result.success:
      return result

  return ComparisonResult(success=True)


def compare_session(
    actual_session: Session, recorded_session: Session
) -> ComparisonResult:
  """Compare actual session with recorded session using comprehensive exclude list.

  Returns:
    ComparisonResult with success status and optional error message
  """
  # Comprehensive exclude dict for all fields that can differ between runs
  excluded_fields = {
      # Session-level fields that vary per run
      "id": True,
      "last_update_time": True,
      # State fields that contain ADK internal configuration
      "state": {
          "_adk_recordings_config": True,
          "_adk_replay_config": True,
      },
      # Events comparison handled separately
      "events": True,
  }

  # Compare sessions using model dumps with comprehensive exclude dict
  actual_dict = actual_session.model_dump(
      exclude_none=True, exclude=excluded_fields
  )
  recorded_dict = recorded_session.model_dump(
      exclude_none=True, exclude=excluded_fields
  )

  if actual_dict != recorded_dict:
    return ComparisonResult(
        success=False,
        error_message=_generate_diff_message(
            "session", actual_dict, recorded_dict
        ),
    )

  return ComparisonResult(success=True)

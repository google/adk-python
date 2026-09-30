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
"""Utility functions for handling artifact URIs."""

from __future__ import annotations

import re
from typing import NamedTuple

from google.genai import types

from ..errors import input_validation_error


class ParsedArtifactUri(NamedTuple):
  """The result of parsing an artifact URI."""

  app_name: str
  user_id: str
  session_id: str | None
  filename: str
  version: int


# The maximum number of `artifact://` references a single lookup may follow
# before giving up. Shared by every artifact service so that a self-referential
# or cyclic reference chain fails with a clear error instead of exhausting the
# Python stack.
_MAX_ARTIFACT_REFERENCE_DEPTH = 5

_WINDOWS_DRIVE_RE = re.compile(r"[A-Za-z]:")

_RESERVED_PATH_SEGMENTS = frozenset({
    "apps",
    "users",
    "sessions",
    "artifacts",
    "versions",
})
_RESERVED_SEGMENTS_LOOKAHEAD = (
    rf"(?!/(?:{'|'.join(sorted(_RESERVED_PATH_SEGMENTS))})/)"
)
_PATH_SEGMENT_PATTERN = rf"(?:{_RESERVED_SEGMENTS_LOOKAHEAD}.)+?"

_SESSION_SCOPED_ARTIFACT_URI_RE = re.compile(
    rf"artifact://apps/({_PATH_SEGMENT_PATTERN})/users/({_PATH_SEGMENT_PATTERN})/sessions/({_PATH_SEGMENT_PATTERN})/artifacts/(.+)/versions/(\d+)"
)
_USER_SCOPED_ARTIFACT_URI_RE = re.compile(
    rf"artifact://apps/({_PATH_SEGMENT_PATTERN})/users/({_PATH_SEGMENT_PATTERN})/artifacts/(.+)/versions/(\d+)"
)


def parse_artifact_uri(uri: str) -> ParsedArtifactUri | None:
  """Parses an artifact URI.

  Args:
      uri: The artifact URI to parse.

  Returns:
      A ParsedArtifactUri if parsing is successful, None otherwise.
  """
  if not uri or not uri.startswith("artifact://"):
    return None

  match = _SESSION_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=match.group(3),
        filename=match.group(4),
        version=int(match.group(5)),
    )

  match = _USER_SCOPED_ARTIFACT_URI_RE.fullmatch(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=None,
        filename=match.group(3),
        version=int(match.group(4)),
    )

  return None


def get_artifact_uri(
    app_name: str,
    user_id: str,
    filename: str,
    version: int,
    session_id: str | None = None,
) -> str:
  """Constructs an artifact URI.

  Args:
      app_name: The name of the application.
      user_id: The ID of the user.
      filename: The name of the artifact file.
      version: The version of the artifact.
      session_id: The ID of the session.

  Returns:
      The constructed artifact URI.
  """
  if session_id:
    return f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
  else:
    return f"artifact://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"


def is_artifact_ref(artifact: types.Part) -> bool:
  """Checks if an artifact part is an artifact reference.

  Args:
      artifact: The artifact part to check.

  Returns:
      True if the artifact part is an artifact reference, False otherwise.
  """
  return bool(
      artifact.file_data
      and artifact.file_data.file_uri
      and artifact.file_data.file_uri.startswith("artifact://")
  )


def validate_artifact_reference_scope(
    *,
    app_name: str,
    user_id: str,
    session_id: str | None,
    parsed_uri: ParsedArtifactUri,
) -> None:
  """Ensures artifact references cannot escape the caller's scope."""
  if parsed_uri.app_name != app_name or parsed_uri.user_id != user_id:
    raise input_validation_error.InputValidationError(
        "Artifact references must stay within the same app and user scope."
    )
  if parsed_uri.session_id is not None and parsed_uri.session_id != session_id:
    raise input_validation_error.InputValidationError(
        "Session-scoped artifact references must stay within the same"
        " session scope."
    )


def resolve_artifact_reference(
    *,
    file_uri: str,
    app_name: str,
    user_id: str,
    session_id: str | None,
    remaining_depth: int,
) -> ParsedArtifactUri:
  """Validates an artifact reference URI before following it.

  An artifact whose payload is an `artifact://` URI points at another artifact,
  which may itself be another reference. Nothing prevents a chain from pointing
  back at one of its own links, so every caller that follows a reference must
  bound how many hops it takes; otherwise a self-referential or cyclic chain
  recurses until the interpreter raises `RecursionError`.

  Args:
      file_uri: The `artifact://` URI held by the artifact being resolved.
      app_name: The app name of the caller's scope.
      user_id: The user id of the caller's scope.
      session_id: The session id of the caller's scope, if any.
      remaining_depth: How many more references the caller may follow. Callers
        start at `_MAX_ARTIFACT_REFERENCE_DEPTH` and pass `remaining_depth - 1`
        when recursing.

  Returns:
      The parsed target of the reference.

  Raises:
      InputValidationError: If the reference chain is longer than
        `_MAX_ARTIFACT_REFERENCE_DEPTH`, the URI is not a valid artifact
        reference, or the target lies outside the caller's scope.
  """
  if remaining_depth <= 0:
    raise input_validation_error.InputValidationError(
        "Exceeded maximum recursion depth resolving artifact reference:"
        f" {file_uri}"
    )
  parsed_uri = parse_artifact_uri(file_uri)
  if not parsed_uri:
    raise input_validation_error.InputValidationError(
        f"Invalid artifact reference URI: {file_uri}"
    )
  validate_artifact_reference_scope(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      parsed_uri=parsed_uri,
  )
  return parsed_uri


def _is_drive_qualified(value: str) -> bool:
  """Checks whether a value starts with a Windows drive letter such as ``C:``."""
  return _WINDOWS_DRIVE_RE.match(value) is not None


def _validate_session_id_for_flat_storage(session_id: str) -> None:
  """Validates a session_id used by flat storage artifact backends.

  In addition to the checks in `validate_path_segment`, rejects values whose
  first path segment is the reserved value "user". Backends that lay out
  session-scoped and user-scoped artifacts in the same flat namespace
  (in-memory, GCS) use that exact string as a reserved segment marking
  user-scoped artifacts, so a session starting with "user" would silently write
  into -- and read out of -- that reserved namespace instead of its own.

  Args:
    session_id: The caller-supplied session id.

  Raises:
    InputValidationError: If `session_id` fails `validate_path_segment`, or has
      the reserved value "user" as its first path segment.
  """
  validate_path_segment(session_id, "session_id")
  if session_id.replace("\\", "/").split("/")[0] == "user":
    raise input_validation_error.InputValidationError(
        "session_id must not be or start with the reserved value 'user'."
    )


def validate_path_segment(value: str, field_name: str) -> None:
  """Rejects values that could alter the constructed path.

  Args:
    value: The caller-supplied identifier (e.g. user_id or session_id).
    field_name: Human-readable name used in the error message.

  Raises:
    InputValidationError: If the value contains traversal segments, null bytes,
      is an absolute path / starts with a slash, is drive-qualified, or contains
      slashes along with reserved path segments.
  """
  if not value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not be empty."
    )
  if "\x00" in value:
    raise input_validation_error.InputValidationError(
        f"{field_name} must not contain null bytes."
    )
  if isinstance(value, str) and (
      value.startswith("/") or value.startswith("\\")
  ):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not be an absolute path or start with a"
        " slash."
    )
  if isinstance(value, str) and _is_drive_qualified(value):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not be drive-qualified."
    )
  if value in (".", "..") or ".." in value.replace("\\", "/").split("/"):
    raise input_validation_error.InputValidationError(
        f"{field_name} {value!r} must not contain traversal segments."
    )
  if isinstance(value, str) and ("/" in value or "\\" in value):
    segments = {
        segment.casefold() for segment in value.replace("\\", "/").split("/")
    }
    if not _RESERVED_PATH_SEGMENTS.isdisjoint(segments):
      raise input_validation_error.InputValidationError(
          f"{field_name} {value!r} must not contain reserved path segments."
      )

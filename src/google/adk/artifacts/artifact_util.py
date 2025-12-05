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
"""Utility functions for handling artifact URIs."""

from __future__ import annotations

import re
from typing import NamedTuple
from typing import Optional

from google.genai import types


class ParsedArtifactUri(NamedTuple):
  """The result of parsing an artifact URI."""

  app_name: str
  user_id: str
  session_id: Optional[str]
  filename: str
  version: int


_SESSION_SCOPED_ARTIFACT_URI_RE = re.compile(
    r"artifact://apps/([^/]+)/users/([^/]+)/sessions/([^/]+)/artifacts/([^/]+)/versions/(\d+)"
)
_USER_SCOPED_ARTIFACT_URI_RE = re.compile(
    r"artifact://apps/([^/]+)/users/([^/]+)/artifacts/([^/]+)/versions/(\d+)"
)


def parse_artifact_uri(uri: str) -> Optional[ParsedArtifactUri]:
  """Parses an artifact URI.

  Args:
      uri: The artifact URI to parse.

  Returns:
      A ParsedArtifactUri if parsing is successful, None otherwise.
  """
  if not uri or not uri.startswith("artifact://"):
    return None

  match = _SESSION_SCOPED_ARTIFACT_URI_RE.match(uri)
  if match:
    return ParsedArtifactUri(
        app_name=match.group(1),
        user_id=match.group(2),
        session_id=match.group(3),
        filename=match.group(4),
        version=int(match.group(5)),
    )

  match = _USER_SCOPED_ARTIFACT_URI_RE.match(uri)
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
    session_id: Optional[str] = None,
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

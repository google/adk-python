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

"""Helpers for normalizing file path strings produced by the model."""

from __future__ import annotations

import re

_SEGMENT_SPLIT_PATTERN = re.compile(r"([/\\])")
_BOUNDARY_CHARS = " \t\r\n'\"`"


def sanitize_generated_file_path(file_path: str) -> str:
  """Strip stray quotes/whitespace around each path segment.

  The agent occasionally emits quoted paths such as `'tools/web.yaml'` which
  would otherwise create directories literally named `'<name>`. This helper
  removes leading/trailing whitespace and quote-like characters from the path
  and from each path component while preserving intentional interior
  characters.

  Args:
    file_path: Path string provided by the model or user.

  Returns:
    Sanitized path string safe to feed into pathlib.Path.
  """
  if not isinstance(file_path, str):
    file_path = str(file_path)

  trimmed = file_path.strip()
  if not trimmed:
    return trimmed

  segments = _SEGMENT_SPLIT_PATTERN.split(trimmed)
  sanitized_segments: list[str] = []

  for segment in segments:
    if not segment:
      sanitized_segments.append(segment)
      continue
    if segment in ("/", "\\"):
      sanitized_segments.append(segment)
      continue
    sanitized_segments.append(segment.strip(_BOUNDARY_CHARS))

  sanitized = "".join(sanitized_segments).strip(_BOUNDARY_CHARS)
  return sanitized or trimmed

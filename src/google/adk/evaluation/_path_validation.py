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


def validate_path_segment(value: str, field_name: str) -> None:
  """Rejects values that could alter a filesystem path.

  Args:
    value: The caller-supplied identifier.
    field_name: Human-readable field name used in error messages.

  Raises:
    ValueError: If the value contains path separators, traversal segments, or
      null bytes.
  """
  if not value:
    raise ValueError(f"{field_name} must not be empty.")
  if "\x00" in value:
    raise ValueError(f"{field_name} must not contain null bytes.")
  if "/" in value or "\\" in value:
    raise ValueError(
        f"{field_name} {value!r} must not contain path separators."
    )
  if value in (".", ".."):
    raise ValueError(
        f"{field_name} {value!r} must not contain traversal segments."
    )

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

ADK_METADATA_KEY_PREFIX = "adk_"


def _get_adk_metadata_key(key: str) -> str:
  """Gets the A2A event metadata key for the given key.

  Args:
    key: The metadata key to prefix.

  Returns:
    The prefixed metadata key.

  Raises:
    ValueError: If key is empty or None.
  """
  if not key:
    raise ValueError("Metadata key cannot be empty or None")
  return f"{ADK_METADATA_KEY_PREFIX}{key}"

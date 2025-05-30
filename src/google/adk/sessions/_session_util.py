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
"""Utility functions for session service."""
from __future__ import annotations

from typing import Any
from typing import Optional

from google.genai import types


def decode_content(
    content: Optional[dict[str, Any]],
) -> Optional[types.Content]:
  """Decodes a content object from a JSON dictionary."""
  if not content:
    return None
  return types.Content.model_validate(content)


def decode_grounding_metadata(
    grounding_metadata: Optional[dict[str, Any]],
) -> Optional[types.GroundingMetadata]:
  """Decodes a grounding metadata object from a JSON dictionary."""
  if not grounding_metadata:
    return None
  return types.GroundingMetadata.model_validate(grounding_metadata)

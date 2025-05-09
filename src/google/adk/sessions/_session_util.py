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

import base64
from typing import Any, Optional

from google.genai import types


def encode_content(content: types.Content):
  """Encodes a content object to a JSON dictionary."""
  encoded_content = content.model_dump(exclude_none=True)
  for p in encoded_content["parts"]:
    if "inline_data" in p:
      p["inline_data"]["data"] = base64.b64encode(
          p["inline_data"]["data"]
      ).decode("utf-8")
  return encoded_content


def decode_content(
    content: Optional[dict[str, Any]],
) -> Optional[types.Content]:
  """Decodes a content object from a JSON dictionary."""
  if not content:
    return None
  for p in content["parts"]:
    if "inline_data" in p:
      p["inline_data"]["data"] = base64.b64decode(p["inline_data"]["data"])
  return types.Content.model_validate(content)

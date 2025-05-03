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

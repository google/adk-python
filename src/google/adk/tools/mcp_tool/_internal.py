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

"""Internal utilities for MCP tools.

This module contains internal validation and sanitization utilities
that are not part of the public API and follow RFC 7230 properly.

**Security Notes:**

- Header validation implements RFC 7230 §3.2 for proper HTTP header format
- All ASCII control characters (0x00-0x1F) and DEL (0x7F) are removed from
  header values to prevent injection
- All functions log security-relevant warnings when appropriate

**RFC 7230 Compliance:**

- Header names: only letters, digits, and hyphens allowed
- Header values: control characters including CRLF (0x00-0x1F, 0x7F) are
  removed to prevent injection

**Attack Prevention:**

- HTTP header injection attacks via control character filtering
- Response splitting attacks through CRLF handling
- Log injection attacks via character sanitization
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("google_adk." + __name__)

# RFC 7230 compliant header name pattern (allows letters, digits, hyphens)
_HEADER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9-]+\Z")

# Truly dangerous characters that should never appear in header values
# These are characters that can break HTTP parsing or cause injection
_DANGEROUS_CHARS = {
    "\x00",
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",
    "\x08",
    "\x09",
    "\x0a",
    "\x0b",
    "\x0c",
    "\x0d",
    "\x0e",
    "\x0f",
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    "\x7f",
}


def validate_header_name(header_name: str) -> None:
  """Validates that a header name conforms to RFC 7230.
  Only allows printable ASCII, no control chars, spaces, or separators.
  Rejects header names containing invalid characters.
  """
  if not header_name:
    raise ValueError("Header name cannot be empty.")

  if not _HEADER_NAME_PATTERN.match(header_name):
    raise ValueError(
        f'Header name "{header_name}" contains invalid characters. '
        "Header names must conform to RFC 7230 and cannot contain "
        'control characters, spaces, or separators like ():<>@,;:\\"/[]?={}.'
    )


def validate_header_format(header_format: str) -> None:
  """Validates that a header format string doesn't contain CRLF injection.

  This prevents header injection attacks where malicious format strings
  could inject additional headers via CRLF sequences.

  Args:
      header_format: The format string to validate.

  Raises:
      ValueError: If header_format contains CRLF sequences.
  """
  if "\r" in header_format or "\n" in header_format:
    raise ValueError(
        "Header format string cannot contain CRLF (carriage return or line"
        " feed) characters due to header injection risk. Invalid format:"
        f" {repr(header_format)}"
    )


def sanitize_header_value(value: Any) -> str:
  """Sanitizes a header value to prevent injection attacks.

  This is a wrapper that converts non-string values to strings and then
  applies core sanitization logic.

  Args:
      value: The header value to sanitize (any type).

  Returns:
      The sanitized header value as a string.
  """
  if not isinstance(value, str):
    value = str(value)

  # Remove CRLF and control characters to prevent header injection.
  # Header folding (obs-fold) was deprecated by RFC 7230 and obsoleted
  # by RFC 9110. CRLF in header values is the primary vector for
  # header injection and response splitting attacks.
  sanitized_chars = []
  for char in value:
    if char not in _DANGEROUS_CHARS:
      sanitized_chars.append(char)
    else:
      logger.warning(
          f"Removed dangerous character {repr(char)} from header value "
          "for security reasons"
      )

  return "".join(sanitized_chars)


def validate_header_value(
    state_key: str, value: Any, strict: bool = False
) -> None:
  """Validates that a state value is suitable for use in a header.

  Args:
      state_key: The key being validated.
      value: The value to validate.
      strict: If True, raises ValueError for non-primitive types.

  Raises:
      ValueError: If strict=True and value is not a primitive type.
  """
  if not isinstance(value, (str, int, float, bool)):
    msg = (
        f'Value for state key "{state_key}" is of type '
        f"{type(value).__name__}, which may not serialize correctly into a "
        "header. Consider pre-serializing complex values or using "
        "state_header_format."
    )
    if strict:
      raise ValueError(msg)
    else:
      logger.warning(msg)

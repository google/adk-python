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

"""Best-effort redaction of credential material from logs and telemetry.

Logs and traces are routinely shipped to sinks that are broader than the data
they describe. Tool arguments, tool results, and model payloads can carry API
keys, OAuth tokens, or passwords, so anything that renders them for a human or
persists them must strip those values first. This module centralizes that
policy for structured payloads; it is deliberately conservative and never
raises, because a logging path must not fail a request.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

REDACTED = "<redacted>"

_MAX_DEPTH = 32

# Keys whose values are credentials once punctuation and case are ignored.
# Exact names only: a substring match would redact benign fields such as
# `prompt_token_count` or `next_page_token`. Names in `_AMBIGUOUS_KEY_NAMES`
# are deliberately absent: a bare `token` or `apiKey` can be a pagination
# cursor or a public identifier in a tool response, and tool responses are
# recorded as-is so callers can debug them. Logging opts into them separately.
_SENSITIVE_KEY_NAMES = frozenset({
    "password",
    "passwd",
    "secret",
    "clientsecret",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "authtoken",
    "bearertoken",
    "sessiontoken",
    "authorization",
    "proxyauthorization",
    "cookie",
    "setcookie",
    "privatekey",
    "privatekeyid",
    "codeverifier",
    "authcode",
    "authresponseuri",
    "additionalheaders",
    "credentials",
})

_AMBIGUOUS_KEY_NAMES = frozenset({"token", "apikey"})

# Free-text secrets are common in headers and error strings: `Bearer <token>`
# and `api_key=...` shapes. Bound each match to a plausible token so prose is
# left alone.
_TEXT_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|client[_-]?secret|access[_-]?token|"
        r"refresh[_-]?token|password)\b\s*([:=])\s*['\"]?[^\s'\",;}]{4,}"
    ),
)


def _mask_text_match(match: re.Match[str]) -> str:
  """Replaces a matched secret, preserving the key (and separator) before it."""
  groups = match.groups()
  if len(groups) == 1:
    return f"{groups[0]} {REDACTED}"
  return f"{groups[0]}{groups[1]}{REDACTED}"


def _normalize_key(key: Any) -> str:
  return re.sub(r"[^a-z0-9]", "", str(key).lower())


def redact_secrets(
    value: Any,
    *,
    include_ambiguous: bool = False,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
  """Returns a copy of *value* with credential-bearing fields replaced.

  Dict keys are matched case- and punctuation-insensitively against a fixed
  allowlist of credential names. ``include_ambiguous`` also masks bare ``token``
  and ``apiKey`` keys, which a tool response may legitimately carry as a
  pagination cursor, so callers that persist tool output leave it off. Pydantic
  models are dumped before recursing, so secrets nested inside them are found
  too. Cycles and pathological nesting are cut off rather than raising.
  """
  if _depth > _MAX_DEPTH:
    return REDACTED

  if isinstance(value, BaseModel):
    try:
      value = value.model_dump(mode="python")
    except Exception:  # pylint: disable=broad-except
      return value

  sensitive_keys = (
      _SENSITIVE_KEY_NAMES | _AMBIGUOUS_KEY_NAMES
      if include_ambiguous
      else _SENSITIVE_KEY_NAMES
  )

  if isinstance(value, (dict, list, tuple, set, frozenset)):
    # Track only the current path: a cycle must reach `json.dumps` unchanged so
    # it can raise, while a shared non-cyclic child is still redacted per use.
    if _seen is None:
      _seen = set()
    if id(value) in _seen:
      return value
    _seen.add(id(value))
    try:
      if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if _normalize_key(key) in sensitive_keys
                else redact_secrets(
                    val,
                    include_ambiguous=include_ambiguous,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
            )
            for key, val in value.items()
        }
      if isinstance(value, (list, tuple)):
        redacted = [
            redact_secrets(
                item,
                include_ambiguous=include_ambiguous,
                _depth=_depth + 1,
                _seen=_seen,
            )
            for item in value
        ]
        return tuple(redacted) if isinstance(value, tuple) else redacted
      return {
          redact_secrets(
              item,
              include_ambiguous=include_ambiguous,
              _depth=_depth + 1,
              _seen=_seen,
          )
          for item in value
      }
    finally:
      _seen.discard(id(value))
  return value


def redact_text(text: str, *, max_length: int | None = None) -> str:
  """Masks credential-looking fragments in free-form text.

  Only well-known shapes are masked so that ordinary prose survives. A
  ``max_length`` truncates after masking, not before, so a cut cannot hide the
  beginning of a secret.
  """
  redacted = text
  for pattern in _TEXT_SECRET_PATTERNS:
    redacted = pattern.sub(_mask_text_match, redacted)
  if max_length is not None and len(redacted) > max_length:
    redacted = redacted[:max_length] + "..."
  return redacted

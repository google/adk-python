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
"""Restricted unpickler for safe deserialization of v0 EventActions data.

The v0 schema stored EventActions as pickle blobs. This module provides a
safe deserialization path that only allows known ADK and standard types,
blocking arbitrary code execution via crafted pickle payloads.

See: https://docs.python.org/3/library/pickle.html#restricting-globals
"""

from __future__ import annotations

import io
import logging
import os
import pickle
from typing import Any

logger = logging.getLogger("google_adk." + __name__)

_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "google.adk.",
    "google.genai.",
    "pydantic.",
    "pydantic_core.",
)

_ALLOWED_GLOBALS: dict[str, set[str]] = {
    "builtins": {
        "dict",
        "list",
        "set",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
        "True",
        "False",
        "None",
        "type",
        "object",
        "complex",
        "slice",
        "range",
        "int",
        "float",
        "str",
        "bool",
    },
    "collections": {"OrderedDict", "defaultdict"},
    "datetime": {"datetime", "date", "time", "timedelta", "timezone"},
    "copy_reg": {"_reconstructor"},
    "copyreg": {"_reconstructor", "__newobj__"},
    "_codecs": {"encode"},
    "enum": {"__new__", "Enum", "IntEnum", "StrEnum"},
}


class _RestrictedUnpickler(pickle.Unpickler):
  """Unpickler that only allows reconstruction of known-safe types."""

  def find_class(self, module: str, name: str) -> Any:
    for prefix in _ALLOWED_MODULE_PREFIXES:
      if module.startswith(prefix):
        return super().find_class(module, name)
    allowed_names = _ALLOWED_GLOBALS.get(module)
    if allowed_names and name in allowed_names:
      return super().find_class(module, name)
    raise pickle.UnpicklingError(
        f"Blocked unsafe pickle global: {module}.{name}. "
        "If this is a legitimate ADK type, please file an issue at "
        "https://github.com/google/adk-python/issues"
    )


def safe_loads(data: bytes) -> Any:
  """Deserialize pickle bytes using a restricted unpickler.

  If ADK_ALLOW_UNSAFE_V0_PICKLE=1 is set, falls back to unrestricted
  pickle.loads() for compatibility. A deprecation warning is logged.
  """
  if os.environ.get("ADK_ALLOW_UNSAFE_V0_PICKLE") == "1":
    logger.warning(
        "ADK_ALLOW_UNSAFE_V0_PICKLE is set - using unrestricted "
        "pickle.loads(). This is unsafe and will be removed in a "
        "future release. Migrate to the v1 JSON schema."
    )
    return pickle.loads(data)  # noqa: S301
  return _RestrictedUnpickler(io.BytesIO(data)).load()

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
"""Restricted pickle loading for the legacy v0 session schema.

The v0 session schema stored `EventActions` as a pickled blob. Bytes read back
from the database are untrusted input, so they are loaded through an unpickler
that only resolves the globals needed to reconstruct `EventActions`.

The allowed set is the union of two parts:

* A derived part, computed by walking the Pydantic field annotations of
  `EventActions` (and of `AuthConfig`, which `EventActions` types as `Any` at
  runtime) and admitting every reachable `BaseModel` and `Enum` subclass. Those
  are inert data types: unpickling one only calls `__setstate__` or the enum
  constructor. Deriving them keeps the set correct as the ADK and `google.genai`
  models gain fields, which a hand-written list does not.
* A static part, for globals the walk cannot see: primitives and `datetime`
  types, which are not Pydantic models, and model classes reachable only by
  subclassing rather than through an annotation.

Anything else - notably arbitrary callables that older versions allowed into
`state_delta` - is refused rather than resolved. See `_RestrictedUnpickler`.

This module exposes `dumps` and `loads`, so it can also be passed directly as
the `pickler` argument of SQLAlchemy's `PickleType`.
"""

from __future__ import annotations

from enum import Enum
import functools
import io
import logging
import pickle
from typing import Any
from typing import get_args

from pydantic import BaseModel

logger = logging.getLogger("google_adk." + __name__)

_STATIC_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset({
    # Builtin containers/primitives.
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "set"),
    ("builtins", "tuple"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("datetime", "datetime"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    # Auth models reachable only by subclassing or as a union base, so the
    # annotation walk below does not reach them.
    ("fastapi.openapi.models", "OAuthFlow"),
    ("fastapi.openapi.models", "SecurityBase"),
    ("fastapi.openapi.models", "SecurityScheme"),
    ("google.adk.auth.auth_schemes", "ExtendedOAuth2"),
    ("google.adk.auth.auth_schemes", "OAuthGrantType"),
})


def _walk_model_tree(roots: list[Any]) -> set[tuple[str, str]]:
  """Returns the `BaseModel`/`Enum` classes reachable from `roots`.

  Follows Pydantic field annotations and the type arguments of generics such as
  `Optional[...]`, `list[...]` and `dict[...]`, so a class is admitted only if
  some field of some reachable model can actually hold it.
  """
  allowed: set[tuple[str, str]] = set()
  seen: set[Any] = set()
  stack = list(roots)
  while stack:
    annotation = stack.pop()
    if annotation is None:
      continue
    try:
      if annotation in seen:
        continue
      seen.add(annotation)
    except TypeError:
      # Unhashable annotation (e.g. a `Literal` of a mutable default).
      continue
    args = get_args(annotation)
    if args:
      stack.extend(args)
      continue
    if not isinstance(annotation, type):
      continue
    try:
      if issubclass(annotation, BaseModel):
        allowed.add((annotation.__module__, annotation.__qualname__))
        stack.extend(
            field.annotation for field in annotation.model_fields.values()
        )
      elif issubclass(annotation, Enum):
        allowed.add((annotation.__module__, annotation.__qualname__))
    except TypeError:
      # `isinstance(x, type)` is True for bare generic aliases on some Python
      # versions, and `issubclass` rejects them.
      continue
  return allowed


@functools.lru_cache(maxsize=1)
def _allowed_globals() -> frozenset[tuple[str, str]]:
  """Returns the globals `loads` may resolve, computed once per process."""
  # Imported lazily: this module is imported by the v0 schema and by the
  # migration tool, neither of which should pay for the auth/event imports
  # until a legacy payload is actually loaded.
  try:
    from ..auth.auth_tool import AuthConfig
    from ..events.event_actions import EventActions

    # `EventActions.requested_auth_configs` is annotated `dict[str, Any]` at
    # runtime to avoid an import cycle, so `AuthConfig` has to be a root.
    derived = _walk_model_tree([EventActions, AuthConfig])
  except Exception:  # pylint: disable=broad-except
    logger.warning(
        "Could not derive the allowed types for legacy session payloads;"
        " falling back to the built-in types only. Loading a legacy"
        " `events.actions` value may fail.",
        exc_info=True,
    )
    derived = set()
  return _STATIC_ALLOWED_GLOBALS | frozenset(derived)


class _RestrictedUnpickler(pickle.Unpickler):
  """Unpickler that only resolves the types `EventActions` can hold."""

  def find_class(self, module: str, name: str) -> Any:  # noqa: ANN001
    if (module, name) in _allowed_globals():
      return super().find_class(module, name)
    raise pickle.UnpicklingError(
        f"Refusing to load {module}.{name} from a legacy pickled"
        " `events.actions` value: it is not a type that `EventActions` can"
        " hold. This value was either not written by ADK, or it holds session"
        " state that is not plain data (for example a callable). To recover a"
        " database whose contents you trust, migrate it offline with"
        " `adk migrate session --allow-unsafe-unpickling`, which also moves it"
        " off the pickled schema."
    )


def loads(data: bytes | bytearray) -> Any:
  """Loads a pickle payload using the restricted unpickler."""
  return _RestrictedUnpickler(io.BytesIO(data)).load()


def dumps(obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
  """Dumps a pickle payload; the restriction only applies when loading."""
  return pickle.dumps(obj, protocol)

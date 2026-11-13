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

"""Shared serialization helpers used by telemetry modules."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry.util.types import AnyValue
from pydantic import BaseModel
from pydantic import RootModel
from pydantic_core import to_jsonable_python

from ..auth.auth_credential import _redact_credential_secrets
from ..utils._credential_names import is_credential_type
from ..utils._credential_names import REDACTED

if TYPE_CHECKING:
  from pydantic.fields import ComputedFieldInfo
  from pydantic.fields import FieldInfo


def _mask_credential_values(
    value: object, dumped: object, *, by_alias: bool | None
) -> object:
  """Returns `dumped`, a JSON dump of `value`, with credential values masked.

  Dumping flattens nested models to dicts, which loses the type the credential
  check needs, so the live value is walked alongside its dump: model and
  dataclass fields, mapping values and list items alike. `by_alias` is the
  value the dump was made with; `None` means each model's own
  `serialize_by_alias` config decided.
  """
  if is_credential_type(type(value)):
    return REDACTED
  if isinstance(value, RootModel):
    # A root model dumps its root directly, with no `root` key.
    return _mask_credential_values(value.root, dumped, by_alias=by_alias)
  is_dataclass_instance = dataclasses.is_dataclass(value) and not isinstance(
      value, type
  )
  if (isinstance(value, BaseModel) or is_dataclass_instance) and isinstance(
      dumped, dict
  ):
    masked = dict(dumped)
    # Models and pydantic dataclasses declare `FieldInfo`s, and computed fields
    # `ComputedFieldInfo`s, whose aliases a dump by alias keys a field by; a
    # stdlib dataclass has only its field names.
    pydantic_fields: Mapping[str, FieldInfo] | None = getattr(
        type(value), "__pydantic_fields__", None
    )
    use_alias = by_alias
    if use_alias is None:
      config = getattr(
          type(value),
          "model_config"
          if isinstance(value, BaseModel)
          else "__pydantic_config__",
          None,
      )
      use_alias = bool(config and config.get("serialize_by_alias"))
    infos: dict[str, FieldInfo | ComputedFieldInfo | None]
    if pydantic_fields is None:
      infos = {
          f.name: None
          for f in (
              dataclasses.fields(value)
              if dataclasses.is_dataclass(value)
              else ()
          )
      }
    else:
      infos = dict(pydantic_fields)
      decorators = getattr(type(value), "__pydantic_decorators__", None)
      if decorators is not None:
        infos.update(
            (name, dec.info) for name, dec in decorators.computed_fields.items()
        )
    matched: set[str] = set()
    unmatched: list[str] = []
    for name, info in infos.items():
      # Only the key the dump used: another field's name can equal this alias.
      key = name
      if use_alias and info is not None:
        key = getattr(info, "serialization_alias", None) or info.alias or name
      if key not in masked:
        unmatched.append(name)
        continue
      matched.add(key)
      masked[key] = _mask_credential_values(
          getattr(value, name, None), masked[key], by_alias=by_alias
      )
    if isinstance(value, BaseModel):
      for key, extra_value in (value.__pydantic_extra__ or {}).items():
        if key in masked:
          matched.add(key)
          masked[key] = _mask_credential_values(
              extra_value, masked[key], by_alias=by_alias
          )
    # A stdlib dataclass can take an alias from `Annotated` metadata, which no
    # `FieldInfo` records; its dump keeps declaration order, so its unmatched
    # fields and leftover keys pair up in order. A model's unmatched fields are
    # ones left out of the dump, so they are never paired.
    leftover = [key for key in masked if key not in matched]
    if pydantic_fields is None and len(leftover) == len(unmatched):
      for name, key in zip(unmatched, leftover):
        masked[key] = _mask_credential_values(
            getattr(value, name, None), masked[key], by_alias=by_alias
        )
    return masked
  if isinstance(value, Mapping) and isinstance(dumped, dict):
    # Each key is formatted the way the dump formats it (`True` as "true",
    # `(1, 2)` as "1,2"), not paired by position, which a serializer that
    # reorders keys would break. Keys that format alike, such as `1` and "1",
    # collapse into one dump key that holds the last of their values.
    by_dump_key = {
        next(iter(to_jsonable_python({key: None}, fallback=str))): item
        for key, item in value.items()
    }
    masked = dict(dumped)
    for dump_key, item in by_dump_key.items():
      if dump_key in masked:
        masked[dump_key] = _mask_credential_values(
            item, masked[dump_key], by_alias=by_alias
        )
    return masked
  if (
      isinstance(value, Sequence)
      and not isinstance(value, (str, bytes, bytearray, memoryview))
      and isinstance(dumped, list)
      and len(value) == len(dumped)
  ):
    return [
        _mask_credential_values(item, item_dump, by_alias=by_alias)
        for item, item_dump in zip(value, dumped)
    ]
  return dumped


def _redact_credentials(obj: object, active: set[int]) -> object:
  """Returns ``obj`` as JSON-ready data with credential-typed values masked.

  `model_dump()` renders every field, including the ones `AuthCredential` and
  friends declare `Field(repr=False)`, so a credential that reaches a span
  attribute through this serializer would be exported in the clear even though
  `repr()` masks it. Masking here is by declared credential type, from
  `..utils._credential_names`, so the rule is the same rule `AutoTracingPlugin`
  applies to captured arguments. Plain dicts have no type to check;
  `safe_json_serialize` strips those by shape afterwards.

  Args:
    obj: The value to redact.
    active: Ids of the containers on the path to `obj`, so a cycle is caught
      rather than rebuilt into a finite tree.

  Returns:
    A JSON-ready copy of `obj` with credentials replaced by a marker.

  Raises:
    ValueError: If `obj` contains a reference cycle. `json.dumps` would raise
      the same error on it, and `safe_json_serialize` reports it the same way.
  """
  if is_credential_type(type(obj)):
    return REDACTED
  if isinstance(obj, (str, int, float, bool, type(None))):
    return obj
  if id(obj) in active:
    raise ValueError("Circular reference detected")
  active.add(id(obj))
  try:
    if isinstance(obj, BaseModel):
      dumped = _mask_credential_values(
          obj, obj.model_dump(mode="json"), by_alias=None
      )
      return _redact_credentials(dumped, active)
    if isinstance(obj, Mapping):
      # Keys `json.dumps` accepts stay as they are, so it renders them as it
      # would without redaction (a `(str, Enum)` member by its value).
      return {
          (
              key
              if key is None or isinstance(key, (str, int, float, bool))
              else str(key)
          ): _redact_credentials(value, active)
          for key, value in obj.items()
      }
    if isinstance(obj, Sequence) and not isinstance(
        obj, (bytes, bytearray, memoryview)
    ):
      return [_redact_credentials(item, active) for item in obj]
  finally:
    active.discard(id(obj))
  # Anything else is left to `json.dumps`'s `default` hook below.
  return obj


def safe_json_serialize(obj: object) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Handles Pydantic `BaseModel` instances (common as tool return types) by
  calling `model_dump(mode="json")` before JSON encoding, and masks credential
  material on the way: credential-typed values by type (see
  `_redact_credentials`), and credential payloads that arrive as plain dicts,
  such as `adk_request_credential` arguments, by shape with
  `_redact_credential_secrets`. Ordinary keys such as `token` outside a
  credential payload are kept. Masking applies only to values passed through
  this function: `serialize_content` dumps models without it, and
  `_experimental_semconv` uses its own serializer.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or `<not serializable>` if the object
    cannot be serialized.
  """

  def _default(o: object) -> object:
    if is_credential_type(type(o)):
      return REDACTED
    if isinstance(o, BaseModel):
      return _redact_credential_secrets(_redact_credentials(o, set()))
    return "<not serializable>"

  try:
    return json.dumps(
        _redact_credential_secrets(_redact_credentials(obj, set())),
        ensure_ascii=False,
        default=_default,
    )
  except (TypeError, ValueError, OverflowError, RecursionError):
    return "<not serializable>"


def serialize_content(content: types.ContentUnion | None) -> AnyValue:
  """Serialize a `types.ContentUnion` value into an OTel-friendly form.

  - `None` is preserved.
  - Pydantic models are dumped via `model_dump()`.
  - Strings are returned as-is.
  - Lists are recursively serialized.
  - Anything else falls back to `safe_json_serialize`.
  """
  if content is None:
    return None
  if isinstance(content, BaseModel):
    return content.model_dump()
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return [serialize_content(part) for part in content]
  return safe_json_serialize(content)

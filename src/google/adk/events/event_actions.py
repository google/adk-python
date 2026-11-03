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

import base64
import logging
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from google.genai.types import Content
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_serializer
from pydantic import field_validator
from pydantic import SerializerFunctionWrapHandler
from pydantic_core import PydanticSerializationError
from pydantic_core import to_jsonable_python

from ..tools.tool_confirmation import ToolConfirmation
from .ui_widget import UiWidget

if TYPE_CHECKING:
  from ..auth.auth_tool import AuthConfig
else:
  AuthConfig = Any

logger = logging.getLogger('google_adk.' + __name__)


def _coerce_non_utf8_bytes(obj: Any) -> Any:
  """Recursively unwraps BaseModels and base64-encodes non-UTF-8 bytes."""
  if isinstance(obj, (bytes, bytearray)):
    raw = bytes(obj)
    try:
      return raw.decode('utf-8')
    except UnicodeDecodeError:
      return base64.b64encode(raw).decode('ascii')
  if isinstance(obj, BaseModel):
    return _coerce_non_utf8_bytes(obj.model_dump(mode='python'))
  if isinstance(obj, dict):
    return {
        _coerce_non_utf8_bytes(k): _coerce_non_utf8_bytes(v)
        for k, v in obj.items()
    }
  if isinstance(obj, list):
    return [_coerce_non_utf8_bytes(v) for v in obj]
  if isinstance(obj, tuple):
    return tuple(_coerce_non_utf8_bytes(v) for v in obj)
  if isinstance(obj, set):
    return [_coerce_non_utf8_bytes(v) for v in obj]
  return obj


def _make_json_serializable(obj: Any) -> Any:
  """Converts an object into a JSON-serializable form.

  Used as a fallback when the default Pydantic serialization fails. Delegates to
  `pydantic_core.to_jsonable_python` so rich types (e.g. datetimes, bytes,
  Pydantic models) are serialized faithfully instead of being discarded.
  Non-UTF-8 `bytes` values are encoded as base64 strings and are not restored to
  `bytes` on read. Values that pydantic-core cannot serialize (e.g. Python
  callables stored in session state) are replaced with their `repr` via
  `serialize_unknown=True` so the overall structure can still be persisted
  without crashing.
  """
  try:
    return to_jsonable_python(obj, serialize_unknown=True)
  except (PydanticSerializationError, UnicodeDecodeError):
    return to_jsonable_python(
        _coerce_non_utf8_bytes(obj),
        serialize_unknown=True,
    )


class EventCompaction(BaseModel):  # type: ignore[misc]
  """The compaction of the events."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  start_timestamp: float
  """The start timestamp of the compacted events, in seconds."""

  end_timestamp: float
  """The end timestamp of the compacted events, in seconds."""

  compacted_content: Content
  """The compacted content of the events."""


class EventActions(BaseModel):  # type: ignore[misc]
  """Represents the actions attached to an event."""

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  skip_summarization: bool | None = None
  """If true, it won't call model to summarize function response.

  Only used for function_response event.
  """

  state_delta: dict[str, Any] = Field(default_factory=dict)
  """Indicates that the event is updating the state with the given delta."""

  @field_serializer('state_delta', mode='wrap')  # type: ignore[misc, untyped-decorator]
  def _serialize_state_delta(
      self, value: dict[str, object], handler: SerializerFunctionWrapHandler
  ) -> dict[str, Any]:
    # Use a wrap serializer so the default serialization (which honors callers'
    # `exclude`/`include` directives, e.g. the conformance harness excluding
    # internal `_adk_*` keys) is preserved. Only fall back to sanitization when
    # the value contains objects Pydantic cannot serialize (e.g. callables).
    try:
      try:
        return cast(dict[str, Any], handler(value))
      except (PydanticSerializationError, UnicodeDecodeError):
        return cast(dict[str, Any], handler(_coerce_non_utf8_bytes(value)))
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          'Failed to serialize `state_delta`; some values are not'
          ' JSON-serializable (e.g. callables) and will be replaced with a'
          ' string representation in the persisted event.',
          exc_info=True,
      )
      # Re-run the handler on the sanitized value so that caller `exclude` /
      # `include` directives are still applied to the fallback output.
      return cast(dict[str, Any], handler(_make_json_serializable(value)))

  artifact_delta: dict[str, int] = Field(default_factory=dict)
  """Indicates that the event is updating an artifact. key is the filename,
  value is the version."""

  transfer_to_agent: str | None = None
  """If set, the event transfers to the specified agent."""

  transfer_reason: str | None = None
  """The reason for transferring to the target agent."""

  escalate: bool | None = None
  """The agent is escalating to a higher level agent."""

  requested_auth_configs: dict[str, AuthConfig] = Field(default_factory=dict)
  """Authentication configurations requested by tool responses.

  This field will only be set by a tool response event indicating tool request
  auth credential.
  - Keys: The function call id. Since one function response event could contain
  multiple function responses that correspond to multiple function calls. Each
  function call could request different auth configs. This id is used to
  identify the function call.
  - Values: The requested auth config.
  """

  @field_validator('requested_auth_configs', mode='after')
  @classmethod
  def _parse_auth_configs(cls, v: dict[str, Any]) -> dict[str, Any]:
    if not v:
      return v
    from ..auth.auth_tool import AuthConfig

    return {
        k: AuthConfig.model_validate(val) if isinstance(val, dict) else val
        for k, val in v.items()
    }

  requested_tool_confirmations: dict[str, ToolConfirmation] = Field(
      default_factory=dict
  )
  """A dict of tool confirmation requested by this event, keyed by
  function call id."""

  compaction: EventCompaction | None = None
  """The compaction of the events."""

  end_of_agent: bool | None = None
  """If true, the current agent has finished its current run. Note that there
  can be multiple events with end_of_agent=True for the same agent within one
  invocation when there is a loop. This should only be set by ADK workflow."""

  agent_state: dict[str, Any] | None = None
  """The agent state at the current event, used for checkpoint and resume. This
  should only be set by ADK workflow."""

  @field_serializer('agent_state', mode='wrap')  # type: ignore[misc, untyped-decorator]
  def _serialize_agent_state(
      self,
      value: dict[str, Any] | None,
      handler: SerializerFunctionWrapHandler,
  ) -> dict[str, Any] | None:
    if value is None:
      return None
    # See `_serialize_state_delta` for why a wrap serializer is used.
    try:
      try:
        return cast(dict[str, Any] | None, handler(value))
      except (PydanticSerializationError, UnicodeDecodeError):
        return cast(
            dict[str, Any] | None, handler(_coerce_non_utf8_bytes(value))
        )
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          'Failed to serialize `agent_state`; some values are not'
          ' JSON-serializable (e.g. callables) and will be replaced with a'
          ' string representation in the persisted event.',
          exc_info=True,
      )
      # Re-run the handler on the sanitized value so that caller `exclude` /
      # `include` directives are still applied to the fallback output.
      return cast(dict[str, Any], handler(_make_json_serializable(value)))

  rewind_before_invocation_id: str | None = None
  """The invocation id to rewind to. This is only set for rewind event."""

  route: bool | int | str | list[bool | int | str] | None = None
  """Route or list of routes for workflow graph edge matching."""

  render_ui_widgets: list[UiWidget] | None = None
  """List of UI widgets to be rendered by the UI."""

  set_model_response: Any | None = None
  """The model response structured output."""

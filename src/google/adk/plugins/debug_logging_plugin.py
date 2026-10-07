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

"""Debug logging plugin for capturing complete interaction data to a file."""

from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import time
from enum import Enum
import logging
import os
from pathlib import Path
import re
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override
import yaml

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..auth.auth_credential import AuthCredential
from ..auth.auth_credential import HttpAuth
from ..auth.auth_credential import HttpCredentials
from ..auth.auth_credential import OAuth2Auth
from ..auth.auth_credential import ServiceAccount
from ..auth.auth_credential import ServiceAccountCredential
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..sessions.state import State
from ..tools.base_tool import BaseTool
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

_REDACTED = "[REDACTED]"

# Models that exist to carry a secret; an instance is replaced wholesale
# rather than dumped field by field.
_CREDENTIAL_MODELS = (
    AuthCredential,
    HttpAuth,
    HttpCredentials,
    OAuth2Auth,
    ServiceAccount,
    ServiceAccountCredential,
)

# Mapping keys whose value is a secret, for credentials that reach the plugin
# as plain dicts rather than as models: session state rehydrated from a session
# service, or the already dumped credential the OpenAPI tool auth handler keeps
# in state. Starts from the set bigquery_agent_analytics_plugin applies, plus
# the OAuth2 authorization-code fields, which ADK itself populates and which
# are enough on their own to complete a token exchange.
_SENSITIVE_KEYS = frozenset({
    "access_token",
    "api_key",
    "auth_code",
    "auth_response_uri",
    "authorization",
    "client_secret",
    "code_verifier",
    "google_access_id",
    "id_token",
    "password",
    "private_key",
    "private_key_id",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "sig",
    "signature",
    "token",
    "x_amz_credential",
    "x_amz_signature",
    "x_api_key",
    "x_goog_credential",
    "x_goog_security_token",
    "x_goog_signature",
})

# Substrings that name a secret wherever they sit in a key. Matched as
# substrings, not whole keys, so that the spellings the exact set above cannot
# enumerate are covered too: `openai_api_key`, `secret_key`,
# `service_account_credentials`.
_SENSITIVE_SUBSTRINGS = (
    "api_key",
    "credentials",
    "passwd",
    "password",
    "private_key",
    "secret",
)

# A key ending in one of these names a secret: `bearer_token`,
# `session_token`. Matched as a suffix rather than as a substring so that the
# usage counters, `prompt_token_count` and its siblings, keep their values.
_SENSITIVE_SUFFIXES = ("_token",)

# Session state keys are namespaced by scope. The scope says nothing about
# whether the value is a secret, so it is stripped before matching, otherwise
# `api_key` is redacted while `user:api_key` is written out.
_STATE_PREFIXES = (State.APP_PREFIX, State.USER_PREFIX)

# Splits a camel-cased key so that `apiKey` and `XApiKey` normalize to the
# same `api_key` that `api-key` and `api_key` do.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# A credential pasted into session state or a tool argument arrives as one
# long string, and a service account file keeps its private key inside that
# string, so no key name identifies it. Only an armored private key block is
# looked for. It is unambiguous, where a general secret scan would be both slow
# and prone to blanking ordinary text. The armor header is matched as a unit so
# that prose quoting one of its fragments is left alone, and only the block
# itself is replaced, so the rest of the string stays readable. A block whose
# footer never arrives is redacted to the end of the string rather than left in
# place.
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----"
    r".*?"
    r"(-----END [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----|\Z)",
    re.DOTALL,
)

# The debug file is written with the process umask otherwise, which commonly
# leaves it world-readable.
_OUTPUT_FILE_MODE = 0o600

# Bounds both walks below, which are otherwise unterminated on a
# self-referential object. Deeper than any credential model nests.
_MAX_WALK_DEPTH = 20


def _is_sensitive_key(key: Any) -> bool:
  """Whether a mapping key names a credential-bearing value."""
  if not isinstance(key, str):
    return False
  # `str.__str__` drops a subclass override of `lower`; hyphens and case
  # boundaries are folded so that `X-Api-Key` and `apiKey` match the same way
  # `api_key` does.
  normalized = (
      _CAMEL_BOUNDARY.sub("_", str.__str__(key)).lower().replace("-", "_")
  )
  # ADK stores exchanged auth credentials under a `temp:`-prefixed state key.
  if normalized.startswith(State.TEMP_PREFIX):
    return True
  for prefix in _STATE_PREFIXES:
    if normalized.startswith(prefix):
      normalized = normalized[len(prefix) :]
      break
  if normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES):
    return True
  return any(marker in normalized for marker in _SENSITIVE_SUBSTRINGS)


def _redact_private_keys(value: str) -> str:
  """Blanks any armored private key block, leaving the rest of the string."""
  # `str.__str__` drops a subclass override of `__contains__`, which `in`
  # would otherwise dispatch to, the same way `_is_sensitive_key` drops one
  # of `lower`.
  return _PRIVATE_KEY_BLOCK.sub(_REDACTED, str.__str__(value))


def _model_items(model: BaseModel) -> list[tuple[str, Any]]:
  """The (name, value) pairs held by a model, including any extra fields."""
  return [
      *model.__dict__.items(),
      *(model.__pydantic_extra__ or {}).items(),
  ]


def _holds_credential(obj: Any, depth: int = 0) -> bool:
  """Whether a credential model instance is reachable from `obj`.

  Dumping a model flattens a credential nested inside it into a plain dict, at
  which point only its key name could still identify it. A model that carries
  one anywhere below it therefore has to be walked field by field instead.
  """
  if depth > _MAX_WALK_DEPTH:
    return False
  if isinstance(obj, _CREDENTIAL_MODELS):
    return True
  child_depth = depth + 1
  if isinstance(obj, BaseModel):
    return any(_holds_credential(v, child_depth) for _, v in _model_items(obj))
  if isinstance(obj, dict):
    return any(_holds_credential(v, child_depth) for v in obj.values())
  if isinstance(obj, (list, tuple, set, frozenset)):
    return any(_holds_credential(v, child_depth) for v in obj)
  return False


class _DebugEntry(BaseModel):
  """A single debug log entry."""

  timestamp: str
  entry_type: str
  invocation_id: str | None = None
  agent_name: str | None = None
  data: dict[str, Any] = Field(default_factory=dict)


class _InvocationDebugState(BaseModel):
  """Per-invocation debug state."""

  invocation_id: str
  session_id: str
  app_name: str
  user_id: str | None = None
  start_time: str
  entries: list[_DebugEntry] = Field(default_factory=list)


class DebugLoggingPlugin(BasePlugin):
  """A plugin that captures complete debug information to a file.

  This plugin records detailed interaction data including:
  - LLM requests (model, system instruction, contents, tools)
  - LLM responses (content, usage metadata, errors)
  - Function calls with arguments
  - Function responses with results
  - Events yielded from the runner
  - Session state at the end of each invocation

  The output is written as YAML format for human readability. Each invocation
  is appended to the file as a separate YAML document (separated by ---).
  This format is easy to read. Credentials are redacted, but the file still
  holds whole prompts and responses, so it is created readable only by its
  owner and is not safe to hand around.

  Redaction covers credential models wherever they appear, mapping keys that
  name a secret with the `app:` or `user:` state scope stripped first, an
  armored private key block found inside any string, and every
  `temp:`-prefixed state key. That last rule blanks all temporary state, not
  only credentials, so an intermediate value passed between agents under a
  `temp:` key reads as `[REDACTED]` here.

  Example:
      >>> debug_plugin = DebugLoggingPlugin(output_path="/tmp/adk_debug.yaml")
      >>> runner = Runner(
      ...     agent=my_agent,
      ...     plugins=[debug_plugin],
      ... )

  Attributes:
      output_path: Path to the output file. Defaults to "adk_debug.yaml".
      include_session_state: Whether to include session state in the output.
      include_system_instruction: Whether to include system instructions.
  """

  def __init__(
      self,
      *,
      name: str = "debug_logging_plugin",
      output_path: str = "adk_debug.yaml",
      include_session_state: bool = True,
      include_system_instruction: bool = True,
  ):
    """Initialize the debug logging plugin.

    Args:
      name: The name of the plugin instance.
      output_path: Path to the output file. Defaults to "adk_debug.yaml".
      include_session_state: Whether to include session state snapshot.
      include_system_instruction: Whether to include full system instructions.
    """
    super().__init__(name)
    self._output_path = Path(output_path)
    self._include_session_state = include_session_state
    self._include_system_instruction = include_system_instruction
    self._invocation_states: dict[str, _InvocationDebugState] = {}
    self._warned_about_output_mode = False

  def _get_timestamp(self) -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

  def _serialize_content(
      self, content: types.Content | None
  ) -> dict[str, Any] | None:
    """Serialize Content to a dictionary."""
    if content is None:
      return None

    parts = []
    if content.parts:
      for part in content.parts:
        part_data: dict[str, Any] = {}
        if part.text:
          part_data["text"] = part.text
        if part.function_call:
          part_data["function_call"] = {
              "id": part.function_call.id,
              "name": part.function_call.name,
              "args": self._safe_serialize(part.function_call.args),
          }
        if part.function_response:
          part_data["function_response"] = {
              "id": part.function_response.id,
              "name": part.function_response.name,
              "response": self._safe_serialize(part.function_response.response),
          }
        if part.inline_data:
          part_data["inline_data"] = {
              "mime_type": part.inline_data.mime_type,
              "display_name": getattr(part.inline_data, "display_name", None),
              # Omit actual data to keep file size manageable
              "_data_omitted": True,
          }
        if part.file_data:
          part_data["file_data"] = {
              "file_uri": part.file_data.file_uri,
              "mime_type": part.file_data.mime_type,
          }
        if part.code_execution_result:
          part_data["code_execution_result"] = {
              "outcome": str(part.code_execution_result.outcome),
              "output": part.code_execution_result.output,
          }
        if part.executable_code:
          part_data["executable_code"] = {
              "language": str(part.executable_code.language),
              "code": part.executable_code.code,
          }
        if part_data:
          parts.append(part_data)

    return {"role": content.role, "parts": parts}

  def _safe_serialize(self, obj: Any, depth: int = 0) -> Any:
    """Safely serialize an object to JSON-compatible format.

    A credential model is replaced with a redaction marker wherever it sits:
    at the top level, or nested inside a dict, list, tuple or another model,
    under any key name. Mapping keys that name a secret are redacted too, for
    credentials that arrive already dumped to a plain dict. An armored private
    key block, which no key name identifies, is cut out of whatever string it
    sits in.
    """
    if obj is None:
      return None
    if isinstance(obj, _CREDENTIAL_MODELS):
      return _REDACTED
    if depth > _MAX_WALK_DEPTH:
      # Terminates a self-referential object. Only the type name survives, so
      # reaching the bound cannot uncover a value.
      return f"<{type(obj).__name__} ...>"
    child_depth = depth + 1
    if isinstance(obj, Enum):
      # A member of a `str` or `int` subclass enum passes the scalar check
      # below unchanged, and then reaches `yaml.dump` as a Python object,
      # which writes a `!!python/object` tag that `yaml.safe_load` refuses.
      return self._safe_serialize(obj.value, child_depth)
    if isinstance(obj, str):
      return _redact_private_keys(obj)
    if isinstance(obj, (int, float, bool)):
      return obj
    if isinstance(obj, (date, time)):
      return obj.isoformat()
    if isinstance(obj, (list, tuple)):
      return [self._safe_serialize(item, child_depth) for item in obj]
    if isinstance(obj, dict):
      return {
          k: (
              _REDACTED
              if _is_sensitive_key(k)
              else self._safe_serialize(v, child_depth)
          )
          for k, v in obj.items()
      }
    if isinstance(obj, BaseModel):
      if _holds_credential(obj):
        # Serialize the raw field values, so that the nested credential is
        # still a model instance when it is reached. Dumping first would leave
        # only its key name to go on, and that name is caller-chosen. The
        # values skip `model_dump`, so each one is normalized on the way
        # through the branches above rather than by pydantic.
        return self._safe_serialize(
            {
                name: value
                for name, value in _model_items(obj)
                if value is not None
            },
            depth,
        )
      try:
        dumped = obj.model_dump(mode="json", exclude_none=True)
      except Exception:
        return str(obj)
      # Recurse so that credential-named keys within the dump are redacted.
      return self._safe_serialize(dumped, depth)
    if isinstance(obj, bytes):
      return f"<bytes: {len(obj)} bytes>"
    try:
      return str(obj)
    except Exception:
      return "<unserializable>"

  def _add_entry(
      self,
      invocation_id: str,
      entry_type: str,
      agent_name: str | None = None,
      **data: Any,
  ) -> None:
    """Add a debug entry to the current invocation state."""
    if invocation_id not in self._invocation_states:
      logger.warning(
          "No debug state for invocation %s, skipping entry", invocation_id
      )
      return

    entry = _DebugEntry(
        timestamp=self._get_timestamp(),
        entry_type=entry_type,
        invocation_id=invocation_id,
        agent_name=agent_name,
        data=self._safe_serialize(data),
    )
    self._invocation_states[invocation_id].entries.append(entry)

  @override
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> types.Content | None:
    """Log user message and invocation start."""
    invocation_id = invocation_context.invocation_id

    self._add_entry(
        invocation_id,
        "user_message",
        content=self._serialize_content(user_message),
    )
    return None

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> types.Content | None:
    """Initialize debug state for this invocation."""
    invocation_id = invocation_context.invocation_id
    session = invocation_context.session

    state = _InvocationDebugState(
        invocation_id=invocation_id,
        session_id=session.id,
        app_name=session.app_name,
        user_id=invocation_context.user_id,
        start_time=self._get_timestamp(),
    )
    self._invocation_states[invocation_id] = state

    self._add_entry(
        invocation_id,
        "invocation_start",
        agent_name=getattr(invocation_context.agent, "name", None),
        branch=invocation_context.branch,
    )
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Event | None:
    """Log events yielded from the runner."""
    invocation_id = invocation_context.invocation_id

    event_data: dict[str, Any] = {
        "event_id": event.id,
        "author": event.author,
        "content": self._serialize_content(event.content),
        "is_final_response": event.is_final_response(),
        "partial": event.partial,
        "turn_complete": event.turn_complete,
        "branch": event.branch,
    }

    if event.actions:
      actions_data: dict[str, Any] = {}
      if event.actions.state_delta:
        actions_data["state_delta"] = self._safe_serialize(
            event.actions.state_delta
        )
      if event.actions.artifact_delta:
        # Preserve filename -> version mapping for debugging
        actions_data["artifact_delta"] = dict(event.actions.artifact_delta)
      if event.actions.transfer_to_agent:
        actions_data["transfer_to_agent"] = event.actions.transfer_to_agent
      if event.actions.escalate:
        actions_data["escalate"] = event.actions.escalate
      if event.actions.requested_auth_configs:
        actions_data["requested_auth_configs"] = len(
            event.actions.requested_auth_configs
        )
      if actions_data:
        event_data["actions"] = actions_data

    if event.grounding_metadata:
      event_data["has_grounding_metadata"] = True

    if event.usage_metadata:
      event_data["usage_metadata"] = {
          "prompt_token_count": event.usage_metadata.prompt_token_count,
          "candidates_token_count": event.usage_metadata.candidates_token_count,
          "total_token_count": event.usage_metadata.total_token_count,
      }

    if event.error_code:
      event_data["error_code"] = event.error_code
      event_data["error_message"] = event.error_message

    if event.long_running_tool_ids:
      event_data["long_running_tool_ids"] = list(event.long_running_tool_ids)

    self._add_entry(
        invocation_id,
        "event",
        agent_name=event.author,
        **event_data,
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Finalize and write debug data to file."""
    invocation_id = invocation_context.invocation_id

    if invocation_id not in self._invocation_states:
      logger.warning(
          "No debug state for invocation %s, skipping write", invocation_id
      )
      return

    state = self._invocation_states[invocation_id]

    # Add session state snapshot if enabled
    if self._include_session_state:
      session = invocation_context.session
      self._add_entry(
          invocation_id,
          "session_state_snapshot",
          state=self._safe_serialize(session.state),
          event_count=len(session.events),
      )

    self._add_entry(invocation_id, "invocation_end")

    # Write to file as YAML
    try:
      output_data = state.model_dump(mode="json", exclude_none=True)
      fd = os.open(
          self._output_path,
          os.O_WRONLY | os.O_CREAT | os.O_APPEND,
          _OUTPUT_FILE_MODE,
      )
      # The mode above only applies to a file this call creates. A file left
      # behind by an earlier run keeps whatever mode it had, so say so rather
      # than silently changing permissions the user may have chosen.
      if not self._warned_about_output_mode and os.fstat(fd).st_mode & 0o077:
        self._warned_about_output_mode = True
        logger.warning(
            "Debug output file %s is readable beyond its owner and holds"
            " whole prompts and responses; restrict it to mode 600.",
            self._output_path,
        )
      with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(
            output_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
      logger.debug(
          "Wrote debug data for invocation %s to %s",
          invocation_id,
          self._output_path,
      )
    except Exception as e:
      logger.error("Failed to write debug data: %s", e)
    finally:
      # Cleanup invocation state
      self._invocation_states.pop(invocation_id, None)

  @override
  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    """Log agent execution start."""
    self._add_entry(
        callback_context.invocation_id,
        "agent_start",
        agent_name=callback_context.agent_name,
        branch=callback_context._invocation_context.branch,
    )
    return None

  @override
  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> types.Content | None:
    """Log agent execution completion."""
    self._add_entry(
        callback_context.invocation_id,
        "agent_end",
        agent_name=callback_context.agent_name,
    )
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse | None:
    """Log LLM request before sending to model."""
    request_data: dict[str, Any] = {
        "model": llm_request.model,
        "content_count": len(llm_request.contents),
        "contents": [self._serialize_content(c) for c in llm_request.contents],
    }

    if llm_request.tools_dict:
      request_data["tools"] = list(llm_request.tools_dict.keys())

    if llm_request.config:
      config = llm_request.config
      config_data: dict[str, Any] = {}

      if self._include_system_instruction and config.system_instruction:
        config_data["system_instruction"] = config.system_instruction
      elif config.system_instruction:
        # Just indicate presence without full content
        si = config.system_instruction
        if isinstance(si, str):
          config_data["system_instruction_length"] = len(si)
        else:
          config_data["has_system_instruction"] = True

      if config.temperature is not None:
        config_data["temperature"] = config.temperature
      if config.top_p is not None:
        config_data["top_p"] = config.top_p
      if config.top_k is not None:
        config_data["top_k"] = config.top_k
      if config.max_output_tokens is not None:
        config_data["max_output_tokens"] = config.max_output_tokens
      if config.response_mime_type:
        config_data["response_mime_type"] = config.response_mime_type
      if config.response_schema:
        config_data["has_response_schema"] = True

      if config_data:
        request_data["config"] = config_data

    self._add_entry(
        callback_context.invocation_id,
        "llm_request",
        agent_name=callback_context.agent_name,
        **request_data,
    )
    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> LlmResponse | None:
    """Log LLM response after receiving from model."""
    response_data: dict[str, Any] = {
        "content": self._serialize_content(llm_response.content),
        "partial": llm_response.partial,
        "turn_complete": llm_response.turn_complete,
    }

    if llm_response.error_code:
      response_data["error_code"] = llm_response.error_code
      response_data["error_message"] = llm_response.error_message

    if llm_response.usage_metadata:
      response_data["usage_metadata"] = {
          "prompt_token_count": llm_response.usage_metadata.prompt_token_count,
          "candidates_token_count": (
              llm_response.usage_metadata.candidates_token_count
          ),
          "total_token_count": llm_response.usage_metadata.total_token_count,
          "cached_content_token_count": (
              llm_response.usage_metadata.cached_content_token_count
          ),
      }

    if llm_response.grounding_metadata:
      response_data["has_grounding_metadata"] = True

    if llm_response.finish_reason:
      response_data["finish_reason"] = str(llm_response.finish_reason)

    if llm_response.model_version:
      response_data["model_version"] = llm_response.model_version

    self._add_entry(
        callback_context.invocation_id,
        "llm_response",
        agent_name=callback_context.agent_name,
        **response_data,
    )
    return None

  @override
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> LlmResponse | None:
    """Log LLM error."""
    self._add_entry(
        callback_context.invocation_id,
        "llm_error",
        agent_name=callback_context.agent_name,
        error_type=type(error).__name__,
        error_message=str(error),
        model=llm_request.model,
    )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> dict[str, Any] | None:
    """Log tool execution start."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_call",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        args=self._safe_serialize(tool_args),
    )
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> dict[str, Any] | None:
    """Log tool execution completion."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_response",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        result=self._safe_serialize(result),
    )
    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, Any] | None:
    """Log tool error."""
    self._add_entry(
        tool_context.invocation_id,
        "tool_error",
        agent_name=tool_context.agent_name,
        tool_name=tool.name,
        function_call_id=tool_context.function_call_id,
        args=self._safe_serialize(tool_args),
        error_type=type(error).__name__,
        error_message=str(error),
    )
    return None

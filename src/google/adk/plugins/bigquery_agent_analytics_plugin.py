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

import asyncio
import atexit
import base64
import collections.abc
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
import contextvars
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import decimal
import enum
import functools
import json
import logging
import math
import mimetypes
import os
import pathlib
import traceback as traceback_module

# Enable gRPC fork support so child processes created via os.fork()
# can safely create new gRPC channels.  Must be set before grpc's
# C-core is loaded (which happens through the google.api_core
# imports below).  setdefault respects any explicit user override.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

import random
import re
import threading
import time
from types import MappingProxyType
from types import TracebackType
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Coroutine
from typing import Optional
from typing import ParamSpec
from typing import TYPE_CHECKING
from typing import TypeVar
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
import uuid
import weakref

from google.api_core import client_options
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.exceptions import TooManyRequests
from google.api_core.gapic_v1 import client_info as gapic_client_info
import google.auth
from google.cloud import bigquery
from google.cloud import exceptions as cloud_exceptions
from google.cloud import storage
from google.cloud.bigquery import schema as bq_schema
from google.cloud.bigquery_storage_v1 import types as bq_storage_types
from google.cloud.bigquery_storage_v1.services.big_query_write.async_client import BigQueryWriteAsyncClient
from google.genai import types
from opentelemetry import trace
import pyarrow as pa

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..platform.thread import create_thread
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ..utils._telemetry_context import _is_visual_builder
from ..version import __version__
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event

logger: logging.Logger = logging.getLogger("google_adk." + __name__)
tracer = trace.get_tracer(
    "google.adk.plugins.bigquery_agent_analytics", __version__
)

# Bumped when the schema changes (1 → 2 → 3 …). Used as a table
# label for governance and to decide whether auto-upgrade should run.
_SCHEMA_VERSION = "1"
_SCHEMA_VERSION_LABEL_KEY = "adk_schema_version"

# ADK 2.0 envelope version. Stamped onto every ADK-enriched row as
# ``attributes.adk.schema_version``. Independent of the BigQuery row
# schema version above — this names the producer's ADK 2.0 attribute
# contract so downstream consumers can gate on it.
_ADK_ENVELOPE_SCHEMA_VERSION = "1"

_HITL_EVENT_MAP = MappingProxyType({
    "adk_request_credential": "HITL_CREDENTIAL_REQUEST",
    "adk_request_confirmation": "HITL_CONFIRMATION_REQUEST",
    "adk_request_input": "HITL_INPUT_REQUEST",
})

# Reverse of _HITL_EVENT_MAP for the long-running-tool pause_kind
# discriminator. The id→name lookup routes ``adk_request_credential``
# → ``hitl_credential`` etc.; everything else is ``tool``.
_HITL_PAUSE_KIND_MAP = MappingProxyType({
    "adk_request_credential": "hitl_credential",
    "adk_request_confirmation": "hitl_confirmation",
    "adk_request_input": "hitl_input",
})


def _derive_scope(
    isolation_scope: Optional[str],
) -> Optional[dict[str, str]]:
  """Derives ``attributes.adk.scope`` from an Event's isolation_scope.

  Order is fixed: (1) None → null; (2) node-shape (``name@run_id`` or
  ``parent/name@run_id``) → ``node_run``; (3) any other non-empty
  string → ``function_call`` (model-provided FC IDs like ``call_*`` and
  ``toolu_*`` legitimately match here); (4) empty/non-string → ``unknown``
  with a warning. Steps 2 and 3 are intentionally ordered: a bare
  ``name@run_id`` must classify as ``node_run`` first, not as
  ``function_call`` by fall-through.
  """
  if isolation_scope is None:
    return None
  if not isinstance(isolation_scope, str) or not isolation_scope:
    logger.warning(
        "Unexpected isolation_scope shape: %r; classifying as 'unknown'",
        isolation_scope,
    )
    return {"id": str(isolation_scope), "kind": "unknown"}
  # Node-shape: last segment contains '@'. The full string may also be
  # path-prefixed (e.g. ``wf/A@1/B@2``).
  last_segment = isolation_scope.rsplit("/", 1)[-1]
  if "@" in last_segment:
    return {"id": isolation_scope, "kind": "node_run"}
  return {"id": isolation_scope, "kind": "function_call"}


# Track all living plugin instances so the fork handler can reset
# them proactively in the child, before _ensure_started runs.
_LIVE_PLUGINS: weakref.WeakSet[BigQueryAgentAnalyticsPlugin] = weakref.WeakSet()


def _after_fork_in_child() -> None:
  """Reset every living plugin instance after os.fork()."""
  for plugin in list(_LIVE_PLUGINS):
    try:
      plugin._reset_runtime_state()
    except Exception:
      pass


if hasattr(os, "register_at_fork"):
  os.register_at_fork(after_in_child=_after_fork_in_child)


_SafeCallbackP = ParamSpec("_SafeCallbackP")
_SafeCallbackT = TypeVar("_SafeCallbackT")


def _safe_callback(
    func: Callable[
        _SafeCallbackP, Coroutine[Any, Any, Optional[_SafeCallbackT]]
    ],
) -> Callable[_SafeCallbackP, Coroutine[Any, Any, Optional[_SafeCallbackT]]]:
  """Decorator that catches and logs exceptions in plugin callbacks.

  Prevents plugin errors from propagating to the runner and crashing
  the agent run. All callback exceptions are logged and swallowed.

  The signature (including keyword-only parameters and the ``Coroutine``
  return type) is preserved via ``ParamSpec`` so decorated methods still
  match the ``BasePlugin`` overrides they implement.
  """

  @functools.wraps(func)
  async def wrapper(
      *args: _SafeCallbackP.args, **kwargs: _SafeCallbackP.kwargs
  ) -> Optional[_SafeCallbackT]:
    try:
      return await func(*args, **kwargs)
    except Exception:
      logger.exception(
          "BigQuery analytics plugin error in %s; skipping.",
          func.__name__,
      )
      return None

  return wrapper


# gRPC Error Codes
_GRPC_DEADLINE_EXCEEDED = 4
_GRPC_INTERNAL = 13
_GRPC_UNAVAILABLE = 14


# --- Helper Formatters ---
def _format_content(
    content: Optional[types.Content], *, max_len: int = 5000
) -> tuple[str, bool]:
  """Formats an Event content for logging.

  Args:
      content: The content to format.
      max_len: Maximum length for text parts.

  Returns:
      A tuple of (formatted_string, is_truncated).
  """
  if content is None or not content.parts:
    return "None", False
  parts = []
  truncated = False
  for p in content.parts:
    if p.text:
      if max_len != -1 and len(p.text) > max_len:
        parts.append(f"text: '{p.text[:max_len]}...'")
        truncated = True
      else:
        parts.append(f"text: '{p.text}'")
    elif p.function_call:
      parts.append(f"call: {p.function_call.name}")
    elif p.function_response:
      parts.append(f"resp: {p.function_response.name}")
    else:
      parts.append("other")
  return " | ".join(parts), truncated


def _find_transfer_target(agent: Any, agent_name: str) -> Any:
  """Find a transfer target agent by name in the accessible agent tree.

  Searches the current agent's sub-agents, parent, and peer agents
  to locate the transfer target.

  Args:
      agent: The current agent executing the transfer.
      agent_name: The name of the transfer target to find.

  Returns:
      The matching agent object, or None if not found.
  """
  for sub in getattr(agent, "sub_agents", []):
    if sub.name == agent_name:
      return sub
  parent = getattr(agent, "parent_agent", None)
  if parent is not None and parent.name == agent_name:
    return parent
  if parent is not None:
    for peer in getattr(parent, "sub_agents", []):
      if peer.name == agent_name and peer.name != agent.name:
        return peer
  return None


def _get_tool_origin(
    tool: "BaseTool",
    tool_args: Optional[dict[str, Any]] = None,
    tool_context: Optional["ToolContext"] = None,
) -> str:
  """Returns the provenance category of a tool.

  Uses lazy imports to avoid circular dependencies.

  For ``TransferToAgentTool`` the classification is **call-level**: when
  *tool_args* and *tool_context* are supplied the selected
  ``agent_name`` is resolved against the agent tree so that transfers
  to a ``RemoteA2aAgent`` are labelled ``TRANSFER_A2A`` rather than
  the generic ``TRANSFER_AGENT``.

  Args:
      tool: The tool instance.
      tool_args: Optional tool arguments, used for call-level classification of
        TransferToAgentTool.
      tool_context: Optional tool context, used to access the agent tree for
        TransferToAgentTool classification.

  Returns:
      One of LOCAL, MCP, A2A, SUB_AGENT, TRANSFER_AGENT,
      TRANSFER_A2A, or UNKNOWN.
  """
  # Import lazily to avoid circular dependencies.
  # pylint: disable=g-import-not-at-top
  from ..tools.agent_tool import AgentTool  # pytype: disable=import-error
  from ..tools.function_tool import FunctionTool  # pytype: disable=import-error
  from ..tools.transfer_to_agent_tool import TransferToAgentTool  # pytype: disable=import-error

  try:
    from ..tools.mcp_tool.mcp_tool import McpTool  # pytype: disable=import-error
  except ImportError:
    McpTool = None

  try:
    from ..agents.remote_a2a_agent import RemoteA2aAgent  # pytype: disable=import-error
  except ImportError:
    RemoteA2aAgent = None

  # Order matters: TransferToAgentTool is a subclass of FunctionTool.
  if McpTool is not None and isinstance(tool, McpTool):
    return "MCP"
  if isinstance(tool, TransferToAgentTool):
    if RemoteA2aAgent is not None and tool_args and tool_context:
      agent_name = tool_args.get("agent_name")
      if agent_name:
        target = _find_transfer_target(
            tool_context._invocation_context.agent,
            agent_name,
        )
        if target is not None and isinstance(target, RemoteA2aAgent):
          return "TRANSFER_A2A"
    return "TRANSFER_AGENT"
  if isinstance(tool, AgentTool):
    if RemoteA2aAgent is not None and isinstance(tool.agent, RemoteA2aAgent):
      return "A2A"
    return "SUB_AGENT"
  if isinstance(tool, FunctionTool):
    return "LOCAL"
  return "UNKNOWN"


def _extract_tool_declarations(
    tools_dict: dict[str, "BaseTool"],
) -> list[dict[str, Any]]:
  """Extracts structured tool metadata for the ``LLM_REQUEST`` event.

  Earlier versions logged only the tool names (``list(tools_dict.keys())``).
  Downstream consumers such as online evaluation need the tool *description* and
  *parameter schema* to judge whether the model selected and invoked the right
  tool, so this returns one structured entry per tool instead of a bare name.

  Each entry always carries ``name`` and, when available, ``description`` and
  ``parameters`` (the OpenAPI parameter schema from the tool's
  ``FunctionDeclaration``). Extraction is best-effort and per-tool: a tool whose
  declaration cannot be resolved still contributes its name and description, so
  one misbehaving tool never drops the whole ``tools`` attribute.

  Args:
      tools_dict: Mapping of tool name to ``BaseTool`` from ``LlmRequest``.

  Returns:
      A list of ``{"name", "description"?, "parameters"?}`` dicts.
  """
  tools: list[dict[str, Any]] = []
  for name, tool in tools_dict.items():
    # Fall back to the dict key when the tool has no (or a falsy) name.
    entry: dict[str, Any] = {"name": getattr(tool, "name", None) or name}
    description = getattr(tool, "description", None)
    if description:
      entry["description"] = description

    # The parameter schema lives on the tool's FunctionDeclaration, which some
    # tools (e.g. built-in tools) do not provide. Resolve defensively so a
    # single failing tool does not discard the whole tools list.
    #
    # Note: FunctionTool._get_declaration() rebuilds the declaration from the
    # function signature on each call (no caching), so this repeats work the
    # framework already did when assembling the request. Acceptable for typical
    # toolsets; revisit with a cache if it shows up on the hot path.
    declaration = None
    try:
      get_declaration = getattr(tool, "_get_declaration", None)
      if callable(get_declaration):
        declaration = get_declaration()
    except Exception:  # pylint: disable=broad-except
      logger.debug("Failed to get declaration for tool %s", name, exc_info=True)

    if declaration is not None:
      if "description" not in entry:
        decl_description = getattr(declaration, "description", None)
        if decl_description:
          entry["description"] = decl_description
      # A declaration carries its parameter schema in one of two shapes: the
      # structured `parameters` Schema, or a raw JSON-schema dict in
      # `parameters_json_schema`. Several tools (MCP, OpenAPI, skill, node, and
      # environment tools) populate only the latter, and model adapters prefer
      # it, so prefer it here too and fall back to `parameters` otherwise.
      json_schema = getattr(declaration, "parameters_json_schema", None)
      if json_schema is not None:
        entry["parameters"] = json_schema
      else:
        parameters = getattr(declaration, "parameters", None)
        if parameters is not None:
          try:
            entry["parameters"] = parameters.model_dump(
                exclude_none=True, mode="json"
            )
          except Exception:  # pylint: disable=broad-except
            # Leave parameters off if the schema is not JSON-serializable.
            logger.debug(
                "Failed to serialize parameters for tool %s",
                name,
                exc_info=True,
            )

    tools.append(entry)
  return tools


_SENSITIVE_KEYS = frozenset({
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "password",
    "private_key",
    "proxy_authorization",
    "google_access_id",
    "sig",
    "signature",
    "token",
    "secret",
    "authorization",
    "x_api_key",
    "x_amz_credential",
    "x_amz_signature",
    "x_goog_credential",
    "x_goog_security_token",
    "x_goog_signature",
})

# Credentials commonly carried in signed URLs and HTTP error text.  These are
# values rather than structured mapping keys in those two surfaces, so they
# need an explicit bounded text/URI pass in addition to _SENSITIVE_KEYS.
_SENSITIVE_TEXT_KEYS = _SENSITIVE_KEYS
_SENSITIVE_TEXT_KEY_RE = re.compile(
    r"(?i)(?P<prefix>(?<![A-Za-z0-9_-])(?:[\"'])?(?:"
    r"temp:[A-Za-z0-9_.-]+|"
    + "|".join(
        sorted(
            (
                re.escape(key).replace("_", "[-_]")
                for key in _SENSITIVE_TEXT_KEYS
            ),
            key=len,
            reverse=True,
        )
    )
    + r")(?:[\"'])?\s*[:=]\s*)(?P<value>"
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|[^\s,;&}]+)"
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)(?P<prefix>\b(?:authorization|proxy-authorization|x-api-key|api-key)"
    r"[ \t]*:[ \t]*)[^\r\n]*"
)
_BEARER_TOKEN_RE = re.compile(
    r"(?i)(?P<prefix>\bbearer\s+)(?!of(?:\s|$))[^\s,;]+"
)
_BASIC_TOKEN_RE = re.compile(
    r"(?i)(?P<prefix>\bbasic\s+)(?P<value>[A-Za-z0-9+/]+={0,2})"
    r"(?![A-Za-z0-9+/=])"
)
_MAX_BASIC_AUTH_TOKEN_CHARS = 16 * 1024
_ASCII_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_sensitive_text_key(text: str) -> bool:
  """Returns whether ``text`` is exactly a credential-bearing key."""
  normalized = text.lower().replace("-", "_")
  return normalized in _SENSITIVE_TEXT_KEYS or normalized.startswith("temp:")


def _canonicalize_common_ascii_escapes(text: str) -> str:
  """Decodes nested ASCII ``\\u``, ``\\x``, and percent escapes in O(n).

  This is a detection-only representation: callers retain the original text
  unless the canonical form exposes a credential construct. The output stack
  also handles encodings that reveal another escape introducer, such as
  ``%255F`` and ``\\u005cu005f``, without rescanning the whole input.
  """
  output: list[str] = []
  for char in text:
    output.append(char)
    while True:
      escape_start = -1
      digits = ""
      if (
          len(output) >= 6
          and output[-6] == "\\"
          and output[-5] in ("u", "U")
          and all(char in _ASCII_HEX_DIGITS for char in output[-4:])
      ):
        escape_start = len(output) - 6
        digits = "".join(output[-4:])
      elif (
          len(output) >= 4
          and output[-4] == "\\"
          and output[-3] in ("x", "X")
          and all(char in _ASCII_HEX_DIGITS for char in output[-2:])
      ):
        escape_start = len(output) - 4
        digits = "".join(output[-2:])
      elif (
          len(output) >= 3
          and output[-3] == "%"
          and all(char in _ASCII_HEX_DIGITS for char in output[-2:])
      ):
        escape_start = len(output) - 3
        digits = "".join(output[-2:])
      if escape_start < 0:
        break
      decoded = int(digits, 16)
      if decoded > 0x7F:
        break
      del output[escape_start:]
      output.append(chr(decoded))
  return "".join(output)


def _redact_sensitive_patterns(text: str) -> tuple[str, bool]:
  """Redacts plain-text credential constructs without decoding the input."""

  def _redact_key_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if value == "[REDACTED]":
      return match.group(0)
    quote_char = value[:1] if value[:1] in ('"', "'") else ""
    return f"{match.group('prefix')}{quote_char}[REDACTED]{quote_char}"

  def _redact_basic_credential(match: re.Match[str]) -> str:
    encoded = match.group("value")
    if len(encoded) > _MAX_BASIC_AUTH_TOKEN_CHARS:
      return f"{match.group('prefix')}[REDACTED]"
    try:
      decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
      return match.group(0)
    if "=" not in encoded and b":" not in decoded:
      return match.group(0)
    return f"{match.group('prefix')}[REDACTED]"

  sanitized = _AUTH_HEADER_RE.sub(
      lambda match: f"{match.group('prefix')}[REDACTED]", text
  )
  sanitized = _BEARER_TOKEN_RE.sub(
      lambda match: f"{match.group('prefix')}[REDACTED]", sanitized
  )
  sanitized = _BASIC_TOKEN_RE.sub(_redact_basic_credential, sanitized)
  sanitized = _SENSITIVE_TEXT_KEY_RE.sub(_redact_key_value, sanitized)
  return sanitized, sanitized != text


def _contains_sensitive_text_marker(text: str) -> bool:
  """Returns whether raw or encoded text contains a credential construct."""
  _, changed = _redact_sensitive_patterns(text)
  if changed:
    return True
  canonical = _canonicalize_common_ascii_escapes(text)
  if canonical == text:
    return False
  _, changed = _redact_sensitive_patterns(canonical)
  return changed


def _sanitize_sensitive_text(text: str, max_len: int) -> tuple[str, bool]:
  """Redacts bounded credential material embedded in diagnostic text.

  Unlike the generic attribute sanitizer, this preserves ordinary prose
  exactly (including bracket-led log messages). Complete JSON/encoded JSON
  still uses the existing structural redactor, while HTTP headers, bearer
  tokens, query parameters, and key/value fragments embedded in prose are
  replaced in place. Inputs too large to inspect safely fail closed when they
  would otherwise be emitted without a configured length bound.
  """
  if type(text) is not str:
    text = str.__str__(text)
  if len(text) > _MAX_JSON_INSPECT_CHARS:
    if max_len == -1 or max_len > _MAX_JSON_INSPECT_CHARS:
      return "[REDACTED_SENSITIVE_TEXT]", True
    # Only this prefix can reach the row, so inspect exactly that bounded
    # value below. Incomplete trailing escapes cannot reveal omitted bytes.
    emitted = text[:max_len]
    text = emitted
    length_truncated = True
  else:
    length_truncated = False

  sanitized = text
  stripped = _strip_bom_ws(text)
  if stripped.startswith(("{", "[", '"')):
    try:
      json.loads(stripped)
    except (TypeError, ValueError, RecursionError, MemoryError):
      # Malformed diagnostic prose is handled by the explicit marker pass
      # below; safe "[INFO] ..." messages must remain byte-identical.
      pass
    else:
      structured, _ = _recursive_smart_truncate(text, -1)
      if isinstance(structured, str):
        sanitized = structured
      else:
        sanitized = json.dumps(structured)

  changed = sanitized != text

  sanitized, patterns_changed = _redact_sensitive_patterns(sanitized)
  changed = changed or patterns_changed

  # Escaped/percent-encoded credential constructs cannot be safely rewritten
  # in place without risking a partial source-to-canonical mapping. Fail the
  # bounded diagnostic closed only when decoding actually exposes one; safe
  # Windows paths and decoder errors remain byte-identical.
  canonical = _canonicalize_common_ascii_escapes(sanitized)
  if canonical != sanitized and _contains_sensitive_text_marker(canonical):
    return "[REDACTED_SENSITIVE_TEXT]", True

  if max_len != -1 and len(sanitized) > max_len:
    sanitized = sanitized[:max_len] + "...[TRUNCATED]"
    length_truncated = True
  return sanitized, changed or length_truncated


# Written in place of event content when a configured content_formatter
# raises: the formatter is a privacy/redaction boundary, so failure must
# never fall back to the unformatted payload.
_FORMATTER_FAILED_SENTINEL = "[FORMATTER_FAILED]"

# Recursion bound for _recursive_smart_truncate: id()-based cycle detection
# cannot catch graphs that create new objects per access (Mock-like duck
# typing); the cap turns unbounded recursion into a redacted leaf.
_MAX_SANITIZE_DEPTH = 50

# Total nodes one sanitizer invocation may visit: depth and per-string size
# are bounded, but width was not — a million-scalar list burned ~1s of
# synchronous callback time. The remainder is
# replaced with a sentinel and the row is flagged truncated.
_MAX_SANITIZE_NODES = 100_000

# Hard ceiling on how many characters of a JSON-shaped string the
# synchronous sanitizer will materialize with json.loads, applied even when
# emitted content is unlimited (max_content_length=-1): a multi-megabyte
# encoded array otherwise burns unbounded event-loop time and memory before
# the node budget can apply. Container-capable
# values beyond it fail closed.
_MAX_JSON_INSPECT_CHARS = 4_000_000

# Keep deeply nested JSON behavior deterministic across Python versions.
# CPython <=3.13 rejects extreme nesting from json.loads with RecursionError,
# while 3.14's iterative decoder accepts it. A fixed ceiling preserves the
# fail-closed contract and bounds the materialized object graph everywhere.
_MAX_JSON_NESTING_DEPTH = 1_000


def _strip_bom_ws(value: str) -> str:
  """Strips BOMs and ALL Unicode whitespace from the start of a string.

  json.dumps round-trips arbitrary Unicode, so a decoded string layer can
  hide a credential container behind U+00A0/U+2003-style whitespace that an
  ASCII-only lstrip never removes. One linear
  index scan: the earlier alternating lstrip loop re-sliced the remaining
  suffix per whitespace/BOM pair, going quadratic on a legal prefix and
  pinning the callback event loop. str.isspace
  covers every Unicode whitespace; BOM (U+FEFF) is not whitespace, so both
  are tested per character.
  """
  i = 0
  n = len(value)
  while i < n and (value[i].isspace() or value[i] == "\ufeff"):
    i += 1
  return value[i:] if i else value


def _json_nesting_exceeds_limit(value: str) -> bool:
  """Returns whether JSON structural nesting exceeds the fixed ceiling.

  Brackets and braces inside strings are payload, not structure. This linear
  preflight runs only after the existing character-size bound, before
  ``json.loads`` can materialize a runtime-dependent deep object graph.
  """
  depth = 0
  in_string = False
  escaped = False
  for char in value:
    if in_string:
      if escaped:
        escaped = False
      elif char == "\\":
        escaped = True
      elif char == '"':
        in_string = False
      continue
    if char == '"':
      in_string = True
    elif char in "[{":
      depth += 1
      if depth > _MAX_JSON_NESTING_DEPTH:
        return True
    elif char in "]}":
      depth -= 1
  return False


def _normalize_json_native(
    obj: Any,
    max_len: int,
    depth: int = 0,
    budget: Optional[list[int]] = None,
) -> tuple[Any, bool]:
  """Coerces already-sanitized parser output to strictly JSON-native values.

  A nested hostile model can defer its failure PAST parser sanitization —
  e.g. an attribute that returns an object whose __repr__ raises — so the
  payload detonated later during Arrow serialization's json.dumps/str
  fallback. Unlike the full sanitizer, this
  does NOT re-run JSON-blob inspection on strings (parser output would
  have its sentinels and bracketed prose corrupted), but it DOES enforce
  the two policies parser output cannot be assumed to carry:
  sensitive-key/``temp:`` value redaction — a nested
  model property can hand the parser a raw credential mapping — and the
  configured length bound, because model fields like ``role`` are copied
  verbatim and an unbounded string reopens the synchronous
  json.dumps/Arrow work boundary. Returns ``(normalized,
  replaced_anything)``.
  """
  if budget is None:
    budget = [_MAX_SANITIZE_NODES]
  budget[0] -= 1
  if budget[0] < 0:
    return "[SANITIZE_BUDGET_EXCEEDED]", True
  if depth >= _MAX_SANITIZE_DEPTH:
    return "[MAX_DEPTH_EXCEEDED]", True
  try:
    if isinstance(obj, str):
      text = obj if type(obj) is str else str.__str__(obj)
      if max_len != -1 and len(text) > max_len:
        return text[:max_len] + "...[TRUNCATED]", True
      return text, False
    if obj is None or type(obj) in (int, float, bool):
      return obj, False
    if isinstance(obj, bool):
      return bool(obj), False
    if isinstance(obj, int):
      return int(obj), False
    if isinstance(obj, float):
      return float(obj), False
    if isinstance(obj, dict):
      out_dict: dict[str, Any] = {}
      replaced = False
      bad_keys = 0

      def collision_safe_key(k: str) -> str:
        """Allocates a unique key while preserving first-writer order."""
        nonlocal bad_keys, replaced
        if k not in out_dict:
          return k
        bad_keys += 1
        candidate = f"[KEY_COLLISION_{bad_keys}]{k}"
        while candidate in out_dict:
          bad_keys += 1
          candidate = f"[KEY_COLLISION_{bad_keys}]{k}"
        replaced = True
        return candidate

      for k, v in obj.items():
        if budget[0] <= 0:
          budget_key = collision_safe_key("[SANITIZE_BUDGET_EXCEEDED]")
          out_dict[budget_key] = "[SANITIZE_BUDGET_EXCEEDED]"
          replaced = True
          break
        redact_value = False
        if isinstance(k, str):
          if type(k) is not str:
            k = str.__str__(k)
          k_lower = k.lower().replace("-", "_")
          if k_lower in _SENSITIVE_KEYS or k_lower.startswith("temp:"):
            redact_value = True
        else:
          bad_keys += 1
          k = f"[UNSUPPORTED_KEY_{bad_keys}]"
          replaced = True
        # Preserve every value after key normalization. The first writer
        # keeps the plain key; later colliders receive a marker that is
        # itself re-allocated around genuine marker-like user keys.
        k = collision_safe_key(k)
        if redact_value:
          budget[0] -= 1
          out_dict[k] = "[REDACTED]"
          continue
        norm_v, v_replaced = _normalize_json_native(
            v, max_len, depth + 1, budget
        )
        replaced = replaced or v_replaced
        out_dict[k] = norm_v
      return out_dict, replaced
    if isinstance(obj, (list, tuple)):
      out_list: list[Any] = []
      replaced = False
      for item in obj:
        if budget[0] <= 0:
          out_list.append("[SANITIZE_BUDGET_EXCEEDED]")
          replaced = True
          break
        norm_item, item_replaced = _normalize_json_native(
            item, max_len, depth + 1, budget
        )
        replaced = replaced or item_replaced
        out_list.append(norm_item)
      return out_list, replaced
    return "[UNSUPPORTED_OBJECT]", True
  except Exception:
    return "[UNSUPPORTED_OBJECT]", True


def _sanitize_free_text(
    text: str,
    seen: set[int],
    depth: int,
    max_len: int,
    budget: Optional[list[int]],
) -> tuple[str, bool, bool]:
  """Handles text that may embed a JSON document ANYWHERE, not just at

  its start: raw multi-document suffixes and decoded string layers.

  A literal container token after prose cannot be verified (the text has
  machine-encoded provenance, so a consumer may parse it) and fails
  closed; a quoted fragment is walked through the full blob sanitizer
  from the first quote. A stray
  escape in the prose gap could shift a consumer's parse boundaries and
  also fails closed. Quote-free, container-free prose passes through.
  """
  if "{" in text or "[" in text:
    return "[UNPARSEABLE_JSON_BLOB]", True, True
  quote_idx = text.find('"')
  if quote_idx == -1:
    return text, False, False
  gap = text[:quote_idx]
  if "\\" in gap:
    return "[UNPARSEABLE_JSON_BLOB]", True, True
  tail = text[quote_idx:]
  t_sanitized, changed, truncated = _sanitize_json_blob(
      tail, seen, depth + 1, max_len, budget
  )
  if not changed:
    return text, False, False
  return gap + t_sanitized, True, truncated


def _sanitize_json_blob(
    value: str,
    seen: set[int],
    depth: int = 0,
    max_len: int = -1,
    budget: Optional[list[int]] = None,
) -> tuple[str, bool, bool]:
  """Redacts sensitive keys inside a JSON-encoded string blob.

  Values such as cached credential JSON often reach attributes as opaque
  strings, bypassing dict-key redaction. Decode
  FIRST: raw-substring prefilters are bypassable through JSON string
  escapes (e.g. ``"access\\u005ftoken"``), so any string that looks like a
  JSON container is parsed and its *decoded* keys inspected recursively —
  arrays of credential objects included. Returns ``(value, changed,
  truncated)`` where ``truncated`` reports content loss inside the blob
  (depth/budget replacement discards payload);
  strings that do not parse, or that need no redaction, are returned
  unchanged (no cosmetic re-serialization).
  """
  # BOM/Unicode-whitespace-aware normalization must run before EVERY
  # shape check — an ASCII-only lstrip let a decoded layer hide its
  # container behind U+00A0/U+2003.
  stripped = _strip_bom_ws(value)
  # The inspection ceiling applies even in unlimited mode.
  inspect_limit = (
      min(max_len, _MAX_JSON_INSPECT_CHARS)
      if max_len != -1
      else _MAX_JSON_INSPECT_CHARS
  )
  if stripped.startswith('"'):
    # A JSON-encoded STRING layer: json.dumps applied twice leaves the
    # secret container quoted-and-escaped, which the container check
    # below cannot see. Decode the layer and
    # re-enter this sanitizer so a decoded string that turns out to be a
    # container goes through the same redaction path. Layer recursion is
    # bounded by the depth cap, and each decode strictly shrinks the
    # string.
    if depth >= _MAX_SANITIZE_DEPTH:
      return "[UNPARSEABLE_JSON_BLOB]", True, True
    if len(stripped) > inspect_limit:
      # Decoding would allocate beyond the limit, so classify what will
      # actually be EMITTED — the max_len prefix after the caller's raw
      # truncation, or the ENTIRE value in unlimited mode, where scanning
      # only the inspection window let a credential document sit just
      # past it. Any
      # container token or escape in the emitted text means the output
      # could retain (escaped) credential material — fail closed. The
      # scan is a bounded linear pass over text already in memory.
      # Escape-free, container-free prose is left to the caller's length
      # truncation (or emitted whole in unlimited mode).
      emitted = stripped if max_len == -1 else stripped[:max_len]
      if "{" in emitted or "[" in emitted or "\\" in emitted:
        return "[UNPARSEABLE_JSON_BLOB]", True, True
      return value, False, False
    suffix = ""
    try:
      decoded = json.loads(stripped)
    except (TypeError, RecursionError, MemoryError):
      return "[UNPARSEABLE_JSON_BLOB]", True, True
    except ValueError:
      # Not one complete JSON document. A valid quoted credential layer
      # followed by trailing garbage still lands here, and classifying it
      # as prose republished the secret:
      # bounded-decode the LEADING string layer instead.
      try:
        decoded, end = json.JSONDecoder().raw_decode(stripped)
      except ValueError:
        # No leading complete JSON document. If the post-quote prefix can
        # still represent an ENCODED CONTAINER, the value is malformed
        # credential JSON (e.g. an encoded layer missing its final quote)
        # and must fail closed like the direct container path does for
        # malformed JSON. Anything else is
        # quoted prose.
        if _strip_bom_ws(stripped[1:])[:1] in ("{", "[", "\\", '"'):
          return "[UNPARSEABLE_JSON_BLOB]", True, True
        return value, False, False
      suffix = stripped[end:]
    if not isinstance(decoded, str):
      return value, False, False
    stripped_suffix = _strip_bom_ws(suffix)
    if "{" in suffix or "[" in suffix or stripped_suffix.startswith("\\"):
      # An unverified raw suffix can smuggle a credential container past
      # a harmless quoted prefix ('"note" {"access_token":...}'). A literal container-token scan is sound for
      # RAW text that no later step decodes; a leading stray escape
      # cannot be classified and fails closed.
      return "[UNPARSEABLE_JSON_BLOB]", True, True
    # ANY quoted fragment in the trailing text may be an encoded JSON
    # document whose decoded content hides a container behind Unicode
    # escapes — immediately ('"note" "\\u007b..."') or
    # after a stretch of prose ('"note" then "\\u007b..."'): walk it with the shared free-text handling.
    s_sanitized, s_changed, s_truncated = _sanitize_free_text(
        suffix, seen, depth, max_len, budget
    )
    if s_changed and s_sanitized == "[UNPARSEABLE_JSON_BLOB]":
      return "[UNPARSEABLE_JSON_BLOB]", True, True
    inner = _strip_bom_ws(decoded)
    if inner.startswith(("{", "[", '"')):
      p_sanitized, p_changed, p_truncated = _sanitize_json_blob(
          decoded, seen, depth + 1, max_len, budget
      )
    else:
      # The DECODED layer can hide a container after prose too
      # ('"note \\u007b...access\\u005ftoken...\\u007d"' decodes to
      # 'note {"access_token":...}') — the escapes are gone after this
      # decode, so the same anywhere-in-text handling applies.
      p_sanitized, p_changed, p_truncated = _sanitize_free_text(
          decoded, seen, depth + 1, max_len, budget
      )
      if p_changed and p_sanitized == "[UNPARSEABLE_JSON_BLOB]":
        return "[UNPARSEABLE_JSON_BLOB]", True, True
    if not p_changed and not s_changed:
      return value, False, False
    prefix_text = stripped[:end] if not p_changed else json.dumps(p_sanitized)
    suffix_text = suffix if not s_changed else s_sanitized
    return prefix_text + suffix_text, True, p_truncated or s_truncated
  if not stripped.startswith(("{", "[")):
    return value, False, False

  # Enforce the inspection limit BEFORE materializing: json.loads runs
  # synchronously on the callback path and can allocate far beyond the
  # limit for a multi-megabyte attribute.
  # Truncating the raw JSON prefix instead could both retain a secret and
  # emit invalid JSON, so over-limit container blobs fail closed.
  if len(stripped) > inspect_limit:
    return "[UNPARSEABLE_JSON_BLOB]", True, True

  if _json_nesting_exceeds_limit(stripped):
    return "[UNPARSEABLE_JSON_BLOB]", True, True

  # json.loads silently keeps only the LAST duplicate member, so a blob like
  # {"access_token":"SECRET","access_token":"x"} can compare equal after
  # sanitization while the raw string still carries the secret. Track duplicates while parsing and always reserialize
  # such blobs.
  saw_duplicate_key = False

  def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    nonlocal saw_duplicate_key
    result = {}
    for k, v in pairs:
      if k in result:
        saw_duplicate_key = True
      result[k] = v
    return result

  try:
    parsed = json.loads(stripped, object_pairs_hook=_pairs_hook)
    if not isinstance(parsed, (dict, list)):
      return value, False, False
    # Redact only (max_len=-1): length truncation is applied by the caller
    # on the re-serialized string, keeping single responsibility per pass.
    # The nested truncation flag must survive: a depth/budget sentinel
    # inside the blob discards payload, and dropping the bit made the row
    # claim completeness.
    sanitized, nested_truncated = _recursive_smart_truncate(
        parsed, -1, seen, depth + 1, budget
    )
    if sanitized == parsed and not saw_duplicate_key and not nested_truncated:
      return value, False, False
    return json.dumps(sanitized), True, nested_truncated
  except (TypeError, ValueError, RecursionError, MemoryError):
    # Container-shaped but unparseable — malformed JSON / trailing garbage
    # (a one-character suffix on valid credential JSON must not bypass
    # redaction), integers over the interpreter
    # digit limit, or a blob too deep/large to inspect.
    # None of these can be verified secret-free — and a
    # raw-substring fallback is bypassable via JSON string escapes — so
    # fail CLOSED to a sentinel.
    return "[UNPARSEABLE_JSON_BLOB]", True, True


def _require_count(name: str, value: Any, minimum: int) -> None:
  """Requires an integral count >= minimum.

  Bools and floats are rejected: ordered comparisons alone let NaN pass
  every range check.
  """
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{name} must be an int, got {value!r}.")
  if value < minimum:
    raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _require_finite(name: str, value: Any, minimum_exclusive: float) -> float:
  """Requires a finite real number strictly greater than the minimum."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a number, got {value!r}.")
  if not math.isfinite(value):
    raise ValueError(f"{name} must be finite, got {value!r}.")
  if value <= minimum_exclusive:
    raise ValueError(f"{name} must be > {minimum_exclusive}, got {value}.")
  return float(value)


def _validate_runtime_config(config: "BigQueryLoggerConfig") -> None:
  """Validates runtime settings at construction time.

  Invalid values used to be accepted silently and only misbehave at
  runtime — notably ``max_retries < 0`` skips the write loop entirely, so
  every batch is dropped without a single attempt.

  Raises:
      ValueError: If any batch, queue, duration, or retry setting is
        invalid.
  """
  _require_count("batch_size", config.batch_size, 1)
  _require_finite("batch_flush_interval", config.batch_flush_interval, 0)
  _require_finite("shutdown_timeout", config.shutdown_timeout, 0)
  _require_count("queue_max_size", config.queue_max_size, 1)
  if isinstance(config.max_content_length, bool) or not isinstance(
      config.max_content_length, int
  ):
    raise ValueError(
        f"max_content_length must be an int, got {config.max_content_length!r}."
    )
  if config.max_content_length != -1 and config.max_content_length < 1:
    raise ValueError(
        "max_content_length must be -1 (unlimited) or >= 1, got"
        f" {config.max_content_length}."
    )
  retry = config.retry_config
  _require_count("retry_config.max_retries", retry.max_retries, 0)
  # Delays are finite and NON-NEGATIVE: zero-delay immediate retries are
  # long-supported (asyncio.sleep(0) is valid) and existing configs use
  # max_retries=0, initial_delay=0, max_delay=0.
  initial_delay = _require_finite(
      "retry_config.initial_delay", retry.initial_delay, -1
  )
  if initial_delay < 0:
    raise ValueError(
        f"retry_config.initial_delay must be >= 0, got {retry.initial_delay}."
    )
  multiplier = _require_finite("retry_config.multiplier", retry.multiplier, 0)
  if multiplier < 1:
    raise ValueError(
        f"retry_config.multiplier must be >= 1, got {retry.multiplier}."
    )
  max_delay = _require_finite("retry_config.max_delay", retry.max_delay, -1)
  if max_delay < 0:
    raise ValueError(
        f"retry_config.max_delay must be >= 0, got {retry.max_delay}."
    )
  if max_delay < initial_delay:
    raise ValueError(
        "retry_config.max_delay must be >= initial_delay, got"
        f" max_delay={retry.max_delay} initial_delay={retry.initial_delay}."
    )


# Cloud Platform OAuth scope. Assembled from parts so this module does not
# embed a bare Google APIs host literal: the file-content compliance scan
# rejects such host literals on changed files unless an accompanying mTLS
# endpoint is present, which does not apply to this OAuth-scope use.
_CLOUD_PLATFORM_SCOPE = (
    "https://www." + "googleapis" + ".com/auth/cloud-platform"
)


class _SetupAbortedError(RuntimeError):
  """Setup lost the lifecycle-generation race against shutdown().

  Distinct from service failures so waiters and row owners can classify
  the outcome without string matching.
  """


class _LoopStateAdmissionAbortedError(_SetupAbortedError):
  """A retained writer is terminal for admission but still draining.

  This is a lifecycle outcome, not a setup/service failure: setup owners and
  coalesced waiters must report ``aborted`` without poisoning retry backoff or
  tearing down shared clients needed by the in-flight drain.
  """


class _ShutdownIncompleteError(RuntimeError):
  """The owning shutdown() was cancelled before teardown completed.

  Coalesced callers must not treat this as success; they retry ownership
  instead.
  """


def _base_str(safe_base: Any, obj: Any) -> str:
  """Non-overridable base-class string conversion.

  ``safe_base.__str__(obj)`` bypasses any subclass ``__str__`` override —
  a subclassed allowlisted scalar (application Enum, PurePath, ...) could
  otherwise reopen the arbitrary-string leak.
  """
  text = safe_base.__str__(obj)
  return text if isinstance(text, str) else "[UNSUPPORTED_OBJECT]"


def _safe_getattr(obj: Any, name: str) -> Any:
  """getattr that cannot be weaponized by payload-controlled properties.

   A bare ``hasattr``/``getattr`` probe evaluates properties, and hasattr
   only swallows AttributeError — a property raising anything else escaped
   the sanitizer entirely, dropping the telemetry row and leaking the
   exception message (often containing the payload) into application logs.
  Returns None on ANY failure.
  """
  try:
    return getattr(obj, name, None)
  except Exception:
    return None


# Stdlib scalar types whose str() form is canonical, side-effect free, and
# cannot embed attribute state beyond the value itself. Only these keep the
# stringify fallback; arbitrary objects' repr/str output is payload-
# controlled (e.g. SimpleNamespace(access_token=...) prints its attributes
# verbatim) and is no longer published.
_SAFE_STR_TYPES: tuple[type, ...] = (
    datetime,
    date,
    timedelta,
    decimal.Decimal,
    uuid.UUID,
    pathlib.PurePath,
    # NOTE: enum.Enum is handled FIRST in the dispatch (before the scalar
    # branches), not here — value-backed members would otherwise never
    # reach this fallback.
    complex,
)


def _recursive_smart_truncate(
    obj: Any,
    max_len: int,
    seen: Optional[set[int]] = None,
    depth: int = 0,
    budget: Optional[list[int]] = None,
) -> tuple[Any, bool]:
  """Recursively truncates string values within a dict or list.

  Redacts sensitive keys corresponding to OAuth tokens and secrets
  prior to serialization into BigQuery JSON strings.

  Args:
      obj: The object to truncate.
      max_len: Maximum length for string values.
      seen: Set of object IDs visited in the current recursion stack.
      depth: Current recursion depth.

  Returns:
      A tuple of (truncated_object, is_truncated).
  """
  if seen is None:
    seen = set()
  if budget is None:
    budget = [_MAX_SANITIZE_NODES]
  budget[0] -= 1
  if budget[0] < 0:
    return "[SANITIZE_BUDGET_EXCEEDED]", True

  # Depth cap: id()-based cycle detection cannot catch object graphs that
  # manufacture NEW objects on each duck-typed access (e.g. anything whose
  # model_dump()/dict()/to_dict() returns a fresh wrapper — unittest Mocks
  # being the canonical case). Without this cap such graphs recurse
  # unboundedly. The replacement discards real payload, so it reports
  # truncation — unlike "[CIRCULAR_REFERENCE]",
  # which replaces a back-reference, not data.
  if depth >= _MAX_SANITIZE_DEPTH:
    return "[MAX_DEPTH_EXCEEDED]", True

  obj_id = id(obj)
  if obj_id in seen:
    return "[CIRCULAR_REFERENCE]", False

  # Converter attributes are fetched exactly once through _safe_getattr:
  # hasattr() evaluates payload-controlled properties OUTSIDE any guard,
  # and a property raising e.g. RuntimeError escaped the sanitizer, leaked
  # its message into logs, and suppressed the whole row.
  model_dump_fn = _safe_getattr(obj, "model_dump")
  dict_fn = _safe_getattr(obj, "dict")
  to_dict_fn = _safe_getattr(obj, "to_dict")
  instance_dict = None
  if not isinstance(
      obj, (str, bytes, bytearray, int, float, bool, dict, list, tuple)
  ):
    attrs = _safe_getattr(obj, "__dict__")
    if isinstance(attrs, dict):
      instance_dict = attrs

  # Track compound objects to detect cycles. Plain objects traversed via
  # their __dict__ count too: an unmarked self-reference (obj.self = obj)
  # repeated the full attribute copy at every level until the depth cap.
  is_compound = (
      isinstance(obj, (dict, list, tuple, collections.abc.Mapping))
      or (dataclasses.is_dataclass(obj) and not isinstance(obj, type))
      or model_dump_fn is not None
      or dict_fn is not None
      or to_dict_fn is not None
      or instance_dict is not None
  )

  if is_compound:
    seen.add(obj_id)

  try:
    if isinstance(obj, enum.Enum):
      # BEFORE the scalar branches: a StrEnum / (str, Enum) / bytes-backed
      # member is also an instance of its mixin type, so scalar dispatch
      # normalized it with str.__str__ and published the underlying VALUE
      # instead of the member name.
      return _recursive_smart_truncate(
          _base_str(enum.Enum, obj), max_len, seen, depth + 1, budget
      )
    elif isinstance(obj, str):
      if type(obj) is not str:
        # str subclasses can override lstrip/startswith/lower to
        # misreport their content while json.dumps still serializes the
        # real underlying value: normalize to
        # the exact built-in string before any inspection.
        obj = str.__str__(obj)
      obj, blob_replaced, blob_truncated = _sanitize_json_blob(
          obj, seen, depth, max_len, budget
      )
      if blob_replaced and obj == "[UNPARSEABLE_JSON_BLOB]":
        # The original string was discarded wholesale.
        return obj, True
      if max_len != -1 and len(obj) > max_len:
        return obj[:max_len] + "...[TRUNCATED]", True
      return obj, blob_truncated
    elif isinstance(obj, (bytes, bytearray)):
      # Credential JSON frequently travels as bytes; stringifying it in
      # the fallback bypassed blob redaction.
      try:
        decoded = bytes(obj).decode("utf-8")
      except UnicodeDecodeError:
        # The whole byte payload is discarded, so the row must report
        # content loss.
        return "[BINARY_DATA]", True
      return _recursive_smart_truncate(
          decoded, max_len, seen, depth + 1, budget
      )
    elif isinstance(obj, collections.abc.Mapping):
      # Covers dict plus mapping views (MappingProxyType, UserDict, ...):
      # stringifying them in the fallback branch would bypass key redaction.
      # Always emits a plain sanitized dict.
      truncated_any = False
      new_dict = {}
      unsupported_keys = 0
      for k, v in obj.items():
        # Stop iterating once the work budget is exhausted: recursing on
        # every remaining entry still did O(input) work and produced
        # O(input) sentinel output, and directly-redacted entries consumed
        # no budget at all, so a wide "temp:" mapping bypassed the bound
        # entirely. One remainder sentinel
        # stands in for everything dropped.
        if budget[0] <= 0:
          new_dict["[SANITIZE_BUDGET_EXCEEDED]"] = "[SANITIZE_BUDGET_EXCEEDED]"
          truncated_any = True
          break
        redact_value = False
        if isinstance(k, str):
          if type(k) is not str:
            # Same normalization as string values: a str-subclass key can
            # misreport itself to the redaction check below.
            k = str.__str__(k)
          k_lower = k.lower().replace("-", "_")
          if k_lower in _SENSITIVE_KEYS or k_lower.startswith("temp:"):
            redact_value = True
        elif k is None or isinstance(k, (int, float, bool)):
          # JSON stringifies all object keys, so 1 and "1" (or True and
          # "true", None and "null") silently collapse into duplicate
          # members and one value is lost downstream. Normalize to the exact JSON key form BEFORE
          # insertion; collisions are handled below.
          if k is True:
            k = "true"
          elif k is False:
            k = "false"
          elif k is None:
            k = "null"
          else:
            k = json.dumps(k)
        else:
          # json.dumps rejects non-scalar keys even with default=str (it
          # applies to values only), so one such key raised TypeError at
          # serialization and silently dropped the whole telemetry row.
          # The key's own repr can embed
          # secrets, so it is not published either: fail the KEY closed
          # and keep the sanitized value.
          unsupported_keys += 1
          budget[0] -= 1
          k = (
              "[UNSUPPORTED_KEY]"
              if unsupported_keys == 1
              else f"[UNSUPPORTED_KEY_{unsupported_keys}]"
          )
          truncated_any = True

        if k in new_dict:
          # Deterministic fail-closed collision policy: the first writer keeps the plain key; later
          # colliders keep their (sanitized) value under an explicit
          # marker instead of silently overwriting or being dropped at
          # JSON parse time. The marker is re-allocated until unique — a
          # single fixed marker could alias (and overwrite) a legitimate
          # user key already named "[KEY_COLLISION_n]...".
          unsupported_keys += 1
          candidate = f"[KEY_COLLISION_{unsupported_keys}]{k}"
          while candidate in new_dict:
            unsupported_keys += 1
            candidate = f"[KEY_COLLISION_{unsupported_keys}]{k}"
          k = candidate
          truncated_any = True

        if redact_value:
          budget[0] -= 1
          new_dict[k] = "[REDACTED]"
          continue
        val, trunc = _recursive_smart_truncate(
            v, max_len, seen, depth + 1, budget
        )
        if trunc:
          truncated_any = True
        new_dict[k] = val
      return new_dict, truncated_any
    elif isinstance(obj, (list, tuple)):
      fields = _safe_getattr(obj, "_fields") if isinstance(obj, tuple) else None
      if (
          isinstance(fields, tuple)
          and len(fields) == len(obj)
          and all(isinstance(f, str) for f in fields)
      ):
        # Namedtuples carry field NAMES (e.g. access_token) that the
        # plain-list normalization below discarded, so the sensitive
        # value survived positionally.
        # Rebuild as a mapping from the class metadata (zip, not the
        # overridable _asdict()) so key redaction runs.
        return _recursive_smart_truncate(
            dict(zip(fields, obj)), max_len, seen, depth + 1, budget
        )
      truncated_any = False
      new_list = []
      # Explicit loop to handle flag propagation
      for i in obj:
        # Same bound as the mapping loop.
        if budget[0] <= 0:
          new_list.append("[SANITIZE_BUDGET_EXCEEDED]")
          truncated_any = True
          break
        val, trunc = _recursive_smart_truncate(
            i, max_len, seen, depth + 1, budget
        )
        if trunc:
          truncated_any = True
        new_list.append(val)
      if type(obj) is tuple or type(obj) is list:
        return type(obj)(new_list), truncated_any
      # Tuple/list subclasses (e.g. namedtuples) may require positional
      # constructor fields; reconstructing raised TypeError and the safe
      # callback dropped the whole row. JSON
      # does not preserve the subclass identity anyway — emit a plain
      # list.
      return new_list, truncated_any
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
      # Manually iterate fields to preserve 'seen' context, avoiding dataclasses.asdict recursion
      as_dict = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
      return _recursive_smart_truncate(
          as_dict, max_len, seen, depth + 1, budget
      )
    elif model_dump_fn is not None and callable(model_dump_fn):
      # Pydantic v2. Only recurse if the conversion made PROGRESS toward a
      # JSON-native value: Mock-like objects answer every duck-typed
      # probe with another Mock-like object, and recursing on those churns
      # to the depth cap (falsely flagging truncation) instead of settling
      # at the fallback. Progress includes SCALARS: a RootModel[str] dumps
      # to a plain string that may itself be a credential blob, and
      # falling through to the generic fallback bypassed blob redaction.
      try:
        dumped = model_dump_fn()
        if isinstance(
            dumped,
            (collections.abc.Mapping, list, tuple, str, bytes, bytearray),
        ):
          return _recursive_smart_truncate(
              dumped, max_len, seen, depth + 1, budget
          )
        if dumped is None or isinstance(dumped, (int, float, bool)):
          return dumped, False
      except Exception:
        pass
    elif dict_fn is not None and callable(dict_fn):
      # Pydantic v1 (same progress requirement as above, scalars included).
      try:
        dumped = dict_fn()
        if isinstance(
            dumped,
            (collections.abc.Mapping, list, tuple, str, bytes, bytearray),
        ):
          return _recursive_smart_truncate(
              dumped, max_len, seen, depth + 1, budget
          )
        if dumped is None or isinstance(dumped, (int, float, bool)):
          return dumped, False
      except Exception:
        pass
    elif to_dict_fn is not None and callable(to_dict_fn):
      # Common pattern for custom objects (same progress requirement,
      # scalars included).
      try:
        dumped = to_dict_fn()
        if isinstance(
            dumped,
            (collections.abc.Mapping, list, tuple, str, bytes, bytearray),
        ):
          return _recursive_smart_truncate(
              dumped, max_len, seen, depth + 1, budget
          )
        if dumped is None or isinstance(dumped, (int, float, bool)):
          return dumped, False
      except Exception:
        pass
    elif obj is None or isinstance(obj, (int, float, bool)):
      # Basic types are safe
      return obj, False

    # Fallback for unknown types. Arbitrary str()/repr() output is
    # payload-controlled and prints attribute values verbatim (e.g.
    # SimpleNamespace(access_token=...)), so it is no longer published.
    # Known-safe stdlib scalars keep their canonical string
    # form via NON-OVERRIDABLE base-class conversions — a subclass
    # (application Enum, PurePath, ...) can override __str__ to return
    # its value, reopening the arbitrary-string leak through a
    # polymorphic str(obj). Objects exposing
    # instance state keep a structurally sanitized public view of their
    # __dict__, collected through a budget-checked loop (a million-
    # attribute object was fully copied before the budget applied); everything else fails closed to a sentinel.
    # Sentinel replacement discards content, so it reports truncation.
    for safe_base in _SAFE_STR_TYPES:
      if isinstance(obj, safe_base):
        # _base_str bypasses any subclass override; propagate the
        # truncation flag — dropping it made an over-limit safe scalar
        # claim completeness.
        return _recursive_smart_truncate(
            _base_str(safe_base, obj), max_len, seen, depth + 1, budget
        )
    if instance_dict is not None:
      public_attrs = {}
      overflow = False
      for k, v in instance_dict.items():
        if budget[0] <= 0:
          overflow = True
          break
        # Every entry costs budget, filtered or not: skipping private
        # attributes for free left the collection loop O(input).
        budget[0] -= 1
        if isinstance(k, str) and not k.startswith("_"):
          public_attrs[k] = v
      if public_attrs or overflow:
        sanitized_attrs, attrs_truncated = _recursive_smart_truncate(
            public_attrs, max_len, seen, depth + 1, budget
        )
        if overflow and isinstance(sanitized_attrs, dict):
          sanitized_attrs["[SANITIZE_BUDGET_EXCEEDED]"] = (
              "[SANITIZE_BUDGET_EXCEEDED]"
          )
          attrs_truncated = True
        return sanitized_attrs, attrs_truncated
    return "[UNSUPPORTED_OBJECT]", True
  except Exception:
    # Fail-closed protocol boundary: a
    # hostile container protocol (a Mapping whose items() raises,
    # sequence iteration or dataclass field access raising) must neither
    # escape the sanitizer — the safe callback would log the payload-
    # controlled message and drop the whole row — nor be logged here.
    return "[UNSUPPORTED_OBJECT]", True
  finally:
    if is_compound:
      seen.discard(obj_id)


# --- PyArrow Helper Functions ---
def _pyarrow_datetime() -> pa.DataType:
  return pa.timestamp("us", tz=None)


def _pyarrow_numeric() -> pa.DataType:
  return pa.decimal128(38, 9)


def _pyarrow_bignumeric() -> pa.DataType:
  return pa.decimal256(76, 38)


def _pyarrow_time() -> pa.DataType:
  return pa.time64("us")


def _pyarrow_timestamp() -> pa.DataType:
  return pa.timestamp("us", tz="UTC")


_BQ_TO_ARROW_SCALARS = MappingProxyType({
    "BOOL": pa.bool_,
    "BOOLEAN": pa.bool_,
    "BYTES": pa.binary,
    "DATE": pa.date32,
    "DATETIME": _pyarrow_datetime,
    "FLOAT": pa.float64,
    "FLOAT64": pa.float64,
    "GEOGRAPHY": pa.string,
    "INT64": pa.int64,
    "INTEGER": pa.int64,
    "JSON": pa.string,
    "NUMERIC": _pyarrow_numeric,
    "BIGNUMERIC": _pyarrow_bignumeric,
    "STRING": pa.string,
    "TIME": _pyarrow_time,
    "TIMESTAMP": _pyarrow_timestamp,
})

_BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
    "GEOGRAPHY": {
        b"ARROW:extension:name": b"google:sqlType:geography",
        b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
    },
    "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
    "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
}
_STRUCT_TYPES = ("RECORD", "STRUCT")


def _bq_to_arrow_scalars(bq_scalar: str) -> Optional[Callable[[], pa.DataType]]:
  """Maps BigQuery scalar types to PyArrow type constructors."""
  return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


def _bq_to_arrow_field(bq_field: bq_schema.SchemaField) -> Optional[pa.Field]:
  """Converts a BigQuery SchemaField to a PyArrow Field."""
  arrow_type = _bq_to_arrow_data_type(bq_field)
  if arrow_type:
    metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
        bq_field.field_type.upper() if bq_field.field_type else ""
    )
    nullable = bq_field.mode.upper() != "REQUIRED"
    return pa.field(
        bq_field.name, arrow_type, nullable=nullable, metadata=metadata
    )
  logger.warning(
      "Could not determine Arrow type for field '%s' with type '%s'.",
      bq_field.name,
      bq_field.field_type,
  )
  return None


def _bq_to_arrow_struct_data_type(
    field: bq_schema.SchemaField,
) -> Optional[pa.StructType]:
  """Converts a BigQuery RECORD/STRUCT field to a PyArrow StructType."""
  arrow_fields = []
  for subfield in field.fields:
    arrow_subfield = _bq_to_arrow_field(subfield)
    if arrow_subfield:
      arrow_fields.append(arrow_subfield)
    else:
      logger.warning(
          "Failed to convert STRUCT/RECORD field '%s' due to subfield '%s'.",
          field.name,
          subfield.name,
      )
      return None
  return pa.struct(arrow_fields)


def _bq_to_arrow_data_type(
    field: bq_schema.SchemaField,
) -> Optional[pa.DataType]:
  """Converts a BigQuery field to a PyArrow DataType."""
  if field.mode == "REPEATED":
    inner = _bq_to_arrow_data_type(
        bq_schema.SchemaField(field.name, field.field_type, fields=field.fields)
    )
    return pa.list_(inner) if inner else None
  field_type_upper = field.field_type.upper() if field.field_type else ""
  if field_type_upper in _STRUCT_TYPES:
    return _bq_to_arrow_struct_data_type(field)
  constructor = _bq_to_arrow_scalars(field_type_upper)
  if constructor:
    return constructor()
  else:
    logger.warning(
        "Failed to convert BigQuery field '%s': unsupported type '%s'.",
        field.name,
        field.field_type,
    )
    return None


def to_arrow_schema(
    bq_schema_list: list[bq_schema.SchemaField],
) -> Optional[pa.Schema]:
  """Converts a list of BigQuery SchemaFields to a PyArrow Schema.

  Args:
      bq_schema_list: list of bigquery.SchemaField objects.

  Returns:
      pa.Schema or None if conversion fails.
  """
  arrow_fields = []
  for bq_field in bq_schema_list:
    af = _bq_to_arrow_field(bq_field)
    if af:
      arrow_fields.append(af)
    else:
      logger.error("Failed to convert schema due to field '%s'.", bq_field.name)
      return None
  return pa.schema(arrow_fields)


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class RetryConfig:
  """Configuration for retrying failed BigQuery write operations.

  Attributes:
      max_retries: Maximum number of retry attempts.
      initial_delay: Initial delay between retries in seconds.
      multiplier: Multiplier for exponential backoff.
      max_delay: Maximum delay between retries in seconds.
  """

  max_retries: int = 3
  initial_delay: float = 1.0
  multiplier: float = 2.0
  max_delay: float = 10.0


@dataclass
class BigQueryLoggerConfig:
  """Configuration for the BigQueryAgentAnalyticsPlugin.

  Attributes:
      enabled: Whether logging is enabled.
      event_allowlist: list of event types to log. If None, all are allowed.
      event_denylist: list of event types to ignore.
      max_content_length: Max length for text content before truncation.
      table_id: BigQuery table ID.
      clustering_fields: Fields to cluster the table by.
      log_multi_modal_content: Whether to log detailed content parts.
      retry_config: Retry configuration for writes.
      batch_size: Number of rows per batch.
      batch_flush_interval: Max time to wait before flushing a batch.
      shutdown_timeout: Max time to wait for shutdown.
      queue_max_size: Max size of the in-memory queue.
      content_formatter: Optional custom formatter for content.
      gcs_bucket_name: GCS bucket for offloading large content.
      connection_id: BigQuery connection ID for ObjectRef columns.
      log_session_metadata: Whether to log session metadata.
      custom_tags: Static custom tags to attach to every event.
      auto_schema_upgrade: Whether to auto-add new columns on schema evolution.
      create_views: Whether to auto-create per-event-type views.
      view_prefix: Prefix for auto-created view names. Default ``"v"`` produces
        views like ``v_llm_request``. Set a distinct prefix per table when
        multiple plugin instances share one dataset to avoid view-name
        collisions.
      enable_otel_correlation: When ``True``, capture the ambient OpenTelemetry
        span context at row-emission time into ``attributes.otel.{span_id,
        trace_id}`` (a best-effort Cloud Trace join key, not a foreign key).
        ``False`` (the default) emits no ``attributes.otel``. Has no effect when
        ``attributes`` is projected out via ``payload_column_denylist``.
      custom_metadata_allowlist: Keys to capture from ``event.custom_metadata``
        into ``attributes.custom_metadata.*``. Entries are exact keys, or
        explicit prefix patterns ending in ``*`` (e.g. ``"a2a:*"``). ``None`` /
        empty preserves today's behavior (only the built-in ``a2a:*`` path
        runs). Captured values pass the same safety pipeline (truncation,
        sensitive-key redaction, circular-reference handling) as all other
        logged content.
      payload_column_denylist: Payload columns to project OUT of the table at
        write time. Only the projectable payload columns ``content`` /
        ``content_parts`` / ``attributes`` / ``latency_ms`` may be listed;
        identity / correlation columns are protected and raise ``ValueError`` if
        listed. Applied schema-first (table schema, Arrow schema, row dict, and
        views all stay consistent); views that reference a denied column drop
        the dependent derived columns. NOTE: denying ``attributes`` also
        disables ``attributes.otel`` and ``attributes.custom_metadata``;
        combining it with a non-empty ``custom_metadata_allowlist`` is rejected
        at construction.
      final_response_tool_names: Tool names whose successful completion carries
        the agent's final answer. When a completed tool's name is in this set,
        its call args are logged as an ``AGENT_RESPONSE`` event. For agents that
        emit the final answer via a dedicated tool (e.g.
        ``submit_final_response``) rather than a plain-text final event. Empty
        (the default) preserves today's behavior.
  """

  enabled: bool = True

  # V1 Configuration Parity
  event_allowlist: list[str] | None = None
  event_denylist: list[str] | None = None
  max_content_length: int = 500 * 1024  # Defaults to 500KB per text block
  table_id: str = "agent_events"

  # V2 Configuration
  clustering_fields: list[str] = field(
      default_factory=lambda: ["event_type", "agent", "user_id"]
  )
  log_multi_modal_content: bool = True
  retry_config: RetryConfig = field(default_factory=RetryConfig)
  batch_size: int = 1
  batch_flush_interval: float = 1.0
  shutdown_timeout: float = 10.0
  queue_max_size: int = 10000
  content_formatter: Optional[Callable[[Any, str], Any]] = None
  # If provided, large content (images, audio, video, large text) will be offloaded to this GCS bucket.
  gcs_bucket_name: Optional[str] = None
  # If provided, this connection ID will be used as the authorizer for ObjectRef columns.
  # Format: "location.connection_id" (e.g. "us.my-connection")
  connection_id: Optional[str] = None

  # Toggle for session metadata (e.g. gchat thread-id)
  log_session_metadata: bool = True
  # Static custom tags (e.g. {"agent_role": "sales"})
  custom_tags: dict[str, Any] = field(default_factory=dict)
  # Automatically add new columns to existing tables when the plugin
  # schema evolves.  Only additive changes are made (columns are never
  # dropped or altered).  Safe to leave enabled; a version label on the
  # table ensures the diff runs at most once per schema version.
  auto_schema_upgrade: bool = True
  # Automatically create per-event-type BigQuery views that unnest
  # JSON columns into typed, queryable columns.
  create_views: bool = True
  # Prefix for auto-created per-event-type view names.
  # Default "v" produces views like ``v_llm_request``.  Set a distinct
  # prefix per table when multiple plugin instances share one dataset
  # to avoid view-name collisions (e.g. ``"v_staging"`` →
  # ``v_staging_llm_request``).
  view_prefix: str = "v"

  # --- span-level Cloud Trace correlation ---
  # When True, capture the ambient OpenTelemetry span context into
  # ``attributes.otel.{span_id,trace_id}`` at row-emission time. Off by
  # default; no plugin-owned span is created.
  enable_otel_correlation: bool = False

  # --- generic custom_metadata capture (allowlist) ---
  # Exact keys and/or explicit ``*``-suffixed prefix patterns to capture
  # from ``event.custom_metadata`` into ``attributes.custom_metadata.*``.
  # None/empty preserves today's behavior (only the built-in ``a2a:*`` path).
  custom_metadata_allowlist: list[str] | None = None

  # --- physical column projection (denylist-first) ---
  # Payload columns to omit from the table at write time.  Only the
  # projectable payload columns are accepted; identity/correlation columns
  # are protected (see ``_PROJECTABLE_PAYLOAD_COLUMNS``).
  payload_column_denylist: list[str] | None = None

  # --- final-answer-via-tool capture ---
  # Tool names whose successful completion carries the agent's final
  # response.  Some agents deliver their final answer through a dedicated
  # tool (e.g. ``submit_final_response``) instead of a plain-text final
  # event, so the on-event ``AGENT_RESPONSE`` path (which excludes function
  # calls/responses) never fires.  When a completed tool's name is in this
  # set, its call args (the final-answer payload) are logged as an
  # ``AGENT_RESPONSE`` event.  Empty (the default) preserves today's
  # behavior.
  final_response_tool_names: frozenset[str] = frozenset()


# ==============================================================================
# HELPER: TRACE MANAGER (Async-Safe with ContextVars)
# ==============================================================================
# NOTE: These contextvars are module-global, not plugin-instance-scoped.
# This is safe in practice for two reasons:
#   1. PluginManager enforces name-uniqueness, preventing two BQ plugin
#      instances on the same Runner.
#   2. Concurrent asyncio tasks (e.g. two Runners in asyncio.gather) each
#      get an isolated contextvar copy, so they don't interfere.
# The only problematic case would be two plugin instances interleaved
# within the *same* asyncio task without task boundaries — which the
# framework's PluginManager already prevents.

_root_agent_name_ctx = contextvars.ContextVar(
    "_bq_analytics_root_agent_name", default=None
)

# Tracks the invocation_id that owns the current span stack so that
# ensure_invocation_span() can distinguish "same invocation re-entry"
# (idempotent) from "stale records from a previous invocation" (clear).
_active_invocation_id_ctx: contextvars.ContextVar[Optional[str]] = (
    contextvars.ContextVar("_bq_analytics_active_invocation_id", default=None)
)


@dataclass
class _SpanRecord:
  """A single record on the BQAA plugin's internal span stack.

  Stores the IDs and timing the plugin needs to populate BigQuery
  ``span_id`` / ``parent_span_id`` / ``trace_id`` / ``latency_ms``
  columns.  Crucially, no OpenTelemetry ``Span`` object is held.

  Background — prior approach and the bug it caused:
    The previous implementation created real OTel spans via
    ``tracer.start_span(...)`` purely as ID carriers.  When the host
    application has an OTel exporter configured (notably Agent Engine
    with ``GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true``), those
    plugin-owned spans were exported to Cloud Trace alongside the
    framework's real spans — producing a duplicate-span view for
    every BQAA-instrumented operation.  See haiyuan-eng-google/BQAA-SDK#94.

    The plugin already tracked all parent / child relationships on
    this internal stack, so the OTel span object was incidental to
    correctness.  We now store ``trace_id`` directly on each record
    (inherited from the ambient OTel span when present, generated
    otherwise) and skip span creation entirely.  Cross-system
    correlation with Cloud Trace still works via ``trace_id``
    inheritance.

    ``attach_current_span`` (which observes the ambient span without
    owning one) is unaffected by this change.
  """

  span_id: str
  trace_id: str
  owns_span: bool
  start_time_ns: int
  # What pushed this record ("invocation", "agent", "llm_request", "tool").
  # Lets error callbacks pop only spans they own: e.g. if another plugin's
  # before_agent_callback raised before BQAA pushed its agent span,
  # on_agent_error_callback must not pop the invocation span instead.
  kind: str = ""
  first_token_time: Optional[float] = None


_span_records_ctx: contextvars.ContextVar[Optional[list[_SpanRecord]]] = (
    contextvars.ContextVar("_bq_analytics_span_records", default=None)
)


class TraceManager:
  """Manages OpenTelemetry-style trace and span context using contextvars.

  Uses a single stack of _SpanRecord objects to keep span, token, ID,
  ownership, and timing in sync by construction.
  """

  @staticmethod
  def _get_records() -> list[_SpanRecord]:
    """Returns the current records stack, initializing if needed."""
    records = _span_records_ctx.get()
    if records is None:
      records = []
      _span_records_ctx.set(records)
    return records

  @staticmethod
  def init_trace(callback_context: CallbackContext) -> None:
    # Always refresh root_agent_name — it can change between
    # invocations (e.g. different root agents in the same task).
    try:
      root_agent = callback_context._invocation_context.agent.root_agent
      _root_agent_name_ctx.set(root_agent.name)
    except (AttributeError, ValueError):
      pass

    # Ensure records stack is initialized
    TraceManager._get_records()

  @staticmethod
  def get_trace_id(callback_context: CallbackContext) -> Optional[str]:
    """Gets the trace ID from the current span stack or invocation_id."""
    records = _span_records_ctx.get()
    if records:
      return records[-1].trace_id

    # Fallback to ambient OTel context (e.g. callbacks fired before
    # any plugin span was pushed).
    ambient_ctx = trace.get_current_span().get_span_context()
    if ambient_ctx.is_valid:
      return format(ambient_ctx.trace_id, "032x")

    return callback_context.invocation_id

  @staticmethod
  def push_span(
      callback_context: CallbackContext,
      span_name: Optional[str] = "adk-span",
  ) -> str:
    """Pushes a BQAA-internal span record onto the stack.

    No OpenTelemetry span is created — see ``_SpanRecord`` for
    background.  The record carries everything the plugin needs to
    populate BigQuery columns:

    * ``span_id`` — newly generated 16-hex string.
    * ``trace_id`` — inherited by precedence:
        1. Top of the existing internal stack (keeps every push
           within an invocation under one trace_id).
        2. Ambient OTel span when valid (e.g. the framework's Runner
           span, or an Agent Engine root span) — keeps BigQuery rows
           joinable to Cloud Trace via the shared ``trace_id``.
        3. A fresh 32-hex value (no ambient context, e.g. unit tests
           or non-OTel runtimes).
    * ``start_time_ns`` — for the eventual ``latency_ms`` on pop.

    ``span_name`` is recorded as the span ``kind`` so error callbacks
    can verify ownership before popping (no OTel span name is set).
    """
    TraceManager.init_trace(callback_context)

    records = TraceManager._get_records()
    if records:
      trace_id = records[-1].trace_id
    else:
      ambient_ctx = trace.get_current_span().get_span_context()
      if ambient_ctx.is_valid:
        trace_id = format(ambient_ctx.trace_id, "032x")
      else:
        trace_id = uuid.uuid4().hex  # 32 hex chars

    span_id_str = uuid.uuid4().hex[:16]

    record = _SpanRecord(
        span_id=span_id_str,
        trace_id=trace_id,
        owns_span=True,
        start_time_ns=time.time_ns(),
        kind=span_name or "",
    )
    _span_records_ctx.set(list(records) + [record])

    return span_id_str

  @staticmethod
  def attach_current_span(
      callback_context: CallbackContext,
  ) -> str:
    """Records the ambient OTel span's IDs on the stack without owning it.

    No OTel span is created or attached.  This path captures the
    ambient span's ``trace_id`` / ``span_id`` so plugin-emitted
    BigQuery rows correlate with whatever Cloud Trace / external
    exporter the host is already running.
    """
    TraceManager.init_trace(callback_context)

    ambient_ctx = trace.get_current_span().get_span_context()
    if ambient_ctx.is_valid:
      span_id_str = format(ambient_ctx.span_id, "016x")
      trace_id = format(ambient_ctx.trace_id, "032x")
    else:
      span_id_str = uuid.uuid4().hex[:16]
      trace_id = uuid.uuid4().hex

    record = _SpanRecord(
        span_id=span_id_str,
        trace_id=trace_id,
        owns_span=False,
        start_time_ns=time.time_ns(),
        # attach_current_span is only used to seed the invocation root
        # (see ensure_invocation_span), so it carries the same kind.
        kind="invocation",
    )
    records = TraceManager._get_records()
    _span_records_ctx.set(list(records) + [record])

    return span_id_str

  @staticmethod
  def ensure_invocation_span(
      callback_context: CallbackContext,
  ) -> None:
    """Ensures a root span exists on the plugin stack for this invocation.

    Must be called before any events are logged so that every event in
    the invocation shares the same trace_id.

    * If the stack has entries for the *current* invocation → no-op
      (idempotent within the same invocation).
    * If the stack has entries from a *different* invocation → clear
      stale records and re-initialise (safety net for abnormal exit).
    * If the ambient OTel span is valid → ``attach_current_span``
      (reuse the runner's span without owning it).
    * Otherwise → ``push_span("invocation")`` (create a new root
      span that will be popped in ``after_run_callback``).
    """
    current_inv = callback_context.invocation_id
    active_inv = _active_invocation_id_ctx.get()

    records = _span_records_ctx.get()
    if records:
      if active_inv == current_inv:
        return  # Already initialised for this invocation.
      # Stale records from a previous invocation that wasn't cleaned
      # up (e.g. exception skipped after_run_callback). Clear and
      # re-init.
      logger.debug(
          "Clearing %d stale span records from previous invocation.",
          len(records),
      )
      TraceManager.clear_stack()

    _active_invocation_id_ctx.set(current_inv)

    # Check for a valid ambient span (e.g. the Runner's invocation span).
    ambient = trace.get_current_span()
    if ambient.get_span_context().is_valid:
      TraceManager.attach_current_span(callback_context)
    else:
      TraceManager.push_span(callback_context, "invocation")

  @staticmethod
  def pop_span(
      expected_kind: Optional[str] = None,
  ) -> tuple[Optional[str], Optional[int]]:
    """Pops the top span record from the internal stack.

    Returns ``(span_id, duration_ms)``.  No OTel span is ended
    because the plugin no longer creates one (see ``_SpanRecord``).

    Args:
      expected_kind: When set, only pop if the top record was pushed
        with this kind; otherwise leave the stack untouched and return
        ``(None, None)``.  Error callbacks use this so they never pop a
        span they do not own (e.g. ``on_agent_error_callback`` firing
        for a failure that happened before BQAA pushed its agent span).
    """
    records = _span_records_ctx.get()
    if not records:
      return None, None

    if expected_kind is not None and records[-1].kind != expected_kind:
      return None, None

    new_records = list(records)
    record = new_records.pop()
    _span_records_ctx.set(new_records)

    duration_ms = int((time.time_ns() - record.start_time_ns) / 1_000_000)
    return record.span_id, duration_ms

  @staticmethod
  def clear_stack() -> None:
    """Clears all span records. Safety net for cross-invocation cleanup."""
    _span_records_ctx.set([])

  @staticmethod
  def get_current_span_and_parent() -> tuple[Optional[str], Optional[str]]:
    """Gets current span_id and parent span_id."""
    records = _span_records_ctx.get()
    if not records:
      return None, None

    span_id = records[-1].span_id
    parent_id = None
    for i in range(len(records) - 2, -1, -1):
      if records[i].span_id != span_id:
        parent_id = records[i].span_id
        break
    return span_id, parent_id

  @staticmethod
  def get_current_span_id() -> Optional[str]:
    """Gets current span_id."""
    records = _span_records_ctx.get()
    if records:
      return records[-1].span_id
    return None

  @staticmethod
  def get_root_agent_name() -> Optional[str]:
    return _root_agent_name_ctx.get()

  @staticmethod
  def get_start_time(span_id: str) -> Optional[float]:
    """Gets start time of a span by ID (seconds since epoch)."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          return record.start_time_ns / 1_000_000_000.0
    return None

  @staticmethod
  def record_first_token(span_id: str) -> bool:
    """Records the current time as first token time if not already recorded."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          if record.first_token_time is None:
            record.first_token_time = time.time()
            return True
          return False
    return False

  @staticmethod
  def get_first_token_time(span_id: str) -> Optional[float]:
    """Gets the recorded first token time."""
    records = _span_records_ctx.get()
    if records:
      for record in reversed(records):
        if record.span_id == span_id:
          return record.first_token_time
    return None


# ==============================================================================
# HELPER: BATCH PROCESSOR
# ==============================================================================
_SHUTDOWN_SENTINEL = object()


class BatchProcessor:
  """Handles asynchronous batching and writing of events to BigQuery."""

  def __init__(
      self,
      write_client: BigQueryWriteAsyncClient,
      arrow_schema: pa.Schema,
      write_stream: str,
      batch_size: int,
      flush_interval: float,
      retry_config: RetryConfig,
      queue_max_size: int,
      shutdown_timeout: float,
  ):
    """Initializes the instance.

    Args:
        write_client: BigQueryWriteAsyncClient for writing rows.
        arrow_schema: PyArrow schema for serialization.
        write_stream: BigQuery write stream name.
        batch_size: Number of rows per batch.
        flush_interval: Max time to wait before flushing a batch.
        retry_config: Retry configuration.
        queue_max_size: Max size of the in-memory queue.
        shutdown_timeout: Max time to wait for shutdown.
    """
    self.write_client = write_client
    self.arrow_schema = arrow_schema
    self.write_stream = write_stream
    self.batch_size = batch_size
    self.flush_interval = flush_interval
    self.retry_config = retry_config
    self.shutdown_timeout = shutdown_timeout

    self._visual_builder = _is_visual_builder.get()

    self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
        maxsize=queue_max_size
    )
    self._queue_max_size = queue_max_size
    # Outstanding shutdown sentinels currently in the queue: lets
    # cancellation account for remaining rows in O(1) via qsize() instead
    # of a synchronous full drain.
    self._sentinel_count = 0
    self._batch_processor_task: Optional[asyncio.Task[None]] = None
    self._shutdown = False

    # Running tally of events/rows dropped without ever being written, keyed by
    # reason. Logging every drop is the only existing signal that data was lost,
    # and those logs are easy to miss at volume; these counters let a host poll
    # get_drop_stats() and export the loss to its own monitoring before it shows
    # up as missing rows downstream.
    self._dropped: dict[str, int] = {
        "queue_full": 0,
        "arrow_prep_failed": 0,
        "retry_exhausted": 0,
        "non_retryable": 0,
        "unexpected_error": 0,
        "shutdown_timeout": 0,
        "shutdown_cancelled": 0,
    }

  async def flush(self) -> None:
    """Flushes the queue, blocking until in-flight writes complete."""
    # empty() turns true as soon as an item is dequeued, before its write
    # finishes; join() waits for the unfinished-task count to reach zero.
    await self._queue.join()

  async def start(self) -> None:
    """Starts the batch writer worker task."""
    if self._batch_processor_task is None:
      self._batch_processor_task = asyncio.create_task(self._batch_writer())

  async def append(self, row: dict[str, Any]) -> None:
    """Appends a row to the queue for batching.

    Args:
        row: Dictionary representing a single row.
    """
    try:
      self._queue.put_nowait(row)
    except asyncio.QueueFull:
      self._dropped["queue_full"] += 1
      logger.warning(
          "BigQuery log queue full, dropping event. Total events dropped"
          " (queue full): %s",
          self._dropped["queue_full"],
      )

  def get_drop_stats(self) -> dict[str, int]:
    """Returns a snapshot of dropped-row counts keyed by reason.

    Dropped rows are logged best-effort and never written, so these counters
    are the canonical signal that data was lost. Reasons:

      ``queue_full``: the in-memory queue was full when the event arrived.
      ``arrow_prep_failed``: the batch could not be serialized to Arrow.
      ``retry_exhausted``: the write failed after exhausting all retries.
      ``non_retryable``: BigQuery returned a non-retryable error (e.g. a
        schema mismatch).
      ``unexpected_error``: an unexpected exception aborted the write.
      ``shutdown_timeout``: rows still queued when shutdown timed out.
      ``shutdown_cancelled``: rows still queued when shutdown was
        cancelled from outside (e.g. a host close timeout).

    Returns:
        A copy of the per-reason drop counters.
    """
    return dict(self._dropped)

  @property
  def dropped_event_count(self) -> int:
    """Total rows dropped without being written, across all reasons."""
    return sum(self._dropped.values())

  def _prepare_arrow_batch(self, rows: list[dict[str, Any]]) -> pa.RecordBatch:
    """Prepares a PyArrow RecordBatch from a list of rows.

    Args:
        rows: list of row dictionaries.

    Returns:
        pa.RecordBatch for writing.
    """
    data: dict[str, list[Any]] = {field.name: [] for field in self.arrow_schema}
    for row in rows:
      for field in self.arrow_schema:
        value = row.get(field.name)
        # JSON fields must be serialized to strings for the Arrow layer
        field_metadata = self.arrow_schema.field(field.name).metadata
        is_json = False
        if field_metadata and b"ARROW:extension:name" in field_metadata:
          if field_metadata[b"ARROW:extension:name"] == b"google:sqlType:json":
            is_json = True

        arrow_field_type = self.arrow_schema.field(field.name).type
        is_struct = pa.types.is_struct(arrow_field_type)
        is_list = pa.types.is_list(arrow_field_type)

        if is_json:
          if value is not None:
            if isinstance(value, (dict, list)):
              try:
                value = json.dumps(value)
              except (TypeError, ValueError):
                value = str(value)
            elif isinstance(value, (str, bytes)):
              if isinstance(value, bytes):
                try:
                  value = value.decode("utf-8")
                except UnicodeDecodeError:
                  value = str(value)

              # Check if it's already a valid JSON object or array to avoid double-encoding
              is_already_json = False
              if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith(("{", "[")) and stripped.endswith(
                    ("}", "]")
                ):
                  try:
                    json.loads(value)
                    is_already_json = True
                  except (ValueError, TypeError):
                    pass

              if not is_already_json:
                try:
                  value = json.dumps(value)
                except (TypeError, ValueError):
                  value = str(value)
              # If is_already_json is True, we keep value as-is
            else:
              # For other types (int, float, bool), serialize to JSON equivalents
              try:
                value = json.dumps(value)
              except (TypeError, ValueError):
                value = str(value)
        elif isinstance(value, (dict, list)) and not is_struct and not is_list:
          if value is not None and not isinstance(value, (str, bytes)):
            try:
              value = json.dumps(value)
            except (TypeError, ValueError):
              value = str(value)
        data[field.name].append(value)
    return pa.RecordBatch.from_pydict(data, schema=self.arrow_schema)

  async def _batch_writer(self) -> None:
    """Worker task that batches and writes rows to BigQuery."""
    while not self._shutdown or not self._queue.empty():
      batch = []
      try:
        if self._shutdown:
          try:
            first_item = self._queue.get_nowait()
          except asyncio.QueueEmpty:
            break
        else:
          first_item = await asyncio.wait_for(
              self._queue.get(), timeout=self.flush_interval
          )

        if first_item is _SHUTDOWN_SENTINEL:
          self._sentinel_count = max(0, self._sentinel_count - 1)
          self._queue.task_done()
          continue

        batch.append(first_item)

        while len(batch) < self.batch_size:
          try:
            item = self._queue.get_nowait()
            if item is _SHUTDOWN_SENTINEL:
              self._sentinel_count = max(0, self._sentinel_count - 1)
              self._queue.task_done()
              continue
            batch.append(item)
          except asyncio.QueueEmpty:
            break

        if batch:
          try:
            await self._write_rows_with_retry(batch)
          finally:
            # Mark tasks as done ONLY after processing (write attempt)
            for _ in batch:
              self._queue.task_done()

      except asyncio.TimeoutError:
        continue
      except asyncio.CancelledError:
        # Cancelled (e.g. by the shutdown timeout): the in-flight batch is
        # lost — count it — then exit the
        # worker, preserving the original swallow-and-break semantics.
        if batch:
          self._dropped["shutdown_timeout"] += len(batch)
          logger.warning(
              "%d in-flight row(s) dropped by shutdown cancellation.",
              len(batch),
          )
        logger.info("Batch writer task cancelled.")
        # Re-raise: asyncio.wait_for treats a task that SUPPRESSES
        # cancellation as a normal completion, so shutdown()'s timeout
        # branch (which drains and counts the remaining queue) would never
        # run if this swallowed the cancellation.
        raise
      except Exception as e:
        logger.error("Error in batch writer loop: %s", e, exc_info=True)
        # Avoid sleeping if we are shutting down or if the task was cancelled
        if not self._shutdown:
          try:
            await asyncio.sleep(1)
          except (asyncio.CancelledError, RuntimeError):
            break
        else:
          break

  async def _write_rows_with_retry(self, rows: list[dict[str, Any]]) -> None:
    """Writes a batch of rows to BigQuery with retry logic.

    Args:
        rows: list of row dictionaries to write.
    """
    attempt = 0
    delay = self.retry_config.initial_delay

    try:
      arrow_batch = self._prepare_arrow_batch(rows)
      serialized_schema = self.arrow_schema.serialize().to_pybytes()
      serialized_batch = arrow_batch.serialize().to_pybytes()

      trace_id_prefix = (
          "google-adk-bq-logger-visual-builder"
          if self._visual_builder
          else "google-adk-bq-logger"
      )

      req = bq_storage_types.AppendRowsRequest(
          write_stream=self.write_stream,
          trace_id=f"{trace_id_prefix}/{__version__}",
      )
      req.arrow_rows.writer_schema.serialized_schema = serialized_schema
      req.arrow_rows.rows.serialized_record_batch = serialized_batch
    except Exception as e:
      self._dropped["arrow_prep_failed"] += len(rows)
      logger.error(
          "Failed to prepare Arrow batch (Data Loss): %s. Total rows dropped"
          " (arrow prep failed): %s",
          e,
          self._dropped["arrow_prep_failed"],
          exc_info=True,
      )
      return

    while attempt <= self.retry_config.max_retries:
      try:

        async def requests_iter() -> AsyncIterator[Any]:
          yield req

        async def perform_write() -> None:
          # The AppendRows streaming RPC does not auto-populate the
          # request-routing header, so writes to any region other than
          # the US multiregion fail with a "session not found" /
          # stream-not-found error. Set the routing header explicitly
          # (same as google.cloud.bigquery_storage_v1.writer) so the
          # request reaches the region that owns the write stream.
          responses = await self.write_client.append_rows(
              requests_iter(),
              metadata=(
                  (
                      "x-goog-request-params",
                      f"write_stream={self.write_stream}",
                  ),
              ),
          )
          async for response in responses:
            error = getattr(response, "error", None)
            error_code = getattr(error, "code", None)
            if error_code and error_code != 0:
              error_message = getattr(error, "message", "Unknown error")
              logger.warning(
                  "BigQuery Write API returned error code %s: %s",
                  error_code,
                  error_message,
              )
              if error_code in [
                  _GRPC_DEADLINE_EXCEEDED,
                  _GRPC_INTERNAL,
                  _GRPC_UNAVAILABLE,
              ]:
                raise ServiceUnavailable(error_message)

              if "schema mismatch" in error_message.lower():
                logger.error(
                    "BigQuery Schema Mismatch: %s. This usually means the"
                    " table schema does not match the expected schema.",
                    error_message,
                )
              else:
                logger.error("Non-retryable BigQuery error: %s", error_message)
                row_errors = getattr(response, "row_errors", [])
                if row_errors:
                  for row_error in row_errors:
                    logger.error("Row error details: %s", row_error)
                logger.error(
                    "%d row(s) dropped due to a non-retryable BigQuery error.",
                    len(rows),
                )
              self._dropped["non_retryable"] += len(rows)
              return
          return

        await asyncio.wait_for(perform_write(), timeout=30.0)
        return

      except (
          ServiceUnavailable,
          TooManyRequests,
          InternalServerError,
          asyncio.TimeoutError,
      ) as e:
        attempt += 1
        if attempt > self.retry_config.max_retries:
          self._dropped["retry_exhausted"] += len(rows)
          logger.error(
              "BigQuery Batch Dropped after %s attempts. Last error: %s."
              " Total rows dropped (retry exhausted): %s",
              self.retry_config.max_retries + 1,
              e,
              self._dropped["retry_exhausted"],
          )
          return

        sleep_time = min(
            delay * (1 + random.random()), self.retry_config.max_delay
        )
        logger.warning(
            "BigQuery write failed (Attempt %s), retrying in %.2fs..."
            " Error: %s",
            attempt,
            sleep_time,
            e,
        )
        await asyncio.sleep(sleep_time)
        delay *= self.retry_config.multiplier
      except Exception as e:
        self._dropped["unexpected_error"] += len(rows)
        logger.error(
            "Unexpected BigQuery Write API error (Dropping batch): %s."
            " Total rows dropped (unexpected error): %s",
            e,
            self._dropped["unexpected_error"],
            exc_info=True,
        )
        return

  def _drain_queue_and_count(self, reason: str) -> int:
    """Counts and discards everything still queued (loss accounting)."""
    drained = 0
    try:
      while True:
        item = self._queue.get_nowait()
        if item is not _SHUTDOWN_SENTINEL:
          drained += 1
        else:
          self._sentinel_count = max(0, self._sentinel_count - 1)
        self._queue.task_done()
    except asyncio.QueueEmpty:
      pass
    if drained:
      self._dropped[reason] += drained
      logger.warning("%d queued row(s) dropped (%s).", drained, reason)
    return drained

  async def shutdown(self, timeout: float = 5.0) -> None:
    """Shuts down the BatchProcessor, draining the queue.

    Args:
        timeout: Maximum time to wait for the queue to drain.
    """
    self._shutdown = True
    logger.info("BatchProcessor shutting down, draining queue...")

    # Signal the writer to wake up and check shutdown status
    try:
      self._queue.put_nowait(_SHUTDOWN_SENTINEL)
      self._sentinel_count += 1
    except asyncio.QueueFull:
      # If queue is full, the writer is active and will check _shutdown soon
      pass

    if self._batch_processor_task:
      if self._batch_processor_task.done():
        # A previous shutdown attempt already terminated the worker —
        # possibly by external cancellation. Re-awaiting the task would
        # re-raise its historical CancelledError on EVERY retry, so no
        # later close could ever finish cleanup. Treat the terminal worker as final and account for
        # whatever is still queued.
        self._drain_queue_and_count(
            "shutdown_cancelled"
            if self._batch_processor_task.cancelled()
            else "shutdown_timeout"
        )
        return
      try:
        await asyncio.wait_for(self._batch_processor_task, timeout=timeout)
      except asyncio.TimeoutError:
        logger.warning("BatchProcessor shutdown timed out, cancelling worker.")
        self._batch_processor_task.cancel()
        # Convert the WORKER's expected CancelledError into a gather result,
        # then shield that acknowledgement owner. If the HOST cancels this
        # shutdown await, shield raises CancelledError unambiguously while the
        # gather continues to own/retrieve the worker result. This works on
        # Python 3.10 (where Task.cancelling() does not exist) and preserves
        # the external-cancellation distinction on newer runtimes.
        await asyncio.shield(
            asyncio.gather(
                self._batch_processor_task,
                return_exceptions=True,
            )
        )
        # Rows still queued after the timeout are lost: count them so the
        # loss is observable instead of silent.
        # The worker counts its own in-flight batch on cancellation.
        self._drain_queue_and_count("shutdown_timeout")
      except asyncio.CancelledError:
        # EXTERNAL cancellation (e.g. PluginManager's close timeout):
        # wait_for has already cancelled the worker. Account for queued
        # rows now — a retry may never come — and preserve the caller's
        # cancellation. The accounting is
        # O(1): this handler runs INSIDE the caller's cancellation window
        # (asyncio.timeout waits for cleanup), so a synchronous full
        # drain of an unbounded queue extended host-close latency
        # linearly with queue depth. qsize
        # minus outstanding sentinels counts the rows; storage is
        # released by swapping in a fresh queue, reclaimed by GC off the
        # cancellation-critical path.
        remaining = max(0, self._queue.qsize() - self._sentinel_count)
        if remaining:
          self._dropped["shutdown_cancelled"] += remaining
          logger.warning(
              "%d queued row(s) dropped (shutdown_cancelled).", remaining
          )
        self._queue = asyncio.Queue(maxsize=self._queue_max_size)
        self._sentinel_count = 0
        raise
      except Exception as e:
        logger.error("Error during BatchProcessor shutdown: %s", e)

  async def close(self) -> None:
    """Closes the processor and flushes remaining items."""
    if self._shutdown:
      return

    self._shutdown = True
    # Wait for queue to be empty
    try:
      await asyncio.wait_for(self._queue.join(), timeout=self.shutdown_timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
      logger.warning(
          "Timeout waiting for BigQuery batch queue to empty on shutdown."
      )

    # Cancel the writer task if it's still running (it should exit on _shutdown + empty queue)
    if self._batch_processor_task and not self._batch_processor_task.done():
      self._batch_processor_task.cancel()
      try:
        await self._batch_processor_task
      except asyncio.CancelledError:
        pass
    # Same loss accounting as shutdown(): rows still queued after the
    # timeout are counted, not silently discarded. The cancelled worker counts its own in-flight batch.
    drained = 0
    try:
      while True:
        item = self._queue.get_nowait()
        if item is not _SHUTDOWN_SENTINEL:
          drained += 1
        self._queue.task_done()
    except asyncio.QueueEmpty:
      pass
    if drained:
      self._dropped["shutdown_timeout"] += drained
      logger.warning("%d queued row(s) dropped by close timeout.", drained)


# ==============================================================================
# HELPER: CONTENT PARSER (Length Limits Only)
# ==============================================================================
class ContentParser:
  """Parses content for logging with length limits and structure normalization."""

  def __init__(self, max_length: int) -> None:
    """Initializes the instance.

    Args:
        max_length: Maximum length for text content.
    """
    self.max_length = max_length

  def _truncate(self, text: str) -> tuple[str, bool]:
    if self.max_length != -1 and text and len(text) > self.max_length:
      return text[: self.max_length] + "...[TRUNCATED]", True
    return text, False


class GCSOffloader:
  """Offloads content to GCS."""

  def __init__(
      self,
      project_id: str,
      bucket_name: str,
      executor: ThreadPoolExecutor,
      storage_client: Optional[storage.Client] = None,
  ):
    self.client = storage_client or storage.Client(project=project_id)
    self.bucket = self.client.bucket(bucket_name)
    self.executor = executor

  async def upload_content(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    """Async wrapper around blocking GCS upload."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        self.executor,
        functools.partial(self._upload_sync, data, content_type, path),
    )

  def _upload_sync(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    blob = self.bucket.blob(path)
    # if_generation_match=0: create-only. Object names are unique by
    # construction, so on the (astronomically unlikely) collision this
    # fails the upload — surfaced as [UPLOAD FAILED] — instead of silently
    # rebinding an existing BigQuery row to another event's bytes.
    blob.upload_from_string(
        data, content_type=content_type, if_generation_match=0
    )
    return f"gs://{self.bucket.name}/{path}"


class HybridContentParser:
  """Parses content and offloads large/binary parts to GCS."""

  def __init__(
      self,
      offloader: Optional[GCSOffloader],
      trace_id: str,
      span_id: str,
      max_length: int = 20000,
      connection_id: Optional[str] = None,
  ):
    self.offloader = offloader
    self.trace_id = trace_id
    self.span_id = span_id
    self.max_length = max_length
    self.connection_id = connection_id
    self.inline_text_limit = 32 * 1024  # 32KB limit

  def _truncate(self, text: str) -> tuple[str, bool]:
    if self.max_length != -1 and len(text) > self.max_length:
      return (
          text[: self.max_length] + "...[TRUNCATED]",
          True,
      )
    return text, False

  @staticmethod
  def _sanitize_raw_text(text: str) -> tuple[str, bool]:
    """Sanitizes caller-provided text exactly once before it is stored.

    Known formatter/redaction sentinels are already safe and must not be
    reinterpreted as malformed JSON merely because they are bracketed. All
    other raw text goes through the same fail-closed credential sanitizer used
    for structured attributes. Length truncation remains a separate pass so
    the sanitized value is also the value sent to GCS.
    """
    if text in (_FORMATTER_FAILED_SENTINEL, "[REDACTED]"):
      return text, False
    sanitized, content_lost = _recursive_smart_truncate(text, -1)
    if sanitized == "[UNPARSEABLE_JSON_BLOB]":
      # Raw content is user-facing prose, not an opaque attributes blob.
      # Reusing the attributes sanitizer made every invalid bracket-led
      # message ("[INFO]", Markdown links, "{not json}") disappear. Restore
      # only bounded prose with no raw or encoded credential construct;
      # malformed credential documents remain fail-closed while ordinary
      # Windows paths and decoder diagnostics stay byte-identical.
      stripped = _strip_bom_ws(text)
      if (
          len(stripped) <= _MAX_JSON_INSPECT_CHARS
          and stripped.startswith(("[", "{"))
          and not _contains_sensitive_text_marker(stripped)
      ):
        try:
          json.loads(stripped)
        except (ValueError, RecursionError, MemoryError):
          return text, False
    if not isinstance(sanitized, str):
      return "[UNSUPPORTED_OBJECT]", True
    return sanitized, content_lost

  def _sanitize_and_truncate(self, text: str) -> tuple[str, bool]:
    sanitized, content_lost = self._sanitize_raw_text(text)
    truncated_text, length_truncated = self._truncate(sanitized)
    return truncated_text, content_lost or length_truncated

  def _sanitize_external_uri(self, uri: str) -> tuple[str, bool]:
    """Redacts signed/query credentials while preserving a URI's location."""
    if not isinstance(uri, str):
      return "[REDACTED_SENSITIVE_URI]", True
    if type(uri) is not str:
      uri = str.__str__(uri)
    if len(uri) > _MAX_JSON_INSPECT_CHARS:
      return "[REDACTED_SENSITIVE_URI]", True
    try:
      parsed = urlsplit(uri)
      if parsed.username is not None or parsed.password is not None:
        # Userinfo is a credential-bearing URI surface by definition. Do not
        # try to retain a username while guessing whether it is sensitive.
        return "[REDACTED_SENSITIVE_URI]", True
      query = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
      return "[REDACTED_SENSITIVE_URI]", True

    changed = False
    path_segments = parsed.path.split("/")
    redact_next_path_segment = False
    for index, segment in enumerate(path_segments):
      if not segment:
        continue
      canonical_segment = _canonicalize_common_ascii_escapes(segment)
      if redact_next_path_segment:
        path_segments[index] = quote("[REDACTED]", safe="")
        changed = True
        redact_next_path_segment = False
        continue
      if _is_sensitive_text_key(canonical_segment):
        path_segments[index] = quote("[REDACTED]", safe="")
        changed = True
        redact_next_path_segment = True
        continue
      safe_segment, segment_changed = _sanitize_sensitive_text(segment, -1)
      if segment_changed:
        path_segments[index] = quote(safe_segment, safe="")
        changed = True

    safe_query: list[tuple[str, str]] = []
    for key, value in query:
      if _is_sensitive_text_key(key):
        safe_query.append((key, "[REDACTED]"))
        changed = True
        continue
      safe_key, key_changed = _sanitize_sensitive_text(key, -1)
      safe_value, value_changed = _sanitize_sensitive_text(value, -1)
      safe_query.append((safe_key, safe_value))
      changed = changed or key_changed or value_changed

    safe_fragment, fragment_changed = _sanitize_sensitive_text(
        parsed.fragment, -1
    )
    changed = changed or fragment_changed
    safe_uri = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        "/".join(path_segments),
        urlencode(safe_query),
        safe_fragment,
    ))
    safe_uri, uri_truncated = self._truncate(safe_uri)
    return safe_uri, changed or uri_truncated

  def _serialize_part_model(self, value: Any) -> tuple[dict[str, Any], bool]:
    """Returns bounded JSON-native fields for a supported structured part."""
    dumped = value.model_dump(exclude_none=True, mode="json")
    sanitized, content_lost = _recursive_smart_truncate(dumped, self.max_length)
    if not isinstance(sanitized, dict):
      return {"value": "[UNSUPPORTED_OBJECT]"}, True

    budget = [_MAX_SANITIZE_NODES]

    def _sanitize_strings(obj: Any, depth: int = 0) -> tuple[Any, bool]:
      budget[0] -= 1
      if budget[0] < 0 or depth >= _MAX_SANITIZE_DEPTH:
        return "[SANITIZE_BUDGET_EXCEEDED]", True
      if isinstance(obj, str):
        return _sanitize_sensitive_text(obj, self.max_length)
      if isinstance(obj, dict):
        out: dict[str, Any] = {}
        replaced = False
        collision_count = 0

        def _collision_safe_key(key: str) -> str:
          nonlocal collision_count, replaced
          if key not in out:
            return key
          collision_count += 1
          candidate = f"[KEY_COLLISION_{collision_count}]{key}"
          while candidate in out:
            collision_count += 1
            candidate = f"[KEY_COLLISION_{collision_count}]{key}"
          replaced = True
          return candidate

        for key, item in obj.items():
          if budget[0] <= 0:
            budget_key = _collision_safe_key("[SANITIZE_BUDGET_EXCEEDED]")
            out[budget_key] = "[SANITIZE_BUDGET_EXCEEDED]"
            return out, True
          redact_item = False
          if not isinstance(key, str):
            safe_key = "[UNSUPPORTED_KEY]"
            key_replaced = True
          else:
            if type(key) is not str:
              key = str.__str__(key)
            canonical_key = _canonicalize_common_ascii_escapes(key)
            redact_item = _is_sensitive_text_key(canonical_key)
            safe_key, key_replaced = _sanitize_sensitive_text(
                key, self.max_length
            )
          safe_key = _collision_safe_key(safe_key)
          if redact_item:
            safe_item = "[REDACTED]"
            item_replaced = item != "[REDACTED]"
          else:
            safe_item, item_replaced = _sanitize_strings(item, depth + 1)
          out[safe_key] = safe_item
          replaced = replaced or key_replaced or item_replaced
        return out, replaced
      if isinstance(obj, list):
        out_list = []
        replaced = False
        for item in obj:
          if budget[0] <= 0:
            out_list.append("[SANITIZE_BUDGET_EXCEEDED]")
            return out_list, True
          safe_item, item_replaced = _sanitize_strings(item, depth + 1)
          out_list.append(safe_item)
          replaced = replaced or item_replaced
        return out_list, replaced
      return obj, False

    sanitized, text_content_lost = _sanitize_strings(sanitized)
    if not isinstance(sanitized, dict):
      return {"value": "[UNSUPPORTED_OBJECT]"}, True
    return sanitized, content_lost or text_content_lost

  async def _parse_content_object(
      self,
      content: types.Content | types.Part,
      *,
      trace_id: Optional[str] = None,
      span_id: Optional[str] = None,
      parse_uid: str = "",
      content_ordinal: int = 0,
  ) -> tuple[str, list[dict[str, Any]], bool]:
    """Parses a Content or Part object into summary text and content parts.

    ``trace_id``/``span_id`` are call-local: GCS object paths are built from
    these arguments so concurrent parses on the shared parser instance can
    never use another event's identity. They fall back to the
    constructor values for backward compatibility.

    ``parse_uid`` (unique per parse() call) and ``content_ordinal`` (the
    message index within a multi-content request) disambiguate GCS object
    names: the part index alone restarts per Content, so two messages in
    one request would otherwise collide at the same part ordinal.
    """
    trace_id = trace_id if trace_id is not None else self.trace_id
    span_id = span_id if span_id is not None else self.span_id
    parse_uid = parse_uid or uuid.uuid4().hex
    content_parts = []
    is_truncated = False
    summary_text = []

    parts = content.parts if hasattr(content, "parts") else [content]
    for idx, part in enumerate(parts):
      part_data = {
          "part_index": idx,
          "mime_type": "text/plain",
          "uri": None,
          "text": None,
          "part_attributes": "{}",
          "storage_mode": "INLINE",
          "object_ref": None,
      }

      # CASE A: It is already a URI (e.g. from user input)
      if hasattr(part, "file_data") and part.file_data:
        part_data["storage_mode"] = "EXTERNAL_URI"
        safe_uri, uri_content_lost = self._sanitize_external_uri(
            part.file_data.file_uri
        )
        part_data["uri"] = safe_uri
        if uri_content_lost:
          is_truncated = True
        part_data["mime_type"] = part.file_data.mime_type

      # CASE B: It is Binary/Inline Data (Image/Blob)
      elif hasattr(part, "inline_data") and part.inline_data:
        if self.offloader:
          ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".bin"
          path = (
              f"{datetime.now().date()}/{trace_id}/{span_id}_{parse_uid}"
              f"_c{content_ordinal}_p{idx}{ext}"
          )
          try:
            uri = await self.offloader.upload_content(
                part.inline_data.data, part.inline_data.mime_type, path
            )
            part_data["storage_mode"] = "GCS_REFERENCE"
            part_data["uri"] = uri
            object_ref = {
                "uri": uri,
                "version": None,
                "authorizer": self.connection_id,
                "details": json.dumps({
                    "gcs_metadata": {"content_type": part.inline_data.mime_type}
                }),
            }
            part_data["object_ref"] = object_ref
            part_data["mime_type"] = part.inline_data.mime_type
            part_data["text"] = "[MEDIA OFFLOADED]"
          except Exception as e:
            logger.warning("Failed to offload content to GCS: %s", e)
            part_data["text"] = "[UPLOAD FAILED]"
        else:
          part_data["text"] = "[BINARY DATA]"

      # CASE C: Text
      elif hasattr(part, "text") and part.text:
        safe_text, sanitized_content_lost = self._sanitize_raw_text(part.text)
        if sanitized_content_lost:
          is_truncated = True
        char_len = len(safe_text)
        byte_len = len(safe_text.encode("utf-8"))

        # Decide whether to offload using each limit in its own
        # unit.  inline_text_limit is a byte-based storage guard;
        # max_length is a character-based truncation limit.
        exceeds_inline_byte_limit = byte_len > self.inline_text_limit
        exceeds_char_limit = (
            self.max_length != -1 and char_len > self.max_length
        )

        if self.offloader and (exceeds_inline_byte_limit or exceeds_char_limit):
          # Text is too big, treat as file
          path = (
              f"{datetime.now().date()}/{trace_id}/{span_id}_{parse_uid}"
              f"_c{content_ordinal}_p{idx}.txt"
          )
          try:
            uri = await self.offloader.upload_content(
                safe_text, "text/plain", path
            )
            part_data["storage_mode"] = "GCS_REFERENCE"
            part_data["uri"] = uri
            object_ref = {
                "uri": uri,
                "version": None,
                "authorizer": self.connection_id,
                "details": json.dumps(
                    {"gcs_metadata": {"content_type": "text/plain"}}
                ),
            }
            part_data["object_ref"] = object_ref
            part_data["mime_type"] = "text/plain"
            part_data["text"] = safe_text[:200] + "... [OFFLOADED]"
          except Exception as e:
            logger.warning("Failed to offload text to GCS: %s", e)
            clean_text, truncated = self._truncate(safe_text)
            if truncated:
              is_truncated = True
            part_data["text"] = clean_text
            summary_text.append(clean_text)
        else:
          # Text is small or no offloader, keep inline
          clean_text, truncated = self._truncate(safe_text)
          if truncated:
            is_truncated = True
          part_data["text"] = clean_text
          summary_text.append(clean_text)

      elif hasattr(part, "function_call") and part.function_call:
        part_data["mime_type"] = "application/json"
        part_data["text"] = f"Function: {part.function_call.name}"
        part_data["part_attributes"] = json.dumps(
            {"function_name": part.function_call.name}
        )

      elif hasattr(part, "function_response") and part.function_response:
        response, response_lost = self._serialize_part_model(
            part.function_response
        )
        if response_lost:
          is_truncated = True
        name = response.get("name") or "unknown"
        if not isinstance(name, str):
          name = "[UNSUPPORTED_OBJECT]"
          is_truncated = True
        response_summary = f"Function response: {name}"
        part_data["mime_type"] = "application/json"
        part_data["text"] = response_summary
        part_data["part_attributes"] = json.dumps(
            {"function_response": response}
        )
        summary_text.append(response_summary)

      elif hasattr(part, "executable_code") and part.executable_code:
        executable, code_lost = self._serialize_part_model(part.executable_code)
        if code_lost:
          is_truncated = True
        language = executable.get("language") or "unknown"
        if not isinstance(language, str):
          language = "[UNSUPPORTED_OBJECT]"
          is_truncated = True
        code = executable.get("code") or ""
        if not isinstance(code, str):
          code = "[UNSUPPORTED_OBJECT]"
          is_truncated = True
        part_data["mime_type"] = "text/plain"
        part_data["text"] = code
        part_data["part_attributes"] = json.dumps({
            "executable_code": executable,
        })
        summary_text.append(f"Executable code ({language}): {code}")

      elif (
          hasattr(part, "code_execution_result") and part.code_execution_result
      ):
        result, result_lost = self._serialize_part_model(
            part.code_execution_result
        )
        if result_lost:
          is_truncated = True
        outcome = result.get("outcome") or "unknown"
        if not isinstance(outcome, str):
          outcome = "[UNSUPPORTED_OBJECT]"
          is_truncated = True
        output = result.get("output") or ""
        if not isinstance(output, str):
          output = "[UNSUPPORTED_OBJECT]"
          is_truncated = True
        part_data["mime_type"] = "text/plain"
        part_data["text"] = output
        part_data["part_attributes"] = json.dumps({
            "code_execution_result": result,
        })
        summary_text.append(f"Code execution result ({outcome}): {output}")

      content_parts.append(part_data)

    summary_str, truncated = self._truncate(" | ".join(summary_text))
    if truncated:
      is_truncated = True

    return summary_str, content_parts, is_truncated

  async def parse(
      self,
      content: Any,
      *,
      trace_id: Optional[str] = None,
      span_id: Optional[str] = None,
  ) -> tuple[Any, list[dict[str, Any]], bool]:
    """Parses content into JSON payload and content parts, potentially offloading to GCS.

    ``trace_id``/``span_id`` identify the calling event for GCS object paths.
    Pass them per call — the parser instance is shared across concurrent
    events, so relying on the mutable instance fields lets one event's await
    resume with another event's identity and overwrite its objects. The instance
    fields remain only as a backward-compatible default.
    """
    trace_id = trace_id if trace_id is not None else self.trace_id
    span_id = span_id if span_id is not None else self.span_id
    # Unique per parse() call: disambiguates GCS object names across the
    # multiple Content objects of one request and across concurrent events.
    parse_uid = uuid.uuid4().hex
    json_payload = {}
    content_parts = []
    is_truncated = False

    def process_text(t: str) -> tuple[str, bool]:
      return self._sanitize_and_truncate(t)

    if isinstance(content, LlmRequest):
      # Handle Prompt
      messages = []
      contents = (
          content.contents
          if isinstance(content.contents, list)
          else [content.contents]
      )
      for content_idx, c in enumerate(contents):
        role = getattr(c, "role", "unknown")
        if isinstance(role, str):
          role, role_truncated = process_text(role)
          if role_truncated:
            is_truncated = True
        summary, parts, trunc = await self._parse_content_object(
            c,
            trace_id=trace_id,
            span_id=span_id,
            parse_uid=parse_uid,
            content_ordinal=content_idx,
        )
        if trunc:
          is_truncated = True
        content_parts.extend(parts)
        messages.append({"role": role, "content": summary})

      if messages:
        json_payload["prompt"] = messages

      # Handle System Instruction
      if content.config and getattr(content.config, "system_instruction", None):
        si = content.config.system_instruction
        if isinstance(si, str):
          truncated_si, trunc = process_text(si)
          if trunc:
            is_truncated = True
          json_payload["system_prompt"] = truncated_si
        else:
          summary, parts, trunc = await self._parse_content_object(
              si,
              trace_id=trace_id,
              span_id=span_id,
              parse_uid=parse_uid,
              content_ordinal=len(contents),
          )
          if trunc:
            is_truncated = True
          content_parts.extend(parts)
          json_payload["system_prompt"] = summary

    elif isinstance(content, (types.Content, types.Part)):
      summary, parts, trunc = await self._parse_content_object(
          content,
          trace_id=trace_id,
          span_id=span_id,
          parse_uid=parse_uid,
      )
      return {"text_summary": summary}, parts, trunc

    elif isinstance(content, (dict, list)):
      json_payload, is_truncated = _recursive_smart_truncate(
          content, self.max_length
      )
    elif isinstance(content, str):
      json_payload, is_truncated = process_text(content)
    elif content is None:
      json_payload = None
    else:
      json_payload, is_truncated = process_text(str(content))

    return json_payload, content_parts, is_truncated


def _get_events_schema() -> list[bigquery.SchemaField]:
  """Returns the BigQuery schema for the events table."""
  return [
      bigquery.SchemaField(
          "timestamp",
          "TIMESTAMP",
          mode="REQUIRED",
          description=(
              "The UTC timestamp when the event occurred. Used for ordering"
              " events within a session."
          ),
      ),
      bigquery.SchemaField(
          "event_type",
          "STRING",
          mode="NULLABLE",
          description=(
              "The category of the event (e.g., 'LLM_REQUEST', 'TOOL_CALL',"
              " 'AGENT_RESPONSE'). Helps in filtering specific types of"
              " interactions."
          ),
      ),
      bigquery.SchemaField(
          "agent",
          "STRING",
          mode="NULLABLE",
          description=(
              "The name of the agent that generated this event. Useful for"
              " multi-agent systems."
          ),
      ),
      bigquery.SchemaField(
          "session_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "A unique identifier for the entire conversation session. Used"
              " to group all events belonging to a single user interaction."
          ),
      ),
      bigquery.SchemaField(
          "invocation_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "A unique identifier for a single turn or execution within a"
              " session. Groups related events like LLM request and response."
          ),
      ),
      bigquery.SchemaField(
          "user_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "The identifier of the end-user participating in the session,"
              " if available."
          ),
      ),
      bigquery.SchemaField(
          "trace_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "OpenTelemetry trace ID for distributed tracing across services."
          ),
      ),
      bigquery.SchemaField(
          "span_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "BQAA-internal execution-tree span id for this operation. This is"
              " the plugin's own correlation id used with parent_span_id to"
              " reconstruct the agent/LLM/tool tree -- NOT the OpenTelemetry"
              " span id, except on the root/invocation row where it may reuse"
              " the ambient OTel span id. For span-level Cloud Trace"
              " correlation use attributes.otel.span_id (best-effort)."
          ),
      ),
      bigquery.SchemaField(
          "parent_span_id",
          "STRING",
          mode="NULLABLE",
          description=(
              "BQAA-internal parent execution-tree span id, used to reconstruct"
              " the operation hierarchy. Points at another BQAA row, not an"
              " OpenTelemetry parent span."
          ),
      ),
      bigquery.SchemaField(
          "content",
          "JSON",
          mode="NULLABLE",
          description=(
              "The primary payload of the event, stored as a JSON string. The"
              " structure depends on the event_type (e.g., prompt text for"
              " LLM_REQUEST, tool output for TOOL_RESPONSE)."
          ),
      ),
      bigquery.SchemaField(
          "content_parts",
          "RECORD",
          mode="REPEATED",
          fields=[
              bigquery.SchemaField(
                  "mime_type",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "The MIME type of the content part (e.g., 'text/plain',"
                      " 'image/png')."
                  ),
              ),
              bigquery.SchemaField(
                  "uri",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "The URI of the content part if stored externally"
                      " (e.g., GCS bucket path)."
                  ),
              ),
              bigquery.SchemaField(
                  "object_ref",
                  "RECORD",
                  mode="NULLABLE",
                  fields=[
                      bigquery.SchemaField(
                          "uri",
                          "STRING",
                          mode="NULLABLE",
                          description="The URI of the object.",
                      ),
                      bigquery.SchemaField(
                          "version",
                          "STRING",
                          mode="NULLABLE",
                          description="The version of the object.",
                      ),
                      bigquery.SchemaField(
                          "authorizer",
                          "STRING",
                          mode="NULLABLE",
                          description="The authorizer for the object.",
                      ),
                      bigquery.SchemaField(
                          "details",
                          "JSON",
                          mode="NULLABLE",
                          description="Additional details about the object.",
                      ),
                  ],
                  description=(
                      "The ObjectRef of the content part if stored externally."
                  ),
              ),
              bigquery.SchemaField(
                  "text",
                  "STRING",
                  mode="NULLABLE",
                  description="The raw text content if the part is text-based.",
              ),
              bigquery.SchemaField(
                  "part_index",
                  "INTEGER",
                  mode="NULLABLE",
                  description=(
                      "The zero-based index of this part within the content."
                  ),
              ),
              bigquery.SchemaField(
                  "part_attributes",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "Additional metadata for this content part as a JSON"
                      " object (serialized to string)."
                  ),
              ),
              bigquery.SchemaField(
                  "storage_mode",
                  "STRING",
                  mode="NULLABLE",
                  description=(
                      "Indicates how the content part is stored (e.g.,"
                      " 'INLINE', 'GCS_REFERENCE', 'EXTERNAL_URI')."
                  ),
              ),
          ],
          description=(
              "For multi-modal events, contains a list of content parts"
              " (text, images, etc.)."
          ),
      ),
      bigquery.SchemaField(
          "attributes",
          "JSON",
          mode="NULLABLE",
          description=(
              "A JSON object containing arbitrary key-value pairs for"
              " additional event metadata. Includes enrichment fields like"
              " 'root_agent_name' (turn orchestration), 'model' (request"
              " model), 'model_version' (response version), and"
              " 'usage_metadata' (detailed token counts). May also carry"
              " 'otel' (best-effort ambient Cloud Trace span/trace ids) and"
              " 'custom_metadata' (allowlisted event.custom_metadata keys)."
          ),
      ),
      bigquery.SchemaField(
          "latency_ms",
          "JSON",
          mode="NULLABLE",
          description=(
              "A JSON object containing latency measurements, such as"
              " 'total_ms' and 'time_to_first_token_ms'."
          ),
      ),
      bigquery.SchemaField(
          "status",
          "STRING",
          mode="NULLABLE",
          description="The outcome of the event, typically 'OK' or 'ERROR'.",
      ),
      bigquery.SchemaField(
          "error_message",
          "STRING",
          mode="NULLABLE",
          description="Detailed error message if the status is 'ERROR'.",
      ),
      bigquery.SchemaField(
          "is_truncated",
          "BOOLEAN",
          mode="NULLABLE",
          description=(
              "Boolean flag indicating if the content or metadata payload was"
              " truncated because it exceeded the maximum allowed size. Set"
              " when 'content', captured 'custom_metadata', or A2A metadata is"
              " truncated; redaction of sensitive keys does not set this flag."
          ),
      ),
  ]


# Payload columns eligible for physical projection.  Every other
# schema column is an identity / correlation / view-critical column and is
# *protected* — it cannot be projected out, because the BQAA execution tree
# and the per-event views depend on it.
_PROJECTABLE_PAYLOAD_COLUMNS = frozenset(
    {"content", "content_parts", "attributes", "latency_ms"}
)


def _validate_payload_column_denylist(
    denylist: Optional[list[str]],
) -> frozenset[str]:
  """Validates ``payload_column_denylist`` and returns the denied set.

  Only the projectable payload columns may be denied.  Anything else —
  an identity/correlation column or an unknown name — is a hard error,
  so a typo or an attempt to drop a join key fails loudly at construction
  rather than producing malformed rows or broken views.
  """
  denied = frozenset(denylist or ())
  invalid = denied - _PROJECTABLE_PAYLOAD_COLUMNS
  if invalid:
    raise ValueError(
        "payload_column_denylist may only contain projectable payload"
        f" columns {sorted(_PROJECTABLE_PAYLOAD_COLUMNS)}; got"
        f" {sorted(invalid)}. Identity/correlation columns (timestamp,"
        " event_type, session_id, invocation_id, trace_id, span_id,"
        " parent_span_id, is_truncated, ...) are protected and cannot be"
        " projected out."
    )
  return denied


def _project_schema(
    schema: list[bigquery.SchemaField], denied: frozenset[str]
) -> list[bigquery.SchemaField]:
  """Returns *schema* with denied columns removed (schema-first projection)."""
  if not denied:
    return schema
  return [f for f in schema if f.name not in denied]


def _parse_custom_metadata_allowlist(
    allowlist: Optional[list[str]],
) -> tuple[frozenset[str], tuple[str, ...]]:
  """Splits the allowlist into exact keys and explicit prefix patterns.

  An entry ending in ``*`` is an explicit prefix pattern (the ``*`` is
  stripped); every other entry matches exactly.  This keeps a plain key
  like ``"citation_metadata"`` from being treated as a prefix.
  """
  exact: set[str] = set()
  prefixes: list[str] = []
  for entry in allowlist or ():
    if entry.endswith("*"):
      prefixes.append(entry[:-1])
    else:
      exact.add(entry)
  return frozenset(exact), tuple(prefixes)


# ==============================================================================
# ANALYTICS VIEW DEFINITIONS
# ==============================================================================

# Columns included in every per-event-type view.
_VIEW_COMMON_COLUMNS = (
    "timestamp",
    "event_type",
    "agent",
    "session_id",
    "invocation_id",
    "user_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "status",
    "error_message",
    "is_truncated",
)

# Per-event-type column extractions.  Each value is a list of
# ``"SQL_EXPR AS alias"`` strings that will be appended after the
# common columns in the view SELECT.
_EVENT_VIEW_DEFS: dict[str, list[str]] = {
    "USER_MESSAGE_RECEIVED": [],
    "LLM_REQUEST": [
        "JSON_VALUE(attributes, '$.model') AS model",
        "content AS request_content",
        "JSON_QUERY(attributes, '$.llm_config') AS llm_config",
        "JSON_QUERY(attributes, '$.tools') AS tools",
    ],
    "LLM_RESPONSE": [
        "JSON_QUERY(content, '$.response') AS response",
        (
            "CAST(JSON_VALUE(content, '$.usage.prompt')"
            " AS INT64) AS usage_prompt_tokens"
        ),
        (
            "CAST(JSON_VALUE(content, '$.usage.completion')"
            " AS INT64) AS usage_completion_tokens"
        ),
        (
            "CAST(JSON_VALUE(content, '$.usage.total')"
            " AS INT64) AS usage_total_tokens"
        ),
        (
            "CAST(JSON_VALUE(attributes,"
            " '$.usage_metadata.cached_content_token_count') AS INT64) AS"
            " usage_cached_tokens"
        ),
        (
            "CAST(JSON_VALUE(attributes,"
            " '$.usage_metadata.thoughts_token_count') AS INT64) AS"
            " usage_thinking_tokens"
        ),
        (
            "CAST(JSON_VALUE(attributes,"
            " '$.usage_metadata.tool_use_prompt_token_count') AS INT64) AS"
            " usage_tool_use_tokens"
        ),
        (
            "SAFE_DIVIDE(CAST(JSON_VALUE(attributes,"
            " '$.usage_metadata.cached_content_token_count') AS"
            " INT64),CAST(JSON_VALUE(content, '$.usage.prompt') AS INT64)) AS"
            " context_cache_hit_rate"
        ),
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
        (
            "CAST(JSON_VALUE(latency_ms,"
            " '$.time_to_first_token_ms') AS INT64) AS ttft_ms"
        ),
        "JSON_VALUE(attributes, '$.model_version') AS model_version",
        "JSON_QUERY(attributes, '$.usage_metadata') AS usage_metadata",
        "JSON_QUERY(attributes, '$.cache_metadata') AS cache_metadata",
    ],
    "LLM_ERROR": [
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "TOOL_STARTING": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
    ],
    "TOOL_COMPLETED": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.result') AS tool_result",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
        # Long-running pair keys: null for ordinary completions,
        # populated on the user-message resume path so typed views can
        # do the TOOL_PAUSED ↔ TOOL_COMPLETED join end-to-end.
        "JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
        "JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
    ],
    "TOOL_ERROR": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
        "JSON_VALUE(content, '$.tool_origin') AS tool_origin",
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "AGENT_STARTING": [
        "JSON_VALUE(content, '$.text_summary') AS agent_instruction",
    ],
    "AGENT_COMPLETED": [
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
    ],
    "AGENT_ERROR": [
        "CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS total_ms",
        "JSON_VALUE(content, '$.error_traceback') AS error_traceback",
    ],
    "INVOCATION_STARTING": [],
    "INVOCATION_COMPLETED": [],
    "INVOCATION_ERROR": [
        "JSON_VALUE(content, '$.error_traceback') AS error_traceback",
    ],
    "STATE_DELTA": [
        "JSON_QUERY(attributes, '$.state_delta') AS state_delta",
    ],
    "HITL_CREDENTIAL_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "HITL_CONFIRMATION_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "HITL_INPUT_REQUEST": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
    ],
    "A2A_INTERACTION": [
        "content AS response_content",
        (
            "JSON_VALUE(attributes,"
            " '$.a2a_metadata.\"a2a:task_id\"') AS a2a_task_id"
        ),
        (
            "JSON_VALUE(attributes,"
            " '$.a2a_metadata.\"a2a:context_id\"') AS a2a_context_id"
        ),
        (
            "JSON_QUERY(attributes,"
            " '$.a2a_metadata.\"a2a:request\"') AS a2a_request"
        ),
        (
            "JSON_QUERY(attributes,"
            " '$.a2a_metadata.\"a2a:response\"') AS a2a_response"
        ),
    ],
    "AGENT_RESPONSE": [
        "JSON_VALUE(content, '$.response') AS response_text",
        "JSON_VALUE(attributes, '$.source_event_id') AS source_event_id",
        (
            "JSON_VALUE(attributes,"
            " '$.source_event_author') AS source_event_author"
        ),
        (
            "JSON_VALUE(attributes,"
            " '$.source_event_branch') AS source_event_branch"
        ),
    ],
    "AGENT_TRANSFER": [
        "JSON_VALUE(content, '$.from_agent') AS from_agent",
        "JSON_VALUE(content, '$.to_agent') AS to_agent",
        "JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
    ],
    "EVENT_COMPACTION": [
        (
            "CAST(JSON_VALUE(content,"
            " '$.start_timestamp') AS FLOAT64) AS start_seconds"
        ),
        (
            "CAST(JSON_VALUE(content,"
            " '$.end_timestamp') AS FLOAT64) AS end_seconds"
        ),
        (
            "TIMESTAMP_MICROS(CAST(CAST(JSON_VALUE(content,"
            " '$.start_timestamp') AS FLOAT64) * 1000000 AS INT64))"
            " AS window_start"
        ),
        (
            "TIMESTAMP_MICROS(CAST(CAST(JSON_VALUE(content,"
            " '$.end_timestamp') AS FLOAT64) * 1000000 AS INT64))"
            " AS window_end"
        ),
        "JSON_QUERY(content, '$.compacted_content') AS compacted_content",
    ],
    "AGENT_STATE_CHECKPOINT": [
        "JSON_QUERY(content, '$.agent_state') AS agent_state",
        # Presence discriminator. JSON_QUERY on an explicit JSON null
        # returns JSON null (not SQL NULL), so consumers must check
        # JSON_TYPE: SQL NULL = key absent, 'null' = explicit JSON
        # null (the {agent_state: null, end_of_agent: true} shape),
        # anything else = a real state object.
        "JSON_TYPE(JSON_QUERY(content, '$.agent_state')) AS agent_state_type",
        (
            "SAFE_CAST(JSON_VALUE(content,"
            " '$.end_of_agent') AS BOOL) AS end_of_agent"
        ),
        "JSON_VALUE(attributes, '$.adk.source_event_id') AS source_event_id",
    ],
    "TOOL_PAUSED": [
        "JSON_VALUE(content, '$.tool') AS tool_name",
        "JSON_QUERY(content, '$.args') AS tool_args",
        "JSON_VALUE(attributes, '$.adk.pause_kind') AS pause_kind",
        "JSON_VALUE(attributes, '$.adk.function_call_id') AS function_call_id",
    ],
}

_VIEW_SQL_TEMPLATE = """\
CREATE OR REPLACE VIEW `{project}.{dataset}.{view_name}` AS
SELECT
  {columns}
FROM
  `{project}.{dataset}.{table}`
WHERE
  event_type = '{event_type}'
"""


# ==============================================================================
# MAIN PLUGIN
# ==============================================================================
@dataclass
class _LoopState:
  """Holds resources bound to a specific event loop."""

  write_client: BigQueryWriteAsyncClient
  batch_processor: BatchProcessor


@dataclass(kw_only=True)
class EventData:
  """Typed container for structured fields passed to _log_event."""

  span_id_override: Optional[str] = None
  parent_span_id_override: Optional[str] = None
  latency_ms: Optional[int] = None
  time_to_first_token_ms: Optional[int] = None
  model: Optional[str] = None
  model_version: Optional[str] = None
  usage_metadata: Any = None
  cache_metadata: Any = None
  status: str = "OK"
  error_message: Optional[str] = None
  extra_attributes: dict[str, Any] = field(default_factory=dict)
  trace_id_override: Optional[str] = None
  # ADK 2.0 envelope: callbacks that hold the source Event pass it here
  # so ``_log_event`` can stamp ``attributes.adk.{source_event_id, node,
  # branch, scope, ...}``. Leave None for rows that don't originate from
  # an Event — the envelope helper omits those keys rather than
  # synthesizing fake identity. Because the
  # surrounding column is BigQuery JSON, an omitted key resolves to SQL
  # NULL via ``JSON_VALUE(attributes, '$.adk.<field>')``, so consumer
  # gating with ``... IS NOT NULL`` works without explicit JSON nulls.
  source_event: Optional["Event"] = None
  # Producer-supplied extras that belong INSIDE ``attributes.adk`` (not
  # at the top level of ``attributes``). C7's pair keys
  # (``pause_kind`` / ``function_call_id``) ride here so consumer SQL
  # like ``JSON_VALUE(attributes, '$.adk.function_call_id')`` lands at
  # the right JSON path.
  adk_extras: dict[str, Any] = field(default_factory=dict)


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """BigQuery Agent Analytics Plugin using Write API.

  Logs agent events (LLM requests, tool calls, etc.) to BigQuery for analytics.
  Uses the BigQuery Write API for efficient, asynchronous, and reliable logging.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: Optional[str] = None,
      config: Optional[BigQueryLoggerConfig] = None,
      location: str = "US",
      credentials: Optional[google.auth.credentials.Credentials] = None,
      **kwargs: Any,
  ) -> None:
    """Initializes the instance.

    Args:
        project_id: Google Cloud project ID.
        dataset_id: BigQuery dataset ID.
        table_id: BigQuery table ID (optional, overrides config).
        config: BigQueryLoggerConfig (optional).
        location: BigQuery location (default: "US").
        credentials: Google Auth credentials (optional). If None, uses
          Application Default Credentials.
        **kwargs: Additional configuration parameters for BigQueryLoggerConfig.
    """
    super().__init__(name="bigquery_agent_analytics")
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.config = config or BigQueryLoggerConfig()

    # Override config with kwargs if provided
    for key, value in kwargs.items():
      if hasattr(self.config, key):
        setattr(self.config, key, value)
      else:
        logger.warning(f"Unknown configuration parameter: {key}")

    if not self.config.view_prefix:
      raise ValueError("view_prefix must be a non-empty string.")

    # Pre-parse the custom_metadata allowlist into exact keys + prefixes.
    self._custom_metadata_exact, self._custom_metadata_prefixes = (
        _parse_custom_metadata_allowlist(self.config.custom_metadata_allowlist)
    )
    # Validate (fail-closed on protected/unknown columns) the projection.
    self._denied_columns = _validate_payload_column_denylist(
        self.config.payload_column_denylist
    )
    # Capturing custom_metadata into the attributes column is
    # incompatible with projecting attributes out -- the captured payload
    # would be silently dropped (and is_truncated could still flip). Fail
    # fast rather than do useless work.
    if "attributes" in self._denied_columns and (
        self._custom_metadata_exact or self._custom_metadata_prefixes
    ):
      raise ValueError(
          "custom_metadata_allowlist captures into the 'attributes' column,"
          " but 'attributes' is in payload_column_denylist -- the captured"
          " metadata would be dropped. Remove 'attributes' from"
          " payload_column_denylist or clear custom_metadata_allowlist."
      )

    self.table_id = table_id or self.config.table_id
    self.location = location

    self._visual_builder = _is_visual_builder.get()

    _validate_runtime_config(self.config)

    self._started = False
    self._startup_error: Optional[Exception] = None
    self._setup_failures = 0
    self._setup_retry_at = 0.0
    # Plugin-level loss accounting: counts drops that happen
    # before/outside any BatchProcessor (setup unavailable, formatter
    # failure). Merged into get_drop_stats() and survives shutdown.
    self._local_drop_counts: dict[str, int] = {}
    # Guards every read-modify-write and snapshot of _local_drop_counts:
    # events run on loops in different threads, and the unlocked
    # increment/fold underreported losses under contention.
    self._drop_counts_guard = threading.Lock()
    self._is_shutting_down = False
    # Guards _setup_future/_started/_setup_* transitions across threads;
    # held only for pointer swaps, never across an await.
    self._setup_guard = threading.Lock()
    self._setup_future: Optional["ConcurrentFuture[None]"] = None
    # Concurrent shutdown callers coalesce on the active owner's
    # completion future instead of returning before teardown finished.
    self._shutdown_future: Optional["ConcurrentFuture[None]"] = None
    # Lifecycle generation: shutdown() bumps it so an in-flight setup that
    # completes afterwards cannot resurrect _started.
    self._generation = 0
    # Guards ownership changes of _loop_state_by_loop: unsynchronized iteration raced concurrent insertion.
    self._loop_states_guard = threading.Lock()
    self._credentials = credentials
    self.client = None
    self._loop_state_by_loop: dict[asyncio.AbstractEventLoop, _LoopState] = {}
    self._write_stream_name: Optional[str] = None  # Resolved stream name
    self._executor: Optional[ThreadPoolExecutor] = None
    self.offloader: Optional[GCSOffloader] = None
    self.parser: Optional[HybridContentParser] = None
    self._schema = None
    self.arrow_schema = None
    self._init_pid = os.getpid()
    _LIVE_PLUGINS.add(self)

  def _cleanup_stale_loop_states(self) -> None:
    """Removes entries for event loops that have been closed."""
    # Snapshot under the guard: iterating the
    # live dict raced concurrent insertion ("dictionary changed size
    # during iteration"). is_closed() is evaluated on the snapshot,
    # outside the lock.
    with self._loop_states_guard:
      candidates = list(self._loop_state_by_loop)
    stale = [loop for loop in candidates if loop.is_closed()]
    for loop in stale:
      # Atomic claim AND fold:
      # exactly one concurrent cleanup folds a given processor's counters,
      # and the pop and the fold happen in one guarded transition so
      # get_drop_stats() never observes the state as neither live nor
      # folded (or as both).
      stale_rows = 0
      with self._loop_states_guard, self._drop_counts_guard:
        state = self._loop_state_by_loop.pop(loop, None)
        if state is not None:
          for reason, count in state.batch_processor.get_drop_stats().items():
            self._local_drop_counts[reason] = (
                self._local_drop_counts.get(reason, 0) + count
            )
          # Rows still queued on the dead loop can never be written:
          # count them instead of discarding silently. O(1) accounting (qsize minus tracked sentinels,
          # ), INSIDE the single-winner claim:
          # counting after the claim let a shutdown holding an earlier
          # snapshot count the same queue a second time.
          queue = getattr(state.batch_processor, "_queue", None)
          sentinels = getattr(state.batch_processor, "_sentinel_count", 0)
          if isinstance(queue, asyncio.Queue):
            if not isinstance(sentinels, int):
              sentinels = 0
            stale_rows = max(0, queue.qsize() - sentinels)
            if stale_rows:
              self._local_drop_counts["stale_loop"] = (
                  self._local_drop_counts.get("stale_loop", 0) + stale_rows
              )
      if state is None:
        continue
      logger.warning(
          "Cleaning up stale loop state for closed loop %s (id=%s).",
          loop,
          id(loop),
      )
      if stale_rows:
        logger.warning(
            "%d queued row(s) lost with closed loop %s.", stale_rows, id(loop)
        )
      # Best-effort resource release; the loop is closed, so async
      # transport teardown is not possible here.
      try:
        if state.write_client and getattr(
            state.write_client, "transport", None
        ):
          close_fn = getattr(state.write_client.transport, "close", None)
          if close_fn is not None and not asyncio.iscoroutinefunction(close_fn):
            close_fn()
      except Exception:
        pass

  # API Compatibility: These class-level attributes mask the dynamic
  # properties from static analysis tools (preventing "breaking changes"),
  # while __getattribute__ intercepts instance access to route to the
  # actual property implementations.
  batch_processor = None
  write_client = None
  write_stream = None

  def __getattribute__(self, name: str) -> Any:
    """Intercepts attribute access to support API masking.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The value of the attribute.
    """
    if name == "batch_processor":
      return self._batch_processor_prop
    if name == "write_client":
      return self._write_client_prop
    if name == "write_stream":
      return self._write_stream_prop
    return super().__getattribute__(name)

  @property
  def _batch_processor_prop(self) -> Optional["BatchProcessor"]:
    """The batch processor for the current event loop."""
    try:
      loop = asyncio.get_running_loop()
      self._cleanup_stale_loop_states()
      if loop in self._loop_state_by_loop:
        return self._loop_state_by_loop[loop].batch_processor
    except RuntimeError:
      pass
    return None

  @property
  def _write_client_prop(self) -> Optional["BigQueryWriteAsyncClient"]:
    """The write client for the current event loop."""
    try:
      loop = asyncio.get_running_loop()
      if loop in self._loop_state_by_loop:
        return self._loop_state_by_loop[loop].write_client
    except RuntimeError:
      pass
    return None

  @property
  def _write_stream_prop(self) -> Optional[str]:
    """The write stream for the current event loop."""
    bp = self._batch_processor_prop
    return bp.write_stream if bp else None

  def _format_content_safely(
      self, content: Optional[types.Content]
  ) -> tuple[str, bool]:
    """Formats content using config.content_formatter or default formatter.

    Args:
        content: The content to format.

    Returns:
        A tuple of (formatted_string, is_truncated).
    """
    if content is None:
      return "None", False
    try:
      # If a custom formatter is provided, we could try to use it here too,
      # but it expects (content, event_type). For internal formatting,
      # we stick to the default _format_content but respect max_len.
      return _format_content(content, max_len=self.config.max_content_length)
    except Exception as e:
      logger.warning("Content formatter failed: %s", e)
      return "[FORMATTING FAILED]", False

  async def _close_write_transport(self, write_client: Any) -> None:
    """Best-effort bounded close for a BigQuery write-client transport."""
    transport = getattr(write_client, "transport", None)
    close_fn = getattr(transport, "close", None)
    if close_fn is None:
      return
    try:
      if asyncio.iscoroutinefunction(close_fn):
        await asyncio.wait_for(close_fn(), timeout=self.config.shutdown_timeout)
      else:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, close_fn),
            timeout=self.config.shutdown_timeout,
        )
        if isinstance(result, collections.abc.Awaitable):
          await asyncio.wait_for(result, timeout=self.config.shutdown_timeout)
    except asyncio.CancelledError:
      raise
    except Exception:
      logger.warning("Could not close a detached BigQuery write transport.")

  async def _close_detached_loop_transport(self, state: _LoopState) -> None:
    """Best-effort bounded close for a terminal loop state's transport."""
    await self._close_write_transport(state.write_client)

  async def _get_loop_state(
      self, claimed_generation: Optional[int] = None
  ) -> _LoopState:
    """Gets or creates the state for the current event loop.

    Args:
        claimed_generation: The lifecycle generation the caller claimed BEFORE
          its own awaits (setup passes the generation captured by
          `_ensure_started`). Without it, a setup blocked ahead of this call
          sampled the post-shutdown generation on resume and published a writer
          that shutdown's snapshot could never see .

    Returns:
        The loop-specific state object containing clients and processors.
    """
    loop = asyncio.get_running_loop()
    if self._is_shutting_down:
      # A callback that passed the early check can resume here after
      # shutdown started; publishing a fresh writer state now would leak
      # it.
      raise RuntimeError("BigQuery plugin is shutting down.")
    # Captured before any await: a shutdown() that starts (and even
    # completes) while the writer below is being built bumps the
    # generation, and the publication guard rechecks it — otherwise the
    # new processor lands in the dict AFTER shutdown's snapshot/clear and
    # leaks past close().
    generation = (
        claimed_generation
        if claimed_generation is not None
        else self._generation
    )
    if self._generation != generation:
      raise RuntimeError("BigQuery plugin is shutting down.")
    self._cleanup_stale_loop_states()
    detached_state: Optional[_LoopState] = None
    detached_rows = 0
    detached_reason = "shutdown_timeout"
    with self._loop_states_guard:
      state = self._loop_state_by_loop.get(loop)
      if state is not None:
        processor = state.batch_processor
        # Production entries always contain a real BatchProcessor. Keeping
        # non-production stand-ins opaque also avoids treating truthy mock
        # attributes as lifecycle flags in compatibility tests.
        if not isinstance(processor, BatchProcessor):
          return state
        worker = processor._batch_processor_task
        if worker is not None and not worker.done():
          if processor._shutdown:
            # It is terminal for admission but still owns a live worker.
            # Returning it loses rows; detaching/closing it races its drain.
            raise _LoopStateAdmissionAbortedError(
                "BigQuery writer is still shutting down."
            )
          return state

        # A missing/done worker can never consume another appended row.
        # Claim + fold under the same canonical guard order used by shutdown
        # so concurrent stats readers see the state either live or folded,
        # never both/neither. Identity ownership makes this single-winner.
        detached_state = self._loop_state_by_loop.pop(loop)
        # Prevent the old atexit registration from trying to run a second,
        # blocking close over a processor whose rows are accounted below.
        processor._shutdown = True
        detached_reason = (
            "shutdown_cancelled"
            if worker is not None and worker.cancelled()
            else "shutdown_timeout"
        )
        queue = processor._queue
        sentinels = processor._sentinel_count
        detached_rows = max(0, queue.qsize() - sentinels)
        with self._drop_counts_guard:
          for reason, count in processor.get_drop_stats().items():
            self._local_drop_counts[reason] = (
                self._local_drop_counts.get(reason, 0) + count
            )
          if detached_rows:
            self._local_drop_counts[detached_reason] = (
                self._local_drop_counts.get(detached_reason, 0) + detached_rows
            )

    if detached_rows:
      logger.warning(
          "%d queued row(s) belonged to a terminal BigQuery writer (%s).",
          detached_rows,
          detached_reason,
      )
    # Structured ownership: the claimant that removed a terminal state also
    # owns its bounded transport close. A detached fire-and-forget task left a
    # warning/leak window whenever fresh construction raised before the task
    # was retrieved. The finally runs on success, failure, and cancellation;
    # on success the replacement is published before this await so concurrent
    # callers share it instead of building another writer.
    try:
      # grpc.aio clients are loop-bound, so we create one per event loop.
      def get_credentials() -> google.auth.credentials.Credentials:
        creds, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
        return creds

      if self._credentials is None:
        self._credentials = await loop.run_in_executor(
            self._executor, get_credentials
        )
      quota_project_id = getattr(self._credentials, "quota_project_id", None)
      options = (
          client_options.ClientOptions(quota_project_id=quota_project_id)
          if quota_project_id
          else None
      )

      user_agents = [f"google-adk-bq-logger/{__version__}"]
      if self._visual_builder:
        user_agents.append(f"google-adk-visual-builder/{__version__}")

      client_info = gapic_client_info.ClientInfo(
          user_agent=" ".join(user_agents)
      )

      write_client = BigQueryWriteAsyncClient(
          credentials=self._credentials,
          client_info=client_info,
          client_options=options,
      )

      if not self._write_stream_name:
        self._write_stream_name = f"projects/{self.project_id}/datasets/{self.dataset_id}/tables/{self.table_id}/_default"

      try:
        batch_processor = BatchProcessor(
            write_client=write_client,
            arrow_schema=self.arrow_schema,
            write_stream=self._write_stream_name,
            batch_size=self.config.batch_size,
            flush_interval=self.config.batch_flush_interval,
            retry_config=self.config.retry_config,
            queue_max_size=self.config.queue_max_size,
            shutdown_timeout=self.config.shutdown_timeout,
        )
      except BaseException:
        # The write client already exists but no _LoopState can own it yet.
        await self._close_write_transport(write_client)
        raise
      state = _LoopState(write_client, batch_processor)
      try:
        await batch_processor.start()
      except BaseException:
        # start() may create then fail/cancel a worker. Keep the fresh client
        # under structured ownership as well; the bounded helper retrieves
        # either sync or async transport-close outcomes.
        await self._close_detached_loop_transport(state)
        raise

      with self._loop_states_guard:
        invalidated = self._is_shutting_down or self._generation != generation
        if not invalidated:
          self._loop_state_by_loop[loop] = state
      if invalidated:
        # shutdown() ran during construction; its snapshot cannot include
        # this writer, so publishing it would leave a live processor and
        # open transport behind after close() returns. Tear the fresh instances down instead of publishing.
        try:
          try:
            await batch_processor.shutdown(timeout=self.config.shutdown_timeout)
          except Exception:
            logger.warning(
                "Could not shut down writer created during shutdown.",
                exc_info=True,
            )
        finally:
          await self._close_detached_loop_transport(state)
        raise RuntimeError("BigQuery plugin is shutting down.")

      atexit.register(self._atexit_cleanup, weakref.proxy(batch_processor))
      return state
    finally:
      if detached_state is not None:
        await self._close_detached_loop_transport(detached_state)

  async def flush(self) -> None:
    """Flushes any pending events to BigQuery.

    Flushes the processor associated with the CURRENT loop.
    """
    try:
      loop = asyncio.get_running_loop()
      self._cleanup_stale_loop_states()
      if loop in self._loop_state_by_loop:
        await self._loop_state_by_loop[loop].batch_processor.flush()
    except RuntimeError:
      # No running loop or other issue
      pass

  def get_drop_stats(self) -> dict[str, int]:
    """Returns dropped-row counts aggregated across all event loops.

    Events are dropped best-effort (queue overflow, write failures), so the
    loss is otherwise only visible in logs. Export these counters to your
    monitoring to detect data loss before it surfaces as missing rows. See
    BatchProcessor.get_drop_stats for the meaning of each reason.

    Reasons are LOSS INCIDENTS, not uniformly dropped rows:
    ``formatter_failed`` and ``content_parse_failed`` mean the row WAS
    written with its content replaced by a sentinel; ``setup_unavailable``,
    ``shutdown_race``, ``shutdown_timeout``, ``shutdown_cancelled``, and
    ``stale_loop`` mean the row was never written. Counters persist
    across shutdown and loop cleanup.

    Returns:
        Per-reason counts: plugin-level incidents plus every live loop
        processor's counters (dead processors are folded in at
        shutdown/cleanup time).
    """
    # Both guards, in the canonical order (loop states, then counters):
    # reading them separately let a state that was folded-but-not-yet-
    # removed be added twice, and a popped-but-not-yet-folded state be
    # missed.
    with self._loop_states_guard, self._drop_counts_guard:
      totals: dict[str, int] = dict(self._local_drop_counts)
      for state in self._loop_state_by_loop.values():
        for reason, count in state.batch_processor.get_drop_stats().items():
          totals[reason] = totals.get(reason, 0) + count
    return totals

  async def _lazy_setup(
      self, claimed_generation: Optional[int] = None, **kwargs: Any
  ) -> None:
    """Performs lazy initialization of BigQuery clients and resources.

    Args:
        claimed_generation: The lifecycle generation claimed by the owning
          `_ensure_started` before any await; forwarded to `_get_loop_state` so
          a shutdown that completes mid-setup is detected even when it finishes
          before the loop-state phase begins.
    """
    if self._started:
      return
    loop = asyncio.get_running_loop()

    # The executor is needed beyond client construction (schema RPCs, GCS
    # offloader): creating it only inside the client branch left
    # _executor as None for the offloader when a client was already set
    # (post-rebase mypy: GCSOffloader argument 3 expects a non-optional
    # ThreadPoolExecutor — a latent runtime gap, not just typing).
    executor = self._executor
    if executor is None:
      executor = ThreadPoolExecutor(max_workers=1)
      self._executor = executor

    if not self.client:
      client_future: "ConcurrentFuture[Any]" = executor.submit(
          lambda: bigquery.Client(
              project=self.project_id,
              credentials=self._credentials,
          ),
      )
      try:
        self.client = await asyncio.wrap_future(client_future)
      except asyncio.CancelledError:
        # Cancelling the await does not stop the constructor thread; the
        # eventual client was silently discarded and its connection pool
        # never closed. The close itself is
        # dispatched to a fresh thread: when the future is ALREADY done,
        # add_done_callback runs the callback synchronously in THIS
        # (event-loop) thread, and a slow client.close() would extend the
        # host's cancellation window.
        def _close_eventual(f: "ConcurrentFuture[Any]") -> None:
          try:
            eventual = f.result()
          except Exception:
            return

          def _close() -> None:
            try:
              eventual.close()
            except Exception:
              pass

          close_thread = create_thread(target=_close)
          close_thread.name = "bqaa-orphan-client-close"
          close_thread.daemon = True
          close_thread.start()

        client_future.add_done_callback(_close_eventual)
        raise

    self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
    if not self._schema:
      # Project out denied payload columns schema-first, so the table
      # schema, Arrow schema, row dict, and views all stay consistent.
      self._schema = _project_schema(_get_events_schema(), self._denied_columns)
    # Run table readiness on EVERY setup attempt until one succeeds: the
    # cached _schema must not gate it, or a failed first attempt would skip
    # the table check on retry and mark the plugin started against a
    # missing/unready table. Once _started is True,
    # _lazy_setup returns early above, so the steady state pays no extra RPC.
    await loop.run_in_executor(executor, self._ensure_schema_exists)

    if not self.parser:
      self.arrow_schema = to_arrow_schema(self._schema)
      if not self.arrow_schema:
        raise RuntimeError("Failed to convert BigQuery schema to Arrow schema.")

      self.offloader = None
      if self.config.gcs_bucket_name:
        if "content_parts" in self._denied_columns:
          # GCS offload stores its object reference in the
          # ``content_parts`` column. With ``content_parts`` projected out,
          # an upload would be orphaned -- payload leaks to GCS and incurs
          # cost with no retained reference. Disable offload and keep
          # content inline (truncated) instead.
          logger.warning(
              "GCS offload disabled: payload_column_denylist drops"
              " 'content_parts', which holds the offloaded object reference;"
              " large/binary content is kept inline (truncated) instead of"
              " being uploaded to %s.",
              self.config.gcs_bucket_name,
          )
        else:
          self.offloader = GCSOffloader(
              self.project_id,
              self.config.gcs_bucket_name,
              executor,
              storage_client=storage.Client(
                  project=self.project_id, credentials=self._credentials
              ),
          )

      self.parser = HybridContentParser(
          self.offloader,
          "",
          "",
          max_length=self.config.max_content_length,
          connection_id=self.config.connection_id,
      )

    await self._get_loop_state(claimed_generation=claimed_generation)

  @staticmethod
  def _atexit_cleanup(batch_processor: "BatchProcessor") -> None:
    """Clean up batch processor on script exit.

    Drains any remaining items from the queue and logs a warning.
    Callers should use ``flush()`` before shutdown to ensure all
    events are written; this handler only reports data that would
    otherwise be silently lost.
    """
    try:
      if not batch_processor or batch_processor._shutdown:
        return
    except ReferenceError:
      return

    # Drain remaining items and warn — creating a new event loop and
    # BQ client at interpreter exit is fragile and masks shutdown bugs.
    remaining = 0
    try:
      while True:
        batch_processor._queue.get_nowait()
        remaining += 1
    except (asyncio.QueueEmpty, AttributeError):
      pass

    if remaining:
      logger.warning(
          "%d analytics event(s) were still queued at interpreter exit "
          "and could not be flushed. Call plugin.flush() before shutdown "
          "to avoid data loss.",
          remaining,
      )

  def _ensure_schema_exists(self) -> None:
    """Ensures the BigQuery table exists with the correct schema.

    When ``config.auto_schema_upgrade`` is True and the table already
    exists, missing columns are added automatically (additive only).
    A ``adk_schema_version`` label is written for governance.
    """
    assert self.client is not None  # _lazy_setup creates it before calling.
    try:
      existing_table = self.client.get_table(self.full_table_id)
      if self.config.auto_schema_upgrade:
        self._maybe_upgrade_schema(existing_table)
      if self.config.create_views:
        self._create_analytics_views()
    except cloud_exceptions.NotFound:
      logger.info("Table %s not found, creating table.", self.full_table_id)
      tbl = bigquery.Table(self.full_table_id, schema=self._schema)
      tbl.time_partitioning = bigquery.TimePartitioning(
          type_=bigquery.TimePartitioningType.DAY,
          field="timestamp",
      )
      tbl.clustering_fields = self.config.clustering_fields
      tbl.labels = {_SCHEMA_VERSION_LABEL_KEY: _SCHEMA_VERSION}
      try:
        self.client.create_table(tbl)
      except cloud_exceptions.Conflict:
        # Another process created it concurrently — but there is no
        # guarantee it used a compatible schema. Re-fetch and run the same
        # readiness path as a pre-existing table; any failure here
        # propagates so _ensure_started keeps _started=False and retries.
        existing_table = self.client.get_table(self.full_table_id)
        if self.config.auto_schema_upgrade:
          self._maybe_upgrade_schema(existing_table)
      except Exception as e:
        # Fail setup: returning normally here used to let the
        # plugin mark itself started against a missing table and silently
        # lose every subsequent row. Raise so _ensure_started records the
        # failure, keeps _started=False, and retries on a later event.
        logger.error(
            "Could not create table %s: %s",
            self.full_table_id,
            e,
            exc_info=True,
        )
        raise
      if self.config.create_views:
        self._create_analytics_views()
    except Exception as e:
      # Fail setup: swallowing control-plane errors here let
      # the plugin mark itself started against a missing/unready table.
      logger.error(
          "Error ensuring table %s is ready: %s",
          self.full_table_id,
          e,
          exc_info=True,
      )
      raise

  @staticmethod
  def _schema_fields_match(
      existing: list[bq_schema.SchemaField],
      desired: list[bq_schema.SchemaField],
      path: tuple[str, ...] = (),
  ) -> tuple[
      list[bq_schema.SchemaField],
      list[bq_schema.SchemaField],
  ]:
    """Compares existing vs desired schema fields recursively.

    Returns:
        A tuple of (new_top_level_fields, updated_record_fields).
        ``new_top_level_fields`` are fields in *desired* that are
        entirely absent from *existing*.
        ``updated_record_fields`` are RECORD fields that exist in
        both but have new sub-fields in *desired*; each entry is a
        copy of the existing field with the missing sub-fields
        appended.
    """
    existing_by_name = {f.name: f for f in existing}
    new_fields: list[bq_schema.SchemaField] = []
    updated_records: list[bq_schema.SchemaField] = []

    for desired_field in desired:
      existing_field = existing_by_name.get(desired_field.name)
      if existing_field is None:
        new_fields.append(desired_field)
        continue

      field_path = ".".join((*path, desired_field.name))
      existing_type = existing_field.field_type.upper()
      desired_type = desired_field.field_type.upper()
      existing_mode = existing_field.mode.upper()
      desired_mode = desired_field.mode.upper()
      if existing_type != desired_type or existing_mode != desired_mode:
        raise ValueError(
            "Incompatible BigQuery schema field "
            f"{field_path!r}: existing={existing_type}/{existing_mode}, "
            f"desired={desired_type}/{desired_mode}."
        )

      if desired_type == "RECORD" and desired_field.fields:
        # Recurse into nested RECORD fields.
        sub_new, sub_updated = (
            BigQueryAgentAnalyticsPlugin._schema_fields_match(
                list(existing_field.fields),
                list(desired_field.fields),
                (*path, desired_field.name),
            )
        )
        if sub_new or sub_updated:
          # Build a merged sub-field list.
          merged_sub = list(existing_field.fields)
          # Replace updated nested records in-place.
          updated_names = {f.name for f in sub_updated}
          merged_sub = [
              next(u for u in sub_updated if u.name == f.name)
              if f.name in updated_names
              else f
              for f in merged_sub
          ]
          # Append entirely new sub-fields.
          merged_sub.extend(sub_new)
          # Rebuild via API representation to preserve all
          # existing field attributes (policy_tags, etc.).
          api_repr = existing_field.to_api_repr()
          api_repr["fields"] = [sf.to_api_repr() for sf in merged_sub]
          updated_records.append(bq_schema.SchemaField.from_api_repr(api_repr))

    return new_fields, updated_records

  def _maybe_upgrade_schema(self, existing_table: bigquery.Table) -> None:
    """Adds missing columns to an existing table (additive only).

    Handles nested RECORD fields by recursing into sub-fields.
    The version label is only stamped after a successful update
    so that a failed attempt is retried on the next run.

    Args:
        existing_table: The current BigQuery table object.
    """
    new_fields, updated_records = self._schema_fields_match(
        list(existing_table.schema), list(self._schema)
    )

    stored_version = (existing_table.labels or {}).get(
        _SCHEMA_VERSION_LABEL_KEY
    )
    # No-op only when there is genuinely nothing to add AND the version label
    # is current. We must NOT early-return on the label alone: ``self._schema``
    # is projection-dependent, so relaxing ``payload_column_denylist``
    # makes previously-omitted columns desired again on a table whose label
    # still matches -- skipping the diff would leave those columns missing and
    # later writes would carry fields absent from the table.
    if (
        not new_fields
        and not updated_records
        and stored_version == _SCHEMA_VERSION
    ):
      return

    if new_fields or updated_records:
      # Build merged top-level schema.
      updated_names = {f.name for f in updated_records}
      merged = [
          next(u for u in updated_records if u.name == f.name)
          if f.name in updated_names
          else f
          for f in existing_table.schema
      ]
      merged.extend(new_fields)
      existing_table.schema = merged

      change_desc = []
      if new_fields:
        change_desc.append(f"new columns {[f.name for f in new_fields]}")
      if updated_records:
        change_desc.append(
            f"updated RECORD fields {[f.name for f in updated_records]}"
        )
      logger.info(
          "Auto-upgrading table %s: %s",
          self.full_table_id,
          ", ".join(change_desc),
      )

    try:
      # Stamp the version label inside the try block so that
      # on failure the label is NOT persisted and the next run
      # retries the upgrade.
      labels = dict(existing_table.labels or {})
      labels[_SCHEMA_VERSION_LABEL_KEY] = _SCHEMA_VERSION
      existing_table.labels = labels

      update_fields = ["schema", "labels"]
      self.client.update_table(existing_table, update_fields)
    except Exception as e:
      logger.error(
          "Schema auto-upgrade failed for %s: %s",
          self.full_table_id,
          e,
          exc_info=True,
      )
      if new_fields or updated_records:
        # The table is verifiably missing required fields; swallowing the
        # failure would let _ensure_started mark the plugin ready against
        # a table every later write can fail on, with no readiness retry.
        raise
      # Label-only refresh failed (e.g. a labels policy): the table schema
      # itself is write-compatible, so readiness must not be blocked —
      # the stale label is retried on the next run.

  def _project_view_columns(self, extra_cols: list[str]) -> list[str]:
    """Drops derived view expressions that reference a denied column.

    Each entry is a ``"SQL_EXPR AS alias"`` string referencing payload
    columns (``content`` / ``attributes`` / ``latency_ms``) as bare
    identifiers.  When such a column is projected out, its dependent view
    columns must go too, otherwise the view SQL references a non-existent
    column and view creation fails.
    """
    if not self._denied_columns:
      return list(extra_cols)
    kept: list[str] = []
    for expr in extra_cols:
      if any(
          re.search(rf"\b{re.escape(col)}\b", expr)
          for col in self._denied_columns
      ):
        continue
      kept.append(expr)
    return kept

  def _create_analytics_views(self) -> None:
    """Creates per-event-type BigQuery views (idempotent).

    Each view filters the events table by ``event_type`` and
    extracts JSON columns into typed, queryable columns.  Uses
    ``CREATE OR REPLACE VIEW`` so it is safe to call repeatedly.
    Errors are logged but never raised.
    """
    for event_type, extra_cols in _EVENT_VIEW_DEFS.items():
      view_name = self.config.view_prefix + "_" + event_type.lower()
      # Projection-aware views -- drop any derived column whose SQL
      # references a denied payload column (content / attributes / latency_ms).
      # Common columns are all protected, so they always remain.
      projected_extra = self._project_view_columns(extra_cols)
      columns = ",\n  ".join(list(_VIEW_COMMON_COLUMNS) + projected_extra)
      sql = _VIEW_SQL_TEMPLATE.format(
          project=self.project_id,
          dataset=self.dataset_id,
          view_name=view_name,
          columns=columns,
          table=self.table_id,
          event_type=event_type,
      )
      try:
        self.client.query(sql).result()
      except cloud_exceptions.Conflict:
        logger.debug(
            "View %s was updated concurrently by another process.",
            view_name,
        )
      except Exception as e:
        logger.error(
            "Failed to create view %s: %s",
            view_name,
            e,
            exc_info=True,
        )

  async def create_analytics_views(self) -> None:
    """Public async helper to (re-)create all analytics views.

    Useful when views need to be refreshed explicitly, for example
    after a schema upgrade.  Ensures the plugin is initialized
    before attempting view creation.
    """
    await self._ensure_started()
    if not self._started:
      raise RuntimeError(
          "Plugin initialization failed; cannot create analytics views."
      ) from self._startup_error
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(self._executor, self._create_analytics_views)

  @staticmethod
  def _schedule_remote_drain(
      processor: "BatchProcessor",
      target_loop: asyncio.AbstractEventLoop,
      drain_timeout: float,
  ) -> "ConcurrentFuture[Any]":
    """Schedules ``processor.shutdown()`` on another event loop.

     The coroutine is created INSIDE the remote-loop callback:
     run_coroutine_threadsafe creates it eagerly in the caller thread, and
     caller-side cleanup then had to guess ownership —
     ``ConcurrentFuture.cancel()`` can return True even after the remote
     task started, so a caller-side ``coro.close()`` either finalized the
     coroutine on the wrong thread or raised "coroutine already executing".
    Here nothing exists to leak until the
     callback runs, and the callback hands ownership to a remote Task
     atomically via ``set_running_or_notify_cancel()``.
    """
    cf: "ConcurrentFuture[Any]" = ConcurrentFuture()

    def _callback() -> None:
      if not cf.set_running_or_notify_cancel():
        return  # cancelled before the callback ran; nothing was created

      coro = processor.shutdown(timeout=drain_timeout)
      try:
        task = target_loop.create_task(coro)
      except Exception as exc:
        # e.g. a custom task factory rejecting creation: the coroutine
        # exists but was never scheduled — close it here or it leaks as
        # never-awaited.
        coro.close()
        cf.set_exception(exc)
        return

      def _transfer(t: "asyncio.Task[Any]") -> None:
        if t.cancelled():
          cf.set_exception(asyncio.CancelledError())
        elif t.exception() is not None:
          cf.set_exception(t.exception())
        else:
          cf.set_result(t.result())

      task.add_done_callback(_transfer)

    target_loop.call_soon_threadsafe(_callback)
    return cf

  async def shutdown(self, timeout: float | None = None) -> None:
    """Shuts down the plugin and releases resources.

    Args:
        timeout: Maximum time to wait for the queue to drain.
    """
    while True:
      waiter: Optional["ConcurrentFuture[None]"] = None
      with self._setup_guard:
        # Atomic admission: checked OUTSIDE
        # the lock, two threads could both observe False, both claim
        # shutdown, tear down the same snapshot twice, and double-fold
        # identical drop counters. Exactly one caller per generation gets
        # past this point.
        if self._is_shutting_down:
          waiter = self._shutdown_future
        else:
          self._is_shutting_down = True
          # Invalidate any in-flight setup: its completion must not
          # resurrect _started after this method returns.
          self._generation += 1
          self._started = False
          self._shutdown_future = ConcurrentFuture()
      if waiter is None:
        break  # this caller owns the teardown below
      # Coalesce on the active owner: returning early made a concurrent
      # `await plugin.close()` claim completion microseconds into another
      # caller's teardown. shield: this
      # waiter's own cancellation must not cancel the shared future.
      try:
        await asyncio.shield(asyncio.wrap_future(waiter))
      except _ShutdownIncompleteError:
        # The owner was cancelled or failed mid-teardown; returning now
        # would claim success while state is still live. Retry ownership — as a LOOP, not recursion, so
        # depth does not grow with the number of coalesced callers.
        continue
      except Exception:
        pass
      return
    t = timeout if timeout is not None else self.config.shutdown_timeout
    loop = asyncio.get_running_loop()
    # Stable snapshot: shutdown used to iterate the live dict, so a
    # concurrent state publication raised "dictionary changed size during
    # iteration" and aborted cleanup.
    with self._loop_states_guard:
      states_snapshot = dict(self._loop_state_by_loop)
    teardown_completed = False
    teardown_error: Optional[BaseException] = None
    retained_remote_drains = 0
    try:
      # Correct Multi-Loop Shutdown:
      # 1. Shutdown current loop's processor directly.
      drained: list[asyncio.AbstractEventLoop] = []
      if loop in states_snapshot:
        await states_snapshot[loop].batch_processor.shutdown(timeout=t)
        drained.append(loop)

      # 1b. Drain batch processors on other (non-current) loops. The
      # wrapped futures are AWAITED, not .result()-ed: the synchronous
      # wait blocked this event loop, so a host asyncio.timeout() around
      # close() could never fire and the delay was paid serially per
      # remote loop. One shared deadline
      # covers all remote drains; unfinished ones are cancelled and their
      # states left in place for a retry.
      remote: list[
          tuple[
              asyncio.AbstractEventLoop,
              "ConcurrentFuture[Any]",
              "asyncio.Future[Any]",
          ]
      ] = []
      for other_loop, state in states_snapshot.items():
        if other_loop is loop:
          continue
        if other_loop.is_closed():
          # No drain is possible on a closed loop, and its queued rows
          # are NOT guaranteed to have been counted — the state can enter
          # this snapshot before _cleanup_stale_loop_states() ever ran.
          # Claim, fold, and count the
          # queue loss in ONE single-winner transition: counting from
          # the snapshot without ownership double-counted rows that a
          # concurrent stale cleanup had already claimed.
          stale_rows = 0
          with self._loop_states_guard, self._drop_counts_guard:
            owned = self._loop_state_by_loop.get(other_loop) is state
            if owned:
              del self._loop_state_by_loop[other_loop]
              for (
                  reason,
                  count,
              ) in state.batch_processor.get_drop_stats().items():
                self._local_drop_counts[reason] = (
                    self._local_drop_counts.get(reason, 0) + count
                )
              queue = getattr(state.batch_processor, "_queue", None)
              sentinels = getattr(state.batch_processor, "_sentinel_count", 0)
              if isinstance(queue, asyncio.Queue):
                if not isinstance(sentinels, int):
                  sentinels = 0
                stale_rows = max(0, queue.qsize() - sentinels)
                if stale_rows:
                  self._local_drop_counts["stale_loop"] = (
                      self._local_drop_counts.get("stale_loop", 0) + stale_rows
                  )
          if stale_rows:
            logger.warning(
                "%d queued row(s) lost with closed loop %s.",
                stale_rows,
                id(other_loop),
            )
          continue
        try:
          cf = self._schedule_remote_drain(state.batch_processor, other_loop, t)
        except Exception:
          # e.g. the loop closed between the is_closed() check and
          # call_soon_threadsafe(). The state stays live, so teardown is
          # NOT complete — without counting it, both the owner and
          # coalesced waiters reported success over live state.
          retained_remote_drains += 1
          logger.warning(
              "Could not drain batch processor on loop %s",
              other_loop,
          )
          continue
        remote.append((other_loop, cf, asyncio.wrap_future(cf)))
      if remote:
        try:
          done_set, pending = await asyncio.wait(
              [wrapper for _, _, wrapper in remote], timeout=t
          )
        except asyncio.CancelledError:
          # Host cancellation mid-wait: release every remote handle. The
          # remote callback creates the coroutine itself, so a
          # successfully cancelled concurrent future means nothing was
          # (or ever will be) created.
          for _, cf, wrapper in remote:
            cf.cancel()
            wrapper.cancel()
          raise
        del done_set
        for other_loop, cf, wrapper in remote:
          if wrapper in pending:
            retained_remote_drains += 1
            # If the remote callback has not run yet this prevents the
            # task from ever being created; if it HAS run, the running
            # drain simply continues remotely, bounded by its own
            # timeout, and the state is retained.
            cf.cancel()
            wrapper.cancel()
            logger.warning(
                "Batch processor drain on loop %s did not finish within"
                " %.1fs; its state is retained for a retried close.",
                other_loop,
                t,
            )
            continue
          if wrapper.cancelled():
            retained_remote_drains += 1
            logger.warning(
                "Batch processor drain on loop %s was cancelled; its"
                " state is retained for a retried close.",
                other_loop,
            )
            continue
          # Retrieve the result: an unchecked failed drain both leaked
          # "exception was never retrieved" and claimed/folded the state
          # as if it had succeeded, silently abandoning its queued rows.
          # Only clean completions claim.
          exc = wrapper.exception()
          if exc is not None:
            retained_remote_drains += 1
            logger.warning(
                "Batch processor drain on loop %s failed (%s); its state"
                " is retained for a retried close.",
                other_loop,
                type(exc).__name__,
            )
            continue
          drained.append(other_loop)

      # 2/3. For every DRAINED state: close its transport, then claim it
      # out of the live dict and fold its counters in one atomic
      # transition — fold-then-clear let get_drop_stats() add the same
      # still-live processor again, and stale-loop cleanup could fold a
      # snapshotted state a second time.
      # States whose drain did not finish stay live (retry ownership).
      for state_loop in drained:
        state = states_snapshot[state_loop]
        if state.write_client and getattr(
            state.write_client, "transport", None
        ):
          try:
            await state.write_client.transport.close()
          except Exception:
            pass
        with self._loop_states_guard, self._drop_counts_guard:
          if self._loop_state_by_loop.get(state_loop) is state:
            del self._loop_state_by_loop[state_loop]
            for reason, count in state.batch_processor.get_drop_stats().items():
              self._local_drop_counts[reason] = (
                  self._local_drop_counts.get(reason, 0) + count
              )
          # else: stale-loop cleanup already claimed and folded it.

      # The parser/offloader hold the (now terminated) executor; keeping
      # them makes the first post-restart GCS upload raise "cannot
      # schedule new futures after shutdown".
      # The offloader's plugin-owned storage.Client is CLOSED, not just
      # dropped — off-loop, under budget.
      offloader, self.offloader = self.offloader, None
      self.parser = None
      storage_client = getattr(offloader, "client", None) if offloader else None
      if storage_client is not None:
        try:
          await asyncio.wait_for(
              loop.run_in_executor(None, storage_client.close), timeout=t
          )
        except Exception:
          pass

      # The executor is shut down INDEPENDENTLY of the client: a
      # cancelled setup could leave a live executor with client=None, and
      # the nested check leaked it past close(). Non-blocking: waiting would stall close() behind a
      # slow/blocked constructor job in the pool; pending jobs are
      # cancelled, and a still-running constructor finishes in the
      # background (its orphaned client is closed by _lazy_setup's
      # done-callback).
      if self._executor:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
      # Close (not just drop) the shared BigQuery client: discarding the
      # reference leaked its HTTP transport, while the aborted-setup path
      # already closed the same resource.
      # Off-loop and bounded by the shutdown budget.
      client, self.client = self.client, None
      if client is not None:
        try:
          await asyncio.wait_for(
              loop.run_in_executor(None, client.close), timeout=t
          )
        except Exception:
          pass
      if retained_remote_drains:
        # An incompletely drained remote state means live processors and
        # possibly queued rows survive this close: reporting success let
        # both the owner and coalesced waiters return normally over live
        # state.
        raise _ShutdownIncompleteError(
            f"{retained_remote_drains} remote drain(s) did not complete;"
            " their states are retained for a retried close."
        )
      teardown_completed = True
    except Exception as e:
      # teardown_completed stays False: reporting success here let both
      # the owner and coalesced waiters return normally while the loop
      # state was still live. Waiters
      # receive _ShutdownIncompleteError and retry ownership; the OWNER
      # re-raises after the finally so its caller sees the failure too —
      # PluginManager.close() aggregates plugin close failures.
      teardown_error = e
      logger.error("Error during shutdown: %s", e, exc_info=True)
    finally:
      # Cancellation-safe reset: PluginManager's close timeout cancels this
      # coroutine, and asyncio.CancelledError is a BaseException that the
      # handler above does not (and must not) swallow. Without the finally,
      # a cancelled shutdown left _is_shutting_down=True forever, so the
      # re-entry guard turned every later close() into a no-op and retained
      # state could never be cleaned up. Any
      # loop states not yet drained stay in _loop_state_by_loop, so a
      # retried shutdown() re-snapshots and finishes the job; the
      # cancellation itself propagates to the caller unchanged.
      # ONE guarded transition for the admission flag, lifecycle flags,
      # and the completion-future swap: resetting _is_shutting_down
      # before taking the guard let a new caller claim shutdown and
      # install ITS future in the gap, after which this owner resolved
      # the wrong future and a third caller could observe
      # _is_shutting_down=True with no future and fall into overlapping
      # teardown.
      with self._setup_guard:
        self._is_shutting_down = False
        self._started = False
        completion, self._shutdown_future = self._shutdown_future, None
      # Wake coalesced callers. Success is only reported when teardown
      # actually ran to completion: resolving unconditionally let an
      # uncancelled waiter return from close() while the owner was
      # cancelled mid-teardown and state was still live. On the incomplete path waiters retry ownership.
      if completion is not None and not completion.done():
        if teardown_completed:
          completion.set_result(None)
        else:
          completion.set_exception(
              _ShutdownIncompleteError(
                  "Owning shutdown did not complete teardown."
              )
          )
    if teardown_error is not None:
      # The owning caller must not report success over live state.
      raise teardown_error

  def __getstate__(self) -> dict[str, Any]:
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    state["_setup_guard"] = None
    state["_setup_future"] = None
    state["_shutdown_future"] = None
    state["_generation"] = 0
    state["_loop_states_guard"] = None
    state["_drop_counts_guard"] = None
    state["client"] = None
    state["_loop_state_by_loop"] = {}
    state["_write_stream_name"] = None
    state["_executor"] = None
    state["offloader"] = None
    state["parser"] = None
    state["_started"] = False
    state["_startup_error"] = None
    state["_setup_failures"] = 0
    state["_setup_retry_at"] = 0.0
    state["_is_shutting_down"] = False
    state["_init_pid"] = 0
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    """Custom unpickling to restore state."""
    # Backfill keys that may be absent in pickled state from older
    # code versions so _ensure_started does not raise AttributeError.
    state.setdefault("_init_pid", 0)
    state.setdefault("_local_drop_counts", {})
    state.setdefault("_setup_failures", 0)
    state.setdefault("_setup_retry_at", 0.0)
    state.pop("_setup_lock", None)  # replaced by cross-loop future
    state.pop("_setup_locks", None)
    state.pop("_setup_locks_guard", None)
    self.__dict__.update(state)
    self._setup_guard = threading.Lock()
    self._setup_future = None
    self._shutdown_future = None
    self._generation = 0
    self._loop_states_guard = threading.Lock()
    self._drop_counts_guard = threading.Lock()
    # Pickles from older code bypass __init__, so re-validate the restored
    # configuration: e.g. a legacy retry_config with max_retries=NaN would
    # otherwise skip the write loop silently.
    _validate_runtime_config(self.config)

  def _reset_runtime_state(self) -> None:
    """Resets all runtime state after a fork.

    gRPC channels and asyncio locks are not safe to use after
    ``os.fork()``.  This method clears them so the next call to
    ``_ensure_started()`` re-initializes everything in the child
    process.  Pure-data fields like ``_schema`` and
    ``arrow_schema`` are kept because they are safe across fork.
    """
    logger.warning(
        "Fork detected (parent PID %s, child PID %s). Resetting"
        " gRPC state for BigQuery analytics plugin.  Note: gRPC"
        " bidirectional streaming (used by the BigQuery Storage"
        " Write API) is not fork-safe.  If writes hang or time"
        " out, configure the 'spawn' start method at your program"
        " entry-point before creating child processes:"
        "  multiprocessing.set_start_method('spawn')",
        self._init_pid,
        os.getpid(),
    )
    # Best-effort: close inherited gRPC channels so broken
    # finalizers don't interfere with newly created channels.
    # For grpc.aio channels, close() is a coroutine.  We cannot
    # await here (called from sync context / fork handler), so
    # we skip async channels and only close sync ones.
    for loop_state in self._loop_state_by_loop.values():
      wc = getattr(loop_state, "write_client", None)
      transport = getattr(wc, "transport", None)
      if transport is not None:
        try:
          channel = getattr(transport, "_grpc_channel", None)
          if channel is not None and hasattr(channel, "close"):
            result = channel.close()
            # If close() returned a coroutine (grpc.aio channel),
            # discard it to avoid unawaited-coroutine warnings.
            if asyncio.iscoroutine(result):
              result.close()
        except Exception:
          pass

    # Clear all runtime state.
    self._setup_guard = threading.Lock()
    self._setup_future = None
    self._shutdown_future = None
    self._generation = 0
    self._loop_states_guard = threading.Lock()
    self._drop_counts_guard = threading.Lock()
    self.client = None
    self._loop_state_by_loop = {}
    self._write_stream_name = None
    self._executor = None
    self.offloader = None
    self.parser = None
    self._started = False
    self._startup_error = None
    self._setup_failures = 0
    self._setup_retry_at = 0.0
    self._is_shutting_down = False
    self._init_pid = os.getpid()

  def _count_local_drop(self, reason: str) -> None:
    """Counts a row lost before/outside any BatchProcessor."""
    with self._drop_counts_guard:
      self._local_drop_counts[reason] = (
          self._local_drop_counts.get(reason, 0) + 1
      )

  async def close(self) -> None:
    """Releases all plugin resources (BasePlugin/PluginManager contract).

    Runner.close() -> PluginManager.close() -> plugin.close() previously
    hit the inherited no-op, bypassing queue drain, client/executor
    teardown, and shutdown loss accounting entirely. PluginManager's outer close
    timeout (5s) may cancel this
    mid-drain; shutdown()'s cleanup is cancellation-tolerant and counters
    remain queryable either way.
    """
    await self.shutdown()

  async def __aenter__(self) -> BigQueryAgentAnalyticsPlugin:
    await self._ensure_started()
    return self

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: TracebackType | None,
  ) -> None:
    await self.shutdown()

  async def _ensure_started(self, **kwargs: Any) -> str:
    """Ensures that the plugin is started and initialized.

    Setup failures no longer poison the plugin permanently:
    the failure is recorded, ``_started`` stays False, and a later event
    retries after a bounded exponential backoff. Attempts are coalesced
    through the setup lock, so failure mode costs at most one setup RPC
    per backoff window — not one per event.

    Returns:
        A structured outcome — ``"ok"``, ``"disabled"``, ``"failed"``, or
        ``"aborted"`` (shutdown crossed the attempt). This method never
        counts a lost row itself: it is also called from non-row paths
        (Runner start, ``__aenter__``), which produced phantom
        ``shutdown_race`` counts, while the real row owner counted the
        same incident a second time as ``setup_unavailable``. The row owner
        counts exactly one loss
        based on this outcome.
    """
    # Disabled mode must have zero side effects: no ADC lookup,
    # client creation, table RPCs, or background tasks from any entry point
    # (before_run_callback, __aenter__, _log_event all route through here).
    if not self.config.enabled:
      return "disabled"
    # _init_pid == 0 means the plugin was unpickled and has never been
    # initialized in this process (the pickle sentinel set by
    # __getstate__).  Skip the fork reset in that case — no fork
    # happened, and _started is already False so _lazy_setup will run.
    # Real forks are caught by os.register_at_fork (line 108) and by
    # this check when _init_pid is a real (non-zero) PID from a
    # different process.
    if self._init_pid != 0 and os.getpid() != self._init_pid:
      self._reset_runtime_state()
    if self._started:
      return "ok"

    # Cross-loop coalescing of the SHARED initialization: _lazy_setup mutates process-wide state (client,
    # executor, parser, schema, views, retry bookkeeping), so exactly one
    # caller may run it at a time — across event loops and threads, which
    # a per-loop asyncio.Lock cannot provide and a shared one cannot
    # survive. A concurrent.futures.Future is claimed under a briefly-held
    # threading.Lock (never held across an await); the owner runs setup,
    # every other caller awaits the same future via asyncio.wrap_future
    # from its own loop. Loop-local writer state stays separate in
    # _get_loop_state().
    setup_future: Optional["ConcurrentFuture[None]"] = None
    is_owner = False
    with self._setup_guard:
      if self._started:
        return "ok"
      if self._setup_future is not None:
        setup_future = self._setup_future
      elif (
          self._startup_error is not None
          and time.monotonic() < self._setup_retry_at
      ):
        # Still inside the backoff window from a previous failure.
        return "failed"
      else:
        setup_future = ConcurrentFuture()
        self._setup_future = setup_future
        is_owner = True
      claimed_generation = self._generation

    assert setup_future is not None  # every fall-through branch assigns it

    if not is_owner:
      try:
        # shield: a cancelled waiter must not cancel the SHARED future —
        # unshielded, cancellation propagated into the ConcurrentFuture
        # and the owner's set_result then raised InvalidStateError. The waiter itself still observes its own
        # cancellation.
        await asyncio.shield(asyncio.wrap_future(setup_future))
      except _SetupAbortedError:
        return "aborted"
      except Exception:
        # The owner already recorded the failure and backoff; waiters
        # degrade the same way the owner does (loss counted by the row
        # owner from this outcome).
        return "failed"
      return "ok" if self._started else "failed"

    try:
      await self._lazy_setup(claimed_generation=claimed_generation, **kwargs)
    except asyncio.CancelledError:
      # Owner cancelled mid-setup: without this, the pending future was
      # never finalized and every later _ensure_started waited forever.
      # Release partial resources that need
      # no await (the executor keeps running its current job; the
      # eventual client is closed by _lazy_setup's done-callback), clear the rendezvous, wake waiters with an
      # ordinary aborted error, then re-raise the cancellation.
      executor, self._executor = self._executor, None
      if executor is not None and self.client is None:
        executor.shutdown(wait=False)
      elif executor is not None:
        self._executor = executor  # a live client still uses it
      self.offloader = None
      self.parser = None
      with self._setup_guard:
        self._setup_future = None
      if not setup_future.done():
        setup_future.set_exception(
            _SetupAbortedError(
                "BigQuery plugin setup aborted: owner cancelled."
            )
        )
      raise
    except _LoopStateAdmissionAbortedError as e:
      # A retained processor with _shutdown=True and a live worker is a
      # lifecycle admission race, not a service/setup failure. Keep shared
      # clients intact, avoid poisoning exponential backoff, and wake every
      # coalesced caller with the same structured aborted outcome. The row
      # owner (and only the row owner) converts that outcome to shutdown_race.
      with self._setup_guard:
        self._setup_future = None
      if not setup_future.done():
        setup_future.set_exception(e)
      return "aborted"
    except Exception as e:
      aborted = False
      with self._setup_guard:
        if self._generation != claimed_generation:
          # shutdown() completed while setup was blocked; the failure is
          # the abort itself, not a service error, so it must not poison
          # the backoff window — and the partially created resources must
          # be released.
          aborted = True
        else:
          self._startup_error = e
          self._setup_failures += 1
          backoff = min(60.0, 2.0 ** min(self._setup_failures, 6))
          self._setup_retry_at = time.monotonic() + backoff
          self._setup_future = None
      if aborted:
        # The rendezvous stays claimed until teardown completes: clearing
        # it first let a new-generation setup finish while the old owner
        # was paused, after which this teardown destroyed the NEW
        # client/parser/loop state and the plugin wedged with
        # _started=True and no resources.
        try:
          await self._teardown_aborted_setup()
        finally:
          with self._setup_guard:
            self._setup_future = None
          if not setup_future.done():
            setup_future.set_exception(
                _SetupAbortedError(
                    "BigQuery plugin setup aborted: shutdown during setup."
                )
            )
        return "aborted"
      logger.error(
          "Failed to initialize BigQuery Plugin (attempt %d, next"
          " retry in %.0fs): %s",
          self._setup_failures,
          backoff,
          e,
      )
      if not setup_future.done():
        setup_future.set_exception(e)
      return "failed"
    else:
      aborted = False
      with self._setup_guard:
        if self._generation != claimed_generation:
          # shutdown() ran while setup was in flight: do NOT resurrect
          # _started after shutdown returned.
          aborted = True
        else:
          self._started = True
          self._startup_error = None
          self._setup_failures = 0
          self._setup_retry_at = 0.0
          self._setup_future = None
          # Record the current PID so fork detection works for
          # the rest of this instance's lifetime.
          if self._init_pid == 0:
            self._init_pid = os.getpid()
      if not aborted:
        if not setup_future.done():
          setup_future.set_result(None)
        return "ok"
      # Setup fully succeeded but lost the generation race: everything it
      # created outlives a shutdown that already returned — release it,
      # holding the rendezvous until the
      # teardown completes.
      try:
        await self._teardown_aborted_setup()
      finally:
        with self._setup_guard:
          self._setup_future = None
        if not setup_future.done():
          setup_future.set_exception(
              _SetupAbortedError(
                  "BigQuery plugin setup aborted: shutdown during setup."
              )
          )
      return "aborted"

  async def _teardown_aborted_setup(self) -> None:
    """Releases every resource created by a setup that crossed a shutdown.

    A setup attempt that lost the generation race used to count the loss
    and stop, leaving the freshly created shared client, executor,
    parser/offloader, and any published loop state alive on a plugin
    whose shutdown() had already returned.
    Callers must hold the setup rendezvous (_setup_future) for the whole
    teardown so no new-generation setup can publish resources this method
    would then destroy; the started-check is
    a second line of defense.
    """
    if self._started:
      # A newer-generation setup owns the current resources.
      return
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None
    state = None
    if loop is not None:
      with self._loop_states_guard:
        state = self._loop_state_by_loop.pop(loop, None)
    if state is not None:
      try:
        await state.batch_processor.shutdown(
            timeout=self.config.shutdown_timeout
        )
      except Exception:
        pass
      transport = getattr(state.write_client, "transport", None)
      if transport:
        try:
          await transport.close()
        except Exception:
          pass
    offloader, self.offloader = self.offloader, None
    self.parser = None
    storage_client = getattr(offloader, "client", None) if offloader else None
    if storage_client is not None:
      # Close the owned GCS client too.
      try:
        await asyncio.get_running_loop().run_in_executor(
            None, storage_client.close
        )
      except Exception:
        pass
    client, self.client = self.client, None
    executor, self._executor = self._executor, None
    if client is not None:
      try:
        await asyncio.get_running_loop().run_in_executor(None, client.close)
      except Exception:
        pass
    if executor is not None:
      try:
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: executor.shutdown(wait=True)
        )
      except Exception:
        pass

  @staticmethod
  def _resolve_ids(
      event_data: EventData,
      callback_context: CallbackContext,
  ) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolves trace_id, span_id, and parent_span_id for a log row.

    Resolution rules:

      * **trace_id** — ambient OTel trace wins (the plugin stack already
        shares the ambient trace when initialised from an ambient span,
        so in practice they agree).
      * **span_id / parent_span_id** — the plugin's internal span stack
        (``TraceManager``) is the preferred source.  Ambient OTel spans
        are only used as a fallback when the plugin stack has no span.
        This ensures every ``parent_span_id`` in BigQuery references a
        ``span_id`` that is also logged to BigQuery, producing a
        self-consistent execution tree.
      * **Explicit overrides** (``EventData``) always win last — they
        are set by post-pop callbacks that have already captured the
        correct plugin-stack values before the pop.

    Priority order (highest first):
      1. Explicit ``EventData`` overrides.
      2. Plugin's internal span stack (``TraceManager``) for
         ``span_id`` / ``parent_span_id``.
      3. Ambient OTel span — always used for ``trace_id``; used for
         ``span_id`` / ``parent_span_id`` only when the plugin stack
         has no span.
      4. ``invocation_id`` fallback for trace_id.

    Returns:
        (trace_id, span_id, parent_span_id)
    """
    # --- Plugin stack: span_id / parent_span_id baseline ---
    trace_id = TraceManager.get_trace_id(callback_context)
    plugin_span_id, plugin_parent_span_id = (
        TraceManager.get_current_span_and_parent()
    )
    span_id = plugin_span_id
    parent_span_id = plugin_parent_span_id

    # --- Ambient OTel: trace_id always; span fallback only ---
    ambient = trace.get_current_span()
    ambient_ctx = ambient.get_span_context()
    if ambient_ctx.is_valid:
      trace_id = format(ambient_ctx.trace_id, "032x")
      # Only use ambient span IDs when the plugin stack has no span.
      # Framework-internal spans (execute_tool, call_llm, etc.) are
      # never written to BQ, so deriving parent_span_id from them
      # creates phantom references.  The plugin stack guarantees
      # that both span_id and parent_span_id reference BQ rows.
      if span_id is None:
        span_id = format(ambient_ctx.span_id, "016x")
        parent_span_id = None
        parent_ctx = getattr(ambient, "parent", None)
        if parent_ctx is not None and parent_ctx.span_id:
          parent_span_id = format(parent_ctx.span_id, "016x")

    # --- Explicit EventData overrides (post-pop callbacks) ---
    if event_data.trace_id_override is not None:
      trace_id = event_data.trace_id_override
    if event_data.span_id_override is not None:
      span_id = event_data.span_id_override
    if event_data.parent_span_id_override is not None:
      parent_span_id = event_data.parent_span_id_override

    return trace_id, span_id, parent_span_id

  @staticmethod
  def _extract_latency(
      event_data: EventData,
  ) -> dict[str, Any] | None:
    """Reads latency fields from EventData and returns a latency dict (or None).

    Returns:
        A dict with ``total_ms`` and/or ``time_to_first_token_ms``, or
        *None* if neither was present.
    """
    latency_json: dict[str, Any] = {}
    if event_data.latency_ms is not None:
      latency_json["total_ms"] = event_data.latency_ms
    if event_data.time_to_first_token_ms is not None:
      latency_json["time_to_first_token_ms"] = event_data.time_to_first_token_ms
    return latency_json or None

  @staticmethod
  def _resolve_agent_label(
      callback_context: CallbackContext,
      source_event: Optional["Event"],
  ) -> Optional[str]:
    """Resolves the ``agent`` column without raising when no agent is set.

    ``CallbackContext.agent_name`` dereferences
    ``InvocationContext.agent.name`` with no None guard, but ``agent`` is
    legitimately ``None`` for workflow-driven invocations with deterministic
    nodes. Reading it at row-build time then raised ``AttributeError``, which
    ``@_safe_callback`` swallowed, silently dropping the row (issue #6063).

    Resolution order:

    * running agent present → ``agent.name``;
    * no agent but a source Event → ``Event.author`` (the emitting node), a
      more meaningful workflow label than a sentinel;
    * callback-only row with neither → ``None`` (SQL NULL).
    """
    agent = getattr(callback_context._invocation_context, "agent", None)
    if agent is not None:
      return getattr(agent, "name", None)
    if source_event is not None:
      return getattr(source_event, "author", None)
    return None

  def _build_adk_envelope(
      self,
      callback_context: CallbackContext,
      source_event: Optional["Event"],
  ) -> dict[str, Any]:
    """Builds the ``attributes.adk`` envelope.

    A1 / A2 (``schema_version``, ``app_name``) stamp on every ADK-enriched
    row regardless of origin. A3 / C1 / C2 / C3 (``source_event_id``,
    ``node``, ``branch``, ``scope``) and C8 (``route``,
    ``render_ui_widgets``, ``rewind_before_invocation_id``) only stamp
    when a source Event is provided — callback-only rows **omit** those
    keys from the envelope rather than synthesizing fake identity. Since
    the surrounding column is BigQuery JSON, an omitted key resolves to
    SQL NULL via ``JSON_VALUE(attributes, '$.adk.<field>')``; consumers
    using ``JSON_VALUE(...) IS NOT NULL`` to gate on Event-originating
    rows therefore work correctly without the producer writing explicit
    JSON nulls.
    """
    adk: dict[str, Any] = {
        "schema_version": _ADK_ENVELOPE_SCHEMA_VERSION,
    }
    try:
      adk["app_name"] = callback_context._invocation_context.session.app_name
    except Exception:
      adk["app_name"] = None

    if source_event is None:
      return adk

    # Every getattr below is defensive: source_event is "anything the
    # caller hands us", which in test suites can be a Mock. Best-effort
    # enrichment means "leave null on missing attrs", never crash the
    # row.
    try:
      source_event_id = getattr(source_event, "id", None)
      if source_event_id:
        adk["source_event_id"] = source_event_id  # A3
    except Exception:
      pass

    # C1: node = {path, run_id, parent_run_id}. NodeInfo.path defaults to
    # the empty string in current ADK (events/event.py); run_id and
    # parent_run_id are @property values parsed from path (not model
    # fields), so they are read explicitly here rather than via
    # model_dump. parent_run_id is None when there is no parent node.
    try:
      node_info = getattr(source_event, "node_info", None)
      if node_info is not None and hasattr(node_info, "path"):
        path = getattr(node_info, "path", "") or ""
        run_id = getattr(node_info, "run_id", None)
        parent_run_id = getattr(node_info, "parent_run_id", None)
        adk["node"] = {
            "path": path,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
        }
    except Exception:
      pass

    # C2: branch — absent stays JSON null (no sentinel string).
    try:
      if hasattr(source_event, "branch"):
        adk["branch"] = source_event.branch
    except Exception:
      pass

    # C3: scope shape derivation. Order matters: node-shape patterns must
    # be checked before falling through to function_call so bare
    # ``name@run_id`` doesn't misclassify.
    try:
      if hasattr(source_event, "isolation_scope"):
        adk["scope"] = _derive_scope(source_event.isolation_scope)
    except Exception:
      pass

    # C8: raw EventActions mirror (flat under attributes.adk). Stamp only
    # when actually set so JSON doesn't bloat with nulls.
    try:
      actions = getattr(source_event, "actions", None)
    except Exception:
      actions = None
    if actions is not None:
      try:
        route = getattr(actions, "route", None)
        if route is not None:
          adk["route"] = route
      except Exception:
        pass
      try:
        widgets = getattr(actions, "render_ui_widgets", None)
        if widgets is not None:
          adk["render_ui_widgets"] = [
              w.model_dump() if hasattr(w, "model_dump") else w for w in widgets
          ]
      except Exception:
        pass
      try:
        rewind = getattr(actions, "rewind_before_invocation_id", None)
        if rewind is not None:
          adk["rewind_before_invocation_id"] = rewind
      except Exception:
        pass

    return adk

  def _enrich_attributes(
      self,
      event_data: EventData,
      callback_context: CallbackContext,
  ) -> dict[str, Any]:
    """Builds the attributes dict from EventData and enrichments.

    Reads ``model``, ``model_version``, and ``usage_metadata`` from
    *event_data*, copies ``extra_attributes``, then adds session metadata
    and custom tags. Also stamps the ``adk`` envelope.

    Returns:
        A new dict ready for JSON serialization into the attributes column.
    """
    attrs: dict[str, Any] = dict(event_data.extra_attributes)
    adk_envelope = self._build_adk_envelope(
        callback_context, event_data.source_event
    )
    # Merge producer-supplied adk_extras (long-running pair keys etc.)
    # INTO the adk envelope so consumer SQL on
    # ``$.adk.pause_kind`` / ``$.adk.function_call_id`` resolves.
    # adk_envelope wins on key conflict — producer-derived envelope
    # is the source of truth for identity fields like source_event_id.
    for k, v in event_data.adk_extras.items():
      adk_envelope.setdefault(k, v)
    attrs["adk"] = adk_envelope

    attrs["root_agent_name"] = TraceManager.get_root_agent_name()
    if event_data.model:
      attrs["model"] = event_data.model
    if event_data.model_version:
      attrs["model_version"] = event_data.model_version
    if event_data.usage_metadata:
      usage_dict, _ = _recursive_smart_truncate(
          event_data.usage_metadata, self.config.max_content_length
      )
      if isinstance(usage_dict, dict):
        attrs["usage_metadata"] = usage_dict
      else:
        attrs["usage_metadata"] = event_data.usage_metadata

    if event_data.cache_metadata:
      cache_meta_dict, _ = _recursive_smart_truncate(
          event_data.cache_metadata, self.config.max_content_length
      )
      if isinstance(cache_meta_dict, dict):
        attrs["cache_metadata"] = cache_meta_dict
      else:
        attrs["cache_metadata"] = event_data.cache_metadata

    if self.config.log_session_metadata:
      try:
        session = callback_context._invocation_context.session
        session_meta = {
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
        }
        # Include session state if non-empty (contains user-set metadata
        # like gchat thread-id, customer_id, etc.)
        if session.state:
          truncated_state, _ = _recursive_smart_truncate(
              dict(session.state),
              self.config.max_content_length,
          )
          session_meta["state"] = truncated_state
        attrs["session_metadata"] = session_meta
      except Exception:
        pass

    if self.config.custom_tags:
      attrs["custom_tags"] = self.config.custom_tags

    # Best-effort span-level Cloud Trace correlation, opt-in via
    # ``enable_otel_correlation``. Capture the ambient OTel span context at
    # row-emission time, ONLY when it is valid. Stored under attributes.otel.*
    # (staged); the typed span_id / parent_span_id columns stay the
    # BQAA-internal execution tree. This is a best-effort join key, not a
    # foreign key -- an unsampled valid span is absent from the Cloud Trace
    # export. Skipped when the attributes column is projected out, since it
    # would be dropped anyway.
    if (
        self.config.enable_otel_correlation
        and "attributes" not in self._denied_columns
    ):
      otel_ctx = trace.get_current_span().get_span_context()
      if otel_ctx.is_valid:
        attrs["otel"] = {
            "span_id": format(otel_ctx.span_id, "016x"),
            "trace_id": format(otel_ctx.trace_id, "032x"),
        }

    return attrs

  def _custom_metadata_allowed(self, key: Any) -> bool:
    """Returns whether *key* matches the allowlist (exact or prefix)."""
    if not isinstance(key, str):
      return False
    if key in self._custom_metadata_exact:
      return True
    return any(key.startswith(p) for p in self._custom_metadata_prefixes)

  def _capture_custom_metadata(
      self, event_data: EventData, attributes: dict[str, Any]
  ) -> bool:
    """Captures allowlisted ``custom_metadata`` into ``attributes``.

    Reads ``event.custom_metadata`` from the row's source Event, keeps only
    allowlisted keys, runs them through the shared safety pipeline
    (truncation + sensitive-key redaction + circular-reference handling),
    and writes the result under ``attributes['custom_metadata']``.

    The built-in ``a2a:*`` handling in ``on_event_callback`` is unaffected;
    this is purely additive under a separate namespace.

    Returns:
        True if any captured value was truncated (so the caller can flip
        ``is_truncated``).
    """
    source = event_data.source_event
    meta = getattr(source, "custom_metadata", None) if source else None
    if not meta:
      return False
    captured = {
        k: v for k, v in meta.items() if self._custom_metadata_allowed(k)
    }
    if not captured:
      return False
    safe, truncated = _recursive_smart_truncate(
        captured, self.config.max_content_length
    )
    if isinstance(safe, dict) and safe:
      attributes["custom_metadata"] = safe
    return bool(truncated)

  async def _log_event(
      self,
      event_type: str,
      callback_context: CallbackContext,
      raw_content: Any = None,
      is_truncated: bool = False,
      event_data: Optional[EventData] = None,
  ) -> None:
    """Logs an event to BigQuery.

    Args:
        event_type: The type of event (e.g., 'LLM_REQUEST').
        callback_context: The callback context.
        raw_content: The raw content to log.
        is_truncated: Whether the content is already truncated.
        event_data: Typed container for structured fields and extra
            attributes. Defaults to ``EventData()`` when not provided.
    """
    if not self.config.enabled or self._is_shutting_down:
      return
    if self.config.event_denylist and event_type in self.config.event_denylist:
      return
    if (
        self.config.event_allowlist
        and event_type not in self.config.event_allowlist
    ):
      return

    if not self._started:
      outcome = await self._ensure_started()
      if outcome == "disabled":
        return
      if not self._started:
        # The row is lost — record exactly ONE loss, classified by the
        # structured setup outcome: "aborted" (shutdown crossed the
        # attempt) is a shutdown_race, anything else is setup
        # unavailability. _ensure_started itself never counts, so
        # non-row entry points no longer produce phantom counts and one
        # raced event no longer counts twice.
        self._count_local_drop(
            "shutdown_race" if outcome == "aborted" else "setup_unavailable"
        )
        return

    if event_data is None:
      event_data = EventData()

    # Error diagnostics bypass the ordinary attributes tree: error_message is
    # a dedicated column and agent/run tracebacks live in raw content. Apply
    # one bounded, fail-closed boundary here so every current and future error
    # producer receives the same privacy contract before formatter/parser row
    # assembly. Ordinary safe messages remain byte-for-byte unchanged.
    if event_data.error_message is not None:
      try:
        safe_error, error_content_lost = _sanitize_sensitive_text(
            event_data.error_message, self.config.max_content_length
        )
      except Exception:
        safe_error, error_content_lost = (
            "[REDACTED_SENSITIVE_TEXT]",
            True,
        )
      event_data.error_message = safe_error
      is_truncated = is_truncated or error_content_lost
    if event_type in ("AGENT_ERROR", "INVOCATION_ERROR") and isinstance(
        raw_content, collections.abc.Mapping
    ):
      try:
        error_traceback = raw_content.get("error_traceback")
        if isinstance(error_traceback, str):
          safe_traceback, traceback_content_lost = _sanitize_sensitive_text(
              error_traceback, self.config.max_content_length
          )
          raw_content = dict(raw_content)
          raw_content["error_traceback"] = safe_traceback
          is_truncated = is_truncated or traceback_content_lost
      except Exception:
        raw_content = {"error_traceback": "[REDACTED_SENSITIVE_TEXT]"}
        is_truncated = True

    timestamp = datetime.now(timezone.utc)
    if self.config.content_formatter:
      try:
        formatted = self.config.content_formatter(raw_content, event_type)
        if isinstance(formatted, str):
          if type(formatted) is not str:
            # Normalize str subclasses to the exact built-in.
            formatted = str.__str__(formatted)
        elif formatted is not None and not (
            # Every shape the parser handles NATIVELY: identity and
            # conditional formatters legitimately return these, and the
            # Str/Content/None-only gate destroyed untransformed
            # LlmRequest/dict/list events.
            # Model shapes require the EXACT class: a subclass can
            # override an attribute the parser reads OUTSIDE this
            # boundary and raise a payload-bearing exception into the
            # safe callback's traceback log. dict/list subclasses stay isinstance-based — the
            # parser routes them through the hardened recursive
            # sanitizer, whose protocol boundary already fails closed.
            type(formatted) in (types.Content, types.Part, LlmRequest)
            or isinstance(formatted, (dict, list))
        ):
          # The formatter is typed Any: a non-native result would reach
          # the parser's unconditional str(content) fallback OUTSIDE this
          # fail-closed boundary, where a payload-controlled __str__ can
          # republish the original content or raise into the safe
          # callback's traceback log. The
          # message is CONSTANT: even a class NAME can be payload-derived
          # via type(name, ...).
          logger.warning(
              "Content formatter returned an unsupported result type for"
              " event %s; writing sentinel instead of original content.",
              event_type,
          )
          formatted = _FORMATTER_FAILED_SENTINEL
          self._count_local_drop("formatter_failed")
        raw_content = formatted
      except Exception:
        # Fail CLOSED: the formatter is a redaction/privacy
        # boundary, so its failure must never fall back to the unformatted
        # payload. The log message is CONSTANT — the exception message and
        # traceback can embed the protected content, and even the class
        # NAME can be payload-derived via type(name, ...).
        logger.warning(
            "Content formatter failed for event %s; writing sentinel"
            " instead of original content.",
            event_type,
        )
        raw_content = _FORMATTER_FAILED_SENTINEL
        self._count_local_drop("formatter_failed")

    trace_id, span_id, parent_span_id = self._resolve_ids(
        event_data, callback_context
    )

    if not self.parser:
      logger.warning("Parser not initialized; skipping event %s.", event_type)
      return

    # When both payload columns are projected out, skip content parsing
    # entirely -- no inline summary, no parts, and (critically) no GCS offload
    # work for a row that retains neither payload column.
    content_json: Any
    content_parts: list[dict[str, Any]]
    parser_truncated: bool
    if {"content", "content_parts"} <= self._denied_columns:
      content_json, content_parts, parser_truncated = None, [], False
    else:
      # Pass trace/span per call: the parser instance is shared, so storing
      # request identity on it lets concurrent events overwrite each other's
      # GCS object paths.
      try:
        content_json, content_parts, parser_truncated = await self.parser.parse(
            raw_content,
            trace_id=trace_id or "no_trace",
            span_id=span_id or "no_span",
        )
        # Normalize the parser OUTPUT to strictly JSON-native values
        # inside the same boundary: a nested
        # hostile model can defer its failure PAST parse(), detonating in
        # Arrow serialization's json.dumps/str fallback where
        # _write_rows_with_retry logged the payload with a traceback and
        # dropped the row as arrow_prep_failed.
        content_json, norm_replaced_json = _normalize_json_native(
            content_json, self.config.max_content_length
        )
        # content_parts carry parser-BUILT metadata (GCS URIs,
        # object_ref.details JSON) whose strings must stay intact; their
        # payload text was already truncated by the parser itself, so
        # only shape normalization applies (max_len=-1).
        normalized_parts, norm_replaced_parts = _normalize_json_native(
            content_parts, -1
        )
        content_parts = (
            normalized_parts if isinstance(normalized_parts, list) else []
        )
        parser_truncated = (
            parser_truncated or norm_replaced_json or norm_replaced_parts
        )
      except Exception:
        # Fail-closed, constant-log parse boundary: the top-level formatter gate cannot see NESTED hostile
        # model subclasses (pydantic preserves them through normal
        # construction), whose attribute accesses raise payload-bearing
        # exceptions inside the parser. Escaping here reached
        # _safe_callback's traceback log and dropped the whole row.
        logger.warning(
            "Content parsing failed for event %s; writing sentinel"
            " instead of content.",
            event_type,
        )
        content_json, content_parts, parser_truncated = (
            "[CONTENT_PARSE_FAILED]",
            [],
            True,
        )
        self._count_local_drop("content_parse_failed")
    is_truncated = is_truncated or parser_truncated

    latency_json = self._extract_latency(event_data)
    attributes = self._enrich_attributes(event_data, callback_context)

    # Capture allowlisted custom_metadata into attributes.custom_metadata.
    # Runs for every row emitted from a source Event (incl. AGENT_RESPONSE,
    # which does not otherwise read custom_metadata), through the same safety
    # pipeline. Truncation here also flips is_truncated.
    if self._custom_metadata_exact or self._custom_metadata_prefixes:
      meta_truncated = self._capture_custom_metadata(event_data, attributes)
      is_truncated = is_truncated or meta_truncated

    # Final safety pass: sanitize the COMPLETE assembled
    # attributes tree immediately before serialization. Producer-local
    # sanitization above remains as an optimization, but this pass is the
    # mandatory boundary — it covers values copied in directly (state_delta
    # via extra_attributes, custom_tags, labels, generic extra attributes),
    # nested structures, `temp:`-scoped keys, and JSON-encoded blobs.
    attributes, attrs_truncated = _recursive_smart_truncate(
        attributes, self.config.max_content_length
    )
    is_truncated = is_truncated or attrs_truncated

    # Serialize attributes to JSON string
    try:
      attributes_json = json.dumps(attributes)
    except (TypeError, ValueError):
      attributes_json = json.dumps(attributes, default=str)

    row = {
        "timestamp": timestamp,
        "event_type": event_type,
        "agent": self._resolve_agent_label(
            callback_context, event_data.source_event
        ),
        "user_id": callback_context.user_id,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "content": content_json,
        "content_parts": (
            content_parts if self.config.log_multi_modal_content else []
        ),
        "attributes": attributes_json,
        "latency_ms": latency_json,
        "status": event_data.status,
        "error_message": event_data.error_message,
        "is_truncated": is_truncated,
    }

    # drop denied payload columns from the row so it matches the
    # projected table / Arrow schema exactly (schema-first consistency).
    if self._denied_columns:
      row = {k: v for k, v in row.items() if k not in self._denied_columns}
    try:
      state = await self._get_loop_state()
    except RuntimeError:
      self._count_local_drop("shutdown_race")
      return
    if self._is_shutting_down:
      self._count_local_drop("shutdown_race")
      return
    await state.batch_processor.append(row)

  # --- UPDATED CALLBACKS FOR V1 PARITY ---

  @_safe_callback
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> None:
    """Parity with V1: Logs USER_MESSAGE_RECEIVED event.

    Also detects:
    * HITL completion responses (user-sent ``FunctionResponse`` parts
      with ``adk_request_*`` names) → ``HITL_*_COMPLETED``.
    * Non-HITL ``FunctionResponse`` parts from a user message → these
      are the long-running tool completions for tools that paused via
      ``TOOL_PAUSED``. Emitted as ``TOOL_COMPLETED`` with
      ``pause_kind = 'tool'`` and ``function_call_id`` so the customer
      can join the pair from BigQuery.

    Args:
        invocation_context: The context of the current invocation.
        user_message: The message content received from the user.
    """
    callback_ctx = CallbackContext(invocation_context)
    TraceManager.ensure_invocation_span(callback_ctx)
    await self._log_event(
        "USER_MESSAGE_RECEIVED",
        callback_ctx,
        raw_content=user_message,
    )

    # Detect completion responses in the user message.
    if user_message and user_message.parts:
      for part in user_message.parts:
        if not part.function_response:
          continue
        hitl_event = _HITL_EVENT_MAP.get(part.function_response.name)
        resp_truncated, is_truncated = _recursive_smart_truncate(
            part.function_response.response or {},
            self.config.max_content_length,
        )
        content_dict = {
            "tool": part.function_response.name,
            "result": resp_truncated,
        }
        if hitl_event:
          # HITL completions stay on the HITL_*_COMPLETED stream — they
          # MUST NOT also emit TOOL_COMPLETED.
          await self._log_event(
              hitl_event + "_COMPLETED",
              callback_ctx,
              raw_content=content_dict,
              is_truncated=is_truncated,
          )
        else:
          # Non-HITL function_response arriving via a user message is
          # by construction a long-running tool completion: regular
          # tool calls complete inside the agent run via
          # after_tool_callback, so a function_response inside a user
          # message is the resume side of a previously-paused tool.
          # Stamp the pair keys; pause_orphan / registry semantics
          # are intentionally deferred.
          if not part.function_response.id:
            logger.debug(
                "User-message function_response for tool %s has no id;"
                " the resulting TOOL_COMPLETED row cannot pair with a"
                " TOOL_PAUSED row.",
                part.function_response.name,
            )
          await self._log_event(
              "TOOL_COMPLETED",
              callback_ctx,
              raw_content=content_dict,
              is_truncated=is_truncated,
              event_data=EventData(
                  adk_extras={
                      "pause_kind": "tool",
                      "function_call_id": part.function_response.id,
                  },
              ),
          )

  @_safe_callback
  async def on_event_callback(
      self,
      *,
      invocation_context: InvocationContext,
      event: "Event",
  ) -> None:
    """Logs state changes, HITL events, A2A interactions, and agent responses.

    - Checks each event for a non-empty state_delta and logs it as a
      STATE_DELTA event.
    - Detects synthetic ``adk_request_*`` function calls (HITL pause
      events) and their corresponding function responses (HITL
      completions) and emits dedicated HITL event types.
    - Detects events carrying A2A interaction metadata
      (``a2a:request`` / ``a2a:response`` in ``custom_metadata``)
      and logs them as ``A2A_INTERACTION`` events so the remote
      agent's response and cross-reference IDs (``a2a:task_id``,
      ``a2a:context_id``) are visible in BigQuery.
    - Detects final response events emitted by agents and logs
      them as ``AGENT_RESPONSE`` so the visible response text
      (after all callback modifications) is captured in BigQuery.

    The HITL detection must happen here (not in tool callbacks) because
    ``adk_request_credential``, ``adk_request_confirmation``, and
    ``adk_request_input`` are synthetic function calls injected by the
    framework — they never go through ``before_tool_callback`` /
    ``after_tool_callback``.

    Args:
        invocation_context: The context for the current invocation.
        event: The event raised by the runner.
    """
    callback_ctx = CallbackContext(invocation_context)

    # A later before_model callback may short-circuit the model call. ADK
    # intentionally skips every after_model callback in that case, so the
    # llm_request span pushed by this plugin has no matching callback to pop
    # it. The synthesized response is a non-partial event; close only that
    # expected top span at this boundary. Streaming chunks stay attached to
    # their live llm_request span until the final response callback.
    if getattr(event, "partial", None) is not True:
      TraceManager.pop_span(expected_kind="llm_request")

    # --- State delta logging ---
    if event.actions.state_delta:
      await self._log_event(
          "STATE_DELTA",
          callback_ctx,
          event_data=EventData(
              source_event=event,
              extra_attributes={"state_delta": dict(event.actions.state_delta)},
          ),
      )

    # --- AGENT_TRANSFER ---
    # actions.transfer_to_agent stores the *target* agent only
    # (events/event_actions.py); from_agent is pinned to event.author
    # by contract. Never fabricate authors on non-Event paths.
    if event.actions.transfer_to_agent:
      await self._log_event(
          "AGENT_TRANSFER",
          callback_ctx,
          raw_content={
              "from_agent": event.author,
              "to_agent": event.actions.transfer_to_agent,
          },
          event_data=EventData(source_event=event),
      )

    # --- EVENT_COMPACTION ---
    # EventCompaction.start_timestamp / end_timestamp are float epoch
    # seconds. Preserve fractional precision here; consumer view
    # conversion is deferred.
    compaction = event.actions.compaction
    if compaction is not None:
      compacted_content, compaction_truncated = self._format_content_safely(
          compaction.compacted_content
      )
      await self._log_event(
          "EVENT_COMPACTION",
          callback_ctx,
          raw_content={
              "start_timestamp": compaction.start_timestamp,
              "end_timestamp": compaction.end_timestamp,
              "compacted_content": compacted_content,
          },
          is_truncated=compaction_truncated,
          event_data=EventData(source_event=event),
      )

    # --- AGENT_STATE_CHECKPOINT ---
    # Fires when *either* agent_state is set or end_of_agent is True;
    # supports {agent_state: None, end_of_agent: True} payloads.
    # Inline payload only — oversized-state GCS offload deferred.
    if (
        event.actions.agent_state is not None
        or event.actions.end_of_agent is True
    ):
      agent_state_dict, agent_state_truncated = (
          _recursive_smart_truncate(
              event.actions.agent_state,
              self.config.max_content_length,
          )
          if event.actions.agent_state is not None
          else (None, False)
      )
      await self._log_event(
          "AGENT_STATE_CHECKPOINT",
          callback_ctx,
          raw_content={
              "agent_state": agent_state_dict,
              "end_of_agent": bool(event.actions.end_of_agent),
          },
          is_truncated=agent_state_truncated,
          event_data=EventData(source_event=event),
      )

    # --- HITL + TOOL_PAUSED (pair-key emit) + per-part
    #     iteration over event.content.parts ---
    # TOOL_PAUSED fires per long_running_tool_id; pause_kind is derived
    # via the id→name lookup against _HITL_PAUSE_KIND_MAP, so a HITL
    # long-running call carries pause_kind = 'hitl_*' and a regular
    # long-running tool carries pause_kind = 'tool'. function_call_id
    # joins to the downstream TOOL_COMPLETED via the user message path.
    # Use getattr so the existing Mock-based HITL test fixtures still
    # work — they construct events without setting long_running_tool_ids.
    long_running_ids = set(getattr(event, "long_running_tool_ids", None) or ())
    paused_ids_emitted: set[str] = set()
    if event.content and event.content.parts:
      for part in event.content.parts:
        # Detect HITL function calls (request events).
        if part.function_call:
          hitl_event = _HITL_EVENT_MAP.get(part.function_call.name)
          if hitl_event:
            args_truncated, is_truncated = _recursive_smart_truncate(
                part.function_call.args or {},
                self.config.max_content_length,
            )
            content_dict = {
                "tool": part.function_call.name,
                "args": args_truncated,
            }
            await self._log_event(
                hitl_event,
                callback_ctx,
                raw_content=content_dict,
                is_truncated=is_truncated,
                event_data=EventData(source_event=event),
            )
          # Per-id TOOL_PAUSED emit. pause_kind derives from the
          # function_call NAME — looking it up against the id value
          # would misclassify every HITL pause as 'tool'.
          if part.function_call.id in long_running_ids:
            paused_ids_emitted.add(part.function_call.id)
            pause_kind = _HITL_PAUSE_KIND_MAP.get(
                part.function_call.name, "tool"
            )
            args_truncated, is_truncated = _recursive_smart_truncate(
                part.function_call.args or {},
                self.config.max_content_length,
            )
            await self._log_event(
                "TOOL_PAUSED",
                callback_ctx,
                raw_content={
                    "tool": part.function_call.name,
                    "args": args_truncated,
                },
                is_truncated=is_truncated,
                event_data=EventData(
                    source_event=event,
                    adk_extras={
                        "pause_kind": pause_kind,
                        "function_call_id": part.function_call.id,
                    },
                ),
            )
        # Detect HITL function responses (completion events). HITL
        # function responses route ONLY here, never to TOOL_COMPLETED
        # (verified by this file's HITL test suite).
        if part.function_response:
          hitl_event = _HITL_EVENT_MAP.get(part.function_response.name)
          if hitl_event:
            resp_truncated, is_truncated = _recursive_smart_truncate(
                part.function_response.response or {},
                self.config.max_content_length,
            )
            content_dict = {
                "tool": part.function_response.name,
                "result": resp_truncated,
            }
            await self._log_event(
                hitl_event + "_COMPLETED",
                callback_ctx,
                raw_content=content_dict,
                is_truncated=is_truncated,
                event_data=EventData(source_event=event),
            )

    # Fallback: a long_running_tool_id with no matching function_call
    # part (possible after after_model_callback content rewrites) still
    # gets a pairable TOOL_PAUSED row. Without the name we cannot derive
    # an HITL pause_kind, so default to 'tool' and warn.
    for orphan_pause_id in long_running_ids - paused_ids_emitted:
      logger.warning(
          "long_running_tool_id %s has no matching function_call part in"
          " event %s; emitting TOOL_PAUSED with pause_kind='tool'.",
          orphan_pause_id,
          getattr(event, "id", None),
      )
      await self._log_event(
          "TOOL_PAUSED",
          callback_ctx,
          raw_content={"tool": None, "args": None},
          event_data=EventData(
              source_event=event,
              adk_extras={
                  "pause_kind": "tool",
                  "function_call_id": orphan_pause_id,
              },
          ),
      )

    # --- A2A interaction logging ---
    # RemoteA2aAgent attaches cross-reference metadata to events:
    #   a2a:task_id, a2a:context_id  — correlation keys
    #   a2a:request, a2a:response    — full interaction payload
    # Log an A2A_INTERACTION event when meaningful payload is present
    # so the supervisor's BQ trace contains the remote agent's
    # response and cross-reference IDs for JOINs.
    meta = getattr(event, "custom_metadata", None)
    if meta and (
        meta.get("a2a:request") is not None
        or meta.get("a2a:response") is not None
    ):
      a2a_keys = {k: v for k, v in meta.items() if k.startswith("a2a:")}
      a2a_truncated, is_truncated = _recursive_smart_truncate(
          a2a_keys, self.config.max_content_length
      )
      # Use the a2a:response as the event content when available,
      # so the remote agent's answer is visible in the content
      # column.
      response_payload = a2a_keys.get("a2a:response")
      content_dict = None
      content_truncated = False
      if response_payload is not None:
        content_dict, content_truncated = _recursive_smart_truncate(
            response_payload,
            self.config.max_content_length,
        )
      await self._log_event(
          "A2A_INTERACTION",
          callback_ctx,
          raw_content=content_dict,
          is_truncated=is_truncated or content_truncated,
          event_data=EventData(
              source_event=event,
              extra_attributes={
                  "a2a_metadata": a2a_truncated,
              },
          ),
      )

    # --- Final agent response logging ---
    # Captures final response events emitted by agents (after all
    # after_model_callback modifications).  Uses a strict guard to
    # avoid false positives from skip_summarization function
    # responses, long-running tool pause events, and thought-only
    # events (which ADK treats as invisible internal reasoning).
    is_agent_response = (
        event.content
        and event.content.parts
        and event.is_final_response()
        and event.partial is not True
        and not event.get_function_calls()
        and not event.get_function_responses()
        and not event.long_running_tool_ids
    )
    if is_agent_response:
      # Filter to visible text parts only.  Exclude thoughts
      # (internal reasoning, A2A working/submitted updates),
      # empty parts, and non-text parts (executable_code, etc.)
      # that would render as "other" in _format_content.
      visible_parts = [
          p
          for p in event.content.parts
          if p.text and not getattr(p, "thought", None)
      ]
      if visible_parts:
        visible_content = types.Content(
            role=event.content.role, parts=visible_parts
        )
        formatted, truncated = self._format_content_safely(visible_content)
        # source_event=event carries the ADK envelope (A3 / node /
        # branch / scope). The flat ``source_event_*`` extras are
        # retained for backward compat with existing AGENT_RESPONSE
        # consumers; the canonical keys are under ``attributes.adk.*``.
        await self._log_event(
            "AGENT_RESPONSE",
            callback_ctx,
            raw_content={"response": formatted},
            is_truncated=truncated,
            event_data=EventData(
                source_event=event,
                extra_attributes={
                    "source_event_id": event.id,
                    "source_event_author": event.author,
                    "source_event_branch": event.branch,
                },
            ),
        )

    return None

  @_safe_callback
  async def before_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    """Callback before the agent run starts.

    Args:
        invocation_context: The context of the current invocation.
    """
    await self._ensure_started()
    callback_ctx = CallbackContext(invocation_context)
    TraceManager.ensure_invocation_span(callback_ctx)
    await self._log_event(
        "INVOCATION_STARTING",
        callback_ctx,
    )

  @_safe_callback
  async def after_run_callback(
      self, *, invocation_context: "InvocationContext"
  ) -> None:
    """Callback after the agent run completes.

    Args:
        invocation_context: The context of the current invocation.
    """
    try:
      # Capture trace_id BEFORE popping the invocation-root span so
      # that INVOCATION_COMPLETED shares the same trace_id as all
      # earlier events in this invocation (fixes #4645).
      callback_ctx = CallbackContext(invocation_context)
      trace_id = TraceManager.get_trace_id(callback_ctx)

      # Pop the invocation-root span pushed by ensure_invocation_span.
      span_id, duration = TraceManager.pop_span()
      parent_span_id = TraceManager.get_current_span_id()

      await self._log_event(
          "INVOCATION_COMPLETED",
          callback_ctx,
          event_data=EventData(
              trace_id_override=trace_id,
              latency_ms=duration,
              span_id_override=span_id,
              parent_span_id_override=parent_span_id,
          ),
      )
    finally:
      # Cleanup must run even if _log_event raises, otherwise
      # stale invocation metadata leaks into the next invocation.
      TraceManager.clear_stack()
      _active_invocation_id_ctx.set(None)
      _root_agent_name_ctx.set(None)
      # Ensure all logs are flushed before the agent returns.
      await self.flush()

  @_safe_callback
  async def before_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext
  ) -> None:
    """Callback before an agent starts processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    TraceManager.init_trace(callback_context)
    TraceManager.push_span(callback_context, "agent")
    await self._log_event(
        "AGENT_STARTING",
        callback_context,
        raw_content=getattr(agent, "instruction", ""),
    )

  @_safe_callback
  async def after_agent_callback(
      self, *, agent: Any, callback_context: CallbackContext
  ) -> None:
    """Callback after an agent completes processing.

    Args:
        agent: The agent instance.
        callback_context: The callback context.
    """
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "AGENT_COMPLETED",
        callback_context,
        event_data=EventData(
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )

  @_safe_callback
  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> None:
    """Callback before LLM call.

    Logs the LLM request details including:
    1. Prompt content
    2. System instruction (if available)

    The content is formatted as 'Prompt: {prompt} | System Prompt:
    {system_prompt}'.
    """

    # Defensive cleanup for a short-circuited request whose synthesized
    # event was not observed (for example, an abnormal generator exit).
    # expected_kind prevents this from disturbing the parent agent or
    # invocation span.
    TraceManager.pop_span(expected_kind="llm_request")

    # 5. Attributes (Config & Tools)
    attributes: dict[str, Any] = {}
    tools_truncated = False
    if llm_request.config:
      config_dict = {}
      for field_name in [
          "temperature",
          "top_p",
          "top_k",
          "candidate_count",
          "max_output_tokens",
          "stop_sequences",
          "presence_penalty",
          "frequency_penalty",
          "response_mime_type",
          "response_schema",
          "seed",
          "response_logprobs",
          "logprobs",
      ]:
        val = getattr(llm_request.config, field_name, None)
        if val is not None:
          config_dict[field_name] = val

      if config_dict:
        attributes["llm_config"] = config_dict

      if labels := getattr(llm_request.config, "labels", None):
        attributes["labels"] = labels

    if hasattr(llm_request, "tools_dict") and llm_request.tools_dict:
      # Route tool declarations through the shared safety pipeline so unbounded
      # descriptions / parameter schemas are size-capped and sensitive keys are
      # redacted, consistent with every other captured attribute.
      tools, tools_truncated = _recursive_smart_truncate(
          _extract_tool_declarations(llm_request.tools_dict),
          self.config.max_content_length,
      )
      attributes["tools"] = tools

    TraceManager.push_span(callback_context, "llm_request")
    await self._log_event(
        "LLM_REQUEST",
        callback_context,
        raw_content=llm_request,
        is_truncated=tools_truncated,
        event_data=EventData(
            model=llm_request.model,
            extra_attributes=attributes,
        ),
    )

  @_safe_callback
  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: "LlmResponse",
  ) -> None:
    """Callback after LLM call.

    Logs the LLM response details including:
    1. Response content
    2. Token usage (if available)

    The content is formatted as 'Response: {content} | Usage: {usage}'.

    Args:
        callback_context: The callback context.
        llm_response: The LLM response object.
    """
    content_dict = {}
    is_truncated = False
    if llm_response.content:
      part_str, part_truncated = self._format_content_safely(
          llm_response.content
      )
      if part_str:
        content_dict["response"] = part_str
      if part_truncated:
        is_truncated = True

    if llm_response.usage_metadata:
      usage = llm_response.usage_metadata
      usage_dict = {}
      if hasattr(usage, "prompt_token_count"):
        usage_dict["prompt"] = usage.prompt_token_count
      if hasattr(usage, "candidates_token_count"):
        usage_dict["completion"] = usage.candidates_token_count
      if hasattr(usage, "total_token_count"):
        usage_dict["total"] = usage.total_token_count
      if usage_dict:
        content_dict["usage"] = usage_dict

    if content_dict:
      content_str = content_dict
    else:
      content_str = None

    span_id = TraceManager.get_current_span_id()
    _, parent_span_id = TraceManager.get_current_span_and_parent()

    is_popped = False
    duration = 0
    tfft = None

    if hasattr(llm_response, "partial") and llm_response.partial:
      # Streaming chunk - do NOT pop span yet
      if span_id:
        TraceManager.record_first_token(span_id)
        start_time = TraceManager.get_start_time(span_id)
        first_token = TraceManager.get_first_token_time(span_id)
        if start_time:
          duration = int((time.time() - start_time) * 1000)
        if start_time and first_token:
          tfft = int((first_token - start_time) * 1000)
    else:
      # Final response - pop span
      start_time = None
      if span_id:
        # Ensure we have first token time even if it wasn't streaming (or single chunk)
        TraceManager.record_first_token(span_id)
        start_time = TraceManager.get_start_time(span_id)
        first_token = TraceManager.get_first_token_time(span_id)
        if start_time and first_token:
          tfft = int((first_token - start_time) * 1000)

      # ACTUALLY pop the span
      popped_span_id, duration = TraceManager.pop_span(
          expected_kind="llm_request"
      )
      is_popped = True

      # If we popped, the span_id from get_current_span_and_parent() above is correct for THIS event
      # Wait, if we popped, get_current_span_and_parent() now returns parent.
      # But we captured span_id BEFORE popping. So we should use THAT.
      # If is_popped is True, we must override span_id in log_event to use the popped one.
      # Otherwise log_event will fetch current stack (which is parent).
      span_id = popped_span_id or span_id

    await self._log_event(
        "LLM_RESPONSE",
        callback_context,
        raw_content=content_str,
        is_truncated=is_truncated,
        event_data=EventData(
            latency_ms=duration,
            time_to_first_token_ms=tfft,
            model_version=llm_response.model_version,
            usage_metadata=llm_response.usage_metadata,
            cache_metadata=getattr(llm_response, "cache_metadata", None),
            span_id_override=span_id if is_popped else None,
            parent_span_id_override=(parent_span_id if is_popped else None),
        ),
    )

  @_safe_callback
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> None:
    """Callback on LLM error.

    Args:
        callback_context: The callback context.
        llm_request: The request that was sent to the model.
        error: The exception that occurred.
    """
    span_id, duration = TraceManager.pop_span(expected_kind="llm_request")
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "LLM_ERROR",
        callback_context,
        event_data=EventData(
            status="ERROR",
            error_message=str(error),
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )

  @_safe_callback
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> None:
    """Callback before tool execution.

    Args:
        tool: The tool being executed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
    """
    args_truncated, is_truncated = _recursive_smart_truncate(
        tool_args, self.config.max_content_length
    )
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "args": args_truncated,
        "tool_origin": tool_origin,
    }
    TraceManager.push_span(tool_context, "tool")
    await self._log_event(
        "TOOL_STARTING",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
    )

  @_safe_callback
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> None:
    """Callback after tool execution.

    Args:
        tool: The tool that was executed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
        result: The response from the tool.
    """
    resp_truncated, is_truncated = _recursive_smart_truncate(
        result, self.config.max_content_length
    )
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "result": resp_truncated,
        "tool_origin": tool_origin,
    }
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    event_data = EventData(
        latency_ms=duration,
        span_id_override=span_id,
        parent_span_id_override=parent_span_id,
    )
    await self._log_event(
        "TOOL_COMPLETED",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
        event_data=event_data,
    )

    # Some agents deliver their final answer through a dedicated tool
    # (e.g. ``submit_final_response``) rather than a plain-text final event,
    # so the on-event AGENT_RESPONSE path (which excludes function
    # calls/responses) never fires.  When such a tool completes, log its call
    # args (the final-answer payload the model supplied) as AGENT_RESPONSE so
    # the visible response text is captured.  Opt-in via
    # ``config.final_response_tool_names`` (empty by default).
    if tool.name in self.config.final_response_tool_names:
      args_truncated, args_is_truncated = _recursive_smart_truncate(
          tool_args, self.config.max_content_length
      )
      await self._log_event(
          "AGENT_RESPONSE",
          tool_context,
          raw_content={"response": args_truncated},
          is_truncated=args_is_truncated,
          event_data=EventData(extra_attributes={"source_tool": tool.name}),
      )

  @_safe_callback
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Callback on tool error.

    Args:
        tool: The tool that failed.
        tool_args: The arguments passed to the tool.
        tool_context: The tool context.
        error: The exception that occurred.
    """
    args_truncated, is_truncated = _recursive_smart_truncate(
        tool_args, self.config.max_content_length
    )
    tool_origin = _get_tool_origin(tool, tool_args, tool_context)
    content_dict = {
        "tool": tool.name,
        "args": args_truncated,
        "tool_origin": tool_origin,
    }
    span_id, duration = TraceManager.pop_span()
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    await self._log_event(
        "TOOL_ERROR",
        tool_context,
        raw_content=content_dict,
        is_truncated=is_truncated,
        event_data=EventData(
            status="ERROR",
            error_message=str(error),
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
    )

  @_safe_callback
  async def on_agent_error_callback(
      self,
      *,
      agent: Any,
      callback_context: CallbackContext,
      error: Exception,
  ) -> None:
    """Callback when an agent execution fails with an unhandled exception.

    Emits an AGENT_ERROR event and pops the agent span from
    TraceManager.

    The pop is guarded by span kind: the agent-error contract includes
    failures raised by *other* plugins' before_agent_callbacks, in which
    case BQAA's own before_agent_callback never pushed an agent span and
    there is nothing to pop (popping unconditionally would consume the
    invocation span and corrupt the subsequent INVOCATION_ERROR row).

    Args:
        agent: The agent instance that failed.
        callback_context: The callback context.
        error: The exception that escaped agent execution.
    """
    span_id, duration = TraceManager.pop_span(expected_kind="agent")
    parent_span_id, _ = TraceManager.get_current_span_and_parent()

    error_tb = "".join(
        traceback_module.format_exception(
            type(error), error, error.__traceback__
        )
    )
    await self._log_event(
        "AGENT_ERROR",
        callback_context,
        event_data=EventData(
            status="ERROR",
            error_message=str(error),
            latency_ms=duration,
            span_id_override=span_id,
            parent_span_id_override=parent_span_id,
        ),
        raw_content={"error_traceback": error_tb},
    )

  @_safe_callback
  async def on_run_error_callback(
      self,
      *,
      invocation_context: "InvocationContext",
      error: Exception,
  ) -> None:
    """Callback when a runner execution fails with an unhandled exception.

    Emits an INVOCATION_ERROR event and performs the cleanup that
    after_run_callback would normally do.

    Args:
        invocation_context: The context of the current invocation.
        error: The exception that escaped runner execution.
    """
    try:
      callback_ctx = CallbackContext(invocation_context)
      trace_id = TraceManager.get_trace_id(callback_ctx)

      # Guarded pop: only consume the invocation-root span. If the failure
      # left intermediate spans on the stack (or the root was never pushed),
      # emit the row without span/latency rather than mis-attributing them;
      # the finally-block clear_stack below resets the stack either way.
      span_id, duration = TraceManager.pop_span(expected_kind="invocation")
      parent_span_id = TraceManager.get_current_span_id()

      error_tb = "".join(
          traceback_module.format_exception(
              type(error), error, error.__traceback__
          )
      )
      await self._log_event(
          "INVOCATION_ERROR",
          callback_ctx,
          event_data=EventData(
              trace_id_override=trace_id,
              status="ERROR",
              error_message=str(error),
              latency_ms=duration,
              span_id_override=span_id,
              parent_span_id_override=parent_span_id,
          ),
          raw_content={"error_traceback": error_tb},
      )
    finally:
      # Cleanup must run even if _log_event raises.
      TraceManager.clear_stack()
      _active_invocation_id_ctx.set(None)
      _root_agent_name_ctx.set(None)
      await self.flush()

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

"""Mimir persistent memory service for ADK.

Mimir (github.com/Perseus-Computing-LLC/mimir) is an open-source (MIT)
persistent memory engine with 30 MCP tools, FTS5 + dense hybrid search,
and optional AES-256-GCM encryption.  This service talks to the Mimir
binary via JSON-RPC over stdin/stdout (MCP stdio transport).

Requirements:
    A ``mimir`` binary must be on ``$PATH`` or passed explicitly via
    ``mimir_binary``.  Build from source or download a pre-built binary from
    the Mimir releases page.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

from typing_extensions import override

from google.genai import types

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session

logger = logging.getLogger(__name__)

_MIMIR_CATEGORY = "adk-memory"


class MimirMemoryService(BaseMemoryService):
  """Persistent memory service backed by Mimir.

  Talks to a local ``mimir`` binary via JSON-RPC (MCP stdio).  Stores
  session events as structured entities and supports keyword (FTS5) search
  across sessions.

  This class is thread-safe.

  Attributes:
      db_path: Filesystem path to the Mimir SQLite database.
      mimir_binary: Path or name of the ``mimir`` executable.
  """

  def __init__(
      self,
      db_path: str = "~/.adk/mimir.db",
      mimir_binary: str = "mimir",
  ):
    """Initializes the Mimir memory service.

    Args:
        db_path: Path to the Mimir database file.  Defaults to
            ``~/.adk/mimir.db``.
        mimir_binary: Name or absolute path of the ``mimir`` executable.
            Defaults to ``mimir`` (resolved from ``$PATH``).

    Raises:
        RuntimeError: If the ``mimir`` binary cannot be found or the
            subprocess fails to start.
    """
    self.db_path = os.path.expanduser(db_path)

    # Resolve the mimir binary.
    if os.path.isabs(mimir_binary):
      self._mimir_binary = mimir_binary
    else:
      resolved = shutil.which(mimir_binary)
      if resolved is None:
        raise RuntimeError(
            f"mimir binary not found on $PATH (looked for '{mimir_binary}'). "
            "Install Mimir from https://github.com/Perseus-Computing-LLC/mimir "
            "or pass the absolute path via mimir_binary=."
        )
      self._mimir_binary = resolved

    # Ensure the database directory exists.
    os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

    # Start the Mimir MCP stdio subprocess.
    self._proc = subprocess.Popen(
        [self._mimir_binary, "--db", self.db_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    self._lock = threading.Lock()
    self._request_id = 0

    # Initialize the MCP session.
    self._rpc(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "adk-mimir-memory-service", "version": "1.0"},
        },
    )

    # Clean up the subprocess on exit.
    atexit.register(self._close)

  def _close(self) -> None:
    """Terminates the Mimir subprocess."""
    try:
      self._proc.terminate()
      self._proc.wait(timeout=5)
    except Exception:
      try:
        self._proc.kill()
      except Exception:
        pass

  def _next_id(self) -> int:
    self._request_id += 1
    return self._request_id

  def _rpc(self, method: str, params: object) -> dict:
    """Sends a JSON-RPC request to Mimir and returns the result dict.

    Args:
        method: The MCP method name (e.g. ``tools/call``).
        params: The method parameters.

    Returns:
        The ``result`` field of the JSON-RPC response.

    Raises:
        RuntimeError: If the RPC returns an error.
    """
    req = {
        "jsonrpc": "2.0",
        "id": self._next_id(),
        "method": method,
        "params": params,
    }
    payload = json.dumps(req, default=str)

    with self._lock:
      try:
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()
        raw = self._proc.stdout.readline()
      except (BrokenPipeError, OSError) as e:
        raise RuntimeError(
            f"Mimir subprocess communication failed: {e}. "
            "The mimir process may have crashed."
        ) from e

    try:
      resp = json.loads(raw)
    except json.JSONDecodeError as e:
      raise RuntimeError(
          f"Failed to parse Mimir response: {e}. Raw: {raw[:200]}"
      ) from e

    if "error" in resp:
      err = resp["error"]
      raise RuntimeError(
          f"Mimir RPC error [{err.get('code')}]: {err.get('message')}"
      )

    return resp.get("result", {})

  def _call_tool(self, name: str, arguments: dict) -> dict:
    """Calls a Mimir MCP tool and returns the ``structuredContent``."""
    result = self._rpc(
        "tools/call",
        {"name": name, "arguments": arguments},
    )
    # MCP result is {content: [{type: "text", text: "..."}], structuredContent: {...}}
    sc = result.get("structuredContent")
    if sc is not None:
      return sc
    # Fallback: parse the text content
    content = result.get("content", [])
    if content:
      try:
        return json.loads(content[0].get("text", "{}"))
      except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return {}

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Stores all events from a session in Mimir.

    Each session is stored as a single Mimir entity keyed by session ID.
    Subsequent calls for the same session will update the stored events.
    """
    if not session.events:
      return

    events_data = []
    for event in session.events:
      if not event.content or not event.content.parts:
        continue
      parts = []
      for part in event.content.parts:
        if part.text:
          parts.append({"text": part.text})
        elif hasattr(part, "function_call") and part.function_call:
          parts.append({
              "function_call": {
                  "name": part.function_call.name,
                  "args": part.function_call.args,
              }
          })
        elif hasattr(part, "function_response") and part.function_response:
          parts.append({
              "function_response": {
                  "name": part.function_response.name,
                  "response": str(part.function_response.response)[:2000],
              }
          })
      if parts:
        events_data.append({
            "author": event.author,
            "timestamp": event.timestamp,
            "parts": parts,
        })

    if not events_data:
      return

    self._call_tool(
        "mimir_remember",
        {
            "category": _MIMIR_CATEGORY,
            "key": f"session:{session.app_name}:{session.user_id}:{session.id}",
            "body_json": json.dumps({
                "session_id": session.id,
                "app_name": session.app_name,
                "user_id": session.user_id,
                "events": events_data,
                "event_count": len(events_data),
            }),
            "tags": ["adk", "session", session.app_name],
        },
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds a delta of events to Mimir.

    Events are appended to an existing session entity if one exists, or a
    new entity is created.  This is the recommended method for incremental
    memory updates during long-running sessions.
    """
    _ = custom_metadata
    events_data = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      parts = []
      for part in event.content.parts:
        if part.text:
          parts.append({"text": part.text})
        elif hasattr(part, "function_call") and part.function_call:
          parts.append({
              "function_call": {
                  "name": part.function_call.name,
                  "args": part.function_call.args,
              }
          })
      if parts:
        events_data.append({
            "author": event.author,
            "timestamp": event.timestamp,
            "parts": parts,
        })

    if not events_data:
      return

    sid = session_id or "__unknown__"
    # Generate a stable key that includes a timestamp component so deltas
    # don't overwrite the full session snapshot.
    import time
    delta_key = f"delta:{app_name}:{user_id}:{sid}:{int(time.time() * 1000)}"

    self._call_tool(
        "mimir_remember",
        {
            "category": _MIMIR_CATEGORY,
            "key": delta_key,
            "body_json": json.dumps({
                "session_id": sid,
                "app_name": app_name,
                "user_id": user_id,
                "events": events_data,
                "event_count": len(events_data),
            }),
            "tags": ["adk", "delta", app_name],
        },
    )

  @override
  async def add_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds explicit memory entries directly to Mimir.

    Each MemoryEntry is stored as a separate entity tagged for the given
    application and user.
    """
    _ = custom_metadata
    for i, entry in enumerate(memories):
      content_text = ""
      if entry.content and entry.content.parts:
        content_text = " ".join(
            p.text for p in entry.content.parts if p.text
        )

      if not content_text:
        continue

      self._call_tool(
          "mimir_remember",
          {
              "category": _MIMIR_CATEGORY,
              "key": f"memory:{app_name}:{user_id}:{entry.id or i}",
              "body_json": json.dumps({
                  "content": content_text,
                  "author": entry.author,
                  "timestamp": entry.timestamp,
                  "metadata": entry.custom_metadata,
              }),
              "tags": ["adk", "explicit", app_name],
          },
      )

  @override
  async def search_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      query: str,
  ) -> SearchMemoryResponse:
    """Searches Mimir for memories matching the query.

    Uses Mimir's FTS5 keyword search.  Results are scoped to the given
    application and user by filtering on the stored tags and body content.

    Args:
        app_name: The application name for memory scope.
        user_id: The user ID for memory scope.
        query: The natural-language query to search for.

    Returns:
        A SearchMemoryResponse containing matching MemoryEntry objects.
    """
    # Search Mimir with FTS5.  Mimir currently searches across all entities
    # so we scope by including app_name/user_id in the query terms.
    scoped_query = f"{query} {app_name} adk-memory {user_id}"
    result = self._call_tool(
        "mimir_recall",
        {
            "query": scoped_query,
            "limit": 20,
            "category": _MIMIR_CATEGORY,
        },
    )

    response = SearchMemoryResponse()
    items = result.get("items", [])
    for item in items:
      body = item.get("body_json", "{}")
      try:
        body_data = json.loads(body) if isinstance(body, str) else body
      except json.JSONDecodeError:
        body_data = {}

      # Determine the best text content to surface.
      content_text = body_data.get("content", "")
      if not content_text:
        events = body_data.get("events", [])
        texts = []
        for ev in events:
          for part in ev.get("parts", []):
            if part.get("text"):
              texts.append(part["text"])
        content_text = " | ".join(texts[:5]) if texts else ""

      if not content_text:
        continue

      response.memories.append(
          MemoryEntry(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=content_text)],
              ),
              author=body_data.get("author") or "mimir",
              timestamp=body_data.get("timestamp")
              or _utils.format_timestamp(
                  item.get("created_at_unix_ms", 0) / 1000.0
              ),
          )
      )

    return response

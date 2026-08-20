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

from typing import Awaitable
from typing import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict

if TYPE_CHECKING:
  from ..agents.readonly_context import ReadonlyContext

  HeaderProvider = Callable[
      [ReadonlyContext], dict[str, str] | Awaitable[dict[str, str]]
  ]
else:
  HeaderProvider = Callable[..., dict[str, str] | Awaitable[dict[str, str]]]


class RemoteMcpServer(BaseModel):
  """A remote MCP server executed server-side by the Managed Agents API.

  ``ManagedAgent`` forwards the server's URL and headers to
  ``interactions.create``; the Interactions backend opens the MCP session and
  runs the tools. Only remote (HTTP/streamable) MCP servers are supported.

  This is server-side MCP: unlike ``LlmAgent``'s ``McpToolset`` (which opens the
  session and executes tools client-side), ADK never connects to the MCP server
  here. The reused concept is the ``header_provider`` callback contract.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

  url: str
  """Full URL of the remote MCP server endpoint (e.g.
  'https://api.example.com/mcp'). Maps to ``MCPServerParam.url``."""

  name: str | None = None
  """Optional server label. Maps to ``MCPServerParam.name``."""

  headers: dict[str, str] | None = None
  """Static headers sent on every turn (e.g. a fixed API key). Merged with
  ``header_provider`` output; ``header_provider`` wins on key conflict."""

  allowed_tools: list[str] | None = None
  """Restrict which of the server's tools are exposed. Maps to
  ``MCPServerParam.allowed_tools``."""

  header_provider: HeaderProvider | None = None
  """Runtime callback that mints headers (e.g. a fresh bearer token) at request
  time. Invoked by ``ManagedAgent`` during resolution (runner-driven), once per
  turn. Receives a ``ReadonlyContext`` and returns a headers dict (or an
  awaitable of one). Same contract as ``LlmAgent``'s
  ``McpToolset.header_provider``."""

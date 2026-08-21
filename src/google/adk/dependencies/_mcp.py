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

"""The MCP Python SDK, as ADK's open-source build resolves it.

Upstream publishes the SDK as `mcp`, so this flavor names it directly. Google's
internal build resolves the same module path to a sibling file naming the copy
vendored there, which carries a different distribution name. Every MCP import
in ADK goes through this module so that difference lives in one place.

The pin admits both 1.x and 2.x, so this module also spans them. Fifteen of
the eighteen names below sit at the same path in both. The three that moved
are resolved here, once, rather than at seventeen call sites:

  * `McpError` was renamed `MCPError`
  * `FastMCP` was renamed `MCPServer`, and `mcp.server.fastmcp` became
    `mcp.server.mcpserver`
  * `Context` kept its name but moved with that module

Try 2.x first. A stale 1.x path that still happens to exist in 2.x would
otherwise win and bind the wrong object.

Three further names have no 1.x path at all rather than a moved one: the
client extension seam `ClientSession` grew in 2.x. They are bound to `Any`
on 1.x so that the modules annotating them stay importable there, and the
opt-in that would reach them is refused before it can (`McpToolset`).

This flavor must never name the internal copy. That name is already taken on
PyPI by an unrelated project, so a released wheel importing it would bind to
someone else's package on any machine that happened to have it installed.
"""

from __future__ import annotations

from typing import Any

from mcp import ClientSession as ClientSession
from mcp import SamplingCapability as SamplingCapability
from mcp import StdioServerParameters as StdioServerParameters
from mcp import types as types
from mcp.client.session import ElicitationFnT as ElicitationFnT
from mcp.client.session import SamplingFnT as SamplingFnT
from mcp.client.sse import sse_client as sse_client
from mcp.client.stdio import stdio_client as stdio_client
from mcp.client.streamable_http import create_mcp_http_client as create_mcp_http_client
from mcp.client.streamable_http import streamable_http_client as streamable_http_client
from mcp.server.session import ServerSession as ServerSession
from mcp.types import CallToolResult as CallToolResult
from mcp.types import ListResourcesResult as ListResourcesResult
from mcp.types import ListToolsResult as ListToolsResult
from mcp.types import Tool as Tool

try:
  from mcp.server.mcpserver import Context as Context
  from mcp.server.mcpserver import MCPServer as FastMCP
  from mcp.shared.exceptions import MCPError as McpError

  # Which major resolved. Three places downstream need it and cannot ask the
  # SDK themselves: `_httpx` pairs the HTTP library to it, `session_context`
  # picks the type `read_timeout_seconds` wants (`float` on 2.x, `timedelta`
  # on 1.x -- neither accepts the other), and `mcp_session_manager` skips the
  # httpx-1.x OTel instrumentor.
  IS_MCP_SDK_V2 = True
except ImportError:
  from mcp.server.fastmcp import Context as Context
  from mcp.server.fastmcp import FastMCP as FastMCP
  from mcp.shared.exceptions import McpError as McpError

  IS_MCP_SDK_V2 = False

if IS_MCP_SDK_V2:
  from mcp.client.extension import ClaimContext as ClaimContext
  from mcp.client.extension import NotificationBinding as NotificationBinding
  from mcp.client.extension import ResultClaim as ResultClaim
else:
  # `Any`, not a stand-in class: a stand-in would satisfy an `isinstance` or a
  # constructor call that has no business succeeding here. `Any` is also why
  # the annotations naming these stay unsubscripted downstream -- `Any[Any]`
  # is a `TypeError` to anything that resolves an annotation, and the point of
  # these three is to keep 1.x importable.
  ClaimContext = Any
  NotificationBinding = Any
  ResultClaim = Any

__all__ = [
    "IS_MCP_SDK_V2",
    "CallToolResult",
    "ClaimContext",
    "ClientSession",
    "Context",
    "ElicitationFnT",
    "FastMCP",
    "ListResourcesResult",
    "ListToolsResult",
    "McpError",
    "NotificationBinding",
    "ResultClaim",
    "SamplingCapability",
    "SamplingFnT",
    "ServerSession",
    "StdioServerParameters",
    "Tool",
    "create_mcp_http_client",
    "sse_client",
    "stdio_client",
    "streamable_http_client",
    "types",
]

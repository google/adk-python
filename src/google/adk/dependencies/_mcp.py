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

This flavor must never name the internal copy. That name is already taken on
PyPI by an unrelated project, so a released wheel importing it would bind to
someone else's package on any machine that happened to have it installed.
"""

from __future__ import annotations

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
from mcp.server.fastmcp import Context as Context
from mcp.server.fastmcp import FastMCP as FastMCP
from mcp.server.session import ServerSession as ServerSession
from mcp.shared.exceptions import McpError as McpError
from mcp.types import ListResourcesResult as ListResourcesResult
from mcp.types import ListToolsResult as ListToolsResult
from mcp.types import Tool as Tool

__all__ = [
    "ClientSession",
    "Context",
    "ElicitationFnT",
    "FastMCP",
    "ListResourcesResult",
    "ListToolsResult",
    "McpError",
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

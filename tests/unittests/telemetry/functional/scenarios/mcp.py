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

"""The MCP scenario: the canonical agent, with its tool served over MCP.

A ``FakeMcpSession`` substitutes the live ``McpClientSession`` so the
scenario doesn't need a running MCP server. ``McpToolset.create_session`` is
patched to hand it out instead of dialing ``StdioServerParameters``.
"""

from __future__ import annotations

from datetime import timedelta

from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.mcp_tool.mcp_session_manager import _DebugHttpxClientFactory
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import ProgressFnT
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import httpx
from mcp import ClientSession as McpClientSession
from mcp import StdioServerParameters
from mcp.types import CallToolResult
from mcp.types import ListToolsResult
from mcp.types import PaginatedRequestParams
from mcp.types import TextContent
from mcp.types import Tool as McpTool
import pytest
from typing_extensions import override

from ....testing_utils import TestInMemoryRunner
from .conversation import AGENT_DESCRIPTION
from .conversation import AGENT_NAME
from .conversation import BASE_INSTRUCTION
from .conversation import TOOL_NAME
from .conversation import TOOL_RESULT_PREFIX

# The MCP server resolves the tool the canned conversation calls, under the
# same name and signature the agent's own ``some_tool`` has: one conversation
# then drives every scenario, and what the MCP scenario adds is where the
# tool came from, not what the model said.
MCP_TOOL_DESCRIPTION = "Echoes back its input."

# The one tool a ``FakeMcpSession`` resolves, unless given others.
DEFAULT_MCP_TOOL = McpTool(
    name=TOOL_NAME,
    description=MCP_TOOL_DESCRIPTION,
    inputSchema={
        "type": "object",
        "properties": {"arg1": {"type": "string"}},
        "required": ["arg1"],
    },
)

# The (fake) streamable HTTP server a ``tools/call`` is posted to when the
# scenario is asked for a session that talks over HTTP. Everything about the
# exchange is pinned here, so what the record carries is a value a reader of
# the golden can look up.
MCP_SERVER_URL = "https://mcp.example.com/mcp"
MCP_SESSION_ID = "mcp-session-1"
MCP_PROTOCOL_VERSION = "2025-06-18"
# A credential on the request. The case allowlists `authorization`, so the
# golden shows what asking for it gets you: the marker, never the secret.
MCP_AUTHORIZATION = "Bearer some-secret-token"
MCP_REQUEST_BODY = (
    '{"jsonrpc": "2.0", "id": 1, "method": "tools/call",'
    f' "params": {{"name": "{TOOL_NAME}"}}}}'
)
MCP_RESPONSE_BODY = '{"jsonrpc": "2.0", "id": 1, "result": {"isError": false}}'


class FakeMcpSession(McpClientSession):
  """Minimal ``McpClientSession`` stand-in with a counted ``list_tools()``.

  Subclasses ``McpClientSession`` (and skips its real ``__init__``) so that
  every ``isinstance(x, McpClientSession)`` check in ADK and in the MCP
  Python client passes, without needing to wire up the underlying anyio
  memory streams + peer process.

  With ``over_http``, ``call_tool`` additionally posts the JSON-RPC call
  through the httpx client ADK builds for a streamable HTTP connection --
  hook, redaction and all -- against a canned server. That is what makes the
  transport itself observable: the exchange is recorded from inside the
  ``execute_tool`` span, exactly where a real MCP tool call would record it.
  """

  def __init__(  # pyright: ignore[reportMissingSuperCall]
      self, *, tools: list[McpTool] | None = None, over_http: bool = False
  ) -> None:
    # Deliberately skip ``McpClientSession.__init__``: the real one wants
    # live anyio streams + a peer process. ``isinstance`` checks still
    # succeed, which is all ADK's MCP plumbing requires.
    self._tools: list[McpTool] = (
        tools if tools is not None else [DEFAULT_MCP_TOOL]
    )
    self._over_http = over_http
    self.list_tools_call_count: int = 0

  async def _post_tool_call(self) -> None:
    """Posts one ``tools/call`` over ADK's instrumented httpx client."""

    def respond(_request: httpx.Request) -> httpx.Response:
      return httpx.Response(
          200,
          headers={
              "content-type": "application/json",
              "mcp-session-id": MCP_SESSION_ID,
              "mcp-protocol-version": MCP_PROTOCOL_VERSION,
          },
          text=MCP_RESPONSE_BODY,
      )

    def base_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
      del timeout, auth  # The canned server has neither to honour.
      return httpx.AsyncClient(
          headers=headers, transport=httpx.MockTransport(respond)
      )

    # The same wrapper `MCPSessionManager._create_client` puts around the
    # connection's factory for an HTTP transport.
    factory = _DebugHttpxClientFactory(base_factory)
    async with factory(headers={"Authorization": MCP_AUTHORIZATION}) as client:
      await client.post(MCP_SERVER_URL, content=MCP_REQUEST_BODY)

  @override
  async def list_tools(
      self,
      cursor: str | None = None,
      *,
      params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    self.list_tools_call_count += 1
    return ListToolsResult(tools=list(self._tools))

  @override
  async def call_tool(
      self,
      name: str,
      arguments: dict[str, object] | None = None,
      read_timeout_seconds: timedelta | None = None,
      progress_callback: ProgressFnT | None = None,
      *,
      meta: dict[str, object] | None = None,
  ) -> CallToolResult:
    """Answers like the agent's own ``some_tool``, over MCP."""
    if self._over_http:
      await self._post_tool_call()
    argument = (arguments or {}).get("arg1", "")
    return CallToolResult(
        content=[
            TextContent(type="text", text=f"{TOOL_RESULT_PREFIX}{argument}")
        ]
    )


def build_mcp_test_runner(
    model: BaseLlm,
    monkeypatch: pytest.MonkeyPatch,
    fake_session: FakeMcpSession,
) -> TestInMemoryRunner:
  """Builds an agent runner whose only tool source is a (fake) MCP server.

  Patches the toolset's ``MCPSessionManager`` so ``create_session`` returns
  ``fake_session`` (no socket / subprocess) and ``close`` is a no-op. The
  model answers in one turn, so an assertion on
  ``fake_session.list_tools_call_count`` is unambiguous: exactly one agent
  invocation is performed.
  """
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(command="unused-by-test"),
      )
  )

  async def _create_session(
      *_args, **_kwargs
  ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return fake_session

  async def _close(
      *_args, **_kwargs
  ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return None

  monkeypatch.setattr(
      toolset._mcp_session_manager,
      "create_session",
      _create_session,  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]
  )
  monkeypatch.setattr(
      toolset._mcp_session_manager, "close", _close
  )  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]

  return TestInMemoryRunner(
      node=Agent(
          name=AGENT_NAME,
          description=AGENT_DESCRIPTION,
          instruction=BASE_INSTRUCTION,
          model=model,
          tools=[toolset],
      )
  )

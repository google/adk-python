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

"""Expose an ADK agent as an MCP server."""

from __future__ import annotations

import base64
from typing import MutableMapping
from typing import Optional
import weakref

from google.genai import types
from mcp import types as mcp_types
from mcp.server.fastmcp import Context
from mcp.server.fastmcp import FastMCP

from ...agents.base_agent import BaseAgent
from ...artifacts.in_memory_artifact_service import InMemoryArtifactService
from ...auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ...features import experimental
from ...features import FeatureName
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.in_memory_session_service import InMemorySessionService

_MCP_USER_ID = "mcp_user"
_INLINE_RESOURCE_URI = "resource://adk-agent/inline-data"


def _build_runner(agent: BaseAgent) -> Runner:
  """Builds a Runner for the agent using in-memory services."""
  return Runner(
      app_name=agent.name or "adk_agent",
      agent=agent,
      artifact_service=InMemoryArtifactService(),
      session_service=InMemorySessionService(),
      memory_service=InMemoryMemoryService(),
      credential_service=InMemoryCredentialService(),
  )


def _part_to_content(part: types.Part) -> Optional[mcp_types.ContentBlock]:
  """Maps one ADK content part to an MCP content block.

  Args:
    part: An ADK content part from the agent's response.

  Returns:
    The matching MCP content block (text, image, audio, or embedded resource),
    or None for a part with no renderable content (e.g. a function call).
  """
  if part.text:
    return mcp_types.TextContent(type="text", text=part.text)
  blob = part.inline_data
  if blob is not None and blob.data is not None:
    data = base64.b64encode(blob.data).decode("ascii")
    mime = blob.mime_type or "application/octet-stream"
    if mime.startswith("image/"):
      return mcp_types.ImageContent(type="image", data=data, mimeType=mime)
    if mime.startswith("audio/"):
      return mcp_types.AudioContent(type="audio", data=data, mimeType=mime)
    return mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.BlobResourceContents(
            uri=_INLINE_RESOURCE_URI, blob=data, mimeType=mime
        ),
    )
  return None


async def _run_agent(
    runner: Runner,
    request: str,
    ctx: Optional[Context] = None,
    sessions: Optional[MutableMapping[object, str]] = None,
) -> list[mcp_types.ContentBlock]:
  """Runs the agent for one request and returns its final response content.

  When ``ctx`` and ``sessions`` are supplied, one ADK session is reused per MCP
  connection, so successive calls form a single conversation; otherwise a fresh
  session is created. Intermediate (non-final) text events are forwarded as MCP
  progress notifications when ``ctx`` is supplied; progress is a no-op unless
  the host requested it.

  Args:
    runner: The Runner that executes the agent.
    request: The user request text for this call.
    ctx: The MCP tool call context, used for progress and session reuse.
    sessions: Per-connection map from MCP connection to ADK session id.

  Returns:
    The agent's final response as a list of MCP content blocks (text plus any
    images, audio, or other data the agent produced).
  """
  session_id: Optional[str] = None
  if ctx is not None and sessions is not None:
    session_id = sessions.get(ctx.session)
  if session_id is None:
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=_MCP_USER_ID
    )
    session_id = session.id
    if ctx is not None and sessions is not None:
      sessions[ctx.session] = session_id
  new_message = types.Content(role="user", parts=[types.Part(text=request)])
  final_content: list[mcp_types.ContentBlock] = []
  async for event in runner.run_async(
      user_id=_MCP_USER_ID,
      session_id=session_id,
      new_message=new_message,
  ):
    if not (event.content and event.content.parts):
      continue
    if event.is_final_response():
      for part in event.content.parts:
        block = _part_to_content(part)
        if block is not None:
          final_content.append(block)
    elif ctx is not None:
      text = "".join(part.text or "" for part in event.content.parts)
      if text:
        await ctx.report_progress(progress=0.0, message=text)
  return final_content


@experimental(FeatureName.MCP_AGENT_SERVER)
def to_mcp_server(
    agent: BaseAgent,
    *,
    name: Optional[str] = None,
    instructions: Optional[str] = None,
    runner: Optional[Runner] = None,
) -> FastMCP:
  """Exposes an ADK agent as an MCP server.

  The returned server registers a single MCP tool that runs the agent: an MCP
  host (e.g. Claude Code, OpenAI Codex, an IDE, or any MCP client) sends a
  request string and receives the agent's final response, including any images
  or audio the agent produced. This is the MCP counterpart of ``to_a2a``; it
  lets harnesses that speak MCP drive an ADK agent.

  One ADK session is kept per MCP connection, so successive tool calls on the
  same connection form a single multi-turn conversation.

  The caller chooses the transport, e.g. ``server.run(transport="stdio")`` for
  a local host or ``server.run(transport="streamable-http")`` for a networked
  one.

  Args:
    agent: The ADK agent to serve.
    name: The MCP server and tool name. Defaults to the agent's name.
    instructions: Optional instructions the MCP host may show to its model.
    runner: A pre-built Runner. If omitted, one is created with in-memory
      services.

  Returns:
    A ``FastMCP`` server exposing the agent as a single tool.

  Example::

      agent = LlmAgent(name="assistant", model="gemini-2.0-flash", ...)
      server = to_mcp_server(agent)
      server.run(transport="stdio")
  """
  tool_name = name or agent.name or "adk_agent"
  server = FastMCP(name=tool_name, instructions=instructions)
  agent_runner = runner if runner is not None else _build_runner(agent)
  # Maps each MCP connection to its ADK session; WeakKeyDictionary drops the
  # entry when the connection is garbage-collected. pylint wrongly flags the
  # WeakKeyDictionary() instantiation below as abstract-class-instantiated.
  # pylint: disable-next=abstract-class-instantiated
  sessions: MutableMapping[object, str] = weakref.WeakKeyDictionary()

  async def call_agent(
      request: str, ctx: Context
  ) -> list[mcp_types.ContentBlock]:
    return await _run_agent(agent_runner, request, ctx, sessions)

  server.add_tool(
      call_agent,
      name=tool_name,
      description=agent.description or f"Run the {tool_name} agent.",
      structured_output=False,
  )
  return server

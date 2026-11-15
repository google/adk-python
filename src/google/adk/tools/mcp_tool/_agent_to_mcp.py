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

import asyncio
import base64
import logging
from typing import Any
from typing import MutableMapping
from typing import Optional
import weakref

from google.genai import types

from ...agents.base_agent import BaseAgent
from ...artifacts.in_memory_artifact_service import InMemoryArtifactService
from ...auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ...dependencies._mcp import Context
from ...dependencies._mcp import FastMCP
from ...dependencies._mcp import ServerSession
from ...dependencies._mcp import types as mcp_types
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.in_memory_session_service import InMemorySessionService

logger = logging.getLogger("google_adk." + __name__)

_MCP_USER_ID = "mcp_user"
_INLINE_RESOURCE_URI = "resource://adk-agent/inline-data"
_MAX_REAP_ATTEMPTS = 3


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
    or None for a part with no renderable content (e.g. a function call) and
    for the model's thoughts, which are not part of its answer.
  """
  if part.thought:
    return None
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


def _tool_result(
    content: list[mcp_types.ContentBlock], session_id: str
) -> mcp_types.CallToolResult:
  """Builds a tool result carrying the ADK session id as structured content.

  The field is ``structuredContent`` in MCP SDK 1.x and ``structured_content``
  in 2.x, which renamed it and kept the old spelling as an alias. Validation
  accepts the 1.x spelling under both majors; a keyword argument does not, so
  the payload goes through ``model_validate``.

  Args:
    content: The agent's response content blocks.
    session_id: The ADK session the response was produced in.

  Returns:
    The tool result the host receives.
  """
  return mcp_types.CallToolResult.model_validate({
      "content": content,
      "structuredContent": {"session_id": session_id},
  })


def _connection_key(ctx: Context[ServerSession, Any]) -> object:
  """Returns the object that identifies the MCP connection behind ``ctx``.

  The MCP SDK exposes no public per-connection handle. In SDK 1.x
  ``ctx.session`` is itself one object per connection. In 2.x the server
  builds a fresh ``ServerSession`` for every inbound message and keeps the
  connection on its private ``_connection``, so ``ctx.session`` changes on
  every call. Reading ``_connection`` when it is there gives one key per
  connection on both versions.

  TODO: Use a public accessor once the SDK adds one. SDK 2.x already defines
  ``mcp.server.context.Context.connection``, but the server does not hand that
  class to tool functions yet.

  Args:
    ctx: The MCP tool call context.

  Returns:
    The per-connection object, or the per-request session when the SDK gives
    no connection. That fallback degrades to one agent session per request.
    It must never fall back to an object shared by all connections, because
    separate clients would then share one conversation.
  """
  session = ctx.session
  return getattr(session, "_connection", session)


async def _reap_orphaned_sessions(
    runner: Runner,
    sessions: MutableMapping[object, str],
    created: set[str],
    reap_failures: Optional[MutableMapping[str, int]] = None,
) -> None:
  """Deletes ADK sessions whose MCP connection is gone.

  ``sessions`` holds its connections weakly, so an entry vanishes when its
  connection is garbage-collected; the ADK session it pointed to would stay
  in the session service forever. Under a stateless streamable HTTP transport
  the connection lives for a single request, which turns that into one leaked
  session per tool call. Reaping runs lazily from the next tool call because
  a GC callback may fire without a running event loop.

  Args:
    runner: The Runner whose session service owns the sessions.
    sessions: Per-connection map from MCP connection to ADK session id.
    created: Ids of every session ever entered into ``sessions``. Ids no
      longer reachable through ``sessions`` are deleted and removed from it.
    reap_failures: Optional map tracking failure counts for failed session
      deletions, bounding retries to at most `_MAX_REAP_ATTEMPTS`.
  """
  live = set(sessions.values())
  if not isinstance(runner.session_service, InMemorySessionService):
    return
  for session_id in created - live:
    if session_id not in created:
      # A concurrent reap already took this one; the discard below and this
      # check share one synchronous stretch, so each id is deleted once.
      continue
    created.discard(session_id)
    try:
      await runner.session_service.delete_session(
          app_name=runner.app_name,
          user_id=_MCP_USER_ID,
          session_id=session_id,
      )
      if reap_failures is not None:
        reap_failures.pop(session_id, None)
    except Exception:  # pylint: disable=broad-exception-caught
      # Reaping is housekeeping; it must not fail the tool call that
      # triggered it. Put the id back so a later call retries the delete,
      # up to _MAX_REAP_ATTEMPTS times.
      attempts = (
          reap_failures.get(session_id, 0) + 1
          if reap_failures is not None
          else 1
      )
      if reap_failures is None or attempts < _MAX_REAP_ATTEMPTS:
        if reap_failures is not None:
          reap_failures[session_id] = attempts
        created.add(session_id)
        logger.warning(
            "Failed to delete orphaned MCP agent session %s; will retry on a"
            " later tool call.",
            session_id,
            exc_info=True,
        )
      else:
        reap_failures.pop(session_id, None)
        logger.warning(
            "Failed to delete orphaned MCP agent session %s after %d attempts;"
            " dropping.",
            session_id,
            _MAX_REAP_ATTEMPTS,
            exc_info=True,
        )
    except BaseException:
      created.add(session_id)
      raise


async def _run_agent(
    runner: Runner,
    request: str,
    ctx: Optional[Context[ServerSession, Any]] = None,
    sessions: Optional[MutableMapping[object, str]] = None,
    created: Optional[set[str]] = None,
    locks: Optional[MutableMapping[object, asyncio.Lock]] = None,
    session_id: Optional[str] = None,
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
    created: Set recording the id of every session entered into ``sessions``,
      so `_reap_orphaned_sessions` can delete the ones whose connection dies.
    locks: Per-connection locks preventing concurrent session creation on the
      same connection.
    session_id: An existing ADK session to run in. Takes precedence over the
      per-connection lookup, so a caller that names its conversation keeps it
      regardless of which connection the call arrived on. The caller is
      responsible for creating the session first.

  Returns:
    The agent's final response as a list of MCP content blocks (text plus any
    images, audio, or other data the agent produced).
  """
  resolved_session_id: Optional[str] = session_id
  connection: Optional[object] = None
  if resolved_session_id is None and ctx is not None and sessions is not None:
    connection = _connection_key(ctx)
    resolved_session_id = sessions.get(connection)
  if resolved_session_id is None:
    if locks is not None and connection is not None and sessions is not None:
      lock = locks.get(connection)
      if lock is None:
        lock = asyncio.Lock()
        locks[connection] = lock
      async with lock:
        resolved_session_id = sessions.get(connection)
        if resolved_session_id is None:
          session = await runner.session_service.create_session(
              app_name=runner.app_name, user_id=_MCP_USER_ID
          )
          resolved_session_id = session.id
          # No await between the two writes: an id is either absent from both or
          # present in both, so the reaper never sees a session it cannot delete.
          sessions[connection] = resolved_session_id
          if created is not None:
            created.add(resolved_session_id)
    else:
      session = await runner.session_service.create_session(
          app_name=runner.app_name, user_id=_MCP_USER_ID
      )
      resolved_session_id = session.id
      if sessions is not None and connection is not None:
        # No await between the two writes: an id is either absent from both or
        # present in both, so the reaper never sees a session it cannot delete.
        sessions[connection] = resolved_session_id
        if created is not None:
          created.add(resolved_session_id)
  new_message = types.Content(role="user", parts=[types.Part(text=request)])
  final_content: list[mcp_types.ContentBlock] = []
  async for event in runner.run_async(
      user_id=_MCP_USER_ID,
      session_id=resolved_session_id,
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
      text = "".join(
          part.text or "" for part in event.content.parts if not part.thought
      )
      if text:
        await ctx.report_progress(progress=0.0, message=text)
  return final_content


def to_mcp_server(
    agent: BaseAgent,
    *,
    name: Optional[str] = None,
    instructions: Optional[str] = None,
    runner: Optional[Runner] = None,
    session_id_arg: bool = False,
) -> FastMCP:
  """Exposes an ADK agent as an MCP server.

  The returned server registers a single MCP tool that runs the agent: an MCP
  host (e.g. Claude Code, OpenAI Codex, an IDE, or any MCP client) sends a
  request string and receives the agent's final response, including any images
  or audio the agent produced. This is the MCP counterpart of ``to_a2a``; it
  lets harnesses that speak MCP drive an ADK agent.

  One ADK session is kept per MCP connection, so successive tool calls on the
  same connection form a single multi-turn conversation. When using an
  in-memory session service (such as the default), orphaned ADK sessions whose
  connection has closed are deleted on subsequent tool calls so a long-running
  server does not accumulate dead conversations. Caller-supplied persistent
  session services retain their sessions.

  The caller chooses the transport, e.g. ``server.run(transport="stdio")`` for
  a local host or ``server.run(transport="streamable-http")`` for a networked
  one. A stateless streamable HTTP deployment (``stateless_http=True``, e.g.
  behind an autoscaler) gets a fresh connection per request, so every tool
  call is its own single-turn conversation whose session is likewise
  reclaimed when using an in-memory session service.

  Some deployments cannot rely on the connection to identify the conversation:
  a stateless transport, several server instances behind a load balancer, or an
  MCP protocol revision that neither initializes a session nor sends an
  ``Mcp-Session-Id`` header. For those, pass ``session_id_arg=True``. The tool
  then takes an optional ``session_id`` argument and returns the id of the
  session it ran in as structured content, so the host can carry the
  conversation itself by passing the id back on the next call. The id is the
  only handle on the conversation: any caller that knows it joins it, so only
  enable this for hosts you trust to hold it.

  Args:
    agent: The ADK agent to serve.
    name: The MCP server and tool name. Defaults to the agent's name.
    instructions: Optional instructions the MCP host may show to its model.
    runner: A pre-built Runner. If omitted, one is created with in-memory
      services.
    session_id_arg: Whether the tool exposes a ``session_id`` argument and
      returns the session id instead of keying the conversation on the MCP
      connection. Off by default, which keeps the per-connection behavior.

  Returns:
    A ``FastMCP`` server exposing the agent as a single tool.

  Example::

      agent = LlmAgent(name="assistant", model="gemini-3.5-flash", ...)
      server = to_mcp_server(agent)
      server.run(transport="stdio")
  """
  tool_name = name or agent.name or "adk_agent"
  server = FastMCP(name=tool_name, instructions=instructions)
  agent_runner = runner if runner is not None else _build_runner(agent)
  tool_description = agent.description or f"Run the {tool_name} agent."
  if session_id_arg:
    tool_description = (
        f"{tool_description} Pass back the returned session_id on every later"
        " call of the same conversation. Omit it to start a new one."
    )

    # The host owns the conversation lifetime, so there is no per-connection
    # map and no orphan reaper: a session lives exactly as long as the host
    # keeps handing its id back.
    async def call_agent_with_session_id(
        request: str,
        ctx: Context[ServerSession, Any],
        session_id: Optional[str] = None,
    ) -> mcp_types.CallToolResult:
      """Runs the agent for one turn of a conversation.

      Args:
        request: The user request text for this turn.
        ctx: The MCP tool call context, used for progress notifications.
        session_id: The id an earlier call returned, to continue that
          conversation. Omit it to start a new one.

      Returns:
        The agent's response, plus the id of the session it ran in.
      """
      session = None
      if session_id:
        session = await agent_runner.session_service.get_session(
            app_name=agent_runner.app_name,
            user_id=_MCP_USER_ID,
            session_id=session_id,
        )
      if session is None:
        session = await agent_runner.session_service.create_session(
            app_name=agent_runner.app_name, user_id=_MCP_USER_ID
        )
      content = await _run_agent(
          agent_runner, request, ctx, session_id=session.id
      )
      return _tool_result(content, session.id)

    server.add_tool(
        call_agent_with_session_id,
        name=tool_name,
        description=tool_description,
        structured_output=False,
    )
    return server

  reap_sessions = isinstance(
      agent_runner.session_service, InMemorySessionService
  )
  # Maps each MCP connection to its ADK session; WeakKeyDictionary drops the
  # entry when the connection is garbage-collected. pylint wrongly flags the
  # WeakKeyDictionary() instantiation below as abstract-class-instantiated.
  # pylint: disable-next=abstract-class-instantiated
  sessions: MutableMapping[object, str] = weakref.WeakKeyDictionary()
  # Ids of every session in `sessions`, kept strongly so the sessions of
  # collected connections can still be found and deleted.
  created_session_ids: set[str] = set()
  # Failure count per session id to bound retries on service outages.
  reap_failures: dict[str, int] = {}
  # Per-connection lock ensuring concurrent first calls on one connection do
  # not both create and overwrite an ADK session.
  # pylint: disable-next=abstract-class-instantiated
  locks: MutableMapping[object, asyncio.Lock] = weakref.WeakKeyDictionary()

  async def call_agent(
      request: str, ctx: Context[ServerSession, Any]
  ) -> list[mcp_types.ContentBlock]:
    if reap_sessions:
      await _reap_orphaned_sessions(
          agent_runner, sessions, created_session_ids, reap_failures
      )
    return await _run_agent(
        agent_runner,
        request,
        ctx,
        sessions,
        created_session_ids if reap_sessions else None,
        locks,
    )

  server.add_tool(
      call_agent,
      name=tool_name,
      description=tool_description,
      structured_output=False,
  )
  return server

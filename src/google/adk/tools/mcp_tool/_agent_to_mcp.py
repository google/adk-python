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
import contextvars
import logging
from typing import Any
from typing import Iterator
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
from ...features import experimental
from ...features import FeatureName
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.in_memory_session_service import InMemorySessionService

logger = logging.getLogger("google_adk." + __name__)

_MCP_USER_ID = "mcp_user"
_INLINE_RESOURCE_URI = "resource://adk-agent/inline-data"
# Caps the deletes one reap has in flight, so a large backlog does not hit a
# database-backed session service all at once. Arbitrary, not measured.
_MAX_CONCURRENT_DELETES = 8
# Same bound runners.py gives toolset cleanup. Arbitrary, not measured.
_DELETE_TIMEOUT_SECONDS = 10.0


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
) -> None:
  """Deletes ADK sessions whose MCP connection is gone.

  ``sessions`` holds its connections weakly, so an entry vanishes when its
  connection is garbage-collected; the ADK session it pointed to would stay
  in the session service forever. Under a stateless streamable HTTP transport
  the connection lives for a single request, which turns that into one leaked
  session per tool call. Reaping is started from tool calls rather than from
  a GC callback, because a GC callback may fire without a running event loop.

  The reap repeats until no orphan is left, including connections that close
  while it runs, so a backlog drains without waiting for further tool calls.
  Each batch is deleted by ``_MAX_CONCURRENT_DELETES`` workers sharing one
  iterator: concurrently, because deleting one at a time caps throughput at
  one session per delete round trip, which a slow session service cannot
  sustain under frequent short connections; and through a fixed pool rather
  than a task per session, because after an outage the backlog can hold
  many thousands of ids. Once as many deletes have failed as there are
  workers, the workers stop deleting and the pass ends, so an unavailable
  service sees fewer than twice the pool size of attempts per pass, while a
  single session that keeps failing does not hold up the others. Every id
  the pass did not delete is put back, and a later tool call retries it.

  Args:
    runner: The Runner whose session service owns the sessions.
    sessions: Per-connection map from MCP connection to ADK session id.
    created: Ids of every session ever entered into ``sessions``. Ids no
      longer reachable through ``sessions`` are deleted and removed from it.
  """

  async def delete_pending(
      pending: Iterator[str], errors: list[Exception], max_errors: int
  ) -> None:
    for session_id in pending:
      if len(errors) >= max_errors:
        created.add(session_id)
        continue
      try:
        # Only one reap runs at a time, so a delete that never returns would
        # stop all reaping; time it out and retry it on a later call instead.
        await asyncio.wait_for(
            runner.session_service.delete_session(
                app_name=runner.app_name,
                user_id=_MCP_USER_ID,
                session_id=session_id,
            ),
            timeout=_DELETE_TIMEOUT_SECONDS,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Reaping is housekeeping; it must not fail the tool call that
        # triggered it. Put the id back so a later call retries it.
        created.add(session_id)
        errors.append(e)

  while True:
    batch = list(created - set(sessions.values()))
    if not batch:
      return
    # Claimed in the same synchronous stretch that computed it, so a
    # concurrent reap never deletes the same id twice.
    created.difference_update(batch)
    pending = iter(batch)
    errors: list[Exception] = []
    workers = min(_MAX_CONCURRENT_DELETES, len(batch))
    # Stopping only once there have been as many failures as workers tells an
    # outage apart from a single session that keeps failing, which must not
    # hold up the rest of the batch.
    await asyncio.gather(
        *(delete_pending(pending, errors, workers) for _ in range(workers))
    )
    if errors:
      logger.warning(
          "Failed to delete %d orphaned MCP agent session(s); will retry on a"
          " later tool call.",
          len(errors),
          exc_info=errors[0],
      )
      return


def _log_reap_failure(task: asyncio.Task[None]) -> None:
  """Logs an unexpected error from a background reap.

  Retrieving the exception here also keeps asyncio from reporting it later as
  "Task exception was never retrieved".

  Args:
    task: The finished reap task.
  """
  if task.cancelled():
    return
  exc = task.exception()
  if exc is not None:
    logger.warning(
        "Background reap of orphaned MCP agent sessions failed.",
        exc_info=exc,
    )


def _start_background_reap(
    runner: Runner,
    sessions: MutableMapping[object, str],
    created: set[str],
) -> asyncio.Task[None]:
  """Starts `_reap_orphaned_sessions` as a task with failure logging.

  Callers run this inside an empty ``contextvars.Context``. Both the task and
  its done callback capture the current context, and a copy of the calling
  tool request's context would keep that request, and through it the MCP
  connection, alive for as long as the task exists (MCP SDK 1.x stores the
  request in a ContextVar). A stuck reap would then pin a connection whose
  session it should delete. Running in an empty context also works on Python
  3.10, which lacks ``create_task(context=...)``.

  Args:
    runner: The Runner whose session service owns the sessions.
    sessions: Per-connection map from MCP connection to ADK session id.
    created: Ids of every session entered into ``sessions``.

  Returns:
    The started reap task.
  """
  task = asyncio.create_task(_reap_orphaned_sessions(runner, sessions, created))
  task.add_done_callback(_log_reap_failure)
  return task


async def _run_agent(
    runner: Runner,
    request: str,
    ctx: Optional[Context[ServerSession, Any]] = None,
    sessions: Optional[MutableMapping[object, str]] = None,
    created: Optional[set[str]] = None,
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

  Returns:
    The agent's final response as a list of MCP content blocks (text plus any
    images, audio, or other data the agent produced).
  """
  session_id: Optional[str] = None
  connection: Optional[object] = None
  if ctx is not None and sessions is not None:
    connection = _connection_key(ctx)
    session_id = sessions.get(connection)
  if session_id is None:
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=_MCP_USER_ID
    )
    session_id = session.id
    if sessions is not None and connection is not None:
      # No await between the two writes: an id is either absent from both or
      # present in both, so the reaper never sees a session it cannot delete.
      sessions[connection] = session_id
      if created is not None:
        created.add(session_id)
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
    delete_orphaned_sessions: bool = True,
) -> FastMCP:
  """Exposes an ADK agent as an MCP server.

  The returned server registers a single MCP tool that runs the agent: an MCP
  host (e.g. Claude Code, OpenAI Codex, an IDE, or any MCP client) sends a
  request string and receives the agent's final response, including any images
  or audio the agent produced. This is the MCP counterpart of ``to_a2a``; it
  lets harnesses that speak MCP drive an ADK agent.

  One ADK session is kept per MCP connection, so successive tool calls on the
  same connection form a single multi-turn conversation. When a connection
  goes away its ADK session is deleted from the session service in the
  background, by a reap that tool calls start, so a long-running server does
  not accumulate dead conversations.

  The caller chooses the transport, e.g. ``server.run(transport="stdio")`` for
  a local host or ``server.run(transport="streamable-http")`` for a networked
  one. A stateless streamable HTTP deployment (``stateless_http=True``, e.g.
  behind an autoscaler) gets a fresh connection per request, so every tool
  call is its own single-turn conversation whose session is likewise
  reclaimed.

  Args:
    agent: The ADK agent to serve.
    name: The MCP server and tool name. Defaults to the agent's name.
    instructions: Optional instructions the MCP host may show to its model.
    runner: A pre-built Runner. If omitted, one is created with in-memory
      services.
    delete_orphaned_sessions: Whether to delete a connection's ADK session
      from the session service once the connection is gone. Defaults to True,
      which keeps a long-running server's memory bounded. Set to False to
      retain finished conversations in the session service, e.g. when a
      caller-supplied ``runner`` uses a persistent session service whose
      records are read after the fact; the caller then owns their cleanup.

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
  # Ids of every session in `sessions`, kept strongly so the sessions of
  # collected connections can still be found and deleted. None disables the
  # tracking and with it the reaping.
  created_session_ids: Optional[set[str]] = (
      set() if delete_orphaned_sessions else None
  )

  # The in-flight reap, if any. Held here so the task is not garbage-collected
  # mid-run, and so at most one reap runs at a time.
  reap_task: Optional[asyncio.Task[None]] = None

  async def call_agent(
      request: str, ctx: Context[ServerSession, Any]
  ) -> list[mcp_types.ContentBlock]:
    nonlocal reap_task
    # A task from another event loop may never finish from this loop's point
    # of view (e.g. that loop stopped without cancelling it), so it must not
    # block reaping here.
    if created_session_ids is not None and (
        reap_task is None
        or reap_task.done()
        or reap_task.get_loop() is not asyncio.get_running_loop()
    ):
      # Reap in the background so a slow session service (e.g. a database)
      # never adds delete latency to the live call.
      reap_task = contextvars.Context().run(
          _start_background_reap,
          agent_runner,
          sessions,
          created_session_ids,
      )
    return await _run_agent(
        agent_runner, request, ctx, sessions, created_session_ids
    )

  server.add_tool(
      call_agent,
      name=tool_name,
      description=agent.description or f"Run the {tool_name} agent.",
      structured_output=False,
  )
  return server

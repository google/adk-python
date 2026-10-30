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
import contextlib
import hashlib
import logging
import warnings
import weakref

from google.genai import types

from ...features import experimental
from ...features import FeatureName
from ...runners import Runner

try:
  from aiohttp import web
  from microsoft_agents.activity import Activity
  from microsoft_agents.hosting.aiohttp import CloudAdapter
  from microsoft_agents.hosting.aiohttp import jwt_authorization_middleware
  from microsoft_agents.hosting.aiohttp import start_agent_process
  from microsoft_agents.hosting.core import AgentApplication
  from microsoft_agents.hosting.core import AgentAuthConfiguration
  from microsoft_agents.hosting.core import ClaimsIdentity
  from microsoft_agents.hosting.core import TurnContext
  from microsoft_agents.hosting.core import TurnState
  from microsoft_agents.hosting.core import TypingIndicator
except ImportError as e:
  raise ImportError(
      "The Microsoft 365 Agents SDK is not installed. Please install it with "
      '`pip install "google-adk[teams]"`.'
  ) from e

logger = logging.getLogger("google_adk." + __name__)

_MESSAGES_PATH = "/api/messages"
_ERROR_REPLY = "Sorry, something went wrong while handling your message."


def _session_id_for(conversation_id: str) -> str:
  # Teams conversation ids contain characters such as ':' and ';' that some
  # session services reject, so derive a stable id that any service accepts.
  digest = hashlib.sha256(conversation_id.encode()).hexdigest()[:32]
  return f"teams-{digest}"


@experimental(FeatureName.TEAMS)
class TeamsRunner:
  """Runner for ADK agents on Microsoft Teams.

  Registers a message handler on a Microsoft 365 Agents SDK
  `AgentApplication` that forwards each message to an ADK `Runner` and posts
  the agent's text responses back to the same conversation. Each Teams
  conversation maps to one ADK session per user.

  Teams retries any request that takes longer than 15 seconds, so each message
  is acknowledged immediately and the agent runs in the background, replying
  through a proactive turn. Messages in the same session run one at a time.
  """

  def __init__(
      self,
      *,
      runner: Runner,
      agent_app: AgentApplication[TurnState],
  ):
    """Initializes the runner and registers its Teams message handler.

    Args:
      runner: The ADK runner that runs the agent for each message.
      agent_app: The Microsoft 365 Agents SDK app that receives Teams
        activities. Must be created with an aiohttp `CloudAdapter` to use
        `create_app` or `start`.
    """
    self.runner = runner
    self.agent_app = agent_app
    self._background_tasks: set[asyncio.Task[None]] = set()
    self._session_locks: weakref.WeakValueDictionary[
        tuple[str, str], asyncio.Lock
    ] = weakref.WeakValueDictionary()
    self.agent_app.activity("message")(self._handle_message)

  async def _handle_message(
      self, context: TurnContext, state: TurnState
  ) -> None:
    """Acknowledges a Teams message and starts the agent run."""
    del state  # Unused; conversation history lives in the ADK session.
    activity = context.activity
    # AgentApplication strips the bot's @mention from the text by default.
    text = (activity.text or "").strip()
    conversation_id = (
        activity.conversation.id if activity.conversation else None
    )
    sender = activity.from_property
    # The Entra object id is stable across chats; the channel account id is
    # the fallback for senders without one.
    user_id = (sender.aad_object_id or sender.id) if sender else None

    if not text or not conversation_id or not user_id or not context.identity:
      return

    continuation = (
        activity.get_conversation_reference().get_continuation_activity()
    )
    # Reply to the user's message, as an in-turn reply would, rather than to
    # the continuation's generated id.
    continuation.id = activity.id
    task = asyncio.create_task(
        self._reply_in_background(
            identity=context.identity,
            continuation=continuation,
            user_id=user_id,
            session_id=_session_id_for(conversation_id),
            text=text,
        )
    )
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def _reply_in_background(
      self,
      *,
      identity: ClaimsIdentity,
      continuation: Activity,
      user_id: str,
      session_id: str,
      text: str,
  ) -> None:
    """Runs the agent in a proactive turn that reuses the caller's identity."""

    async def run_turn(context: TurnContext) -> None:
      async with TypingIndicator(context):
        await self._run_agent(context, user_id, session_id, text)

    lock = self._session_locks.setdefault((user_id, session_id), asyncio.Lock())
    async with lock:
      try:
        await self.agent_app.adapter.continue_conversation_with_claims(
            identity, continuation, run_turn
        )
      except Exception:
        logger.exception("Error replying to Teams conversation:")

  async def _run_agent(
      self, context: TurnContext, user_id: str, session_id: str, text: str
  ) -> None:
    new_message = types.Content(role="user", parts=[types.Part(text=text)])
    try:
      async with contextlib.aclosing(
          self.runner.run_async(
              user_id=user_id,
              session_id=session_id,
              new_message=new_message,
          )
      ) as events:
        async for event in events:
          if event.partial:
            continue
          if event.error_code or event.error_message:
            logger.warning(
                "ADK agent run for Teams ended with %s: %s",
                event.error_code,
                event.error_message,
            )
            if not event.content or not event.content.parts:
              await context.send_activity(_ERROR_REPLY)
              continue
          if not event.content or not event.content.parts:
            continue
          for part in event.content.parts:
            if part.text and not part.thought:
              await context.send_activity(part.text)
    except Exception:
      logger.exception("Error running ADK agent for Teams:")
      # The exception text can contain internal details, so the user only
      # gets a generic reply.
      await context.send_activity(_ERROR_REPLY)

  def create_app(
      self, auth_configuration: AgentAuthConfiguration
  ) -> web.Application:
    """Creates an aiohttp app that serves the Teams messaging endpoint.

    Incoming requests are authenticated with the Microsoft 365 Agents SDK JWT
    middleware. Use this to mount the endpoint in your own server; otherwise
    call `start`.

    Args:
      auth_configuration: Authentication settings for the bot's Entra app
        registration, used to validate tokens on incoming requests.

    Returns:
      An aiohttp application that handles POST requests to /api/messages.

    Raises:
      TypeError: If `agent_app` was not created with an aiohttp
        `CloudAdapter`.
    """
    adapter = self.agent_app.adapter
    if not isinstance(adapter, CloudAdapter):
      raise TypeError(
          "TeamsRunner requires an AgentApplication created with"
          " microsoft_agents.hosting.aiohttp.CloudAdapter."
      )

    async def handle_messages(request: web.Request) -> web.StreamResponse:
      response = await start_agent_process(request, self.agent_app, adapter)
      return response or web.Response(status=200)

    async def drain_background_tasks(_: web.Application) -> None:
      if self._background_tasks:
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    app = web.Application(middlewares=[jwt_authorization_middleware])
    app.router.add_post(_MESSAGES_PATH, handle_messages)
    # Lets replies already in progress finish when the server shuts down.
    app.on_shutdown.append(drain_background_tasks)
    # The SDK's JWT middleware looks this up by its plain string key.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", web.NotAppKeyWarning)
      app["agent_configuration"] = auth_configuration
    return app

  async def start(
      self,
      auth_configuration: AgentAuthConfiguration,
      *,
      host: str = "localhost",
      port: int = 3978,
  ) -> None:
    """Serves the Teams messaging endpoint until cancelled.

    Args:
      auth_configuration: Authentication settings for the bot's Entra app
        registration, used to validate tokens on incoming requests.
      host: Interface to listen on.
      port: Port to listen on.
    """
    app_runner = web.AppRunner(self.create_app(auth_configuration))
    await app_runner.setup()
    try:
      await web.TCPSite(app_runner, host, port).start()
      logger.info(
          "Teams endpoint listening on http://%s:%d%s",
          host,
          port,
          _MESSAGES_PATH,
      )
      await asyncio.Event().wait()
    finally:
      await app_runner.cleanup()

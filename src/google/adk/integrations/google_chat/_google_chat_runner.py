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
import dataclasses
import hashlib
import json
import logging
from typing import Any
import weakref

import google.auth
from google.auth import credentials as auth_credentials
from google.auth.transport import requests as auth_requests
from google.genai import types
from google.oauth2 import id_token
import httpx

from ...features import experimental
from ...features import FeatureName
from ...runners import Runner
from ...utils._mtls_utils import client_cert_httpx_kwargs
from ...utils._mtls_utils import get_api_endpoint

try:
  from aiohttp import web
except ImportError as e:
  raise ImportError(
      "aiohttp is not installed. Please install it with "
      '`pip install "google-adk[google-chat]"`.'
  ) from e

logger = logging.getLogger("google_adk." + __name__)

_EVENTS_PATH = "/"
_CHAT_SCOPE = "https://www.googleapis.com/auth/chat.bot"
_CHAT_ISSUER = "chat@system.gserviceaccount.com"
_CERTS_URL = "https://www.googleapis.com/service_accounts/v1/metadata/x509/"
_DEFAULT_API_HOST = "chat.googleapis.com"
_DEFAULT_MTLS_API_HOST = "chat.mtls.googleapis.com"
_ERROR_REPLY = "Sorry, something went wrong while handling your message."


@dataclasses.dataclass(frozen=True)
class _IncomingMessage:
  text: str
  user_id: str
  space_name: str
  # Set when the reply belongs in a thread, which is also the session scope.
  thread_name: str | None


def _session_id_for(key: str) -> str:
  # Chat resource names contain '/', which some session services reject, so
  # derive a stable id that any service accepts.
  digest = hashlib.sha256(key.encode()).hexdigest()[:32]
  return f"google-chat-{digest}"


def _is_threaded(space: dict[str, Any]) -> bool:
  """Returns whether conversations in the space are organized by thread."""
  # Direct messages with conversation sessions turned on keep each session in
  # its own thread.
  if space.get("sessionState") == "THREAD":
    return True
  threading_state = space.get("spaceThreadingState")
  if threading_state:
    return bool(threading_state == "THREADED_MESSAGES")
  # Older events only carry the deprecated space type.
  return bool(space.get("spaceType") == "SPACE" or space.get("type") == "ROOM")


def _parse_message_event(event: dict[str, Any]) -> _IncomingMessage | None:
  """Extracts a user message from a Chat app or Workspace add-on event."""
  if isinstance(event.get("chat"), dict):
    # Chat apps built as Google Workspace add-ons.
    chat = event["chat"]
    payload = chat.get("messagePayload")
    if not isinstance(payload, dict):
      return None
    message = payload.get("message") or {}
    space = payload.get("space") or message.get("space") or {}
    sender = chat.get("user") or message.get("sender") or {}
  elif event.get("type") == "MESSAGE":
    message = event.get("message") or {}
    space = event.get("space") or message.get("space") or {}
    sender = event.get("user") or message.get("sender") or {}
  else:
    return None

  if sender.get("type") == "BOT":
    return None
  # argumentText is the message without the app's @mention.
  text = message.get("argumentText", message.get("text")) or ""
  text = text.strip()
  user_id = sender.get("name")
  space_name = space.get("name")
  if not text or not user_id or not space_name:
    return None

  thread_name = None
  if _is_threaded(space):
    thread_name = (message.get("thread") or {}).get("name")
  return _IncomingMessage(
      text=text,
      user_id=user_id,
      space_name=space_name,
      thread_name=thread_name,
  )


@experimental(FeatureName.GOOGLE_CHAT)
class GoogleChatRunner:
  """Runner for ADK agents on Google Chat.

  Serves the HTTP endpoint of a Google Chat app, forwards each message to an
  ADK `Runner`, and posts the agent's text responses back with the Chat API.
  In spaces organized by thread, each thread maps to one ADK session per user;
  in direct messages and group conversations, the whole conversation does.

  Chat expects a response within 30 seconds, so each event is acknowledged
  immediately and the agent runs in the background. Messages in the same
  session run one at a time.
  """

  def __init__(
      self,
      *,
      runner: Runner,
      credentials: auth_credentials.Credentials | None = None,
      api_endpoint: str | None = None,
  ):
    """Initializes the runner.

    Args:
      runner: The ADK runner that runs the agent for each message.
      credentials: Credentials of the Chat app's service account, used to post
        replies with the `chat.bot` scope. Defaults to Application Default
        Credentials.
      api_endpoint: Base URL of the Google Chat API. Defaults to
        `https://chat.googleapis.com`, or its mTLS variant when a client
        certificate is configured.
    """
    self.runner = runner
    self._credentials = credentials
    if api_endpoint is None:
      host = get_api_endpoint(
          location="",
          default_template=_DEFAULT_API_HOST,
          mtls_template=_DEFAULT_MTLS_API_HOST,
      )
      api_endpoint = f"https://{host}"
    self._api_endpoint = api_endpoint.rstrip("/")
    self._credentials_lock = asyncio.Lock()
    self._http: httpx.AsyncClient | None = None
    self._background_tasks: set[asyncio.Task[None]] = set()
    self._session_locks: weakref.WeakValueDictionary[
        tuple[str, str], asyncio.Lock
    ] = weakref.WeakValueDictionary()

  def _schedule_reply(self, message: _IncomingMessage) -> None:
    session_key = message.thread_name or message.space_name
    task = asyncio.create_task(
        self._reply_in_background(message, _session_id_for(session_key))
    )
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def _reply_in_background(
      self, message: _IncomingMessage, session_id: str
  ) -> None:
    lock = self._session_locks.setdefault(
        (message.user_id, session_id), asyncio.Lock()
    )
    async with lock:
      try:
        await self._run_agent(message, session_id)
      except Exception:
        logger.exception("Error replying to Google Chat message:")

  async def _run_agent(
      self, message: _IncomingMessage, session_id: str
  ) -> None:
    new_message = types.Content(
        role="user", parts=[types.Part(text=message.text)]
    )
    try:
      async with contextlib.aclosing(
          self.runner.run_async(
              user_id=message.user_id,
              session_id=session_id,
              new_message=new_message,
          )
      ) as events:
        async for event in events:
          if event.partial:
            continue
          if event.error_code or event.error_message:
            logger.warning(
                "ADK agent run for Google Chat ended with %s: %s",
                event.error_code,
                event.error_message,
            )
            if not event.content or not event.content.parts:
              await self._post_message(message, _ERROR_REPLY)
              continue
          if not event.content or not event.content.parts:
            continue
          for part in event.content.parts:
            if part.text and not part.thought:
              await self._post_message(message, part.text)
    except Exception:
      logger.exception("Error running ADK agent for Google Chat:")
      # The exception text can contain internal details, so the user only
      # gets a generic reply.
      await self._post_message(message, _ERROR_REPLY)

  async def _access_token(self) -> str:
    async with self._credentials_lock:
      if self._credentials is None:
        self._credentials, _ = await asyncio.to_thread(
            google.auth.default, scopes=[_CHAT_SCOPE]
        )
      if not self._credentials.valid:
        await asyncio.to_thread(
            self._credentials.refresh, auth_requests.Request()
        )
      token: str = self._credentials.token
      return token

  async def _post_message(self, message: _IncomingMessage, text: str) -> None:
    """Posts a message as the Chat app, in the thread when there is one."""
    body: dict[str, Any] = {"text": text}
    params: dict[str, str] = {}
    if message.thread_name:
      body["thread"] = {"name": message.thread_name}
      params["messageReplyOption"] = "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
    if self._http is None:
      raise RuntimeError("GoogleChatRunner is not serving requests.")
    token = await self._access_token()
    response = await self._http.post(
        f"{self._api_endpoint}/v1/{message.space_name}/messages",
        params=params,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.is_error:
      raise RuntimeError(
          f"Chat API returned {response.status_code} when posting a message:"
          f" {response.text}"
      )

  def create_app(
      self,
      *,
      audience: str | None,
      service_account_email: str = _CHAT_ISSUER,
  ) -> web.Application:
    """Creates an aiohttp app that serves the Chat app's HTTP endpoint.

    Each request must carry a bearer token signed by Google Chat. Use this to
    mount the endpoint in your own server; otherwise call `start`.

    Args:
      audience: The Authentication Audience configured for the Chat app: the
        HTTP endpoint URL, or the Google Cloud project number. `None` turns off
        request verification, which is only safe for local testing.
      service_account_email: The account that signs requests. Chat apps use
        `chat@system.gserviceaccount.com`; Chat apps built as Google Workspace
        add-ons use their per-project
        `service-PROJECT_NUMBER@gcp-sa-gsuiteaddons.iam.gserviceaccount.com`.

    Returns:
      An aiohttp application that handles POST requests to /.
    """
    if audience is None:
      logger.warning(
          "Google Chat request verification is off; anyone who can reach the"
          " endpoint can send messages to the agent."
      )
    token_request = auth_requests.Request()

    def verify_token(token: str, expected_audience: str) -> bool:
      if expected_audience.isdigit():
        # The Project Number audience uses a JWT self-signed by Chat.
        claims = id_token.verify_token(
            token,
            token_request,
            audience=expected_audience,
            certs_url=_CERTS_URL + service_account_email,
        )
        return claims.get("iss") == service_account_email
      claims = id_token.verify_oauth2_token(
          token, token_request, expected_audience
      )
      return bool(
          claims.get("email") == service_account_email
          and claims.get("email_verified")
      )

    async def is_authorized(request: web.Request) -> bool:
      if audience is None:
        return True
      scheme, _, token = request.headers.get("Authorization", "").partition(" ")
      if scheme.lower() != "bearer" or not token:
        return False
      try:
        return await asyncio.to_thread(verify_token, token, audience)
      except Exception:
        logger.warning("Rejected a Google Chat request with an invalid token.")
        return False

    async def handle_event(request: web.Request) -> web.StreamResponse:
      if not await is_authorized(request):
        return web.Response(status=401)
      try:
        event = await request.json()
      except json.JSONDecodeError:
        return web.Response(status=400)
      message = _parse_message_event(event) if isinstance(event, dict) else None
      if message:
        self._schedule_reply(message)
      # An empty response acknowledges the event without a synchronous reply.
      return web.json_response({})

    async def open_http_session(_: web.Application) -> None:
      # Presents the default client certificate to googleapis.com hosts when
      # one is configured.
      self._http = httpx.AsyncClient(
          **client_cert_httpx_kwargs(url=self._api_endpoint)
      )

    async def drain_background_tasks(_: web.Application) -> None:
      if self._background_tasks:
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def close_http_session(_: web.Application) -> None:
      if self._http is not None:
        await self._http.aclose()
        self._http = None

    app = web.Application()
    app.router.add_post(_EVENTS_PATH, handle_event)
    app.on_startup.append(open_http_session)
    # Lets replies already in progress finish when the server shuts down.
    app.on_shutdown.append(drain_background_tasks)
    app.on_cleanup.append(close_http_session)
    return app

  async def start(
      self,
      *,
      audience: str | None,
      service_account_email: str = _CHAT_ISSUER,
      host: str = "localhost",
      port: int = 8080,
  ) -> None:
    """Serves the Chat app's HTTP endpoint until cancelled.

    Args:
      audience: The Authentication Audience configured for the Chat app: the
        HTTP endpoint URL, or the Google Cloud project number. `None` turns off
        request verification, which is only safe for local testing.
      service_account_email: The account that signs requests. See
        `create_app`.
      host: Interface to listen on.
      port: Port to listen on.
    """
    app_runner = web.AppRunner(
        self.create_app(
            audience=audience, service_account_email=service_account_email
        )
    )
    await app_runner.setup()
    try:
      await web.TCPSite(app_runner, host, port).start()
      logger.info(
          "Google Chat endpoint listening on http://%s:%d%s",
          host,
          port,
          _EVENTS_PATH,
      )
      await asyncio.Event().wait()
    finally:
      await app_runner.cleanup()

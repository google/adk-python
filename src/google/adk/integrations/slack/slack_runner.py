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

import logging
from typing import Any

from google.adk.runners import Runner
from google.genai import types

try:
  from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
  from slack_bolt.app.async_app import AsyncApp
except ImportError as e:
  raise ImportError(
      "slack_bolt is not installed. Please install it with "
      '`pip install "google-adk[slack]"`.'
  ) from e

logger = logging.getLogger("google_adk." + __name__)


class SlackRunner:
  """Runner for ADK agents on Slack."""

  def __init__(
      self,
      runner: Runner,
      slack_app: AsyncApp,
  ):
    self.runner = runner
    self.slack_app = slack_app
    self._setup_handlers()

  def _setup_handlers(self):
    """Sets up event handlers for Slack."""

    @self.slack_app.event("app_mention")
    async def handle_app_mentions(event, say):
      await self._handle_message(event, say)

    @self.slack_app.event("message")
    async def handle_message_events(event, say):
      # Skip bot messages to avoid loops
      if event.get("bot_id") or event.get("bot_profile"):
        return

      is_im = event.get("channel_type") == "im"
      in_thread = event.get("thread_ts") is not None

      if is_im or in_thread:
        await self._handle_message(event, say)

  async def _handle_message(self, event: dict[str, Any], say: Any):
    """Handles a message or app_mention event."""
    text = event.get("text", "")
    user_id = event.get("user")
    channel_id = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    if not text or not user_id or not channel_id:
      return

    # In Slack, we can use the channel_id (and optionally thread_ts) as a session ID.
    session_id = f"{channel_id}-{thread_ts}" if thread_ts else channel_id

    new_message = types.Content(role="user", parts=[types.Part(text=text)])

    thinking_ts: str | None = None
    try:
      thinking_response = await say(text="_Thinking..._", thread_ts=thread_ts)
      thinking_ts = thinking_response.get("ts")

      async for event in self.runner.run_async(
          user_id=user_id,
          session_id=session_id,
          new_message=new_message,
      ):
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.text:
              if thinking_ts:
                await self.slack_app.client.chat_update(
                    channel=channel_id,
                    ts=thinking_ts,
                    text=part.text,
                )
                thinking_ts = None
              else:
                await say(text=part.text, thread_ts=thread_ts)
      if thinking_ts:
        await self.slack_app.client.chat_delete(
            channel=channel_id, ts=thinking_ts
        )
        thinking_ts = None
    except Exception as e:
      error_message = f"Sorry, I encountered an error: {str(e)}"
      logger.exception("Error running ADK agent for Slack:")
      if thinking_ts:
        await self.slack_app.client.chat_update(
            channel=channel_id,
            ts=thinking_ts,
            text=error_message,
        )
      else:
        await say(text=error_message, thread_ts=thread_ts)

  async def start(self, app_token: str):
    """Starts the Slack app using Socket Mode."""
    handler = AsyncSocketModeHandler(self.slack_app, app_token)
    await handler.start_async()

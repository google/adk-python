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

from collections.abc import AsyncIterator
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

pytest.importorskip("slack_bolt")

from google.adk.events.event import Event
from google.adk.integrations.slack import slack_runner as slack_runner_module
from google.adk.integrations.slack import SlackRunner
from google.adk.runners import Runner
from google.genai import types
from slack_bolt.app.async_app import AsyncApp


def _make_thread_event(
    ts: str | None, channel: str = "C67890"
) -> dict[str, str]:
  event = {
      "text": "<@UBOT> status?",
      "user": "U12345",
      "channel": channel,
      "thread_ts": "1234567890.123456",
      "channel_type": "channel",
  }
  if ts is not None:
    event["ts"] = ts
  return event


async def _empty_run_async(
    *args: object, **kwargs: object
) -> AsyncIterator[Event]:
  return
  yield


class TestSlackRunner(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.mock_runner = MagicMock(spec=Runner)
    self.mock_slack_app = MagicMock(spec=AsyncApp)
    self.mock_slack_app.client = MagicMock()
    self.mock_slack_app.client.chat_update = AsyncMock()
    self.mock_slack_app.client.chat_delete = AsyncMock()
    self.slack_runner = SlackRunner(self.mock_runner, self.mock_slack_app)

  @patch("google.adk.integrations.slack.slack_runner.logger")
  async def test_handle_message_success(self, mock_logger):
    # Setup mocks
    mock_say = AsyncMock()
    mock_say.return_value = {"ts": "thinking_ts"}
    event = {
        "text": "Hello bot",
        "user": "U12345",
        "channel": "C67890",
        "ts": "1234567890.123456",
    }

    # Mock runner.run_async to yield a response
    mock_event = MagicMock()
    mock_event.content = types.Content(
        role="model", parts=[types.Part(text="Hi user!")]
    )

    async def mock_run_async(*args, **kwargs):
      yield mock_event

    self.mock_runner.run_async.side_effect = mock_run_async

    # Call the handler
    await self.slack_runner._handle_message(event, mock_say)

    # Verify calls
    self.mock_runner.run_async.assert_called_once()
    mock_say.assert_called_once_with(
        text="_Thinking..._", thread_ts="1234567890.123456"
    )
    self.mock_slack_app.client.chat_update.assert_called_once_with(
        channel="C67890",
        ts="thinking_ts",
        text="Hi user!",
    )

  @patch("google.adk.integrations.slack.slack_runner.logger")
  async def test_handle_message_sanitizes_slack_thread_ts_session_id(
      self, mock_logger
  ):
    mock_say = AsyncMock()
    mock_say.return_value = {"ts": "thinking_ts"}
    event = {
        "text": "Hello bot",
        "user": "U12345",
        "channel": "C67890",
        "thread_ts": "1234567890.123456",
        "ts": "1234567899.999999",
    }

    async def mock_run_async(*args, **kwargs):
      return
      yield

    self.mock_runner.run_async.side_effect = mock_run_async

    await self.slack_runner._handle_message(event, mock_say)

    self.mock_runner.run_async.assert_called_once_with(
        user_id="U12345",
        session_id="C67890-1234567890_123456",
        new_message=types.Content(
            role="user", parts=[types.Part(text="Hello bot")]
        ),
    )
    mock_say.assert_called_once_with(
        text="_Thinking..._", thread_ts="1234567890.123456"
    )

  @patch("google.adk.integrations.slack.slack_runner.logger")
  async def test_handle_message_multi_turn(self, mock_logger):
    # Setup mocks
    mock_say = AsyncMock()
    mock_say.return_value = {"ts": "thinking_ts"}
    event = {
        "text": "Tell me two things",
        "user": "U12345",
        "channel": "C67890",
        "ts": "1234567890.123456",
    }

    # Mock runner.run_async to yield two responses
    e1 = MagicMock()
    e1.content = types.Content(
        role="model", parts=[types.Part(text="First thing.")]
    )
    e2 = MagicMock()
    e2.content = types.Content(
        role="model", parts=[types.Part(text="Second thing.")]
    )

    async def mock_run_async(*args, **kwargs):
      yield e1
      yield e2

    self.mock_runner.run_async.side_effect = mock_run_async

    await self.slack_runner._handle_message(event, mock_say)

    # First message uses chat_update
    self.mock_slack_app.client.chat_update.assert_called_once_with(
        channel="C67890",
        ts="thinking_ts",
        text="First thing.",
    )
    # Second message uses say
    self.assertEqual(mock_say.call_count, 2)
    mock_say.assert_any_call(
        text="_Thinking..._", thread_ts="1234567890.123456"
    )
    mock_say.assert_any_call(
        text="Second thing.", thread_ts="1234567890.123456"
    )

  @patch("google.adk.integrations.slack.slack_runner.logger")
  async def test_handle_message_error(self, mock_logger):
    mock_say = AsyncMock()
    mock_say.return_value = {"ts": "thinking_ts"}
    event = {
        "text": "Trigger error",
        "user": "U12345",
        "channel": "C67890",
        "ts": "1234567890.123456",
    }

    async def mock_run_async_error(*args, **kwargs):
      raise Exception("Something went wrong")
      yield  # To make it a generator

    self.mock_runner.run_async.side_effect = mock_run_async_error

    await self.slack_runner._handle_message(event, mock_say)

    mock_say.assert_called_once_with(
        text="_Thinking..._", thread_ts="1234567890.123456"
    )
    self.mock_slack_app.client.chat_update.assert_called_once()
    self.assertIn(
        "Sorry, I encountered an error",
        self.mock_slack_app.client.chat_update.call_args[1]["text"],
    )

  async def test_event_delivered_as_app_mention_and_message_runs_once(
      self,
  ) -> None:
    for order in (("app_mention", "message"), ("message", "app_mention")):
      with self.subTest(order=order):
        slack_app = MagicMock(spec=AsyncApp)
        runner = MagicMock(spec=Runner)
        runner.run_async.side_effect = _empty_run_async
        SlackRunner(runner, slack_app)
        names = [c.args[0] for c in slack_app.event.call_args_list]
        handlers = [
            c.args[0] for c in slack_app.event.return_value.call_args_list
        ]
        handler_by_name = dict(zip(names, handlers))
        say = AsyncMock(return_value={})

        for name in order:
          await handler_by_name[name](
              _make_thread_event("1234567899.000001"), say
          )

        runner.run_async.assert_called_once()

  async def test_processed_events_expire_by_ttl_and_capacity(self) -> None:
    self.mock_runner.run_async.side_effect = _empty_run_async
    say = AsyncMock(return_value={})
    mock_time = MagicMock()
    mock_time.monotonic.return_value = 1000.0

    async def handle(ts: str | None, channel: str = "C67890") -> int:
      await self.slack_runner._handle_message(
          _make_thread_event(ts, channel), say
      )
      return int(self.mock_runner.run_async.call_count)

    with (
        patch.object(slack_runner_module, "_MAX_PROCESSED_EVENTS", 2),
        patch.object(slack_runner_module, "_PROCESSED_EVENT_TTL_SECONDS", 60.0),
        patch.object(slack_runner_module, "time", mock_time),
    ):
      self.assertEqual(await handle("100.1"), 1)
      self.assertEqual(await handle("100.1", channel="C11111"), 2)
      mock_time.monotonic.return_value = 1050.0
      self.assertEqual(await handle("100.1"), 2)
      mock_time.monotonic.return_value = 1061.0
      self.assertEqual(await handle("100.1"), 3)
      self.assertEqual(await handle("100.2"), 4)
      self.assertEqual(await handle("100.3"), 5)
      self.assertEqual(await handle("100.3"), 5)
      self.assertEqual(await handle("100.1"), 6)
      self.assertEqual(await handle(None), 7)
      self.assertEqual(await handle(None), 8)


if __name__ == "__main__":
  unittest.main()

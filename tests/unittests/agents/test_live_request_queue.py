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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_close_queue():
  queue = LiveRequestQueue()

  with patch.object(queue._queue, "put_nowait") as mock_put_nowait:
    queue.close()
    mock_put_nowait.assert_called_once_with(LiveRequest(close=True))


def test_send_content():
  queue = LiveRequestQueue()
  content = MagicMock(spec=types.Content)

  with patch.object(queue._queue, "put_nowait") as mock_put_nowait:
    queue.send_content(content)
    mock_put_nowait.assert_called_once_with(LiveRequest(content=content))


def test_send_realtime():
  queue = LiveRequestQueue()
  blob = MagicMock(spec=types.Blob)

  with patch.object(queue._queue, "put_nowait") as mock_put_nowait:
    queue.send_realtime(blob)
    mock_put_nowait.assert_called_once_with(LiveRequest(blob=blob))


def test_send():
  queue = LiveRequestQueue()
  req = LiveRequest(content=MagicMock(spec=types.Content))

  with patch.object(queue._queue, "put_nowait") as mock_put_nowait:
    queue.send(req)
    mock_put_nowait.assert_called_once_with(req)


@pytest.mark.asyncio
async def test_get():
  queue = LiveRequestQueue()
  res = MagicMock(spec=types.Content)

  with patch.object(queue._queue, "get", return_value=res) as mock_get:
    result = await queue.get()

    assert result == res
    mock_get.assert_called_once()

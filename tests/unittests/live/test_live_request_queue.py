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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.live import LiveRequest
from google.adk.live import LiveRequestQueue
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


def test_send_realtime_blob():
  queue = LiveRequestQueue()
  blob = types.Blob(data=b"test", mime_type="audio/pcm")

  with patch.object(queue._queue, "put_nowait") as mock_put_nowait:
    queue.send_realtime(blob)
    mock_put_nowait.assert_called_once_with(LiveRequest(blob=blob))


def test_closed_is_false_before_close():
  queue = LiveRequestQueue()

  assert queue.closed is False


def test_close_sets_closed_stickily():
  """The flag has to outlive the one-shot sentinel it accompanies."""
  queue = LiveRequestQueue()

  queue.close()

  assert queue.closed is True
  # Draining the sentinel does not clear it: reconnect logic asks afterwards.
  queue._queue.get_nowait()
  assert queue.closed is True


def test_send_with_close_request_sets_closed():
  """`send()` is the generic path into the queue and must record it too."""
  queue = LiveRequestQueue()

  queue.send(LiveRequest(close=True))

  assert queue.closed is True


def test_send_without_close_request_leaves_queue_open():
  queue = LiveRequestQueue()

  queue.send(LiveRequest(content=types.Content(parts=[])))

  assert queue.closed is False

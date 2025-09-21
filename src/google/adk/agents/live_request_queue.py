# Copyright 2025 Google LLC
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
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


class LiveRequest(BaseModel):
  """Request send to live agents."""

  model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """If set, send the content to the model in turn-by-turn mode."""
  blob: Optional[types.Blob] = None
  """If set, send the blob to the model in realtime mode."""
  activity_start: Optional[types.ActivityStart] = None
  """If set, signal the start of user activity to the model."""
  activity_end: Optional[types.ActivityEnd] = None
  """If set, signal the end of user activity to the model."""
  close: bool = False
  """If set, close the queue. queue.shutdown() is only supported in Python 3.13+."""


class LiveRequestQueue:
  """Queue used to send LiveRequest in a live(bidirectional streaming) way."""

  def __init__(self):
    # Ensure there's an event loop available in this thread
    try:
      asyncio.get_running_loop()
    except RuntimeError:
      # No running loop, create one
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

    # Now create the queue (it will use the event loop we just ensured exists)
    self._queue = asyncio.Queue()

  def close(self):
    self._queue.put_nowait(LiveRequest(close=True))

  def send_content(self, content: types.Content):
    self._queue.put_nowait(LiveRequest(content=content))

  def send_realtime(self, blob: types.Blob):
    self._queue.put_nowait(LiveRequest(blob=blob))

  def send_activity_start(self):
    """Sends an activity start signal to mark the beginning of user input."""
    self._queue.put_nowait(LiveRequest(activity_start=types.ActivityStart()))

  def send_activity_end(self):
    """Sends an activity end signal to mark the end of user input."""
    self._queue.put_nowait(LiveRequest(activity_end=types.ActivityEnd()))

  def send(self, req: LiveRequest):
    self._queue.put_nowait(req)

  async def get(self) -> LiveRequest:
    return await self._queue.get()

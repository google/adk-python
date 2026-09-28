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

"""Live request queue for bidirectional streaming agents."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict


class LiveRequest(BaseModel):
  """Request send to live agents.

  When multiple fields are set, they are processed by priority (highest first):
  activity_start > activity_end > audio_stream_end > blob > content.
  state_delta, if set, is always applied regardless of the other fields.
  """

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

  audio_stream_end: bool = False
  """If set, signal the end of the audio stream to the model. This is only used
  when Voice Activity Detection is enabled.
  """

  close: bool = False
  """If set, close the queue. queue.shutdown() is only supported in Python 3.13+."""

  partial: bool = False
  """If set, the content is a partial turn update that does not complete the current model turn."""

  state_delta: Optional[dict[str, Any]] = None
  """If set, these state changes are applied to the session, so they take
  effect even when the request carries no content or a partial/
  function-response turn."""


class LiveRequestQueue:
  """Queue used to send LiveRequest in a live(bidirectional streaming) way."""

  def __init__(self) -> None:
    self._queue: asyncio.Queue[LiveRequest] = asyncio.Queue()
    self._closed = False

  @property
  def closed(self) -> bool:
    """Whether the client has closed this queue.

    Sticky: the `close=True` sentinel is gone by the time reconnect logic asks.
    """
    return self._closed

  def close(self) -> None:
    self._closed = True
    self._queue.put_nowait(LiveRequest(close=True))

  def send_content(self, content: types.Content, partial: bool = False) -> None:
    self._queue.put_nowait(LiveRequest(content=content, partial=partial))

  def send_realtime(self, blob: types.Blob) -> None:
    self._queue.put_nowait(LiveRequest(blob=blob))

  def send_activity_start(self) -> None:
    """Sends an activity start signal to mark the beginning of user input."""
    self._queue.put_nowait(LiveRequest(activity_start=types.ActivityStart()))

  def send_activity_end(self) -> None:
    """Sends an activity end signal to mark the end of user input."""
    self._queue.put_nowait(LiveRequest(activity_end=types.ActivityEnd()))

  def send_audio_stream_end(self) -> None:
    """Sends an audio stream end signal to force flush audio."""
    self._queue.put_nowait(LiveRequest(audio_stream_end=True))

  def send(self, req: LiveRequest) -> None:
    if req.close:
      self._closed = True
    self._queue.put_nowait(req)

  async def get(self) -> LiveRequest:
    return await self._queue.get()

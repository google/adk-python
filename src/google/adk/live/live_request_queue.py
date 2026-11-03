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
from pydantic import Field
from pydantic import model_validator

from ..tools.base_tool import BaseTool


class LiveContextUpdate(BaseModel):
  """An in-session update to the model's system instruction and/or tools.

  Only the fields that are set are updated. The Live API processes the update
  in order with the other inputs on the connection.

  Both fields are part of the model preamble, so applying an update
  invalidates the Live prefix cache and the server re-prefills the system
  instruction, the tool declarations, and the session history on the next
  model call. Send one only on a real phase transition of the conversation,
  not on every turn. Turn-specific context such as a retrieved policy snippet
  belongs at the tail of the history instead, where it leaves the cached
  prefix intact: send it with `LiveRequestQueue.send_content(..., partial=True)`
  or return it as part of a tool's function response.

  An update applies to the current live invocation only. The flow carries it
  across reconnects of that invocation, but the next `run_live` call builds
  its request from the agent definition again.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

  system_instruction: Optional[str] = None
  """If set, replaces the session's system instruction.

  This is a full replacement: it overrides the instruction ADK assembled from
  the agent's `instruction`, `global_instruction`, and the request processors
  that append to it, so include everything the model still needs.
  """

  tools: Optional[list[BaseTool]] = Field(default=None, exclude=True)
  """If set, replaces the session's tools with these tools.

  The tools are registered with the invocation so that the model's function
  calls to them are executed like calls to the agent's own tools. Pass an
  empty list to clear all tools. Leave unset to keep the current tools.
  """

  @model_validator(mode='after')
  def _require_a_field(self) -> LiveContextUpdate:
    if self.system_instruction is None and self.tools is None:
      raise ValueError(
          'At least one of `system_instruction` or `tools` must be set.'
      )
    return self


class LiveRequest(BaseModel):
  """Request send to live agents.

  When multiple fields are set, they are processed by priority (highest first):
  activity_start > activity_end > audio_stream_end > blob > content.
  state_delta, if set, is always applied regardless of the other fields.
  context_update, if set, is applied before any of the fields above so that a
  content sent in the same request already sees the new instruction and tools.
  """

  model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """If set, send the content to the model in turn-by-turn mode."""

  blob: Optional[types.Blob] = None
  """If set, send the blob to the model in realtime mode."""

  activity_start: Optional[types.ActivityStart] = None
  """If set, signal the start of user activity to the model.

  Only used when server-side Voice Activity Detection is disabled, i.e. when
  the client performs its own (client-side) VAD.
  """

  activity_end: Optional[types.ActivityEnd] = None
  """If set, signal the end of user activity to the model.

  Only used when server-side Voice Activity Detection is disabled, i.e. when
  the client performs its own (client-side) VAD.
  """

  audio_stream_end: bool = False
  """If set, signal the end of the audio stream to the model. This is only used
  when Voice Activity Detection is enabled.

  For example, when the user mutes the microphone, send this signal so the
  server flushes any cached audio instead of waiting for more input.
  """

  close: bool = False
  """If set, close the queue. queue.shutdown() is only supported in Python 3.13+."""

  partial: bool = False
  """If set, the content is a partial turn update that does not complete the current model turn."""

  state_delta: Optional[dict[str, Any]] = None
  """If set, these state changes are applied to the session, so they take
  effect even when the request carries no content or a partial/
  function-response turn."""

  context_update: Optional[LiveContextUpdate] = None
  """If set, update the session's system instruction and/or tools in-stream.

  See `LiveContextUpdate` for the cache cost and when to use it.
  """


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
    """Sends an activity start signal to mark the beginning of user input.

    Only use this when server-side Voice Activity Detection is disabled, i.e.
    when the client performs its own (client-side) VAD.
    """
    self._queue.put_nowait(LiveRequest(activity_start=types.ActivityStart()))

  def send_activity_end(self) -> None:
    """Sends an activity end signal to mark the end of user input.

    Only use this when server-side Voice Activity Detection is disabled, i.e.
    when the client performs its own (client-side) VAD.
    """
    self._queue.put_nowait(LiveRequest(activity_end=types.ActivityEnd()))

  def send_audio_stream_end(self) -> None:
    """Sends an audio stream end signal to force flush audio.

    For example, when the user mutes the microphone or the audio stream pauses
    for more than a second, call this so the server flushes any cached audio
    instead of waiting for more input.
    """
    self._queue.put_nowait(LiveRequest(audio_stream_end=True))

  def send_context_update(
      self,
      *,
      system_instruction: Optional[str] = None,
      tools: Optional[list[BaseTool]] = None,
  ) -> None:
    """Updates the session's system instruction and/or tools in-stream.

    Both are part of the model preamble, so the update invalidates the Live
    prefix cache and the next model call re-prefills the instruction, the
    tool declarations, and the history. Use it for phase transitions of the
    conversation, not per turn; per-turn context goes to the tail of the
    history via `send_content(..., partial=True)` or a tool response, which
    keeps the cached prefix intact.

    Args:
      system_instruction: Replaces the whole system instruction, including
        what ADK assembled from the agent definition.
      tools: Replaces the session's tools. The model's calls to them are
        executed by ADK like calls to the agent's own tools. Pass `[]` to
        clear all tools; leave unset to keep the current ones.

    Raises:
      ValueError: If neither argument is set.
    """
    self._queue.put_nowait(
        LiveRequest(
            context_update=LiveContextUpdate(
                system_instruction=system_instruction, tools=tools
            )
        )
    )

  def send(self, req: LiveRequest) -> None:
    if req.close:
      self._closed = True
    self._queue.put_nowait(req)

  async def get(self) -> LiveRequest:
    return await self._queue.get()

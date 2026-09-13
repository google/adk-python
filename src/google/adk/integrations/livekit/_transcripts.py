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

"""Publishing ADK transcriptions on LiveKit's transcription topic."""

from __future__ import annotations

import logging
from typing import Optional
import uuid

from ._rtc import rtc

logger = logging.getLogger("google_adk." + __name__)

LK_TRANSCRIPTION_TOPIC = "lk.transcription"

_LK_TRANSCRIBED_TRACK_ID_ATTRIBUTE = "lk.transcribed_track_id"
_LK_SEGMENT_ID_ATTRIBUTE = "lk.segment_id"
_LK_TRANSCRIPTION_FINAL_ATTRIBUTE = "lk.transcription_final"


class TranscriptSegment:
  """One utterance being transcribed.

  A stable id shared by the interim stream and the final stream that replaces
  it, plus the track and identity of whoever spoke.
  """

  def __init__(
      self,
      *,
      track_sid: Optional[str] = None,
      identity: Optional[str] = None,
  ):
    self._track_sid = track_sid
    self._identity = identity
    self._segment_id = uuid.uuid4().hex
    self._writer: Optional[rtc.TextStreamWriter] = None

  def bind(self, *, track_sid: Optional[str], identity: Optional[str]) -> None:
    """Attaches the speaker's track and identity, once they are known."""
    if track_sid is not None:
      self._track_sid = track_sid
    if identity is not None:
      self._identity = identity

  def attributes(self, *, final: bool) -> dict[str, str]:
    """Returns the LiveKit stream attributes describing this segment."""
    attributes = {
        _LK_SEGMENT_ID_ATTRIBUTE: self._segment_id,
        _LK_TRANSCRIPTION_FINAL_ATTRIBUTE: "true" if final else "false",
    }
    if self._track_sid:
      attributes[_LK_TRANSCRIBED_TRACK_ID_ATTRIBUTE] = self._track_sid
    return attributes

  async def write(self, local: rtc.LocalParticipant, text: str) -> None:
    """Appends an incremental fragment to this segment's interim stream."""
    if self._writer is None:
      self._writer = await self._open(local, final=False)
    await self._writer.write(text)

  async def finish(self, local: rtc.LocalParticipant, text: str) -> None:
    """Replaces the interim stream with the completed utterance."""
    await self.close()
    writer = await self._open(local, final=True)
    await writer.write(text)
    await writer.aclose()
    self.reset()

  async def _open(
      self, local: rtc.LocalParticipant, *, final: bool
  ) -> rtc.TextStreamWriter:
    # Not `send_text`, which cannot set a sender identity, so the caller's own
    # words would be attributed to the agent.
    return await local.stream_text(
        topic=LK_TRANSCRIPTION_TOPIC,
        attributes=self.attributes(final=final),
        sender_identity=self._identity,
    )

  async def close(self) -> None:
    """Closes the interim stream, if one is open."""
    if self._writer is None:
      return
    writer, self._writer = self._writer, None
    await writer.aclose()

  def reset(self) -> None:
    """Starts a fresh segment for this speaker's next utterance."""
    self._writer = None
    self._segment_id = uuid.uuid4().hex

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

"""Room and runner doubles shared by the LiveKit connector tests."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.tool_context import ToolContext
import pytest

pytest.importorskip("livekit.rtc")

from google.adk.integrations.livekit import _livekit_runner
from google.adk.integrations.livekit import _transcripts as _transcripts_module
from google.adk.integrations.livekit import LiveKitCall
from google.adk.integrations.livekit import LiveKitRunner
from livekit import rtc


class FakeTextStreamWriter:
  """Records what a caller streams, standing in for LiveKit's writer.

  Deliberately mirrors `rtc.TextStreamWriter`'s real surface -- `aclose`, not
  `close`. A fake shaped to the implementation instead of to the SDK will
  happily green-light code that cannot work against a real room.
  """

  def __init__(self, topic, attributes, sender_identity):
    self.topic = topic
    self.attributes = attributes or {}
    self.sender_identity = sender_identity
    self.chunks: list[str] = []
    self.closed = False

  async def write(self, text: str) -> None:
    self.chunks.append(text)

  async def aclose(self) -> None:
    self.closed = True

  @property
  def text(self) -> str:
    return "".join(self.chunks)

  @property
  def is_final(self) -> bool:
    return self.attributes.get("lk.transcription_final") == "true"


def make_room(remote_participants: dict | None = None):
  """A connected LiveKit room with async publish methods and no tracks.

  `local_participant` is spec'd against the real `rtc.LocalParticipant`, so a
  call with a keyword the SDK does not accept fails here rather than on a real
  room.
  """
  room = MagicMock(spec=rtc.Room)
  room.name = "test-room"
  room.remote_participants = remote_participants or {}
  room.stream_writers: list[FakeTextStreamWriter] = []

  local = MagicMock(spec=rtc.LocalParticipant)
  local.identity = "adk-agent"
  local.publish_track = AsyncMock(
      return_value=MagicMock(spec=rtc.LocalTrackPublication, sid="TR_agent")
  )
  local.publish_data = AsyncMock()
  local.send_text = AsyncMock()
  local.set_attributes = AsyncMock()
  local.publish_dtmf = AsyncMock()

  async def stream_text(*, topic="", attributes=None, sender_identity=None):
    writer = FakeTextStreamWriter(topic, attributes, sender_identity)
    room.stream_writers.append(writer)
    return writer

  local.stream_text = stream_text
  room.local_participant = local
  return room


def sip_participant(attributes: dict[str, str], identity: str = "sip_caller"):
  """A telephony caller carrying the given `sip.*` attributes."""
  participant = MagicMock()
  participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
  participant.identity = identity
  participant.attributes = attributes
  participant.track_publications = {}
  return participant


def webrtc_participant(identity: str = "browser"):
  """A browser caller, which is not a phone call."""
  participant = MagicMock()
  participant.kind = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
  participant.identity = identity
  participant.attributes = {}
  participant.track_publications = {}
  return participant


def make_lk_runner(runner, room, **kwargs) -> LiveKitRunner:
  """Builds a runner with the outbound audio track stubbed out.

  `rtc.AudioSource` reaches into the LiveKit FFI, which needs a live worker.
  The replacement is spec'd against the real class so a call the SDK does not
  support fails here rather than on a real room.

  The wait for a caller is off unless a test asks for it, so a room built
  without participants starts at once instead of stalling.
  """
  kwargs.setdefault("wait_for_participant", None)
  with (
      patch.object(_livekit_runner.rtc, "AudioSource"),
      patch.object(_livekit_runner.rtc, "LocalAudioTrack"),
  ):
    lk_runner = LiveKitRunner(
        runner=runner, room=room, user_id="u1", session_id="s1", **kwargs
    )
  lk_runner._audio_source = MagicMock(spec=rtc.AudioSource)
  lk_runner._audio_source.capture_frame = AsyncMock()
  lk_runner._audio_source.wait_for_playout = AsyncMock()
  return lk_runner


def make_call(
    room=None,
    hang_up_callback=None,
    user_id: str = "u1",
    session_id: str = "s1",
) -> LiveKitCall:
  """A call handle over `room`, on the session `make_tool_context` names."""
  return LiveKitCall(
      room=room or make_room(),
      user_id=user_id,
      session_id=session_id,
      hang_up_callback=hang_up_callback or (lambda: None),
  )


def idle_runner() -> Runner:
  """A Runner whose `run_live` never finishes, like a real idle call."""
  runner = MagicMock(spec=Runner)
  runner.app_name = "test_app"
  runner.session_service = MagicMock()
  runner.session_service.get_session = AsyncMock(return_value=MagicMock())

  async def run_live(**kwargs):
    await asyncio.Event().wait()
    yield  # pragma: no cover - unreachable, keeps this an async generator

  runner.run_live = run_live
  return runner


@contextlib.asynccontextmanager
async def patched_livekit_api():
  """Patches LiveKit's server API and yields the client the code will use."""
  from livekit import api

  client = MagicMock()
  client.sip.transfer_sip_participant = AsyncMock()
  client.room.delete_room = AsyncMock()

  @contextlib.asynccontextmanager
  async def _session(*args, **kwargs):
    del args, kwargs
    yield client

  with patch.object(api, "LiveKitAPI", _session):
    yield client


def transfer_request(client):
  """The `TransferSIPParticipantRequest` sent through a patched API client."""
  return client.sip.transfer_sip_participant.await_args.args[0]


def make_tool_context(
    user_id: str = "u1", session_id: str = "s1"
) -> ToolContext:
  """The context ADK hands a tool, naming the session the call is on.

  Real rather than a mock, because the lookup reads `user_id` and `session.id`
  off it and a mock would answer those whatever the plumbing did.
  """
  return ToolContext(
      InvocationContext(
          session_service=InMemorySessionService(),
          invocation_id="test-invocation",
          agent=None,
          session=Session(id=session_id, app_name="test_app", user_id=user_id),
      )
  )


def _transcripts(room) -> list[FakeTextStreamWriter]:
  """Text streams on the transcription topic, excluding chat and the rest."""
  return [
      writer
      for writer in room.stream_writers
      if writer.topic == _transcripts_module.LK_TRANSCRIPTION_TOPIC
  ]


def final_transcripts(room) -> list[FakeTextStreamWriter]:
  """Every completed transcript published as a LiveKit text stream."""
  return [writer for writer in _transcripts(room) if writer.is_final]


def interim_transcripts(room) -> list[FakeTextStreamWriter]:
  """Every in-progress transcript stream."""
  return [writer for writer in _transcripts(room) if not writer.is_final]


def agent_states(room) -> list[str]:
  """Every agent state published, in order."""
  return [
      call.args[0][_livekit_runner._LK_AGENT_STATE_ATTRIBUTE]
      for call in room.local_participant.set_attributes.await_args_list
  ]

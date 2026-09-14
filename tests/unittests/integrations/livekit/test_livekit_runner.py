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

"""Tests for the LiveKit connector's two frame bridges.

Verifies that `LiveKitRunner` forwards inbound room media into the
`LiveRequestQueue` in the formats ADK's live contract expects, and pushes
outbound `run_live` events back to the room as audio, LiveKit-standard
transcription streams, agent state, and tool activity.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner
from google.genai import types
import pytest

pytest.importorskip("livekit.rtc")

from google.adk.integrations.livekit import _livekit_runner
from google.adk.integrations.livekit import _transcripts
from google.adk.integrations.livekit import LiveKitRunner
from livekit import rtc
import numpy as np

from tests.unittests.integrations.livekit.conftest import agent_states
from tests.unittests.integrations.livekit.conftest import final_transcripts
from tests.unittests.integrations.livekit.conftest import interim_transcripts
from tests.unittests.integrations.livekit.conftest import make_lk_runner
from tests.unittests.integrations.livekit.conftest import make_room
from tests.unittests.integrations.livekit.conftest import sip_participant
from tests.unittests.isolated_import_utils import run_isolated
from tests.unittests.testing_utils import MockModel

# --- Fixtures (minimal, one purpose each) ---


def _make_runner(events: list[Event]) -> Runner:
  """A Runner whose run_live yields the given events then finishes.

  The session already exists, so bridge tests exercise frame handling rather
  than session setup. What `run_live` was called with is recorded on
  `runner.live_kwargs`, so tests can assert on the config the connector
  actually hands the runner instead of on its own attributes.
  """
  runner = MagicMock(spec=Runner)
  runner.live_kwargs = {}

  async def run_live(**kwargs):
    runner.live_kwargs.update(kwargs)
    for event in events:
      yield event

  runner.run_live = run_live
  runner.app_name = "test_app"
  runner.session_service = MagicMock()
  runner.session_service.get_session = AsyncMock(return_value=MagicMock())
  runner.session_service.create_session = AsyncMock()
  return runner


def _never_ending_runner() -> Runner:
  """A Runner whose run_live never finishes, like a real idle call."""
  runner = _make_runner([])

  async def run_live(**kwargs):
    await asyncio.Event().wait()
    yield  # pragma: no cover - unreachable, keeps this an async generator

  runner.run_live = run_live
  return runner


def _audio_event(data: bytes) -> Event:
  return Event(
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  inline_data=types.Blob(mime_type="audio/pcm", data=data)
              )
          ],
      ),
  )


def _function_call_event(
    name: str, args: dict, *, call_id: str | None = None
) -> Event:
  return Event(
      author="agent",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name=name, args=args
                  )
              )
          ],
      ),
  )


def _function_response_event(
    name: str, response: dict, *, call_id: str | None = None
) -> Event:
  return Event(
      author="agent",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name=name, response=response
                  )
              )
          ],
      ),
  )


def _transcript_event(text: str, *, role: str, partial: bool) -> Event:
  transcription = types.Transcription(text=text, finished=not partial)
  return Event(
      author=role,
      partial=partial,
      input_transcription=transcription if role == "user" else None,
      output_transcription=transcription if role == "agent" else None,
  )


def _published(room) -> list[dict]:
  """Decodes every payload published on the ADK data topic."""
  return [
      json.loads(call.args[0])
      for call in room.local_participant.publish_data.await_args_list
  ]


class _FakeStream:
  """An async-iterable stand-in for rtc.AudioStream / rtc.VideoStream."""

  def __init__(self, events):
    self._events = events

  def __call__(self, *args, **kwargs):
    return self

  def __aiter__(self):
    async def gen():
      for event in self._events:
        yield event

    return gen()


def _rgba_frame(width: int = 16, height: int = 16) -> rtc.VideoFrame:
  return rtc.VideoFrame(
      width=width,
      height=height,
      type=rtc.VideoBufferType.RGBA,
      data=bytearray(np.zeros((height, width, 4), dtype=np.uint8).tobytes()),
  )


# --- Outbound bridge: Event stream -> room ---


async def test_the_default_run_config_survives_a_reconnect():
  """Voice calls outlive a model connection, so resumption is on by default.

  Without a resumption handle the reconnect that follows `go_away` replays
  only pre-call history, silently restarting the conversation mid-call.
  """
  runner = _make_runner([])
  lk_runner = make_lk_runner(runner, make_room())

  await lk_runner._forward_events()

  run_config = runner.live_kwargs["run_config"]
  assert run_config.response_modalities == [types.Modality.AUDIO]
  assert run_config.session_resumption is not None


async def test_the_default_run_config_produces_captions():
  """`lk.transcription` is silent unless transcription is asked for.

  `RunConfig` defaults both transcription fields to enabled, so the connector
  does nothing here. Asserted anyway: if that default ever flips back to None,
  captions stop and this is the test that says why.
  """
  runner = _make_runner([])
  lk_runner = make_lk_runner(runner, make_room())

  await lk_runner._forward_events()

  run_config = runner.live_kwargs["run_config"]
  assert run_config.input_audio_transcription is not None
  assert run_config.output_audio_transcription is not None


async def test_a_partial_run_config_keeps_its_own_settings():
  """A caller who sets one field must not lose the rest of their config."""
  transcription = types.AudioTranscriptionConfig()
  run_config = RunConfig(
      response_modalities=[types.Modality.AUDIO],
      output_audio_transcription=transcription,
      max_llm_calls=7,
  )
  runner = _make_runner([])
  lk_runner = make_lk_runner(runner, make_room(), run_config=run_config)

  await lk_runner._forward_events()

  used = runner.live_kwargs["run_config"]
  assert used.output_audio_transcription is transcription
  assert used.max_llm_calls == 7


async def test_a_partial_run_config_gains_the_voice_defaults():
  """Supplying a run_config for one reason must not mute the call.

  A config built to set `max_llm_calls` used to replace the defaults
  wholesale, dropping audio output and resumption without saying so.
  """
  runner = _make_runner([])
  lk_runner = make_lk_runner(
      runner, make_room(), run_config=RunConfig(max_llm_calls=7)
  )

  await lk_runner._forward_events()

  used = runner.live_kwargs["run_config"]
  assert used.response_modalities == [types.Modality.AUDIO]
  assert used.session_resumption is not None


async def test_the_callers_run_config_is_left_alone():
  """The caller's object may be a shared module-level default."""
  run_config = RunConfig(max_llm_calls=7)
  runner = _make_runner([])
  lk_runner = make_lk_runner(runner, make_room(), run_config=run_config)

  await lk_runner._forward_events()

  assert run_config.session_resumption is None
  assert run_config.response_modalities is None
  assert runner.live_kwargs["run_config"] is not run_config


async def test_transcription_can_still_be_switched_off():
  """`None` is the only way to say "no captions", so it has to survive.

  Transcription defaults to on, so filling a `None` back in would enable it
  for a caller who had just gone out of their way to disable it.
  """
  runner = _make_runner([])
  lk_runner = make_lk_runner(
      runner,
      make_room(),
      run_config=RunConfig(
          input_audio_transcription=None, output_audio_transcription=None
      ),
  )

  await lk_runner._forward_events()

  used = runner.live_kwargs["run_config"]
  assert used.input_audio_transcription is None
  assert used.output_audio_transcription is None
  # The fields it does fill are unaffected.
  assert used.response_modalities == [types.Modality.AUDIO]


async def test_output_audio_is_played_out_before_the_session_returns():
  """A session that ends on its own does not cut the agent off mid-word.

  Handing frames over is not the same as the caller having heard them: the
  audio source holds up to a second of speech after the last `capture_frame`
  returns, so returning early would clip the agent's last words.
  """
  events = [_audio_event(b"\x01\x02"), _audio_event(b"\x03\x04")]
  lk_runner = make_lk_runner(_make_runner(events), make_room())

  await lk_runner._forward_events()

  assert lk_runner._audio_source.capture_frame.await_count == 2
  lk_runner._audio_source.wait_for_playout.assert_awaited_once()


async def test_a_dead_playback_task_ends_the_call():
  """A broken audio source ends the call rather than going quietly mute.

  Nothing awaits the playback task while the call runs, so without this the
  caller would hear silence for the rest of the call with no error anywhere.
  """
  lk_runner = make_lk_runner(_never_ending_runner(), make_room())
  lk_runner._audio_source.capture_frame = AsyncMock(
      side_effect=RuntimeError("audio device gone")
  )

  session = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)
  lk_runner._playback.put_nowait(b"\x01\x02")

  await asyncio.wait_for(session, timeout=5)


async def test_event_pump_does_not_wait_on_audio_playback():
  """Playback pacing must not hold up transcripts, tools, or barge-in.

  `capture_frame` blocks once the audio buffer is full. If the pump drained
  playback inline it would run at realtime speed, and every later event --
  including the interruption that is supposed to stop playback -- would queue
  up behind the speech it is meant to cancel.
  """
  room = make_room()
  events = [_audio_event(b"\x01\x02"), _function_call_event("roll_die", {})]
  lk_runner = make_lk_runner(_make_runner(events), room)
  playing = asyncio.Event()
  release = asyncio.Event()

  async def _slow_playback(_frame):
    playing.set()
    await release.wait()

  lk_runner._audio_source.capture_frame = AsyncMock(side_effect=_slow_playback)

  session = asyncio.create_task(lk_runner._forward_events())
  await asyncio.wait_for(playing.wait(), timeout=5)
  await asyncio.sleep(0)

  # The tool call reached the room while playback was still blocked.
  assert [payload["type"] for payload in _published(room)] == ["function_call"]

  release.set()
  await asyncio.wait_for(session, timeout=5)


async def test_interrupted_event_drops_unplayed_speech():
  """Barge-in drops queued speech instead of talking over the user.

  Setup: two audio chunks queued behind a playback task blocked on the first,
    then an interruption.
  Act: run the pump to completion.
  Assert: the buffer is cleared and the second chunk is never captured.

  Clearing the source is also what releases a task suspended inside
  `capture_frame`, which is why playback is torn down rather than drained.
  """
  playing = asyncio.Event()
  captured: list[bytes] = []

  async def _blocking_playback(frame):
    captured.append(bytes(frame.data))
    playing.set()
    await asyncio.Event().wait()  # Never completes; the interrupt cancels it.

  runner = _make_runner([])

  async def run_live(**kwargs):
    yield _audio_event(b"\x01\x02")
    yield _audio_event(b"\x03\x04")
    # Interrupt only once the first chunk is genuinely on the wire, so the
    # second is provably still queued rather than merely un-scheduled.
    await playing.wait()
    yield Event(author="agent", interrupted=True)

  runner.run_live = run_live
  lk_runner = make_lk_runner(runner, make_room())
  lk_runner._audio_source.capture_frame = AsyncMock(
      side_effect=_blocking_playback
  )

  await lk_runner._forward_events()

  assert captured == [b"\x01\x02"]
  lk_runner._audio_source.clear_queue.assert_called_once()


async def test_tool_activity_is_published_on_the_adk_data_topic():
  """A function_call event reaches clients as JSON on the ADK topic."""
  room = make_room()
  lk_runner = make_lk_runner(
      _make_runner([_function_call_event("roll_die", {"sides": 6})]), room
  )

  await lk_runner._forward_events()

  (payload,) = _published(room)
  assert payload["type"] == "function_call"
  assert payload["name"] == "roll_die"
  assert payload["args"] == {"sides": 6}
  _, kwargs = room.local_participant.publish_data.await_args
  assert kwargs["topic"] == _livekit_runner.DATA_TOPIC


async def test_tool_payloads_carry_the_call_id_that_pairs_them():
  """A client has to know which call a result answers.

  Matching on the tool name is wrong as soon as one tool is called twice in a
  turn, which parallel calls make routine, so the ADK function call id goes
  on the wire and both halves carry it.
  """
  room = make_room()
  events = [
      _function_call_event("roll_die", {"sides": 20}, call_id="call-1"),
      _function_response_event("roll_die", {"result": 19}, call_id="call-1"),
  ]
  lk_runner = make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  call, response = _published(room)
  assert call["id"] == response["id"] == "call-1"
  assert call["type"] == "function_call"
  assert response["type"] == "function_response"
  assert response["response"] == {"result": 19}


# --- Outbound transcripts: LiveKit's standard channel ---


async def test_final_transcript_published_and_closed_on_livekit_topic():
  """Completed transcripts go where every LiveKit client already listens.

  An unterminated stream leaves the caption hanging open on every client, so
  the stream is closed as well as written.
  """
  room = make_room()
  lk_runner = make_lk_runner(
      _make_runner(
          [_transcript_event("you rolled a four", role="agent", partial=False)]
      ),
      room,
  )

  await lk_runner._forward_events()

  (final,) = final_transcripts(room)
  assert final.text == "you rolled a four"
  assert final.topic == _transcripts.LK_TRANSCRIPTION_TOPIC
  assert final.closed


async def test_the_final_transcript_replaces_the_interim_one():
  """A client replaces the caption it drew; it does not extend it.

  Setup: two partial fragments and then the completed utterance, which is
    exactly what ADK emits and what LiveKit's interim/final pair models.
  Act: run the pump to completion.
  Assert: both streams carry the whole utterance under one segment id, and
    the interim stream is closed by the final one.

  Send a remainder rather than the whole utterance here and every caption in
  every client renders the tail twice.
  """
  room = make_room()
  events = [
      _transcript_event("you ", role="agent", partial=True),
      _transcript_event("rolled a four", role="agent", partial=True),
      _transcript_event("you rolled a four", role="agent", partial=False),
  ]
  lk_runner = make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  (interim,) = interim_transcripts(room)
  (final,) = final_transcripts(room)
  assert interim.text == final.text == "you rolled a four"
  assert (
      interim.attributes["lk.segment_id"] == final.attributes["lk.segment_id"]
  )
  assert interim.closed


async def test_consecutive_utterances_get_their_own_segment_ids():
  """Two sentences must not be rendered as one endlessly-growing caption."""
  room = make_room()
  events = [
      _transcript_event("first", role="agent", partial=False),
      _transcript_event("second", role="agent", partial=False),
  ]
  lk_runner = make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  segment_ids = {
      writer.attributes["lk.segment_id"] for writer in final_transcripts(room)
  }
  assert len(segment_ids) == 2


async def test_agent_transcript_carries_the_agent_audio_track():
  """The transcribed track id is how a client tells the speakers apart."""
  room = make_room()
  lk_runner = make_lk_runner(
      _make_runner([_transcript_event("hello", role="agent", partial=False)]),
      room,
  )
  await lk_runner._publish_output_audio_track()

  await lk_runner._forward_events()

  assert (
      final_transcripts(room)[0].attributes["lk.transcribed_track_id"]
      == "TR_agent"
  )


@pytest.mark.parametrize("partial", [True, False])
async def test_caller_transcript_is_attributed_to_the_caller(partial):
  """The user's words must appear to come from the user, not the agent.

  Both sides are published by the agent's participant, so without an explicit
  sender identity a client renders the caller's own speech as the agent's.
  Interim and final are checked together because they take different code
  paths.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)
  with patch.object(_livekit_runner.rtc, "AudioStream", _FakeStream([])):
    await lk_runner._forward_audio(
        MagicMock(), track_sid="TR_caller", identity="caller-1"
    )

  await lk_runner._publish_transcript(
      role="user", text="roll a die", final=not partial
  )

  (writer,) = room.stream_writers
  assert writer.sender_identity == "caller-1"
  assert writer.attributes["lk.transcribed_track_id"] == "TR_caller"


async def test_a_failing_transcript_does_not_end_the_call():
  """Captions are cosmetic; a client that cannot render them is not fatal."""
  room = make_room()
  room.local_participant.stream_text = AsyncMock(
      side_effect=RuntimeError("nope")
  )
  lk_runner = make_lk_runner(
      _make_runner([_transcript_event("hello", role="agent", partial=False)]),
      room,
  )

  await lk_runner._forward_events()  # Must not raise.


# --- Outbound agent state ---


async def test_agent_state_starts_listening():
  """A connected agent tells clients it is ready before anyone speaks."""
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)

  await lk_runner._forward_events()

  assert agent_states(room)[0] == "listening"


async def test_agent_state_reports_speaking_while_audio_flows():
  """Voice UI needs to know when the agent has the floor."""
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([_audio_event(b"\x01\x02")]), room)

  await lk_runner._forward_events()

  assert "speaking" in agent_states(room)


async def test_agent_state_reports_thinking_while_a_tool_runs():
  """A tool call is dead air; clients show a thinking indicator instead."""
  room = make_room()
  lk_runner = make_lk_runner(
      _make_runner([_function_call_event("roll_die", {"sides": 6})]), room
  )

  await lk_runner._forward_events()

  assert "thinking" in agent_states(room)


async def test_agent_state_returns_to_listening_after_a_turn():
  """The floor goes back to the caller when the agent finishes."""
  room = make_room()
  events = [
      _audio_event(b"\x01\x02"),
      Event(author="agent", turn_complete=True),
  ]
  lk_runner = make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  assert agent_states(room)[-1] == "listening"


async def test_unchanged_agent_state_is_not_republished():
  """Every attribute update is a room message; repeats are pure noise."""
  room = make_room()
  events = [
      _audio_event(b"\x01\x02"),
      _audio_event(b"\x03\x04"),
      _audio_event(b"\x05\x06"),
  ]
  lk_runner = make_lk_runner(_make_runner(events), room)

  await lk_runner._forward_events()

  assert agent_states(room).count("speaking") == 1


# --- Inbound bridge: room -> LiveRequestQueue ---


async def test_inbound_audio_track_forwarded_as_pcm_blob():
  """Frames from a room audio track land on the queue as 16kHz PCM blobs."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())

  frame_event = MagicMock()
  frame_event.frame.data = b"\x10\x20"
  with patch.object(
      _livekit_runner.rtc, "AudioStream", _FakeStream([frame_event])
  ):
    await lk_runner._forward_audio(MagicMock())

  blob = (await lk_runner._queue.get()).blob
  # The rate belongs in the mime type; a bare `audio/pcm` leaves the model
  # guessing at the sample rate.
  assert blob.mime_type == "audio/pcm;rate=16000"
  assert blob.data == b"\x10\x20"


async def test_audio_stream_end_signalled_when_track_ends():
  """A muted or unpublished track flushes the model's audio buffer.

  Without the flush a server-VAD turn hangs waiting for input that will never
  arrive.
  """
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.send_audio_stream_end = MagicMock()

  with patch.object(_livekit_runner.rtc, "AudioStream", _FakeStream([])):
    await lk_runner._forward_audio(MagicMock())

  lk_runner._queue.send_audio_stream_end.assert_called_once()


async def test_inbound_video_track_forwarded_as_real_jpeg():
  """Video frames are JPEG-encoded, not raw buffers labelled image/jpeg."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  captured: list[types.Blob] = []
  lk_runner._queue.send_realtime = captured.append

  frame_event = MagicMock()
  frame_event.frame = _rgba_frame(64, 48)
  with patch.object(
      _livekit_runner.rtc, "VideoStream", _FakeStream([frame_event])
  ):
    await lk_runner._forward_video(MagicMock())

  assert len(captured) == 1
  assert captured[0].mime_type == "image/jpeg"
  assert captured[0].data.startswith(b"\xff\xd8")  # JPEG SOI marker.


async def test_video_frames_are_rate_limited():
  """Live models sample video; forwarding at capture rate floods the queue."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  captured: list[types.Blob] = []
  lk_runner._queue.send_realtime = captured.append

  frame = _rgba_frame()
  frame_events = []
  for _ in range(30):  # One second of 30fps capture.
    event = MagicMock()
    event.frame = frame
    frame_events.append(event)

  with patch.object(
      _livekit_runner.rtc, "VideoStream", _FakeStream(frame_events)
  ):
    await lk_runner._forward_video(MagicMock())

  assert len(captured) == 1


# --- Inbound text ---


def _data_packet(message: dict, topic: str | None = None):
  packet = MagicMock()
  packet.topic = topic or _livekit_runner.DATA_TOPIC
  packet.data = json.dumps(message).encode("utf-8")
  return packet


async def test_a_chat_message_becomes_a_user_turn():
  """Text typed in any LiveKit client reaches the agent with no ADK code.

  LiveKit hands the reader over synchronously, so the read is spawned as a
  task; that task has to be referenced or the garbage collector can cancel
  the message out from under the caller.
  """
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  reader = MagicMock()
  reader.read_all = AsyncMock(return_value="roll a die")

  lk_runner._on_chat_stream(reader, "caller-1")
  await asyncio.gather(*lk_runner._forward_tasks)

  request = await lk_runner._queue.get()
  assert request.content.role == "user"
  assert request.content.parts[0].text == "roll a die"


async def test_an_empty_chat_message_is_ignored():
  """An accidental empty send must not start a model turn."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.send_content = MagicMock()
  reader = MagicMock()
  reader.read_all = AsyncMock(return_value="   ")

  await lk_runner._read_chat_stream(reader)

  lk_runner._queue.send_content.assert_not_called()


async def test_chat_topic_is_livekits_own():
  """Registering elsewhere would mean every client had to be taught ADK."""
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)

  await lk_runner.start()

  topics = {
      call.args[0] for call in room.register_text_stream_handler.mock_calls
  }
  assert _livekit_runner.LK_CHAT_TOPIC in topics


async def test_inbound_text_on_the_adk_topic_still_works():
  """A client written against the ADK topic keeps working."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  captured: list[types.Content] = []
  lk_runner._queue.send_content = captured.append

  lk_runner._on_data_received(
      _data_packet({"type": "text", "text": "roll a die"})
  )

  assert captured[0].parts[0].text == "roll a die"


async def test_data_on_another_topic_is_ignored():
  """The bridge only claims its own topic; the room is shared."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.send_content = MagicMock()

  lk_runner._on_data_received(
      _data_packet({"type": "text", "text": "not for us"}, topic="other-app")
  )

  lk_runner._queue.send_content.assert_not_called()


async def test_malformed_data_message_is_ignored():
  """A non-JSON payload must not take down the session."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.send_content = MagicMock()

  packet = MagicMock()
  packet.topic = _livekit_runner.DATA_TOPIC
  packet.data = b"\xff\xfe not json"

  lk_runner._on_data_received(packet)

  lk_runner._queue.send_content.assert_not_called()


# --- Inbound DTMF ---


def _dtmf(digit: str, identity: str | None = "sip_caller"):
  packet = MagicMock()
  packet.digit = digit
  packet.participant = MagicMock(identity=identity)
  return packet


def _chat_messages(room) -> list:
  """Every message published on LiveKit's chat topic."""
  return [
      writer
      for writer in room.stream_writers
      if writer.topic == _livekit_runner.LK_CHAT_TOPIC
  ]


async def test_keypad_entry_reaches_the_agent_as_one_turn():
  """A caller keying an account number is one input, not six.

  Forwarding each keypress on its own would start a model turn per digit, so
  digits are held until the caller signals the end with `#`.
  """
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  captured: list[types.Content] = []
  lk_runner._queue.send_content = captured.append

  for digit in "4321#":
    lk_runner._on_dtmf_received(_dtmf(digit))

  assert len(captured) == 1
  assert "4321#" in captured[0].parts[0].text


async def test_keypad_entry_is_flushed_when_the_caller_stops_typing():
  """Not every IVR entry ends in `#`, so an idle pause ends it too."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  captured: list[types.Content] = []
  lk_runner._queue.send_content = captured.append

  with patch.object(_livekit_runner, "_DTMF_IDLE_FLUSH_SECONDS", 0.01):
    lk_runner._on_dtmf_received(_dtmf("7"))
    await asyncio.sleep(0.05)

  assert len(captured) == 1
  assert "7" in captured[0].parts[0].text


async def test_partial_keypad_entry_is_not_forwarded_early():
  """Digits still being typed must not wake the model mid-entry."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.send_content = MagicMock()

  lk_runner._on_dtmf_received(_dtmf("1"))
  lk_runner._on_dtmf_received(_dtmf("2"))

  lk_runner._queue.send_content.assert_not_called()


async def test_a_keypad_entry_is_published_to_the_room():
  """A keypress makes a turn nothing transcribes.

  LiveKit relays the tones but not the turn assembled from them, so without
  this a client shows the agent answering a question that appears nowhere.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)
  lk_runner._queue.send_content = MagicMock()

  for digit in "4321#":
    lk_runner._on_dtmf_received(_dtmf(digit))
  await asyncio.gather(*lk_runner._forward_tasks)

  (published,) = _chat_messages(room)
  assert published.text == "4321#"
  assert published.closed


async def test_the_keypad_turn_is_attributed_to_the_caller():
  """Otherwise the caller's own entry reads as something the agent said."""
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)
  lk_runner._queue.send_content = MagicMock()

  lk_runner._on_dtmf_received(_dtmf("#", identity="sip_caller"))
  await asyncio.gather(*lk_runner._forward_tasks)

  assert _chat_messages(room)[0].sender_identity == "sip_caller"


async def test_a_keypress_does_not_truncate_speech_being_transcribed():
  """A caller can press a key while still talking.

  Publishing onto their open transcription segment would close it, cutting
  the caption short and stranding the rest of the sentence in a dead stream.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)
  lk_runner._queue.send_content = MagicMock()
  await lk_runner._publish_transcript(
      role=_livekit_runner._USER_ROLE, text="my account is ", final=False
  )

  lk_runner._on_dtmf_received(_dtmf("#"))
  await asyncio.gather(*lk_runner._forward_tasks)

  (interim,) = interim_transcripts(room)
  assert not interim.closed  # Still open for the rest of the sentence.
  assert _chat_messages(room)[0] is not interim


async def test_a_failing_keypad_publish_does_not_end_the_call():
  """Publishing the turn is cosmetic; the model still has to get the digits."""
  room = make_room()
  room.local_participant.stream_text = AsyncMock(
      side_effect=RuntimeError("nope")
  )
  lk_runner = make_lk_runner(_make_runner([]), room)
  lk_runner._queue.send_content = MagicMock()

  lk_runner._on_dtmf_received(_dtmf("#"))
  await asyncio.gather(*lk_runner._forward_tasks)  # Must not raise.

  lk_runner._queue.send_content.assert_called_once()


# --- Waiting for the caller ---


async def _registered_handler(room, event: str, tries: int = 50):
  """Yields to the loop until `room.on(event, ...)` has been called."""
  for _ in range(tries):
    for call in room.on.call_args_list:
      if call.args[0] == event:
        return call.args[1]
    await asyncio.sleep(0)
  raise AssertionError(f"{event} was never subscribed to")


async def test_the_session_waits_for_a_caller_before_it_starts():
  """What a call can offer the model is read off the room.

  A live connection fixes its tool declarations when it opens, so an agent
  that starts before the caller arrives offers the wrong set for the whole
  call -- on an outbound call, never the telephony tools.
  """
  room = make_room()  # Nobody has joined yet.
  lk_runner = make_lk_runner(_make_runner([]), room, wait_for_participant=5)

  session = asyncio.create_task(lk_runner.start())
  on_connected = await _registered_handler(room, "participant_connected")
  assert not session.done()

  room.remote_participants = {"browser": MagicMock()}
  on_connected(room.remote_participants["browser"])

  await asyncio.wait_for(session, timeout=1)


async def test_a_caller_already_in_the_room_is_not_waited_for():
  """Inbound dispatch hands over a room that already holds the caller."""
  room = make_room({"sip_caller": MagicMock()})
  lk_runner = make_lk_runner(_make_runner([]), room, wait_for_participant=5)

  await asyncio.wait_for(lk_runner.start(), timeout=1)


async def test_the_session_starts_anyway_when_nobody_joins():
  """An agent alone in a room is odd, not an error.

  Failing here would turn a caller who hung up during the ring into a crashed
  worker.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room, wait_for_participant=0.01)

  await asyncio.wait_for(lk_runner.start(), timeout=1)


async def test_the_wait_can_be_turned_off():
  """Callers who join their own room know there is nobody to wait for."""
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room, wait_for_participant=None)

  await asyncio.wait_for(lk_runner.start(), timeout=1)


# --- Caller identity ---


async def test_phone_call_seeds_the_caller_number_into_session_state():
  """An agent should know who is calling before they say a word."""
  participant = sip_participant({
      "sip.phoneNumber": "+15105550100",
      "sip.trunkPhoneNumber": "+15105550199",
      "sip.callID": "call-1",
  })
  runner = _make_runner([])
  runner.session_service.get_session = AsyncMock(return_value=None)
  lk_runner = make_lk_runner(runner, make_room({"sip_caller": participant}))

  await lk_runner._ensure_session()

  state = runner.session_service.create_session.await_args.kwargs["state"]
  assert state["livekit_caller_phone_number"] == "+15105550100"
  assert state["livekit_called_phone_number"] == "+15105550199"
  assert state["livekit_is_phone_call"] is True


async def test_webrtc_call_adds_no_telephony_state():
  """A browser caller has no phone number; state stays clean."""
  runner = _make_runner([])
  runner.session_service.get_session = AsyncMock(return_value=None)
  lk_runner = make_lk_runner(runner, make_room())

  await lk_runner._ensure_session()

  assert runner.session_service.create_session.await_args.kwargs["state"] == {}


async def test_late_sip_attributes_reach_the_running_session():
  """Attributes mapped from SIP headers arrive after the participant does.

  A second identical change is not resent, since that would append a junk
  event to the session.
  """
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  lk_runner = make_lk_runner(
      _make_runner([]), make_room({"sip_caller": participant})
  )
  captured = []
  lk_runner._queue.send = captured.append

  lk_runner._on_participant_attributes({}, participant)
  lk_runner._on_participant_attributes({}, participant)

  assert len(captured) == 1
  assert captured[0].state_delta["livekit_caller_phone_number"] == (
      "+15105550100"
  )


async def test_the_agents_own_attributes_are_not_echoed_back():
  """Publishing agent state fires this handler; that must not loop.

  `lk.agent.state` is a participant attribute, so every listening/thinking/
  speaking transition comes straight back as an attribute change on the local
  participant.
  """
  participant = sip_participant({"sip.phoneNumber": "+15105550100"})
  room = make_room({"sip_caller": participant})
  lk_runner = make_lk_runner(_make_runner([]), room)
  captured = []
  lk_runner._queue.send = captured.append

  lk_runner._on_participant_attributes(
      {"lk.agent.state": "speaking"}, room.local_participant
  )

  assert captured == []


# --- Lifecycle ---


async def test_start_closes_queue_when_session_ends():
  """When run_live finishes, the live request queue is closed."""
  lk_runner = make_lk_runner(_make_runner([]), make_room())
  lk_runner._queue.close = MagicMock()

  await lk_runner.start()

  lk_runner._queue.close.assert_called_once()


async def test_a_failed_setup_still_tears_the_call_down():
  """A call that never starts must not leak the queue it would have used.

  Setup fails before the event pump exists, so teardown has to cope with
  there being nothing yet to cancel.
  """
  runner = _make_runner([])
  runner.session_service.get_session = AsyncMock(return_value=None)
  runner.session_service.create_session = AsyncMock(
      side_effect=RuntimeError("session service is down")
  )
  lk_runner = make_lk_runner(runner, make_room())
  lk_runner._queue.close = MagicMock()

  with pytest.raises(RuntimeError, match="session service is down"):
    await lk_runner.start()

  lk_runner._queue.close.assert_called_once()


async def test_the_room_is_released_when_the_session_ends():
  """A room can outlive one call, and LiveKit allows one handler per topic.

  Leaving the chat handler registered makes the *next* call fail rather than
  this one, which is a miserable way to find out.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)

  await lk_runner.start()

  room.unregister_text_stream_handler.assert_called_once_with(
      _livekit_runner.LK_CHAT_TOPIC
  )
  assert room.off.call_count == room.on.call_count


async def test_forwarders_are_awaited_before_the_queue_closes():
  """A forwarder's teardown still pushes onto the queue being closed.

  `cancel()` only schedules cancellation, so a bridge that closes the queue
  without awaiting its forwarders races its own shutdown.
  """
  room = make_room()
  lk_runner = make_lk_runner(_make_runner([]), room)
  stopped = asyncio.Event()

  async def _forwarder():
    try:
      await asyncio.Event().wait()
    finally:
      stopped.set()

  lk_runner._forward_tasks.add(asyncio.create_task(_forwarder()))

  await lk_runner.start()

  assert stopped.is_set()


async def test_room_disconnect_ends_the_session():
  """Losing the room connection ends the call rather than hanging."""
  lk_runner = make_lk_runner(_never_ending_runner(), make_room())

  start = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)
  lk_runner._on_disconnected()

  await asyncio.wait_for(start, timeout=5)


async def test_last_participant_leaving_ends_the_session():
  """The caller hanging up is the end of the call.

  A dispatched worker is torn down by LiveKit, but a room joined directly
  would otherwise leave the agent alone in it holding a live model connection.
  Closing the queue is not enough: with session resumption enabled `run_live`
  reads that as a dropped connection and reconnects.
  """
  room = make_room()
  lk_runner = make_lk_runner(_never_ending_runner(), room)

  start = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)
  lk_runner._on_participant_disconnected(MagicMock())

  await asyncio.wait_for(start, timeout=5)


async def test_session_survives_one_of_several_participants_leaving():
  """Someone else leaving a multi-party room does not end the call."""
  room = make_room({"still-here": MagicMock(track_publications={})})
  lk_runner = make_lk_runner(_never_ending_runner(), room)

  start = asyncio.create_task(lk_runner.start())
  await asyncio.sleep(0)
  lk_runner._on_participant_disconnected(MagicMock())

  with pytest.raises(asyncio.TimeoutError):
    await asyncio.wait_for(asyncio.shield(start), timeout=0.2)
  start.cancel()
  with contextlib.suppress(asyncio.CancelledError):
    await start


# --- Sessions ---


def _real_runner(app_name: str = "dice") -> InMemoryRunner:
  """A real Runner whose agent answers over a mocked live connection."""
  model = MockModel.create(responses=["you rolled a four"])
  return InMemoryRunner(
      agent=LlmAgent(name="dice_agent", model=model), app_name=app_name
  )


async def test_session_created_for_a_new_room():
  """A freshly joined room has no ADK session, so the connector makes one.

  `Runner` raises `SessionNotFoundError` rather than creating sessions, so a
  brand new room used to fail before a single frame moved.
  """
  runner = _real_runner()
  lk_runner = make_lk_runner(runner, make_room())

  await lk_runner._ensure_session()

  assert (
      await runner.session_service.get_session(
          app_name="dice", user_id="u1", session_id="s1"
      )
      is not None
  )


async def test_existing_session_is_reused():
  """An out-of-band session is picked up, not replaced."""
  runner = _real_runner()
  created = await runner.session_service.create_session(
      app_name="dice", user_id="u1", session_id="s1"
  )
  lk_runner = make_lk_runner(runner, make_room())

  await lk_runner._ensure_session()

  found = await runner.session_service.get_session(
      app_name="dice", user_id="u1", session_id="s1"
  )
  assert found.id == created.id


async def test_concurrent_create_is_tolerated():
  """Two workers on the same room must not fight over creating its session.

  Dispatch retries and rejoins both land two callers on one room; the loser of
  the create race should carry on with the session the winner made.
  """
  runner = _real_runner()
  lk_runner = make_lk_runner(runner, make_room())
  real_create = runner.session_service.create_session

  async def _create_then_conflict(**kwargs):
    await real_create(**kwargs)  # the other worker won
    raise AlreadyExistsError("Session with id s1 already exists.")

  runner.session_service.create_session = _create_then_conflict

  await lk_runner._ensure_session()  # must not raise

  assert (
      await runner.session_service.get_session(
          app_name="dice", user_id="u1", session_id="s1"
      )
      is not None
  )


async def test_create_session_can_be_opted_out():
  """With create_session False the caller owns session lifecycle."""
  runner = _real_runner()
  lk_runner = make_lk_runner(runner, make_room(), create_session=False)

  with pytest.raises(SessionNotFoundError):
    await lk_runner.start()

  assert (
      await runner.session_service.get_session(
          app_name="dice", user_id="u1", session_id="s1"
      )
      is None
  )


async def test_room_drives_a_real_run_live():
  """End to end: a connected room reaches the model over a real Runner."""
  runner = _real_runner()
  room = make_room()
  lk_runner = make_lk_runner(runner, room)

  # `run_live` stays open for the life of the call, so stop it once the
  # session is established and the outbound track is published.
  task = asyncio.create_task(lk_runner.start())
  try:
    session = await _await_session(runner, "dice", "u1", "s1")
    assert session is not None, _failure_of(task) or (
        "run_live never created the room's session."
    )
  finally:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task

  room.local_participant.publish_track.assert_awaited_once()


async def _await_session(runner: Runner, app_name: str, user_id: str, sid: str):
  """Polls the runner's session service until the session shows up."""
  for _ in range(100):
    session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=sid
    )
    if session is not None:
      return session
    await asyncio.sleep(0.05)
  return None


def _failure_of(task: asyncio.Task) -> str:
  """Renders a task's exception, so a crashed session reports its own cause."""
  if not task.done() or task.cancelled():
    return ""
  exc = task.exception()
  return f"Entrypoint raised {type(exc).__name__}: {exc}" if exc else ""


# --- The optional dependency ---


def test_using_the_connector_without_the_extra_names_it():
  """A missing `livekit` must say which extra installs it.

  Run in a fresh interpreter: blocking an import is process-global, and the
  rest of this file depends on the real SDK being importable.
  """
  result = run_isolated("""
import sys


class _NoLiveKit:

  def find_spec(self, name, path=None, target=None):
    if name == 'livekit' or name.startswith('livekit.'):
      raise ImportError('livekit is not installed')
    return None


sys.meta_path.insert(0, _NoLiveKit())

# The package itself imports lazily, so this much works with no SDK present.
import google.adk.integrations.livekit as livekit_integration

try:
  livekit_integration.LiveKitRunner
except ImportError as e:
  assert 'google-adk[livekit]' in str(e), str(e)
else:
  raise AssertionError('Expected an ImportError naming the extra.')
""")

  assert result.returncode == 0, result.stderr

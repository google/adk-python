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

"""Tests for GeminiSTT.

The Live API is faked by sessions whose server messages each test scripts in
response to what the transform sends.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Callable
from typing import Optional
from unittest import mock

from google.adk.live import _gemini_stt
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import CascadeLive
from google.adk.live import GeminiSTT
from google.adk.live import LiveEgress
from google.adk.live import PartialTranscript
from google.adk.live import UserSpeechStarted
from google.adk.live import UserTurnFinished
from google.adk.live._gemini_stt import _AudioStream
from google.adk.live._gemini_stt import _is_transient
from google.adk.live._gemini_stt import _TranscriptionState
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors
from google.genai import types
import pytest
from typing_extensions import override
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError
from websockets.exceptions import InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

_MIME_TYPE = 'audio/pcm;rate=16000'

_START = types.VoiceActivityType.ACTIVITY_START
_END = types.VoiceActivityType.ACTIVITY_END


def _activity(kind: types.VoiceActivityType) -> types.LiveServerMessage:
  return types.LiveServerMessage(
      voice_activity=types.VoiceActivity(voice_activity_type=kind)
  )


def _interim(text: str) -> types.LiveServerMessage:
  return types.LiveServerMessage(
      server_content=types.LiveServerContent(
          interim_input_transcription=types.Transcription(text=text)
      )
  )


def _final(text: str) -> types.LiveServerMessage:
  return types.LiveServerMessage(
      server_content=types.LiveServerContent(
          input_transcription=types.Transcription(text=text)
      )
  )


_GENERATION_COMPLETE = types.LiveServerMessage(
    server_content=types.LiveServerContent(generation_complete=True)
)
_GO_AWAY = types.LiveServerMessage(go_away=types.LiveServerGoAway())


class _FakeSession:
  """A Live API session; `script` answers each client message."""

  def __init__(
      self, script: Callable[[_FakeSession, dict[str, Any]], None]
  ) -> None:
    self.sent: list[dict[str, Any]] = []
    self._script = script
    self._messages: asyncio.Queue[Any] = asyncio.Queue()

  @property
  def audio(self) -> list[bytes]:
    return [
        message['audio'].data for message in self.sent if 'audio' in message
    ]

  async def send_realtime_input(self, **kwargs: Any) -> None:
    self.sent.append(kwargs)
    self._script(self, kwargs)

  def reply(self, *messages: Any) -> None:
    for message in messages:
      self._messages.put_nowait(message)

  async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
    while True:
      message = await self._messages.get()
      if isinstance(message, Exception):
        raise message
      yield message


class _FakeLive:
  """Stands in for `client.aio.live`, recording each session it opens."""

  def __init__(self) -> None:
    self.script: Callable[[_FakeSession, dict[str, Any]], None] = (
        lambda session, message: None
    )
    self.sessions: list[_FakeSession] = []
    self.configs: list[types.LiveConnectConfig] = []
    # Raised, in order, by the next attempts to open a session; `None` lets
    # that attempt succeed.
    self.connect_errors: list[Optional[Exception]] = []
    self.connect_attempts = 0

  @contextlib.asynccontextmanager
  async def connect(
      self, *, model: str, config: types.LiveConnectConfig
  ) -> AsyncIterator[_FakeSession]:
    """Opens a session that answers with `script`, or fails as told."""
    del model
    self.connect_attempts += 1
    error = self.connect_errors.pop(0) if self.connect_errors else None
    if error is not None:
      raise error
    session = _FakeSession(self.script)
    self.sessions.append(session)
    self.configs.append(config)
    yield session


@pytest.fixture(name='live')
def _live(monkeypatch):
  monkeypatch.setattr(_gemini_stt, '_FLUSH_QUIET_S', 0.01)
  fake = _FakeLive()
  client = mock.MagicMock()
  client.aio.live.connect = fake.connect
  with mock.patch.object(
      Gemini, 'api_client', new=property(lambda self: client)
  ):
    yield fake


def _blob(index: int, mime_type: str = _MIME_TYPE) -> types.Blob:
  """Returns 10 ms of 16 kHz audio, marked with its index."""
  return types.Blob(data=index.to_bytes(2, 'little') * 160, mime_type=mime_type)


async def _audio(
    count: int, *, mime_type: str = _MIME_TYPE, endless: bool = False
) -> AsyncIterator[types.Blob]:
  for index in range(1, count + 1):
    # Lets the session's messages be handled between frames, as with a
    # real microphone.
    await asyncio.sleep(0)
    yield _blob(index, mime_type)
  if endless:
    # The microphone stays open without sending more audio.
    await asyncio.Event().wait()


# A server that hears speech in the first frame and finalizes it at the end of
# the stream.
def _answers_one_question(session: _FakeSession, message: dict[str, Any]):
  if session.audio == [_blob(1).data]:
    session.reply(
        _activity(_START),
        _interim('what is'),
        _interim('what is'),
        _interim('what is the weather'),
    )
  if message.get('audio_stream_end'):
    session.reply(
        _final(' What is the weather? '), _GENERATION_COMPLETE, _activity(_END)
    )


async def _transcribe(stt: GeminiSTT, audio: AsyncIterator[types.Blob]):
  return [event async for event in stt(audio)]


async def test_server_messages_become_ingress_events(live):
  """Speech start, changed captions and the final transcript are reported."""
  live.script = _answers_one_question

  events = await _transcribe(GeminiSTT(), _audio(3))

  assert events == [
      UserSpeechStarted(),
      PartialTranscript(text='what is'),
      PartialTranscript(text='what is the weather'),
      UserTurnFinished(text='What is the weather?'),
  ]


async def test_audio_reaches_the_server_in_order(live):
  """Every client frame is streamed to the session unchanged."""
  await _transcribe(GeminiSTT(), _audio(3))

  assert [message['audio'] for message in live.sessions[0].sent[:3]] == [
      _blob(1),
      _blob(2),
      _blob(3),
  ]


async def test_segment_end_finalizes_the_speech_in_progress(live):
  """At the end of a segment the transform asks for the final transcript.

  The stream end finalizes speech the server is in the middle of, and the
  trailing silence (the 500 ms silence duration plus a margin) ends speech it
  has not processed yet.
  """
  await _transcribe(GeminiSTT(), _audio(1))

  assert live.sessions[0].sent[-2:] == [
      {'audio_stream_end': True},
      {
          'audio': types.Blob(
              data=bytes(16000 * 700 // 1000 * 2), mime_type=_MIME_TYPE
          )
      },
  ]


async def test_turn_ends_after_half_a_second_of_silence_by_default(live):
  """Unless configured, the server ends the turn after 500 ms of silence."""
  await _transcribe(GeminiSTT(), _audio(1))

  detection = live.configs[0].realtime_input_config.automatic_activity_detection
  assert detection.silence_duration_ms == 500


async def test_activity_detection_settings_are_sent_to_the_server(live):
  """Explicit activity detection settings reach the server as given."""
  detection = types.AutomaticActivityDetection(
      silence_duration_ms=300,
      start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
  )

  await _transcribe(GeminiSTT(activity_detection=detection), _audio(1))

  assert (
      live.configs[0].realtime_input_config.automatic_activity_detection
      == detection
  )


async def test_transcription_settings_are_sent_to_the_server(live):
  """Language hints and custom vocabulary reach the server."""
  transcription = types.AudioTranscriptionConfig(
      language_codes=['en-US'], custom_vocabulary=['ADK']
  )

  await _transcribe(
      GeminiSTT(audio_transcription_config=transcription), _audio(1)
  )

  assert live.configs[0].input_audio_transcription == transcription


async def test_client_turn_detection_brackets_each_segment(live):
  """With detection disabled, the segment is one activity the client ended."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if 'activity_start' in message:
      session.reply(_activity(_START))
    if 'activity_end' in message:
      session.reply(_final('Hello.'), _GENERATION_COMPLETE, _activity(_END))

  live.script = script

  events = await _transcribe(
      GeminiSTT(
          activity_detection=types.AutomaticActivityDetection(disabled=True)
      ),
      _audio(2),
  )

  assert [list(message) for message in live.sessions[0].sent] == [
      ['activity_start'],
      ['audio'],
      ['audio'],
      ['activity_end'],
  ]
  assert events == [UserSpeechStarted(), UserTurnFinished(text='Hello.')]


async def test_session_error_fails_the_ingress_while_audio_is_awaited(live):
  """An error the session reports surfaces even if no more audio comes."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    session.reply(RuntimeError('session failed'))

  live.script = script

  with pytest.raises(RuntimeError, match='session failed'):
    await asyncio.wait_for(
        _transcribe(GeminiSTT(), _audio(1, endless=True)), timeout=5
    )
  assert len(live.sessions) == 1


async def test_go_away_replaces_the_session_right_away(live):
  """GoAway in the middle of an utterance replaces the session at once.

  Setup: the first session hears speech at frame 1 and sends GoAway; the new
    session hears the speech sent again and finalizes it at the end.
  Assert:
    - The new session gets every frame since the utterance began.
    - The start of speech and the final transcript are each reported once.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if len(session.audio) == 1 and 'audio' in message:
      session.reply(_activity(_START), _interim('what is'))
      if session is live.sessions[0]:
        session.reply(_GO_AWAY)
    if message.get('audio_stream_end'):
      session.reply(_final('What is the weather?'), _GENERATION_COMPLETE)

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(6))

  first, second = live.sessions
  assert len(first.audio) < 6
  assert second.audio[:6] == [_blob(i).data for i in range(1, 7)]
  assert events.count(UserSpeechStarted()) == 1
  assert [e for e in events if isinstance(e, UserTurnFinished)] == [
      UserTurnFinished(text='What is the weather?')
  ]


@pytest.mark.parametrize(
    ('speaking', 'rotates'), [(False, True), (True, False)]
)
async def test_session_is_replaced_by_age_only_between_utterances(
    live, monkeypatch, speaking, rotates
):
  """A session past its age limit is replaced only while nobody speaks."""
  del live
  monkeypatch.setattr(_gemini_stt, '_SESSION_ROTATION_S', 0)
  session = _FakeSession(lambda session, message: None)
  state = _TranscriptionState(lambda event: None, on_finalized=lambda: None)
  state.speaking = speaking
  stream = _AudioStream(_audio(3), tail_ms=500, max_ms=10_000)

  ended = await GeminiSTT()._send(session, stream, state)  # pylint: disable=protected-access

  assert ended is not rotates
  assert len(session.audio) == (1 if rotates else 3)


async def test_replacing_session_gets_the_audio_since_the_last_final(live):
  """A session replaced between utterances hands over the recent audio.

  Setup: the first session finalizes an utterance at frame 70 and is retired
    with GoAway at frame 80. Each frame is 10 ms.
  Assert:
    - The new session starts 500 ms before the final transcript (frame 21).
    - It then gets every later frame, in order.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if session is not live.sessions[0]:
      return
    if len(session.audio) == 1:
      session.reply(_activity(_START))
    if len(session.audio) == 70:
      session.reply(_final('Hi.'), _GENERATION_COMPLETE, _activity(_END))
    if len(session.audio) == 80:
      session.reply(_GO_AWAY)

  live.script = script

  await _transcribe(GeminiSTT(), _audio(100))

  first, second = live.sessions
  handed_over = len(first.audio) - 20
  assert second.audio[:handed_over] == first.audio[20:]
  assert second.audio[handed_over] == _blob(len(first.audio) + 1).data


async def test_short_silence_keeps_less_audio_after_a_final(live):
  """Audio kept after a final transcript is no longer than the silence.

  Setup: the silence duration is 200 ms; the first session finalizes an
    utterance at frame 70, hears new speech at frame 80 and drops at frame 90.
    Each frame is 10 ms.
  Assert: the reopened session starts 200 ms before the final transcript
    (frame 51), so none of the finalized speech is sent again.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if session is live.sessions[0]:
      if len(session.audio) in (1, 80):
        session.reply(_activity(_START))
      if len(session.audio) == 70:
        session.reply(_final('First.'))
      if len(session.audio) == 90:
        session.reply(ConnectionError('connection reset'))
    elif message.get('audio_stream_end'):
      session.reply(_final('Second.'), _GENERATION_COMPLETE)

  live.script = script

  await _transcribe(
      GeminiSTT(
          activity_detection=types.AutomaticActivityDetection(
              silence_duration_ms=200
          )
      ),
      _audio(95),
  )

  assert live.sessions[1].audio[0] == _blob(51).data


async def test_dropped_session_is_reopened_with_the_unfinished_utterance(live):
  """A session that drops mid-utterance is reopened, and nothing is lost.

  Setup: the first session hears speech at frame 1 and drops at frame 3.
  Assert:
    - The reopened session gets every frame since the utterance began.
    - The start of speech and the final transcript are each reported once.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if len(session.audio) == 1 and 'audio' in message:
      session.reply(_activity(_START), _interim('what is'))
    if session is live.sessions[0] and len(session.audio) == 3:
      session.reply(ConnectionError('connection reset'))
    if session is not live.sessions[0] and message.get('audio_stream_end'):
      session.reply(_final('What is the weather?'), _GENERATION_COMPLETE)

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(5))

  assert len(live.sessions) == 2
  assert live.sessions[1].audio[:5] == [_blob(i).data for i in range(1, 6)]
  # The reopened session hears the speech start again, but it is one utterance.
  assert events.count(UserSpeechStarted()) == 1
  assert [e for e in events if isinstance(e, UserTurnFinished)] == [
      UserTurnFinished(text='What is the weather?')
  ]


async def test_sessions_that_keep_dropping_fail_the_ingress(live):
  """After repeated drops, the last session's own error surfaces unchanged."""
  raised: list[Exception] = []

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if len(session.audio) == 1:
      raised.append(errors.APIError(1011, 'internal error'))
      session.reply(raised[-1])

  live.script = script

  with pytest.raises(errors.APIError) as failure:
    await asyncio.wait_for(
        _transcribe(GeminiSTT(), _audio(1, endless=True)), timeout=5
    )
  assert failure.value is raised[-1]
  # The first session and two reopened ones.
  assert len(live.sessions) == 3


async def test_non_transient_session_error_is_not_retried(live):
  """An error a new session would hit again surfaces without reopening."""
  error = errors.APIError(1008, 'policy violation')

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if len(session.audio) == 1:
      session.reply(error)

  live.script = script

  with pytest.raises(errors.APIError) as failure:
    await asyncio.wait_for(
        _transcribe(GeminiSTT(), _audio(1, endless=True)), timeout=5
    )
  assert failure.value is error
  assert len(live.sessions) == 1


async def test_session_dropped_while_finalizing_is_reopened(live):
  """A drop after the segment ended still yields the last transcript.

  Setup: the first session hears speech and drops when asked to finalize it.
  Assert: the reopened session gets the speech again and finalizes it once.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if session is live.sessions[0]:
      if len(session.audio) == 1 and 'audio' in message:
        session.reply(_activity(_START), _interim('what is'))
      if message.get('audio_stream_end'):
        session.reply(ConnectionError('connection reset'))
    elif message.get('audio_stream_end'):
      session.reply(_final('What is the weather?'), _GENERATION_COMPLETE)

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(2))

  assert live.sessions[1].audio[:2] == [_blob(1).data, _blob(2).data]
  assert [e for e in events if isinstance(e, UserTurnFinished)] == [
      UserTurnFinished(text='What is the weather?')
  ]


@pytest.mark.parametrize(
    ('finalization', 'expected_turns'),
    [
        ([_final('First.')], ['First.', 'Second.']),
        # An utterance in which the server found no words.
        ([_GENERATION_COMPLETE], ['Second.']),
    ],
)
async def test_reopened_session_skips_utterances_already_finalized(
    live, finalization, expected_turns
):
  """Speech the server is done with is not sent again after a drop.

  Setup: the first session finalizes an utterance at frame 70, hears new
    speech at frame 80 and drops at frame 90. Each frame is 10 ms.
  Assert:
    - The reopened session starts 500 ms before the end of the finalized
      utterance (frame 21), not at the first frame.
    - Each utterance is reported once.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if session is live.sessions[0]:
      if len(session.audio) in (1, 80):
        session.reply(_activity(_START))
      if len(session.audio) == 70:
        session.reply(*finalization)
      if len(session.audio) == 90:
        session.reply(ConnectionError('connection reset'))
    elif message.get('audio_stream_end'):
      session.reply(_final('Second.'), _GENERATION_COMPLETE)

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(95))

  second = live.sessions[1]
  assert second.audio[0] == _blob(21).data
  assert _blob(90).data in second.audio
  assert [
      e.text for e in events if isinstance(e, UserTurnFinished)
  ] == expected_turns


async def test_drop_while_waiting_for_audio_keeps_the_segment_open(live):
  """A session that drops between frames does not end the client's audio.

  Setup: the first session drops right after frame 1, while the transform
    waits for frame 2, which arrives only once the session is reopened.
  Assert:
    - The reopened session gets frames 1 to 3, then the end of the segment.
    - No third session is opened.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if session is live.sessions[0] and len(session.audio) == 1:
      session.reply(ConnectionError('connection reset'))

  live.script = script

  async def audio() -> AsyncIterator[types.Blob]:
    yield _blob(1)
    while len(live.sessions) < 2:
      await asyncio.sleep(0)
    yield _blob(2)
    yield _blob(3)

  await _transcribe(GeminiSTT(), audio())

  assert len(live.sessions) == 2
  second = live.sessions[1]
  assert second.audio[:3] == [_blob(i).data for i in (1, 2, 3)]
  assert second.sent.count({'audio_stream_end': True}) == 1


async def test_closing_the_ingress_stops_reading_the_audio(live):
  """Closing the transform stops a read in progress.

  Otherwise the read would take the first frame of the next audio segment.
  """
  stopped = asyncio.Event()

  async def audio() -> AsyncIterator[types.Blob]:
    try:
      yield _blob(1)
      await asyncio.Event().wait()
    finally:
      stopped.set()

  events = GeminiSTT()(audio())
  reading = asyncio.create_task(anext(events))
  for _ in range(100):
    if live.sessions and live.sessions[0].audio:
      break
    await asyncio.sleep(0)
  reading.cancel()
  await asyncio.gather(reading, return_exceptions=True)
  await events.aclose()

  assert stopped.is_set()


@pytest.mark.parametrize(
    'trigger',
    [ConnectionError('connection reset'), _GO_AWAY],
    ids=['drop', 'go_away'],
)
async def test_client_turn_detection_new_session_starts_the_activity_first(
    live, trigger
):
  """With client turn detection, a new session starts the activity first.

  The speech sent again belongs to the client's activity, so it follows
  `ActivityStart`.

  Setup: the first session drops, or sends GoAway, after frame 2.
  Assert:
    - The new session gets `ActivityStart`, then frames 1 to 4, then
      `ActivityEnd`, and finalizes the whole segment.
    - The first session's activity is never ended.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if session is live.sessions[0]:
      if len(session.audio) == 2 and 'audio' in message:
        session.reply(trigger)
    elif 'activity_end' in message:
      session.reply(_final('Hello.'), _GENERATION_COMPLETE, _activity(_END))

  live.script = script

  events = await _transcribe(
      GeminiSTT(
          activity_detection=types.AutomaticActivityDetection(disabled=True)
      ),
      _audio(4),
  )

  first, second = live.sessions
  assert {'activity_end': types.ActivityEnd()} not in first.sent
  assert list(second.sent[0]) == ['activity_start']
  assert second.audio[:4] == [_blob(i).data for i in range(1, 5)]
  assert list(second.sent[-1]) == ['activity_end']
  assert UserTurnFinished(text='Hello.') in events


async def test_max_replay_ms_limits_the_speech_sent_again(live):
  """A smaller `max_replay_ms` sends less of a long utterance again."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if session is live.sessions[0]:
      if len(session.audio) == 1:
        session.reply(_activity(_START))
      if len(session.audio) == 50:
        session.reply(ConnectionError('connection reset'))

  live.script = script

  await _transcribe(GeminiSTT(max_replay_ms=200), _audio(55))

  first, second = live.sessions
  # Each frame is 10 ms.
  assert second.audio[:20] == first.audio[-20:]
  assert second.audio[20] == _blob(len(first.audio) + 1).data


def test_negative_max_replay_ms_is_rejected():
  """A negative replay limit is a configuration error."""
  with pytest.raises(ValueError, match='max_replay_ms'):
    GeminiSTT(max_replay_ms=-1)


async def test_reopened_session_that_hears_no_speech_ends_promptly(
    live, monkeypatch, caplog
):
  """Speech the old session heard does not stay pending on the new one.

  Setup: the first session hears speech and drops; the reopened session
    reports nothing for the audio sent again.
  Assert: the segment ends without waiting for a transcript that never comes.
  """
  monkeypatch.setattr(_gemini_stt, '_FLUSH_TIMEOUT_S', 1.0)

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if session is live.sessions[0]:
      if len(session.audio) == 1:
        session.reply(_activity(_START))
      if len(session.audio) == 3:
        session.reply(ConnectionError('connection reset'))

  live.script = script

  await _transcribe(GeminiSTT(), _audio(5))

  assert len(live.sessions) == 2
  assert 'returned no transcript' not in caplog.text


async def test_reopened_session_gets_at_most_ten_seconds_again(live):
  """Only the last 10 s of a long unfinished utterance are sent again."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if session is live.sessions[0]:
      if len(session.audio) == 1:
        session.reply(_activity(_START))
      if len(session.audio) == 1100:
        session.reply(ConnectionError('connection reset'))

  live.script = script

  await _transcribe(GeminiSTT(), _audio(1105))

  first, second = live.sessions
  assert len(first.audio) >= 1100
  # Each frame is 10 ms.
  assert second.audio[:1000] == first.audio[-1000:]


async def test_failure_to_open_a_reopened_session_counts_as_a_drop(live):
  """A reopened session that fails to open is attempted again."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if session is live.sessions[0] and len(session.audio) == 1:
      session.reply(ConnectionError('connection reset'))
    elif message.get('audio_stream_end'):
      session.reply(_final('Hi.'), _GENERATION_COMPLETE)

  live.script = script
  live.connect_errors = [None, ConnectionError('no route to host')]

  events = await _transcribe(GeminiSTT(), _audio(3))

  assert UserTurnFinished(text='Hi.') in events
  assert live.connect_attempts == 3


async def test_non_transient_failure_to_open_is_not_retried(live):
  """An error opening a session, such as bad credentials, surfaces."""
  live.connect_errors = [PermissionError('denied')]

  with pytest.raises(PermissionError, match='denied'):
    await _transcribe(GeminiSTT(), _audio(1))
  assert live.connect_attempts == 1


async def test_transient_failure_to_open_the_first_session_is_retried(live):
  """A first session that fails to open for a passing reason is retried."""

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del session
    if message.get('audio_stream_end'):
      live.sessions[-1].reply(_final('Hi.'), _GENERATION_COMPLETE)

  live.script = script
  live.connect_errors = [ConnectionError('no route to host')]

  events = await _transcribe(GeminiSTT(), _audio(2))

  assert UserTurnFinished(text='Hi.') in events
  assert live.connect_attempts == 2
  assert live.sessions[0].audio[:2] == [_blob(1).data, _blob(2).data]


def _http_error(code: int) -> errors.APIError:
  error_type = errors.ServerError if code >= 500 else errors.ClientError
  return error_type(
      code, {'error': {'code': code, 'message': 'failed', 'status': 'ERROR'}}
  )


@pytest.mark.parametrize(
    ('error', 'transient'),
    [
        (errors.APIError(1011, 'internal error'), True),
        (errors.APIError(1006, 'abnormal closure'), True),
        (errors.APIError(1000, 'normal closure'), True),
        (errors.APIError(1008, 'policy violation'), False),
        (_http_error(503), True),
        (_http_error(429), True),
        (_http_error(403), False),
        (ConnectionClosedError(None, None), True),
        (ConnectionClosedError(Close(1008, 'policy violation'), None), False),
        (InvalidStatus(Response(503, 'Service Unavailable', Headers())), True),
        (InvalidStatus(Response(401, 'Unauthorized', Headers())), False),
        (ConnectionError('connection reset'), True),
        (TimeoutError(), True),
        (PermissionError('denied'), False),
        (RuntimeError('bug'), False),
    ],
)
def test_transient_errors_are_told_apart(error, transient):
  """Only errors after which a new session may succeed are transient."""
  assert _is_transient(error) is transient


async def test_sessions_that_ran_stably_reset_the_reconnect_limit(
    live, monkeypatch
):
  """Occasional drops in a long conversation do not add up to a failure."""
  monkeypatch.setattr(_gemini_stt, '_STABLE_SESSION_S', 0)

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if len(live.sessions) <= 4 and len(session.audio) == 1:
      session.reply(ConnectionError('connection reset'))
    elif message.get('audio_stream_end'):
      session.reply(_final('Hi.'), _GENERATION_COMPLETE)

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(10))

  assert len(live.sessions) == 5
  assert UserTurnFinished(text='Hi.') in events


async def test_speech_never_finalized_ends_the_segment_after_a_wait(
    live, monkeypatch, caplog
):
  """If the server never finalizes the last speech, the segment still ends."""
  monkeypatch.setattr(_gemini_stt, '_FLUSH_TIMEOUT_S', 0.05)

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    del message
    if len(session.audio) == 1:
      session.reply(_activity(_START))

  live.script = script

  events = await _transcribe(GeminiSTT(), _audio(2))

  assert events == [UserSpeechStarted()]
  assert 'returned no transcript' in caplog.text


async def test_empty_audio_frames_are_not_sent(live):
  """Frames without audio data are skipped."""

  async def audio() -> AsyncIterator[types.Blob]:
    yield _blob(1)
    yield types.Blob(data=b'', mime_type=_MIME_TYPE)
    yield _blob(2)

  await _transcribe(GeminiSTT(), audio())

  assert live.sessions[0].audio[:2] == [_blob(1).data, _blob(2).data]
  assert b'' not in live.sessions[0].audio


@pytest.mark.parametrize(
    ('mime_type', 'sample_rate'),
    [('audio/pcm;rate=24000', 24000), ('audio/pcm', 16000)],
)
async def test_trailing_silence_matches_the_audio_sample_rate(
    live, mime_type, sample_rate
):
  """The silence that ends the segment uses the client's sample rate."""
  await _transcribe(GeminiSTT(), _audio(1, mime_type=mime_type))

  assert live.sessions[0].sent[-1] == {
      'audio': types.Blob(
          data=bytes(sample_rate * 700 // 1000 * 2), mime_type=mime_type
      )
  }


@pytest.mark.usefixtures('live')
async def test_client_turn_detection_stops_waiting_for_the_activity_end(
    monkeypatch, caplog
):
  """If the server never confirms the end of the activity, the segment ends."""
  monkeypatch.setattr(_gemini_stt, '_FLUSH_TIMEOUT_S', 0.05)

  events = await _transcribe(
      GeminiSTT(
          activity_detection=types.AutomaticActivityDetection(disabled=True)
      ),
      _audio(1),
  )

  assert not events
  assert 'did not end the activity' in caplog.text


class _ScriptedLlm(BaseLlm):
  """Answers every request with the same streamed reply."""

  model: str = 'scripted'
  reply: str = 'It is sunny.'
  requests: list[LlmRequest] = []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request)
    yield LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text=self.reply)])
    )


class _EchoTts(LiveEgress):
  """Speaks each reply as one chunk of its UTF-8 bytes."""

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[Any, None]:
    del cancel
    reply = ''.join([delta async for delta in text])
    yield AudioChunk(
        blob=types.Blob(data=reply.encode(), mime_type='audio/pcm')
    )
    yield AgentSpokenOutput(text=reply)


async def test_cascade_answers_a_spoken_question(live):
  """A transcript from GeminiSTT reaches the reasoner as the user's turn."""
  live.script = _answers_one_question
  llm = _ScriptedLlm()
  model = CascadeLive(model=llm, stt=GeminiSTT(), tts=_EchoTts())
  responses: list[LlmResponse] = []

  async with model.connect(LlmRequest()) as connection:
    for index in (1, 2):
      await connection.send_realtime(_blob(index))
    await connection.send_realtime(types.ActivityEnd())

    async def read_turn() -> None:
      async with contextlib.aclosing(connection.receive()) as stream:
        async for response in stream:
          responses.append(response)
          if response.turn_complete:
            return

    await asyncio.wait_for(read_turn(), timeout=5)

  assert llm.requests[0].contents == [
      types.Content(
          role='user', parts=[types.Part(text='What is the weather?')]
      )
  ]
  assert any(
      response.content
      and response.content.parts[0].inline_data
      and response.content.parts[0].inline_data.data == b'It is sunny.'
      for response in responses
  )


async def test_speech_over_the_agent_interrupts_it(live):
  """Speech GeminiSTT detects while the agent speaks interrupts the reply.

  Setup: the server finalizes a question at frame 2, and the TTS speaks the
    first chunk of the reply, then waits.
  Act: the server detects new speech at frame 3.
  Assert: the connection reports the interruption.
  """

  def script(session: _FakeSession, message: dict[str, Any]) -> None:
    if 'audio' not in message:
      return
    if len(session.audio) == 1:
      session.reply(_activity(_START), _interim('what is the weather'))
    elif len(session.audio) == 2:
      session.reply(
          _final('What is the weather?'),
          _GENERATION_COMPLETE,
          _activity(_END),
      )
    elif len(session.audio) == 3:
      session.reply(_activity(_START))

  live.script = script
  speaking = asyncio.Event()

  class _WaitingTts(LiveEgress):
    """Speaks the first chunk of a reply, then waits to be cancelled."""

    async def __call__(
        self, text: AsyncIterator[str], *, cancel: asyncio.Event
    ) -> AsyncGenerator[Any, None]:
      async for _ in text:
        pass
      # One second of audio, so the reply is still playing at frame 3.
      yield AudioChunk(
          blob=types.Blob(data=bytes(48000), mime_type='audio/pcm;rate=24000')
      )
      speaking.set()
      await cancel.wait()

  model = CascadeLive(model=_ScriptedLlm(), stt=GeminiSTT(), tts=_WaitingTts())
  responses: list[LlmResponse] = []

  async with model.connect(LlmRequest()) as connection:

    async def read_until_interrupted() -> None:
      async with contextlib.aclosing(connection.receive()) as stream:
        async for response in stream:
          responses.append(response)
          if response.interrupted:
            return

    reading = asyncio.create_task(read_until_interrupted())
    for index in (1, 2):
      await connection.send_realtime(_blob(index))
    await asyncio.wait_for(speaking.wait(), timeout=5)
    await connection.send_realtime(_blob(3))
    await asyncio.wait_for(reading, timeout=5)

  assert responses[-1].interrupted
  assert not any(response.turn_complete for response in responses)

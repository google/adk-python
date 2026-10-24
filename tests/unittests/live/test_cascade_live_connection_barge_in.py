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

"""Tests cascade live barge-in: the user speaking over the agent interrupts it.

Onset detected while a turn is in progress cancels the synthesis still to come
and records only what the user actually heard, whether the onset arrives
through the ingress transform or as a client-sent activity signal.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Optional

from google.adk.live import _cascade_live_connection
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import LiveEgress
from google.adk.live import LiveIngress
from google.adk.live import UserSpeechStarted
from google.adk.live import UserTurnFinished
from google.adk.live._cascade_live_connection import CascadeLiveConnection
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest
from websockets.exceptions import ConnectionClosedOK

from .test_cascade_live_connection import _ANSWER
from .test_cascade_live_connection import _blob
from .test_cascade_live_connection import _MIME_TYPE
from .test_cascade_live_connection import _model_text
from .test_cascade_live_connection import _user
from .test_cascade_live_connection import _UTTERANCE
from .test_cascade_live_connection import FakeStt
from .test_cascade_live_connection import FakeTts
from .test_cascade_live_connection import ScriptedReasoner


class SingleWordGatedTts(LiveEgress):
  """Speaks one chunk, signals, then parks until `cancel` fires.

  Lets a test interrupt precisely mid-utterance: the second chunk is never
  synthesized because the transform is blocked on `cancel`.
  """

  PREFIX = b'<tts>'

  def __init__(self) -> None:
    self.synthesized: list[str] = []
    self.speaking = asyncio.Event()

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for chunk in text:
      if cancel.is_set():
        return
      self.synthesized.append(chunk)
      yield AudioChunk(
          blob=types.Blob(
              data=self.PREFIX + chunk.encode(), mime_type=_MIME_TYPE
          )
      )
      yield AgentSpokenOutput(text=chunk)
      self.speaking.set()
      await cancel.wait()
      return


class SilentGateTts(LiveEgress):
  """Consumes the first chunk, signals, then parks without emitting audio.

  Holds the connection in its pre-audio phase so a test can barge in before
  the agent has spoken anything.
  """

  def __init__(self) -> None:
    self.reached = asyncio.Event()

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for _ in text:
      self.reached.set()
      await cancel.wait()
      return
    if False:  # pragma: no cover -- marks this an async generator.
      yield


class BurstTts(LiveEgress):
  """Synthesizes every sentence up front, then parks until `cancel` fires.

  Models the real egress, which produces audio far faster than real time: all
  `AgentSpokenOutput` text exists well before its audio would have played.
  Each sentence carries a fixed audio duration so a test can place the barge-in
  precisely on the timeline.
  """

  RATE = 24000

  def __init__(
      self,
      ms_per_sentence: float,
      *,
      hold: bool = True,
      fail_on: Optional[str] = None,
      trailing_ms: float = 0,
  ) -> None:
    self.synthesized: list[str] = []
    self.burst_done = asyncio.Event()
    # 16-bit mono: two bytes per sample.
    self._bytes = int(ms_per_sentence / 1000 * self.RATE) * 2
    # Whether to park after bursting (mimicking synthesis still in flight) or
    # end the stream (mimicking synthesis complete, playback ongoing).
    self._hold = hold
    # Raises instead of synthesizing this chunk.
    self._fail_on = fail_on
    # Audio of a next sentence that is still being synthesized: sent after the
    # last sentence, with no `AgentSpokenOutput`.
    self._trailing_bytes = int(trailing_ms / 1000 * self.RATE) * 2

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[AudioChunk | AgentSpokenOutput, None]:
    async for chunk in text:
      if cancel.is_set():
        return
      if chunk == self._fail_on:
        raise RuntimeError(f'TTS failed on {chunk!r}')
      self.synthesized.append(chunk)
      yield AudioChunk(
          blob=types.Blob(
              data=b'\x00' * self._bytes,
              mime_type=f'audio/pcm;rate={self.RATE}',
          )
      )
      yield AgentSpokenOutput(text=chunk)
    if self._trailing_bytes:
      yield AudioChunk(
          blob=types.Blob(
              data=b'\x00' * self._trailing_bytes,
              mime_type=f'audio/pcm;rate={self.RATE}',
          )
      )
    self.burst_done.set()
    if self._hold:
      # Everything is synthesized but "still playing"; hold the stream open.
      await cancel.wait()


class _FakeClock:
  """A settable stand-in for `_cascade_live_connection._now`."""

  def __init__(self, value: float = 0.0) -> None:
    self.value = value

  def __call__(self) -> float:
    return self.value


def _build_partials_and_full(words: list[str]) -> list[LlmResponse]:
  """Streams each word as its own partial, then the repeated aggregate."""
  whole = ''.join(words)
  return [_model_text(word, partial=True) for word in words] + [
      _model_text(whole, partial=False)
  ]


class BargingStt(LiveIngress):
  """Reads a sentinel blob as speech onset, any other blob as an utterance.

  A blob that starts with the sentinel and has more text yields the onset and
  then that text as an utterance, back to back.
  """

  BARGE = '<barge>'

  def __init__(self) -> None:
    self.heard: list[bytes] = []

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[UserSpeechStarted | UserTurnFinished, None]:
    async for blob in audio:
      self.heard.append(blob.data)
      text = blob.data.decode()
      if text.startswith(self.BARGE):
        yield UserSpeechStarted()
        text = text[len(self.BARGE) :]
      if text:
        yield UserTurnFinished(text=text)


async def _open(
    reasoner: BaseLlm, stt: LiveIngress, tts: LiveEgress
) -> CascadeLiveConnection:
  """A connection driven directly, the way the partial-content test does."""
  return CascadeLiveConnection(
      LlmRequest(),
      reasoner,
      stt,
      tts,
  )


def _collect(
    connection: CascadeLiveConnection,
) -> tuple[list[LlmResponse], asyncio.Task[None]]:
  """Drains `receive()` into a list in the background, as the flow would."""
  received: list[LlmResponse] = []

  async def _read() -> None:
    try:
      async with aclosing(connection.receive()) as agen:
        async for response in agen:
          received.append(response)
    except ConnectionClosedOK:
      pass

  return received, asyncio.create_task(_read())


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
  """Spins until `predicate()` holds, so a poll can't hang the suite."""

  async def _spin() -> None:
    while not predicate():
      await asyncio.sleep(0.01)

  await asyncio.wait_for(_spin(), timeout)


def _fragments(text: str, whole: str) -> list[LlmResponse]:
  """Two streamed fragments plus the aggregate the contract repeats."""
  return [
      _model_text(text, partial=True),
      _model_text(whole[len(text) :], partial=True),
      _model_text(whole, partial=False),
  ]


@pytest.mark.asyncio
async def test_barge_in_interrupts_and_keeps_only_the_heard_prefix(monkeypatch):
  """Onset mid-utterance stops synthesis and records what was actually heard.

  The reasoner generates "Hello world." but the user speaks after "Hello ";
  the remainder is never synthesized and history keeps the spoken prefix, not
  the full generation. Onset arrives through the ingress transform, the way a
  VAD-driven STT reports it. A fake clock places the barge-in well past the
  short chunk's playback so "Hello " counts as fully heard, not a fraction.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  stt, tts = BargingStt(), SingleWordGatedTts()
  reasoner = ScriptedReasoner(turns=[_fragments('Hello ', 'Hello world.')])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  # Speak an utterance and let the agent get its first chunk out.
  await connection.send_realtime(_blob(_UTTERANCE))
  await asyncio.wait_for(tts.speaking.wait(), timeout=5.0)

  # The user speaks over the agent, a full second into the chunk's playback.
  clock.value = 1.0
  await connection.send_realtime(_blob(BargingStt.BARGE))

  await _wait_until(lambda: any(r.interrupted for r in received))
  assert connection._cancel.is_set()

  # The remainder was never synthesized, and history holds only the prefix.
  assert tts.synthesized == ['Hello ']
  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='Hello ')]
  )

  # The final transcription matches history and precedes the interrupt.
  finals = [
      (i, r.output_transcription.text)
      for i, r in enumerate(received)
      if r.output_transcription and r.output_transcription.finished
  ]
  assert [text for _, text in finals] == ['Hello ']
  assert finals[0][0] < next(i for i, r in enumerate(received) if r.interrupted)

  # No turn_complete follows the interrupt.
  assert not any(r.turn_complete for r in received)

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_user_turn_right_after_barge_in_is_answered(monkeypatch):
  """The user turn that follows a barge-in starts a new turn.

  The ingress reports the onset and the finished utterance back to back, so the
  utterance arrives while the interrupted turn is still shutting down. The next
  request must see the heard prefix followed by the new user turn.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  stt, tts = BargingStt(), SingleWordGatedTts()
  reasoner = ScriptedReasoner(
      turns=[_fragments('Hello ', 'Hello world.'), [_model_text(_ANSWER)]]
  )
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_realtime(_blob(_UTTERANCE))
  await asyncio.wait_for(tts.speaking.wait(), timeout=5.0)

  clock.value = 1.0
  await connection.send_realtime(_blob(BargingStt.BARGE + 'and tomorrow'))

  await _wait_until(lambda: len(reasoner.requests) == 2)
  assert reasoner.requests[1].contents == [
      _user(_UTTERANCE),
      types.Content(role='model', parts=[types.Part(text='Hello ')]),
      _user('and tomorrow'),
  ]
  await _wait_until(lambda: tts.synthesized == ['Hello ', _ANSWER])

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_client_activity_start_triggers_barge_in(monkeypatch):
  """A manual-VAD client interrupts by sending `ActivityStart`."""
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  stt, tts = FakeStt(), SingleWordGatedTts()
  reasoner = ScriptedReasoner(turns=[_fragments('Hello ', 'Hello world.')])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.speaking.wait(), timeout=5.0)

  clock.value = 1.0
  await connection.send_realtime(types.ActivityStart())

  await _wait_until(lambda: any(r.interrupted for r in received))
  assert tts.synthesized == ['Hello ']

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_speech_onset_while_idle_is_not_a_barge_in():
  """Onset with no turn in progress is a turn start, not an interruption."""
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  # Nothing is speaking; onset must not interrupt or cancel anything.
  await connection.send_realtime(types.ActivityStart())

  assert not connection._cancel.is_set()
  assert not any(r.interrupted for r in received)

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_before_any_audio_records_no_speech():
  """Interrupting while the agent is still thinking leaves history unchanged.

  Nothing was spoken, so `AgentSpokenOutput` -- the only writer of assistant
  text -- has nothing to record: history keeps just the user turn.
  """
  stt, tts = FakeStt(), SilentGateTts()
  reasoner = ScriptedReasoner(turns=[[_model_text(_ANSWER)]])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.reached.wait(), timeout=5.0)

  await connection.send_realtime(types.ActivityStart())

  await _wait_until(lambda: any(r.interrupted for r in received))
  assert connection._contents == [_user(_UTTERANCE)]
  assert not any(r.output_transcription for r in received)

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_records_only_played_prefix_not_synthesized_burst(
    monkeypatch,
):
  """Barge-in during synthesis records the heard prefix, not the full burst.

  This is the barge-*during-synthesis* case: the egress holds its stream open
  after bursting, so synthesis is still in progress when the barge lands and the
  turn stays open through the egress coroutine itself. Its counterpart
  test_barge_in_after_synthesis_during_playback_still_truncates ends the stream
  first, so there the turn is held open only by the deferred-completion playback
  wait. Same inputs, same truncated result; different mechanism keeps the turn
  open.

  The egress synthesizes five one-second sentences up front. A fake clock puts
  the barge-in 2.5 s into playback: the sentences starting at 0 s and 1 s played
  in full, and the one starting at 2 s is only half heard. Under the whole-word
  "finished" policy that half-played word had not completed, so it is dropped
  and history keeps just the two fully-played sentences.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  words = ['one ', 'two ', 'three ', 'four ', 'five ']
  stt, tts = FakeStt(), BurstTts(ms_per_sentence=1000)
  reasoner = ScriptedReasoner(turns=[_build_partials_and_full(words)])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  # Playback starts at t=0; let the whole burst synthesize.
  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.burst_done.wait(), timeout=5.0)

  # 2.5 s of playback elapsed: 'one' and 'two' fully played; 'three' is only
  # half heard, so its single word had not finished.
  clock.value = 2.5
  await connection.send_realtime(types.ActivityStart())
  await _wait_until(lambda: any(r.interrupted for r in received))

  # All five were synthesized, but only the fully-played prefix is recorded.
  assert tts.synthesized == words
  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='one two ')]
  )

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_after_synthesis_during_playback_still_truncates(
    monkeypatch,
):
  """Barge-in after synthesis, mid-playback, still records the heard prefix.

  The counterpart to the during-synthesis case above
  (test_barge_in_records_only_played_prefix_not_synthesized_burst): there the
  egress holds its stream open, so synthesis is still running when the barge
  lands; here it bursts every sentence and then ends the stream, so synthesis is
  complete and the turn is held open solely by the deferred-completion playback
  wait. Both barge halfway through 'three' and truncate to the same heard
  prefix.

  Because the turn does not complete the instant synthesis ends, the barge still
  lands while the agent is speaking: it records the heard prefix and emits no
  turn_complete, rather than arriving against a fully-recorded, already-completed
  turn.

  Each sentence is 60 s long. The playback wait runs in real time, so the long
  audio keeps the turn open until the barge-in, even on a slow machine.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  words = ['one ', 'two ', 'three ', 'four ', 'five ']
  stt, tts = FakeStt(), BurstTts(ms_per_sentence=60_000, hold=False)
  reasoner = ScriptedReasoner(turns=[_build_partials_and_full(words)])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  # The whole burst is synthesized and the egress stream has ended; the turn is
  # now held open only by the deferred-completion playback wait.
  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.burst_done.wait(), timeout=5.0)

  # 150 s into playback: 'one' and 'two' fully played, 'three' still mid-word.
  clock.value = 150.0
  await connection.send_realtime(types.ActivityStart())
  await _wait_until(lambda: any(r.interrupted for r in received))

  # The barge truncated to the heard prefix; the turn never completed.
  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='one two ')]
  )
  assert not any(r.turn_complete for r in received)

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_during_playback_after_a_tts_failure_interrupts(
    monkeypatch, caplog
):
  """A failed turn stays open while its audio plays, so barge-in still lands.

  The egress bursts three sentences, then fails. That audio keeps playing on the
  client, so the turn must not complete before it would end: a barge-in halfway
  through 'two' interrupts and keeps the heard prefix, instead of arriving after
  turn_complete as a new turn.

  Each sentence is 60 s long. The playback wait runs in real time, so the long
  audio keeps the turn open until the barge-in, even on a slow machine.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  words = ['one ', 'two ', 'three ', 'four ']
  stt = FakeStt()
  tts = BurstTts(ms_per_sentence=60_000, hold=False, fail_on='four ')
  reasoner = ScriptedReasoner(turns=[_build_partials_and_full(words)])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  # The turn's only await after this log is the playback wait.
  await _wait_until(lambda: 'Cascade reasoning turn failed.' in caplog.text)
  assert not any(r.turn_complete for r in received)

  # 90 s into playback: 'one' fully played, 'two' still mid-word.
  clock.value = 90.0
  await connection.send_realtime(types.ActivityStart())
  await _wait_until(lambda: any(r.interrupted for r in received))

  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='one ')]
  )
  assert not any(r.turn_complete or r.error_code for r in received)

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_truncates_straddled_sentence_to_finished_words(
    monkeypatch,
):
  """The sentence the interrupt lands in keeps only its fully-played words.

  One three-second sentence, "alpha bravo charlie", is synthesized up front.
  The barge-in lands 2 s in (two thirds of the way), and audio time is
  apportioned across the sentence by character length. "alpha" and "bravo" fall
  inside that budget; "charlie" is still playing, so it is dropped.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  stt, tts = FakeStt(), BurstTts(ms_per_sentence=3000)
  reasoner = ScriptedReasoner(
      turns=[_build_partials_and_full(['alpha bravo charlie'])]
  )
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.burst_done.wait(), timeout=5.0)

  # 2 s of a 3 s sentence: char budget covers "alpha bravo" but not "charlie".
  clock.value = 2.0
  await connection.send_realtime(types.ActivityStart())
  await _wait_until(lambda: any(r.interrupted for r in received))

  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='alpha bravo')]
  )

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_barge_in_while_next_sentence_streams_keeps_the_finished_one(
    monkeypatch,
):
  """A finished sentence is kept whole while the next one is still streaming.

  The egress sends a sentence's audio before its `AgentSpokenOutput`, so the
  next sentence can have audio in the turn but no text yet. Here "Hello there. "
  is 1 s, followed by 0.5 s of the next sentence. A barge-in at 1.25 s lands in
  the next sentence, so "Hello there. " played in full and must not be cut.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  stt = FakeStt()
  tts = BurstTts(ms_per_sentence=1000, trailing_ms=500)
  reasoner = ScriptedReasoner(turns=[[_model_text('Hello there. ')]])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  await asyncio.wait_for(tts.burst_done.wait(), timeout=5.0)

  clock.value = 1.25
  await connection.send_realtime(types.ActivityStart())
  await _wait_until(lambda: any(r.interrupted for r in received))

  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='Hello there. ')]
  )
  assert [
      r.output_transcription.text
      for r in received
      if r.output_transcription and r.output_transcription.finished
  ] == ['Hello there. ']

  await connection.close()
  await reader


@pytest.mark.asyncio
async def test_completed_turn_records_full_spoken_text(monkeypatch):
  """A turn that finishes normally records everything, regardless of the clock.

  The heard-prefix truncation is a barge-in behavior only; a completed turn was
  heard in full even though the same burst outran the playback clock.
  """
  clock = _FakeClock(0.0)
  monkeypatch.setattr(_cascade_live_connection, '_now', clock)

  words = ['one ', 'two ', 'three ']
  stt, tts = FakeStt(), FakeTts()
  reasoner = ScriptedReasoner(turns=[_build_partials_and_full(words)])
  connection = await _open(reasoner, stt, tts)
  received, reader = _collect(connection)

  await connection.send_content(_user(_UTTERANCE))
  await _wait_until(lambda: any(r.turn_complete for r in received))

  assert connection._contents[-1] == types.Content(
      role='model', parts=[types.Part(text='one two three ')]
  )

  await connection.close()
  await reader

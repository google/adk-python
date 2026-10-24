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

"""A `BaseLlmConnection` adapting STT ingress, text reasoning, and TTS egress into a live connection."""

from __future__ import annotations

import asyncio
import enum
import logging
import re
import time
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from ..features import experimental
from ..features import FeatureName
from ..models.base_llm_connection import BaseLlmConnection
from ..models.base_llm_connection import RealtimeInput
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..utils.context_utils import Aclosing
from ._cascade_live_events import AgentSpokenOutput
from ._cascade_live_events import AudioChunk
from ._cascade_live_events import IngressEvent
from ._cascade_live_events import PartialTranscript
from ._cascade_live_events import UserSpeechStarted
from ._cascade_live_events import UserTurnFinished

if TYPE_CHECKING:
  from ..models.base_llm import BaseLlm
  from ._transforms import LiveEgress
  from ._transforms import LiveIngress

logger = logging.getLogger('google_adk.' + __name__)

# Sentinel pushed onto the outbound queue to wake `receive()` on close.
_CLOSED = object()

# Close frame emitted when terminating the live connection.
_CLOSE_FRAME = Close(1000, 'CascadeLive connection closed.')

# Assumed PCM sample rate when a blob mime type omits `rate=`.
_DEFAULT_PCM_RATE = 24000

# Clock for playback timing. Tests patch this instead of `time.perf_counter`.
_now = time.perf_counter


def _pcm_duration_ms(blob: types.Blob) -> float:
  """Playback duration in ms of a 16-bit mono PCM blob."""
  rate = _DEFAULT_PCM_RATE
  for token in (blob.mime_type or '').split(';'):
    token = token.strip()
    if token.startswith('rate='):
      try:
        rate = int(token[len('rate=') :])
      except ValueError:
        pass
      break
  return len(blob.data or b'') / 2 / rate * 1000.0


def _played_prefix(text: str, frac: float) -> str:
  """The whole-word prefix of `text` whose audio had played by fraction `frac`.

  Estimates audio playback progress by character count. Includes only fully
  played words, dropping any word interrupted in progress. Preserves original
  punctuation and spacing.
  """
  budget = max(0.0, min(1.0, frac)) * len(text)
  cut = 0
  for match in re.finditer(r'\S+', text):
    if match.end() <= budget:
      cut = match.end()
    else:
      break
  return text[:cut]


class _State(enum.Enum):
  """Turn-taking phase. Decides whether speech onset is a barge-in.

  Onset while ``LISTENING`` starts a new turn; onset while ``THINKING`` or
  ``AGENT_SPEAKING`` interrupts the turn in progress.
  """

  LISTENING = 'listening'
  THINKING = 'thinking'
  AGENT_SPEAKING = 'agent_speaking'


@experimental(FeatureName.CASCADE_LIVE)
class CascadeLiveConnection(BaseLlmConnection):
  """Live connection bridging STT ingress, text reasoning, and TTS egress.

  Audio input from the client is processed by the STT transform. When endpointing
  occurs, the recognized text triggers a reasoning turn on the underlying LLM.
  Streaming text responses are synthesized by the TTS transform into audio chunks
  and output transcriptions delivered to the client.
  """

  def __init__(
      self,
      llm_request: LlmRequest,
      llm: BaseLlm,
      stt: LiveIngress,
      tts: LiveEgress,
  ) -> None:
    self._llm_request = llm_request

    self._llm = llm
    self._stt = stt
    self._tts = tts

    # Replayed to the LLM on every turn. One rule governs what lands
    # here: only *completed* content, and assistant text only as the egress
    # reported it (`AgentSpokenOutput`), never as the LLM generated it. See
    # `_run_turn`.
    self._contents: list[types.Content] = list(llm_request.contents or [])

    # Client audio frames consumed by the STT transform; `None` ends the
    # current audio segment.
    self._audio_in: asyncio.Queue[Optional[types.Blob]] = asyncio.Queue()
    # Non-audio MIME types already logged as dropped.
    self._dropped_mime_types: set[Optional[str]] = set()
    # Responses yielded by `receive()`; `_CLOSED` ends the stream.
    self._out: asyncio.Queue[Any] = asyncio.Queue()

    # Signals cancellation of in-flight synthesis on barge-in or close.
    self._cancel = asyncio.Event()

    # Turn phase; drives barge-in detection.
    self._state = _State.LISTENING

    # Spoken sentences in this turn as (audio_start_ms, text) offsets.
    self._spoken_marks: list[tuple[float, str]] = []
    # Total synthesized audio duration for the current turn (in ms).
    self._turn_audio_ms: float = 0.0
    # Start offset on the audio timeline for the sentence currently being
    # synthesized (in ms).
    self._sentence_start_ms: float = 0.0
    # Wall-clock timestamp (`_now()`) when audio playback began, or None.
    self._playback_start: Optional[float] = None

    # Set by `close()`; stops new turns and drops further emitted responses.
    self._closed = False
    # Ingress failure that closed the connection; re-raised by `receive()`.
    self._error: Optional[Exception] = None
    # Function call from the LLM, emitted after the turn's speech ends.
    self._pending_function_call: Optional[LlmResponse] = None
    # Set when new input needs a turn; `_run_turns` starts it once the current
    # turn has ended.
    self._input_ready = asyncio.Event()
    # Runs `_run_turns`, started by the first input.
    self._turn_loop: Optional[asyncio.Task[None]] = None
    # Runs `_run_turn` for the current turn.
    self._turn: Optional[asyncio.Task[None]] = None
    # Runs `_pump_ingress` for the connection's lifetime.
    self._ingress_task = asyncio.create_task(self._pump_ingress())

  # ------------------------------------------------------------------
  # Inbound: `send_to_model` calls these.
  # ------------------------------------------------------------------

  async def send_history(self, history: list[types.Content]) -> None:
    """Seeds the conversation. Does not itself trigger reasoning."""
    self._contents = list(history)

  async def send_content(self, content: types.Content) -> None:
    """Appends a turn-completing content and starts reasoning.

    Carries both real user text and tool results: after a tool runs,
    ``run_live`` routes the function response back through here, which is
    what closes the tool loop.
    """
    self._contents.append(content)
    self._start_reasoning()

  async def _send_content(
      self, content: types.Content, *, partial: bool = False
  ) -> None:
    """Appends non-partial content and triggers reasoning."""
    if partial:
      return
    await self.send_content(content)

  async def send_realtime(self, blob: RealtimeInput) -> None:
    """Accepts audio frames and realtime control messages from the client."""
    # TODO: Keep the STT session open across `ActivityEnd` with an in-band
    # commit, and close it only on `audio_stream_end` or when idle, to avoid a
    # reconnect per turn.
    if isinstance(blob, types.ActivityStart):
      await self._on_speech_started(UserSpeechStarted())
    elif isinstance(blob, types.ActivityEnd):
      # End of the user's turn; finalize the current audio segment.
      await self._audio_in.put(None)
    elif isinstance(blob, types.LiveClientRealtimeInput):
      if blob.audio_stream_end:
        # Client microphone closed; finalize the current audio segment.
        await self._audio_in.put(None)
    elif (blob.mime_type or '').startswith('audio/'):
      await self._audio_in.put(blob)
    elif blob.mime_type not in self._dropped_mime_types:
      # The STT accepts only audio; drop video frames and other blobs.
      self._dropped_mime_types.add(blob.mime_type)
      logger.warning(
          'CascadeLive accepts only audio input; dropping %s blobs.',
          blob.mime_type,
      )

  # ------------------------------------------------------------------
  # Outbound: `receive_from_model` drives this.
  # ------------------------------------------------------------------

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Yields responses until the connection is closed.

    Raises the STT error if ingress failed, so the failure is not reported as
    a normal close.
    """
    while True:
      item = await self._out.get()
      if item is _CLOSED:
        if self._error is not None:
          raise self._error
        raise ConnectionClosedOK(None, _CLOSE_FRAME)
      yield item

  async def close(self) -> None:
    """Tears down the ingress pump and any in-flight reasoning turn."""
    if self._closed:
      return
    self._closed = True

    self._cancel.set()
    await self._audio_in.put(None)

    for task in (self._turn, self._turn_loop, self._ingress_task):
      if task is not None and not task.done():
        task.cancel()

    self._out.put_nowait(_CLOSED)

  # ------------------------------------------------------------------
  # Ingress.
  # ------------------------------------------------------------------

  async def _drain_audio(
      self, first: types.Blob
  ) -> AsyncGenerator[types.Blob, None]:
    """Yields one audio segment: `first`, then frames until the next `None`."""
    yield first
    while True:
      blob = await self._audio_in.get()
      if blob is None:
        return
      yield blob

  async def _pump_ingress(self) -> None:
    """Runs one STT session per audio segment until the connection closes.

    A segment ends on `ActivityEnd` or `audio_stream_end`, which lets the STT
    transform finalize the utterance. The next audio frame opens a new session,
    so the client can pause and resume speaking.
    """
    try:
      while not self._closed:
        first = await self._audio_in.get()
        if first is None:
          # Segment ended with no audio, or the connection is closing.
          continue
        async for ingress_event in self._stt(self._drain_audio(first)):
          await self._on_ingress_event(ingress_event)
    except asyncio.CancelledError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      logger.exception('Cascade ingress failed; closing the live connection.')
      self._error = e
      await self.close()

  async def _on_ingress_event(self, ingress_event: IngressEvent) -> None:
    """Dispatches one event from the STT transform."""
    if isinstance(ingress_event, PartialTranscript):
      # Forwarded as the full hypothesis so far, replacing the previous one.
      # Unlike `GeminiLlmConnection`, whose partials are deltas.
      self._emit(
          LlmResponse(
              input_transcription=types.Transcription(
                  text=ingress_event.text, finished=False
              ),
              partial=True,
          )
      )
    elif isinstance(ingress_event, UserSpeechStarted):
      await self._on_speech_started(ingress_event)
    elif isinstance(ingress_event, UserTurnFinished):
      await self._on_user_turn_finished(ingress_event)
    else:
      logger.warning('Ignoring unknown ingress event: %r', ingress_event)

  async def _on_speech_started(self, event: UserSpeechStarted) -> None:
    """Interrupts the turn in progress when the user speaks over the agent.

    Onset while idle is a normal turn start and is ignored here; the turn
    begins when ingress emits `UserTurnFinished`.
    """
    del event
    if self._state not in (_State.THINKING, _State.AGENT_SPEAKING):
      return

    # Stop synthesis and tear down the in-flight turn.
    self._cancel.set()
    if self._turn is not None and not self._turn.done():
      self._turn.cancel()
    self._input_ready.clear()
    self._pending_function_call = None

    # Record only the prefix the user actually heard, then report the interrupt,
    # which triggers the framework's audio-cache flush.
    self._record_spoken(heard_only=True)
    self._state = _State.LISTENING
    self._emit(LlmResponse(interrupted=True))

  async def _on_user_turn_finished(self, event: UserTurnFinished) -> None:
    """Emits user transcription and triggers reasoning."""
    self._emit(
        LlmResponse(
            input_transcription=types.Transcription(
                text=event.text, finished=True
            ),
            partial=False,
        )
    )
    # TODO: If a tool call is waiting for its response, this user turn lands
    # between the call and its response in `_contents`, which breaks this and
    # later requests. Hold user turns until the response arrives, except for
    # long-running tools that return no immediate response.
    self._contents.append(
        types.Content(role='user', parts=[types.Part(text=event.text)])
    )
    self._start_reasoning()

  # ------------------------------------------------------------------
  # Reasoning + egress.
  # ------------------------------------------------------------------

  def _start_reasoning(self) -> None:
    """Requests a turn. It starts once the current turn, if any, has ended."""
    if self._closed:
      return
    self._input_ready.set()
    if self._turn_loop is None:
      self._turn_loop = asyncio.create_task(self._run_turns())

  async def _run_turns(self) -> None:
    """Runs one turn at a time, each after the previous one has ended."""
    while True:
      await self._input_ready.wait()
      self._input_ready.clear()
      self._cancel.clear()
      self._state = _State.THINKING
      self._turn = asyncio.create_task(self._run_turn())
      # Also returns when a barge-in cancels the turn, unlike awaiting the task.
      await asyncio.wait([self._turn])

  async def _run_turn(self) -> None:
    """Runs one reasoning turn and synthesizes its output."""
    # If the previous turn failed, drop its function call so the tool does not
    # run. That turn already ended with an error, and the user may not have
    # heard what the agent said before the call. The model can call the tool
    # again on a later turn.
    self._pending_function_call = None
    try:
      self._spoken_marks = []
      self._turn_audio_ms = 0.0
      self._sentence_start_ms = 0.0
      self._playback_start = None

      # Closing both streams on exit means an early exit (barge-in cancel or a
      # TTS that returns early) also closes the LLM stream.
      async with (
          Aclosing(self._stream_text()) as text,
          Aclosing(self._tts(text, cancel=self._cancel)) as events,
      ):
        async for event in events:
          if isinstance(event, AudioChunk):
            self._state = _State.AGENT_SPEAKING
            # Stamp playback start on the first chunk; advance the audio clock.
            if self._playback_start is None:
              self._playback_start = _now()
            self._turn_audio_ms += _pcm_duration_ms(event.blob)
            self._emit(
                LlmResponse(
                    content=types.Content(
                        role='model',
                        parts=[types.Part(inline_data=event.blob)],
                    )
                )
            )
          elif isinstance(event, AgentSpokenOutput):
            # The egress emits a sentence's audio before its
            # `AgentSpokenOutput`, so this sentence spans
            # [_sentence_start_ms, _turn_audio_ms) on the timeline. Tag it with
            # its start and advance to the next.
            self._spoken_marks.append((self._sentence_start_ms, event.text))
            self._sentence_start_ms = self._turn_audio_ms
            self._emit(
                LlmResponse(
                    output_transcription=types.Transcription(
                        text=event.text, finished=False
                    ),
                    partial=True,
                )
            )
          else:
            logger.warning('Ignoring unknown egress event: %r', event)

      # A barge-in during synthesis already recorded the heard prefix and
      # reported the interrupt; a torn-down turn's history is discarded.
      if self._cancel.is_set():
        return

      # Synthesis finished, but the egress bursts audio far ahead of playback,
      # so the turn is not over until that audio would finish playing. Hold it
      # open until then (interruptible) so a barge-in during playback still
      # lands while AGENT_SPEAKING and truncates to the heard prefix, rather
      # than arriving after turn_complete as a fresh turn.
      await self._await_playback()
      if self._cancel.is_set():
        return

      self._record_spoken(heard_only=False)

      if self._pending_function_call is not None:
        response, self._pending_function_call = (
            self._pending_function_call,
            None,
        )
        # Keep only function call parts: text reaches history and the client
        # only as spoken output, and thoughts never do.
        function_calls = types.Content(
            role='model',
            parts=[
                part for part in response.content.parts if part.function_call
            ],
        )
        self._contents.append(function_calls)
        # Do not complete the turn: it resumes when the tool response arrives
        # back through `_send_content`.
        self._state = _State.LISTENING
        self._emit(response.model_copy(update={'content': function_calls}))
        return

      self._state = _State.LISTENING
      self._emit(LlmResponse(turn_complete=True))
    except asyncio.CancelledError:
      raise
    except Exception:  # pylint: disable=broad-except
      logger.exception('Cascade reasoning turn failed.')
      # Audio sent before the failure is still playing. As on success, hold the
      # turn open until it ends so a barge-in during it still interrupts.
      await self._await_playback()
      # A barge-in or close() already took over the turn.
      if self._cancel.is_set():
        return
      # Keep the sentences the user already heard.
      self._record_spoken(heard_only=False)
      self._state = _State.LISTENING
      self._emit(
          LlmResponse(
              error_code='CASCADE_TURN_FAILED',
              error_message='Cascaded reasoning turn encountered an error.',
              turn_complete=True,
          )
      )

  async def _await_playback(self) -> None:
    """Blocks until this turn's synthesized audio would finish playing.

    The egress bursts audio far faster than real time, so synthesis completes
    long before playback does. Deferring turn completion to the end of playback
    (as native live does) keeps the turn in `AGENT_SPEAKING` for its whole
    audible span, so a barge-in during playback is detected instead of arriving
    after `turn_complete` as a fresh turn. Playback end is estimated from wall
    time since the first chunk; the client's true position is not observable.
    Returns early if `_cancel` fires (barge-in or teardown).
    """
    if self._playback_start is None:
      return
    remaining = (self._playback_start + self._turn_audio_ms / 1000.0) - _now()
    if remaining <= 0:
      return
    try:
      await asyncio.wait_for(self._cancel.wait(), timeout=remaining)
    except asyncio.TimeoutError:
      pass  # Playback finished without a barge-in.

  def _record_spoken(self, *, heard_only: bool) -> None:
    """Records this turn's spoken text and emits its final transcription.

    Appends to history at most once per turn.

    On a completed turn (`heard_only=False`) the whole turn was heard. On a
    barge-in (`heard_only=True`) the egress bursts audio faster than real time,
    so text was synthesized well ahead of playback; keep only what had played by
    the interrupt. Playback position is estimated from wall time since the first
    chunk (the client's true position is not observable) - safe here because
    bursting never lets the buffer underrun. Fully-played sentences are kept
    whole; the sentence the interrupt lands in is truncated to the words whose
    audio finished, dropping the word in progress.
    """
    if not heard_only:
      texts = [text for _, text in self._spoken_marks]
    elif self._playback_start is None:
      texts = []  # Interrupted before any audio played: nothing was heard.
    else:
      texts = self._extract_heard_sentences(self._playback_start)
    self._spoken_marks = []
    if not texts:
      return

    spoken_text = ''.join(texts)
    self._contents.append(
        types.Content(role='model', parts=[types.Part(text=spoken_text)])
    )
    # Final transcription for the turn, matching history; the per-sentence ones
    # are partial.
    self._emit(
        LlmResponse(
            output_transcription=types.Transcription(
                text=spoken_text, finished=True
            ),
            partial=False,
        )
    )

  def _extract_heard_sentences(self, playback_start: float) -> list[str]:
    """Returns the text the user heard before the barge-in.

    `_spoken_marks` has one `(start_ms, text)` entry per finished sentence,
    where `start_ms` is when that sentence's audio starts in the turn. For
    example, `[(0, 'Hi. '), (800, 'How are you? ')]`.

    Each sentence starts right after the previous one ends, so the next
    sentence's `start_ms` is the end of the current one. The last sentence
    ends at `_sentence_start_ms`, which records where the next sentence
    would start.
    """
    elapsed_ms = (_now() - playback_start) * 1000.0
    texts = []
    for i, (start_ms, text) in enumerate(self._spoken_marks):
      if start_ms >= elapsed_ms:
        break  # This sentence had not started; nor has any later one.
      end_ms = (
          self._spoken_marks[i + 1][0]
          if i + 1 < len(self._spoken_marks)
          else self._sentence_start_ms
      )
      if end_ms <= elapsed_ms:
        texts.append(text)  # Audio fully played.
      else:
        # The interrupt landed inside this sentence; keep the words whose
        # audio finished, then stop.
        frac = (elapsed_ms - start_ms) / (end_ms - start_ms)
        prefix = _played_prefix(text, frac)
        if prefix:
          texts.append(prefix)
        break
    return texts

  async def _stream_text(self) -> AsyncGenerator[str, None]:
    """Streams LLM text deltas to TTS, diverting function calls."""
    request = self._build_request()
    streamed = False

    async with Aclosing(
        self._llm.generate_content_async(request, stream=True)
    ) as responses:
      async for response in responses:
        if response.error_code:
          self._emit(response)
          return
        if not response.content or not response.content.parts:
          continue
        if any(part.function_call for part in response.content.parts):
          # Wait for the complete aggregate response before emitting function calls.
          if response.partial:
            continue
          # Stashed before yielding, so the call survives a TTS that stops
          # reading early, for `_run_turn` to emit.
          self._pending_function_call = response
          # Synthesize any text in the aggregate if it was not streamed earlier.
          if not streamed:
            for part in response.content.parts:
              if part.text and not part.thought:
                yield part.text
          return
        if streamed and not response.partial:
          # The turn's final response repeats every fragment already spoken.
          continue
        for part in response.content.parts:
          # Skip model thoughts; only synthesize spoken text.
          if part.text and not part.thought:
            streamed = True
            yield part.text

  def _build_request(self) -> LlmRequest:
    """Rebuilds the text request from the live request plus accumulated turns."""
    request = self._llm_request.model_copy(deep=False)
    request.model = self._llm.model
    request.contents = list(self._contents)
    return request

  # ------------------------------------------------------------------

  def _emit(self, response: LlmResponse) -> None:
    if not self._closed:
      self._out.put_nowait(response)

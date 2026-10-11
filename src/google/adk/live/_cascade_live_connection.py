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
import logging
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

    # Set by `close()`; stops new turns and drops further emitted responses.
    self._closed = False
    # Ingress failure that closed the connection; re-raised by `receive()`.
    self._error: Optional[Exception] = None
    # Function call from the LLM, emitted after the turn's speech ends.
    self._pending_function_call: Optional[LlmResponse] = None
    # Set when new input arrives mid-turn; reruns reasoning after the turn.
    self._rerun_requested = False
    # Runs `_drive_turns`: reasoning plus TTS for one or more turns.
    self._reasoning_task: Optional[asyncio.Task[None]] = None
    # Runs `_pump_ingress` for the connection's lifetime.
    self._ingress_task = asyncio.create_task(self._pump_ingress())

  # ------------------------------------------------------------------
  # Inbound: the flow's `_send_to_model` calls these.
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
  # Outbound: the flow's `_receive_from_model` drives this.
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

    for task in (self._reasoning_task, self._ingress_task):
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
    """Handles speech onset detection."""
    del event
    # TODO: Add barge-in handling when agent is speaking.

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
    if self._closed:
      return
    if self._reasoning_task is not None and not self._reasoning_task.done():
      # If a turn is currently running, queue a rerun for when it finishes.
      self._rerun_requested = True
      return
    self._cancel.clear()
    self._reasoning_task = asyncio.create_task(self._drive_turns())

  async def _drive_turns(self) -> None:
    """Executes reasoning turns while rerun triggers are queued."""
    while True:
      await self._run_turn()
      if self._closed or not self._rerun_requested:
        return
      self._rerun_requested = False
      self._cancel.clear()

  async def _run_turn(self) -> None:
    """Runs one reasoning turn and synthesizes its output."""
    # If the previous turn failed, drop its function call so the tool does not
    # run. That turn already ended with an error, and the user may not have
    # heard what the agent said before the call. The model can call the tool
    # again on a later turn.
    self._pending_function_call = None
    spoken: list[str] = []
    try:
      # Closing both streams on exit means a TTS that returns early also
      # closes the LLM stream instead of leaving it suspended.
      async with (
          Aclosing(self._stream_text()) as text,
          Aclosing(self._tts(text, cancel=self._cancel)) as events,
      ):
        async for event in events:
          if isinstance(event, AudioChunk):
            self._emit(
                LlmResponse(
                    content=types.Content(
                        role='model',
                        parts=[types.Part(inline_data=event.blob)],
                    )
                )
            )
          elif isinstance(event, AgentSpokenOutput):
            # Record assistant spoken text.
            spoken.append(event.text)
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

      self._record_spoken(spoken)

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
        self._emit(response.model_copy(update={'content': function_calls}))
        return

      self._emit(LlmResponse(turn_complete=True))
    except asyncio.CancelledError:
      raise
    except Exception:  # pylint: disable=broad-except
      logger.exception('Cascade reasoning turn failed.')
      # Keep the sentences the user already heard.
      self._record_spoken(spoken)
      self._emit(
          LlmResponse(
              error_code='CASCADE_TURN_FAILED',
              error_message='Cascaded reasoning turn encountered an error.',
              turn_complete=True,
          )
      )

  def _record_spoken(self, spoken: list[str]) -> None:
    """Records this turn's spoken text and emits its final transcription."""
    if not spoken:
      return
    spoken_text = ''.join(spoken)
    self._contents.append(
        types.Content(role='model', parts=[types.Part(text=spoken_text)])
    )
    # Final transcription for the turn; the per-sentence ones are partial.
    self._emit(
        LlmResponse(
            output_transcription=types.Transcription(
                text=spoken_text, finished=True
            ),
            partial=False,
        )
    )

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

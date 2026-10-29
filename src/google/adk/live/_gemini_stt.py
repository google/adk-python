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

"""A `LiveIngress` backed by Gemini streaming transcription."""

from __future__ import annotations

import asyncio
import collections
import logging
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types

from ..features import experimental
from ..features import FeatureName
from ._cascade_live_events import IngressEvent
from ._cascade_live_events import PartialTranscript
from ._cascade_live_events import UserSpeechStarted
from ._cascade_live_events import UserTurnFinished
from ._transforms import LiveIngress

if TYPE_CHECKING:
  from google.genai import Client
  from google.genai.live import AsyncSession

  from ..models.google_llm import Gemini

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_MODEL = 'gemini-3.5-transcribe-live'

# Silence that ends the user's turn, unless `activity_detection` sets it. The
# server's own default is about a second.
_DEFAULT_SILENCE_DURATION_MS = 500

_DEFAULT_SAMPLE_RATE = 16000

# 16-bit mono, so two bytes per sample.
_BYTES_PER_SAMPLE = 2

# Once the audio segment ends: the longest wait for the transcript of the
# speech in progress, and how long the server must stay quiet before no
# speech is assumed to be left.
_FLUSH_TIMEOUT_S = 5.0
_FLUSH_QUIET_S = 0.5

# Silence sent after the end of the stream, beyond the silence duration, so
# the server ends speech in audio it had not processed yet.
_FLUSH_SILENCE_MARGIN_MS = 200

# A transcription session streams for at most 10 minutes, and the server
# sends GoAway a minute before it closes one. A session is replaced between
# utterances after this long, so it rarely has to be replaced in the middle of
# one.
_SESSION_ROTATION_S = 480.0

# Audio from before a final transcript that a new session still gets, so
# speech that started just before the switch keeps its first syllable. It is
# never longer than the silence duration, so it is silence the server already
# heard, and the finalized speech is not transcribed twice.
_REPLAY_MS = 500

# Longest stretch of audio sent again to a new session, unless
# `max_replay_ms` sets it.
_DEFAULT_MAX_REPLAY_MS = 10_000

# A session that fails with a transient error is reopened at most this many
# times in a row; a session that lasted `_STABLE_SESSION_S` resets the count.
_MAX_RECONNECTS = 2
_STABLE_SESSION_S = 30.0

# WebSocket close codes after which a new session may succeed: normal closure
# and going away (the server ended the session), abnormal closure (the
# connection was cut), internal error, service restart, try again later, and
# bad gateway.
_TRANSIENT_CLOSE_CODES = frozenset({1000, 1001, 1006, 1011, 1012, 1013, 1014})

# HTTP status codes after which a new session may succeed.
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Pushed onto the event queue when the ingress has nothing further to report.
_DONE = object()


def _is_transient(error: Exception) -> bool:
  """Whether a failed session may succeed if opened again."""
  # Imported here: the Live API errors are only needed once a session fails.
  from google.genai import errors  # pylint: disable=g-import-not-at-top
  from websockets.exceptions import ConnectionClosed  # pylint: disable=g-import-not-at-top
  from websockets.exceptions import InvalidStatus  # pylint: disable=g-import-not-at-top

  if isinstance(error, errors.APIError):
    # The SDK reports a closed connection as an APIError with its close code.
    code = error.code
  elif isinstance(error, ConnectionClosed):
    # Raised by a send; no close frame means the connection was cut.
    code = error.rcvd.code if error.rcvd else 1006
  elif isinstance(error, InvalidStatus):
    # The server refused the WebSocket handshake.
    code = error.response.status_code
  else:
    return isinstance(error, (ConnectionError, TimeoutError))
  return code in _TRANSIENT_CLOSE_CODES or code in _TRANSIENT_STATUS_CODES


def _sample_rate_of(blob: types.Blob) -> int:
  """Returns the sample rate from a PCM blob's mime type, defaulting to 16000."""
  for parameter in (blob.mime_type or '').lower().split(';')[1:]:
    name, _, value = parameter.partition('=')
    if name.strip() == 'rate' and value.strip().isdigit():
      return int(value)
  return _DEFAULT_SAMPLE_RATE


class _AudioStream:
  """Reads the client's audio and keeps what a new session may need again.

  One audio segment can span several sessions, but its audio is read only
  once, so the audio since the last final transcript is kept for the next
  session to start with. For example, with 100 ms frames and the default
  500 ms of silence:

  - The user says "what time is it" in frames 1 to 15, and the server
    finalizes it at frame 21. `mark_finalized` keeps only frames 17 to 21,
    which are silence.
  - If the session drops at frame 45, after the user started "and the date"
    at frame 40, the new session is sent frames 17 to 45 first, so it
    transcribes "and the date" whole.
  - If the session is instead replaced between utterances at frame 30, the
    new session is sent frames 17 to 30 first, so speech that began just
    before the switch keeps its first syllable.

  At most the last `max_ms` is kept, 10 s by default, so a drop deep into a
  longer utterance loses its start. A session that fails while waiting for
  the next frame does not end the client's audio: the read carries over to
  the next session.
  """

  def __init__(
      self, audio: AsyncIterator[types.Blob], *, tail_ms: int, max_ms: int
  ) -> None:
    """Wraps the client's audio.

    Args:
      audio: The audio segment.
      tail_ms: How much of the audio before a final transcript to keep.
      max_ms: The most audio to keep.
    """
    self._audio = audio
    self._tail_ms = tail_ms
    self._max_ms = max_ms
    self._ended = False
    # The read of the next frame, kept when the session waiting for it stops.
    self._read: Optional[asyncio.Future[types.Blob]] = None
    self.mime_type = f'audio/pcm;rate={_DEFAULT_SAMPLE_RATE}'
    self.sample_rate = _DEFAULT_SAMPLE_RATE
    # The audio since the last final transcript, at most `max_ms` of it, and
    # the last `tail_ms` before that transcript.
    self._unfinalized: collections.deque[types.Blob] = collections.deque()
    self._unfinalized_bytes = 0

  async def next(self) -> Optional[types.Blob]:
    """Returns the next audio blob, or `None` once the segment has ended."""
    while not self._ended:
      if self._read is None:
        self._read = asyncio.ensure_future(self._audio.__anext__())
      try:
        # Shielded: cancelling the read would end the client's audio stream.
        # If this session stops waiting, the read and any frame it returns
        # are kept for the next one.
        blob = await asyncio.shield(self._read)
      except StopAsyncIteration:
        self._ended = True
        break
      self._read = None
      if not blob.data:
        continue
      self.mime_type = blob.mime_type or self.mime_type
      self.sample_rate = _sample_rate_of(blob)
      self._unfinalized.append(blob)
      self._unfinalized_bytes += len(blob.data)
      self._trim(self.bytes_for(self._max_ms))
      return blob
    return None

  def mark_finalized(self) -> None:
    """Forgets the audio of the utterance just finalized, but its last moments."""
    self._trim(self.bytes_for(self._tail_ms))

  def replay(self) -> list[types.Blob]:
    """Returns the audio to send again to a new session, oldest first."""
    return list(self._unfinalized)

  async def aclose(self) -> None:
    """Stops a read in progress, so it does not take the next segment's audio."""
    if self._read is not None:
      self._read.cancel()
      await asyncio.gather(self._read, return_exceptions=True)
      self._read = None

  def bytes_for(self, duration_ms: int) -> int:
    """Returns the size of `duration_ms` of audio at the current rate."""
    return self.sample_rate * duration_ms // 1000 * _BYTES_PER_SAMPLE

  def _trim(self, limit: int) -> None:
    """Drops the oldest audio while at least `limit` bytes would remain."""
    while (
        self._unfinalized
        and self._unfinalized_bytes - len(self._unfinalized[0].data or b'')
        >= limit
    ):
      self._unfinalized_bytes -= len(self._unfinalized.popleft().data or b'')


class _TranscriptionState:
  """What the transcription sessions of one audio segment have reported.

  A session that replaces another in the middle of an utterance hears that
  utterance again, so the first start of speech it reports is not reported
  twice.
  """

  def __init__(
      self,
      emit: Callable[[IngressEvent], None],
      on_finalized: Callable[[], None],
  ) -> None:
    self._emit = emit
    # Called once the server is done with an utterance.
    self._on_finalized = on_finalized
    # Between the server detecting speech and it ending.
    self.speaking = False
    # The latest interim transcript of the utterance not yet finalized.
    self.partial = ''
    # Set when the session replaced the previous one in the middle of an
    # utterance, whose start was already reported.
    self._resumed = False
    # Set when the server asks the client to reconnect.
    self.go_away = False
    # Set when the server confirms the end of the user's activity.
    self.activity_ended = asyncio.Event()
    # Set on every server message, for waiting until the server goes quiet.
    self.message_received = asyncio.Event()

  def start_session(self) -> None:
    """Starts over for a new session, which first hears the audio sent again."""
    self._resumed = self.pending
    self.speaking = False
    self.partial = ''
    self.go_away = False
    self.activity_ended.clear()
    self.message_received.clear()

  @property
  def pending(self) -> bool:
    """Whether an utterance is in progress or awaits its final transcript."""
    return self.speaking or bool(self.partial)

  def on_message(self, message: types.LiveServerMessage) -> None:
    """Turns one server message into ingress events."""
    self.message_received.set()
    if message.go_away:
      self.go_away = True
    activity = message.voice_activity
    if activity and activity.voice_activity_type:
      if activity.voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
        self._start_speech()
      elif activity.voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
        self.speaking = False
        self.activity_ended.set()
    content = message.server_content
    if not content:
      return
    interim = content.interim_input_transcription
    if interim and interim.text and interim.text.strip():
      self._start_speech()
      text = interim.text.strip()
      if text != self.partial:
        self.partial = text
        # The full hypothesis so far, which replaces the previous one.
        self._emit(PartialTranscript(text=text))
    final = content.input_transcription
    if final and final.text and final.text.strip():
      self._finalize()
      self._emit(UserTurnFinished(text=final.text.strip()))
    if content.generation_complete:
      # The server is done with the utterance, whether or not it had words.
      self._finalize()

  def _start_speech(self) -> None:
    if not self.speaking:
      self.speaking = True
      self.activity_ended.clear()
      if self._resumed:
        self._resumed = False
      else:
        self._emit(UserSpeechStarted())

  def _finalize(self) -> None:
    """Forgets the utterance the server is done with."""
    self.speaking = False
    self.partial = ''
    self._resumed = False
    self._on_finalized()


@experimental(FeatureName.CASCADE_LIVE)
class GeminiSTT(LiveIngress):
  """LiveIngress adapter for Gemini streaming transcription.

  Streams the user's audio to `gemini-3.5-transcribe-live` over the Live API,
  so the audio is transcribed while the user speaks. The server detects the
  start and end of speech; interim transcripts are reported as
  `PartialTranscript`, and the final transcript of each utterance as
  `UserTurnFinished` once the user has been silent for `silence_duration_ms`.

  If the client signals the end of the user's turn itself, for example by
  sending `ActivityEnd` when a push-to-talk button is released, or by closing
  the microphone (`audio_stream_end`), the final transcript arrives right away
  instead of after the silence. To leave turn detection entirely to the
  client, set
  `activity_detection=types.AutomaticActivityDetection(disabled=True)`: each
  audio segment, from the first frame to `ActivityEnd`, is then one
  utterance.

  Sessions are replaced before the Live API's 10-minute limit, between
  utterances when the user pauses, and right away when the server sends
  GoAway. A session that fails with a transient error, such as a dropped
  connection, is reopened. Either way the new session is sent the speech the
  old one had not finished transcribing, up to `max_replay_ms` of it.

  Authenticates like the `Gemini` text model, with `GOOGLE_API_KEY` for the
  Gemini API or `GOOGLE_GENAI_USE_ENTERPRISE` plus a project and location for
  Vertex AI. On Vertex AI the model is served from the `global`, `us` and
  `eu` locations.
  """

  def __init__(
      self,
      *,
      model: Union[str, Gemini] = _DEFAULT_MODEL,
      audio_transcription_config: Optional[
          types.AudioTranscriptionConfig
      ] = None,
      activity_detection: Optional[types.AutomaticActivityDetection] = None,
      max_replay_ms: int = _DEFAULT_MAX_REPLAY_MS,
  ) -> None:
    """Configures the recognizer.

    Args:
      model: The Gemini streaming transcription model name, or a `Gemini`
        instance whose client settings (`client_kwargs`, `base_url`) should be
        used.
      audio_transcription_config: Transcription settings such as
        `language_codes` and `custom_vocabulary`. Defaults to automatic
        language detection.
      activity_detection: The server's voice activity detection settings, in
        the vocabulary of `RunConfig.realtime_input_config`. A shorter
        `silence_duration_ms` (500 ms unless set) ends the user's turn sooner
        but may split it at a pause. `disabled=True` leaves turn detection to
        the client's `ActivityStart` and `ActivityEnd`.
      max_replay_ms: The most speech sent again to a session that replaces
        another in the middle of an utterance. A longer utterance loses its
        start. The audio is kept in memory, 32 KB per second at 16 kHz.

    Raises:
      ValueError: If `max_replay_ms` is negative.
    """
    if max_replay_ms < 0:
      raise ValueError('max_replay_ms must not be negative.')
    # Imported here: `google_llm` imports the agents package, which imports
    # this package.
    from ..models.google_llm import Gemini  # pylint: disable=g-import-not-at-top

    self._llm = model if isinstance(model, Gemini) else Gemini(model=model)
    detection = activity_detection or types.AutomaticActivityDetection()
    # The client marks each utterance with `ActivityStart` and `ActivityEnd`,
    # and the server's voice activity detection is off.
    self._client_vad = bool(detection.disabled)
    if not self._client_vad and detection.silence_duration_ms is None:
      detection = detection.model_copy(
          update={'silence_duration_ms': _DEFAULT_SILENCE_DURATION_MS}
      )
    self._silence_duration_ms = (
        detection.silence_duration_ms or _DEFAULT_SILENCE_DURATION_MS
    )
    self._max_replay_ms = max_replay_ms
    self._config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        input_audio_transcription=(
            audio_transcription_config or types.AudioTranscriptionConfig()
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=detection
        ),
    )

  async def __call__(
      self, audio: AsyncIterator[types.Blob]
  ) -> AsyncGenerator[IngressEvent, None]:
    """Transcribes `audio`, yielding speech starts, captions and user turns."""
    client = self._llm.api_client
    events: asyncio.Queue[Any] = asyncio.Queue()

    async def transcribe() -> None:
      stream = _AudioStream(
          audio,
          tail_ms=min(_REPLAY_MS, self._silence_duration_ms),
          max_ms=self._max_replay_ms,
      )
      try:
        await self._transcribe(client, stream, events.put_nowait)
        events.put_nowait(_DONE)
      except Exception as e:  # pylint: disable=broad-except
        events.put_nowait(e)
      finally:
        await stream.aclose()

    task = asyncio.create_task(transcribe())
    try:
      while True:
        item = await events.get()
        if item is _DONE:
          return
        if isinstance(item, Exception):
          raise item
        yield item
    finally:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)

  async def _transcribe(
      self,
      client: Client,
      stream: _AudioStream,
      emit: Callable[[IngressEvent], None],
  ) -> None:
    """Runs transcription sessions until the audio segment is transcribed.

    A session is replaced before its time limit. A session that fails with a
    transient error, whether opening or streaming, is reopened, unless
    sessions keep failing. Each new session starts with the audio since the
    last final transcript, so speech is not lost in the switch. Any other
    error is raised unchanged.

    Args:
      client: The client that opens the sessions.
      stream: The client's audio.
      emit: Receives the ingress events.
    """
    loop = asyncio.get_running_loop()
    state = _TranscriptionState(emit, on_finalized=stream.mark_finalized)
    failures = 0
    while True:
      started_at = loop.time()
      try:
        if await self._run_session(client, stream, state):
          return
        logger.debug('Replacing the Gemini transcription session.')
      except Exception as e:  # pylint: disable=broad-except
        if not _is_transient(e):
          raise
        if loop.time() - started_at >= _STABLE_SESSION_S:
          failures = 0
        failures += 1
        if failures > _MAX_RECONNECTS:
          raise
        logger.warning(
            'The Gemini transcription session failed; reopening it.',
            exc_info=True,
        )

  async def _run_session(
      self,
      client: Client,
      stream: _AudioStream,
      state: _TranscriptionState,
  ) -> bool:
    """Transcribes on one Live API session.

    Args:
      client: The client that opens the session.
      stream: The client's audio, shared by successive sessions.
      state: What the server has reported about the audio segment, shared by
        successive sessions.

    Returns:
      True once the audio segment has ended and been transcribed, or False
      when the session should be replaced.
    """
    async with client.aio.live.connect(
        model=self._llm.model, config=self._config
    ) as session:
      state.start_session()
      sender = asyncio.create_task(self._send(session, stream, state))
      receiver = asyncio.create_task(self._receive(session, state))
      try:
        await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        if not sender.done():
          # The receiver only stops when the session fails.
          receiver.result()
          raise RuntimeError('The Gemini transcription session ended early.')
        ended = sender.result()
        if ended:
          await self._flush(session, stream, state, receiver)
        return ended
      finally:
        for task in (sender, receiver):
          task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)

  async def _send(
      self,
      session: AsyncSession,
      stream: _AudioStream,
      state: _TranscriptionState,
  ) -> bool:
    """Streams audio to `session` until the segment ends or it must rotate.

    Starts the activity first when the client detects turns, so the audio
    sent again belongs to it. Then sends the audio kept from earlier sessions,
    then each new frame.

    Args:
      session: The session to stream to.
      stream: The client's audio, shared by successive sessions.
      state: What the server has reported, which decides when to rotate.

    Returns:
      True once the client has ended the audio segment, with `ActivityEnd` or
      `audio_stream_end`, so the speech left must be finalized on this
      session. False when this session must be replaced first: right away
      once the server sends GoAway, or after `_SESSION_ROTATION_S` once the
      user is not speaking.
    """
    if self._client_vad:
      await session.send_realtime_input(activity_start=types.ActivityStart())
    for replayed in stream.replay():
      await session.send_realtime_input(audio=replayed)
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    while (blob := await stream.next()) is not None:
      await session.send_realtime_input(audio=blob)
      if state.go_away or (
          # An utterance the client detects lasts the whole segment, so only
          # GoAway splits it.
          not self._client_vad
          and not state.pending
          and loop.time() - started_at >= _SESSION_ROTATION_S
      ):
        return False
    return True

  async def _receive(
      self, session: AsyncSession, state: _TranscriptionState
  ) -> None:
    """Dispatches server messages until the session closes."""
    while True:
      # `receive()` returns at the end of each server turn.
      async for message in session.receive():
        state.on_message(message)

  async def _flush(
      self,
      session: AsyncSession,
      stream: _AudioStream,
      state: _TranscriptionState,
      receiver: asyncio.Task[None],
  ) -> None:
    """Finalizes the speech left when the audio segment ends."""
    if self._client_vad:
      await session.send_realtime_input(activity_end=types.ActivityEnd())
      try:
        await asyncio.wait_for(state.activity_ended.wait(), _FLUSH_TIMEOUT_S)
      except asyncio.TimeoutError:
        logger.warning(
            'Gemini transcription did not end the activity within %.1fs; the'
            ' last utterance may be lost.',
            _FLUSH_TIMEOUT_S,
        )
      return

    # Ends speech the server is in the middle of without waiting for the
    # silence. It is ignored if the server has not reached that speech yet,
    # so trailing silence follows to end it the usual way.
    await session.send_realtime_input(audio_stream_end=True)
    await session.send_realtime_input(
        audio=types.Blob(
            data=bytes(
                stream.bytes_for(
                    self._silence_duration_ms + _FLUSH_SILENCE_MARGIN_MS
                )
            ),
            mime_type=stream.mime_type,
        )
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _FLUSH_TIMEOUT_S
    # Done once nothing is pending and the server has been quiet for
    # `_FLUSH_QUIET_S`. It may not have reported the last speech yet, so the
    # quiet period starts now.
    while not receiver.done() and loop.time() < deadline:
      state.message_received.clear()
      try:
        await asyncio.wait_for(
            state.message_received.wait(),
            min(_FLUSH_QUIET_S, deadline - loop.time()),
        )
      except asyncio.TimeoutError:
        if not state.pending:
          return
    if receiver.done():
      receiver.result()
    if state.pending:
      logger.warning(
          'Gemini transcription returned no transcript for the end of the'
          ' audio segment within %.1fs; the last utterance was dropped.',
          _FLUSH_TIMEOUT_S,
      )

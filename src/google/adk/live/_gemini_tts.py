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

"""A `LiveEgress` backed by Gemini text-to-speech models."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
import weakref

from google.genai import types

from ..features import experimental
from ..features import FeatureName
from ..utils.context_utils import Aclosing
from ._cascade_live_events import AgentSpokenOutput
from ._cascade_live_events import AudioChunk
from ._cascade_live_events import EgressEvent
from ._transforms import LiveEgress

if TYPE_CHECKING:
  from google.genai import Client

  from ..models.google_llm import Gemini

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_MODEL = 'gemini-3.8-flash-lite-tts'

_DEFAULT_VOICE = 'Kore'

# Streamed Gemini TTS audio is 16-bit mono PCM at 24 kHz.
_DEFAULT_SAMPLE_RATE = 24000

# Longest a synthesis request waits for the warm-up's connection.
_MAX_WARM_UP_WAIT_S = 2.0

# Delays before sending a failed synthesis request again, one per retry. A
# request is retried only for a transient error, and only before it returned
# any audio.
_SYNTHESIS_RETRY_DELAYS_S = (0.1, 0.3)

# HTTP status codes of errors worth retrying.
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# Ends the queue of a segment's audio, or the queue of segments.
_END = object()

# A sentence ends at Latin or Devanagari terminal punctuation followed by
# whitespace, at CJK full-width terminal punctuation (which is not followed by
# a space), or at a line break. Closing quotes and brackets stay with the
# sentence they close.
_SENTENCE_END = re.compile(
    r'[.!?\u2026\u0964]["\'\u201d\u2019)\]]*\s+'
    r'|[\u3002\uff01\uff1f]["\'\u201d\u2019)\]\u300d\u300f\uff09]*\s*'
    r'|\n+'
)

# Where an over-long segment with no whitespace may be cut.
_CJK_CLAUSE_END = re.compile(r'[\uff0c\u3001\uff1b\uff1a]')


def _split_segment(
    buffer: str,
    *,
    max_chars: int,
    min_chars: int = 0,
) -> tuple[Optional[str], str]:
  """Splits the leading segment off `buffer`.

  The segment keeps the whitespace that followed it, so joining every
  returned segment reproduces the original text.

  Args:
    buffer: The text received so far that has not been split off yet.
    max_chars: Length at which text with no boundary is cut anyway: at the
      last space, else after the last CJK clause punctuation, else at
      `max_chars` itself.
    min_chars: The segment ends only at a sentence boundary at least this
      many characters in, so a shorter sentence stays with the text after it.

  Returns:
    `(segment, rest)`, or `(None, buffer)` when `buffer` holds no complete
    segment yet.
  """
  for match in _SENTENCE_END.finditer(buffer):
    if match.end() >= min_chars:
      return buffer[: match.end()], buffer[match.end() :]

  if len(buffer) < max_chars:
    return None, buffer

  head = buffer[:max_chars]
  cut = head.rfind(' ')
  if cut > 0:
    return buffer[: cut + 1], buffer[cut + 1 :]
  clause_ends = list(_CJK_CLAUSE_END.finditer(head))
  if clause_ends:
    cut = clause_ends[-1].end()
    return buffer[:cut], buffer[cut:]
  return head, buffer[max_chars:]


def _as_pcm(blob: types.Blob) -> types.Blob:
  """Relabels streamed Gemini audio as `audio/pcm;rate=N`, which clients play.

  Streamed audio is raw 16-bit PCM labeled `audio/l16; rate=24000;
  channels=1` or `audio/L16;codec=pcm;rate=24000`, depending on the backend.

  Args:
    blob: Audio streamed by a Gemini TTS model.

  Returns:
    The same audio labeled `audio/pcm;rate=N`, or `blob` unchanged if it is
    not linear PCM.
  """
  media_type, *parameters = (blob.mime_type or '').lower().split(';')
  if media_type.strip() not in ('audio/l16', 'audio/pcm'):
    return blob
  rate = _DEFAULT_SAMPLE_RATE
  for parameter in parameters:
    name, _, value = parameter.partition('=')
    if name.strip() == 'rate' and value.strip().isdigit():
      rate = int(value)
  return types.Blob(data=blob.data, mime_type=f'audio/pcm;rate={rate}')


def _is_transient(error: Exception) -> bool:
  """Whether a failed synthesis request may succeed if sent again."""
  # Imported here: the API errors are only needed once a request fails.
  from google.genai import errors  # pylint: disable=g-import-not-at-top

  if isinstance(error, errors.APIError):
    return error.code in _TRANSIENT_STATUS_CODES
  return isinstance(error, (ConnectionError, asyncio.TimeoutError))


class _Segment:
  """A piece of the reply and the audio synthesized for it so far."""

  def __init__(self, text: str) -> None:
    self.text = text
    # Audio blobs, then `_END`, or the exception that ended synthesis.
    self.audio: asyncio.Queue[Any] = asyncio.Queue()
    # Set once any audio is queued, after which a failure is not retried.
    self.has_audio = False
    self.task: Optional[asyncio.Task[None]] = None


@experimental(FeatureName.CASCADE_LIVE)
class GeminiTTS(LiveEgress):
  """LiveEgress adapter for Gemini text-to-speech models.

  Splits the reasoner's streamed text into segments of one or more sentences,
  and synthesizes each with a streaming `generate_content` request, so a
  segment's audio reaches the client while it is still being generated. The
  requests for the next segments start while the current one is still being
  delivered, so there is no pause between sentences, and the first turn warms
  the HTTP connection while the reasoner is still thinking. A request that
  fails with a transient error before returning any audio is retried.

  Authenticates like the `Gemini` text model, with `GOOGLE_API_KEY` for the
  Gemini API or `GOOGLE_GENAI_USE_ENTERPRISE` plus a project and location for
  Vertex AI. On Vertex AI, the Gemini 3.8 TTS models are served only from the
  `global` location.

  Example::

      CascadeLive(
          model='gemini-3.8-flash',
          stt=GeminiSTT(),
          tts=GeminiTTS(speech_config='Puck'),
      )
  """

  def __init__(
      self,
      *,
      model: Union[str, Gemini] = _DEFAULT_MODEL,
      speech_config: types.SpeechConfigUnion = _DEFAULT_VOICE,
      min_segment_chars: int = 50,
      max_segment_chars: int = 240,
      prefetch_segments: int = 1,
  ) -> None:
    """Configures the synthesizer.

    Args:
      model: The Gemini TTS model name, or a `Gemini` instance whose client
        settings (`client_kwargs`, `base_url`, `retry_options`) should be used,
        e.g. `Gemini(model=..., client_kwargs={'location': 'global'})`.
      speech_config: A prebuilt voice name such as `'Kore'`, or a full
        `types.SpeechConfig` to also set the language or another voice.
      min_segment_chars: Length a segment after the first must reach before
        it may end at a sentence boundary. A shorter sentence is synthesized
        together with the text after it, which gives the model context: a
        letter spelled out on its own, such as "Y.", may not be read as its
        name. The first segment ends at its first sentence boundary
        regardless, so the time to the first audio is unchanged.
      max_segment_chars: Length at which a segment with no sentence boundary
        is cut anyway.
      prefetch_segments: How many segments after the one being delivered may
        be synthesized ahead of time. 0 synthesizes one segment at a time.
    """
    if prefetch_segments < 0:
      raise ValueError('prefetch_segments must not be negative.')
    if max_segment_chars < 1:
      raise ValueError('max_segment_chars must be positive.')
    if min_segment_chars < 0:
      raise ValueError('min_segment_chars must not be negative.')
    # Imported here: `google_llm` imports the agents package, which imports
    # this package.
    from ..models.google_llm import Gemini  # pylint: disable=g-import-not-at-top

    self._llm = model if isinstance(model, Gemini) else Gemini(model=model)
    self._config = types.GenerateContentConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=speech_config,
    )
    self._min_segment_chars = min_segment_chars
    self._max_segment_chars = max_segment_chars
    self._prefetch_segments = prefetch_segments
    # A warm-up request per client, so the first request on a client waits
    # for its connection instead of opening another one.
    self._warm_ups: weakref.WeakKeyDictionary[Client, asyncio.Task[None]] = (
        weakref.WeakKeyDictionary()
    )

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[EgressEvent, None]:
    """Speaks `text`, yielding audio and a record of what was said."""
    client = self._llm.api_client
    warm_up = self._warm_up(client)
    # `_Segment`s in reply order, then `_END`, or the exception that stopped
    # reading `text`.
    segments: asyncio.Queue[Any] = asyncio.Queue()
    # Segments started and not yet delivered, which bounds the work wasted
    # when the reply is cancelled.
    slots = asyncio.Semaphore(self._prefetch_segments + 1)
    started: list[_Segment] = []

    async def start(piece: str) -> None:
      await slots.acquire()
      segment = _Segment(piece)
      segment.task = asyncio.create_task(
          self._synthesize(client, segment, warm_up)
      )
      started.append(segment)
      segments.put_nowait(segment)

    async def split_text() -> None:
      try:
        buffer = ''
        # Bare line breaks, carried into the next segment so the spoken
        # record still reproduces the reply.
        whitespace = ''
        # The first segment ends as early as it can, for the time to the
        # first audio; later ones wait until they are long enough.
        min_chars = 0
        async for delta in text:
          buffer += delta
          while True:
            piece, buffer = _split_segment(
                buffer,
                max_chars=self._max_segment_chars,
                min_chars=min_chars,
            )
            if piece is None:
              break
            if not piece.strip():
              whitespace += piece
              continue
            min_chars = self._min_segment_chars
            await start(whitespace + piece)
            whitespace = ''
        if buffer.strip():
          await start(whitespace + buffer)
        segments.put_nowait(_END)
      except Exception as e:  # pylint: disable=broad-except
        segments.put_nowait(e)

    # `CascadeLive` stops a reply by cancelling the task that reads it, which
    # stops this generator at once, so `cancel` is only checked as each item
    # arrives.
    splitter = asyncio.create_task(split_text())
    try:
      while True:
        segment = await segments.get()
        if segment is _END or cancel.is_set():
          return
        if isinstance(segment, Exception):
          raise segment
        produced_audio = False
        while True:
          audio = await segment.audio.get()
          if cancel.is_set():
            # Barge-in: drop the rest of the audio, and do not record the
            # segment as spoken.
            return
          if audio is _END:
            break
          if isinstance(audio, Exception):
            raise audio
          produced_audio = True
          yield AudioChunk(blob=audio)
        slots.release()
        if not produced_audio:
          logger.warning(
              'Gemini TTS returned no audio for a segment; it is not recorded'
              ' as spoken.'
          )
          continue
        # Keeps the whitespace that followed the segment, so the connection's
        # join of every spoken segment reads like the model's text.
        yield AgentSpokenOutput(text=segment.text)
    finally:
      tasks = [splitter, *(segment.task for segment in started)]
      for task in tasks:
        if task is not None:
          task.cancel()
      await asyncio.gather(
          *(task for task in tasks if task is not None),
          return_exceptions=True,
      )

  async def _synthesize(
      self,
      client: Client,
      segment: _Segment,
      warm_up: asyncio.Task[None],
  ) -> None:
    """Streams the audio for `segment` into its queue, retrying if needed."""
    try:
      if not warm_up.done():
        # Reuses the connection the warm-up is opening, rather than opening
        # a second one.
        await asyncio.wait({warm_up}, timeout=_MAX_WARM_UP_WAIT_S)
      retries = 0
      while True:
        try:
          await self._stream_audio(client, segment)
          break
        except Exception as e:  # pylint: disable=broad-except
          # Audio already delivered cannot be taken back, so a request is
          # sent again only if it failed before returning any.
          if (
              segment.has_audio
              or retries == len(_SYNTHESIS_RETRY_DELAYS_S)
              or not _is_transient(e)
          ):
            raise
          logger.warning('Gemini TTS request failed; retrying.', exc_info=True)
          await asyncio.sleep(_SYNTHESIS_RETRY_DELAYS_S[retries])
          retries += 1
      segment.audio.put_nowait(_END)
    except Exception as e:  # pylint: disable=broad-except
      segment.audio.put_nowait(e)

  async def _stream_audio(self, client: Client, segment: _Segment) -> None:
    """Sends one synthesis request and queues the audio it streams."""
    stream = await client.aio.models.generate_content_stream(
        model=self._llm.model,
        contents=segment.text.strip(),
        config=self._config,
    )
    # The SDK returns an async generator, typed as an AsyncIterator.
    async with Aclosing(
        cast(AsyncGenerator[types.GenerateContentResponse, None], stream)
    ) as responses:
      async for response in responses:
        if not response.candidates or not response.candidates[0].content:
          continue
        for part in response.candidates[0].content.parts or []:
          if part.inline_data and part.inline_data.data:
            segment.has_audio = True
            segment.audio.put_nowait(_as_pcm(part.inline_data))

  def _warm_up(self, client: Client) -> asyncio.Task[None]:
    """Opens `client`'s connection the first time a reply uses it.

    A new client spends about a second fetching credentials and opening a
    connection. Doing that while the reasoner generates the first sentence
    keeps it out of the time to the first audio.

    Args:
      client: The client the reply synthesizes with.

    Returns:
      The warm-up request, started by the first reply on `client`.
    """
    task = self._warm_ups.get(client)
    if task is None:
      task = asyncio.create_task(self._fetch_model_metadata(client))
      self._warm_ups[client] = task
    return task

  async def _fetch_model_metadata(self, client: Client) -> None:
    try:
      await client.aio.models.get(model=self._llm.model)
    except Exception:  # pylint: disable=broad-except
      # Only a warm-up: synthesis reports any real problem.
      logger.debug('Gemini TTS warm-up request failed.', exc_info=True)

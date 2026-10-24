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

"""Tests for GeminiTTS.

The Gemini API is faked: each segment is "synthesized" as its UTF-8 bytes, so
the audio shows which text was synthesized and in what order it was
delivered.
"""

from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Optional
from unittest import mock

from google.adk.live import _gemini_tts
from google.adk.live import AgentSpokenOutput
from google.adk.live import AudioChunk
from google.adk.live import GeminiTTS
from google.adk.live._gemini_tts import _split_segment
from google.adk.models.google_llm import Gemini
from google.genai import errors
from google.genai import types
import pytest

_PCM = 'audio/pcm;rate=24000'


def _audio_response(data: bytes) -> types.GenerateContentResponse:
  return types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part(
                          inline_data=types.Blob(
                              data=data,
                              mime_type='audio/l16; rate=24000; channels=1',
                          )
                      )
                  ],
              )
          )
      ]
  )


class _FakeSynthesizer:
  """Stands in for `client.aio.models` of the Gemini API."""

  def __init__(self) -> None:
    self.texts: list[str] = []
    # Texts whose audio streams only once their event is set.
    self.held: dict[str, asyncio.Event] = {}
    # Texts whose stream was closed, finished or not.
    self.closed: list[str] = []
    self.silent: set[str] = set()
    self.failing: set[str] = set()
    # Raised, in order, by the next requests for a text, before any audio.
    self.request_errors: dict[str, list[Exception]] = {}
    # Raised by the stream of a text after its audio.
    self.errors_after_audio: dict[str, Exception] = {}
    # Streamed before the audio of every text.
    self.preamble: list[types.GenerateContentResponse] = []
    self.metadata_error: Optional[Exception] = None
    self.metadata_requests = 0

  async def generate_content_stream(
      self, *, model: str, contents: str, config: Any
  ) -> AsyncIterator[types.GenerateContentResponse]:
    """Records `contents` and streams it back as its "audio"."""
    del model, config
    self.texts.append(contents)
    if self.request_errors.get(contents):
      raise self.request_errors[contents].pop(0)

    async def stream() -> AsyncIterator[types.GenerateContentResponse]:
      try:
        if contents in self.held:
          await self.held[contents].wait()
        if contents in self.failing:
          raise RuntimeError('synthesis failed')
        for response in self.preamble:
          yield response
        if contents not in self.silent:
          yield _audio_response(contents.encode())
        if contents in self.errors_after_audio:
          raise self.errors_after_audio[contents]
      finally:
        self.closed.append(contents)

    return stream()

  async def get(self, *, model: str) -> None:
    del model
    self.metadata_requests += 1
    if self.metadata_error:
      raise self.metadata_error


def _api_error(code: int) -> errors.APIError:
  error_type = errors.ServerError if code >= 500 else errors.ClientError
  return error_type(
      code, {'error': {'code': code, 'message': 'failed', 'status': 'ERROR'}}
  )


@pytest.fixture(name='synthesizer')
def _synthesizer(monkeypatch):
  monkeypatch.setattr(_gemini_tts, '_SYNTHESIS_RETRY_DELAYS_S', (0, 0))
  fake = _FakeSynthesizer()
  client = mock.MagicMock()
  client.aio.models.generate_content_stream = fake.generate_content_stream
  client.aio.models.get = fake.get
  with mock.patch.object(
      Gemini, 'api_client', new=property(lambda self: client)
  ):
    yield fake


async def _text(*deltas: str) -> AsyncIterator[str]:
  for delta in deltas:
    yield delta


async def _speak(
    tts: GeminiTTS, *deltas: str, cancel: Optional[asyncio.Event] = None
) -> list[Any]:
  return [
      event
      async for event in tts(_text(*deltas), cancel=cancel or asyncio.Event())
  ]


async def _until(condition: Callable[[], bool]) -> None:
  for _ in range(100):
    if condition():
      return
    await asyncio.sleep(0)
  raise AssertionError('The condition was never met.')


def _audio(text: str) -> AudioChunk:
  return AudioChunk(blob=types.Blob(data=text.encode(), mime_type=_PCM))


async def test_reply_is_spoken_sentence_by_sentence(synthesizer):
  """Each sentence is synthesized, delivered, then recorded with its spaces."""
  events = await _speak(GeminiTTS(), 'Hello there. How ', 'are you?')

  assert synthesizer.texts == ['Hello there.', 'How are you?']
  assert events == [
      _audio('Hello there.'),
      AgentSpokenOutput(text='Hello there. '),
      _audio('How are you?'),
      AgentSpokenOutput(text='How are you?'),
  ]


async def test_audio_keeps_reply_order_when_a_later_segment_is_ready_first(
    synthesizer,
):
  """A segment synthesized ahead waits for the one before it."""
  synthesizer.held['One.'] = asyncio.Event()
  speaking = asyncio.create_task(_speak(GeminiTTS(), 'One. Two.'))
  await _until(lambda: 'Two.' in synthesizer.closed)

  synthesizer.held['One.'].set()
  events = await speaking

  assert events == [
      _audio('One.'),
      AgentSpokenOutput(text='One. '),
      _audio('Two.'),
      AgentSpokenOutput(text='Two.'),
  ]


async def test_prefetch_bounds_the_segments_synthesized_ahead(synthesizer):
  """With one segment of prefetch, only the next segment starts early."""
  synthesizer.held['One.'] = asyncio.Event()
  speaking = asyncio.create_task(
      _speak(
          GeminiTTS(prefetch_segments=1, min_segment_chars=0),
          'One. Two. Three.',
      )
  )
  await _until(lambda: len(synthesizer.texts) == 2)
  await asyncio.sleep(0)

  assert synthesizer.texts == ['One.', 'Two.']
  synthesizer.held['One.'].set()
  await speaking
  assert synthesizer.texts == ['One.', 'Two.', 'Three.']


async def test_no_prefetch_synthesizes_one_segment_at_a_time(synthesizer):
  """With no prefetch, the next segment starts once the current is spoken."""
  synthesizer.held['One.'] = asyncio.Event()
  speaking = asyncio.create_task(
      _speak(GeminiTTS(prefetch_segments=0), 'One. Two.')
  )
  await _until(lambda: synthesizer.texts == ['One.'])
  for _ in range(10):
    await asyncio.sleep(0)

  assert synthesizer.texts == ['One.']
  synthesizer.held['One.'].set()
  await speaking
  assert synthesizer.texts == ['One.', 'Two.']


async def test_cancel_drops_the_segment_being_delivered(synthesizer):
  """On barge-in the unfinished segment is neither played nor recorded."""
  synthesizer.held['Two.'] = asyncio.Event()
  cancel = asyncio.Event()
  events = []

  async for event in GeminiTTS()(_text('One. Two.'), cancel=cancel):
    events.append(event)
    if isinstance(event, AgentSpokenOutput):
      await _until(lambda: 'Two.' in synthesizer.texts)
      cancel.set()

  assert events == [_audio('One.'), AgentSpokenOutput(text='One. ')]
  # The request still waiting for its audio was abandoned.
  assert 'Two.' in synthesizer.closed


async def test_cancel_between_chunks_drops_the_rest_of_the_segment(
    synthesizer,
):
  """`cancel` set mid-segment ends the reply at the segment's next chunk."""
  synthesizer.preamble = [_audio_response(b'Start.')]
  cancel = asyncio.Event()
  events = []

  async for event in GeminiTTS()(_text('One.'), cancel=cancel):
    events.append(event)
    cancel.set()

  assert events == [_audio('Start.')]


async def test_cancelling_the_reading_task_abandons_synthesis(synthesizer):
  """A barge-in that cancels the turn's task stops synthesis at once.

  Setup: the first segment is delivered and the second's audio is held.
  Act: the task reading the egress is cancelled, as `CascadeLive` does on
    barge-in, without setting `cancel`.
  Assert:
    - The request waiting for its audio is closed.
    - The interrupted segment is not recorded as spoken.
  """
  synthesizer.held['Two.'] = asyncio.Event()
  events = []
  first_spoken = asyncio.Event()

  async def read() -> None:
    async for event in GeminiTTS()(_text('One. Two.'), cancel=asyncio.Event()):
      events.append(event)
      if isinstance(event, AgentSpokenOutput):
        first_spoken.set()

  reading = asyncio.create_task(read())
  await first_spoken.wait()
  await _until(lambda: 'Two.' in synthesizer.texts)
  reading.cancel()
  await asyncio.gather(reading, return_exceptions=True)

  assert events == [_audio('One.'), AgentSpokenOutput(text='One. ')]
  assert 'Two.' in synthesizer.closed


async def test_closing_early_also_closes_the_text_stream(synthesizer):
  """Closing the egress mid-reply leaves the text stream closable."""
  del synthesizer
  reply_continues = asyncio.Event()

  async def text() -> AsyncIterator[str]:
    yield 'One. '
    await reply_continues.wait()
    yield 'Two.'

  text_stream = text()
  events = GeminiTTS()(text_stream, cancel=asyncio.Event())
  async for event in events:
    if isinstance(event, AgentSpokenOutput):
      break

  await events.aclose()

  # Raises if the text stream were still being read.
  await text_stream.aclose()


async def test_short_sentences_after_the_first_are_synthesized_together(
    synthesizer,
):
  """Sentences after the first are grouped until `min_segment_chars`.

  The first sentence is synthesized on its own, for the earliest audio; the
  letters spelled after it are synthesized with each other, for context.
  """
  events = await _speak(
      GeminiTTS(min_segment_chars=20), 'Hi. Yellow. Y. E. L. L. O. W. Done.'
  )

  assert synthesizer.texts == ['Hi.', 'Yellow. Y. E. L. L.', 'O. W. Done.']
  spoken = [e.text for e in events if isinstance(e, AgentSpokenOutput)]
  assert ''.join(spoken) == 'Hi. Yellow. Y. E. L. L. O. W. Done.'


async def test_synthesis_error_fails_the_reply(synthesizer):
  """An error from the TTS model ends the reply with that error."""
  synthesizer.failing.add('Two.')

  with pytest.raises(RuntimeError, match='synthesis failed'):
    await _speak(GeminiTTS(), 'One. Two.')


async def test_segment_without_audio_is_not_recorded_as_spoken(synthesizer):
  """Text the model returned no audio for stays out of the spoken record."""
  synthesizer.silent.add('One.')

  events = await _speak(GeminiTTS(), 'One. Two.')

  assert events == [_audio('Two.'), AgentSpokenOutput(text='Two.')]


async def test_leading_line_break_is_kept_in_the_spoken_text(synthesizer):
  """A bare line break joins the next segment instead of being dropped."""
  events = await _speak(GeminiTTS(), '\n', 'Hello.')

  assert synthesizer.texts == ['Hello.']
  assert events[-1] == AgentSpokenOutput(text='\nHello.')


async def test_first_reply_warms_up_the_client_once(synthesizer):
  """Only the first reply on a client spends a request warming it up."""
  tts = GeminiTTS()

  await _speak(tts, 'One.')
  await _speak(tts, 'Two.')

  assert synthesizer.metadata_requests == 1


async def test_transient_error_is_retried_before_any_audio(synthesizer):
  """A request that fails transiently before any audio is sent again."""
  synthesizer.request_errors['One.'] = [_api_error(503)]

  events = await _speak(GeminiTTS(), 'One.')

  assert synthesizer.texts == ['One.', 'One.']
  assert events == [_audio('One.'), AgentSpokenOutput(text='One.')]


async def test_error_after_audio_is_not_retried(synthesizer):
  """Audio already delivered is not synthesized and delivered again."""
  synthesizer.errors_after_audio['One.'] = _api_error(503)
  events = []

  with pytest.raises(errors.ServerError):
    async for event in GeminiTTS()(_text('One.'), cancel=asyncio.Event()):
      events.append(event)

  assert synthesizer.texts == ['One.']
  assert events == [_audio('One.')]


async def test_persistent_transient_error_fails_the_reply(synthesizer):
  """A request still failing after its retries ends the reply."""
  synthesizer.request_errors['One.'] = [_api_error(429) for _ in range(3)]

  with pytest.raises(errors.ClientError):
    await _speak(GeminiTTS(), 'One.')
  assert synthesizer.texts == ['One.'] * 3


async def test_non_transient_error_is_not_retried(synthesizer):
  """A request rejected as invalid is not sent again."""
  synthesizer.request_errors['One.'] = [_api_error(400)]

  with pytest.raises(errors.ClientError):
    await _speak(GeminiTTS(), 'One.')
  assert synthesizer.texts == ['One.']


async def test_reasoner_failure_fails_the_reply(synthesizer):
  """An error from the reasoner's text stream ends the reply with it."""
  del synthesizer

  async def text() -> AsyncIterator[str]:
    yield 'One. '
    raise RuntimeError('reasoner failed')

  with pytest.raises(RuntimeError, match='reasoner failed'):
    async for _ in GeminiTTS()(text(), cancel=asyncio.Event()):
      pass


async def test_failed_warm_up_does_not_block_synthesis(synthesizer):
  """The reply is spoken even if the warm-up request fails."""
  synthesizer.metadata_error = RuntimeError('metadata unavailable')

  events = await _speak(GeminiTTS(), 'One.')

  assert events == [_audio('One.'), AgentSpokenOutput(text='One.')]


async def test_reply_cancelled_before_it_starts_is_not_spoken(synthesizer):
  """A reply already cancelled produces no events."""
  del synthesizer
  cancel = asyncio.Event()
  cancel.set()

  events = await _speak(GeminiTTS(), 'One. Two.', cancel=cancel)

  assert not events


async def test_responses_without_audio_are_skipped(synthesizer):
  """Responses with no candidates or no audio part produce no audio."""
  synthesizer.preamble = [
      types.GenerateContentResponse(candidates=[]),
      types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=types.Content(
                      role='model', parts=[types.Part(text='not audio')]
                  )
              )
          ]
      ),
  ]

  events = await _speak(GeminiTTS(), 'One.')

  assert events == [_audio('One.'), AgentSpokenOutput(text='One.')]


def test_negative_prefetch_is_rejected():
  with pytest.raises(ValueError, match='prefetch_segments'):
    GeminiTTS(prefetch_segments=-1)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'max_segment_chars': 0}, 'max_segment_chars'),
        ({'min_segment_chars': -1}, 'min_segment_chars'),
    ],
)
def test_invalid_segment_lengths_are_rejected(kwargs, message):
  """Lengths that could not split any text are rejected up front."""
  with pytest.raises(ValueError, match=message):
    GeminiTTS(**kwargs)


# --- Splitting the reply into segments ---


def test_sentence_keeps_its_trailing_whitespace():
  """A sentence is split off with the whitespace that followed it."""
  assert _split_segment('Hello world! How are you?', max_chars=100) == (
      'Hello world! ',
      'How are you?',
  )


def test_unfinished_sentence_is_not_split():
  """Text with no boundary waits for more text."""
  assert _split_segment('Hello world', max_chars=100) == (None, 'Hello world')


def test_decimal_point_is_not_a_sentence_end():
  """A period inside a number does not end the sentence."""
  assert _split_segment('It is 3.5 degrees', max_chars=100) == (
      None,
      'It is 3.5 degrees',
  )


def test_cjk_sentence_ends_without_a_space():
  """CJK full-width terminal punctuation ends a sentence on its own."""
  assert _split_segment('今天天气很好。明天下雨', max_chars=100) == (
      '今天天气很好。',
      '明天下雨',
  )


def test_closing_quote_stays_with_its_sentence():
  """A closing quote after the period belongs to the sentence it closes."""
  assert _split_segment('He said "stop." Then', max_chars=100) == (
      'He said "stop." ',
      'Then',
  )


def test_short_sentences_stay_with_the_text_after_them():
  """With `min_chars`, the segment ends at the first late enough boundary."""
  assert _split_segment(
      'Yellow. Y. E. L. L. O. W. Next one. ', max_chars=240, min_chars=20
  ) == ('Yellow. Y. E. L. L. ', 'O. W. Next one. ')


def test_short_sentence_waits_for_more_text():
  """With `min_chars`, an abbreviation does not end the segment."""
  assert _split_segment(
      'Dr. Smith is here. Then', max_chars=240, min_chars=15
  ) == ('Dr. Smith is here. ', 'Then')
  assert _split_segment('Dr. Smith is in', max_chars=240, min_chars=15) == (
      None,
      'Dr. Smith is in',
  )


def test_comma_does_not_end_a_segment():
  """A comma, however late, does not end the segment."""
  assert _split_segment('For October in Paris, you', max_chars=100) == (
      None,
      'For October in Paris, you',
  )


@pytest.mark.parametrize(
    ('buffer', 'max_chars', 'expected'),
    [
        ('one two three four five', 12, ('one two ', 'three four five')),
        ('一二三，四五六七八九', 8, ('一二三，', '四五六七八九')),
        ('abcdefghij', 4, ('abcd', 'efghij')),
    ],
)
def test_long_text_without_boundary_is_cut(buffer, max_chars, expected):
  """Text with no boundary is cut at a space, a CJK clause, or `max_chars`."""
  assert _split_segment(buffer, max_chars=max_chars) == expected

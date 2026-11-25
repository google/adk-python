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

"""A classifier backed by the Gemini Decisions API."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
import json
import math
import os
from typing import Any
from typing import ClassVar
import urllib.parse

from google.genai import types
import httpx
from pydantic import Field
import tenacity
from typing_extensions import override

from ...labs.classifiers._base_classifier import BaseClassifier
from ...labs.classifiers._types import Answer
from ...labs.classifiers._types import Binary
from ...labs.classifiers._types import BinaryAnswer
from ...labs.classifiers._types import Choice
from ...labs.classifiers._types import ChoiceAnswer
from ...labs.classifiers._types import ClassifierInput
from ...labs.classifiers._types import ClassifierResponse
from ...labs.classifiers._types import JSONContent
from ...labs.classifiers._types import Question
from ...labs.classifiers._types import RefusalAnswer
from ...labs.classifiers._types import Score
from ...labs.classifiers._types import ScoreAnswer
from ...utils import _mtls_utils
from ...utils._google_client_headers import get_tracking_headers

# With a client certificate, _mtls_utils sends this to its
# generativelanguage.mtls.googleapis.com host.
_DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
_PATH = '/v1beta/decisions'
_API_KEY_ENVS = ('GEMINI_API_KEY', 'GOOGLE_API_KEY')
_DEFAULT_TIMEOUT = 30.0
# The most options and levels the API accepts in one question.
_MAX_OPTIONS = 255

# Rate-limited (429) and server-side failures are worth another attempt.
_RETRY_WAIT = tenacity.wait_exponential_jitter(initial=0.5, max=8.0)
# The longest a retry-after header can make a retry wait.
_MAX_RETRY_AFTER = 30.0
# Other transport errors, such as a bad URL or proxy, fail the same way on
# every attempt.
_RETRY_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
# Calibrated values can land a rounding error outside their range.
_TOLERANCE = 1e-6

# The error codes of an HTTP 400 that says the safety filters blocked the input.
_SAFETY_ERROR_CODES = frozenset({
    'safety',
    'prohibited_content',
    'blocklist',
    'image_safety',
    'content_blocked',
})

# The media the API accepts in an input block, by block type.
_MEDIA_MIME_TYPES = {
    'image': frozenset({
        'image/png',
        'image/jpeg',
        'image/webp',
        'image/heic',
        'image/heif',
        'image/gif',
        'image/bmp',
        'image/tiff',
    }),
    'audio': frozenset({
        'audio/wav',
        'audio/mp3',
        'audio/mpeg',
        'audio/aiff',
        'audio/aac',
        'audio/ogg',
        'audio/flac',
        'audio/m4a',
        'audio/opus',
        'audio/webm',
        'audio/l16',
        'audio/alaw',
        'audio/mulaw',
    }),
    'video': frozenset({
        'video/mp4',
        'video/mpeg',
        'video/mov',
        'video/avi',
        'video/webm',
        'video/wmv',
        'video/x-flv',
        'video/3gpp',
    }),
    'document': frozenset({'application/pdf'}),
}
_BLOCK_TYPE_BY_MIME_TYPE = {
    mime_type: block_type
    for block_type, mime_types in _MEDIA_MIME_TYPES.items()
    for mime_type in mime_types
}


class _RetryableStatusError(Exception):

  def __init__(self, response: httpx.Response):
    super().__init__(response.status_code)
    self.response = response


def _render(content: JSONContent) -> str:
  if isinstance(content, str):
    return content
  return json.dumps(content, ensure_ascii=False)


def _media_block(part: types.Part) -> dict[str, Any]:
  """Builds the input block for a part that carries inline media."""
  blob = part.inline_data
  if blob is None or not blob.data or not blob.mime_type:
    raise ValueError(
        'GeminiDecisionsClassifier accepts text, JSON and inline media only;'
        ' the input has a part that is none of these.'
    )
  mime_type = blob.mime_type.split(';')[0].strip().lower()
  block_type = _BLOCK_TYPE_BY_MIME_TYPE.get(mime_type)
  if block_type is None:
    raise ValueError(
        f'Gemini Decision 1 does not accept {blob.mime_type!r} input. It'
        ' accepts text, images, audio, video and PDF documents.'
    )
  return {
      'type': block_type,
      'data': base64.b64encode(blob.data).decode('ascii'),
      'mime_type': mime_type,
  }


def _input_to_wire(input: ClassifierInput) -> str | list[dict[str, Any]]:
  """Builds the request's input: a string, or content blocks for media."""
  if not isinstance(input, types.Content):
    # JSON goes as text so its keys stay plain data, not content blocks.
    wire: str | list[dict[str, Any]] = _render(input)
  elif all(part.text is not None for part in input.parts or []):
    # The API joins adjacent text blocks with no separator, as this does.
    wire = ''.join(part.text for part in input.parts or [])
  else:
    wire = [
        {'type': 'text', 'text': part.text}
        if part.text is not None
        else _media_block(part)
        for part in input.parts or []
    ]
  if not wire:
    raise ValueError('The API does not accept empty input.')
  return wire


def _check_option_count(name: str, options: str, count: int) -> None:
  if count > _MAX_OPTIONS:
    raise ValueError(
        f'Gemini Decision 1 accepts at most {_MAX_OPTIONS} {options} in a'
        f' question; question {name!r} has {count}.'
    )


def _question_to_wire(name: str, question: Question) -> dict[str, Any]:
  if question.instructions is None:
    raise ValueError(
        f'GeminiDecisionsClassifier needs instructions for question {name!r}.'
    )
  body: dict[str, Any] = {'instructions': _render(question.instructions)}
  if isinstance(question, Choice):
    _check_option_count(name, 'labels', len(question.criteria))
    body['type'] = 'choice'
    body['criteria'] = {
        label: '' if description is None else _render(description)
        for label, description in question.criteria.items()
    }
  elif isinstance(question, Binary):
    body['type'] = 'noul'
    if question.criteria is not None:
      body['criteria'] = {'true_description': _render(question.criteria)}
  elif isinstance(question, Score):
    _check_option_count(name, 'levels', len(question.criteria))
    body['type'] = 'score'
    body['criteria'] = [_render(level) for level in question.criteria]
  else:
    raise TypeError(f'Unsupported question type: {type(question).__name__}.')
  return body


def _read_number(name: str, key: str, value: Any, high: float = 1.0) -> float:
  """Reads a number in [0, high], clamping rounding errors at either end."""
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(value)
  ):
    raise ValueError(
        f'Gemini returned {value!r} for {key!r} in {name!r}, not a number.'
    )
  if not -_TOLERANCE <= value <= high + _TOLERANCE:
    raise ValueError(
        f'Gemini returned {value} for {key!r} in {name!r}, outside [0, {high}].'
    )
  return min(max(float(value), 0.0), high)


def _read_probabilities(
    name: str, raw: Mapping[str, Any], keys: list[str]
) -> dict[str, float]:
  probabilities = raw['probabilities']
  if not isinstance(probabilities, Mapping):
    raise ValueError(
        f'Gemini returned a malformed {raw["type"]!r} answer to {name!r},'
        ' whose probabilities are not an object.'
    )
  if set(probabilities) != set(keys):
    raise ValueError(
        f'Gemini returned probabilities for {sorted(probabilities)} in'
        f' {name!r}, expected {keys}.'
    )
  return {key: _read_number(name, key, probabilities[key]) for key in keys}


def _to_answer(name: str, question: Question, raw: Any) -> Answer:
  if not isinstance(raw, Mapping):
    raise ValueError(f'Gemini returned no answer to {name!r}.')
  expected = 'noul' if isinstance(question, Binary) else question.type
  if raw.get('type') != expected:
    raise ValueError(
        f'Gemini returned a {raw.get("type")!r} answer to {name!r}, which is'
        f' a {expected!r} question.'
    )
  required = [expected] if expected == 'noul' else [expected, 'probabilities']
  missing = [key for key in required if key not in raw]
  if missing:
    raise ValueError(
        f'Gemini returned a malformed {expected!r} answer to {name!r},'
        f' missing {missing}.'
    )
  confidence = raw.get('confidence')
  if confidence is not None:
    confidence = _read_number(name, 'confidence', confidence)

  if isinstance(question, Binary):
    return BinaryAnswer(
        probability=_read_number(name, 'noul', raw['noul']),
        confidence=confidence,
    )
  if isinstance(question, Choice):
    choice = raw['choice']
    if not isinstance(choice, str) or choice not in question.criteria:
      raise ValueError(
          f'Gemini chose {choice!r} for {name!r}, which is not one of its'
          ' labels.'
      )
    return ChoiceAnswer(
        choice=choice,
        probabilities=_read_probabilities(name, raw, list(question.criteria)),
        confidence=confidence,
    )
  top = len(question.criteria) - 1
  probabilities = _read_probabilities(
      name, raw, [str(level) for level in range(top + 1)]
  )
  return ScoreAnswer(
      score=_read_number(name, 'score', raw['score'], high=top),
      probabilities={int(level): p for level, p in probabilities.items()},
      confidence=confidence,
  )


def _is_safety_refusal(response: httpx.Response) -> bool:
  """Whether the response says that the safety filters blocked the input."""
  if response.status_code != 400:
    return False
  try:
    error = response.json().get('error')
  except (ValueError, AttributeError):
    return False
  return isinstance(error, Mapping) and error.get('code') in _SAFETY_ERROR_CODES


def _retry_after(response: httpx.Response) -> float | None:
  """Reads a retry-after header given in seconds, if there is one."""
  try:
    seconds = float(response.headers['retry-after'])
  except (KeyError, ValueError):
    return None
  return (
      min(max(seconds, 0.0), _MAX_RETRY_AFTER)
      if math.isfinite(seconds)
      else None
  )


def _wait(retry_state: tenacity.RetryCallState) -> float:
  """Waits as long as the API asks, or else backs off with jitter."""
  error = retry_state.outcome.exception() if retry_state.outcome else None
  if isinstance(error, _RetryableStatusError):
    delay = _retry_after(error.response)
    if delay is not None:
      return delay
  return _RETRY_WAIT(retry_state)


def _is_google_api_url(url: str) -> bool:
  parsed = urllib.parse.urlparse(url)
  host = parsed.hostname or ''
  return parsed.scheme == 'https' and (
      host == 'googleapis.com' or host.endswith('.googleapis.com')
  )


class GeminiDecisionsClassifier(BaseClassifier):
  """Answers typed questions with Gemini Decision 1.

  Gemini Decision 1 returns calibrated probabilities from a single forward
  pass rather than generating text, so it is much faster than asking an LLM.
  Every question needs ``instructions``; ``Binary`` questions are sent as
  ``noul`` (yes/no) questions. If the API's safety filters block the input,
  every question gets a ``RefusalAnswer``.

  Notes on the answers:

  * The API sorts the keys of a ``Choice`` alphabetically before it scores
    them, so name a catch-all label ``other``, not ``none_of_the_above``.
  * ``confidence`` is the probability of the top answer, ``max(p, 1 - p)`` for
    a ``Binary``.
  * Treat the input as untrusted. Text in it can argue for its own
    classification, so do not make a classifier the only gate on an action that
    changes access or moves money.

  Usage::

      from google.adk.labs.classifiers import Choice
      from google.adk.integrations.gemini_decisions import (
          GeminiDecisionsClassifier,
      )

      classifier = GeminiDecisionsClassifier()  # reads GEMINI_API_KEY
      response = await classifier.classify(
          input='My invoice was charged twice.',
          questions={
              'department': Choice(
                  instructions='Which team should handle this ticket?',
                  criteria={'billing': None, 'technical': None},
              ),
          },
      )
      response.answers['department'].choice  # 'billing'
  """

  _CLIENT_LABEL_KIND: ClassVar[str] = 'gemini_decisions'

  model: str = 'gemini-decision-1'
  """The Gemini Decisions model."""

  api_key: str | None = Field(default=None, exclude=True, repr=False)
  """The API key. Defaults to the ``GEMINI_API_KEY`` environment variable,
  then ``GOOGLE_API_KEY``."""

  base_url: str | None = None
  """The API base URL. Defaults to the base URL of ``http_client``, then
  ``https://generativelanguage.googleapis.com``."""

  timeout: float | None = None
  """Seconds to wait for each request. Defaults to the timeout of
  ``http_client``, or 30 seconds without one."""

  max_attempts: int = Field(default=3, ge=1)
  """Attempts per call, counting the first. Rate-limited (429) and server
  error (5xx) responses, timeouts and dropped connections are retried with
  backoff, or after the delay in a ``retry-after`` header."""

  http_client: httpx.AsyncClient | None = Field(
      default=None, exclude=True, repr=False
  )
  """A client to send requests with, such as one configured with a proxy.

  When unset, each call opens and closes its own client.
  """

  def _default_url(self) -> str:
    return (self.base_url or _DEFAULT_BASE_URL).rstrip('/') + _PATH

  def _url(self, client: httpx.AsyncClient) -> str:
    if self.base_url is None and str(client.base_url):
      return _PATH
    return self._default_url()

  def _headers(self, url: str) -> dict[str, str]:
    api_key = self.api_key or next(
        (os.environ[env] for env in _API_KEY_ENVS if os.environ.get(env)), None
    )
    if not api_key:
      raise ValueError(
          'GeminiDecisionsClassifier needs an API key. Set'
          f' {" or ".join(_API_KEY_ENVS)} or pass api_key.'
      )
    # The hosting runtime is sent to Google APIs only.
    headers = get_tracking_headers(
        exclude_runtime_tag=not _is_google_api_url(url)
    )
    headers['x-goog-api-key'] = api_key
    return headers

  async def _post(
      self, client: httpx.AsyncClient, body: dict[str, Any]
  ) -> httpx.Response:
    url = self._url(client)
    response = await client.post(
        url,
        json=body,
        headers=self._headers(str(client.base_url.join(url))),
        timeout=(
            httpx.USE_CLIENT_DEFAULT if self.timeout is None else self.timeout
        ),
    )
    if response.status_code == 429 or response.status_code >= 500:
      raise _RetryableStatusError(response)
    return response

  async def _send(
      self, client: httpx.AsyncClient, body: dict[str, Any]
  ) -> httpx.Response:
    try:
      return await tenacity.AsyncRetrying(
          stop=tenacity.stop_after_attempt(self.max_attempts),
          wait=_wait,
          retry=tenacity.retry_if_exception_type(
              (_RetryableStatusError, *_RETRY_ERRORS)
          ),
          reraise=True,
      )(self._post, client, body)
    except _RetryableStatusError as e:
      return e.response

  @override
  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    """Asks Gemini every question in one Decisions request.

    Args:
      input: What the questions are about: text, JSON, or a ``types.Content``
        whose parts are text or inline images, audio, video or PDF documents.
      questions: Maps a caller-chosen name to each question. Every question
        needs ``instructions``.

    Returns:
      One answer per question, keyed by the question's name. If the safety
      filters block the input, every answer is a ``RefusalAnswer``.

    Raises:
      ValueError: Before the request, if no API key is set, ``questions`` is
        empty, a question has no instructions or too many labels or levels,
        or the input is empty or has a part the model does not accept. After
        the request, if the response does not answer the questions that were
        asked.
      RuntimeError: If the API responds with an error status, including one
        that is still returned after the last attempt.
    """
    if not questions:
      raise ValueError('classify() needs at least one question.')
    body = {
        'model': self.model,
        'input': _input_to_wire(input),
        'questions': {
            name: _question_to_wire(name, question)
            for name, question in questions.items()
        },
    }
    if self.http_client is not None:
      response = await self._send(self.http_client, body)
    else:
      # Empty unless the environment has a client certificate to present.
      mtls_kwargs = await asyncio.to_thread(
          _mtls_utils.client_cert_httpx_kwargs, url=self._default_url()
      )
      async with httpx.AsyncClient(
          timeout=_DEFAULT_TIMEOUT, **mtls_kwargs
      ) as client:
        response = await self._send(client, body)
    if _is_safety_refusal(response):
      return ClassifierResponse(
          answers={name: RefusalAnswer() for name in questions}
      )
    if response.is_error:
      raise RuntimeError(
          f'Gemini Decisions request failed with {response.status_code}:'
          f' {response.text[:500]}'
      )

    data = response.json()
    if not isinstance(data, Mapping):
      raise ValueError('Gemini did not return a JSON object.')
    raw_answers = data.get('answers')
    if not isinstance(raw_answers, Mapping):
      raise ValueError('Gemini returned no answers.')
    return ClassifierResponse(
        # The API does not report a model version today.
        model_version=data.get('modelVersion') or data.get('model_version'),
        answers={
            name: _to_answer(name, question, raw_answers.get(name))
            for name, question in questions.items()
        },
    )

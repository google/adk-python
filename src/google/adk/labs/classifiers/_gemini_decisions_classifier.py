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
from collections.abc import Collection
from collections.abc import Mapping
import json
import os
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import TypeVar

from google.genai import types
import httpx
import pydantic
from pydantic import Field
from typing_extensions import override

from ...utils._event_loop_cache import PerLoopCachedProperty
from ...utils._google_client_headers import get_tracking_headers
from ...utils._mtls_utils import client_cert_httpx_kwargs
from ._base_classifier import BaseClassifier
from ._types import Answer
from ._types import Binary
from ._types import BinaryAnswer
from ._types import Choice
from ._types import ChoiceAnswer
from ._types import ClassifierInput
from ._types import ClassifierResponse
from ._types import JSONContent
from ._types import Question
from ._types import RefusalAnswer
from ._types import ScoreAnswer

_DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
_API_KEY_ENVS = ('GOOGLE_API_KEY', 'GEMINI_API_KEY')
_PATH = '/v1beta/decisions'
# A classifier usually runs in the path of a turn, so it gives up sooner than
# httpx's default would suggest.
_DEFAULT_TIMEOUT = 30.0
# The Gemini Developer API reports blocked input as an HTTP 400 with one of
# these error codes. Vertex AI reports it as `prompt_feedback` instead.
_BLOCKED_CODES = frozenset({
    'safety',
    'prohibited_content',
    'blocklist',
    'image_safety',
    'jailbreak',
    'model_armor',
    'content_blocked',
})
# Lets rounding in the API put a probability a hair outside [0, 1].
_TOLERANCE = 1e-6

_K = TypeVar('_K', str, int)


class _WireAnswer(pydantic.BaseModel):
  """Common fields of every answer the API returns."""

  model_config = pydantic.ConfigDict(extra='ignore')

  confidence: float


class _NoulAnswer(_WireAnswer):
  type: Literal['noul']
  noul: float


class _ChoiceAnswer(_WireAnswer):
  type: Literal['choice']
  choice: str
  probabilities: dict[str, float]


class _ScoreAnswer(_WireAnswer):
  type: Literal['score']
  score: float
  probabilities: dict[int, float]


class _Response(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(extra='ignore')

  answers: dict[
      str,
      Annotated[
          _NoulAnswer | _ChoiceAnswer | _ScoreAnswer,
          Field(discriminator='type'),
      ],
  ] = {}
  prompt_feedback: dict[str, Any] | None = None


def _render(content: JSONContent) -> str:
  if isinstance(content, str):
    return content
  return json.dumps(content, ensure_ascii=False)


def _media_block(mime_type: str | None) -> dict[str, Any]:
  if not mime_type:
    raise ValueError(
        'GeminiDecisionsClassifier needs a MIME type on every inline data and'
        ' file part.'
    )
  mime_type = mime_type.lower()
  prefix = mime_type.split('/', 1)[0]
  return {
      'type': prefix if prefix in ('image', 'audio', 'video') else 'document',
      'mime_type': mime_type,
  }


def _part_to_json(part: types.Part) -> dict[str, Any]:
  if part.text is not None:
    return {'type': 'text', 'text': part.text}
  blob = part.inline_data
  if blob is not None and blob.data is not None:
    return {
        **_media_block(blob.mime_type),
        'data': base64.b64encode(blob.data).decode('ascii'),
    }
  file = part.file_data
  if file is not None and file.file_uri is not None:
    return {**_media_block(file.mime_type), 'uri': file.file_uri}
  raise ValueError(
      'GeminiDecisionsClassifier accepts text, inline data and file parts'
      ' only; the input has another kind of part.'
  )


def _input_to_json(input: ClassifierInput) -> str | list[dict[str, Any]]:
  if not isinstance(input, types.Content):
    return _render(input)
  return [_part_to_json(part) for part in input.parts or [] if not part.thought]


def _question_to_json(name: str, question: Question) -> dict[str, Any]:
  if question.instructions is None:
    raise ValueError(
        f'GeminiDecisionsClassifier needs instructions for question {name!r}.'
    )
  body: dict[str, Any] = {'instructions': _render(question.instructions)}
  if isinstance(question, Choice):
    body['type'] = 'choice'
    body['criteria'] = {
        label: '' if description is None else _render(description)
        for label, description in question.criteria.items()
    }
  elif isinstance(question, Binary):
    body['type'] = 'noul'
    if question.criteria is not None:
      body['criteria'] = {'true_description': _render(question.criteria)}
  else:
    body['type'] = 'score'
    body['criteria'] = [_render(level) for level in question.criteria]
  return body


def _is_blocked(response: httpx.Response) -> bool:
  if response.status_code != 400:
    return False
  try:
    error = response.json().get('error')
  except (ValueError, AttributeError):
    return False
  return isinstance(error, Mapping) and error.get('code') in _BLOCKED_CODES


def _in_range(name: str, field: str, value: float, high: float) -> float:
  if not -_TOLERANCE <= value <= high + _TOLERANCE:
    raise ValueError(
        f'Gemini returned {field} {value} for {name!r}, outside [0, {high}].'
    )
  return min(max(value, 0.0), high)


def _probabilities(
    name: str, raw: Mapping[_K, float], keys: Collection[_K]
) -> dict[_K, float]:
  if set(raw) != set(keys):
    raise ValueError(
        f'Gemini returned probabilities for {sorted(raw)} in {name!r}, not for'
        f' its options {sorted(keys)}.'
    )
  return {key: _in_range(name, 'probability', raw[key], 1.0) for key in keys}


def _to_answer(name: str, question: Question, raw: Any) -> Answer:
  if isinstance(question, Binary) and isinstance(raw, _NoulAnswer):
    return BinaryAnswer(
        probability=_in_range(name, 'probability', raw.noul, 1.0),
        confidence=_in_range(name, 'confidence', raw.confidence, 1.0),
    )
  if isinstance(question, Choice) and isinstance(raw, _ChoiceAnswer):
    if raw.choice not in question.criteria:
      raise ValueError(
          f'Gemini chose {raw.choice!r} for {name!r}, which is not one of its'
          ' labels.'
      )
    return ChoiceAnswer(
        choice=raw.choice,
        probabilities=_probabilities(
            name, raw.probabilities, list(question.criteria)
        ),
        confidence=_in_range(name, 'confidence', raw.confidence, 1.0),
    )
  if isinstance(question, (Binary, Choice)) or not isinstance(
      raw, _ScoreAnswer
  ):
    raise ValueError(
        f'Gemini returned {type(raw).__name__} for {name!r}, which is a'
        f' {question.type!r} question.'
    )
  top = len(question.criteria) - 1
  return ScoreAnswer(
      score=_in_range(name, 'score', raw.score, top),
      probabilities=_probabilities(
          name, raw.probabilities, range(len(question.criteria))
      ),
      confidence=_in_range(name, 'confidence', raw.confidence, 1.0),
  )


class _ClientSlot:
  """One event loop's default client and the TLS context it was built with."""

  client: httpx.AsyncClient | None = None
  verify: Any = None


class GeminiDecisionsClassifier(BaseClassifier):
  """Answers typed questions with the Gemini Decisions API.

  The Decisions API scores every question about an input in one request and
  returns calibrated probabilities instead of generated text. ``Binary``
  questions are sent as ``noul`` questions, and every question needs
  ``instructions``. Input can be text, JSON, or a ``types.Content`` with text,
  image, audio, video or document parts. If Gemini blocks the input, every
  answer is a ``RefusalAnswer``.

  The API is served by the Gemini Developer API only, so the classifier needs
  an API key and does not use Vertex AI.

  Usage::

      from google.adk.labs.classifiers import Choice
      from google.adk.labs.classifiers import GeminiDecisionsClassifier

      classifier = GeminiDecisionsClassifier()  # reads GOOGLE_API_KEY
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

  # Pydantic exempts functools.cached_property by module name rather than by
  # type, so the per-loop descriptor has to be named here.
  model_config = pydantic.ConfigDict(ignored_types=(PerLoopCachedProperty,))

  _CLIENT_LABEL_KIND: ClassVar[str] = 'gemini'

  model: str = 'gemini-decision-1'
  """The Decisions model."""

  api_key: str | None = Field(default=None, exclude=True, repr=False)
  """The API key. Defaults to the ``GOOGLE_API_KEY`` environment variable,
  then ``GEMINI_API_KEY``."""

  base_url: str | None = None
  """The API base URL. Defaults to the base URL of ``http_client``, then
  ``https://generativelanguage.googleapis.com``."""

  timeout: float | None = None
  """Seconds to wait for each request. Defaults to the timeout of
  ``http_client``, or 30 seconds without one."""

  http_client: httpx.AsyncClient | None = Field(
      default=None, exclude=True, repr=False
  )
  """A client to send requests with, such as one configured with a proxy.

  When unset, the classifier reuses one client per event loop, which presents
  the default client certificate and uses the ``.mtls.googleapis.com`` host
  when one is configured. The client is rebuilt when the certificate is
  reloaded.
  """

  @PerLoopCachedProperty
  def _client_slot(self) -> _ClientSlot:
    return _ClientSlot()

  async def _client(self) -> httpx.AsyncClient:
    if self.http_client is not None:
      return self.http_client
    # Loading the certificate can run a cert provider, so keep it off the
    # event loop. The loaded context is reused for a few minutes, and a new
    # one means the certificate rotated.
    kwargs = await asyncio.to_thread(
        client_cert_httpx_kwargs, url=self.base_url or _DEFAULT_BASE_URL
    )
    slot = self._client_slot
    if slot.client is None or slot.verify is not kwargs.get('verify'):
      slot.client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, **kwargs)
      slot.verify = kwargs.get('verify')
    return slot.client

  def _url(self, client: httpx.AsyncClient) -> str:
    if self.base_url is None and str(client.base_url):
      return _PATH
    return (self.base_url or _DEFAULT_BASE_URL).rstrip('/') + _PATH

  def _headers(self) -> dict[str, str]:
    api_key = self.api_key or next(
        (os.environ[name] for name in _API_KEY_ENVS if os.environ.get(name)),
        None,
    )
    if not api_key:
      raise ValueError(
          'GeminiDecisionsClassifier needs an API key. Set GOOGLE_API_KEY or'
          ' pass api_key.'
      )
    return {'x-goog-api-key': api_key, **get_tracking_headers()}

  async def _post(
      self,
      client: httpx.AsyncClient,
      body: dict[str, Any],
      headers: dict[str, str],
  ) -> httpx.Response:
    return await client.post(
        self._url(client),
        json=body,
        headers=headers,
        timeout=(
            httpx.USE_CLIENT_DEFAULT if self.timeout is None else self.timeout
        ),
    )

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
        whose parts are text, inline data or file URIs. Thought parts are
        skipped.
      questions: Maps a name to each question. Every question needs
        ``instructions``.

    Returns:
      One answer per question, keyed by the question's name. If Gemini blocked
      the input, every answer is a ``RefusalAnswer``.

    Raises:
      ValueError: If `questions` is empty, no API key is set, a question has
        no instructions, the input has an unsupported part or a media part
        without a MIME type, or an answer does not match its question.
      RuntimeError: If the request fails, including a request over the API's
        limits.
    """
    if not questions:
      raise ValueError('classify() needs at least one question.')
    body = {
        'model': self.model,
        'input': _input_to_json(input),
        'questions': {
            name: _question_to_json(name, question)
            for name, question in questions.items()
        },
    }
    headers = self._headers()
    client = await self._client()
    response = await self._post(client, body, headers)

    if _is_blocked(response):
      return ClassifierResponse(
          answers={name: RefusalAnswer() for name in questions}
      )
    if response.is_error:
      raise RuntimeError(
          f'Gemini Decisions request failed with {response.status_code}:'
          f' {response.text[:500]}'
      )
    data = _Response.model_validate(response.json())
    if (data.prompt_feedback or {}).get('block_reason'):
      return ClassifierResponse(
          answers={name: RefusalAnswer() for name in questions}
      )
    return ClassifierResponse(
        answers={
            name: _to_answer(name, question, data.answers.get(name))
            for name, question in questions.items()
        },
    )

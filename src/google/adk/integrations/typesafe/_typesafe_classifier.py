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

"""A classifier backed by TypeSafe's Jev decision models."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any

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
from ...labs.classifiers._types import ScoreAnswer

_DEFAULT_BASE_URL = 'https://api.typesafe.ai'
_API_KEY_ENV = 'TYPESAFE_API_KEY'
_BASE_URL_ENV = 'TYPESAFE_BASE_URL'
_PATH = '/v1/systemone'
_DEFAULT_TIMEOUT = 30.0

# The API asks clients to back off and retry on these.
_RETRY_STATUS_CODES = frozenset({429, 529})
_RETRY_WAIT = tenacity.wait_exponential_jitter(initial=0.5, max=8.0)
# Other transport errors, such as a bad URL or proxy, fail the same way on
# every attempt.
_RETRY_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
# Calibrated values can land a rounding error outside their range.
_TOLERANCE = 1e-6


class _RetryableStatusError(Exception):

  def __init__(self, response: httpx.Response):
    super().__init__(response.status_code)
    self.response = response


def _input_to_json(input: ClassifierInput) -> JSONContent:
  if not isinstance(input, types.Content):
    return input
  texts = []
  for part in input.parts or []:
    if part.text is None:
      raise ValueError(
          'TypesafeClassifier accepts text and JSON input only; the input has'
          ' a non-text part.'
      )
    texts.append(part.text)
  return ''.join(texts)


def _question_to_json(name: str, question: Question) -> dict[str, Any]:
  if question.instructions is None:
    raise ValueError(
        f'TypesafeClassifier needs instructions for question {name!r}.'
    )
  body: dict[str, Any] = {'instructions': question.instructions}
  if isinstance(question, Choice):
    body['type'] = 'choice'
    body['criteria'] = dict(question.criteria)
  elif isinstance(question, Binary):
    body['type'] = 'noul'
    if question.criteria is not None:
      body['criteria'] = {'true': question.criteria}
  else:
    body['type'] = 'score'
    body['criteria'] = list(question.criteria)
  return body


def _read_number(name: str, key: str, value: Any, high: float = 1.0) -> float:
  """Reads a number in [0, high], clamping rounding errors at either end."""
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(value)
  ):
    raise ValueError(
        f'TypeSafe returned {value!r} for {key!r} in {name!r}, not a number.'
    )
  if not -_TOLERANCE <= value <= high + _TOLERANCE:
    raise ValueError(
        f'TypeSafe returned {value} for {key!r} in {name!r}, outside'
        f' [0, {high}].'
    )
  return min(max(float(value), 0.0), high)


def _read_probabilities(
    name: str, raw: Mapping[str, Any], keys: list[str]
) -> dict[str, float]:
  probabilities = raw['probabilities']
  if not isinstance(probabilities, Mapping):
    raise ValueError(
        f'TypeSafe returned a malformed {raw["type"]!r} answer to {name!r},'
        ' whose probabilities are not an object.'
    )
  if set(probabilities) != set(keys):
    raise ValueError(
        f'TypeSafe returned probabilities for {sorted(probabilities)} in'
        f' {name!r}, expected {keys}.'
    )
  return {key: _read_number(name, key, probabilities[key]) for key in keys}


def _to_answer(name: str, question: Question, raw: Any) -> Answer:
  if not isinstance(raw, Mapping):
    raise ValueError(f'TypeSafe returned no answer to {name!r}.')
  expected = 'noul' if isinstance(question, Binary) else question.type
  if raw.get('type') != expected:
    raise ValueError(
        f'TypeSafe returned a {raw.get("type")!r} answer to {name!r}, which'
        f' is a {expected!r} question.'
    )
  required = [expected] if expected == 'noul' else [expected, 'probabilities']
  missing = [key for key in required if key not in raw]
  if missing:
    raise ValueError(
        f'TypeSafe returned a malformed {expected!r} answer to {name!r},'
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
          f'TypeSafe chose {choice!r} for {name!r}, which is not one of its'
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


class TypesafeClassifier(BaseClassifier):
  """Answers typed questions with TypeSafe's Jev models.

  Jev returns calibrated probabilities from a single forward pass rather than
  generating text, so it is much faster than asking an LLM. Every question
  needs ``instructions``; ``Binary`` questions are sent as Jev's ``noul``
  questions.

  Usage::

      from google.adk.labs.classifiers import Choice
      from google.adk.integrations.typesafe import TypesafeClassifier

      classifier = TypesafeClassifier()  # reads TYPESAFE_API_KEY
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

  model: str = 'jev-latest'
  """The TypeSafe model, such as ``jev-latest`` or a pinned ``jev-1.13``."""

  api_key: str | None = Field(default=None, exclude=True, repr=False)
  """The API key. Defaults to the ``TYPESAFE_API_KEY`` environment variable."""

  base_url: str | None = None
  """The API base URL. Defaults to the base URL of ``http_client``, then
  ``TYPESAFE_BASE_URL``, then ``https://api.typesafe.ai``."""

  timeout: float | None = None
  """Seconds to wait for each request. Defaults to the timeout of
  ``http_client``, or 30 seconds without one."""

  max_attempts: int = Field(default=3, ge=1)
  """Attempts per call, counting the first. Rate-limited (429) and overloaded
  (529) responses, timeouts and dropped connections are retried with
  backoff."""

  http_client: httpx.AsyncClient | None = Field(
      default=None, exclude=True, repr=False
  )
  """A client to send requests with, such as one configured with a proxy.

  When unset, each call opens and closes its own client.
  """

  def _url(self, client: httpx.AsyncClient) -> str:
    if self.base_url is None and str(client.base_url):
      return _PATH
    base = self.base_url or os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL
    return base.rstrip('/') + _PATH

  def _headers(self) -> dict[str, str]:
    api_key = self.api_key or os.environ.get(_API_KEY_ENV)
    if not api_key:
      raise ValueError(
          f'TypesafeClassifier needs an API key. Set {_API_KEY_ENV} or pass'
          ' api_key.'
      )
    return {'Authorization': f'Bearer {api_key}'}

  async def _post(
      self, client: httpx.AsyncClient, body: dict[str, Any]
  ) -> httpx.Response:
    response = await client.post(
        self._url(client),
        json=body,
        headers=self._headers(),
        timeout=(
            httpx.USE_CLIENT_DEFAULT if self.timeout is None else self.timeout
        ),
    )
    if response.status_code in _RETRY_STATUS_CODES:
      raise _RetryableStatusError(response)
    return response

  async def _send(
      self, client: httpx.AsyncClient, body: dict[str, Any]
  ) -> httpx.Response:
    try:
      return await tenacity.AsyncRetrying(
          stop=tenacity.stop_after_attempt(self.max_attempts),
          wait=_RETRY_WAIT,
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
    body = {
        'model': self.model,
        'state': _input_to_json(input),
        'questions': {
            name: _question_to_json(name, question)
            for name, question in questions.items()
        },
    }
    if self.http_client is not None:
      response = await self._send(self.http_client, body)
    else:
      async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await self._send(client, body)
    if response.is_error:
      raise RuntimeError(
          f'TypeSafe request failed with {response.status_code}:'
          f' {response.text[:500]}'
      )

    data = response.json()
    if not isinstance(data, Mapping):
      raise ValueError('TypeSafe did not return a JSON object.')
    raw_answers = data.get('answers')
    if not isinstance(raw_answers, Mapping):
      raise ValueError('TypeSafe returned no answers.')
    return ClassifierResponse(
        model_version=data.get('model'),
        answers={
            name: _to_answer(name, question, raw_answers.get(name))
            for name, question in questions.items()
        },
    )

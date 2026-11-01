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

"""A classifier backed by the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
import json
import math
from typing import Any
from typing import TypeVar

from google.genai import types

try:
  from openai import AsyncOpenAI
except ImportError as e:
  raise ImportError(
      "The 'openai' package is not installed. Please install it with "
      '`pip install "google-adk[openai]"` to use the'
      ' OpenAIDecisionsClassifier.'
  ) from e

from pydantic import Field
from pydantic import PrivateAttr
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
from ._openai_common import build_api_key

# The openai package versions ADK supports have no typed Decisions resource,
# so requests go through the client's generic post() under this path.
_PATH = '/decisions'
# A classifier usually runs in the path of a turn, so the client it builds
# gives up much sooner than the openai package's 10-minute default.
_DEFAULT_TIMEOUT = 30.0
# The fewest and most options the API accepts in one question.
_CHOICE_LIMITS = (2, 255)
_SCORE_LIMITS = (2, 10)
# Calibrated values can land a rounding error outside their range.
_TOLERANCE = 1e-6

_K = TypeVar('_K', str, int)


def _render(content: JSONContent) -> str:
  if isinstance(content, str):
    return content
  return json.dumps(content, ensure_ascii=False)


def _input_to_text(input: ClassifierInput) -> str:
  """Renders the input as the text that the request sends."""
  if isinstance(input, types.Content):
    texts = []
    for part in input.parts or []:
      if part.text is None:
        raise ValueError(
            'OpenAIDecisionsClassifier accepts text and JSON input only; the'
            ' input has a non-text part.'
        )
      texts.append(part.text)
    return ''.join(texts)
  # JSON goes as text rather than as an input_json part. OpenAI renders
  # input_json as text too, but it holds only objects and caps their size.
  return _render(input)


def _check_option_count(
    name: str, options: str, count: int, limits: tuple[int, int]
) -> None:
  low, high = limits
  if not low <= count <= high:
    raise ValueError(
        f'OpenAI accepts {low} to {high} {options} in a question; question'
        f' {name!r} has {count}.'
    )


def _question_to_wire(name: str, question: Question) -> dict[str, Any]:
  """Builds the OpenAI form of one question, named after its key."""
  if question.instructions is None:
    raise ValueError(
        f'OpenAIDecisionsClassifier needs instructions for question {name!r}.'
    )
  instructions = _render(question.instructions)
  if isinstance(question, Binary):
    # A predicate has no criteria field, so the criteria go at the end of the
    # instructions.
    if question.criteria is not None:
      instructions += (
          f'\nTrue means: {_render(question.criteria)}\nAnything else is false.'
      )
    return {'type': 'predicate', 'name': name, 'instructions': instructions}
  if isinstance(question, Choice):
    _check_option_count(name, 'labels', len(question.criteria), _CHOICE_LIMITS)
    choices = []
    for label, description in question.criteria.items():
      choice: dict[str, Any] = {'value': label}
      if description is not None:
        choice['description'] = _render(description)
      choices.append(choice)
    return {
        'type': 'choice',
        'name': name,
        'instructions': instructions,
        'choices': choices,
    }
  if isinstance(question, Score):
    _check_option_count(name, 'levels', len(question.criteria), _SCORE_LIMITS)
    return {
        'type': 'score',
        'name': name,
        'instructions': instructions,
        'levels': [
            {'label': str(i), 'description': _render(level)}
            for i, level in enumerate(question.criteria)
        ],
    }
  raise TypeError(f'Unsupported question type: {type(question).__name__}.')


def _read_number(name: str, key: str, value: Any, high: float = 1.0) -> float:
  """Reads a number in [0, high], clamping rounding errors at either end."""
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(value)
  ):
    raise ValueError(
        f'OpenAI returned {value!r} for {key!r} in {name!r}, not a number.'
    )
  if not -_TOLERANCE <= value <= high + _TOLERANCE:
    raise ValueError(
        f'OpenAI returned {value} for {key!r} in {name!r}, outside [0, {high}].'
    )
  return min(max(float(value), 0.0), high)


def _read_probabilities(
    name: str, raw: Mapping[str, Any], keys: list[_K]
) -> dict[_K, float]:
  """Reads OpenAI's list of option probabilities into a map over `keys`."""
  items = raw['probabilities']
  if not isinstance(items, list) or not all(
      isinstance(item, Mapping) and 'value' in item and 'probability' in item
      for item in items
  ):
    raise ValueError(
        f'OpenAI returned a malformed {raw["type"]!r} answer to {name!r},'
        ' whose probabilities are not a list of values and probabilities.'
    )
  values = [item['value'] for item in items]
  # Match exact types: OpenAI treats the boolean true and the string "true" as
  # different choices, and Python counts True as the int 1, a valid level.
  key_type = type(keys[0])
  same_type = all(
      isinstance(value, key_type) and not isinstance(value, bool)
      for value in values
  )
  if not same_type or sorted(values) != sorted(keys):
    raise ValueError(
        f'OpenAI returned probabilities for {values} in {name!r}, expected'
        f' {keys}.'
    )
  by_value = {item['value']: item['probability'] for item in items}
  return {key: _read_number(name, str(key), by_value[key]) for key in keys}


def _answers_by_name(raw_answers: list[Any]) -> dict[str, Mapping[str, Any]]:
  """Indexes the answers by the question name that each one echoes."""
  answers: dict[str, Mapping[str, Any]] = {}
  for raw in raw_answers:
    name = raw.get('name') if isinstance(raw, Mapping) else None
    if not isinstance(name, str):
      raise ValueError('OpenAI returned an answer that names no question.')
    if name in answers:
      raise ValueError(f'OpenAI returned more than one answer to {name!r}.')
    answers[name] = raw
  return answers


def _to_answer(
    name: str, question: Question, raw: Mapping[str, Any] | None
) -> Answer:
  """Checks one raw OpenAI answer and builds the ADK answer from it."""
  if raw is None:
    raise ValueError(f'OpenAI returned no answer to {name!r}.')
  if raw.get('type') == 'refusal':
    return RefusalAnswer()
  expected = 'predicate' if isinstance(question, Binary) else question.type
  if raw.get('type') != expected:
    raise ValueError(
        f'OpenAI returned a {raw.get("type")!r} answer to {name!r}, which is'
        f' a {expected!r} question.'
    )
  required = (
      ['probability']
      if expected == 'predicate'
      else [expected, 'probabilities']
  )
  missing = [key for key in required if key not in raw]
  if missing:
    raise ValueError(
        f'OpenAI returned a malformed {expected!r} answer to {name!r},'
        f' missing {missing}.'
    )
  confidence = raw.get('confidence')
  if confidence is not None:
    confidence = _read_number(name, 'confidence', confidence)

  if isinstance(question, Binary):
    return BinaryAnswer(
        probability=_read_number(name, 'probability', raw['probability']),
        confidence=confidence,
    )
  if isinstance(question, Choice):
    choice = raw['choice']
    if not isinstance(choice, str) or choice not in question.criteria:
      raise ValueError(
          f'OpenAI chose {choice!r} for {name!r}, which is not one of its'
          ' labels.'
      )
    return ChoiceAnswer(
        choice=choice,
        probabilities=_read_probabilities(name, raw, list(question.criteria)),
        confidence=confidence,
    )
  top = len(question.criteria) - 1
  return ScoreAnswer(
      score=_read_number(name, 'score', raw['score'], high=top),
      probabilities=_read_probabilities(name, raw, list(range(top + 1))),
      confidence=confidence,
  )


class OpenAIDecisionsClassifier(BaseClassifier):
  """Answers typed questions with the OpenAI Decisions API.

  The Decisions API answers every question about an input in one request and
  returns probabilities instead of generated text, so it is much faster than
  asking a chat model. ``Binary`` questions are sent as OpenAI predicates, and
  every question needs ``instructions``. OpenAI can decline to answer a
  question while it answers the others; that question's answer is a
  ``RefusalAnswer``.

  Usage::

      from google.adk.integrations.openai import OpenAIDecisionsClassifier
      from google.adk.labs.classifiers import Choice

      classifier = OpenAIDecisionsClassifier()  # reads OPENAI_API_KEY
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

  model: str = 'gpt-6-luna'
  """The OpenAI model. ``gpt-6-luna`` is the only model the Decisions API
  supports today."""

  api_key: str | Callable[[], str] | Callable[[], Awaitable[str]] | None = (
      Field(default=None, exclude=True, repr=False)
  )
  """The API key, or a zero-argument callable, sync or async, that returns
  one. The client calls a callable on every request, so it can supply a key
  that expires. Defaults to the ``OPENAI_API_KEY`` environment variable.
  Ignored when ``client`` is set."""

  base_url: str | None = None
  """The API base URL, such as ``https://api.openai.com/v1``. Defaults to the
  ``OPENAI_BASE_URL`` environment variable, then OpenAI's API. Ignored when
  ``client`` is set."""

  client: AsyncOpenAI | None = Field(default=None, exclude=True, repr=False)
  """A configured client to send requests with, for settings such as a
  timeout, retries, an organization, or a proxy.

  When unset, the classifier builds a client from ``api_key`` and
  ``base_url`` on its first call, with a 30-second timeout, and reuses it
  until either of them changes.
  """

  _built_client: AsyncOpenAI | None = PrivateAttr(default=None)
  """The client built when ``client`` is unset."""

  _built_from: tuple[object, object] | None = PrivateAttr(default=None)
  """The ``api_key`` and ``base_url`` that ``_built_client`` was built from."""

  def _get_client(self) -> AsyncOpenAI:
    if self.client is not None:
      return self.client
    settings = (self.api_key, self.base_url)
    if self._built_client is None or self._built_from != settings:
      kwargs: dict[str, Any] = {'timeout': _DEFAULT_TIMEOUT}
      api_key = build_api_key(self.api_key)
      if api_key is not None:
        kwargs['api_key'] = api_key
      if self.base_url is not None:
        kwargs['base_url'] = self.base_url
      self._built_client = AsyncOpenAI(**kwargs)
      self._built_from = settings
    return self._built_client

  @override
  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    """Asks OpenAI every question in one Decisions request.

    Args:
      input: What the questions are about: text, JSON, or a ``types.Content``
        whose parts are all text.
      questions: Maps a caller-chosen name to each question. Every question
        needs ``instructions``.

    Returns:
      One answer per question, keyed by the question's name. A question that
      OpenAI declined to answer gets a ``RefusalAnswer``.

    Raises:
      ValueError: Before the request, if ``questions`` is empty, a question
        has no instructions or more labels or levels than OpenAI accepts, or
        the input has a part that is not text. After the request, if the
        response does not answer the questions that were asked.
      openai.OpenAIError: If no API key is set, or the request still fails
        after the client's retries.
    """
    if not questions:
      raise ValueError('classify() needs at least one question.')
    body = {
        'model': self.model,
        'input': _input_to_text(input),
        'questions': [
            _question_to_wire(name, question)
            for name, question in questions.items()
        ],
    }
    data = await self._get_client().post(_PATH, cast_to=object, body=body)
    if not isinstance(data, Mapping):
      raise ValueError('OpenAI did not return a JSON object.')
    raw_answers = data.get('answers')
    if not isinstance(raw_answers, list):
      raise ValueError('OpenAI returned no answers.')
    answers = _answers_by_name(raw_answers)
    return ClassifierResponse(
        model_version=data.get('model'),
        answers={
            name: _to_answer(name, question, answers.get(name))
            for name, question in questions.items()
        },
    )

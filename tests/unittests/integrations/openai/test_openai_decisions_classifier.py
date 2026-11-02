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

"""Tests for OpenAIDecisionsClassifier."""

from __future__ import annotations

import json
from typing import Any

from google.adk.integrations.openai import _openai_decisions_classifier
from google.adk.integrations.openai import OpenAIDecisionsClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import RefusalAnswer
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
from google.genai import types
import httpx
import openai
from openai import AsyncOpenAI
import pytest

_ROUTE = Choice(
    instructions='Which team should handle this?',
    criteria={'billing': 'Invoices and refunds', 'technical': None},
)
_URGENT = Binary(instructions='Is this urgent?')
_FRUSTRATION = Score(
    instructions='How frustrated is the customer?',
    criteria=['Calm', 'Frustrated', 'Very angry'],
)


class _Server:
  """Records requests and replies with queued responses."""

  def __init__(self, *responses: httpx.Response):
    self.responses = list(responses)
    self.requests: list[httpx.Request] = []

  def __call__(self, request: httpx.Request) -> httpx.Response:
    self.requests.append(request)
    return self.responses.pop(0)

  def body(self, index: int = 0) -> dict[str, Any]:
    return json.loads(self.requests[index].content)


def _ok(*answers: Any, model: str = 'gpt-6-luna-2026-10-01') -> httpx.Response:
  return httpx.Response(
      200,
      json={
          'model': model,
          'answers': list(answers),
          'usage': {'input_tokens': 42, 'output_tokens': 0, 'total_tokens': 42},
      },
  )


def _predicate(name: str, probability: Any) -> dict[str, Any]:
  return {'type': 'predicate', 'name': name, 'probability': probability}


def _choice(name: str, choice: Any, probabilities: Any) -> dict[str, Any]:
  return {
      'type': 'choice',
      'name': name,
      'choice': choice,
      'probabilities': probabilities,
      'confidence': 0.9,
  }


def _score(name: str, score: Any, probabilities: Any) -> dict[str, Any]:
  return {
      'type': 'score',
      'name': name,
      'score': score,
      'probabilities': probabilities,
      'confidence': 0.9,
  }


def _options(values: list[Any], chosen: Any) -> list[dict[str, Any]]:
  return [
      {'value': value, 'probability': 1.0 if value == chosen else 0.0}
      for value in values
  ]


def _every_type_ok() -> httpx.Response:
  """Answers _ROUTE, _URGENT, and _FRUSTRATION, asked as their names."""
  return _ok(
      {
          'type': 'choice',
          'name': 'route',
          'choice': 'billing',
          'probabilities': [
              {'value': 'billing', 'probability': 0.9},
              {'value': 'technical', 'probability': 0.1},
          ],
          'confidence': 0.8,
      },
      _predicate('urgent', 0.95),
      {
          'type': 'score',
          'name': 'frustration',
          'score': 1.05,
          'probabilities': [
              {'value': 0, 'label': '0', 'probability': 0.0},
              {'value': 1, 'label': '1', 'probability': 0.95},
              {'value': 2, 'label': '2', 'probability': 0.05},
          ],
          'confidence': 0.92,
      },
  )


_EVERY_TYPE = {'route': _ROUTE, 'urgent': _URGENT, 'frustration': _FRUSTRATION}


def _client(server: _Server, *, max_retries: int = 0) -> AsyncOpenAI:
  return AsyncOpenAI(
      api_key='test-key',
      base_url='https://api.test/v1',
      max_retries=max_retries,
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
  )


def _classifier(server: _Server) -> OpenAIDecisionsClassifier:
  return OpenAIDecisionsClassifier(client=_client(server))


def _send_built_clients_to(server: _Server, monkeypatch) -> list[AsyncOpenAI]:
  """Routes the clients the classifier builds itself through `server`."""
  real_client = AsyncOpenAI
  clients = []

  def _build(**kwargs: Any) -> AsyncOpenAI:
    client = real_client(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **kwargs,
    )
    clients.append(client)
    return client

  monkeypatch.setattr(_openai_decisions_classifier, 'AsyncOpenAI', _build)
  return clients


async def test_sends_every_question_type_in_one_request():
  """All questions go to /decisions in one request, in OpenAI's format."""
  server = _Server(_every_type_ok())

  await _classifier(server).classify(
      input='I was charged twice!', questions=_EVERY_TYPE
  )

  assert len(server.requests) == 1
  request = server.requests[0]
  assert str(request.url) == 'https://api.test/v1/decisions'
  assert request.headers['Authorization'] == 'Bearer test-key'
  assert server.body() == {
      'model': 'gpt-6-luna',
      'input': 'I was charged twice!',
      'questions': [
          {
              'type': 'choice',
              'name': 'route',
              'instructions': 'Which team should handle this?',
              'choices': [
                  {'value': 'billing', 'description': 'Invoices and refunds'},
                  {'value': 'technical'},
              ],
          },
          {
              'type': 'predicate',
              'name': 'urgent',
              'instructions': 'Is this urgent?',
          },
          {
              'type': 'score',
              'name': 'frustration',
              'instructions': 'How frustrated is the customer?',
              'levels': [
                  {'label': '0', 'description': 'Calm'},
                  {'label': '1', 'description': 'Frustrated'},
                  {'label': '2', 'description': 'Very angry'},
              ],
          },
      ],
  }


async def test_maps_every_answer_type():
  """Each OpenAI answer becomes the ADK answer for its question type."""
  server = _Server(_every_type_ok())

  response = await _classifier(server).classify(
      input='I was charged twice!', questions=_EVERY_TYPE
  )

  assert response.model_version == 'gpt-6-luna-2026-10-01'
  assert response.answers == {
      'route': ChoiceAnswer(
          choice='billing',
          probabilities={'billing': 0.9, 'technical': 0.1},
          confidence=0.8,
      ),
      'urgent': BinaryAnswer(probability=0.95),
      'frustration': ScoreAnswer(
          score=1.05, probabilities={0: 0.0, 1: 0.95, 2: 0.05}, confidence=0.92
      ),
  }


async def test_answers_are_matched_to_questions_by_name():
  """An answer reaches the question it names, whatever its position."""
  server = _Server(_ok(_predicate('second', 0.2), _predicate('first', 0.7)))

  response = await _classifier(server).classify(
      input='hi', questions={'first': _URGENT, 'second': _URGENT}
  )

  assert response.answers == {
      'first': BinaryAnswer(probability=0.7),
      'second': BinaryAnswer(probability=0.2),
  }


async def test_refused_question_gets_a_refusal_and_others_keep_answers():
  """A refused question gets a RefusalAnswer; the rest keep theirs."""
  server = _Server(
      _ok({'type': 'refusal', 'name': 'route'}, _predicate('urgent', 0.6))
  )

  response = await _classifier(server).classify(
      input='hi', questions={'route': _ROUTE, 'urgent': _URGENT}
  )

  assert response.answers == {
      'route': RefusalAnswer(),
      'urgent': BinaryAnswer(probability=0.6),
  }


async def test_binary_criteria_are_appended_to_the_instructions():
  """A predicate has no criteria field, so they join the instructions."""
  server = _Server(_ok(_predicate('risky', 0.1)))

  await _classifier(server).classify(
      input='hi',
      questions={
          'risky': Binary(
              instructions='Is this call risky?', criteria='It deletes data.'
          )
      },
  )

  assert server.body()['questions'][0]['instructions'] == (
      'Is this call risky?\nTrue means: It deletes data.\nAnything else is'
      ' false.'
  )


async def test_json_instructions_and_descriptions_are_sent_as_text():
  """Instructions and descriptions given as JSON are sent as JSON text."""
  server = _Server(_ok(_choice('q', 'a', _options(['a', 'b'], 'a'))))
  question = Choice(
      instructions={'task': 'route', 'channels': ['email']},
      criteria={'a': {'examples': ['refund']}, 'b': ['bug', 'outage']},
  )

  await _classifier(server).classify(input='hi', questions={'q': question})

  sent = server.body()['questions'][0]
  assert sent['instructions'] == '{"task": "route", "channels": ["email"]}'
  assert sent['choices'] == [
      {'value': 'a', 'description': '{"examples": ["refund"]}'},
      {'value': 'b', 'description': '["bug", "outage"]'},
  ]


@pytest.mark.parametrize(
    'input,expected',
    [
        (
            {'ticket': 'Refund the café order.', 'charges': [49, 49]},
            '{"ticket": "Refund the café order.", "charges": [49, 49]}',
        ),
        (
            [{'role': 'user', 'text': 'Hi'}],
            '[{"role": "user", "text": "Hi"}]',
        ),
        (['First message', 2], '["First message", 2]'),
    ],
    ids=['object', 'list-of-objects', 'list'],
)
async def test_json_input_is_sent_as_json_text(input, expected):
  """Dictionaries and lists are sent as JSON text, not input_json parts."""
  server = _Server(_ok(_predicate('urgent', 0.1)))

  await _classifier(server).classify(input=input, questions={'urgent': _URGENT})

  assert server.body()['input'] == expected


async def test_text_content_is_sent_as_a_string():
  """The text parts of a types.Content are joined into one string."""
  server = _Server(_ok(_predicate('urgent', 0.1)))

  await _classifier(server).classify(
      input=types.Content(
          role='user', parts=[types.Part(text='Help, '), types.Part(text='now')]
      ),
      questions={'urgent': _URGENT},
  )

  assert server.body()['input'] == 'Help, now'


async def test_non_text_content_is_rejected():
  """A types.Content with a non-text part fails before any request."""
  server = _Server()
  content = types.Content(
      role='user',
      parts=[types.Part.from_bytes(data=b'png', mime_type='image/png')],
  )

  with pytest.raises(ValueError, match='non-text part'):
    await _classifier(server).classify(
        input=content, questions={'urgent': _URGENT}
    )
  assert not server.requests


async def test_question_without_instructions_is_rejected():
  """A question without instructions fails before any request."""
  server = _Server()

  with pytest.raises(ValueError, match="instructions for question 'urgent'"):
    await _classifier(server).classify(
        input='hi', questions={'urgent': Binary()}
    )
  assert not server.requests


async def test_no_questions_is_rejected():
  """Classifying with no questions fails before any request."""
  server = _Server()

  with pytest.raises(ValueError, match='at least one question'):
    await _classifier(server).classify(input='hi', questions={})
  assert not server.requests


@pytest.mark.parametrize(
    'question,match',
    [
        (
            Score(
                instructions='How bad?', criteria=[str(i) for i in range(11)]
            ),
            "2 to 10 levels in a question; question 'q' has 11",
        ),
        (
            Choice(
                instructions='Which one?',
                criteria={str(i): None for i in range(256)},
            ),
            "2 to 255 labels in a question; question 'q' has 256",
        ),
    ],
    ids=['score', 'choice'],
)
async def test_more_options_than_openai_accepts_are_rejected(question, match):
  """Over 10 score levels or 255 labels fails before any request."""
  server = _Server()

  with pytest.raises(ValueError, match=match):
    await _classifier(server).classify(input='hi', questions={'q': question})
  assert not server.requests


@pytest.mark.parametrize(
    'question,answer,options_key',
    [
        (
            Score(
                instructions='How bad?', criteria=[str(i) for i in range(10)]
            ),
            _score('q', 9.0, _options(list(range(10)), 9)),
            'levels',
        ),
        (
            Choice(
                instructions='Which one?',
                criteria={str(i): None for i in range(255)},
            ),
            _choice('q', '0', _options([str(i) for i in range(255)], '0')),
            'choices',
        ),
    ],
    ids=['score', 'choice'],
)
async def test_the_most_options_openai_accepts_are_sent(
    question, answer, options_key
):
  """Exactly 10 score levels or 255 labels are sent to OpenAI."""
  server = _Server(_ok(answer))

  await _classifier(server).classify(input='hi', questions={'q': question})

  assert len(server.body()['questions'][0][options_key]) == len(
      question.criteria
  )


async def test_builds_a_client_from_api_key_and_base_url(monkeypatch):
  """Without a client, requests use api_key and base_url."""
  server = _Server(_ok(_predicate('urgent', 0.1)))
  _send_built_clients_to(server, monkeypatch)
  classifier = OpenAIDecisionsClassifier(
      api_key='explicit-key', base_url='https://proxy.test/openai/v1'
  )

  await classifier.classify(input='hi', questions={'urgent': _URGENT})

  request = server.requests[0]
  assert str(request.url) == 'https://proxy.test/openai/v1/decisions'
  assert request.headers['Authorization'] == 'Bearer explicit-key'


async def test_built_client_times_out_after_30_seconds(monkeypatch):
  """The client the classifier builds waits 30 seconds per attempt."""
  server = _Server(_ok(_predicate('urgent', 0.1)))
  _send_built_clients_to(server, monkeypatch)

  await OpenAIDecisionsClassifier(api_key='test-key').classify(
      input='hi', questions={'urgent': _URGENT}
  )

  assert server.requests[0].extensions['timeout']['read'] == 30.0


async def test_reads_key_and_base_url_from_the_environment(monkeypatch):
  """OPENAI_API_KEY and OPENAI_BASE_URL apply when the fields are unset."""
  monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
  monkeypatch.setenv('OPENAI_BASE_URL', 'https://env.test/v1')
  server = _Server(_ok(_predicate('urgent', 0.1)))
  _send_built_clients_to(server, monkeypatch)

  await OpenAIDecisionsClassifier().classify(
      input='hi', questions={'urgent': _URGENT}
  )

  request = server.requests[0]
  assert str(request.url) == 'https://env.test/v1/decisions'
  assert request.headers['Authorization'] == 'Bearer env-key'


async def test_callable_api_key_is_called_for_every_request(monkeypatch):
  """A callable api_key is called again for every request."""
  keys = iter(['key-1', 'key-2'])
  server = _Server(
      _ok(_predicate('urgent', 0.1)), _ok(_predicate('urgent', 0.1))
  )
  _send_built_clients_to(server, monkeypatch)
  classifier = OpenAIDecisionsClassifier(api_key=lambda: next(keys))

  for _ in range(2):
    await classifier.classify(input='hi', questions={'urgent': _URGENT})

  assert [r.headers['Authorization'] for r in server.requests] == [
      'Bearer key-1',
      'Bearer key-2',
  ]


async def test_built_client_is_reused_across_calls(monkeypatch):
  """Calls with the same settings share one built client."""
  server = _Server(
      _ok(_predicate('urgent', 0.1)), _ok(_predicate('urgent', 0.1))
  )
  clients = _send_built_clients_to(server, monkeypatch)
  classifier = OpenAIDecisionsClassifier(api_key='test-key')

  for _ in range(2):
    await classifier.classify(input='hi', questions={'urgent': _URGENT})

  assert len(clients) == 1
  assert len(server.requests) == 2


async def test_new_api_key_builds_a_new_client(monkeypatch):
  """A copy with a new api_key sends its calls with the new key."""
  server = _Server(
      _ok(_predicate('urgent', 0.1)), _ok(_predicate('urgent', 0.1))
  )
  _send_built_clients_to(server, monkeypatch)
  classifier = OpenAIDecisionsClassifier(api_key='key-1')
  await classifier.classify(input='hi', questions={'urgent': _URGENT})

  await classifier.model_copy(update={'api_key': 'key-2'}).classify(
      input='hi', questions={'urgent': _URGENT}
  )

  assert server.requests[1].headers['Authorization'] == 'Bearer key-2'


async def test_missing_api_key_raises(monkeypatch):
  """Without an API key, classify raises the openai package's error."""
  monkeypatch.delenv('OPENAI_API_KEY', raising=False)

  with pytest.raises(openai.OpenAIError, match='OPENAI_API_KEY'):
    await OpenAIDecisionsClassifier().classify(
        input='hi', questions={'urgent': _URGENT}
    )


def test_api_key_and_client_are_not_serialized_or_shown():
  """api_key and client stay out of model_dump and repr."""
  classifier = OpenAIDecisionsClassifier(
      api_key='secret', client=AsyncOpenAI(api_key='other-secret')
  )

  assert 'secret' not in repr(classifier)
  assert classifier.model_dump() == {
      'model': 'gpt-6-luna',
      'base_url': None,
  }


async def test_error_status_raises_the_openai_error():
  """An error status from OpenAI raises the openai package's error."""
  server = _Server(
      httpx.Response(400, json={'error': {'message': 'Too many questions.'}})
  )

  with pytest.raises(openai.BadRequestError, match='Too many questions'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )
  assert len(server.requests) == 1


async def test_client_retries_a_rate_limited_request():
  """The client's retries apply, so a 429 is retried."""
  server = _Server(
      httpx.Response(
          429,
          headers={'retry-after-ms': '1'},
          json={'error': {'message': 'Slow down.'}},
      ),
      _ok(_predicate('urgent', 0.7)),
  )
  classifier = OpenAIDecisionsClassifier(client=_client(server, max_retries=1))

  response = await classifier.classify(
      input='hi', questions={'urgent': _URGENT}
  )

  assert len(server.requests) == 2
  assert response.answers['urgent'] == BinaryAnswer(probability=0.7)


async def test_missing_answer_raises():
  """A question with no answer in the response raises."""
  server = _Server(_ok())

  with pytest.raises(ValueError, match="no answer to 'urgent'"):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


async def test_answer_of_the_wrong_type_raises():
  """An answer whose type does not match its question raises."""
  server = _Server(_ok(_choice('urgent', 'a', _options(['a', 'b'], 'a'))))

  with pytest.raises(
      ValueError, match="'choice' answer to 'urgent', which is a 'predicate'"
  ):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


@pytest.mark.parametrize(
    'answers,match',
    [
        ([{'type': 'predicate', 'probability': 0.5}], 'names no question'),
        (['not an object'], 'names no question'),
        (
            [_predicate('urgent', 0.5), _predicate('urgent', 0.6)],
            "more than one answer to 'urgent'",
        ),
    ],
    ids=['unnamed', 'not-an-object', 'duplicate'],
)
async def test_answers_that_cannot_be_matched_raise(answers, match):
  """An unnamed, non-object, or repeated answer raises."""
  server = _Server(_ok(*answers))

  with pytest.raises(ValueError, match=match):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


@pytest.mark.parametrize(
    'question,raw',
    [
        (_URGENT, {'type': 'predicate', 'name': 'q'}),
        (
            _ROUTE,
            {
                'type': 'choice',
                'name': 'q',
                'probabilities': _options(['billing', 'technical'], 'billing'),
            },
        ),
        (_ROUTE, {'type': 'choice', 'name': 'q', 'choice': 'billing'}),
        (_ROUTE, _choice('q', 'billing', {'billing': 1.0, 'technical': 0.0})),
        (_ROUTE, _choice('q', 'billing', [{'value': 'billing'}])),
        (
            _FRUSTRATION,
            {
                'type': 'score',
                'name': 'q',
                'probabilities': _options([0, 1, 2], 0),
            },
        ),
        (_FRUSTRATION, {'type': 'score', 'name': 'q', 'score': 0.0}),
    ],
    ids=[
        'predicate-no-probability',
        'choice-no-choice',
        'choice-no-probabilities',
        'choice-probabilities-not-a-list',
        'choice-option-without-probability',
        'score-no-score',
        'score-no-probabilities',
    ],
)
async def test_malformed_answer_raises_value_error(question, raw):
  """An answer missing a field its type needs raises."""
  server = _Server(_ok(raw))

  with pytest.raises(ValueError, match="malformed .* answer to 'q'"):
    await _classifier(server).classify(input='hi', questions={'q': question})


async def test_non_object_response_raises():
  """A JSON response that is not an object raises."""
  server = _Server(httpx.Response(200, json=['not', 'an', 'object']))

  with pytest.raises(ValueError, match='not return a JSON object'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


async def test_non_json_response_raises():
  """A response body that is not JSON, such as HTML, raises."""
  server = _Server(
      httpx.Response(
          200,
          headers={'content-type': 'text/html'},
          text='<html>Proxy sign-in</html>',
      )
  )

  with pytest.raises(ValueError, match='not return a JSON object'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


async def test_response_without_answers_raises():
  """A response without an answers list raises."""
  server = _Server(httpx.Response(200, json={'model': 'gpt-6-luna'}))

  with pytest.raises(ValueError, match='returned no answers'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


@pytest.mark.parametrize('value', ['NaN', 'Infinity', 'true', '"0.5"', 'null'])
async def test_non_number_probability_raises(value):
  """A probability that is not a finite number raises."""
  server = _Server(
      httpx.Response(
          200,
          headers={'content-type': 'application/json'},
          text=(
              '{"answers": [{"type": "predicate", "name": "q", "probability":'
              f' {value}}}]}}'
          ),
      )
  )

  with pytest.raises(
      ValueError, match="for 'probability' in 'q', not a number"
  ):
    await _classifier(server).classify(input='hi', questions={'q': _URGENT})


@pytest.mark.parametrize(
    'question,raw,match',
    [
        (_URGENT, _predicate('q', 70), "70 for 'probability'"),
        (
            _ROUTE,
            {
                **_choice(
                    'q',
                    'billing',
                    _options(['billing', 'technical'], 'billing'),
                ),
                'confidence': 1.5,
            },
            "1.5 for 'confidence'",
        ),
        (
            _ROUTE,
            _choice(
                'q',
                'billing',
                [
                    {'value': 'billing', 'probability': -0.2},
                    {'value': 'technical', 'probability': 1.0},
                ],
            ),
            "-0.2 for 'billing'",
        ),
        (
            _FRUSTRATION,
            _score('q', 3.0, _options([0, 1, 2], 2)),
            "3.0 for 'score'",
        ),
    ],
    ids=['predicate', 'confidence', 'choice', 'score'],
)
async def test_out_of_range_value_raises(question, raw, match):
  """A value well outside its range raises."""
  server = _Server(_ok(raw))

  with pytest.raises(ValueError, match=match):
    await _classifier(server).classify(input='hi', questions={'q': question})


@pytest.mark.parametrize(
    'question,raw,expected',
    [
        (
            _URGENT,
            _predicate('q', 1.0000001),
            BinaryAnswer(probability=1.0),
        ),
        (
            _ROUTE,
            {
                **_choice(
                    'q',
                    'billing',
                    [
                        {'value': 'billing', 'probability': 1.0000001},
                        {'value': 'technical', 'probability': -1e-9},
                    ],
                ),
                'confidence': 1.0000001,
            },
            ChoiceAnswer(
                choice='billing',
                probabilities={'billing': 1.0, 'technical': 0.0},
                confidence=1.0,
            ),
        ),
        (
            _FRUSTRATION,
            {
                **_score(
                    'q',
                    2.0000001,
                    [
                        {'value': 0, 'probability': -1e-9},
                        {'value': 1, 'probability': 0.0},
                        {'value': 2, 'probability': 1.0000001},
                    ],
                ),
                'confidence': -1e-9,
            },
            ScoreAnswer(
                score=2.0,
                probabilities={0: 0.0, 1: 0.0, 2: 1.0},
                confidence=0.0,
            ),
        ),
    ],
    ids=['binary', 'choice', 'score'],
)
async def test_values_a_rounding_error_out_of_range_are_clamped(
    question, raw, expected
):
  """Values a rounding error outside their range are clamped."""
  server = _Server(_ok(raw))

  response = await _classifier(server).classify(
      input='hi', questions={'q': question}
  )

  assert response.answers['q'] == expected


@pytest.mark.parametrize('choice', ['sales', True])
async def test_choice_outside_the_labels_raises(choice):
  """A choice that is not one of the question's labels raises."""
  server = _Server(
      _ok(_choice('q', choice, _options(['billing', 'technical'], 'billing')))
  )

  with pytest.raises(ValueError, match=f"chose {choice!r} for 'q'"):
    await _classifier(server).classify(input='hi', questions={'q': _ROUTE})


@pytest.mark.parametrize(
    'question,raw',
    [
        (_ROUTE, _choice('q', 'billing', _options(['billing'], 'billing'))),
        (
            _ROUTE,
            _choice(
                'q',
                'billing',
                _options(['billing', 'technical', 'sales'], 'billing'),
            ),
        ),
        (
            _ROUTE,
            _choice(
                'q', 'billing', _options(['billing', 'billing'], 'billing')
            ),
        ),
        (
            Choice(
                instructions='Yes or no?',
                criteria={'true': None, 'false': None},
            ),
            _choice('q', 'true', _options([True, False], True)),
        ),
        (_FRUSTRATION, _score('q', 1.0, _options([0, 1], 1))),
        (_FRUSTRATION, _score('q', 1.0, _options([False, True, 2], True))),
    ],
    ids=[
        'choice-missing-label',
        'choice-extra-label',
        'choice-repeated-label',
        'choice-boolean-values',
        'score-missing-level',
        'score-boolean-levels',
    ],
)
async def test_probabilities_must_match_the_options(question, raw):
  """Probabilities must cover each label or level exactly once."""
  server = _Server(_ok(raw))

  with pytest.raises(ValueError, match="probabilities for .* in 'q', expected"):
    await _classifier(server).classify(input='hi', questions={'q': question})

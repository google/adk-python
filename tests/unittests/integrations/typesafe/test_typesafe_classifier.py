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

"""Tests for TypesafeClassifier."""

from __future__ import annotations

import json
from typing import Any

from google.adk.integrations.typesafe import _typesafe_classifier
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
from google.genai import types
import httpx
import pytest
import tenacity

_ROUTE = Choice(
    instructions='Which team should handle this?',
    criteria={'billing': 'Invoices and refunds', 'technical': None},
)
_URGENT = Binary(
    instructions='Is this urgent?',
    criteria='Time-sensitive',
)
_FRUSTRATION = Score(
    instructions='How frustrated is the customer?',
    criteria=['Calm', 'Frustrated', 'Very angry'],
)


class _Server:
  """Records requests and replies with queued responses or errors."""

  def __init__(self, *responses: httpx.Response | Exception):
    self.responses = list(responses)
    self.requests: list[httpx.Request] = []

  def __call__(self, request: httpx.Request) -> httpx.Response:
    self.requests.append(request)
    response = self.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return response

  def body(self, index: int = 0) -> dict[str, Any]:
    return json.loads(self.requests[index].content)


def _ok(answers: dict[str, Any], model: str = 'jev-1.13.0') -> httpx.Response:
  return httpx.Response(
      200,
      json={
          'model': model,
          'answers': answers,
          'usage': {'input_tokens': 10, 'output_tokens': 2},
      },
  )


def _classifier(server: _Server, **kwargs: Any) -> TypesafeClassifier:
  return TypesafeClassifier(
      api_key='test-key',
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
      **kwargs,
  )


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
  monkeypatch.setattr(_typesafe_classifier, '_RETRY_WAIT', tenacity.wait_none())


async def test_sends_every_question_type_in_one_request():
  server = _Server(
      _ok({
          'route': {
              'type': 'choice',
              'choice': 'billing',
              'probabilities': {'billing': 0.9, 'technical': 0.1},
              'confidence': 0.8,
          },
          'urgent': {'type': 'noul', 'noul': 0.95},
          'frustration': {
              'type': 'score',
              'score': 1.05,
              'legend': {'0': 'Calm', '1': 'Frustrated', '2': 'Very angry'},
              'probabilities': {'0': 0.0, '1': 0.95, '2': 0.05},
              'confidence': 0.92,
          },
      })
  )

  response = await _classifier(server).classify(
      input={'message': 'I was charged twice!'},
      questions={
          'route': _ROUTE,
          'urgent': _URGENT,
          'frustration': _FRUSTRATION,
      },
  )

  assert len(server.requests) == 1
  request = server.requests[0]
  assert str(request.url) == 'https://api.typesafe.ai/v1/systemone'
  assert request.headers['Authorization'] == 'Bearer test-key'
  assert server.body() == {
      'model': 'jev-latest',
      'state': {'message': 'I was charged twice!'},
      'questions': {
          'route': {
              'type': 'choice',
              'instructions': 'Which team should handle this?',
              'criteria': {
                  'billing': 'Invoices and refunds',
                  'technical': None,
              },
          },
          'urgent': {
              'type': 'noul',
              'instructions': 'Is this urgent?',
              'criteria': {'true': 'Time-sensitive'},
          },
          'frustration': {
              'type': 'score',
              'instructions': 'How frustrated is the customer?',
              'criteria': ['Calm', 'Frustrated', 'Very angry'],
          },
      },
  }
  assert response.model_version == 'jev-1.13.0'
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


async def test_text_content_is_sent_as_a_string():
  server = _Server(_ok({'urgent': {'type': 'noul', 'noul': 0.1}}))

  await _classifier(server).classify(
      input=types.Content(
          role='user', parts=[types.Part(text='Help, '), types.Part(text='now')]
      ),
      questions={'urgent': _URGENT},
  )

  assert server.body()['state'] == 'Help, now'


async def test_non_text_content_is_rejected():
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
  server = _Server()

  with pytest.raises(ValueError, match="instructions for question 'urgent'"):
    await _classifier(server).classify(
        input='hi', questions={'urgent': Binary()}
    )
  assert not server.requests


async def test_reads_key_and_base_url_from_the_environment(monkeypatch):
  monkeypatch.setenv('TYPESAFE_API_KEY', 'env-key')
  monkeypatch.setenv('TYPESAFE_BASE_URL', 'http://localhost:8088/')
  server = _Server(_ok({'urgent': {'type': 'noul', 'noul': 0.1}}))
  classifier = TypesafeClassifier(
      model='jev-1.13',
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
  )

  await classifier.classify(input='hi', questions={'urgent': _URGENT})

  request = server.requests[0]
  assert str(request.url) == 'http://localhost:8088/v1/systemone'
  assert request.headers['Authorization'] == 'Bearer env-key'
  assert server.body()['model'] == 'jev-1.13'


async def test_uses_base_url_and_timeout_of_the_supplied_client(monkeypatch):
  monkeypatch.setenv('TYPESAFE_BASE_URL', 'http://env-host')
  server = _Server(_ok({'urgent': {'type': 'noul', 'noul': 0.1}}))
  classifier = _classifier(server)
  classifier.http_client = httpx.AsyncClient(
      base_url='http://proxy/typesafe',
      timeout=7.0,
      transport=httpx.MockTransport(server),
  )

  await classifier.classify(input='hi', questions={'urgent': _URGENT})

  request = server.requests[0]
  assert str(request.url) == 'http://proxy/typesafe/v1/systemone'
  assert request.extensions['timeout']['read'] == 7.0


async def test_fields_override_the_supplied_client():
  server = _Server(_ok({'urgent': {'type': 'noul', 'noul': 0.1}}))
  classifier = TypesafeClassifier(
      api_key='test-key',
      base_url='http://explicit',
      timeout=3.0,
      http_client=httpx.AsyncClient(
          base_url='http://proxy',
          timeout=7.0,
          transport=httpx.MockTransport(server),
      ),
  )

  await classifier.classify(input='hi', questions={'urgent': _URGENT})

  request = server.requests[0]
  assert str(request.url) == 'http://explicit/v1/systemone'
  assert request.extensions['timeout']['read'] == 3.0


async def test_default_client_waits_30_seconds(monkeypatch):
  server = _Server(_ok({'urgent': {'type': 'noul', 'noul': 0.1}}))
  real_client = httpx.AsyncClient
  monkeypatch.setattr(
      _typesafe_classifier.httpx,
      'AsyncClient',
      lambda **kwargs: real_client(
          transport=httpx.MockTransport(server), **kwargs
      ),
  )

  await TypesafeClassifier(api_key='test-key').classify(
      input='hi', questions={'urgent': _URGENT}
  )

  request = server.requests[0]
  assert str(request.url) == 'https://api.typesafe.ai/v1/systemone'
  assert request.extensions['timeout']['read'] == 30.0


async def test_missing_api_key_raises(monkeypatch):
  monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
  classifier = TypesafeClassifier(
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(_Server()))
  )

  with pytest.raises(ValueError, match='TYPESAFE_API_KEY'):
    await classifier.classify(input='hi', questions={'urgent': _URGENT})


async def test_default_client_is_closed_after_each_call(monkeypatch):
  urgent = {'urgent': {'type': 'noul', 'noul': 0.5}}
  server = _Server(_ok(urgent), _ok(urgent))
  real_client = httpx.AsyncClient
  clients = []

  def _client(**kwargs):
    client = real_client(transport=httpx.MockTransport(server), **kwargs)
    clients.append(client)
    return client

  monkeypatch.setattr(_typesafe_classifier.httpx, 'AsyncClient', _client)
  classifier = TypesafeClassifier(api_key='test-key')

  for _ in range(2):
    await classifier.classify(input='hi', questions={'urgent': _URGENT})

  assert len(clients) == 2
  assert all(client.is_closed for client in clients)


def test_api_key_is_not_serialized_or_shown():
  classifier = TypesafeClassifier(api_key='secret')

  assert 'secret' not in repr(classifier)
  assert 'api_key' not in classifier.model_dump()


@pytest.mark.parametrize('status', [429, 529])
async def test_retries_rate_limited_and_overloaded_responses(status):
  server = _Server(
      httpx.Response(status, json={'detail': 'slow down'}),
      _ok({'urgent': {'type': 'noul', 'noul': 0.7}}),
  )

  response = await _classifier(server).classify(
      input='hi', questions={'urgent': _URGENT}
  )

  assert len(server.requests) == 2
  assert response.answers['urgent'] == BinaryAnswer(probability=0.7)


async def test_gives_up_after_max_attempts():
  server = _Server(*[httpx.Response(529, text='overloaded')] * 2)

  with pytest.raises(RuntimeError, match='529: overloaded'):
    await _classifier(server, max_attempts=2).classify(
        input='hi', questions={'urgent': _URGENT}
    )
  assert len(server.requests) == 2


@pytest.mark.parametrize(
    'error',
    [
        httpx.ConnectError('refused'),
        httpx.ReadTimeout('slow'),
        httpx.RemoteProtocolError('disconnected'),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_retries_transient_transport_errors(error):
  server = _Server(error, _ok({'urgent': {'type': 'noul', 'noul': 0.7}}))

  response = await _classifier(server).classify(
      input='hi', questions={'urgent': _URGENT}
  )

  assert len(server.requests) == 2
  assert response.answers['urgent'] == BinaryAnswer(probability=0.7)


@pytest.mark.parametrize(
    'error',
    [
        httpx.UnsupportedProtocol('ftp'),
        httpx.ProxyError('bad proxy'),
        httpx.LocalProtocolError('bad header'),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_permanent_transport_errors_are_not_retried(error):
  server = _Server(error)

  with pytest.raises(type(error)):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )
  assert len(server.requests) == 1


async def test_client_errors_are_not_retried():
  server = _Server(httpx.Response(422, text='bad question'))

  with pytest.raises(RuntimeError, match='422: bad question'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )
  assert len(server.requests) == 1


async def test_missing_answer_raises():
  server = _Server(_ok({}))

  with pytest.raises(ValueError, match="no answer to 'urgent'"):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


async def test_answer_of_the_wrong_type_raises():
  server = _Server(_ok({'urgent': {'type': 'choice', 'choice': 'x'}}))

  with pytest.raises(ValueError, match="'choice' answer to 'urgent'"):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


@pytest.mark.parametrize(
    'question,raw,expected',
    [
        (
            _URGENT,
            {'type': 'noul', 'noul': 1.0000001, 'confidence': -1e-9},
            BinaryAnswer(probability=1.0, confidence=0.0),
        ),
        (
            _ROUTE,
            {
                'type': 'choice',
                'choice': 'billing',
                'probabilities': {'billing': 1.0000001, 'technical': -1e-9},
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
                'type': 'score',
                'score': 0.0,
                'probabilities': {'0': 1.0000001, '1': -1e-9, '2': 0.0},
            },
            ScoreAnswer(score=0.0, probabilities={0: 1.0, 1: 0.0, 2: 0.0}),
        ),
    ],
    ids=['binary', 'choice', 'score'],
)
async def test_probabilities_are_clamped(question, raw, expected):
  server = _Server(_ok({'q': raw}))

  response = await _classifier(server).classify(
      input='hi', questions={'q': question}
  )

  assert response.answers['q'] == expected


@pytest.mark.parametrize(
    'question,raw',
    [
        (_URGENT, {'type': 'noul'}),
        (_ROUTE, {'type': 'choice', 'probabilities': {'billing': 1.0}}),
        (_ROUTE, {'type': 'choice', 'choice': 'billing'}),
        (_ROUTE, {'type': 'choice', 'choice': 'billing', 'probabilities': 1}),
        (_FRUSTRATION, {'type': 'score', 'probabilities': {'0': 1.0}}),
        (_FRUSTRATION, {'type': 'score', 'score': 0.0}),
    ],
    ids=[
        'binary-no-noul',
        'choice-no-choice',
        'choice-no-probabilities',
        'choice-bad-probabilities',
        'score-no-score',
        'score-no-probabilities',
    ],
)
async def test_malformed_answer_raises_value_error(question, raw):
  server = _Server(_ok({'q': raw}))

  with pytest.raises(ValueError, match="malformed .* answer to 'q'"):
    await _classifier(server).classify(input='hi', questions={'q': question})


async def test_non_object_response_raises():
  server = _Server(httpx.Response(200, json=['not', 'an', 'object']))

  with pytest.raises(ValueError, match='not return a JSON object'):
    await _classifier(server).classify(
        input='hi', questions={'urgent': _URGENT}
    )


@pytest.mark.parametrize('value', ['NaN', 'Infinity', 'true', '"0.5"', 'null'])
async def test_non_number_probability_raises(value):
  server = _Server(
      httpx.Response(
          200,
          text=f'{{"answers": {{"q": {{"type": "noul", "noul": {value}}}}}}}',
      )
  )

  with pytest.raises(ValueError, match="for 'noul' in 'q', not a number"):
    await _classifier(server).classify(input='hi', questions={'q': _URGENT})


@pytest.mark.parametrize(
    'question,raw,match',
    [
        (_URGENT, {'type': 'noul', 'noul': 70}, "70 for 'noul'"),
        (
            _URGENT,
            {'type': 'noul', 'noul': 0.5, 'confidence': 1.5},
            "1.5 for 'confidence'",
        ),
        (
            _ROUTE,
            {
                'type': 'choice',
                'choice': 'billing',
                'probabilities': {'billing': -0.2, 'technical': 1.0},
            },
            "-0.2 for 'billing'",
        ),
        (
            _FRUSTRATION,
            {
                'type': 'score',
                'score': 3.0,
                'probabilities': {'0': 0.0, '1': 0.0, '2': 1.0},
            },
            "3.0 for 'score'",
        ),
    ],
    ids=['binary', 'confidence', 'choice', 'score'],
)
async def test_out_of_range_value_raises(question, raw, match):
  server = _Server(_ok({'q': raw}))

  with pytest.raises(ValueError, match=match):
    await _classifier(server).classify(input='hi', questions={'q': question})


async def test_choice_outside_the_labels_raises():
  server = _Server(
      _ok({
          'q': {
              'type': 'choice',
              'choice': 'sales',
              'probabilities': {'billing': 0.5, 'technical': 0.5},
          }
      })
  )

  with pytest.raises(ValueError, match="chose 'sales' for 'q'"):
    await _classifier(server).classify(input='hi', questions={'q': _ROUTE})


@pytest.mark.parametrize(
    'question,raw',
    [
        (
            _ROUTE,
            {
                'type': 'choice',
                'choice': 'billing',
                'probabilities': {'billing': 1.0},
            },
        ),
        (
            _ROUTE,
            {
                'type': 'choice',
                'choice': 'billing',
                'probabilities': {
                    'billing': 0.8,
                    'technical': 0.1,
                    'sales': 0.1,
                },
            },
        ),
        (
            _FRUSTRATION,
            {
                'type': 'score',
                'score': 1.0,
                'probabilities': {'0': 0.0, '1': 1.0},
            },
        ),
    ],
    ids=['choice-missing-label', 'choice-extra-label', 'score-missing-level'],
)
async def test_probabilities_must_match_the_options(question, raw):
  server = _Server(_ok({'q': raw}))

  with pytest.raises(ValueError, match="probabilities for .* in 'q', expected"):
    await _classifier(server).classify(input='hi', questions={'q': question})

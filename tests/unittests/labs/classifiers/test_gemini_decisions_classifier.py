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


"""Tests for GeminiDecisionsClassifier."""

from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any

from google.adk.labs.classifiers import _gemini_decisions_classifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import GeminiDecisionsClassifier
from google.adk.labs.classifiers import RefusalAnswer
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
from google.adk.utils._client_labels_utils import classifier_client_label
from google.genai import types
import httpx
import pytest

_ROUTE = Choice(
    instructions='Which team should handle this?',
    criteria={'billing': 'Invoices and refunds', 'technical': None},
)
_URGENT = Binary(instructions='Is this urgent?', criteria='Time-sensitive')
_FRUSTRATION = Score(
    instructions='How frustrated is the customer?',
    criteria=['Calm', 'Frustrated', 'Very angry'],
)
_CHOICE_ANSWER = {
    'type': 'choice',
    'choice': 'billing',
    'probabilities': {'billing': 0.9, 'technical': 0.1},
    'confidence': 0.9,
}
_NOUL_ANSWER = {'type': 'noul', 'noul': 0.95, 'confidence': 0.95}


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


def _ok(body: dict[str, Any]) -> httpx.Response:
  return httpx.Response(200, json=body)


def _bad_request(code: str) -> httpx.Response:
  return httpx.Response(
      400, json={'error': {'code': code, 'message': 'Request blocked.'}}
  )


def _classifier(server: _Server, **kwargs: Any) -> GeminiDecisionsClassifier:
  return GeminiDecisionsClassifier(
      api_key='test-key',
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
      **kwargs,
  )


async def test_sends_every_question_type_in_one_request():
  server = _Server(
      _ok({
          'answers': {
              'route': _CHOICE_ANSWER,
              'urgent': _NOUL_ANSWER,
              'frustration': {
                  'type': 'score',
                  'score': 1.05,
                  'legend': {'0': 'Calm', '1': 'Frustrated', '2': 'Very angry'},
                  'probabilities': {'0': 0.0, '1': 0.95, '2': 0.05},
                  'confidence': 0.95,
              },
          },
          'usage': {'total_input_tokens': 100, 'total_output_tokens': 0},
      })
  )

  response = await _classifier(server).classify(
      input='I was charged twice!',
      questions={
          'route': _ROUTE,
          'urgent': _URGENT,
          'frustration': _FRUSTRATION,
      },
  )

  assert len(server.requests) == 1
  request = server.requests[0]
  assert str(request.url) == (
      _gemini_decisions_classifier._DEFAULT_BASE_URL  # pylint: disable=protected-access
      + '/v1beta/decisions'
  )
  assert request.headers['x-goog-api-key'] == 'test-key'
  assert classifier_client_label('gemini') in request.headers[
      'x-goog-api-client'
  ].split(' ')
  body = server.body()
  assert body['model'] == 'gemini-decision-1'
  assert body['input'] == 'I was charged twice!'
  assert body['questions'] == {
      'route': {
          'type': 'choice',
          'instructions': 'Which team should handle this?',
          'criteria': {'billing': 'Invoices and refunds', 'technical': ''},
      },
      'urgent': {
          'type': 'noul',
          'instructions': 'Is this urgent?',
          'criteria': {'true_description': 'Time-sensitive'},
      },
      'frustration': {
          'type': 'score',
          'instructions': 'How frustrated is the customer?',
          'criteria': ['Calm', 'Frustrated', 'Very angry'],
      },
  }
  assert response.answers == {
      'route': ChoiceAnswer(
          choice='billing',
          probabilities={'billing': 0.9, 'technical': 0.1},
          confidence=0.9,
      ),
      'urgent': BinaryAnswer(probability=0.95, confidence=0.95),
      'frustration': ScoreAnswer(
          score=1.05,
          probabilities={0: 0.0, 1: 0.95, 2: 0.05},
          confidence=0.95,
      ),
  }


async def test_sends_json_input_and_criteria_as_json_text():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))

  await _classifier(server).classify(
      input={'message': 'Héllo', 'priority': 2},
      questions={
          'urgent': Binary(
              instructions={'ask': 'Is this urgent?'}, criteria=['p0', 'p1']
          )
      },
  )

  body = server.body()
  assert body['input'] == '{"message": "Héllo", "priority": 2}'
  assert body['questions']['urgent'] == {
      'type': 'noul',
      'instructions': '{"ask": "Is this urgent?"}',
      'criteria': {'true_description': '["p0", "p1"]'},
  }


async def test_sends_multimodal_content_as_typed_parts():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))

  await _classifier(server).classify(
      input=types.Content(
          role='user',
          parts=[
              types.Part.from_text(text='What is in this photo?'),
              types.Part.from_bytes(data=b'png-bytes', mime_type='image/png'),
              types.Part.from_uri(
                  file_uri='https://example.com/clip.mp4',
                  mime_type='video/mp4',
              ),
              types.Part.from_bytes(data=b'%PDF', mime_type='application/pdf'),
          ],
      ),
      questions={'urgent': Binary(instructions='Is this urgent?')},
  )

  assert server.body()['input'] == [
      {'type': 'text', 'text': 'What is in this photo?'},
      {'type': 'image', 'data': 'cG5nLWJ5dGVz', 'mime_type': 'image/png'},
      {
          'type': 'video',
          'uri': 'https://example.com/clip.mp4',
          'mime_type': 'video/mp4',
      },
      {'type': 'document', 'data': 'JVBERg==', 'mime_type': 'application/pdf'},
  ]


async def test_skips_thought_parts():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))

  await _classifier(server).classify(
      input=types.Content(
          role='model',
          parts=[
              types.Part(text='Let me think.', thought=True),
              types.Part.from_text(text='Answer'),
          ],
      ),
      questions={'urgent': _URGENT},
  )

  assert server.body()['input'] == [{'type': 'text', 'text': 'Answer'}]


async def test_lowercases_the_mime_type():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))

  await _classifier(server).classify(
      input=types.Content(
          role='user',
          parts=[types.Part.from_bytes(data=b'png', mime_type='IMAGE/PNG')],
      ),
      questions={'urgent': _URGENT},
  )

  assert server.body()['input'] == [
      {'type': 'image', 'mime_type': 'image/png', 'data': 'cG5n'}
  ]


async def test_rejects_a_media_part_without_a_mime_type():
  server = _Server()
  content = types.Content(
      role='user',
      parts=[types.Part(inline_data=types.Blob(data=b'x'))],
  )

  with pytest.raises(ValueError, match='MIME type'):
    await _classifier(server).classify(
        input=content, questions={'urgent': _URGENT}
    )

  assert not server.requests


@pytest.mark.parametrize(
    'reply',
    [
        _ok({'prompt_feedback': {'block_reason': 'safety'}}),
        _bad_request('safety'),
        _bad_request('jailbreak'),
        _bad_request('model_armor'),
    ],
)
async def test_blocked_input_refuses_every_question(reply):
  server = _Server(reply)

  response = await _classifier(server).classify(
      input='bad input', questions={'route': _ROUTE, 'urgent': _URGENT}
  )

  assert response.answers == {
      'route': RefusalAnswer(),
      'urgent': RefusalAnswer(),
  }


@pytest.mark.parametrize(
    'reply',
    [_bad_request('invalid_argument'), httpx.Response(500, text='boom')],
)
async def test_raises_other_errors(reply):
  server = _Server(reply)

  with pytest.raises(RuntimeError, match=str(reply.status_code)):
    await _classifier(server).classify(input='x', questions={'urgent': _URGENT})


async def test_reads_api_key_from_the_environment(monkeypatch):
  monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
  monkeypatch.setenv('GEMINI_API_KEY', 'env-key')
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))
  classifier = GeminiDecisionsClassifier(
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
  )

  await classifier.classify(input='x', questions={'urgent': _URGENT})

  assert server.requests[0].headers['x-goog-api-key'] == 'env-key'


async def test_requires_an_api_key(monkeypatch):
  monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
  monkeypatch.delenv('GEMINI_API_KEY', raising=False)

  def fail(*args, **kwargs):
    raise AssertionError('built a client without an API key')

  monkeypatch.setattr(
      _gemini_decisions_classifier, 'client_cert_httpx_kwargs', fail
  )

  with pytest.raises(ValueError, match='GOOGLE_API_KEY'):
    await GeminiDecisionsClassifier().classify(
        input='x', questions={'urgent': _URGENT}
    )


async def test_uses_the_base_url_of_the_http_client():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))
  classifier = GeminiDecisionsClassifier(
      api_key='k',
      http_client=httpx.AsyncClient(
          base_url='https://proxy.example/gemini',
          transport=httpx.MockTransport(server),
      ),
  )

  await classifier.classify(input='x', questions={'urgent': _URGENT})

  assert (
      str(server.requests[0].url)
      == 'https://proxy.example/gemini/v1beta/decisions'
  )


async def test_base_url_overrides_the_http_client():
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))
  classifier = GeminiDecisionsClassifier(
      api_key='k',
      base_url='https://other.example/',
      http_client=httpx.AsyncClient(
          base_url='https://proxy.example/gemini',
          transport=httpx.MockTransport(server),
      ),
  )

  await classifier.classify(input='x', questions={'urgent': _URGENT})

  assert str(server.requests[0].url) == 'https://other.example/v1beta/decisions'


@pytest.mark.parametrize(
    ('timeout', 'expected'),
    [(None, 12.0), (3.0, 3.0)],
)
async def test_timeout_defaults_to_the_http_client(timeout, expected):
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))
  classifier = _classifier(server, timeout=timeout)
  classifier.http_client.timeout = httpx.Timeout(12.0)

  await classifier.classify(input='x', questions={'urgent': _URGENT})

  assert server.requests[0].extensions['timeout']['read'] == expected


async def test_default_client_presents_the_client_cert(monkeypatch):
  server = _Server(_ok({'answers': {'urgent': _NOUL_ANSWER}}))
  seen: dict[str, Any] = {}

  def fake_kwargs(verify=True, *, url=None):
    seen['url'] = url
    return {'transport': httpx.MockTransport(server)}

  monkeypatch.setattr(
      _gemini_decisions_classifier, 'client_cert_httpx_kwargs', fake_kwargs
  )

  await GeminiDecisionsClassifier(api_key='k').classify(
      input='x', questions={'urgent': _URGENT}
  )

  assert seen['url'] == _gemini_decisions_classifier._DEFAULT_BASE_URL  # pylint: disable=protected-access
  assert len(server.requests) == 1


def _count_clients(monkeypatch) -> list[httpx.AsyncClient]:
  clients = []

  class CountingClient(httpx.AsyncClient):

    def __init__(self, **kwargs):
      super().__init__(**kwargs)
      clients.append(self)

  monkeypatch.setattr(
      _gemini_decisions_classifier.httpx, 'AsyncClient', CountingClient
  )
  return clients


def test_default_client_is_reused_per_event_loop(monkeypatch):
  server = _Server(
      *[_ok({'answers': {'urgent': _NOUL_ANSWER}}) for _ in range(3)]
  )
  transport = httpx.MockTransport(server)
  monkeypatch.setattr(
      _gemini_decisions_classifier,
      'client_cert_httpx_kwargs',
      lambda verify=True, *, url=None: {'transport': transport},
  )
  clients = _count_clients(monkeypatch)
  classifier = GeminiDecisionsClassifier(api_key='k')

  async def classify_twice():
    for _ in range(2):
      await classifier.classify(input='x', questions={'urgent': _URGENT})

  async def classify_once():
    await classifier.classify(input='x', questions={'urgent': _URGENT})

  asyncio.run(classify_twice())
  asyncio.run(classify_once())

  assert len(server.requests) == 3
  assert len(clients) == 2


async def test_default_client_is_rebuilt_when_the_cert_reloads(monkeypatch):
  server = _Server(
      *[_ok({'answers': {'urgent': _NOUL_ANSWER}}) for _ in range(3)]
  )
  transport = httpx.MockTransport(server)
  first = ssl.create_default_context()
  second = ssl.create_default_context()
  contexts = iter([first, first, second])
  monkeypatch.setattr(
      _gemini_decisions_classifier,
      'client_cert_httpx_kwargs',
      lambda verify=True, *, url=None: {
          'transport': transport,
          'verify': next(contexts),
      },
  )
  clients = _count_clients(monkeypatch)
  classifier = GeminiDecisionsClassifier(api_key='k')

  for _ in range(3):
    await classifier.classify(input='x', questions={'urgent': _URGENT})

  assert len(server.requests) == 3
  assert len(clients) == 2


async def test_rejects_empty_questions():
  server = _Server()

  with pytest.raises(ValueError, match='at least one question'):
    await _classifier(server).classify(input='x', questions={})

  assert not server.requests


async def test_rejects_a_question_without_instructions():
  server = _Server()

  with pytest.raises(ValueError, match='needs instructions'):
    await _classifier(server).classify(
        input='x', questions={'urgent': Binary()}
    )

  assert not server.requests


async def test_rejects_unsupported_parts_before_the_request():
  server = _Server()
  content = types.Content(
      role='user',
      parts=[types.Part.from_function_call(name='f', args={})],
  )

  with pytest.raises(ValueError, match='another kind of part'):
    await _classifier(server).classify(
        input=content, questions={'urgent': _URGENT}
    )

  assert not server.requests


_SCORE_ANSWER = {
    'type': 'score',
    'score': 1.0,
    'probabilities': {'0': 0.0, '1': 1.0, '2': 0.0},
    'confidence': 1.0,
}


@pytest.mark.parametrize(
    ('answers', 'message'),
    [
        ({}, "NoneType for 'route'"),
        ({'route': _NOUL_ANSWER}, "_NoulAnswer for 'route'"),
        ({'route': {**_CHOICE_ANSWER, 'choice': 'sales'}}, "chose 'sales'"),
        (
            {
                'route': {
                    **_CHOICE_ANSWER,
                    'probabilities': {'billing': 0.9, 'sales': 0.1},
                }
            },
            'not for its options',
        ),
        (
            {'route': {**_CHOICE_ANSWER, 'confidence': 1.1}},
            'confidence 1.1',
        ),
        ({'frustration': {**_SCORE_ANSWER, 'score': 2.5}}, 'score 2.5'),
        (
            {
                'frustration': {
                    **_SCORE_ANSWER,
                    'probabilities': {'0': 0.5, '1': 0.5},
                }
            },
            'not for its options',
        ),
    ],
)
async def test_rejects_answers_that_do_not_match_the_question(answers, message):
  server = _Server(_ok({'answers': answers}))
  name = next(iter(answers), 'route')
  question = _FRUSTRATION if name == 'frustration' else _ROUTE

  with pytest.raises(ValueError, match=message):
    await _classifier(server).classify(input='x', questions={name: question})


async def test_clamps_rounding_errors():
  server = _Server(
      _ok({
          'answers': {
              'urgent': {'type': 'noul', 'noul': 1.0000004, 'confidence': 1.0},
              'frustration': {
                  **_SCORE_ANSWER,
                  'score': -0.0000002,
                  'probabilities': {'0': 1.0000003, '1': -0.0000001, '2': 0.0},
              },
          }
      })
  )

  response = await _classifier(server).classify(
      input='x', questions={'urgent': _URGENT, 'frustration': _FRUSTRATION}
  )

  assert response.answers == {
      'urgent': BinaryAnswer(probability=1.0, confidence=1.0),
      'frustration': ScoreAnswer(
          score=0.0, probabilities={0: 1.0, 1: 0.0, 2: 0.0}, confidence=1.0
      ),
  }

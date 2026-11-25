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

import base64
import json
from types import SimpleNamespace
from typing import Any

from google.adk import version
from google.adk.integrations.gemini_decisions import _gemini_decisions_classifier
from google.adk.integrations.gemini_decisions import GeminiDecisionsClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import ClassifierNode
from google.adk.labs.classifiers import RefusalAnswer
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
from google.adk.utils._client_labels_utils import classifier_client_label
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._errors import WorkflowDataError
from google.genai import types
import httpx
import pytest
import tenacity

from ...workflow.workflow_testing_utils import InputCapturingNode
from ...workflow.workflow_testing_utils import run_workflow

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
# With a client certificate, _mtls_utils sends requests to the
# generativelanguage.mtls.googleapis.com host instead.
_URL = 'https://generativelanguage.googleapis.com/v1beta/decisions'


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


def _ok(answers: dict[str, Any], **extra: Any) -> httpx.Response:
  return httpx.Response(
      200,
      json={
          'answers': answers,
          'usage': {
              'total_input_tokens': 10,
              'total_output_tokens': 0,
              'total_tokens': 10,
          },
          **extra,
      },
  )


def _blocked(code: str = 'safety') -> httpx.Response:
  return httpx.Response(
      400, json={'error': {'code': code, 'message': 'Request blocked.'}}
  )


def _classifier(server: _Server, **kwargs: Any) -> GeminiDecisionsClassifier:
  return GeminiDecisionsClassifier(
      api_key='test-key',
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
      **kwargs,
  )


async def _classify_one(
    question: Any, raw: dict[str, Any], input: Any = 'text'
) -> Any:
  server = _Server(_ok({'q': raw}))
  response = await _classifier(server).classify(
      input=input, questions={'q': question}
  )
  return response.answers['q']


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
  monkeypatch.setattr(
      _gemini_decisions_classifier, '_RETRY_WAIT', tenacity.wait_none()
  )


@pytest.fixture(autouse=True)
def _no_api_key_in_environment(monkeypatch):
  monkeypatch.delenv('GEMINI_API_KEY', raising=False)
  monkeypatch.delenv('GOOGLE_API_KEY', raising=False)


# Request building.


async def test_sends_every_question_type_in_one_request():
  server = _Server(
      _ok({
          'route': {
              'type': 'choice',
              'choice': 'billing',
              'probabilities': {'billing': 0.9, 'technical': 0.1},
              'confidence': 0.9,
          },
          'urgent': {'type': 'noul', 'noul': 0.95, 'confidence': 0.95},
          'frustration': {
              'type': 'score',
              'score': 1.05,
              'legend': {'0': 'Calm', '1': 'Frustrated', '2': 'Very angry'},
              'probabilities': {'0': 0.0, '1': 0.95, '2': 0.05},
              'confidence': 0.95,
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
  assert request.method == 'POST'
  assert str(request.url) == _URL
  assert request.headers['x-goog-api-key'] == 'test-key'
  assert 'authorization' not in request.headers
  assert server.body() == {
      'model': 'gemini-decision-1',
      'input': '{"message": "I was charged twice!"}',
      'questions': {
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
  assert response.model_version is None


async def test_binary_without_criteria_omits_them():
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))

  await _classifier(server).classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert server.body()['questions']['q'] == {
      'type': 'noul',
      'instructions': 'Is it?',
  }


async def test_json_instructions_and_criteria_are_sent_as_text():
  server = _Server(
      _ok({
          'c': {
              'type': 'choice',
              'choice': 'a',
              'probabilities': {'a': 0.5, 'b': 0.5},
          },
          'b': {'type': 'noul', 'noul': 0.5},
          's': {
              'type': 'score',
              'score': 0.5,
              'probabilities': {'0': 0.5, '1': 0.5},
          },
      })
  )

  await _classifier(server).classify(
      input=['a', 1],
      questions={
          'c': Choice(
              instructions={'ask': 'which'},
              criteria={'a': {'when': 'x'}, 'b': None},
          ),
          'b': Binary(instructions={'ask': 'is it'}, criteria={'true': 'if x'}),
          's': Score(instructions=['rate'], criteria=[{'level': 0}, 'high']),
      },
  )

  body = server.body()
  assert body['input'] == '["a", 1]'
  assert body['questions']['c']['instructions'] == '{"ask": "which"}'
  assert body['questions']['c']['criteria'] == {'a': '{"when": "x"}', 'b': ''}
  assert body['questions']['b']['criteria'] == {
      'true_description': '{"true": "if x"}'
  }
  assert body['questions']['s']['instructions'] == '["rate"]'
  assert body['questions']['s']['criteria'] == ['{"level": 0}', 'high']


async def test_text_content_is_sent_as_a_joined_string():
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  content = types.Content(
      role='user',
      parts=[types.Part(text='Hello, '), types.Part(text='world')],
  )

  await _classifier(server).classify(
      input=content, questions={'q': Binary(instructions='Is it?')}
  )

  assert server.body()['input'] == 'Hello, world'


@pytest.mark.parametrize(
    ('mime_type', 'block_type'),
    [
        ('application/pdf', 'document'),
        ('image/png', 'image'),
        ('image/jpeg; charset=binary', 'image'),
        ('audio/mpeg', 'audio'),
        ('video/mp4', 'video'),
    ],
)
async def test_inline_media_is_sent_as_content_blocks(mime_type, block_type):
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='Invoice for review.'),
          types.Part(
              inline_data=types.Blob(data=b'\x00\x01bytes', mime_type=mime_type)
          ),
      ],
  )

  await _classifier(server).classify(
      input=content, questions={'q': Binary(instructions='Is it?')}
  )

  assert server.body()['input'] == [
      {'type': 'text', 'text': 'Invoice for review.'},
      {
          'type': block_type,
          'data': base64.b64encode(b'\x00\x01bytes').decode(),
          'mime_type': mime_type.split(';')[0],
      },
  ]


async def test_unsupported_mime_type_is_rejected_before_sending():
  server = _Server()
  content = types.Content(
      role='user',
      parts=[
          types.Part(
              inline_data=types.Blob(data=b'x', mime_type='text/csv'),
          )
      ],
  )

  with pytest.raises(ValueError, match='text/csv'):
    await _classifier(server).classify(
        input=content, questions={'q': Binary(instructions='Is it?')}
    )

  assert server.requests == []


async def test_non_media_part_is_rejected_before_sending():
  server = _Server()
  content = types.Content(
      role='user',
      parts=[
          types.Part(function_call=types.FunctionCall(name='f', args={})),
      ],
  )

  with pytest.raises(ValueError, match='text, JSON and inline media only'):
    await _classifier(server).classify(
        input=content, questions={'q': Binary(instructions='Is it?')}
    )

  assert server.requests == []


@pytest.mark.parametrize('input', ['', types.Content(role='user', parts=[])])
async def test_empty_input_is_rejected(input):
  server = _Server()

  with pytest.raises(ValueError, match='empty input'):
    await _classifier(server).classify(
        input=input, questions={'q': Binary(instructions='Is it?')}
    )

  assert server.requests == []


async def test_question_without_instructions_is_rejected():
  server = _Server()

  with pytest.raises(ValueError, match="needs instructions for question 'q'"):
    await _classifier(server).classify(input='text', questions={'q': Binary()})

  assert server.requests == []


async def test_no_questions_is_rejected():
  with pytest.raises(ValueError, match='at least one question'):
    await _classifier(_Server()).classify(input='text', questions={})


async def test_too_many_labels_are_rejected():
  question = Choice(
      instructions='Which?', criteria={f'l{i}': None for i in range(256)}
  )

  with pytest.raises(ValueError, match='at most 255 labels'):
    await _classifier(_Server()).classify(
        input='text', questions={'q': question}
    )


# Configuration.


async def test_reads_the_key_from_the_environment(monkeypatch):
  monkeypatch.setenv('GEMINI_API_KEY', 'gemini-key')
  monkeypatch.setenv('GOOGLE_API_KEY', 'google-key')
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  classifier = GeminiDecisionsClassifier(
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
  )

  await classifier.classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert server.requests[0].headers['x-goog-api-key'] == 'gemini-key'


async def test_falls_back_to_google_api_key(monkeypatch):
  monkeypatch.setenv('GOOGLE_API_KEY', 'google-key')
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  classifier = GeminiDecisionsClassifier(
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
  )

  await classifier.classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert server.requests[0].headers['x-goog-api-key'] == 'google-key'


async def test_missing_api_key_raises_before_sending():
  server = _Server()
  classifier = GeminiDecisionsClassifier(
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
  )

  with pytest.raises(ValueError, match='GEMINI_API_KEY or GOOGLE_API_KEY'):
    await classifier.classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )

  assert server.requests == []


async def test_base_url_field_overrides_the_default():
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))

  await _classifier(server, base_url='https://proxy.example.com/').classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert str(server.requests[0].url) == (
      'https://proxy.example.com/v1beta/decisions'
  )


async def test_uses_the_base_url_of_the_supplied_client():
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  classifier = GeminiDecisionsClassifier(
      api_key='test-key',
      http_client=httpx.AsyncClient(
          transport=httpx.MockTransport(server),
          base_url='https://proxy.example.com',
      ),
  )

  await classifier.classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert str(server.requests[0].url) == (
      'https://proxy.example.com/v1beta/decisions'
  )


async def test_default_client_waits_30_seconds(monkeypatch):
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  real_client = httpx.AsyncClient
  monkeypatch.setattr(
      _gemini_decisions_classifier.httpx,
      'AsyncClient',
      lambda **kwargs: real_client(
          transport=httpx.MockTransport(server), **kwargs
      ),
  )

  await GeminiDecisionsClassifier(api_key='test-key').classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  request = server.requests[0]
  assert str(request.url) == _URL
  assert request.extensions['timeout']['read'] == 30.0


async def test_default_client_is_closed_after_each_call(monkeypatch):
  answer = {'q': {'type': 'noul', 'noul': 0.5}}
  server = _Server(_ok(answer), _ok(answer))
  real_client = httpx.AsyncClient
  clients = []

  def _client(**kwargs):
    client = real_client(transport=httpx.MockTransport(server), **kwargs)
    clients.append(client)
    return client

  monkeypatch.setattr(
      _gemini_decisions_classifier.httpx, 'AsyncClient', _client
  )
  classifier = GeminiDecisionsClassifier(api_key='test-key')

  for _ in range(2):
    await classifier.classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )

  assert len(clients) == 2
  assert all(client.is_closed for client in clients)


async def test_default_client_presents_the_client_certificate(monkeypatch):
  calls = []

  def fake_kwargs(*args, **kwargs):
    calls.append(kwargs)
    return {'headers': {'x-from-mtls': '1'}}

  monkeypatch.setattr(
      _gemini_decisions_classifier._mtls_utils,
      'client_cert_httpx_kwargs',
      fake_kwargs,
  )
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  real_client = httpx.AsyncClient
  monkeypatch.setattr(
      _gemini_decisions_classifier.httpx,
      'AsyncClient',
      lambda **kwargs: real_client(
          transport=httpx.MockTransport(server), **kwargs
      ),
  )

  await GeminiDecisionsClassifier(api_key='test-key').classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert calls == [{'url': _URL}]
  assert server.requests[0].headers['x-from-mtls'] == '1'


def test_api_key_is_not_serialized_or_shown():
  classifier = GeminiDecisionsClassifier(api_key='secret-key')

  assert 'secret-key' not in repr(classifier)
  assert 'secret-key' not in classifier.model_dump_json()


async def test_request_headers_name_adk_and_the_classifier():
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))

  await _classifier(server).classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  headers = server.requests[0].headers
  for name in ('user-agent', 'x-goog-api-client'):
    labels = headers[name].split(' ')
    assert labels[0].startswith('google-adk/')
    assert labels[-1] == classifier_client_label('gemini_decisions')


async def test_runtime_tag_is_sent_to_google_only(monkeypatch):
  monkeypatch.setenv('GOOGLE_CLOUD_AGENT_ENGINE_ID', 'engine-id')
  google = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  proxy = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}))
  questions = {'q': Binary(instructions='Is it?')}

  await _classifier(google).classify(input='text', questions=questions)
  await _classifier(proxy, base_url='https://proxy.example.com').classify(
      input='text', questions=questions
  )

  bare = f'google-adk/{version.__version__}'
  assert google.requests[0].headers['user-agent'].split(' ')[0] != bare
  assert proxy.requests[0].headers['user-agent'].split(' ')[0] == bare


# Safety refusals.


@pytest.mark.parametrize(
    'code',
    [
        'safety',
        'prohibited_content',
        'blocklist',
        'image_safety',
        'content_blocked',
    ],
)
async def test_blocked_input_refuses_every_question(code):
  server = _Server(_blocked(code))

  response = await _classifier(server).classify(
      input='text',
      questions={'route': _ROUTE, 'urgent': _URGENT, 'f': _FRUSTRATION},
  )

  assert len(server.requests) == 1
  assert response.answers == {
      'route': RefusalAnswer(),
      'urgent': RefusalAnswer(),
      'f': RefusalAnswer(),
  }


@pytest.mark.parametrize(
    'response',
    [
        httpx.Response(
            400, json={'error': {'code': 'invalid_argument', 'message': 'x'}}
        ),
        httpx.Response(400, json={'error': 'safety'}),
        httpx.Response(400, json=['safety']),
        httpx.Response(400, text='not json'),
        httpx.Response(403, json={'error': {'code': 'safety'}}),
    ],
)
async def test_other_errors_are_not_refusals(response):
  server = _Server(response)

  with pytest.raises(RuntimeError, match='request failed with'):
    await _classifier(server).classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )


async def test_classifier_node_fails_loudly_on_a_refusal():
  node = ClassifierNode(
      name='triage',
      classifier=_classifier(_Server(_blocked())),
      question=_ROUTE,
  )
  wf = Workflow(
      name='wf',
      edges=[
          (START, node),
          (
              node,
              {
                  'billing': InputCapturingNode(name='billing'),
                  'technical': InputCapturingNode(name='technical'),
              },
          ),
      ],
  )

  with pytest.raises(WorkflowDataError, match='declined to answer'):
    await run_workflow(wf, message='something')


# Retries.


@pytest.mark.parametrize('status', [429, 500, 503])
async def test_retries_rate_limited_and_server_error_responses(status):
  server = _Server(
      httpx.Response(status, json={'error': {'code': 'too_many_requests'}}),
      httpx.Response(status),
      _ok({'q': {'type': 'noul', 'noul': 0.7}}),
  )

  response = await _classifier(server).classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert len(server.requests) == 3
  assert response.answers['q'] == BinaryAnswer(probability=0.7)


async def test_gives_up_after_max_attempts():
  server = _Server(
      httpx.Response(429, text='slow down'),
      httpx.Response(429, text='slow down'),
      httpx.Response(429, text='slow down'),
  )

  with pytest.raises(RuntimeError, match='429.*slow down'):
    await _classifier(server, max_attempts=2).classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )

  assert len(server.requests) == 2
  assert len(server.responses) == 1


@pytest.mark.parametrize(
    'error',
    [
        httpx.ReadTimeout('timed out'),
        httpx.ConnectError('refused'),
        httpx.RemoteProtocolError('dropped'),
    ],
)
async def test_retries_transient_transport_errors(error):
  server = _Server(error, _ok({'q': {'type': 'noul', 'noul': 0.7}}))

  response = await _classifier(server).classify(
      input='text', questions={'q': Binary(instructions='Is it?')}
  )

  assert len(server.requests) == 2
  assert response.answers['q'] == BinaryAnswer(probability=0.7)


async def test_permanent_transport_errors_are_not_retried():
  server = _Server(httpx.UnsupportedProtocol('bad url'))

  with pytest.raises(httpx.UnsupportedProtocol):
    await _classifier(server).classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )

  assert len(server.requests) == 1


@pytest.mark.parametrize('status', [400, 401, 403, 404])
async def test_client_errors_are_not_retried(status):
  server = _Server(httpx.Response(status, text='nope'))

  with pytest.raises(RuntimeError, match=f'{status}.*nope'):
    await _classifier(server).classify(
        input='text', questions={'q': Binary(instructions='Is it?')}
    )

  assert len(server.requests) == 1


@pytest.mark.parametrize(
    ('header', 'expected'),
    [
        ('2', 2.0),
        ('0', 0.0),
        ('500', _gemini_decisions_classifier._MAX_RETRY_AFTER),
        ('-3', 0.0),
        ('soon', None),
        ('nan', None),
        (None, None),
    ],
)
def test_retry_after_header(header, expected):
  headers = {} if header is None else {'retry-after': header}

  delay = _gemini_decisions_classifier._retry_after(
      httpx.Response(429, headers=headers)
  )

  assert delay == expected


def _retry_state(response: httpx.Response | None) -> Any:
  error = (
      _gemini_decisions_classifier._RetryableStatusError(response)
      if response is not None
      else httpx.ReadTimeout('timed out')
  )
  return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: error))


def test_waits_as_long_as_retry_after_says():
  state = _retry_state(httpx.Response(429, headers={'retry-after': '7'}))

  assert _gemini_decisions_classifier._wait(state) == 7.0


@pytest.mark.parametrize(
    'response', [httpx.Response(503), httpx.Response(429), None]
)
def test_backs_off_without_a_retry_after_header(monkeypatch, response):
  monkeypatch.setattr(
      _gemini_decisions_classifier, '_RETRY_WAIT', lambda retry_state: 0.25
  )

  assert _gemini_decisions_classifier._wait(_retry_state(response)) == 0.25


# Answers.


@pytest.mark.parametrize(
    ('question', 'raw', 'expected'),
    [
        (
            Binary(instructions='Is it?'),
            {'type': 'noul', 'noul': 1.0000001, 'confidence': 1.0000001},
            BinaryAnswer(probability=1.0, confidence=1.0),
        ),
        (
            Binary(instructions='Is it?'),
            {'type': 'noul', 'noul': -0.0000001},
            BinaryAnswer(probability=0.0),
        ),
        (
            Choice(instructions='Which?', criteria={'a': None, 'b': None}),
            {
                'type': 'choice',
                'choice': 'a',
                'probabilities': {'a': 1.0000001, 'b': -0.0000001},
            },
            ChoiceAnswer(choice='a', probabilities={'a': 1.0, 'b': 0.0}),
        ),
        (
            Score(instructions='How much?', criteria=['lo', 'hi']),
            {
                'type': 'score',
                'score': 1.0000001,
                'probabilities': {'0': -0.0000001, '1': 1.0000001},
            },
            ScoreAnswer(score=1.0, probabilities={0: 0.0, 1: 1.0}),
        ),
    ],
)
async def test_values_a_rounding_error_out_of_range_are_clamped(
    question, raw, expected
):
  assert await _classify_one(question, raw) == expected


_BINARY = Binary(instructions='Is it?')
_CHOICE = Choice(instructions='Which?', criteria={'a': None, 'b': None})
_SCORE = Score(instructions='How much?', criteria=['lo', 'mid', 'hi'])


@pytest.mark.parametrize(
    ('question', 'raw', 'match'),
    [
        (_BINARY, {'type': 'noul'}, r"missing \['noul'\]"),
        (_CHOICE, {'type': 'choice', 'choice': 'a'}, 'missing'),
        (
            _CHOICE,
            {'type': 'choice', 'probabilities': {'a': 1, 'b': 0}},
            'missing',
        ),
        (_SCORE, {'type': 'score', 'probabilities': {'0': 1}}, 'missing'),
        (_BINARY, {'type': 'choice', 'choice': 'a'}, "'choice' answer"),
        (_BINARY, {'noul': 0.5}, 'None answer'),
        (_BINARY, {'type': 'noul', 'noul': 'high'}, 'not a number'),
        (_BINARY, {'type': 'noul', 'noul': True}, 'not a number'),
        (_BINARY, {'type': 'noul', 'noul': 1.5}, r'outside \[0, 1.0\]'),
        (
            _BINARY,
            {'type': 'noul', 'noul': 0.5, 'confidence': 2},
            r'outside \[0, 1.0\]',
        ),
        (
            _CHOICE,
            {
                'type': 'choice',
                'choice': 'c',
                'probabilities': {'a': 0.5, 'b': 0.5},
            },
            'not one of its labels',
        ),
        (
            _CHOICE,
            {
                'type': 'choice',
                'choice': 'a',
                'probabilities': {'a': 0.5, 'c': 0.5},
            },
            'expected',
        ),
        (
            _CHOICE,
            {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 1.0}},
            'expected',
        ),
        (
            _CHOICE,
            {'type': 'choice', 'choice': 'a', 'probabilities': [0.5, 0.5]},
            'not an object',
        ),
        (
            _SCORE,
            {
                'type': 'score',
                'score': 1.0,
                'probabilities': {'0': 0.5, '1': 0.5},
            },
            'expected',
        ),
        (
            _SCORE,
            {
                'type': 'score',
                'score': 2.5,
                'probabilities': {'0': 0.0, '1': 0.0, '2': 1.0},
            },
            r'outside \[0, 2\]',
        ),
    ],
)
async def test_malformed_answer_raises_value_error(question, raw, match):
  with pytest.raises(ValueError, match=match):
    await _classify_one(question, raw)


async def test_missing_answer_raises():
  server = _Server(_ok({}))

  with pytest.raises(ValueError, match="no answer to 'q'"):
    await _classifier(server).classify(input='text', questions={'q': _BINARY})


@pytest.mark.parametrize(
    ('response', 'match'),
    [
        (httpx.Response(200, json=['answers']), 'JSON object'),
        (httpx.Response(200, json={}), 'no answers'),
        (httpx.Response(200, json={'answers': []}), 'no answers'),
    ],
)
async def test_malformed_response_raises(response, match):
  with pytest.raises(ValueError, match=match):
    await _classifier(_Server(response)).classify(
        input='text', questions={'q': _BINARY}
    )


@pytest.mark.parametrize('field', ['modelVersion', 'model_version'])
async def test_reads_a_model_version_when_the_api_reports_one(field):
  server = _Server(_ok({'q': {'type': 'noul', 'noul': 0.5}}, **{field: 'v2'}))

  response = await _classifier(server).classify(
      input='text', questions={'q': _BINARY}
  )

  assert response.model_version == 'v2'


# Routing.


async def test_classifier_node_routes_a_choice():
  node = ClassifierNode(
      name='triage',
      classifier=_classifier(
          _Server(
              _ok({
                  'route': {
                      'type': 'choice',
                      'choice': 'technical',
                      'probabilities': {'billing': 0.1, 'technical': 0.9},
                      'confidence': 0.9,
                  }
              })
          )
      ),
      question=_ROUTE,
  )
  billing = InputCapturingNode(name='billing')
  technical = InputCapturingNode(name='technical')
  wf = Workflow(
      name='wf',
      edges=[
          (START, node),
          (node, {'billing': billing, 'technical': technical}),
      ],
  )

  await run_workflow(wf, message='The API returns 500')

  assert billing.received_inputs == []
  assert len(technical.received_inputs) == 1
  assert technical.received_inputs[0].parts[0].text == 'The API returns 500'


@pytest.mark.parametrize(
    ('probability', 'expected_route'),
    [(0.7, 'yes'), (0.6, 'yes'), (0.59, 'no')],
)
async def test_classifier_node_routes_a_binary_on_its_threshold(
    probability, expected_route
):
  node = ClassifierNode(
      name='gate',
      classifier=_classifier(
          _Server(
              _ok({
                  'route': {
                      'type': 'noul',
                      'noul': probability,
                      'confidence': max(probability, 1 - probability),
                  }
              })
          )
      ),
      question=Binary(instructions='Is this risky?'),
      threshold=0.6,
  )
  nodes = {
      'yes': InputCapturingNode(name='yes'),
      'no': InputCapturingNode(name='no'),
  }
  wf = Workflow(
      name='wf',
      edges=[(START, node), (node, {True: nodes['yes'], False: nodes['no']})],
  )

  await run_workflow(wf, message='delete everything')

  for name, sink in nodes.items():
    assert len(sink.received_inputs) == (1 if name == expected_route else 0)

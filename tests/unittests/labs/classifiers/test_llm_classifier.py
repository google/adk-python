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

"""Tests for LlmClassifier."""

from __future__ import annotations

import json
from typing import Any
from typing import AsyncGenerator

from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import LlmClassifier
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
from google.adk.models import registry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
import pydantic
import pytest


class _FakeLlm(BaseLlm):
  """Replies with fixed text and records the request it was sent."""

  model: str = 'fake-llm'

  reply: str = '{}'

  error_code: str | None = None

  model_version: str | None = None

  finish_reason: types.FinishReason | None = None

  requests: list[LlmRequest] = pydantic.Field(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.requests.append(llm_request)
    if self.error_code:
      yield LlmResponse(error_code=self.error_code, error_message='boom')
      return
    yield LlmResponse(
        content=types.Content(
            role='model',
            parts=[
                types.Part(text='thinking...', thought=True),
                types.Part(text=self.reply),
            ],
        ),
        model_version=self.model_version,
        finish_reason=self.finish_reason,
    )


def _classifier(reply: dict[str, Any] | str) -> tuple[LlmClassifier, _FakeLlm]:
  llm = _FakeLlm(reply=reply if isinstance(reply, str) else json.dumps(reply))
  return LlmClassifier(model=llm), llm


_ROUTE = Choice(
    instructions='Which department handles this?',
    criteria={'billing': 'Invoices and refunds', 'technical': None},
)


async def test_choice_picks_the_most_likely_label():
  classifier, _ = _classifier({'route': {'billing': 0.8, 'technical': 0.2}})

  response = await classifier.classify(
      input='I was charged twice.', questions={'route': _ROUTE}
  )

  assert response.answers['route'] == ChoiceAnswer(
      choice='billing', probabilities={'billing': 0.8, 'technical': 0.2}
  )


async def test_choice_probabilities_are_normalized():
  classifier, _ = _classifier({'route': {'billing': 3, 'technical': 1}})

  response = await classifier.classify(input='x', questions={'route': _ROUTE})

  assert response.answers['route'].probabilities == {
      'billing': 0.75,
      'technical': 0.25,
  }


async def test_binary_returns_probability_of_true():
  classifier, _ = _classifier({'risky': {'probability': 0.3}})

  response = await classifier.classify(
      input={'tool': 'delete_account'},
      questions={'risky': Binary(instructions='Is this destructive?')},
  )

  assert response.answers['risky'] == BinaryAnswer(probability=0.3)


async def test_score_returns_expected_level():
  classifier, _ = _classifier({'urgency': {'0': 0.2, '1': 0.3, '2': 0.5}})

  response = await classifier.classify(
      input='Site is down.',
      questions={'urgency': Score(criteria=['low', 'medium', 'high'])},
  )

  answer = response.answers['urgency']
  assert isinstance(answer, ScoreAnswer)
  assert answer.score == pytest.approx(1.3)
  assert answer.probabilities == {0: 0.2, 1: 0.3, 2: 0.5}


async def test_answers_several_questions_in_one_call():
  classifier, llm = _classifier({
      'route': {'billing': 0.9, 'technical': 0.1},
      'escalate': {'probability': 0.7},
  })

  response = await classifier.classify(
      input='x', questions={'route': _ROUTE, 'escalate': Binary()}
  )

  assert set(response.answers) == {'route', 'escalate'}
  assert len(llm.requests) == 1


async def test_response_reports_the_model_version():
  llm = _FakeLlm(
      reply=json.dumps({'risky': {'probability': 0.5}}),
      model_version='fake-llm-001',
  )

  response = await LlmClassifier(model=llm).classify(
      input='x', questions={'risky': Binary()}
  )

  assert response.model_version == 'fake-llm-001'


async def test_response_model_version_is_none_when_not_reported():
  classifier, _ = _classifier({'risky': {'probability': 0.5}})

  response = await classifier.classify(input='x', questions={'risky': Binary()})

  assert response.model_version is None


async def test_request_constrains_output_to_the_question_labels():
  classifier, llm = _classifier({'route': {'billing': 1, 'technical': 0}})

  await classifier.classify(input='x', questions={'route': _ROUTE})

  config = llm.requests[0].config
  assert config.response_mime_type == 'application/json'
  route_schema = config.response_schema['properties']['route']
  assert route_schema['required'] == ['billing', 'technical']


async def test_request_describes_questions_in_system_instruction():
  classifier, llm = _classifier({'route': {'billing': 1, 'technical': 0}})

  await classifier.classify(input='x', questions={'route': _ROUTE})

  instruction = llm.requests[0].config.system_instruction
  assert 'Which department handles this?' in instruction
  assert '"billing": Invoices and refunds' in instruction


async def test_request_describes_binary_criteria():
  classifier, llm = _classifier({'risky': {'probability': 0.5}})

  await classifier.classify(
      input='x', questions={'risky': Binary(criteria='Deletes user data')}
  )

  instruction = llm.requests[0].config.system_instruction
  assert 'True means: Deletes user data' in instruction


async def test_request_keeps_non_ascii_labels_readable():
  classifier, llm = _classifier({'lang': {'日本語': 0.9, 'English': 0.1}})

  await classifier.classify(
      input='x',
      questions={'lang': Choice(criteria={'日本語': None, 'English': None})},
  )

  assert '"日本語"' in llm.requests[0].config.system_instruction


async def test_json_input_is_sent_as_json_text():
  classifier, llm = _classifier({'risky': {'probability': 0.5}})

  await classifier.classify(
      input={'tool': 'pay', 'amount': 5}, questions={'risky': Binary()}
  )

  sent = llm.requests[0].contents[0]
  assert sent.role == 'user'
  assert json.loads(sent.parts[0].text) == {'tool': 'pay', 'amount': 5}


async def test_content_input_is_sent_as_user_parts():
  classifier, llm = _classifier({'risky': {'probability': 0.5}})
  image = types.Part.from_bytes(data=b'png', mime_type='image/png')

  await classifier.classify(
      input=types.Content(role='model', parts=[image]),
      questions={'risky': Binary()},
  )

  sent = llm.requests[0].contents[0]
  assert sent.role == 'user'
  assert sent.parts == [image]


async def test_generate_content_config_is_passed_through():
  llm = _FakeLlm(reply=json.dumps({'risky': {'probability': 0.5}}))
  classifier = LlmClassifier(
      model=llm,
      generate_content_config=types.GenerateContentConfig(temperature=0.0),
  )

  await classifier.classify(input='x', questions={'risky': Binary()})

  assert llm.requests[0].config.temperature == 0.0
  assert classifier.generate_content_config.system_instruction is None


async def test_missing_label_raises():
  classifier, _ = _classifier({'route': {'billing': 1.0}})

  with pytest.raises(ValueError, match='technical'):
    await classifier.classify(input='x', questions={'route': _ROUTE})


@pytest.mark.parametrize('value', [None, '0.4', True, [1]])
async def test_non_numeric_probability_raises(value):
  classifier, _ = _classifier({'route': {'billing': value, 'technical': 0.5}})

  with pytest.raises(ValueError, match='not a number'):
    await classifier.classify(input='x', questions={'route': _ROUTE})


async def test_non_finite_probability_raises():
  classifier, _ = _classifier('{"risky": {"probability": NaN}}')

  with pytest.raises(ValueError, match='not a number'):
    await classifier.classify(input='x', questions={'risky': Binary()})


async def test_binary_probability_out_of_range_raises():
  classifier, _ = _classifier({'risky': {'probability': 70}})

  with pytest.raises(ValueError, match=r'outside \[0, 1\]'):
    await classifier.classify(input='x', questions={'risky': Binary()})


async def test_missing_question_raises():
  classifier, _ = _classifier({})

  with pytest.raises(ValueError, match='route'):
    await classifier.classify(input='x', questions={'route': _ROUTE})


async def test_invalid_json_raises():
  classifier, _ = _classifier('not json')

  with pytest.raises(ValueError, match='valid JSON'):
    await classifier.classify(input='x', questions={'route': _ROUTE})


async def test_model_error_raises():
  llm = _FakeLlm(error_code='RESOURCE_EXHAUSTED')

  with pytest.raises(RuntimeError, match='RESOURCE_EXHAUSTED'):
    await LlmClassifier(model=llm).classify(
        input='x', questions={'route': _ROUTE}
    )


async def test_truncated_response_raises():
  llm = _FakeLlm(
      reply='{"route": {"bil',
      finish_reason=types.FinishReason.MAX_TOKENS,
  )

  with pytest.raises(RuntimeError, match='output tokens'):
    await LlmClassifier(model=llm).classify(
        input='x', questions={'route': _ROUTE}
    )


async def test_empty_questions_raises():
  classifier, _ = _classifier({})

  with pytest.raises(ValueError, match='at least one question'):
    await classifier.classify(input='x', questions={})


async def test_model_name_resolves_through_registry(monkeypatch):
  class _RegisteredLlm(_FakeLlm):
    reply: str = json.dumps({'risky': {'probability': 0.4}})

  monkeypatch.setitem(
      registry._llm_registry_dict, 'registered-fake-llm', _RegisteredLlm
  )
  LLMRegistry.resolve.cache_clear()
  try:
    response = await LlmClassifier(model='registered-fake-llm').classify(
        input='x', questions={'risky': Binary()}
    )
  finally:
    LLMRegistry.resolve.cache_clear()

  assert response.answers['risky'] == BinaryAnswer(probability=0.4)


async def test_reassigning_model_name_resolves_the_new_model(monkeypatch):
  class _FirstLlm(_FakeLlm):
    reply: str = json.dumps({'risky': {'probability': 0.1}})

  class _SecondLlm(_FakeLlm):
    reply: str = json.dumps({'risky': {'probability': 0.9}})

  monkeypatch.setitem(registry._llm_registry_dict, 'first-fake-llm', _FirstLlm)
  monkeypatch.setitem(
      registry._llm_registry_dict, 'second-fake-llm', _SecondLlm
  )
  LLMRegistry.resolve.cache_clear()
  try:
    classifier = LlmClassifier(model='first-fake-llm')
    await classifier.classify(input='x', questions={'risky': Binary()})
    classifier.model = 'second-fake-llm'
    response = await classifier.classify(
        input='x', questions={'risky': Binary()}
    )
  finally:
    LLMRegistry.resolve.cache_clear()

  assert response.answers['risky'] == BinaryAnswer(probability=0.9)


def test_needs_model():
  with pytest.raises(pydantic.ValidationError, match='model'):
    LlmClassifier()


def test_empty_model_name_raises():
  with pytest.raises(pydantic.ValidationError, match='needs a model'):
    LlmClassifier(model='')


def test_config_with_response_schema_raises():
  with pytest.raises(pydantic.ValidationError, match='response schema'):
    LlmClassifier(
        model=_FakeLlm(),
        generate_content_config=types.GenerateContentConfig(
            response_schema={'type': 'object'}
        ),
    )

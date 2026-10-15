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

"""Tests for the classifier question and answer types."""

from __future__ import annotations

from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import ClassifierResponse
from google.adk.labs.classifiers import Question
from google.adk.labs.classifiers import Score
from google.adk.labs.classifiers import ScoreAnswer
import pydantic
import pytest


def test_choice_needs_at_least_two_labels():
  with pytest.raises(pydantic.ValidationError):
    Choice(criteria={'only': None})


def test_score_needs_at_least_two_levels():
  with pytest.raises(pydantic.ValidationError):
    Score(criteria=['low'])


def test_binary_criteria_describes_true():
  question = Binary(criteria='The action deletes user data.')

  assert question.criteria == 'The action deletes user data.'


def test_question_misspelled_field_raises():
  with pytest.raises(pydantic.ValidationError):
    Choice(criteria={'a': None, 'b': None}, instruction='typo')


def test_question_accepts_json_instructions_and_criteria():
  question = Choice(
      instructions={'task': 'route', 'context': ['ticket']},
      criteria={'a': {'examples': ['x']}, 'b': None},
  )

  assert question.instructions == {'task': 'route', 'context': ['ticket']}


def test_question_rejects_non_json_content():
  with pytest.raises(pydantic.ValidationError):
    Choice(instructions={'tags': {1, 2}}, criteria={'a': None, 'b': None})


@pytest.mark.parametrize(
    'make_answer',
    [
        lambda: BinaryAnswer(probability=0.5, extra=1),
        lambda: ChoiceAnswer(choice='a', probabilities={'a': 1.0}, extra=1),
        lambda: ScoreAnswer(score=0, probabilities={0: 1.0}, extra=1),
        lambda: ClassifierResponse(answers={}, extra=1),
    ],
)
def test_answer_rejects_extra_fields(make_answer):
  with pytest.raises(pydantic.ValidationError):
    make_answer()


def test_question_deserializes_by_type():
  adapter = pydantic.TypeAdapter(dict[str, Question])

  questions = adapter.validate_python({
      'route': {'type': 'choice', 'criteria': {'a': None, 'b': None}},
      'risky': {'type': 'binary'},
      'urgency': {'type': 'score', 'criteria': ['low', 'high']},
  })

  assert isinstance(questions['route'], Choice)
  assert isinstance(questions['risky'], Binary)
  assert isinstance(questions['urgency'], Score)


@pytest.mark.parametrize(
    'make_answer',
    [
        lambda p: BinaryAnswer(probability=p),
        lambda p: BinaryAnswer(probability=0.5, confidence=p),
        lambda p: ChoiceAnswer(choice='a', probabilities={'a': p}),
        lambda p: ChoiceAnswer(
            choice='a', probabilities={'a': 1.0}, confidence=p
        ),
        lambda p: ScoreAnswer(score=0, probabilities={0: p}),
        lambda p: ScoreAnswer(score=0, probabilities={0: 1.0}, confidence=p),
    ],
)
@pytest.mark.parametrize('probability', [-0.1, 1.5])
def test_answer_rejects_probability_outside_zero_to_one(
    make_answer, probability
):
  with pytest.raises(pydantic.ValidationError):
    make_answer(probability)


def test_classifier_response_round_trips_through_json():
  response = ClassifierResponse(
      model_version='m',
      answers={
          'route': ChoiceAnswer(choice='a', probabilities={'a': 0.9, 'b': 0.1}),
          'risky': BinaryAnswer(probability=0.2, confidence=0.9),
          'urgency': ScoreAnswer(
              score=0.75, probabilities={0: 0.25, 1: 0.75}, confidence=0.5
          ),
      },
  )

  restored = ClassifierResponse.model_validate_json(response.model_dump_json())

  assert restored == response

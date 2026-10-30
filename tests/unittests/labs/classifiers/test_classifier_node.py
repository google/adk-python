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

"""Tests for ClassifierNode."""

from __future__ import annotations

from collections.abc import Mapping
import datetime
import enum
import json
from typing import Any

from google.adk.labs.classifiers import Answer
from google.adk.labs.classifiers import BaseClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import ClassifierInput
from google.adk.labs.classifiers import ClassifierNode
from google.adk.labs.classifiers import ClassifierResponse
from google.adk.labs.classifiers import Question
from google.adk.labs.classifiers import RefusalAnswer
from google.adk.labs.classifiers import ScoreAnswer
from google.adk.workflow import DEFAULT_ROUTE
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._errors import WorkflowDataError
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
import pytest

from ...workflow.workflow_testing_utils import InputCapturingNode
from ...workflow.workflow_testing_utils import run_workflow
from ...workflow.workflow_testing_utils import TestingNode

_TRIAGE = Choice(
    instructions='Route this ticket.',
    criteria={'billing': 'Invoices or refunds', 'technical': 'Bugs'},
)


class _FixedClassifier(BaseClassifier):
  """Returns the same answer to every question and records each call."""

  model: str = 'fixed'
  answer: Answer | None
  calls: list[dict[str, Any]] = Field(default_factory=list)

  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    self.calls.append({'input': input, 'questions': dict(questions)})
    answers = {} if self.answer is None else {n: self.answer for n in questions}
    return ClassifierResponse(model_version=self.model, answers=answers)


def _choice(label: str) -> ChoiceAnswer:
  return ChoiceAnswer(
      choice=label,
      probabilities={'billing': 0.9, 'technical': 0.1}
      if label == 'billing'
      else {'billing': 0.1, 'technical': 0.9},
  )


def _triage_node(answer: Answer | None, **kwargs: Any) -> ClassifierNode:
  return ClassifierNode(
      name='triage',
      classifier=_FixedClassifier(answer=answer),
      question=_TRIAGE,
      **kwargs,
  )


def _workflow_from(source: TestingNode, node: ClassifierNode) -> Workflow:
  """START -> source -> node, with every route of node going to one sink."""
  return Workflow(
      name='wf',
      edges=[
          (START, source),
          (source, node),
          (node, InputCapturingNode(name='sink')),
      ],
  )


async def test_choice_routes_to_chosen_label_and_passes_input_through():
  """A Choice routes to the chosen label and forwards the unchanged input."""
  node = _triage_node(_choice('technical'))
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
  received = technical.received_inputs[0]
  assert isinstance(received, types.Content)
  assert received.parts[0].text == 'The API returns 500'
  call = node.classifier.calls[0]
  assert call['input'] == received
  assert call['questions'] == {'route': _TRIAGE}


@pytest.mark.parametrize(
    ('probability', 'expected_route'),
    [(0.7, True), (0.6, True), (0.59, False)],
)
async def test_binary_routes_on_threshold(probability, expected_route):
  """A Binary routes True at or above the threshold, False below it."""
  node = ClassifierNode(
      name='gate',
      classifier=_FixedClassifier(answer=BinaryAnswer(probability=probability)),
      question=Binary(instructions='Is this risky?'),
      threshold=0.6,
  )
  yes = InputCapturingNode(name='yes')
  no = InputCapturingNode(name='no')
  wf = Workflow(
      name='wf', edges=[(START, node), (node, {True: yes, False: no})]
  )

  await run_workflow(wf)

  assert len(yes.received_inputs) == expected_route
  assert len(no.received_inputs) == (not expected_route)


async def test_answer_key_stores_answer_in_state():
  """answer_key stores the full answer in session state."""
  node = _triage_node(_choice('billing'), answer_key='triage_answer')
  billing = InputCapturingNode(name='billing')
  wf = Workflow(
      name='wf',
      edges=[
          (START, node),
          (node, {'billing': billing, DEFAULT_ROUTE: billing}),
      ],
  )

  _, ss, session = await run_workflow(wf)

  stored = await ss.get_session(
      app_name=session.app_name, user_id=session.user_id, session_id=session.id
  )
  assert stored.state['triage_answer'] == _choice('billing').model_dump()


async def test_no_answer_key_leaves_state_untouched():
  """Without answer_key, the node writes nothing to state."""
  node = _triage_node(_choice('billing'))
  billing = InputCapturingNode(name='billing')
  technical = InputCapturingNode(name='technical')
  wf = Workflow(
      name='wf',
      edges=[
          (START, node),
          (node, {'billing': billing, 'technical': technical}),
      ],
  )

  _, ss, session = await run_workflow(wf)

  stored = await ss.get_session(
      app_name=session.app_name, user_id=session.user_id, session_id=session.id
  )
  assert stored.state == {}


class _Ticket(BaseModel):
  subject: str


async def test_model_input_is_classified_as_dict():
  """A model input is classified as a dict and forwarded as the model."""
  node = _triage_node(_choice('billing'))
  source = TestingNode(name='src', output=_Ticket(subject='x'))
  billing = InputCapturingNode(name='billing')
  wf = Workflow(
      name='wf',
      edges=[
          (START, source),
          (source, node),
          (node, {'billing': billing, DEFAULT_ROUTE: billing}),
      ],
  )

  await run_workflow(wf)

  assert node.classifier.calls[0]['input'] == {'subject': 'x'}
  assert billing.received_inputs == [_Ticket(subject='x')]


async def test_model_input_is_dumped_json_safe():
  """Enums and dates in a model input reach the classifier as JSON values."""

  class Kind(enum.Enum):
    BUG = 'bug'

  class Ticket(BaseModel):
    kind: Kind
    opened: datetime.date

  node = _triage_node(_choice('billing'))
  source = TestingNode(
      name='src',
      output=Ticket(kind=Kind.BUG, opened=datetime.date(2026, 1, 2)),
  )

  await run_workflow(_workflow_from(source, node))

  classifier_input = node.classifier.calls[0]['input']
  assert classifier_input == {'kind': 'bug', 'opened': '2026-01-02'}
  json.dumps(classifier_input)


def test_model_dump_keeps_classifier_subclass_fields():
  """model_dump keeps the fields of the classifier's subclass."""
  node = _triage_node(_choice('billing'))

  assert node.model_dump()['classifier']['answer'] == (
      _choice('billing').model_dump()
  )


@pytest.mark.parametrize(
    ('answer', 'match'),
    [
        (None, 'returned no answer'),
        (_choice('billing').model_copy(update={'choice': 'sales'}), 'sales'),
        (BinaryAnswer(probability=0.5), 'binary answer to a choice question'),
        (
            ScoreAnswer(score=1.0, probabilities={0: 0.0, 1: 1.0}),
            'score answer to a choice question',
        ),
        (RefusalAnswer(), 'declined to answer'),
    ],
)
async def test_unusable_answer_raises(answer, match):
  """An answer that cannot pick a route fails the run."""
  node = _triage_node(answer)
  source = TestingNode(name='src', output='x')

  with pytest.raises(WorkflowDataError, match=match):
    await run_workflow(_workflow_from(source, node))


async def test_missing_input_raises():
  """A node with no input to classify fails the run."""
  node = _triage_node(_choice('billing'))
  source = TestingNode(name='src', route='go')
  wf = Workflow(
      name='wf',
      edges=[
          (START, source),
          (source, {'go': node}),
          (node, InputCapturingNode(name='sink')),
      ],
  )

  with pytest.raises(WorkflowDataError, match='no input'):
    await run_workflow(wf)


def test_edge_for_unknown_route_fails_validation():
  """An edge for a route the node never emits fails graph validation."""
  node = _triage_node(_choice('billing'))
  a = InputCapturingNode(name='a')
  b = InputCapturingNode(name='b')

  with pytest.raises(ValidationError, match=r"never emits: \['biling'\]"):
    Workflow(
        name='wf',
        edges=[(START, node), (node, {'biling': a, 'technical': b})],
    )


def test_label_without_edge_fails_validation():
  """A label with no edge fails graph validation."""
  node = _triage_node(_choice('billing'))
  a = InputCapturingNode(name='a')

  with pytest.raises(
      ValidationError, match=r"no edge for routes \['technical'\]"
  ):
    Workflow(name='wf', edges=[(START, node), (node, {'billing': a})])


def test_binary_string_routes_fail_validation():
  """A Binary node's routes are bools, so string routes fail validation."""
  node = ClassifierNode(
      name='gate',
      classifier=_FixedClassifier(answer=BinaryAnswer(probability=1.0)),
      question=Binary(),
  )
  a = InputCapturingNode(name='a')
  b = InputCapturingNode(name='b')

  with pytest.raises(ValidationError, match='never emits'):
    Workflow(name='wf', edges=[(START, node), (node, {'true': a, 'false': b})])


def test_default_route_covers_missing_labels():
  """A DEFAULT_ROUTE edge covers labels without their own edge."""
  node = _triage_node(_choice('billing'))
  a = InputCapturingNode(name='a')
  b = InputCapturingNode(name='b')

  Workflow(
      name='wf', edges=[(START, node), (node, {'billing': a, DEFAULT_ROUTE: b})]
  )


def test_unrouted_edge_covers_missing_labels():
  """An unrouted edge covers labels without their own edge."""
  node = _triage_node(_choice('billing'))
  a = InputCapturingNode(name='a')
  b = InputCapturingNode(name='b')

  Workflow(
      name='wf',
      edges=[(START, node), (node, {'billing': a}), (node, b)],
  )


def test_unrouted_edges_skip_validation():
  """A node with only unrouted edges skips route validation."""
  node = _triage_node(_choice('billing'))
  a = InputCapturingNode(name='a')

  Workflow(name='wf', edges=[(START, node, a)])


def test_score_question_is_rejected():
  """A Score question cannot pick a route, so the node rejects it."""
  with pytest.raises(ValidationError):
    ClassifierNode(
        name='rate',
        classifier=_FixedClassifier(answer=None),
        question={'type': 'score', 'criteria': ['low', 'high']},
    )


def test_threshold_with_choice_question_is_rejected():
  """threshold has no effect on a Choice, so setting it is an error."""
  with pytest.raises(ValidationError, match='only applies to Binary'):
    _triage_node(_choice('billing'), threshold=0.8)

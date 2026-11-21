# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A judge sample contributes at most one consistent score per mapped rubric."""

from __future__ import annotations

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.rubric_based_final_response_quality_v1 import RubricBasedFinalResponseQualityV1Evaluator
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.genai import types
from pydantic import Field
import pytest


def _verdict(rubric_id, value, *, property_text=None, rationale='first'):
  return (
      f'ID: {rubric_id}\n'
      f'Property: {property_text or "Property " + rubric_id}\n'
      f'Rationale: {rationale}\nVerdict: {value}\n'
  )


_POSITIVE = _verdict('r1', 'yes') + _verdict('r2', 'no')
_NEGATIVE = _verdict('r1', 'no') + _verdict('r2', 'no')
_CONFLICT = _verdict('r1', 'yes') + _verdict('r1', 'no') + _verdict('r2', 'yes')
_BOTH_POSITIVE = _verdict('r1', 'yes') + _verdict('r2', 'yes')


@pytest.fixture
def make_evaluator(monkeypatch):
  class ScriptedJudge(BaseLlm):
    model: str = 'offline-rubric-judge'
    responses: list[str] = Field(default_factory=list)
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False):
      self.calls += 1
      yield LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part(text=self.responses.pop(0))]
          )
      )

  def make(responses):
    judge = ScriptedJudge(responses=list(responses))
    monkeypatch.setattr(
        LLMRegistry,
        'resolve',
        staticmethod(lambda model: lambda **kwargs: judge),
    )
    evaluator = RubricBasedFinalResponseQualityV1Evaluator(
        EvalMetric(
            metric_name='rubric_based_final_response_quality_v1',
            criterion=RubricsBasedCriterion(
                threshold=0.5,
                rubrics=[
                    Rubric(
                        rubric_id=rid,
                        rubric_content=RubricContent(
                            text_property=f'Property {rid}'
                        ),
                    )
                    for rid in ('r1', 'r2')
                ],
                judge_model_options=JudgeModelOptions(
                    judge_model='offline-rubric-judge',
                    num_samples=len(responses),
                ),
            ),
        )
    )
    return evaluator, judge

  return make


def _convert(make_evaluator, text):
  evaluator, _ = make_evaluator([text])
  evaluator.create_effective_rubrics_list(None)
  return evaluator.convert_auto_rater_response_to_score(
      LlmResponse(content=types.Content(parts=[types.Part(text=text)]))
  )


@pytest.mark.parametrize(
    'value,score', [('yes', 1.0), ('no', 0.0), ('invalid', None)]
)
def test_identical_scores_keep_one_result_and_first_rationale(
    make_evaluator, value, score
):
  """Repeated identical scores retain one entry and its original explanation."""
  result = _convert(
      make_evaluator,
      _verdict('r1', value)
      + _verdict('r1', value, rationale='repeated')
      + _verdict('r2', 'no'),
  )
  assert [(s.rubric_id, s.score) for s in result.rubric_scores] == [
      ('r1', score),
      ('r2', 0.0),
  ]
  assert result.rubric_scores[0].rationale == 'first'


@pytest.mark.parametrize(
    'first,second',
    [
        ('yes', 'no'),
        ('no', 'yes'),
        ('yes', 'invalid'),
        ('invalid', 'yes'),
        ('no', 'invalid'),
        ('invalid', 'no'),
    ],
)
def test_conflicting_mapped_scores_discard_the_whole_sample(
    make_evaluator, first, second, caplog
):
  """A conflict discards even the sample's previously valid other rubric."""
  result = _convert(
      make_evaluator,
      _verdict('r2', 'yes') + _verdict('r1', first) + _verdict('r1', second),
  )
  assert result.score is None
  assert result.rubric_scores == []
  assert 'conflicting scores' in caplog.text
  assert 'r1' in caplog.text


def test_property_fallback_coalesces_after_effective_id_mapping(make_evaluator):
  """Different raw IDs and normalized properties can denote the same rubric."""
  result = _convert(
      make_evaluator,
      _verdict('r1', 'yes')
      + _verdict('unknown', 'yes', property_text='**PROPERTY   R1**')
      + _verdict('', 'yes', property_text='Property r1')
      + _verdict('r2', 'no'),
  )
  assert [(s.rubric_id, s.score) for s in result.rubric_scores] == [
      ('r1', 1.0),
      ('r2', 0.0),
  ]
  assert result.score == 0.5


def test_property_fallback_conflict_discards_the_whole_sample(make_evaluator):
  """An aliased property cannot contribute a contradictory second verdict."""
  result = _convert(
      make_evaluator,
      _verdict('r1', 'yes')
      + _verdict('r2', 'yes')
      + _verdict('unknown', 'no', property_text='**PROPERTY   R1**'),
  )
  assert result.score is None
  assert result.rubric_scores == []


def test_unmapped_extra_verdicts_do_not_create_a_mapped_conflict(
    make_evaluator,
):
  """Unmatched extras remain ignored when all expected rubrics are answered."""
  result = _convert(
      make_evaluator,
      _POSITIVE + _verdict('unknown', 'yes') + _verdict('unknown', 'no'),
  )
  assert result.score == 0.5
  assert [s.rubric_id for s in result.rubric_scores] == ['r1', 'r2']


@pytest.mark.parametrize(
    'responses,score,status',
    [
        ([_CONFLICT], None, EvalStatus.NOT_EVALUATED),
        (
            [
                _verdict('r1', 'invalid')
                + _verdict('r1', 'yes')
                + _verdict('r2', 'yes')
            ],
            None,
            EvalStatus.NOT_EVALUATED,
        ),
        ([_CONFLICT, _BOTH_POSITIVE, _NEGATIVE], 0.0, EvalStatus.FAILED),
        ([_CONFLICT, _NEGATIVE, _NEGATIVE], 0.0, EvalStatus.FAILED),
        (
            [
                _verdict('r1', 'yes') * 3 + _verdict('r2', 'no'),
                _NEGATIVE,
                _NEGATIVE,
            ],
            0.0,
            EvalStatus.FAILED,
        ),
        (
            [
                _verdict('r1', 'no') * 3 + _verdict('r2', 'no'),
                _POSITIVE,
                _POSITIVE,
            ],
            0.5,
            EvalStatus.PASSED,
        ),
        (
            [
                _verdict('', 'yes', property_text='Property r1') * 3
                + _verdict('r2', 'no'),
                _NEGATIVE,
                _NEGATIVE,
            ],
            0.0,
            EvalStatus.FAILED,
        ),
        ([_POSITIVE, _POSITIVE, _NEGATIVE], 0.5, EvalStatus.PASSED),
        ([_POSITIVE, _NEGATIVE, _NEGATIVE], 0.0, EvalStatus.FAILED),
        ([_POSITIVE], 0.5, EvalStatus.PASSED),
    ],
)
async def test_real_evaluator_counts_independent_samples_once(
    make_evaluator, responses, score, status
):
  """Score real judge responses without letting a sample multiply its vote."""
  evaluator, judge = make_evaluator(responses)
  invocation = Invocation(
      user_content=types.Content(
          role='user', parts=[types.Part(text='Question')]
      ),
      final_response=types.Content(
          role='model', parts=[types.Part(text='Answer')]
      ),
  )
  result = await evaluator.evaluate_invocations([invocation])
  assert judge.calls == len(responses)
  assert result.overall_score == score
  assert result.overall_eval_status == status
  per_invocation = result.per_invocation_results[0]
  assert per_invocation.score == score
  assert per_invocation.eval_status == status
  if score is None:
    assert per_invocation.rubric_scores == []
  else:
    assert [s.rubric_id for s in per_invocation.rubric_scores] == ['r1', 'r2']


async def test_seen_scores_do_not_leak_between_evaluator_calls(make_evaluator):
  """Reusing an evaluator still counts the next invocation's samples afresh."""
  evaluator, judge = make_evaluator([_POSITIVE, _POSITIVE, _NEGATIVE])
  invocation = Invocation(
      user_content=types.Content(
          role='user', parts=[types.Part(text='Question')]
      ),
      final_response=types.Content(
          role='model', parts=[types.Part(text='Answer')]
      ),
  )
  first = await evaluator.evaluate_invocations([invocation])
  assert first.overall_score == 0.5
  judge.responses = [_NEGATIVE, _NEGATIVE, _POSITIVE]
  second = await evaluator.evaluate_invocations([invocation])
  assert second.overall_score == 0.0
  assert second.overall_eval_status == EvalStatus.FAILED
  assert judge.calls == 6

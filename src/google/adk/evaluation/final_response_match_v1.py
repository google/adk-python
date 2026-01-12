# Copyright 2025 Google LLC
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

from __future__ import annotations

from typing import Optional

from google.genai import types as genai_types
from typing_extensions import override

from ..dependencies.rouge_scorer import rouge_scorer
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult
from .text_utils import normalize_text #importing normalize_text function for non-English text comparison


class RougeEvaluator(Evaluator):
  """Evaluates if agent's final response matches a golden/expected final response using Rouge_1 metric.

  Value range for this metric is [0,1], with values closer to 1 more desirable.
  """

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]] = None,
      conversation_scenario: Optional[ConversationScenario] = None,
  ) -> EvaluationResult:
    if expected_invocations is None:
      raise ValueError("expected_invocations is required for this metric.")
    del conversation_scenario  # not used by this metric.

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      reference = _get_text_from_content(expected.final_response)
      response = _get_text_from_content(actual.final_response)
      rouge_1_scores = _calculate_rouge_1_scores(response, reference)
      score = rouge_1_scores.fmeasure
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=_get_eval_status(score, self._eval_metric.threshold),
          )
      )
      total_score += score
      num_invocations += 1

    if per_invocation_results:
      overall_score = total_score / num_invocations
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=_get_eval_status(
              overall_score, self._eval_metric.threshold
          ),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()


def _get_text_from_content(content: Optional[genai_types.Content]) -> str:
  if content and content.parts:
    return "\n".join([part.text for part in content.parts if part.text])

  return ""


def _get_eval_status(score: float, threshold: float):
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


def _calculate_rouge_1_scores(candidate: str, reference: str):
  """Calculates the ROUGE-1 score between a candidate and reference text.

  ROUGE-1 measures the overlap of unigrams (single words) between the
  candidate and reference texts. The score is broken down into:
  - Precision: The proportion of unigrams in the candidate that are also in the
  reference.
  - Recall: The proportion of unigrams in the reference that are also in the
  candidate.
  - F-measure: The harmonic mean of precision and recall.

  Args:
      candidate: The generated text to be evaluated.
      reference: The ground-truth text to compare against.

  Returns:
      A dictionary containing the ROUGE-1 precision, recall, and f-measure.
  """
  # Normalize both texts before scoring to handle Unicode variations
  normalized_candidate = normalize_text(candidate)
  normalized_reference = normalize_text(reference)

  # Check if the text contains spaces (word-separated languages)
  has_spaces = ' ' in normalized_reference or ' ' in normalized_candidate
  
  if has_spaces:
    # Use standard word-level ROUGE for space-separated languages
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(normalized_reference, normalized_candidate)
    return scores["rouge1"]
  else:
    # For non-space-separated languages, use character-level comparison
    return _calculate_character_level_rouge(normalized_candidate, normalized_reference)


def _calculate_character_level_rouge(candidate: str, reference: str):
  """Calculates character-level ROUGE-1 score for non-space-separated text.
  
  Args:
    candidate: The candidate text (already normalized).
    reference: The reference text (already normalized).
  
  Returns:
    A Score namedtuple with precision, recall, and fmeasure.
  """
  from collections import Counter, namedtuple
  
  if not reference or not candidate:
    Score = namedtuple('Score', ['precision', 'recall', 'fmeasure'])
    return Score(precision=0.0, recall=0.0, fmeasure=0.0)
  
  # Count character occurrences
  ref_chars = Counter(reference)
  cand_chars = Counter(candidate)
  
  # Calculate overlapping characters
  overlap = sum((ref_chars & cand_chars).values())
  
  # Calculate precision and recall
  precision = overlap / len(candidate) if len(candidate) > 0 else 0.0
  recall = overlap / len(reference) if len(reference) > 0 else 0.0
  
  # Calculate F-measure
  if precision + recall > 0:
    fmeasure = 2 * (precision * recall) / (precision + recall)
  else:
    fmeasure = 0.0
  
  Score = namedtuple('Score', ['precision', 'recall', 'fmeasure'])
  return Score(precision=precision, recall=recall, fmeasure=fmeasure)
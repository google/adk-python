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

from __future__ import annotations

import unicodedata
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


def _is_latin_script(text: str) -> bool:
  """Checks if text is primarily Latin script.

  This is used to determine whether to apply English-specific stemming.
  Latin script includes English, Portuguese, Spanish, French, German, etc.
  Non-Latin scripts include Thai, Chinese, Arabic, Japanese, Korean, etc.

  Args:
      text: The text to analyze.

  Returns:
      True if the text is primarily Latin script, False otherwise.
  """
  if not text:
    return True

  latin_chars = 0
  letter_chars = 0

  for char in text:
    # Check if character is a letter (category starts with 'L')
    if unicodedata.category(char).startswith("L"):
      letter_chars += 1
      # Check if it's a Latin character by looking at its Unicode name
      char_name = unicodedata.name(char, "")
      if "LATIN" in char_name:
        latin_chars += 1

  # If no letters found, default to Latin (likely punctuation/numbers only)
  if letter_chars == 0:
    return True

  # Consider text as Latin if more than 50% of letters are Latin
  return latin_chars / letter_chars > 0.5


def _calculate_rouge_1_scores(candidate: str, reference: str):
  """Calculates the ROUGE-1 score between a candidate and reference text.

  ROUGE-1 measures the overlap of unigrams (single words) between the
  candidate and reference texts. The score is broken down into:
  - Precision: The proportion of unigrams in the candidate that are also in the
  reference.
  - Recall: The proportion of unigrams in the reference that are also in the
  candidate.
  - F-measure: The harmonic mean of precision and recall.

  Stemming is only applied for Latin script text (English, Portuguese, etc.)
  since the Porter stemmer only works correctly for English. For non-Latin
  scripts (Thai, Chinese, Arabic, etc.), stemming is disabled to ensure
  accurate matching.

  Args:
      candidate: The generated text to be evaluated.
      reference: The ground-truth text to compare against.

  Returns:
      A dictionary containing the ROUGE-1 precision, recall, and f-measure.
  """
  # Use stemmer only for Latin script text (English, Portuguese, Spanish, etc.)
  # Porter stemmer doesn't work for non-Latin scripts (Thai, Chinese, Arabic)
  use_stemmer = _is_latin_script(candidate) and _is_latin_script(reference)
  scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=use_stemmer)

  # The score method returns a dictionary where keys are the ROUGE types
  # and values are Score objects (tuples) with precision, recall, and fmeasure.
  scores = scorer.score(reference, candidate)

  return scores["rouge1"]

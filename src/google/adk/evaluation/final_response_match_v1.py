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

import re
from typing import Optional

from google.genai import types as genai_types
from typing_extensions import override

from ..dependencies.rouge_scorer import rouge_scorer
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import _validate_invocation_lengths
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
    _validate_invocation_lengths(actual_invocations, expected_invocations)
    del conversation_scenario  # not used by this metric.

    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(
        actual_invocations, expected_invocations, strict=True
    ):
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


def _get_eval_status(score: float, threshold: float) -> EvalStatus:
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


def _contains_non_ascii(text: str) -> bool:
  """Returns True if the text contains any non-ASCII characters."""
  return bool(re.search(r"[^\x00-\x7F]", text))


class _UnicodeTokenizer:
  """A tokenizer that handles non-ASCII text by splitting on whitespace and
  decomposing non-ASCII tokens into individual characters."""

  def tokenize(self, text: str) -> list[str]:
    tokens = []
    for word in text.lower().split():
      if _contains_non_ascii(word):
        # For non-ASCII words (e.g. Thai, Chinese, Arabic), treat each
        # Unicode character as a separate token so that ROUGE overlap is
        # computed at the character level instead of being discarded entirely.
        tokens.extend(list(word))
      else:
        # Keep ASCII tokens as-is (drop purely punctuation-only tokens).
        cleaned = re.sub(r"[^a-z0-9]", "", word)
        if cleaned:
          tokens.append(cleaned)
    return tokens


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
  # Use a Unicode-aware tokenizer when either text contains non-ASCII
  # characters (e.g. Thai, Chinese, Arabic) so that the default ASCII-only
  # tokenizer does not strip every token and return a spurious score of 0.
  if _contains_non_ascii(candidate) or _contains_non_ascii(reference):
    tokenizer = _UnicodeTokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizer)
  else:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

  # The score method returns a dictionary where keys are the ROUGE types
  # and values are Score objects (tuples) with precision, recall, and fmeasure.
  scores = scorer.score(reference, candidate)

  return scores["rouge1"]

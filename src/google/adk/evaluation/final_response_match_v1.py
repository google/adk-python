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

from typing import Any
from typing import Optional
import unicodedata

from google.genai import types as genai_types
from typing_extensions import override

from ..dependencies.rouge_scorer import rouge_scorer
from ..dependencies.rouge_scorer import tokenizers
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

    threshold = self._eval_metric.threshold
    assert threshold is not None

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
              eval_status=_get_eval_status(score, threshold),
          )
      )
      total_score += score
      num_invocations += 1

    if per_invocation_results:
      overall_score = total_score / num_invocations
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=_get_eval_status(overall_score, threshold),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()


def _get_text_from_content(content: Optional[genai_types.Content]) -> str:
  if content and content.parts:
    return "\n".join([part.text for part in content.parts if part.text])

  return ""


def _get_eval_status(score: float, threshold: float) -> EvalStatus:
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


def _is_cjk(char: str) -> bool:
  """Checks if a character belongs to CJK (Chinese, Japanese, Korean) scripts."""
  code = ord(char)
  return (
      0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
      or 0x3040 <= code <= 0x309F  # Hiragana
      or 0x30A0 <= code <= 0x30FF  # Katakana
      or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
  )


def _is_non_spaced_script(char: str) -> bool:
  """Checks if a character belongs to non-spaced scripts like Thai, Lao, Khmer, Myanmar."""
  code = ord(char)
  return (
      0x0E00 <= code <= 0x0E7F  # Thai
      or 0x0E80 <= code <= 0x0EFF  # Lao
      or 0x1780 <= code <= 0x17FF  # Khmer
      or 0x1000 <= code <= 0x109F  # Myanmar
  )


def _is_word_char(char: str) -> bool:
  # Combining marks (e.g. Thai vowel signs, Devanagari matras) are not
  # alphanumeric on their own but must stay attached to their base character.
  return char.isalnum() or unicodedata.category(char).startswith("M")


class _UnicodeAwareTokenizer:
  """Tokenizer that keeps non-ASCII word characters and splits CJK/non-spaced characters.

  The default rouge_score tokenizer discards any character outside [a-z0-9],
  so text in non-Latin scripts (e.g. Thai, Chinese, Arabic) tokenizes to
  nothing and always scores 0. This tokenizer keeps Unicode word characters,
  normalizes Unicode variants (NFKC), splits non-spaced CJK characters at the
  character level, bundles non-spaced scripts by grapheme clusters (base
  consonant + attached combining marks), and delegates ASCII tokens to the
  default tokenizer.

  Note:
      Languages written without spaces (e.g. Thai, Chinese, Japanese) are
      tokenized at the character or grapheme cluster level rather than at true
      word granularity, since dictionary-based word segmentation requires heavy
      external NLP dependencies. As a result, ROUGE-1 unigram overlap for these
      languages operates on character/grapheme cluster units rather than full words.
  """

  def __init__(self, use_stemmer: bool = False):
    self._default_tokenizer = tokenizers.DefaultTokenizer(use_stemmer)

  def tokenize(self, text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    processed_chars: list[str] = []
    for char in text:
      if _is_cjk(char):
        processed_chars.extend([" ", char, " "])
      elif _is_non_spaced_script(char):
        if unicodedata.category(char).startswith("M"):
          # Combining mark (vowel/tone mark): attach directly to previous base consonant!
          processed_chars.append(char)
        else:
          # Base consonant: start a new grapheme cluster boundary by prepending a space!
          processed_chars.extend([" ", char])
      elif _is_word_char(char):
        processed_chars.append(char)
      else:
        processed_chars.append(" ")
    words = "".join(processed_chars).split()
    tokens: list[str] = []
    for word in words:
      if word.isascii():
        tokens.extend(self._default_tokenizer.tokenize(word))
      else:
        tokens.append(word)
    return tokens


def _calculate_rouge_1_scores(candidate: str, reference: str) -> Any:
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
  scorer = rouge_scorer.RougeScorer(
      ["rouge1"], tokenizer=_UnicodeAwareTokenizer(use_stemmer=True)
  )

  # The score method returns a dictionary where keys are the ROUGE types
  # and values are Score objects (tuples) with precision, recall, and fmeasure.
  scores = scorer.score(reference, candidate)

  return scores["rouge1"]

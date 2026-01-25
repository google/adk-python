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

import logging
import re
from typing import ClassVar
from typing import List
from typing import Optional

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ..dependencies.rouge_scorer import rouge_scorer
from ..dependencies.rouge_scorer import tokenizers
from .eval_case import ConversationScenario
from .eval_case import Invocation
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .eval_metrics import RougeScoreCriterion
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult

logger = logging.getLogger("google_adk." + __name__)


# =============================================================================
# CJK Character Ranges
# =============================================================================
# Each range is defined separately for maintainability.
# Order: Han (Chinese/Japanese/Korean) -> Japanese Kana -> Korean Hangul

CJK_RANGES = (
    "\u4e00-\u9fff"  # CJK Unified Ideographs (Han)
    "\u3400-\u4dbf"  # CJK Extension A (Han)
    "\u3040-\u309f"  # Hiragana (Japanese)
    "\u30a0-\u30ff"  # Katakana (Japanese)
    "\uac00-\ud7af"  # Hangul Syllables (Korean)
)

# CJK Symbols and Punctuation block (U+3000-U+303F)
# Includes: 。、！？「」『』【】〈〉《》〔〕 etc.
# Note: Fullwidth forms (U+FF00-U+FFEF) are NOT included here.
CJK_PUNCTUATION = "\u3000-\u303f"

CJK_CHAR_PATTERN = re.compile(f"[{CJK_RANGES}]")
CJK_PUNCT_PATTERN = re.compile(f"[{CJK_PUNCTUATION}]")


# Regex pattern for tokenization: matches CJK characters or ASCII alphanumeric words
_CJK_TOKEN_PATTERN = re.compile(f"[{CJK_RANGES}]|[a-z0-9]+")


def _contains_cjk(text: str) -> bool:
  """Check if text contains any CJK characters."""
  return bool(CJK_CHAR_PATTERN.search(text)) if text else False


class CJKTokenizer(tokenizers.Tokenizer):
  """Character-based tokenizer for CJK + ASCII alphanumeric mixed text.

  This tokenizer is designed for evaluating text in CJK languages
  (Chinese, Japanese, Korean) where the default ROUGE tokenizer fails
  because it only recognizes ASCII alphanumeric characters.

  Tokenization strategy:
      - CJK characters: Each character becomes one token
      - ASCII alphanumeric (a-z, 0-9): Word-based tokenization
      - CJK punctuation/symbols (U+3000-U+303F): Removed
      - All other characters: Skipped (not tokenized)

  Limitations:
      - Fullwidth alphanumeric (Ａ-Ｚ, ０-９): Skipped
      - Greek, Cyrillic, accented Latin: Skipped
      - This is NOT a general multilingual tokenizer

  For morphological analysis, consider language-specific tokenizers
  (e.g., MeCab for Japanese).

  Note: Stemming is not applicable to CJK and is always disabled.
  """

  def tokenize(self, text: Optional[str]) -> List[str]:
    """Tokenize text with CJK-aware segmentation.

    Args:
        text: Input text to tokenize. None or empty string returns [].

    Returns:
        List of tokens. CJK characters are individual tokens,
        ASCII words are single tokens.
    """
    if not text:
      return []

    text = text.lower()
    text = CJK_PUNCT_PATTERN.sub(" ", text)
    return _CJK_TOKEN_PATTERN.findall(text)


class RougeEvaluator(Evaluator):
  """Evaluates using Rouge_1 metric with optional CJK support.

  Value range for this metric is [0,1], with values closer to 1 more desirable.

  Warning behavior:
      When CJK characters are detected but no tokenizer is specified,
      a warning is logged. This warning is logged at most ONCE per
      RougeEvaluator instance, even if evaluate_invocations() is called
      multiple times.
  """

  criterion_type: ClassVar[type[BaseCriterion]] = RougeScoreCriterion

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric
    # Warning is logged at most once per instance
    self._warned_about_cjk = False

    tokenizer: Optional[tokenizers.Tokenizer] = None
    use_stemmer = True

    if eval_metric.criterion:
      try:
        criterion = RougeScoreCriterion.model_validate(
            eval_metric.criterion.model_dump()
        )
        if criterion.tokenizer == "cjk":
          tokenizer = CJKTokenizer()
          use_stemmer = False  # Stemming not applicable to CJK
      except ValidationError:
        pass  # Different criterion type, ignore

    # Create scorer once for reuse across invocations (performance optimization)
    if tokenizer:
      self._scorer = rouge_scorer.RougeScorer(
          ["rouge1"], use_stemmer=False, tokenizer=tokenizer
      )
      self._has_cjk_tokenizer = True
    else:
      self._scorer = rouge_scorer.RougeScorer(
          ["rouge1"], use_stemmer=use_stemmer
      )
      self._has_cjk_tokenizer = False

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

      # Log warning once if CJK detected without tokenizer
      self._maybe_warn_cjk(reference, response)

      # Use pre-created scorer for performance
      scores = self._scorer.score(reference, response)
      score = scores["rouge1"].fmeasure
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

  def _maybe_warn_cjk(self, reference: str, response: str) -> None:
    """Log warning if CJK detected without tokenizer (once per instance)."""
    if self._warned_about_cjk:
      return
    if self._has_cjk_tokenizer:
      return
    if _contains_cjk(reference) or _contains_cjk(response):
      logger.warning(
          "CJK characters detected in text but no tokenizer specified. "
          "ROUGE scores will likely be 0.0 for CJK text. "
          "Consider using RougeScoreCriterion(tokenizer='cjk') for "
          "Chinese, Japanese, or Korean language support."
      )
      self._warned_about_cjk = True


def _get_text_from_content(content: Optional[genai_types.Content]) -> str:
  if content and content.parts:
    return "\n".join([part.text for part in content.parts if part.text])

  return ""


def _get_eval_status(score: float, threshold: float):
  return EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED


def _calculate_rouge_1_scores(
    candidate: str,
    reference: str,
    tokenizer: Optional[tokenizers.Tokenizer] = None,
    use_stemmer: bool = True,
):
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
      tokenizer: Custom tokenizer (e.g., CJKTokenizer). None for default.
      use_stemmer: Whether to use Porter stemmer. Ignored if tokenizer is set.

  Returns:
      A Score object containing the ROUGE-1 precision, recall, and f-measure.
  """
  if tokenizer:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1"], use_stemmer=False, tokenizer=tokenizer
    )
  else:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=use_stemmer)

  # The score method returns a dictionary where keys are the ROUGE types
  # and values are Score objects (tuples) with precision, recall, and fmeasure.
  scores = scorer.score(reference, candidate)

  return scores["rouge1"]

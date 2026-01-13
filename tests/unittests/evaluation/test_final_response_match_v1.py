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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.genai import types as genai_types
import pytest


def _create_test_rouge_evaluator(threshold: float) -> RougeEvaluator:
  return RougeEvaluator(
      EvalMetric(metric_name="response_match_score", threshold=threshold)
  )


def _create_test_invocations(
    candidate: str, reference: str
) -> tuple[Invocation, Invocation]:
  """Returns tuple of (actual_invocation, expected_invocation)."""
  return Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=candidate)]
      ),
  ), Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=reference)]
      ),
  )


def test_calculate_rouge_1_scores_empty_candidate_and_reference():
  candidate = ""
  reference = ""
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores_empty_candidate():
  candidate = ""
  reference = "This is a test reference."
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores_empty_reference():
  candidate = "This is a test candidate response."
  reference = ""
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == 0
  assert rouge_1_score.recall == 0
  assert rouge_1_score.fmeasure == 0


def test_calculate_rouge_1_scores():
  candidate = "This is a test candidate response."
  reference = "This is a test reference."
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == pytest.approx(2 / 3)
  assert rouge_1_score.recall == pytest.approx(4 / 5)
  assert rouge_1_score.fmeasure == pytest.approx(8 / 11)


@pytest.mark.parametrize(
    "candidates, references, expected_score, expected_status",
    [
        (
            ["The quick brown fox jumps.", "hello world"],
            ["The quick brown fox jumps over the lazy dog.", "hello"],
            0.69048,  # (5/7 + 2/3) / 2
            EvalStatus.FAILED,
        ),
        (
            ["This is a test.", "Another test case."],
            ["This is a test.", "This is a different test."],
            0.625,  # (1 + 1/4) / 2
            EvalStatus.FAILED,
        ),
        (
            ["No matching words here.", "Second candidate."],
            ["Completely different text.", "Another reference."],
            0.0,  # (0 + 1/2) / 2
            EvalStatus.FAILED,
        ),
        (
            ["Same words", "Same words"],
            ["Same words", "Same words"],
            1.0,
            EvalStatus.PASSED,
        ),
    ],
)
def test_rouge_evaluator_multiple_invocations(
    candidates: list[str],
    references: list[str],
    expected_score: float,
    expected_status: EvalStatus,
):
  rouge_evaluator = _create_test_rouge_evaluator(threshold=0.8)
  actual_invocations = []
  expected_invocations = []
  for candidate, reference in zip(candidates, references):
    actual_invocation, expected_invocation = _create_test_invocations(
        candidate, reference
    )
    actual_invocations.append(actual_invocation)
    expected_invocations.append(expected_invocation)

  evaluation_result = rouge_evaluator.evaluate_invocations(
      actual_invocations, expected_invocations
  )
  assert evaluation_result.overall_score == pytest.approx(
      expected_score, rel=1e-3
  )
  assert evaluation_result.overall_eval_status == expected_status


# =============================================================================
# CJK Tokenizer Tests (Issue #4122)
# =============================================================================

import logging

from google.adk.evaluation.eval_metrics import RougeScoreCriterion
from google.adk.evaluation.final_response_match_v1 import _contains_cjk
from google.adk.evaluation.final_response_match_v1 import CJKTokenizer


class TestCJKTokenizer:
  """Tests for CJKTokenizer tokenization behavior."""

  def test_tokenize_japanese(self):
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("これはテスト")
    assert tokens == ["こ", "れ", "は", "テ", "ス", "ト"]

  def test_tokenize_english(self):
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("This is a test")
    assert tokens == ["this", "is", "a", "test"]

  def test_tokenize_mixed_cjk_and_ascii(self):
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("Hello世界World")
    assert tokens == ["hello", "世", "界", "world"]

  def test_tokenize_fullwidth_alphanumeric_skipped(self):
    """Fullwidth alphanumeric should be skipped."""
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("ＡＢＣ１２３")
    assert tokens == []

  def test_tokenize_greek_skipped(self):
    """Greek and other non-CJK scripts should be skipped."""
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("αβγtest")
    assert tokens == ["test"]

  def test_tokenize_empty_string(self):
    tokenizer = CJKTokenizer()
    assert tokenizer.tokenize("") == []

  def test_tokenize_none(self):
    """None input should return empty list."""
    tokenizer = CJKTokenizer()
    assert tokenizer.tokenize(None) == []

  def test_tokenize_chinese(self):
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("这是测试")
    assert tokens == ["这", "是", "测", "试"]

  def test_tokenize_korean(self):
    tokenizer = CJKTokenizer()
    tokens = tokenizer.tokenize("테스트")
    assert len(tokens) == 3  # 3 Hangul syllables


class TestContainsCJK:
  """Tests for _contains_cjk helper function."""

  def test_contains_cjk_japanese(self):
    assert _contains_cjk("これはテスト") is True

  def test_contains_cjk_english(self):
    assert _contains_cjk("This is a test") is False

  def test_contains_cjk_mixed(self):
    assert _contains_cjk("Hello世界") is True

  def test_contains_cjk_empty(self):
    assert _contains_cjk("") is False

  def test_contains_cjk_none(self):
    assert _contains_cjk(None) is False


class TestRougeScoreWithCJKTokenizer:
  """Tests for ROUGE score calculation with CJK tokenizer."""

  def test_english_identical_default_tokenizer(self):
    """English identical text should score 1.0 with default tokenizer."""
    result = self._evaluate("This is a test", "This is a test", None)
    assert result.overall_score == pytest.approx(1.0)

  def test_english_partial_default_tokenizer(self):
    """English partial match should score between 0 and 1."""
    result = self._evaluate("This is test", "This is a test", None)
    assert 0 < result.overall_score < 1

  def test_japanese_without_tokenizer_scores_zero(self):
    """Japanese text without CJK tokenizer should score 0.0."""
    result = self._evaluate("これはテスト", "これはテスト", None)
    assert result.overall_score == pytest.approx(0.0)

  def test_japanese_identical_with_cjk_tokenizer(self):
    """Japanese identical text with CJK tokenizer should score 1.0."""
    result = self._evaluate("これはテスト", "これはテスト", "cjk")
    assert result.overall_score == pytest.approx(1.0)

  def test_japanese_partial_with_cjk_tokenizer(self):
    """Japanese partial match should score between 0 and 1."""
    result = self._evaluate("これはテスト", "これはサンプル", "cjk")
    assert 0 < result.overall_score < 1

  def test_chinese_identical_with_cjk_tokenizer(self):
    """Chinese identical text with CJK tokenizer should score 1.0."""
    result = self._evaluate("这是测试", "这是测试", "cjk")
    assert result.overall_score == pytest.approx(1.0)

  def test_mixed_text_identical_with_cjk_tokenizer(self):
    """Mixed CJK+ASCII identical text should score 1.0."""
    result = self._evaluate("Hello世界", "Hello世界", "cjk")
    assert result.overall_score == pytest.approx(1.0)

  def test_cjk_punctuation_does_not_affect_score(self):
    """CJK punctuation should be removed, not affecting score."""
    result_with = self._evaluate("これはテスト。", "これはテスト", "cjk")
    result_without = self._evaluate("これはテスト", "これはテスト", "cjk")
    assert result_with.overall_score == pytest.approx(1.0)
    assert result_without.overall_score == pytest.approx(1.0)

  def _evaluate(self, candidate: str, reference: str, tokenizer_type: str):
    """Helper to evaluate ROUGE score."""
    criterion = None
    if tokenizer_type:
      criterion = RougeScoreCriterion(threshold=0.8, tokenizer=tokenizer_type)

    eval_metric = EvalMetric(
        metric_name="response_match_score",
        threshold=0.8,
        criterion=criterion,
    )
    evaluator = RougeEvaluator(eval_metric=eval_metric)

    actual, expected = _create_test_invocations(candidate, reference)

    return evaluator.evaluate_invocations([actual], [expected])


class TestCJKWarning:
  """Tests for CJK detection warning behavior."""

  def test_warning_logged_once_for_multiple_evaluations(self, caplog):
    """Warning should be logged exactly once per evaluator instance."""
    eval_metric = EvalMetric(
        metric_name="response_match_score",
        threshold=0.8,
    )
    evaluator = RougeEvaluator(eval_metric=eval_metric)

    actual1, expected1 = _create_test_invocations(
        "これはテスト", "これはテスト"
    )
    actual2, expected2 = _create_test_invocations("別のテスト", "別のテスト")

    with caplog.at_level(logging.WARNING):
      # First evaluation with CJK - should trigger warning
      evaluator.evaluate_invocations([actual1], [expected1])
      # Second evaluation with CJK - should NOT trigger warning
      evaluator.evaluate_invocations([actual2], [expected2])

    cjk_warnings = [r for r in caplog.records if "CJK" in r.message]
    assert len(cjk_warnings) == 1

  def test_no_warning_when_cjk_tokenizer_specified(self, caplog):
    """No warning when CJK tokenizer is properly specified."""
    criterion = RougeScoreCriterion(threshold=0.8, tokenizer="cjk")
    eval_metric = EvalMetric(
        metric_name="response_match_score",
        threshold=0.8,
        criterion=criterion,
    )
    evaluator = RougeEvaluator(eval_metric=eval_metric)

    actual, expected = _create_test_invocations("これはテスト", "これはテスト")

    with caplog.at_level(logging.WARNING):
      evaluator.evaluate_invocations([actual], [expected])

    cjk_warnings = [r for r in caplog.records if "CJK" in r.message]
    assert len(cjk_warnings) == 0

  def test_no_warning_for_english_text(self, caplog):
    """No warning for ASCII-only text."""
    eval_metric = EvalMetric(
        metric_name="response_match_score",
        threshold=0.8,
    )
    evaluator = RougeEvaluator(eval_metric=eval_metric)

    actual, expected = _create_test_invocations(
        "This is a test", "This is a test"
    )

    with caplog.at_level(logging.WARNING):
      evaluator.evaluate_invocations([actual], [expected])

    cjk_warnings = [r for r in caplog.records if "CJK" in r.message]
    assert len(cjk_warnings) == 0

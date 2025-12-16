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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
from google.adk.evaluation.final_response_match_v1 import _is_latin_script
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



def test_get_metric_info():
  """Test get_metric_info function for response match metric."""
  metric_info = RougeEvaluator.get_metric_info()
  assert metric_info.metric_name == PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
  assert metric_info.metric_value_info.interval.min_value == 0.0
  assert metric_info.metric_value_info.interval.max_value == 1.0


# Tests for _is_latin_script function
class TestIsLatinScript:
  """Tests for the _is_latin_script helper function."""

  def test_empty_string(self):
    """Empty string should default to Latin."""
    assert _is_latin_script("") is True

  def test_english_text(self):
    """English text should be detected as Latin."""
    assert _is_latin_script("Hello world") is True
    assert _is_latin_script("The quick brown fox") is True

  def test_portuguese_text(self):
    """Portuguese with accents should be detected as Latin."""
    assert _is_latin_script("Olá, como você está?") is True
    assert _is_latin_script("São Paulo é uma cidade") is True

  def test_french_text(self):
    """French with accents should be detected as Latin."""
    assert _is_latin_script("Bonjour, comment allez-vous?") is True
    assert _is_latin_script("français café résumé") is True

  def test_german_text(self):
    """German with umlauts should be detected as Latin."""
    assert _is_latin_script("Guten Tag, wie geht es Ihnen?") is True
    assert _is_latin_script("Größe Übung Äpfel") is True

  def test_thai_text(self):
    """Thai text should be detected as non-Latin."""
    assert _is_latin_script("สวัสดี") is False
    assert _is_latin_script("สวัสดีครับ") is False

  def test_chinese_text(self):
    """Chinese text should be detected as non-Latin."""
    assert _is_latin_script("你好") is False
    assert _is_latin_script("中文测试") is False

  def test_arabic_text(self):
    """Arabic text should be detected as non-Latin."""
    assert _is_latin_script("مرحبا") is False
    assert _is_latin_script("اللغة العربية") is False

  def test_japanese_text(self):
    """Japanese text should be detected as non-Latin."""
    assert _is_latin_script("こんにちは") is False
    assert _is_latin_script("日本語テスト") is False

  def test_korean_text(self):
    """Korean text should be detected as non-Latin."""
    assert _is_latin_script("안녕하세요") is False
    assert _is_latin_script("한국어 테스트") is False

  def test_numbers_only(self):
    """Numbers only should default to Latin."""
    assert _is_latin_script("12345") is True

  def test_punctuation_only(self):
    """Punctuation only should default to Latin."""
    assert _is_latin_script("!@#$%") is True

  def test_mixed_latin_dominant(self):
    """Mixed text with Latin dominant should be Latin."""
    assert _is_latin_script("Hello 你好 world test") is True

  def test_mixed_non_latin_dominant(self):
    """Mixed text with non-Latin dominant should be non-Latin."""
    assert _is_latin_script("你好世界 Hi") is False


# Tests for non-English language ROUGE scoring
class TestNonEnglishRougeScoring:
  """Tests for ROUGE scoring with non-English languages (Issue #3111).

  These tests verify that the fix for non-English languages works correctly.
  The key issue was that Porter stemmer only works for English, causing
  match failures for other languages.
  """

  # === Thai Language Tests (Original Issue #3111) ===

  def test_thai_greeting_identical(self):
    """Thai: Identical greeting should have perfect score."""
    # This is the exact case from Issue #3111
    candidate = "สวัสดี"
    reference = "สวัสดี"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_thai_sentence_with_overlap(self):
    """Thai: Sentences with common words should show partial match."""
    # "Hello, how are you today?" vs "Hello, how is the weather?"
    candidate = "สวัสดี คุณ สบายดี ไหม วันนี้"
    reference = "สวัสดี คุณ อากาศ เป็น อย่างไร"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "สวัสดี" and "คุณ" (2 out of 5 words each)
    assert rouge_1_score.fmeasure > 0
    assert rouge_1_score.fmeasure < 1.0

  def test_thai_polite_particle_variation(self):
    """Thai: Same meaning with polite particle should show high match."""
    # "Hello" vs "Hello (polite)"
    candidate = "สวัสดี ครับ"
    reference = "สวัสดี ค่ะ"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "สวัสดี" (1 out of 2 words)
    assert rouge_1_score.fmeasure == pytest.approx(0.5, rel=0.1)

  # === Chinese Language Tests ===

  def test_chinese_greeting_identical(self):
    """Chinese: Identical greeting should have perfect score."""
    candidate = "你好世界"
    reference = "你好世界"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_chinese_sentence_with_overlap(self):
    """Chinese: Sentences with common words should show partial match."""
    # Space-separated for tokenization
    candidate = "今天 天气 很好"  # "Today's weather is good"
    reference = "今天 我 很 开心"  # "Today I am happy"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "今天" and "很"
    assert rouge_1_score.fmeasure > 0
    assert rouge_1_score.fmeasure < 1.0

  def test_chinese_different_sentences(self):
    """Chinese: Completely different sentences should have zero score."""
    candidate = "苹果 橙子 香蕉"  # "Apple orange banana"
    reference = "汽车 飞机 火车"  # "Car airplane train"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 0

  # === Arabic Language Tests ===

  def test_arabic_greeting_identical(self):
    """Arabic: Identical greeting should have perfect score."""
    candidate = "مرحبا بالعالم"
    reference = "مرحبا بالعالم"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_arabic_sentence_with_overlap(self):
    """Arabic: Sentences with common words should show partial match."""
    candidate = "أنا أحب القراءة والكتابة"  # "I love reading and writing"
    reference = "أنا أحب السفر والموسيقى"  # "I love travel and music"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "أنا" and "أحب"
    assert rouge_1_score.fmeasure > 0
    assert rouge_1_score.fmeasure < 1.0

  # === Japanese Language Tests ===

  def test_japanese_greeting_identical(self):
    """Japanese: Identical greeting should have perfect score."""
    candidate = "こんにちは"
    reference = "こんにちは"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_japanese_sentence_with_overlap(self):
    """Japanese: Sentences with common words should show partial match."""
    candidate = "今日 は 天気 が いい です"  # "Today the weather is good"
    reference = "今日 は 仕事 が 忙しい です"  # "Today work is busy"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "今日", "は", "が", "です"
    assert rouge_1_score.fmeasure > 0.5

  # === Korean Language Tests ===

  def test_korean_greeting_identical(self):
    """Korean: Identical greeting should have perfect score."""
    candidate = "안녕하세요"
    reference = "안녕하세요"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_korean_sentence_with_overlap(self):
    """Korean: Sentences with common words should show partial match."""
    candidate = "오늘 날씨가 좋습니다"  # "Today's weather is good"
    reference = "오늘 기분이 좋습니다"  # "Today my mood is good"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "오늘" and "좋습니다"
    assert rouge_1_score.fmeasure > 0
    assert rouge_1_score.fmeasure < 1.0

  # === European Languages (Latin script with accents) ===

  def test_portuguese_sentence_identical(self):
    """Portuguese: Identical sentence with accents should match perfectly."""
    candidate = "Olá, como você está hoje?"
    reference = "Olá, como você está hoje?"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  def test_portuguese_sentence_with_overlap(self):
    """Portuguese: Sentences with common words should show partial match."""
    candidate = "Eu gosto de programação e música"
    reference = "Eu gosto de viajar e cozinhar"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "Eu", "gosto", "de", "e"
    assert rouge_1_score.fmeasure > 0.5

  def test_french_sentence_with_accents(self):
    """French: Accented characters should match correctly."""
    candidate = "Où est la bibliothèque s'il vous plaît?"
    reference = "Où est la gare s'il vous plaît?"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match most words except "bibliothèque" vs "gare"
    assert rouge_1_score.fmeasure > 0.7

  def test_german_sentence_with_umlauts(self):
    """German: Umlauts should be handled correctly."""
    candidate = "Ich möchte ein Brötchen und Käse"
    reference = "Ich möchte ein Brötchen und Wurst"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match everything except "Käse" vs "Wurst"
    assert rouge_1_score.fmeasure > 0.8

  def test_spanish_sentence_with_accents(self):
    """Spanish: Accented characters should match correctly."""
    candidate = "¿Cómo estás? Estoy muy bien gracias"
    reference = "¿Cómo estás? Estoy cansado hoy"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "¿Cómo", "estás?", "Estoy"
    assert rouge_1_score.fmeasure > 0.4

  # === English Stemming Verification ===

  def test_english_stemming_running_vs_run(self):
    """English: Stemming should normalize 'running' to 'run'."""
    candidate = "I am running fast"
    reference = "I am run fast"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # With stemming: "running" -> "run", perfect match
    assert rouge_1_score.fmeasure == 1.0

  def test_english_stemming_multiple_forms(self):
    """English: Multiple word forms should match via stemming."""
    candidate = "The dogs are running and jumping happily"
    reference = "The dog is run and jump happy"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Stemming normalizes: dogs->dog, running->run, jumping->jump, happily->happi
    # Should have high overlap
    assert rouge_1_score.fmeasure > 0.7

  def test_english_preserves_exact_matching(self):
    """English: Exact matches should still work perfectly."""
    candidate = "The quick brown fox jumps over the lazy dog"
    reference = "The quick brown fox jumps over the lazy dog"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    assert rouge_1_score.fmeasure == 1.0

  # === Mixed Script Edge Cases ===

  def test_mixed_english_chinese(self):
    """Mixed: English and Chinese in same text."""
    candidate = "Hello 世界 welcome to Python"
    reference = "Hello 世界 welcome to Java"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "Hello", "世界", "welcome", "to" (4 out of 5)
    assert rouge_1_score.fmeasure > 0.7

  def test_mixed_with_numbers(self):
    """Mixed: Text with numbers should work correctly."""
    candidate = "订单号 12345 已确认"
    reference = "订单号 12345 已发货"
    rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
    # Should match "订单号" and "12345"
    assert rouge_1_score.fmeasure > 0.5

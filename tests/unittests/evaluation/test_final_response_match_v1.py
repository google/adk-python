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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
from google.adk.evaluation.final_response_match_v1 import _is_cjk
from google.adk.evaluation.final_response_match_v1 import _is_non_spaced_script
from google.adk.evaluation.final_response_match_v1 import _is_word_char
from google.adk.evaluation.final_response_match_v1 import _UnicodeAwareTokenizer
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.genai import types as genai_types
import pytest
from rouge_score import tokenizers


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
    "text",
    [
        "สวัสดี",  # Thai
        "你好世界",  # Chinese
        "مرحبا بالعالم",  # Arabic
        "こんにちは",  # Japanese
        "Здравствуйте",  # Russian
    ],
)
def test_calculate_rouge_1_scores_identical_non_english_text(text: str):
  rouge_1_score = _calculate_rouge_1_scores(text, text)
  assert rouge_1_score.precision == pytest.approx(1)
  assert rouge_1_score.recall == pytest.approx(1)
  assert rouge_1_score.fmeasure == pytest.approx(1)


def test_calculate_rouge_1_scores_different_non_english_text():
  candidate = "мир привет"
  reference = "привет только"
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  assert rouge_1_score.precision == pytest.approx(1 / 2)
  assert rouge_1_score.recall == pytest.approx(1 / 2)
  assert rouge_1_score.fmeasure == pytest.approx(1 / 2)


def test_calculate_rouge_1_scores_cjk_partial_overlap_and_inversion():
  candidate = "天气很好今天"
  reference = "今天天气很好"
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  # Character-level matching: 6/6 characters overlap in unigram space.
  assert rouge_1_score.precision == pytest.approx(1.0)
  assert rouge_1_score.recall == pytest.approx(1.0)
  assert rouge_1_score.fmeasure == pytest.approx(1.0)


def test_calculate_rouge_1_scores_mixed_language_text():
  candidate = "hello สวัสดี"
  reference = "hello world"
  rouge_1_score = _calculate_rouge_1_scores(candidate, reference)
  # Candidate tokens: ['hello', 'สั', 'ส', 'ด', 'ดี'] (5 tokens).
  # Reference tokens: ['hello', 'world'] (2 tokens).
  assert rouge_1_score.precision == pytest.approx(1 / 5)
  assert rouge_1_score.recall == pytest.approx(1 / 2)
  assert rouge_1_score.fmeasure == pytest.approx(2 / 7)


def test_unicode_aware_tokenizer_combining_marks_category_m():
  """Tests that combining marks (category 'M', e.g. Thai vowel signs) stay attached to base characters."""
  tokenizer = _UnicodeAwareTokenizer()

  # Thai word "ดี" (Consonant 'ด' + Combining Mark Vowel ' ี' [category Mn]).
  # Verifies that category 'M' combining marks hit the startswith("M") branch and attach to 'ด'.
  # Extracting mark from "ดี"[1] ensures clean visual rendering without font overlap.
  thai_vowel_mark = "ดี"[1]
  assert unicodedata.category(thai_vowel_mark).startswith("M")

  tokens = tokenizer.tokenize("ดี")
  assert len(tokens) == 1
  assert tokens[0] == "ดี"

  # Hindi / Devanagari word "नमस्ते" (contains combining mark matras).
  tokens_hindi = tokenizer.tokenize("नमस्ते")
  assert len(tokens_hindi) == 1
  assert tokens_hindi[0] == "नमस्ते"


@pytest.mark.parametrize(
    "input_text, use_stemmer, expected_tokens",
    [
        # Mixed English + Thai (with stemmer)
        ("hello สวัสดี", True, ["hello", "ส", "วั", "ส", "ดี"]),
        # Branch 1a: CJK Hanzi
        ("中文测试", False, ["中", "文", "测", "试"]),
        ("今天天气很好", False, ["今", "天", "天", "气", "很", "好"]),
        # Branch 1b: CJK Hiragana
        ("ひらがな", False, ["ひ", "ら", "が", "な"]),
        ("こんにちは", False, ["こ", "ん", "に", "ち", "は"]),
        # Branch 1c: CJK Katakana
        ("カタカナ", False, ["カ", "タ", "カ", "ナ"]),
        # Branch 1d: CJK Hangul
        ("한글", False, ["한", "글"]),
        # Branch 2a: Non-spaced script (Thai consonant + combining mark M)
        ("ดี", False, ["ดี"]),
        ("ฉันรักคุณมาก", False, ["ฉั", "น", "รั", "ก", "คุ", "ณ", "ม", "า", "ก"]),
        # Branch 2b: Non-spaced script (Lao)
        ("ດີ", False, ["ດີ"]),
        # Branch 2c: Non-spaced script (Khmer)
        ("ល្អ", False, ["ល្", "អ"]),
        # Branch 2d: Non-spaced script (Myanmar)
        ("မင်္ဂလာ", False, ["မ", "င်္", "ဂ", "လာ"]),
        # Branch 3a: Alphanumeric ASCII (with and without stemmer)
        ("Running jumped 123", True, ["run", "jump", "123"]),
        ("Running jumped 123", False, ["running", "jumped", "123"]),
        # Branch 3b & 3c: Non-ASCII spaced script with combining mark M (Arabic Harakat & Hindi Matra)
        ("مَرْحَبًا", False, ["مَرْحَبًا"]),
        ("नमस्ते", False, ["नमस्ते"]),
        ("Hello World! Привет мир", True, ["hello", "world", "привет", "мир"]),
        # Branch 4: Punctuation and non-word symbols (triggers else: append(" "))
        ("hello, world! @123 #test", True, ["hello", "world", "123", "test"]),
    ],
)
def test_unicode_aware_tokenizer_all_branches_coverage(
    input_text: str, use_stemmer: bool, expected_tokens: list[str]
):
  """Verifies 100% branch coverage for all script types, combining marks, stemmer flag, and punctuation handling."""
  tokenizer = _UnicodeAwareTokenizer(use_stemmer=use_stemmer)
  assert tokenizer.tokenize(input_text) == expected_tokens


@pytest.mark.parametrize(
    "char, expected",
    [
        ("中", True),  # Hanzi
        ("ぁ", True),  # Hiragana
        ("ァ", True),  # Katakana
        ("한", True),  # Hangul
        ("a", False),
        ("1", False),
        ("ส", False),
    ],
)
def test_is_cjk(char: str, expected: bool):
  """Tests _is_cjk helper for Chinese, Hiragana, Katakana, and Hangul boundaries."""
  assert _is_cjk(char) == expected


@pytest.mark.parametrize(
    "char, expected",
    [
        ("ส", True),  # Thai
        ("ກ", True),  # Lao
        ("ក", True),  # Khmer
        ("က", True),  # Myanmar
        ("中", False),
        ("a", False),
    ],
)
def test_is_non_spaced_script(char: str, expected: bool):
  """Tests _is_non_spaced_script helper for Thai, Lao, Khmer, and Myanmar boundaries."""
  assert _is_non_spaced_script(char) == expected


@pytest.mark.parametrize(
    "char, expected",
    [
        ("a", True),
        ("9", True),
        ("中", True),
        ("ส", True),
        ("ดี"[1], True),  # Combining Mark Category Mn (Thai Vowel)
        (" ", False),
        ("!", False),
    ],
)
def test_is_word_char(char: str, expected: bool):
  """Tests _is_word_char helper for alphanumerics and combining marks."""
  assert _is_word_char(char) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "Testing stemmed words like running and jumped, don't split!",
        "Numbers 123 and mixed a1b2 tokens under_scored.",
        "",
    ],
)
def test_unicode_aware_tokenizer_matches_default_tokenizer_for_ascii(
    text: str,
):
  default_tokens = tokenizers.DefaultTokenizer(use_stemmer=True).tokenize(text)
  unicode_tokens = _UnicodeAwareTokenizer(use_stemmer=True).tokenize(text)
  assert unicode_tokens == default_tokens


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
        (
            ["สวัสดี", "你好"],
            ["สวัสดี", "你好"],
            1.0,
            EvalStatus.PASSED,
        ),
        (
            ["今天天气不错", "我想吃炒饭"],
            ["今天天气很好", "我想吃面条"],
            0.63333,  # (2/3 + 3/5) / 2
            EvalStatus.FAILED,
        ),
        (
            ["สวัสดีครับ", "ฉันชอบกินข้าวผัด"],
            ["สวัสดีค่ะ", "ฉันชอบกินก๋วยเตี๋ยว"],
            0.61538,  # (8/13 + 8/13) / 2
            EvalStatus.FAILED,
        ),
        (
            ["你好世界", "人工智能"],
            ["再见", "机器学习"],
            0.0,
            EvalStatus.FAILED,
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


@pytest.mark.parametrize(
    "actual_count, expected_count",
    [
        pytest.param(2, 1, id="extra-actual-turn"),
        pytest.param(1, 2, id="missing-actual-turn"),
    ],
)
def test_rouge_evaluator_rejects_mismatched_invocation_lengths(
    actual_count: int, expected_count: int
):
  actual, expected = _create_test_invocations("same", "same")
  rouge_evaluator = _create_test_rouge_evaluator(threshold=0.8)

  with pytest.raises(
      ValueError,
      match=f"same length; got {actual_count} and {expected_count}",
  ):
    rouge_evaluator.evaluate_invocations(
        [actual] * actual_count, [expected] * expected_count
    )

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

"""Tests for final_response_match_v1."""

from __future__ import annotations
import pytest

def test_normalization_applied_in_rouge():
    """Normalization should make identical Thai strings match."""
    from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
    from google.adk.evaluation.text_utils import normalize_text

    reference = "สวัสดี"
    candidate = "สวัสดี"

    # Verify normalization directly
    assert normalize_text(reference) == normalize_text(candidate)

    # Verify ROUGE score reflects a perfect match
    score = _calculate_rouge_1_scores(candidate, reference)

    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)
    assert score.fmeasure == pytest.approx(1.0)
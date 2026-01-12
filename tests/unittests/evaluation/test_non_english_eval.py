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


def test_debug_normalization():
  """Debug test to see if normalization is being applied."""
  from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
  from google.adk.evaluation.text_utils import normalize_text
  
  reference = "สวัสดี"
  candidate = "สวัสดี"
  
  # Check normalization directly
  norm_ref = normalize_text(reference)
  norm_cand = normalize_text(candidate)
  
  print(f"Reference: {repr(reference)}")
  print(f"Candidate: {repr(candidate)}")
  print(f"Normalized reference: {repr(norm_ref)}")
  print(f"Normalized candidate: {repr(norm_cand)}")
  print(f"Are they equal after normalization? {norm_ref == norm_cand}")
  
  # Now test the actual function
  score = _calculate_rouge_1_scores(candidate, reference)
  print(f"ROUGE score: {score}")
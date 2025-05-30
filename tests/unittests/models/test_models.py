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

from google.adk import models
from google.adk.models.anthropic_llm import Claude
from google.adk.models.google_llm import Gemini
from google.adk.models.registry import LLMRegistry
import pytest


@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-1.5-flash',
        'gemini-1.5-flash-001',
        'gemini-1.5-flash-002',
        'gemini-1.5-pro',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
        'gemini-2.0-flash-exp',
        'projects/123456/locations/us-central1/endpoints/123456',  # finetuned vertex gemini endpoint
        'projects/123456/locations/us-central1/publishers/google/models/gemini-2.0-flash-exp',  # vertex gemini long name
    ],
)
def test_match_gemini_family(model_name):
  assert models.LLMRegistry.resolve(model_name) is Gemini


@pytest.mark.parametrize(
    'model_name',
    [
        'claude-3-5-haiku@20241022',
        'claude-3-5-sonnet-v2@20241022',
        'claude-3-5-sonnet@20240620',
        'claude-3-haiku@20240307',
        'claude-3-opus@20240229',
        'claude-3-sonnet@20240229',
        'claude-sonnet-4@20250514',
        'claude-opus-4@20250514',
    ],
)
def test_match_claude_family(model_name):
  LLMRegistry.register(Claude)

  assert models.LLMRegistry.resolve(model_name) is Claude


def test_non_exist_model():
  with pytest.raises(ValueError) as e_info:
    models.LLMRegistry.resolve('non-exist-model')
  assert 'Model non-exist-model not found.' in str(e_info.value)

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

from google.adk import models
from google.adk.models.anthropic_llm import Claude
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
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
  """Test that Gemini models are resolved correctly."""
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
  """Test that Claude models are resolved correctly."""
  assert models.LLMRegistry.resolve(model_name) is Claude


@pytest.mark.parametrize(
    'model_name',
    [
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'groq/llama3-70b-8192',
        'groq/mixtral-8x7b-32768',
        'anthropic/claude-3-opus-20240229',
        'anthropic/claude-3-5-sonnet-20241022',
    ],
)
def test_match_litellm_family(model_name):
  """Test that LiteLLM models are resolved correctly."""
  assert models.LLMRegistry.resolve(model_name) is LiteLlm


def test_non_exist_model():
  with pytest.raises(ValueError) as e_info:
    models.LLMRegistry.resolve('non-exist-model')
  assert 'Model non-exist-model not found.' in str(e_info.value)


def test_helpful_error_for_claude_without_extensions():
  """Test that missing Claude models show helpful install instructions.

  Note: This test may pass even when anthropic IS installed, because it
  only checks the error message format when a model is not found.
  """
  # Use a non-existent Claude model variant to trigger error
  with pytest.raises(ValueError) as e_info:
    models.LLMRegistry.resolve('claude-nonexistent-model-xyz')

  error_msg = str(e_info.value)
  # The error should mention anthropic package and installation instructions
  # These checks work whether or not anthropic is actually installed
  assert 'Model claude-nonexistent-model-xyz not found' in error_msg
  assert 'anthropic package' in error_msg
  assert 'pip install' in error_msg


def test_helpful_error_for_litellm_without_extensions():
  """Test that missing LiteLLM models show helpful install instructions.

  Note: This test may pass even when litellm IS installed, because it
  only checks the error message format when a model is not found.
  """
  # Use a non-existent provider to trigger error
  with pytest.raises(ValueError) as e_info:
    models.LLMRegistry.resolve('unknown-provider/gpt-4o')

  error_msg = str(e_info.value)
  # The error should mention litellm package for provider-style models
  assert 'Model unknown-provider/gpt-4o not found' in error_msg
  assert 'litellm package' in error_msg
  assert 'pip install' in error_msg
  assert 'Provider-style models' in error_msg

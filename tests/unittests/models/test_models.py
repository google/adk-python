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

from unittest import mock

from google.adk import models
from google.adk.labs.openai._openai_llm import OpenAILlm
from google.adk.models import registry
from google.adk.models.anthropic_llm import Claude
from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
import pytest


@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-1.5-pro',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
        'gemini-2.5-flash',
        'projects/123456/locations/us-central1/endpoints/123456',  # finetuned vertex gemini endpoint
        'projects/123456/locations/us-central1/publishers/google/models/gemini-2.5-flash',  # vertex gemini long name
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
        'claude-sonnet-4-5',
        'claude-haiku-4-5',
        'claude-sonnet-4-6',
        'claude-opus-4-6',
        'claude-opus-4-7',
        'claude-opus-4-8',
        'claude-opus-5',
        'claude-sonnet-5',
        'claude-fable-5',
        'claude-opus-5@default',
        'claude-sonnet-5@default',
        # Anthropic put the generation first at the 4 launch, so these ids
        # matched nothing even though claude-opus-4 did.
        'claude-4-opus-20250514',
        'claude-4-sonnet-20250514',
        # Carries no generation at all.
        'claude-mythos-preview',
        # A generation nobody has added a pattern for yet.
        'claude-opus-6',
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


@pytest.mark.parametrize(
    'model_name',
    [
        'xai/grok-4',
        'gemini/gemini-3.5-flash',
        'openrouter/anthropic/claude-opus-4',
        'cerebras/llama-3.3-70b',
    ],
)
def test_match_litellm_provider_not_spelled_out_in_registry(model_name):
  """Test that any provider LiteLLM knows about resolves to LiteLlm."""
  assert models.LLMRegistry.resolve(model_name) is LiteLlm


@pytest.mark.parametrize(
    'model_name',
    [
        'apigee/gemini-2.5-flash',
        'apigee/v1/gemini-2.5-flash',
        'apigee/vertex_ai/v1beta/gemini-2.5-flash',
    ],
)
def test_match_apigee_family(model_name):
  """Test that Apigee models are resolved correctly."""
  assert models.LLMRegistry.resolve(model_name) is ApigeeLlm


@pytest.mark.parametrize(
    'model_name',
    [
        'o1-preview',
        'o3-mini',
        'o4-mini',
    ],
)
def test_match_openai_reasoning_family(model_name):
  """Test that the OpenAI o-series resolves regardless of generation."""
  assert models.LLMRegistry.resolve(model_name) is OpenAILlm


def test_non_exist_model():
  with pytest.raises(ValueError) as e_info:
    models.LLMRegistry.resolve('non-exist-model')
  assert 'Model non-exist-model not found.' in str(e_info.value)


def test_helpful_error_for_claude_without_extensions():
  """Test that Claude models show install instructions when anthropic is absent.

  This is now the only way to reach that message. Any claude id resolves once
  the package is installed, which is what makes the message true when it does
  appear -- it used to fire for a mistyped id on a machine that had anthropic
  installed the whole time.
  """
  # Point the lazy entry at a module that cannot be imported, which is what an
  # uninstalled anthropic looks like to the registry.
  with mock.patch.dict(
      registry._llm_registry_dict,
      {r'claude-.*': ('google.adk.models.not_installed', 'Claude')},
  ):
    registry.LLMRegistry.resolve.cache_clear()
    with pytest.raises(ValueError) as e_info:
      models.LLMRegistry.resolve('claude-opus-5')
  registry.LLMRegistry.resolve.cache_clear()

  error_msg = str(e_info.value)
  assert 'Model claude-opus-5 not found' in error_msg
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


def test_resolve_with_prefix():
  """Test that model resolution can be overridden with a prefix."""
  assert models.LLMRegistry.resolve('gemini:gemini-1.5-flash') is Gemini
  assert models.LLMRegistry.resolve('Claude:claude-3-opus@20240229') is Claude
  assert models.LLMRegistry.resolve('lite:openai/gpt-4o') is LiteLlm
  assert models.LLMRegistry.resolve('LiteLlm:openai/gpt-4o') is LiteLlm


def test_register_after_resolve_returns_the_new_class():
  """Test that registering over an already-resolved name takes effect."""
  model_name = 'test-registry-override-model'

  class FirstLlm(BaseLlm):

    @classmethod
    def supported_models(cls):
      return [model_name]

  class SecondLlm(BaseLlm):

    @classmethod
    def supported_models(cls):
      return [model_name]

  try:
    models.LLMRegistry.register(FirstLlm)
    assert models.LLMRegistry.resolve(model_name) is FirstLlm

    models.LLMRegistry.register(SecondLlm)
    assert models.LLMRegistry.resolve(model_name) is SecondLlm
  finally:
    registry._llm_registry_dict.pop(model_name, None)
    models.LLMRegistry.resolve.cache_clear()


def test_new_llm_with_prefix(mocker):
  """Test that new_llm strips prefix when creating instance if it matches class."""
  mock_class = mocker.MagicMock()
  mock_class.__name__ = 'MockLlm'
  mocker.patch.object(models.LLMRegistry, 'resolve', return_value=mock_class)

  models.LLMRegistry.new_llm('mock:gpt-4')
  mock_class.assert_called_once_with(model='gpt-4')

  mock_class.reset_mock()
  models.LLMRegistry.new_llm('MockLlm:gpt-4')
  mock_class.assert_called_once_with(model='gpt-4')


def test_new_llm_with_non_matching_prefix(mocker):
  """Test that new_llm keeps prefix if it does not match class."""
  mock_class = mocker.MagicMock()
  mock_class.__name__ = 'MockLlm'
  mocker.patch.object(models.LLMRegistry, 'resolve', return_value=mock_class)

  models.LLMRegistry.new_llm('custom:gpt-4')
  mock_class.assert_called_once_with(model='custom:gpt-4')

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

"""Defines the interface to support a model."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse
from .registry import LLMRegistry

if TYPE_CHECKING:
  from google.adk.labs.openai import OpenAILlm

  from .anthropic_llm import AnthropicGenerateContentConfig
  from .anthropic_llm import Claude
  from .apigee_llm import ApigeeLlm
  from .gemma_llm import Gemma
  from .gemma_llm import Gemma3Ollama
  from .google_llm import Gemini
  from .lite_llm import LiteLlm

__all__ = [
    'AnthropicGenerateContentConfig',
    'ApigeeLlm',
    'BaseLlm',
    'Claude',
    'Gemini',
    'Gemma',
    'Gemma3Ollama',
    'LLMRegistry',
    'LiteLlm',
]

_LAZY_PROVIDERS: dict[str, tuple[list[str], str]] = {
    'Gemini': (
        [
            r'gemini-.*',
            # Gemma 4+ uses Gemini natively; must precede Gemma's gemma-.* so
            # gemma-4-* resolves to Gemini, not the Gemma 3 workaround class.
            r'gemma-4.*',
            r'model-optimizer-.*',
            r'projects\/.+\/locations\/.+\/endpoints\/.+',
            r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
        ],
        'google_llm',
    ),
    # Gemma 3 only (function-calling workarounds). Gemma 4+ resolves to Gemini.
    'Gemma': ([r'gemma-.*'], 'gemma_llm'),
    'ApigeeLlm': ([r'.*-apigee$'], 'apigee_llm'),
    'Claude': ([r'claude-3-.*', r'claude-.*-4.*'], 'anthropic_llm'),
    'Gemma3Ollama': ([r'ollama/gemma3.*'], 'gemma_llm'),
    'OpenAILlm': (
        [r'gpt-.*', r'o1-.*', r'o3-.*'],
        'google.adk.labs.openai',
    ),
    'LiteLlm': (
        [
            r'openai/.*',
            r'azure/.*',
            r'azure_ai/.*',
            r'groq/.*',
            r'anthropic/.*',
            r'bedrock/.*',
            r'ollama/(?!gemma3).*',
            r'ollama_chat/.*',
            r'together_ai/.*',
            r'vertex_ai/.*',
            r'mistral/.*',
            r'deepseek/.*',
            r'fireworks_ai/.*',
            r'cohere/.*',
            r'databricks/.*',
            r'ai21/.*',
        ],
        'lite_llm',
    ),
}

for _name, (_patterns, _module) in _LAZY_PROVIDERS.items():
  _target_module = (
      _module if _module.startswith('google.adk.') else f'{__name__}.{_module}'
  )
  LLMRegistry._register_lazy(_patterns, _target_module, _name)


_OTHER_LAZY_IMPORTS: dict[str, str] = {
    'AnthropicGenerateContentConfig': 'anthropic_llm',
}


def __getattr__(name: str):
  if name in _LAZY_PROVIDERS:
    module_name = _LAZY_PROVIDERS[name][1]
  elif name in _OTHER_LAZY_IMPORTS:
    module_name = _OTHER_LAZY_IMPORTS[name]
  else:
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

  try:
    if module_name.startswith('google.adk.'):
      module = importlib.import_module(module_name)
    else:
      module = importlib.import_module(f'{__name__}.{module_name}')
  except ImportError as e:
    raise ImportError(
        f'`{name}` requires an optional dependency that is not installed.'
        ' Install with: pip install google-adk[extensions]'
    ) from e
  return getattr(module, name)

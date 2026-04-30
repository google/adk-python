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
  from .anthropic_llm import Claude
  from .apigee_llm import ApigeeLlm
  from .gemma_llm import Gemma
  from .gemma_llm import Gemma3Ollama
  from .google_llm import Gemini
  from .lite_llm import LiteLlm

__all__ = [
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
            r'model-optimizer-.*',
            r'projects\/.+\/locations\/.+\/endpoints\/.+',
            r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
        ],
        'google_llm',
    ),
    'Gemma': ([r'gemma-.*'], 'gemma_llm'),
    'ApigeeLlm': ([r'.*-apigee$'], 'apigee_llm'),
    'Claude': ([r'claude-3-.*', r'claude-.*-4.*'], 'anthropic_llm'),
    'Gemma3Ollama': ([r'ollama/gemma3.*'], 'gemma_llm'),
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
  LLMRegistry._register_lazy(_patterns, f'{__name__}.{_module}', _name)


def __getattr__(name: str):
  if name in _LAZY_PROVIDERS:
    module_name = _LAZY_PROVIDERS[name][1]
    try:
      module = importlib.import_module(f'{__name__}.{module_name}')
    except ImportError as e:
      raise ImportError(
          f'`{name}` requires an optional dependency that is not installed.'
          ' Install with: pip install google-adk[extensions]'
      ) from e
    return getattr(module, name)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

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

"""Defines the interface to support a model."""

from .apigee_llm import ApigeeLlm
from .base_llm import BaseLlm
from .gemma_llm import Gemma
from .google_llm import Gemini
from .llm_request import LlmRequest
from .llm_response import LlmResponse
from .registry import LLMRegistry

__all__ = [
    'BaseLlm',
    'Gemini',
    'Gemma',
    'LLMRegistry',
]


LLMRegistry.register(Gemini)
LLMRegistry.register(Gemma)
LLMRegistry.register(ApigeeLlm)

# Optionally register Claude if anthropic package is installed
try:
  from .anthropic_llm import Claude

  LLMRegistry.register(Claude)
  __all__.append('Claude')
except Exception:
  # Claude support requires: pip install google-adk[extensions]
  pass

# Optionally register LiteLlm if litellm package is installed
try:
  from .lite_llm import LiteLlm

  LLMRegistry.register(LiteLlm)
  __all__.append('LiteLlm')
except Exception:
  # LiteLLM support requires: pip install google-adk[extensions]
  pass

# Optionally register Gemma3Ollama if litellm package is installed
try:
  from .gemma_llm import Gemma3Ollama

  LLMRegistry.register(Gemma3Ollama)
  __all__.append('Gemma3Ollama')
except Exception:
  # Gemma3Ollama requires LiteLLM: pip install google-adk[extensions]
  pass

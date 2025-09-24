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

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Optional

from google.genai import types
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)


class KerasLlm(BaseLlm):
  """Wrapper around KerasHub for local model inference.

  This wrapper can be used with any of the models supported by KerasHub. The
  models run locally without requiring external API calls. Generation parameters
  are passed directly to the model's generate() method.

  Example usage:
  ```
  agent = Agent(
      model=KerasLlm(
          model="gpt2_base_en",
          max_length=100,
          temperature=0.7,
          top_k=50,
          top_p=0.9
      ),
      ...
  )
  ```

  Attributes:
      model: The name of the KerasHub model preset.
      _keras_model: The loaded KerasHub model instance.
      _additional_args: Additional generation parameters passed to generate().
  """

  _keras_model: Optional[Any] = None
  """The loaded KerasHub model instance."""

  _additional_args: Dict[str, Any] = None
  """Additional generation parameters."""

  def __init__(self, model: str, **kwargs):
    """Initializes the KerasLlm class.

    Args:
        model: The name of the KerasHub model preset.
        **kwargs: Additional arguments to pass to the model generation.
    """
    super().__init__(model=model, **kwargs)
    self._additional_args = kwargs
    # Remove internal fields from kwargs
    self._additional_args.pop("_keras_model", None)
    self._additional_args.pop("_additional_args", None)

  def _load_model(self):
    """Loads the KerasHub model and preprocessor."""
    try:
      import keras_hub
    except ImportError:
      raise ImportError(
          "KerasHub is not installed. Please install it with: pip install"
          " keras-hub"
      )

    # Strip 'keras/' prefix if present
    preset = self.model
    if preset.startswith("keras/"):
      preset = preset[6:]

    try:
      # Load the model using the appropriate model class
      # For causal language models, use CausalLM.from_preset
      from keras_hub.models import CausalLM

      # Optional: allow skipping weight download for quick tests
      load_weights = self._additional_args.get("load_weights", True)
      self._keras_model = CausalLM.from_preset(preset, load_weights=load_weights)
    except Exception as e:
      raise ValueError(
          f"Failed to load model '{self.model}'. "
          "This might be due to an unsupported preset or network issues. "
          f"Error: {e}"
      )

  def _flatten_conversation_to_prompt(self, llm_request: LlmRequest) -> str:
    """Flattens the conversation into a single text prompt.

    Args:
        llm_request: The LLM request containing conversation history.

    Returns:
        A single text prompt string.
    """
    prompt_parts = []

    for content in llm_request.contents:
      if content.role == "system":
        # Add system instruction at the beginning
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"System: {part.text}")
      elif content.role == "user":
        # Add user message
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"User: {part.text}")
      elif content.role == "assistant":
        # Add assistant response
        for part in content.parts:
          if hasattr(part, "text") and part.text:
            prompt_parts.append(f"Assistant: {part.text}")

    # Add the final "Assistant:" to prompt for completion
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content using the KerasHub model."""
    self._maybe_append_user_content(llm_request)

    if self._keras_model is None:
      self._load_model()

    prompt = self._flatten_conversation_to_prompt(llm_request)
    
    # Extract generation parameters from additional_args
    generation_params = {}
    
    # Basic parameters that most models support
    if "max_length" in self._additional_args:
      generation_params["max_length"] = self._additional_args["max_length"]
    else:
      generation_params["max_length"] = 100  # Default
    
    # Only pass parameters that the model actually supports
    # Different models support different parameters, so we'll be conservative
    supported_params = ["max_length", "temperature", "top_k", "top_p"]
    
    for param in supported_params:
      if param in self._additional_args:
        generation_params[param] = self._additional_args[param]

    def generate_text():
      return self._keras_model.generate(prompt, strip_prompt=True, **generation_params)

    try:
      generated_text = await asyncio.to_thread(generate_text)
    except Exception as e:
      logger.error(f"Error generating text with KerasHub: {e}")
      raise RuntimeError(f"Failed to generate text with KerasHub model: {e}")

    # Process the generated text
    if self._additional_args.get("strip_prompt", False):
      # Remove the original prompt from the response
      if prompt in generated_text:
        generated_text = generated_text[len(prompt):].strip()
      # Also try to remove common prefixes that might be left
      common_prefixes = ["Assistant:", "User:", "System:"]
      for prefix in common_prefixes:
        if generated_text.startswith(prefix):
          generated_text = generated_text[len(prefix):].strip()
    
    # Clean up repetitive patterns
    generated_text = self._clean_repetitive_text(generated_text)

    # Create response content
    response_content = types.Content(
        role="assistant", parts=[types.Part.from_text(text=generated_text)]
    )

    response = LlmResponse(content=response_content)
    yield response

  def _clean_repetitive_text(self, text: str) -> str:
    """Clean up repetitive patterns in generated text."""
    if not text:
      return text
    
    # Split into words and detect repetitive patterns
    words = text.split()
    if len(words) < 3:
      return text
    
    # Look for immediate repetitions (same word repeated 3+ times)
    cleaned_words = []
    i = 0
    while i < len(words):
      word = words[i]
      count = 1
      j = i + 1
      while j < len(words) and words[j] == word:
        count += 1
        j += 1
      
      if count >= 3:
        # Keep only 2 instances of repeated words
        cleaned_words.extend([word, word])
        i = j
      else:
        cleaned_words.append(word)
        i += 1
    
    return " ".join(cleaned_words)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Returns a list of supported models in regex for LlmRegistry.
    
    Based on KerasHub CausalLM presets available in the library.
    These patterns cover the most common model families.
    """
    return [
        # KerasHub prefix for explicit local model selection
        "keras/.*",
        
        # GPT Family
        "gpt2_.*_en",
        
        # OPT Family
        "opt_.*_en",
        
        # BLOOM Family
        "bloom_.*_multi",
        "bloomz_.*_multi",
        
        # LLaMA Family
        "llama2_.*_en",
        "llama3_.*_en",
        "llama3\\.1_.*",
        "llama3\\.2_.*",
        "vicuna_.*_en",
        
        # Gemma Family
        "gemma_.*_en",
        "gemma2_.*_en",
        "gemma3_.*",
        "shieldgemma_.*_en",
        "code_gemma_.*_en",
        
        # Modern Models
        "falcon_.*_en",
        "mistral_.*_en",
        "mixtral_.*_en",
        "qwen_.*_en",
        "qwen1\\.5_.*_en",
        "qwen2\\.5_.*_en",
        "phi3_.*_en",
        
        # Vision-Language Models
        "pali_gemma_.*",
        "pali_gemma2_.*",
        
        # Audio Models
        "moonshine_.*_en",
        
        # BART Family
        "bart_.*_en",
        
        # Generic patterns
        ".*_en",  # English models
        ".*_multi",  # Multilingual models
    ]

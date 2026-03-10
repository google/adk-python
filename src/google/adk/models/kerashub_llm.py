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

"""KerasHub LLM integration for ADK.

This module provides a generic integration with KerasHub CausalLM models,
allowing any locally-run model to be used within the ADK agent framework.

Supported model families include Gemma, Llama, Mistral, GPT-2, OPT, Falcon,
and any other model available as a keras_hub.models.CausalLM preset.

Requires: pip install keras keras-hub

For Kaggle model access, set environment variables:
  KAGGLE_USERNAME=<your_username>
  KAGGLE_KEY=<your_api_key>
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import cached_property
from typing import Any, AsyncGenerator, Optional

from google.genai import types
from pydantic import Field
from typing_extensions import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

# --------------------------------------------------------------------------- #
# Prompt formatting helpers
# --------------------------------------------------------------------------- #

_END_MARKERS = ['<end_of_turn>', '</s>', '<|endoftext|>', '<|im_end|>']


def _extract_text_from_content(content: types.Content) -> str:
  """Extracts text from a Content, serialising function_call/response parts."""
  if not content.parts:
    return ''
  texts = []
  for part in content.parts:
    if part.text:
      texts.append(part.text)
    elif part.function_call:
      texts.append(json.dumps({
          'name': part.function_call.name,
          'parameters': part.function_call.args,
      }))
    elif part.function_response:
      texts.append(
          f'Result of {part.function_response.name}: '
          f'{json.dumps(part.function_response.response)}'
      )
  return ' '.join(texts)


def _contents_to_prompt(
    contents: list[types.Content],
    system_instruction: Optional[str] = None,
    model_id: str = '',
) -> str:
  """Converts ADK Contents to a text prompt string.

  Selects a chat template based on the model preset name.
  Falls back to a generic template for unknown model families.
  """
  model_lower = model_id.lower()

  if 'gemma' in model_lower:
    return _format_gemma_prompt(contents, system_instruction)
  if 'llama' in model_lower:
    return _format_llama_prompt(contents, system_instruction)
  if 'mistral' in model_lower:
    return _format_mistral_prompt(contents, system_instruction)
  return _format_generic_prompt(contents, system_instruction)


def _format_gemma_prompt(
    contents: list[types.Content],
    system_instruction: Optional[str] = None,
) -> str:
  parts = []
  if system_instruction:
    parts.append(f'<start_of_turn>user\n{system_instruction}<end_of_turn>')

  for content in contents:
    text = _extract_text_from_content(content)
    if not text:
      continue
    role = 'model' if content.role == 'model' else 'user'
    parts.append(f'<start_of_turn>{role}\n{text}<end_of_turn>')

  parts.append('<start_of_turn>model')
  return '\n'.join(parts)


def _format_llama_prompt(
    contents: list[types.Content],
    system_instruction: Optional[str] = None,
) -> str:
  parts = []
  if system_instruction:
    parts.append(f'[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n')

  for content in contents:
    text = _extract_text_from_content(content)
    if not text:
      continue
    if content.role == 'model':
      parts.append(f'{text} </s>')
    else:
      parts.append(f'[INST] {text} [/INST]')

  return '\n'.join(parts)


def _format_mistral_prompt(
    contents: list[types.Content],
    system_instruction: Optional[str] = None,
) -> str:
  parts = []
  first_user = True
  for content in contents:
    text = _extract_text_from_content(content)
    if not text:
      continue
    if content.role == 'model':
      parts.append(f'{text}</s>')
    else:
      if first_user and system_instruction:
        text = f'{system_instruction}\n\n{text}'
        first_user = False
      parts.append(f'[INST] {text} [/INST]')

  return ' '.join(parts)


def _format_generic_prompt(
    contents: list[types.Content],
    system_instruction: Optional[str] = None,
) -> str:
  parts = []
  if system_instruction:
    parts.append(f'System: {system_instruction}\n')

  for content in contents:
    text = _extract_text_from_content(content)
    if not text:
      continue
    if content.role == 'model':
      parts.append(f'Assistant: {text}')
    else:
      parts.append(f'User: {text}')

  parts.append('Assistant:')
  return '\n'.join(parts)


# --------------------------------------------------------------------------- #
# Tool / function-call helpers (model-agnostic)
# --------------------------------------------------------------------------- #

def _build_tool_system_instruction(
    function_declarations: list[types.FunctionDeclaration],
) -> str:
  """Builds a system instruction that describes available tools."""
  if not function_declarations:
    return ''

  decls_json = ',\n'.join(
      fd.model_dump_json(exclude_none=True) for fd in function_declarations
  )
  return (
      f'You have access to the following functions:\n[{decls_json}\n]\n'
      'When you call a function, you MUST respond in the format of: '
      '{"name": function name, "parameters": dictionary of argument name and its value}\n'
      'When you call a function, you MUST NOT include any other text in the response.\n'
  )


def _inject_tools_as_instructions(llm_request: LlmRequest) -> None:
  """Moves tool declarations into a system instruction prompt.

  Local CausalLM models don't have native tool-calling APIs, so we
  convert tool declarations into text instructions and convert any
  function_call / function_response parts in the conversation history
  to plain text.
  """
  # Convert function_call / function_response parts in history to text
  new_contents: list[types.Content] = []
  for content in llm_request.contents:
    new_parts: list[types.Part] = []
    has_fc, has_fr = False, False
    for part in content.parts:
      if part.function_response:
        has_fr = True
        new_parts.append(types.Part.from_text(
            text=(
                f'Invoking tool `{part.function_response.name}` produced: '
                f'`{json.dumps(part.function_response.response)}`.'
            )
        ))
      elif part.function_call:
        has_fc = True
        new_parts.append(types.Part.from_text(
            text=part.function_call.model_dump_json(exclude_none=True)
        ))
      else:
        new_parts.append(part)

    if has_fr:
      new_contents.append(types.Content(role='user', parts=new_parts))
    elif has_fc:
      new_contents.append(types.Content(role='model', parts=new_parts))
    else:
      new_contents.append(content)

  llm_request.contents = new_contents

  # Move function declarations into system instruction text
  if not llm_request.config.tools:
    return

  all_fds: list[types.FunctionDeclaration] = []
  for tool in llm_request.config.tools:
    if isinstance(tool, types.Tool) and tool.function_declarations:
      all_fds.extend(tool.function_declarations)

  if all_fds:
    llm_request.append_instructions([_build_tool_system_instruction(all_fds)])

  llm_request.config.tools = []


def _try_extract_function_call(response: LlmResponse) -> None:
  """Attempts to parse a function call from the model's text output."""
  if not response.content or not response.content.parts:
    return
  if len(response.content.parts) != 1:
    return
  text = response.content.parts[0].text
  if not text:
    return

  try:
    json_candidate = None

    # Look inside markdown code blocks first
    block_match = re.search(
        r'```(?:json|tool_code)?\s*(.*?)\s*```', text, re.DOTALL
    )
    if block_match:
      json_candidate = block_match.group(1).strip()
    else:
      # Find last valid JSON object in the text
      decoder = json.JSONDecoder()
      start = 0
      while start < len(text):
        try:
          idx = text.index('{', start)
          _, end = decoder.raw_decode(text[idx:])
          json_candidate = text[idx:idx + end]
          start = idx + end
        except (json.JSONDecodeError, ValueError):
          if '{' in text[start:]:
            start = text.index('{', start) + 1
          else:
            break

    if not json_candidate:
      return

    parsed = json.loads(json_candidate)
    name = parsed.get('name') or parsed.get('function')
    params = parsed.get('parameters') or parsed.get('args')
    if name and isinstance(params, dict):
      response.content.parts = [
          types.Part(function_call=types.FunctionCall(name=name, args=params))
      ]
  except (json.JSONDecodeError, KeyError, TypeError):
    pass


# --------------------------------------------------------------------------- #
# KerasHubLlm
# --------------------------------------------------------------------------- #

class KerasHubLlm(BaseLlm):
  """Integration for any KerasHub CausalLM model.

  KerasHub provides access to pretrained CausalLM models (Gemma, Llama,
  Mistral, Falcon, GPT-2, OPT, and more) via Keras 3. Models are
  downloaded from Kaggle and run locally.

  Requires keras and keras-hub packages. For Kaggle access, set
  KAGGLE_USERNAME and KAGGLE_KEY environment variables.

  Example usage with ADK Agent:
    ```python
    from google.adk.agents import Agent
    from google.adk.models import KerasHubLlm

    model = KerasHubLlm(
        model="kerashub://gemma3_instruct_4b",
        keras_backend="jax",
    )
    agent = Agent(name="my_agent", model=model, instruction="You are helpful.")
    ```

  Attributes:
    model: Model identifier in format "kerashub://<preset_name>".
        The preset can be a built-in name (e.g. "gemma3_instruct_4b"),
        a Kaggle handle ("kaggle://keras/gemma3/keras/gemma3_instruct_4b"),
        or a HuggingFace handle ("hf://username/model").
    keras_backend: The Keras backend to use ("jax", "tensorflow", or "torch").
    dtype: Data type for model weights (e.g., "bfloat16", "float32").
    max_length: Maximum sequence length for generation.
    sampler: Sampler configuration. Can be "greedy", "top_k", "top_p",
        or a dict (e.g. {"type": "top_k", "k": 10, "temperature": 0.7}).
  """

  model: str = 'kerashub://gemma3_instruct_4b'

  keras_backend: str = Field(default='jax')
  """The Keras backend to use: 'jax', 'tensorflow', or 'torch'."""

  dtype: str = Field(default='bfloat16')
  """Data type for model weights."""

  max_length: int = Field(default=512)
  """Maximum sequence length for generation."""

  sampler: Any = Field(default='greedy')
  """Sampler for generation. 'greedy', 'top_k', 'top_p', or a dict config."""

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(model="{self.model}")'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return [r'kerashub://.*']

  @cached_property
  def _preset_name(self) -> str:
    """Extracts the preset name from the model identifier."""
    prefix = 'kerashub://'
    if self.model.startswith(prefix):
      return self.model[len(prefix):]
    return self.model

  @cached_property
  def _keras_model(self):
    """Lazily loads the KerasHub CausalLM model."""
    import os
    os.environ['KERAS_BACKEND'] = self.keras_backend

    try:
      import keras_hub
    except ImportError as e:
      msg = str(e)
      if 'jax' in msg:
        hint = 'pip install jax jaxlib'
      elif 'torch' in msg:
        hint = 'pip install torch'
      elif 'tensorflow' in msg:
        hint = 'pip install tensorflow'
      else:
        hint = 'pip install keras keras-hub'
      raise ImportError(
          f'Failed to import keras-hub: {e}. '
          f'Install the required packages with: {hint}'
      ) from e

    logger.info(
        'Loading KerasHub model: %s (backend=%s, dtype=%s)',
        self._preset_name, self.keras_backend, self.dtype,
    )

    causal_lm = keras_hub.models.CausalLM.from_preset(
        self._preset_name, dtype=self.dtype,
    )

    # Configure sampler
    if isinstance(self.sampler, dict):
      sampler_type = self.sampler.get('type', 'greedy')
      if sampler_type == 'top_k':
        sampler_obj = keras_hub.samplers.TopKSampler(
            k=self.sampler.get('k', 10),
            temperature=self.sampler.get('temperature', 1.0),
        )
      elif sampler_type == 'top_p':
        sampler_obj = keras_hub.samplers.TopPSampler(
            p=self.sampler.get('p', 0.9),
            temperature=self.sampler.get('temperature', 1.0),
        )
      else:
        sampler_obj = 'greedy'
      causal_lm.compile(sampler=sampler_obj)
    elif self.sampler != 'greedy':
      causal_lm.compile(sampler=self.sampler)

    logger.info('KerasHub model loaded successfully: %s', self._preset_name)
    return causal_lm

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content using a KerasHub CausalLM model.

    Args:
      llm_request: The request containing conversation contents and config.
      stream: Whether to stream (not supported for KerasHub, ignored).

    Yields:
      LlmResponse with the model's generated text or a function call.
    """
    # Convert tool declarations to text instructions (model-agnostic)
    _inject_tools_as_instructions(llm_request)

    # Ensure there's user content
    self._maybe_append_user_content(llm_request)

    # Extract system instruction
    system_instruction = None
    if llm_request.config and llm_request.config.system_instruction:
      si = llm_request.config.system_instruction
      system_instruction = si if isinstance(si, str) else str(si)

    # Build prompt
    prompt = _contents_to_prompt(
        llm_request.contents,
        system_instruction=system_instruction,
        model_id=self._preset_name,
    )

    # Resolve max_length
    max_length = self.max_length
    if llm_request.config and llm_request.config.max_output_tokens:
      max_length = llm_request.config.max_output_tokens

    logger.debug(
        'KerasHub generating with preset=%s, max_length=%d',
        self._preset_name, max_length,
    )

    try:
      loop = asyncio.get_event_loop()
      raw_output = await loop.run_in_executor(
          None,
          lambda: self._keras_model.generate(prompt, max_length=max_length),
      )

      # Strip input prompt from output
      output_text = str(raw_output)
      if output_text.startswith(prompt):
        output_text = output_text[len(prompt):]
      output_text = output_text.strip()

      # Strip end-of-turn markers
      for marker in _END_MARKERS:
        idx = output_text.find(marker)
        if idx != -1:
          output_text = output_text[:idx]

      response = LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text=output_text)],
          ),
          model_version=self._preset_name,
      )

      # Try to extract a function call from the text
      _try_extract_function_call(response)

      yield response

    except Exception as e:
      logger.error('KerasHub generation failed: %s', e, exc_info=True)
      yield LlmResponse(
          error_code='KERASHUB_ERROR',
          error_message=str(e),
      )

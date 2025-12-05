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

from functools import cached_property
import json
import logging
import re
from typing import Any
from typing import AsyncGenerator

from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai import types
from google.genai.types import Content
from google.genai.types import FunctionDeclaration
from google.genai.types import Part
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from typing_extensions import override

logger = logging.getLogger('google_adk.' + __name__)


class GemmaFunctionCallingMixin:
  """Mixin providing function calling support for Gemma models.

  Gemma models don't have native function calling support, so this mixin
  provides the logic to:
  1. Convert function declarations to system instruction prompts
  2. Convert function call/response parts to text in the conversation
  3. Extract function calls from model text responses
  """

  def _move_function_calls_into_system_instruction(
      self, llm_request: LlmRequest
  ) -> None:
    """Converts function declarations to system instructions for Gemma."""
    # Convert function calls/responses in contents to text
    new_contents: list[Content] = []
    for content_item in llm_request.contents:
      (
          new_parts_for_content,
          has_function_response_part,
          has_function_call_part,
      ) = _convert_content_parts_for_gemma(content_item)

      if has_function_response_part:
        if new_parts_for_content:
          new_contents.append(Content(role='user', parts=new_parts_for_content))
      elif has_function_call_part:
        if new_parts_for_content:
          new_contents.append(
              Content(role='model', parts=new_parts_for_content)
          )
      else:
        new_contents.append(content_item)

    llm_request.contents = new_contents

    if not llm_request.config.tools:
      return

    all_function_declarations: list[FunctionDeclaration] = []
    for tool_item in llm_request.config.tools:
      if isinstance(tool_item, types.Tool) and tool_item.function_declarations:
        all_function_declarations.extend(tool_item.function_declarations)

    if all_function_declarations:
      system_instruction = _build_gemma_function_system_instruction(
          all_function_declarations
      )
      llm_request.append_instructions([system_instruction])

    llm_request.config.tools = []

  def _extract_function_calls_from_response(
      self, llm_response: LlmResponse
  ) -> None:
    """Extracts function calls from Gemma text responses."""
    if llm_response.partial or (llm_response.turn_complete is True):
      return

    if not llm_response.content:
      return

    if not llm_response.content.parts:
      return

    if len(llm_response.content.parts) > 1:
      return

    response_text = llm_response.content.parts[0].text
    if not response_text:
      return

    try:
      json_candidate = None

      markdown_code_block_pattern = re.compile(
          r'```(?:(json|tool_code))?\s*(.*?)\s*```', re.DOTALL
      )
      block_match = markdown_code_block_pattern.search(response_text)

      if block_match:
        json_candidate = block_match.group(2).strip()
      else:
        found, json_text = _get_last_valid_json_substring(response_text)
        if found:
          json_candidate = json_text

      if not json_candidate:
        return

      function_call_parsed = GemmaFunctionCallModel.model_validate_json(
          json_candidate
      )
      function_call = types.FunctionCall(
          name=function_call_parsed.name,
          args=function_call_parsed.parameters,
      )
      function_call_part = Part(function_call=function_call)
      llm_response.content.parts = [function_call_part]
    except (json.JSONDecodeError, ValidationError) as e:
      logger.debug(
          'Error attempting to parse JSON into function call. Leaving as text'
          ' response. %s',
          e,
      )
    except Exception as e:
      logger.warning(
          'Error processing Gemma function call response: %s',
          e,
          exc_info=True,
      )


class GemmaFunctionCallModel(BaseModel):
  """Flexible Pydantic model for parsing inline Gemma function call responses."""

  name: str = Field(validation_alias=AliasChoices('name', 'function'))
  parameters: dict[str, Any] = Field(
      validation_alias=AliasChoices('parameters', 'args')
  )


class Gemma(GemmaFunctionCallingMixin, Gemini):
  """Integration for Gemma models exposed via the Gemini API.

  Only Gemma 3 models are supported at this time. For agentic use cases,
  use of gemma-3-27b-it and gemma-3-12b-it are strongly recommended.

  For full documentation, see: https://ai.google.dev/gemma/docs/core/

  NOTE: Gemma does **NOT** support system instructions. Any system instructions
  will be replaced with an initial *user* prompt in the LLM request. If system
  instructions change over the course of agent execution, the initial content
  **SHOULD** be replaced. Special care is warranted here.
  See:
  https://ai.google.dev/gemma/docs/core/prompt-structure#system-instructions

  NOTE: Gemma's function calling support is limited. It does not have full
  access to the
  same built-in tools as Gemini. It also does not have special API support for
  tools and
  functions. Rather, tools must be passed in via a `user` prompt, and extracted
  from model
  responses based on approximate shape.

  NOTE: Vertex AI API support for Gemma is not currently included. This **ONLY**
  supports
  usage via the Gemini API.
  """

  model: str = (
      'gemma-3-27b-it'  # Others: [gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it]
  )

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}(model="{self.model}")'

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Provides the list of supported models.

    Returns:
    A list of supported models.
    """

    return [
        r'gemma-3.*',
    ]

  @cached_property
  def _api_backend(self) -> GoogleLLMVariant:
    return GoogleLLMVariant.GEMINI_API

  @override
  async def _preprocess_request(self, llm_request: LlmRequest) -> None:
    self._move_function_calls_into_system_instruction(llm_request=llm_request)

    if system_instruction := llm_request.config.system_instruction:
      contents = llm_request.contents
      instruction_content = Content(
          role='user', parts=[Part.from_text(text=system_instruction)]
      )

      # NOTE: if history is preserved, we must include the system instructions ONLY once at the beginning
      # of any chain of contents.
      if contents:
        if contents[0] != instruction_content:
          # only prepend if it hasn't already been done
          llm_request.contents = [instruction_content] + contents

      llm_request.config.system_instruction = None

    return await super()._preprocess_request(llm_request)

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemma model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """
    # print(f'{llm_request=}')
    assert llm_request.model.startswith('gemma-'), (
        f'Requesting a non-Gemma model ({llm_request.model}) with the Gemma LLM'
        ' is not supported.'
    )

    async for response in super().generate_content_async(llm_request, stream):
      self._extract_function_calls_from_response(response)
      yield response


def _convert_content_parts_for_gemma(
    content_item: Content,
) -> tuple[list[Part], bool, bool]:
  """Converts function call/response parts within a content item to text parts.

  Args:
    content_item: The original Content item.

  Returns:
    A tuple containing:
      - A list of new Part objects with function calls/responses converted to text.
      - A boolean indicating if any function response parts were found.
      - A boolean indicating if any function call parts were found.
  """
  new_parts: list[Part] = []
  has_function_response_part = False
  has_function_call_part = False

  for part in content_item.parts:
    if func_response := part.function_response:
      has_function_response_part = True
      response_text = (
          f'Invoking tool `{func_response.name}` produced:'
          f' `{json.dumps(func_response.response)}`.'
      )
      new_parts.append(Part.from_text(text=response_text))
    elif func_call := part.function_call:
      has_function_call_part = True
      new_parts.append(
          Part.from_text(text=func_call.model_dump_json(exclude_none=True))
      )
    else:
      new_parts.append(part)
  return new_parts, has_function_response_part, has_function_call_part


def _build_gemma_function_system_instruction(
    function_declarations: list[FunctionDeclaration],
) -> str:
  """Constructs the system instruction string for Gemma function calling."""
  if not function_declarations:
    return ''

  system_instruction_prefix = 'You have access to the following functions:\n['
  instruction_parts = []
  for func in function_declarations:
    instruction_parts.append(func.model_dump_json(exclude_none=True))

  separator = ',\n'
  system_instruction = (
      f'{system_instruction_prefix}{separator.join(instruction_parts)}\n]\n'
  )

  system_instruction += (
      'When you call a function, you MUST respond in the format of: '
      """{"name": function name, "parameters": dictionary of argument name and its value}\n"""
      'When you call a function, you MUST NOT include any other text in the'
      ' response.\n'
  )
  return system_instruction


def _get_last_valid_json_substring(text: str) -> tuple[bool, str | None]:
  """Attempts to find and return the last valid JSON object in a string.

  This function is designed to extract JSON that might be embedded in a larger
  text, potentially with introductory or concluding remarks. It will always chose
  the last block of valid json found within the supplied text (if it exists).

  Args:
    text: The input string to search for JSON objects.

  Returns:
    A tuple:
      - bool: True if a valid JSON substring was found, False otherwise.
      - str | None: The last valid JSON substring found, or None if none was
        found.
  """
  decoder = json.JSONDecoder()
  last_json_str = None
  start_pos = 0
  while start_pos < len(text):
    try:
      first_brace_index = text.index('{', start_pos)
      _, end_index = decoder.raw_decode(text[first_brace_index:])
      last_json_str = text[first_brace_index : first_brace_index + end_index]
      start_pos = first_brace_index + end_index
    except json.JSONDecodeError:
      start_pos = first_brace_index + 1
    except ValueError:
      break

  if last_json_str:
    return True, last_json_str
  return False, None


try:
  from google.adk.models.lite_llm import LiteLlm  # noqa: F401
except Exception:
  # LiteLLM not available, Gemma3Ollama will not be defined
  LiteLlm = None

if LiteLlm is not None:

  class Gemma3Ollama(GemmaFunctionCallingMixin, LiteLlm):
    """Integration for Gemma 3 models running locally via Ollama.

    This enables fully local agent workflows using Gemma 3 models.
    Requires Ollama to be running with a Gemma 3 model pulled.

    Example:
      ollama pull gemma3:12b
      model = Gemma3Ollama(model="ollama/gemma3:12b")
    """

    def __init__(self, model: str = 'ollama/gemma3:12b', **kwargs):
      super().__init__(model=model, **kwargs)

    def __repr__(self) -> str:
      return f'{self.__class__.__name__}(model="{self.model}")'

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
      return [
          r'ollama/gemma3.*',
      ]

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      """Sends a request to Gemma via Ollama/LiteLLM.

      Args:
        llm_request: LlmRequest, the request to send.
        stream: bool = False, whether to do streaming call.

      Yields:
        LlmResponse: The model response.
      """
      self._move_function_calls_into_system_instruction(llm_request)

      async for response in super().generate_content_async(llm_request, stream):
        self._extract_function_calls_from_response(response)
        yield response

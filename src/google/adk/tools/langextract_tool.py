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

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Optional

from google.genai import types
from typing_extensions import override

from ..features import FeatureName
from ..features import is_feature_enabled
from .base_tool import BaseTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig
from .tool_context import ToolContext

try:
  import langextract as lx
except ImportError as e:
  raise ImportError(
      "LangExtract tools require pip install 'google-adk[extensions]'."
  ) from e

logger = logging.getLogger('google_adk.' + __name__)


class LangExtractTool(BaseTool):
  """A tool that extracts structured information from text using LangExtract.

  This tool wraps the langextract library to enable LLM agents to extract
  structured data (entities, attributes, relationships) from unstructured
  text. The agent provides the text to extract from and a description of
  what to extract; other parameters are pre-configured at construction time.

  Args:
    name: The name of the tool. Defaults to 'langextract'.
    description: The description of the tool shown to the LLM.
    examples: Optional list of langextract ExampleData for few-shot
      extraction guidance.
    model_id: The model ID for langextract to use internally.
      Defaults to 'gemini-2.5-flash'.
    api_key: Optional API key for langextract. If None, uses the
      LANGEXTRACT_API_KEY environment variable.
    extraction_passes: Number of extraction passes. Defaults to 1.
    max_workers: Maximum worker threads for langextract.
      Defaults to 1.
    max_char_buffer: Maximum character buffer size for text
      chunking. Defaults to 4000.

  Examples::

      from google.adk.tools.langextract_tool import LangExtractTool
      import langextract as lx

      examples = [
          lx.data.ExampleData(
              text="John is a software engineer at Google.",
              extractions=[
                  lx.data.Extraction(
                      extraction_class="person",
                      extraction_text="John",
                      attributes={
                          "role": "software engineer",
                          "company": "Google",
                      },
                  )
              ],
          )
      ]

      tool = LangExtractTool(
          name='extract_people',
          description='Extract person entities from text.',
          examples=examples,
      )
  """

  def __init__(
      self,
      *,
      name: str = 'langextract',
      description: str = (
          'Extracts structured information from unstructured'
          ' text. Provide the text and a description of what'
          ' to extract.'
      ),
      examples: Optional[list[lx.data.ExampleData]] = None,
      model_id: str = 'gemini-2.5-flash',
      api_key: Optional[str] = None,
      extraction_passes: int = 1,
      max_workers: int = 1,
      max_char_buffer: int = 4000,
  ):
    super().__init__(name=name, description=description)
    self._examples = examples or []
    self._model_id = model_id
    self._api_key = api_key
    self._extraction_passes = extraction_passes
    self._max_workers = max_workers
    self._max_char_buffer = max_char_buffer

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              'type': 'object',
              'properties': {
                  'text': {
                      'type': 'string',
                      'description': (
                          'The unstructured text to extract information from.'
                      ),
                  },
                  'prompt_description': {
                      'type': 'string',
                      'description': (
                          'A description of what kind of'
                          ' information to extract from'
                          ' the text.'
                      ),
                  },
              },
              'required': ['text', 'prompt_description'],
          },
      )
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'text': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'The unstructured text to extract information from.'
                    ),
                ),
                'prompt_description': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'A description of what kind of'
                        ' information to extract from'
                        ' the text.'
                    ),
                ),
            },
            required=['text', 'prompt_description'],
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    text = args.get('text')
    prompt_description = args.get('prompt_description')

    if not text:
      return {'error': 'The "text" parameter is required.'}
    if not prompt_description:
      return {
          'error': 'The "prompt_description" parameter is required.',
      }

    try:
      extract_kwargs: dict[str, Any] = {
          'text_or_documents': text,
          'prompt_description': prompt_description,
          'examples': self._examples,
          'model_id': self._model_id,
          'extraction_passes': self._extraction_passes,
          'max_workers': self._max_workers,
          'max_char_buffer': self._max_char_buffer,
      }
      if self._api_key is not None:
        extract_kwargs['api_key'] = self._api_key

      # lx.extract() is synchronous; run in a thread to avoid
      # blocking the event loop.
      result = await asyncio.to_thread(lx.extract, **extract_kwargs)

      extractions = []
      for extraction in result:
        entry = {
            'extraction_class': extraction.extraction_class,
            'extraction_text': extraction.extraction_text,
        }
        if extraction.attributes:
          entry['attributes'] = extraction.attributes
        extractions.append(entry)

      return {'extractions': extractions}

    except Exception as e:
      logger.error('LangExtract extraction failed: %s', e)
      return {'error': f'Extraction failed: {e}'}

  @override
  @classmethod
  def from_config(
      cls: type[LangExtractTool],
      config: ToolArgsConfig,
      config_abs_path: str,
  ) -> LangExtractTool:
    from ..agents import config_agent_utils

    langextract_config = LangExtractToolConfig.model_validate(
        config.model_dump()
    )

    examples = []
    if langextract_config.examples:
      examples = config_agent_utils.resolve_fully_qualified_name(
          langextract_config.examples
      )

    return cls(
        name=langextract_config.name,
        description=langextract_config.description,
        examples=examples,
        model_id=langextract_config.model_id,
        api_key=langextract_config.api_key,
        extraction_passes=langextract_config.extraction_passes,
        max_workers=langextract_config.max_workers,
        max_char_buffer=langextract_config.max_char_buffer,
    )


class LangExtractToolConfig(BaseToolConfig):
  """Configuration for LangExtractTool when loaded from YAML config."""

  name: str = 'langextract'
  """The name of the tool."""

  description: str = 'Extracts structured information from unstructured text.'
  """The description of the tool."""

  examples: str = ''
  """Fully qualified path to a list of ExampleData instances."""

  model_id: str = 'gemini-2.5-flash'
  """The model ID for langextract."""

  api_key: Optional[str] = None
  """Optional API key for langextract."""

  extraction_passes: int = 1
  """Number of extraction passes."""

  max_workers: int = 1
  """Maximum worker threads."""

  max_char_buffer: int = 4000
  """Maximum character buffer size."""

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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

pytest.importorskip('langextract', reason='Requires langextract')

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.langextract_tool import LangExtractTool
from google.adk.tools.langextract_tool import LangExtractToolConfig


def test_langextract_tool_default_initialization():
  """Test that LangExtractTool initializes with correct defaults."""
  tool = LangExtractTool()
  assert tool.name == 'langextract'
  assert 'structured information' in tool.description
  assert tool._model_id == 'gemini-2.5-flash'
  assert tool._examples == []
  assert tool._extraction_passes == 1
  assert tool._max_workers == 1
  assert tool._max_char_buffer == 4000
  assert tool._api_key is None


def test_langextract_tool_custom_initialization():
  """Test that LangExtractTool accepts custom parameters."""
  import langextract as lx

  examples = [
      lx.data.ExampleData(
          text='test text',
          extractions=[
              lx.data.Extraction(
                  extraction_class='entity',
                  extraction_text='test',
              )
          ],
      )
  ]
  tool = LangExtractTool(
      name='my_extractor',
      description='Custom extractor',
      examples=examples,
      model_id='gemini-2.0-flash',
      api_key='test-key',
      extraction_passes=2,
      max_workers=4,
      max_char_buffer=8000,
  )
  assert tool.name == 'my_extractor'
  assert tool.description == 'Custom extractor'
  assert len(tool._examples) == 1
  assert tool._model_id == 'gemini-2.0-flash'
  assert tool._api_key == 'test-key'
  assert tool._extraction_passes == 2
  assert tool._max_workers == 4
  assert tool._max_char_buffer == 8000


@pytest.mark.parametrize('json_schema_enabled', [True, False])
def test_langextract_tool_get_declaration(json_schema_enabled):
  """Test that _get_declaration returns the correct schema."""
  with temporary_feature_override(
      FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, json_schema_enabled
  ):
    tool = LangExtractTool()
    declaration = tool._get_declaration()
    assert declaration is not None
    assert declaration.name == 'langextract'

    if json_schema_enabled:
      params = declaration.parameters_json_schema
      assert params is not None
      props = params['properties']
      required = params['required']
    else:
      params = declaration.parameters
      assert params is not None
      props = params.properties
      required = params.required

    assert 'text' in props
    assert 'prompt_description' in props
    assert 'text' in required
    assert 'prompt_description' in required


@pytest.mark.asyncio
@patch('google.adk.tools.langextract_tool.lx')
async def test_langextract_tool_run_async(mock_lx):
  """Test that run_async calls lx.extract and returns results."""
  mock_extraction = MagicMock()
  mock_extraction.extraction_class = 'person'
  mock_extraction.extraction_text = 'John'
  mock_extraction.attributes = {'role': 'engineer'}
  mock_lx.extract.return_value = [mock_extraction]

  tool = LangExtractTool()
  result = await tool.run_async(
      args={
          'text': 'John is an engineer.',
          'prompt_description': 'Extract people.',
      },
      tool_context=MagicMock(),
  )

  assert 'extractions' in result
  assert len(result['extractions']) == 1
  assert result['extractions'][0]['extraction_class'] == 'person'
  assert result['extractions'][0]['extraction_text'] == 'John'
  assert result['extractions'][0]['attributes'] == {'role': 'engineer'}
  mock_lx.extract.assert_called_once()


@pytest.mark.asyncio
async def test_langextract_tool_missing_text():
  """Test that run_async returns error when text is missing."""
  tool = LangExtractTool()
  result = await tool.run_async(
      args={'prompt_description': 'Extract people.'},
      tool_context=MagicMock(),
  )
  assert 'error' in result
  assert 'text' in result['error']


@pytest.mark.asyncio
async def test_langextract_tool_missing_prompt_description():
  """Test that run_async returns error when prompt_description is missing."""
  tool = LangExtractTool()
  result = await tool.run_async(
      args={'text': 'Some text.'},
      tool_context=MagicMock(),
  )
  assert 'error' in result
  assert 'prompt_description' in result['error']


@pytest.mark.asyncio
@patch('google.adk.tools.langextract_tool.lx')
async def test_langextract_tool_extraction_error(mock_lx):
  """Test that run_async handles extraction errors gracefully."""
  mock_lx.extract.side_effect = RuntimeError('API error')

  tool = LangExtractTool()
  result = await tool.run_async(
      args={
          'text': 'Some text.',
          'prompt_description': 'Extract stuff.',
      },
      tool_context=MagicMock(),
  )
  assert 'error' in result
  assert 'Extraction failed' in result['error']


def test_langextract_tool_config():
  """Test that LangExtractToolConfig validates correctly."""
  config = LangExtractToolConfig(
      name='my_tool',
      description='My custom extractor',
      model_id='gemini-2.0-flash',
      extraction_passes=3,
      max_workers=2,
      max_char_buffer=6000,
  )
  assert config.name == 'my_tool'
  assert config.description == 'My custom extractor'
  assert config.model_id == 'gemini-2.0-flash'
  assert config.extraction_passes == 3
  assert config.max_workers == 2
  assert config.max_char_buffer == 6000
  assert config.examples == ''
  assert config.api_key is None

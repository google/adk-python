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

import os
from unittest import mock
from unittest.mock import AsyncMock

from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from google.genai.types import Content
from google.genai.types import Part
import pytest

BASE_MODEL_ID = 'gemini-2.5-flash'
APIGEE_GEMINI_MODEL_ID = 'apigee/gemini/v1/' + BASE_MODEL_ID
APIGEE_VERTEX_MODEL_ID = 'apigee/vertex_ai/v1beta/gemini-pro'
VERTEX_BASE_MODEL_ID = 'gemini-pro'
PROXY_URL = 'https://test.apigee.net'


@pytest.fixture
def llm_request():
  """Provides a sample LlmRequest for testing."""
  return LlmRequest(
      model=APIGEE_GEMINI_MODEL_ID,
      contents=[
          types.Content(
              role='user', parts=[types.Part.from_text(text='Test prompt')]
          )
      ],
  )


@pytest.mark.asyncio
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_generate_content_async_non_streaming(
    mock_client_constructor, llm_request
):
  """Tests the generate_content_async method for non-streaming responses."""
  apigee_llm_instance = ApigeeLlm(
      model=APIGEE_GEMINI_MODEL_ID,
      proxy_url=PROXY_URL,
  )
  mock_client_instance = mock.Mock()
  mock_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  parts=[Part.from_text(text='Test response')],
                  role='model',
              )
          )
      ]
  )
  mock_client_instance.aio.models.generate_content = AsyncMock(
      return_value=mock_response
  )
  mock_client_constructor.return_value = mock_client_instance

  response_generator = apigee_llm_instance.generate_content_async(llm_request)
  responses = [resp async for resp in response_generator]

  assert len(responses) == 1
  llm_response = responses[0]
  assert llm_response.content.parts[0].text == 'Test response'
  assert llm_response.content.role == 'model'

  mock_client_constructor.assert_called_once()
  _, kwargs = mock_client_constructor.call_args
  assert not kwargs['vertexai']
  http_options = kwargs['http_options']
  assert http_options.base_url == PROXY_URL
  assert http_options.api_version == 'v1'
  assert 'user-agent' in http_options.headers
  assert 'x-goog-api-client' in http_options.headers

  mock_client_instance.aio.models.generate_content.assert_called_once_with(
      model=BASE_MODEL_ID,
      contents=llm_request.contents,
      config=llm_request.config,
  )


@pytest.mark.asyncio
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_generate_content_async_streaming(
    mock_client_constructor, llm_request
):
  """Tests the generate_content_async method for streaming responses."""
  apigee_llm_instance = ApigeeLlm(
      model=APIGEE_GEMINI_MODEL_ID,
      proxy_url=PROXY_URL,
  )
  mock_client_instance = mock.Mock()
  mock_responses = [
      types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=Content(
                      parts=[Part.from_text(text='Hello')],
                  )
              )
          ]
      ),
      types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=Content(
                      parts=[Part.from_text(text=',')],
                  )
              )
          ]
      ),
      types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=Content(
                      parts=[Part.from_text(text=' world!')],
                  )
              )
          ]
      ),
  ]

  async def mock_stream_generator():
    for r in mock_responses:
      yield r

  mock_client_instance.aio.models.generate_content_stream = AsyncMock(
      return_value=mock_stream_generator()
  )
  mock_client_constructor.return_value = mock_client_instance

  response_generator = apigee_llm_instance.generate_content_async(
      llm_request, stream=True
  )
  responses = [resp async for resp in response_generator]

  assert responses
  full_text_parts = []
  for r in responses:
    for p in r.content.parts:
      if p.text:
        full_text_parts.append(p.text)
  full_text = ''.join(full_text_parts)
  assert 'Hello, world!' in full_text

  mock_client_instance.aio.models.generate_content_stream.assert_called_once_with(
      model=BASE_MODEL_ID,
      contents=llm_request.contents,
      config=llm_request.config,
  )


@pytest.mark.asyncio
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_generate_content_async_with_custom_headers(
    mock_client_constructor, llm_request
):
  """Tests that custom headers are passed in the request."""
  custom_headers = {
      'X-Custom-Header': 'custom-value',
  }
  apigee_llm = ApigeeLlm(
      model=APIGEE_GEMINI_MODEL_ID,
      proxy_url=PROXY_URL,
      custom_headers=custom_headers,
  )
  mock_client_instance = mock.Mock()
  mock_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=Content(
                  parts=[Part.from_text(text='Test response')],
                  role='model',
              )
          )
      ]
  )
  mock_client_instance.aio.models.generate_content = AsyncMock(
      return_value=mock_response
  )
  mock_client_constructor.return_value = mock_client_instance

  response_generator = apigee_llm.generate_content_async(llm_request)
  _ = [resp async for resp in response_generator]  # Consume generator

  mock_client_constructor.assert_called_once()
  _, kwargs = mock_client_constructor.call_args
  http_options = kwargs['http_options']
  assert http_options.headers['X-Custom-Header'] == 'custom-value'
  assert 'user-agent' in http_options.headers


@pytest.mark.asyncio
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_vertex_model_path_parsing(mock_client_constructor):
  """Tests that Vertex AI model paths are parsed correctly."""
  apigee_llm = ApigeeLlm(model=APIGEE_VERTEX_MODEL_ID, proxy_url=PROXY_URL)
  llm_request = LlmRequest(
      model=APIGEE_VERTEX_MODEL_ID,
      contents=[
          types.Content(
              role='user', parts=[types.Part.from_text(text='Test prompt')]
          )
      ],
  )
  mock_client_instance = mock.Mock()
  mock_client_instance.aio.models.generate_content = AsyncMock(
      return_value=types.GenerateContentResponse(
          candidates=[
              types.Candidate(
                  content=Content(
                      parts=[Part.from_text(text='Test response')],
                      role='model',
                  )
              )
          ]
      )
  )
  mock_client_constructor.return_value = mock_client_instance

  _ = [resp async for resp in apigee_llm.generate_content_async(llm_request)]

  mock_client_constructor.assert_called_once()
  _, kwargs = mock_client_constructor.call_args
  assert kwargs['vertexai']
  assert kwargs['http_options'].api_version == 'v1beta'

  mock_client_instance.aio.models.generate_content.assert_called_once()
  call_kwargs = (
      mock_client_instance.aio.models.generate_content.call_args.kwargs
  )
  assert call_kwargs['model'] == VERTEX_BASE_MODEL_ID


@pytest.mark.asyncio
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_proxy_url_from_env_variable(mock_client_constructor):
  """Tests that proxy_url is read from environment variable."""
  with mock.patch.dict(
      os.environ, {'APIGEE_PROXY_URL': 'https://env.proxy.url'}
  ):
    apigee_llm = ApigeeLlm(model=APIGEE_GEMINI_MODEL_ID)
    llm_request = LlmRequest(
        model=APIGEE_GEMINI_MODEL_ID,
        contents=[
            types.Content(
                role='user', parts=[types.Part.from_text(text='Test prompt')]
            )
        ],
    )
    mock_client_instance = mock.Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(
        return_value=types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(
                        parts=[Part.from_text(text='Test response')],
                        role='model',
                    )
                )
            ]
        )
    )
    mock_client_constructor.return_value = mock_client_instance

    _ = [resp async for resp in apigee_llm.generate_content_async(llm_request)]

    mock_client_constructor.assert_called_once()
    _, kwargs = mock_client_constructor.call_args
    assert kwargs['http_options'].base_url == 'https://env.proxy.url'


@pytest.mark.parametrize(
    ('model_string', 'env_vars'),
    [
        (
            'apigee/vertex_ai/gemini-2.5-flash',
            {'GOOGLE_CLOUD_LOCATION': 'test-location'},
        ),
        (
            'apigee/vertex_ai/gemini-2.5-flash',
            {'GOOGLE_CLOUD_PROJECT': 'test-project'},
        ),
        (
            'apigee/gemini-2.5-flash',
            {
                'GOOGLE_GENAI_USE_VERTEXAI': 'true',
                'GOOGLE_CLOUD_LOCATION': 'test-location',
            },
        ),
        (
            'apigee/gemini-2.5-flash',
            {
                'GOOGLE_GENAI_USE_VERTEXAI': 'true',
                'GOOGLE_CLOUD_PROJECT': 'test-project',
            },
        ),
    ],
)
def test_vertex_model_missing_project_or_location_raises_error(
    model_string, env_vars
):
  """Tests that ValueError is raised for Vertex models if project or location is missing."""
  with mock.patch.dict(os.environ, env_vars, clear=True):
    with pytest.raises(ValueError, match='environment variable must be set'):
      ApigeeLlm(model=model_string, proxy_url=PROXY_URL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        'model_string',
        'use_vertexai_env',
        'expected_is_vertexai',
        'expected_api_version',
        'expected_model_id',
    ),
    [
        ('apigee/gemini-2.5-flash', None, False, None, 'gemini-2.5-flash'),
        ('apigee/gemini-2.5-flash', 'true', True, None, 'gemini-2.5-flash'),
        ('apigee/gemini-2.5-flash', '1', True, None, 'gemini-2.5-flash'),
        ('apigee/gemini-2.5-flash', 'false', False, None, 'gemini-2.5-flash'),
        ('apigee/gemini-2.5-flash', '0', False, None, 'gemini-2.5-flash'),
        (
            'apigee/v1/gemini-2.5-flash',
            None,
            False,
            'v1',
            'gemini-2.5-flash',
        ),
        (
            'apigee/v1/gemini-2.5-flash',
            'true',
            True,
            'v1',
            'gemini-2.5-flash',
        ),
        (
            'apigee/vertex_ai/gemini-2.5-flash',
            None,
            True,
            None,
            'gemini-2.5-flash',
        ),
        (
            'apigee/vertex_ai/gemini-2.5-flash',
            'false',
            True,
            None,
            'gemini-2.5-flash',
        ),
        (
            'apigee/gemini/v1/gemini-2.5-flash',
            'true',
            False,
            'v1',
            'gemini-2.5-flash',
        ),
        (
            'apigee/vertex_ai/v1beta/gemini-2.5-flash',
            'false',
            True,
            'v1beta',
            'gemini-2.5-flash',
        ),
    ],
)
@mock.patch('google.adk.models.apigee_llm.Client')
async def test_model_string_parsing_and_client_initialization(
    mock_client_constructor,
    model_string,
    use_vertexai_env,
    expected_is_vertexai,
    expected_api_version,
    expected_model_id,
):
  """Tests model string parsing and genai.Client initialization."""
  env_vars = {}
  if use_vertexai_env is not None:
    env_vars['GOOGLE_GENAI_USE_VERTEXAI'] = use_vertexai_env

  if expected_is_vertexai:
    env_vars['GOOGLE_CLOUD_PROJECT'] = 'test-project'
    env_vars['GOOGLE_CLOUD_LOCATION'] = 'test-location'

  # The ApigeeLlm is initialized in the 'with' block to make sure that the mock
  # of the environment variable is active.
  with mock.patch.dict(os.environ, env_vars, clear=True):
    apigee_llm = ApigeeLlm(model=model_string, proxy_url=PROXY_URL)
    request = LlmRequest(model=model_string, contents=[])

    mock_client_instance = mock.Mock()
    mock_client_instance.aio.models.generate_content = AsyncMock(
        return_value=types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=Content(parts=[Part.from_text(text='')])
                )
            ]
        )
    )
    mock_client_constructor.return_value = mock_client_instance

    _ = [resp async for resp in apigee_llm.generate_content_async(request)]

    mock_client_constructor.assert_called_once()
    _, kwargs = mock_client_constructor.call_args
    assert kwargs['vertexai'] == expected_is_vertexai
    if expected_is_vertexai:
      assert kwargs['project'] == 'test-project'
      assert kwargs['location'] == 'test-location'
    http_options = kwargs['http_options']
    assert http_options.api_version == expected_api_version

    (
        mock_client_instance.aio.models.generate_content.assert_called_once_with(
            model=expected_model_id,
            contents=request.contents,
            config=request.config,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'invalid_model_string',
    [
        'apigee/openai/v1/gpt',
        'apigee/',  # Missing model_id
        'apigee',  # Invalid format
        'gemini-pro',  # Invalid format
        'apigee/vertex_ai/v1/model/extra',  # Too many components
        'apigee/unknown/model',
    ],
)
async def test_invalid_model_strings_raise_value_error(invalid_model_string):
  """Tests that invalid model strings raise a ValueError."""
  with pytest.raises(
      ValueError, match=f'Invalid model string: {invalid_model_string}'
  ):
    ApigeeLlm(model=invalid_model_string, proxy_url=PROXY_URL)

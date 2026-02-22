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
from unittest.mock import AsyncMock

from google.adk.models.apigee_llm import CompletionsHTTPClient
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import httpx
import pytest


@pytest.fixture
def client():
  return CompletionsHTTPClient(base_url='https://example.com')


@pytest.fixture(name='llm_request')
def fixture_llm_request():
  return LlmRequest(
      model='apigee/open_llama',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hello')])
      ],
  )


@pytest.mark.asyncio
async def test_construct_payload_basic_payload(client, llm_request):
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{'message': {'role': 'assistant', 'content': 'Hi'}}]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    url = call_args[0][0]
    kwargs = call_args[1]

    assert url == 'https://example.com/chat/completions'
    payload = kwargs['json']
    assert payload['model'] == 'open_llama'
    assert payload['stream'] is False
    assert len(payload['messages']) == 1
    assert payload['messages'][0]['role'] == 'user'
    assert payload['messages'][0]['content'] == 'Hello'


@pytest.mark.asyncio
async def test_construct_payload_with_config(client, llm_request):
  llm_request.config = types.GenerateContentConfig(
      temperature=0.7,
      top_p=0.9,
      max_output_tokens=100,
      stop_sequences=['STOP'],
      frequency_penalty=0.5,
      presence_penalty=0.5,
      seed=42,
      candidate_count=2,
      response_mime_type='application/json',
  )

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{'message': {'role': 'assistant', 'content': 'Hi'}}]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]['json']

    assert payload['temperature'] == 0.7
    assert payload['top_p'] == 0.9
    assert payload['max_tokens'] == 100
    assert payload['stop'] == ['STOP']
    assert payload['frequency_penalty'] == 0.5
    assert payload['presence_penalty'] == 0.5
    assert payload['seed'] == 42
    assert payload['n'] == 2
    assert payload['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_construct_payload_with_tools(client, llm_request):
  tool = types.Tool(
      function_declarations=[
          types.FunctionDeclaration(
              name='get_weather',
              description='Get weather',
              parameters=types.Schema(
                  type=types.Type.OBJECT,
                  properties={'location': types.Schema(type=types.Type.STRING)},
              ),
          )
      ]
  )
  llm_request.config = types.GenerateContentConfig(tools=[tool])

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{'message': {'role': 'assistant', 'content': 'Hi'}}]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]['json']
    assert 'tools' in payload
    assert payload['tools'][0]['function']['name'] == 'get_weather'


@pytest.mark.asyncio
async def test_construct_payload_system_instruction(client, llm_request):
  llm_request.config = types.GenerateContentConfig(
      system_instruction='You are a helpful assistant.'
  )
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{'message': {'role': 'assistant', 'content': 'Hi'}}]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    payload = mock_post.call_args[1]['json']
    assert payload['messages'][0]['role'] == 'system'
    assert payload['messages'][0]['content'] == 'You are a helpful assistant.'
    # Ensure user message follows system
    assert payload['messages'][1]['role'] == 'user'


@pytest.mark.asyncio
async def test_construct_payload_multimodal_content(client):
  # Mock inline_data for image
  image_data = b'fake_image_bytes'
  llm_request = LlmRequest(
      model='apigee/open_llama',
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part.from_text(text='What is this?'),
                  types.Part.from_bytes(
                      data=image_data, mime_type='image/jpeg'
                  ),
              ],
          )
      ],
  )

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [
          {'message': {'role': 'assistant', 'content': 'It is an image'}}
      ]
  }

  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]['json']
    assert len(payload['messages']) == 1
    message = payload['messages'][0]
    assert message['role'] == 'user'
    assert isinstance(message['content'], list)
    assert len(message['content']) == 2
    assert message['content'][0] == {'type': 'text', 'text': 'What is this?'}
    assert message['content'][1]['type'] == 'image_url'
    # Base64 encoding of b'fake_image_bytes' is 'ZmFrZV9pbWFnZV9ieXRlcw=='
    assert message['content'][1]['image_url']['url'] == (
        'data:image/jpeg;base64,ZmFrZV9pbWFnZV9ieXRlcw=='
    )


@pytest.mark.asyncio
async def test_construct_payload_image_file_uri(client):
  llm_request = LlmRequest(
      model='apigee/open_llama',
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part.from_uri(
                      file_uri='https://example.com/image.jpg',
                      mime_type='image/jpeg',
                  )
              ],
          )
      ],
  )

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [
          {'message': {'role': 'assistant', 'content': 'It is an image'}}
      ]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]['json']
    assert len(payload['messages']) == 1
    message = payload['messages'][0]
    assert message['role'] == 'user'
    assert isinstance(message['content'], list)
    assert message['content'][0] == {
        'type': 'image_url',
        'image_url': {'url': 'https://example.com/image.jpg'},
    }


@pytest.mark.asyncio
async def test_generate_content_async_function_call_response(
    client, llm_request
):
  # Mock response with tool call
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{
          'message': {
              'role': 'assistant',
              'content': None,
              'tool_calls': [{
                  'id': 'call_123',
                  'type': 'function',
                  'function': {
                      'name': 'get_weather',
                      'arguments': '{"location": "London"}',
                  },
              }],
          }
      }]
  }
  mock_response.status_code = 200

  with mock.patch.object(httpx.AsyncClient, 'post', return_value=mock_response):
    responses = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    assert len(responses) == 1
    part = responses[0].content.parts[0]
    assert part.function_call
    assert part.function_call.name == 'get_weather'
    assert part.function_call.args == {'location': 'London'}
    assert part.function_call.id == 'call_123'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('response_json_schema', 'response_mime_type', 'expected_response_format'),
    [
        # Case 1: Only response_json_schema is provided
        (
            {'type': 'object', 'properties': {'name': {'type': 'string'}}},
            None,
            {
                'type': 'json_schema',
                'json_schema': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                },
            },
        ),
        # Case 2: Both provided, schema takes precedence
        (
            {'type': 'object', 'properties': {'name': {'type': 'string'}}},
            'application/json',
            {
                'type': 'json_schema',
                'json_schema': {
                    'type': 'object',
                    'properties': {'name': {'type': 'string'}},
                },
            },
        ),
        # Case 3: Only response_mime_type is provided
        (
            None,
            'application/json',
            {'type': 'json_object'},
        ),
    ],
)
async def test_construct_payload_response_format(
    client,
    llm_request,
    response_json_schema,
    response_mime_type,
    expected_response_format,
):
  llm_request.config = types.GenerateContentConfig(
      response_json_schema=response_json_schema,
      response_mime_type=response_mime_type,
  )
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{'message': {'role': 'assistant', 'content': '{}'}}]
  }
  mock_response.status_code = 200

  with mock.patch.object(
      httpx.AsyncClient, 'post', return_value=mock_response
  ) as mock_post:
    _ = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    mock_post.assert_called_once()
    payload = mock_post.call_args[1]['json']
    assert payload['response_format'] == expected_response_format


@pytest.mark.asyncio
async def test_generate_content_async_invalid_tool_call_type_raises_error(
    client, llm_request
):
  # Mock response with invalid tool call type
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{
          'message': {
              'role': 'assistant',
              'content': None,
              'tool_calls': [{
                  'id': 'call_123',
                  # Invalid type
                  'type': 'custom',
                  'custom': {
                      'name': 'read_string',
                      'input': 'Hi! The this is a custom tool call!',
                  },
              }],
          }
      }]
  }
  mock_response.status_code = 200

  with mock.patch.object(httpx.AsyncClient, 'post', return_value=mock_response):
    with pytest.raises(ValueError, match='Unsupported tool_call type: custom'):
      _ = [
          r
          async for r in client.generate_content_async(
              llm_request, stream=False
          )
      ]


@pytest.mark.asyncio
async def test_generate_content_async_function_call_response(
    client, llm_request
):
  # Mock response with deprecated function call
  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.json.return_value = {
      'choices': [{
          'message': {
              'role': 'assistant',
              'content': None,
              'function_call': {
                  'name': 'get_weather',
                  'arguments': '{"location": "London"}',
              },
          }
      }]
  }
  mock_response.status_code = 200

  with mock.patch.object(httpx.AsyncClient, 'post', return_value=mock_response):
    responses = [
        r
        async for r in client.generate_content_async(llm_request, stream=False)
    ]

    assert len(responses) == 1
    part = responses[0].content.parts[0]
    assert part.function_call
    assert part.function_call.name == 'get_weather'
    assert part.function_call.args == {'location': 'London'}
    assert part.function_call.id is None

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

import json
from unittest import mock
from unittest.mock import AsyncMock

from google.adk.models.apigee_llm import CompletionsHTTPClient
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import httpx
import pytest


@pytest.fixture
def client():
  return CompletionsHTTPClient(base_url='https://localhost')


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

    assert url == 'https://localhost/chat/completions'
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
                      file_uri='https://localhost/image.jpg',
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
        'image_url': {'url': 'https://localhost/image.jpg'},
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


@pytest.mark.asyncio
async def test_generate_content_async_streaming_function_call():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  llm_request = LlmRequest(
      model='apigee/test',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
  )

  # Mock chunks simulating split arguments
  chunk_data_0 = {
      'id': 'chatcmpl-123',
      'object': 'chat.completion.chunk',
      'created': 1234567890,
      'model': 'gpt-3.5-turbo',
      'service_tier': 'default',
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [{
                  'index': 0,
                  'id': 'call_123',
                  'type': 'function',
                  'function': {'name': 'get_weather', 'arguments': ''},
              }]
          },
          'finish_reason': None,
      }],
  }
  chunk_data_1 = {
      'id': 'chatcmpl-123',
      'object': 'chat.completion.chunk',
      'created': 1234567890,
      'model': 'gpt-3.5-turbo',
      'service_tier': 'default',
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [{
                  'index': 0,
                  'function': {'arguments': '{"location": "London"}'},
              }]
          },
          'finish_reason': None,
      }],
  }
  chunk_data_2 = {
      'id': 'chatcmpl-123',
      'object': 'chat.completion.chunk',
      'created': 1234567890,
      'model': 'gpt-3.5-turbo',
      'service_tier': 'default',
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [{
                  'index': 0,
                  'function': {'arguments': '{"country": "UK"}'},
              }]
          },
          'finish_reason': None,
      }],
  }
  chunk_data_3 = {
      'id': 'chatcmpl-123',
      'object': 'chat.completion.chunk',
      'created': 1234567890,
      'model': 'gpt-3.5-turbo',
      'service_tier': 'default',
      'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
      'usage': {
          'prompt_tokens': 10,
          'completion_tokens': 20,
          'total_tokens': 30,
      },
  }

  chunks = [
      f'{json.dumps(chunk_data_0)}\n',
      f'{json.dumps(chunk_data_1)}\n',
      f'{json.dumps(chunk_data_2)}\n',
      f'{json.dumps(chunk_data_3)}\n',
  ]

  async def mock_aiter_lines():
    for chunk in chunks:
      yield chunk

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.aiter_lines.return_value = mock_aiter_lines()
  mock_response.status_code = 200

  mock_stream_ctx = mock.AsyncMock()
  mock_stream_ctx.__aenter__.return_value = mock_response

  with mock.patch.object(
      httpx.AsyncClient, 'stream', return_value=mock_stream_ctx
  ):
    responses = [
        r
        async for r in local_client.generate_content_async(
            llm_request, stream=True
        )
    ]
    # Check that we get 5 responses (one per chunk + extra final accumulated)
    assert len(responses) == 5

    # Check 1st response: partial tool call, empty args
    assert responses[0].partial is True
    assert responses[0].content.parts[0].function_call.name == 'get_weather'
    assert responses[0].content.parts[0].function_call.id == 'call_123'

    # Check 2nd response: full args for first update
    assert responses[1].partial is True
    assert responses[1].content.parts[0].function_call.args == {
        'location': 'London'
    }

    # Check 3rd response: full args for second update (merged)
    assert responses[2].partial is True
    assert responses[2].content.parts[0].function_call.args == {'country': 'UK'}

    # Check 4th response: last delta (empty)
    assert responses[3].partial is True
    assert responses[3].content.parts == []

    # Check 5th response: final accumulated
    assert responses[4].finish_reason == types.FinishReason.STOP
    # Full accumulated args
    assert responses[4].content.parts[0].function_call.args == {
        'location': 'London',
        'country': 'UK',
    }

    # Check metadata and usage
    assert responses[4].model_version == 'gpt-3.5-turbo'
    assert responses[4].custom_metadata['id'] == 'chatcmpl-123'
    assert responses[4].custom_metadata['created'], 1234567890
    assert responses[4].custom_metadata['object'], 'chat.completion.chunk'
    assert responses[4].custom_metadata['service_tier'], 'default'
    assert responses[4].usage_metadata is not None
    assert responses[4].usage_metadata.prompt_token_count == 10
    assert responses[4].usage_metadata.candidates_token_count == 20
    assert responses[4].usage_metadata.total_token_count == 30


@pytest.mark.asyncio
async def test_generate_content_async_streaming_multiple_function_calls():
  # Mock streaming response with multiple tool calls
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  llm_request = LlmRequest(
      model='apigee/test',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
  )
  chunk_data_1 = {
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [
                  {
                      'index': 0,
                      'id': 'call_1',
                      'type': 'function',
                      'function': {'name': 'func_1', 'arguments': ''},
                  },
                  {
                      'index': 1,
                      'id': 'call_2',
                      'type': 'function',
                      'function': {'name': 'func_2', 'arguments': ''},
                  },
              ]
          },
          'finish_reason': None,
      }]
  }
  # the tool_call type is optional in chunk responses.
  chunk_data_2 = {
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [
                  {'index': 0, 'function': {'arguments': '{"arg": 1}'}},
                  {'index': 1, 'function': {'arguments': '{"arg": 2}'}},
              ]
          },
          'finish_reason': None,
      }]
  }
  chunk_data_3 = {
      'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]
  }

  chunks = [
      f'{json.dumps(chunk_data_1)}\n',
      f'{json.dumps(chunk_data_2)}\n',
      f'{json.dumps(chunk_data_3)}\n',
  ]

  async def mock_aiter_lines():
    for chunk in chunks:
      yield chunk

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.aiter_lines.return_value = mock_aiter_lines()
  mock_response.status_code = 200

  mock_stream_ctx = mock.AsyncMock()
  mock_stream_ctx.__aenter__.return_value = mock_response

  with mock.patch.object(
      httpx.AsyncClient, 'stream', return_value=mock_stream_ctx
  ):
    responses = [
        r
        async for r in local_client.generate_content_async(
            llm_request, stream=True
        )
    ]

    assert len(responses) == 4
    parts = responses[-1].content.parts
    assert len(parts) == 2

    assert parts[0].function_call.name == 'func_1'
    assert parts[0].function_call.args == {'arg': 1}
    assert parts[0].function_call.id == 'call_1'

    assert parts[1].function_call.name == 'func_2'
    assert parts[1].function_call.args == {'arg': 2}

    assert parts[1].function_call.id == 'call_2'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('chunks', 'expected_response_count'),
    [
        (
            [
                '\n',
                '   \n',
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "Hello"}, "finish_reason": null}]}\n'
                ),
            ],
            1,
        ),
        (
            [
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "Hello"}, "finish_reason": null}]}\n'
                ),
                '[DONE]\n',
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "World"}, "finish_reason": "stop"}]}\n'
                ),
            ],
            1,  # Should stop after [DONE]
        ),
        (
            [
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "Hello"}, "finish_reason": null}]}\n'
                ),
                '   [DONE]   \n',
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "World"}, "finish_reason": "stop"}]}\n'
                ),
            ],
            1,  # Should stop after [DONE]
        ),
        (
            [
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "Hello"}, "finish_reason": null}]}\n'
                ),
                'data: [DONE]\n',
                (
                    'data: {"choices": [{"index": 0, "delta": {"content":'
                    ' "World"}, "finish_reason": "stop"}]}\n'
                ),
            ],
            1,  # Should stop after [DONE]
        ),
    ],
)
async def test_generate_content_async_streaming_parse_lines(
    chunks, expected_response_count
):
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  llm_request = LlmRequest(
      model='apigee/test',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
  )

  async def mock_aiter_lines():
    for chunk in chunks:
      yield chunk

  mock_response = AsyncMock(spec=httpx.Response)
  mock_response.aiter_lines.return_value = mock_aiter_lines()
  mock_response.status_code = 200

  mock_stream_ctx = mock.AsyncMock()
  mock_stream_ctx.__aenter__.return_value = mock_response

  with mock.patch.object(
      httpx.AsyncClient, 'stream', return_value=mock_stream_ctx
  ):
    responses = [
        r
        async for r in local_client.generate_content_async(
            llm_request, stream=True
        )
    ]
    assert len(responses) == expected_response_count
    assert responses[0].content.parts[0].text == 'Hello'

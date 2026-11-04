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

import asyncio
import atexit
import gc
import json
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock
import weakref

from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.apigee_llm import ChatCompletionsResponseHandler
from google.adk.models.apigee_llm import CompletionsHTTPClient
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
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
                  'function': {'arguments': '{"location": "London", '},
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
                  'function': {'arguments': ' "country": "UK"}'},
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

    # Check 2nd response: partial args for first update
    assert responses[1].partial is True
    assert responses[1].content.parts[0].function_call.id == 'call_123'
    assert responses[1].content.parts[0].function_call.name == 'get_weather'
    assert responses[1].content.parts[0].function_call.args is None
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].json_path
        == '$.location'
    )
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].string_value
        == 'London'
    )

    # Check 3rd response: partial args for second update
    assert responses[2].partial is True
    assert responses[2].content.parts[0].function_call.id == 'call_123'
    assert responses[2].content.parts[0].function_call.name == 'get_weather'
    assert responses[2].content.parts[0].function_call.args is None
    assert (
        responses[2].content.parts[0].function_call.partial_args[0].json_path
        == '$.country'
    )
    assert (
        responses[2].content.parts[0].function_call.partial_args[0].string_value
        == 'UK'
    )

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
async def test_generate_content_async_streaming_function_call_complete_json_deltas():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  llm_request = LlmRequest(
      model='apigee/test',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
  )

  # Mock chunks simulating complete JSON objects in arguments
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

    # Check 2nd response: partial args for first update
    assert responses[1].partial is True
    assert responses[1].content.parts[0].function_call.id == 'call_123'
    assert responses[1].content.parts[0].function_call.name == 'get_weather'
    assert responses[1].content.parts[0].function_call.args is None
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].json_path
        == '$.location'
    )
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].string_value
        == 'London'
    )

    # Check 3rd response: partial args for second update
    assert responses[2].partial is True
    assert responses[2].content.parts[0].function_call.id == 'call_123'
    assert responses[2].content.parts[0].function_call.name == 'get_weather'
    assert responses[2].content.parts[0].function_call.args is None
    assert (
        responses[2].content.parts[0].function_call.partial_args[0].json_path
        == '$.country'
    )
    assert (
        responses[2].content.parts[0].function_call.partial_args[0].string_value
        == 'UK'
    )

    # Check 4th response: last delta (empty)
    assert responses[3].partial is True
    assert responses[3].content.parts == []

    # Check 5th response: final accumulated, merged args
    assert responses[4].finish_reason == types.FinishReason.STOP
    assert responses[4].content.parts[0].function_call.args == {
        'location': 'London',
        'country': 'UK',
    }


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

    # Check 1st response: 2 partial tool calls, empty args
    assert responses[0].partial is True
    assert responses[0].content.parts[0].function_call.id == 'call_1'
    assert responses[0].content.parts[0].function_call.name == 'func_1'
    assert responses[0].content.parts[1].function_call.id == 'call_2'
    assert responses[0].content.parts[1].function_call.name == 'func_2'

    # Check 2nd response: 2 partial tool calls, with partial args.
    # verify id and name are carried over.
    assert responses[1].partial is True
    assert responses[1].content.parts[0].function_call.id == 'call_1'
    assert responses[1].content.parts[0].function_call.name == 'func_1'
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].json_path
        == '$.arg'
    )
    assert (
        responses[1].content.parts[0].function_call.partial_args[0].number_value
        == 1
    )
    assert responses[1].content.parts[1].function_call.id == 'call_2'
    assert responses[1].content.parts[1].function_call.name == 'func_2'
    assert (
        responses[1].content.parts[1].function_call.partial_args[0].json_path
        == '$.arg'
    )
    assert (
        responses[1].content.parts[1].function_call.partial_args[0].number_value
        == 2
    )

    parts = responses[-1].content.parts
    assert len(parts) == 2

    assert parts[0].function_call.name == 'func_1'
    assert parts[0].function_call.args == {'arg': 1}
    assert parts[0].function_call.id == 'call_1'

    assert parts[1].function_call.name == 'func_2'
    assert parts[1].function_call.args == {'arg': 2}
    assert parts[1].function_call.id == 'call_2'


@pytest.mark.asyncio
async def test_generate_content_async_streaming_incomplete_stream_no_finish_reason():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  llm_request = LlmRequest(
      model='apigee/test',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
  )
  chunk_data = {
      'choices': [{
          'index': 0,
          'delta': {
              'tool_calls': [{
                  'index': 0,
                  'id': 'call_1',
                  'type': 'function',
                  'function': {'name': 'func_1', 'arguments': '{"arg": 1}'},
              }]
          },
          'finish_reason': None,
      }]
  }

  chunks = [f'{json.dumps(chunk_data)}\n']

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

    assert len(responses) == 1
    assert responses[0].partial is True
    assert responses[0].finish_reason is None


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


def test_process_chunk_with_refusal_streaming():
  handler = ChatCompletionsResponseHandler()

  chunk1 = {
      'choices': [{
          'delta': {
              'role': 'assistant',
              'content': 'Hello',
          },
          'index': 0,
      }]
  }
  responses1 = list(handler.process_chunk(chunk1))
  assert len(responses1) == 1
  assert responses1[0].content.parts[0].text == 'Hello'

  chunk2 = {
      'choices': [{
          'delta': {
              'refusal': 'I refuse',
          },
          'index': 0,
      }]
  }
  responses2 = list(handler.process_chunk(chunk2))
  assert len(responses2) == 1
  assert responses2[0].content.parts[0].text == '\n[[REFUSAL]]: I refuse'

  chunk3 = {
      'choices': [{
          'delta': {
              'refusal': ' to answer',
          },
          'index': 0,
      }]
  }
  responses3 = list(handler.process_chunk(chunk3))
  assert len(responses3) == 1
  assert responses3[0].content.parts[0].text == ' to answer'

  chunk4 = {
      'choices': [{
          'delta': {},
          'finish_reason': 'stop',
          'index': 0,
      }]
  }
  responses4 = list(handler.process_chunk(chunk4))
  assert len(responses4) == 2
  final_response = responses4[1]
  assert final_response.finish_reason == types.FinishReason.STOP
  assert (
      final_response.content.parts[0].text
      == 'Hello\n[[REFUSAL]]: I refuse to answer'
  )


def test_client_creation_registers_atexit_cleanup():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  with mock.patch(
      'google.adk.models.apigee_llm.atexit.register'
  ) as mock_register:
    _ = local_client._client
    mock_register.assert_called_once_with(local_client._atexit_callback)


def _unregister_calls_for(mock_unregister, callback):
  # Other clients can be garbage collected while the global function is patched.
  return [
      call
      for call in mock_unregister.call_args_list
      if call.args and call.args[0] is callback
  ]


def test_close_unregisters_atexit_cleanup():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  _ = local_client._client
  callback = local_client._atexit_callback
  with mock.patch(
      'google.adk.models.apigee_llm.atexit.unregister', wraps=atexit.unregister
  ) as mock_unregister:
    local_client.close()
    assert _unregister_calls_for(mock_unregister, callback) == [
        mock.call(callback)
    ]


@pytest.mark.asyncio
async def test_aclose_unregisters_atexit_cleanup():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  _ = local_client._client
  callback = local_client._atexit_callback
  with mock.patch(
      'google.adk.models.apigee_llm.atexit.unregister', wraps=atexit.unregister
  ) as mock_unregister:
    await local_client.aclose()
    assert _unregister_calls_for(mock_unregister, callback) == [
        mock.call(callback)
    ]
  assert local_client._client.is_closed


@pytest.mark.asyncio
async def test_aclose_preserves_live_peer_during_other_client_collection():
  """Closing one client preserves another client's pool and exit handler."""
  live_peer = CompletionsHTTPClient(base_url='https://localhost')
  live_httpx_client = live_peer._client
  live_callback = live_peer._atexit_callback

  old_client = CompletionsHTTPClient(base_url='https://localhost')
  _ = old_client._client
  old_callback = old_client._atexit_callback
  old_client._cycle = old_client
  old_ref = weakref.ref(old_client)

  local_client = CompletionsHTTPClient(base_url='https://localhost')
  _ = local_client._client
  callback = local_client._atexit_callback
  with mock.patch(
      'google.adk.models.apigee_llm.atexit.unregister', wraps=atexit.unregister
  ) as mock_unregister:
    del old_client
    gc.collect()
    await local_client.aclose()

    assert old_ref() is None
    assert _unregister_calls_for(mock_unregister, old_callback) == [
        mock.call(old_callback)
    ]
    assert _unregister_calls_for(mock_unregister, callback) == [
        mock.call(callback)
    ]
    assert _unregister_calls_for(mock_unregister, live_callback) == []
    assert not live_httpx_client.is_closed

  await live_peer.aclose()
  # Let the background aclose() task scheduled by old_client.__del__() during
  # gc.collect() run before the test's event loop closes.
  await asyncio.sleep(0)


def test_close_without_client_does_not_touch_atexit():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  with mock.patch(
      'google.adk.models.apigee_llm.atexit.unregister',
      wraps=atexit.unregister,
  ) as mock_unregister:
    local_client.close()
    assert (
        _unregister_calls_for(
            mock_unregister, getattr(local_client, '_atexit_callback', None)
        )
        == []
    )


def test_atexit_registration_does_not_pin_client():
  # The atexit handler is registered against a weakref.proxy, so the registry
  # must not keep the underlying httpx client alive after it is closed and its
  # owner is dropped. Regression test for the leaked AsyncClient / connection
  # pool that grew unbounded in long-running deployments.
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  httpx_client = local_client._client
  client_ref = weakref.ref(httpx_client)

  local_client.close()
  del httpx_client
  del local_client
  gc.collect()

  assert client_ref() is None


def test_cleanup_client_tolerates_dead_weakref_proxy():
  local_client = CompletionsHTTPClient(base_url='https://localhost')
  httpx_client = local_client._client
  proxy = weakref.proxy(httpx_client)
  del httpx_client
  del local_client
  gc.collect()

  # The proxy now points at a collected object; cleanup must not raise.
  CompletionsHTTPClient._cleanup_client(proxy)


@pytest.mark.asyncio
async def test_cleanup_client_retains_pending_task():
  from google.adk.models import apigee_llm

  local_client = CompletionsHTTPClient(base_url='https://localhost')
  httpx_client = local_client._client

  CompletionsHTTPClient._cleanup_client(httpx_client)

  # While the aclose() task is pending it must be held in the module-level set
  # so it is not garbage collected before it completes.
  client_tasks = [
      t
      for t in apigee_llm._CLEANUP_TASKS
      if getattr(t.get_coro(), 'cr_frame', None) is not None
      and t.get_coro().cr_frame.f_locals.get('self') is httpx_client
  ]
  assert len(client_tasks) == 1
  task = client_tasks[0]
  await task
  assert task not in apigee_llm._CLEANUP_TASKS
  assert httpx_client.is_closed


def _tool_argument_chunk(
    arguments: str | None, index: int = 0
) -> dict[str, Any]:
  function = {'name': 'weather'}
  if arguments is not None:
    function['arguments'] = arguments
  return {
      'choices': [{
          'delta': {
              'role': 'assistant',
              'tool_calls': [{
                  'index': index,
                  'id': f'call_{index}',
                  'type': 'function',
                  'function': function,
              }],
          }
      }]
  }


@pytest.mark.parametrize(
    'arguments',
    [
        ['{"city": "Par'],
        [''],
        [None],
        ['   '],
        ['{"city": "Paris"}', '{"country":'],
        ['{"country":', '{"city": "Paris"}'],
        ['[1, 2]'],
        ['null'],
    ],
)
def test_length_with_incomplete_tool_arguments_reports_error(
    arguments: list[str | None],
) -> None:
  """Token exhaustion never exposes a final call with incomplete arguments."""
  handler = ChatCompletionsResponseHandler()
  responses: list[LlmResponse] = []
  for fragment in arguments:
    responses.extend(handler.process_chunk(_tool_argument_chunk(fragment)))
  responses.extend(
      handler.process_chunk({
          'choices': [{'delta': {}, 'finish_reason': 'length'}],
          'usage': {
              'prompt_tokens': 3,
              'completion_tokens': 4,
              'total_tokens': 7,
          },
      })
  )

  assert len(responses) == len(arguments) + 2
  assert all(r.partial is True for r in responses[:-1])
  for response in responses[: len(arguments)]:
    (call,) = response.get_function_calls()
    assert call.args is None
    assert call.will_continue is True
  assert responses[-2].content is not None
  assert responses[-2].content.parts == []
  final = responses[-1]
  assert not final.partial
  assert final.content is None
  assert final.finish_reason == types.FinishReason.MAX_TOKENS
  assert final.error_code == types.FinishReason.MAX_TOKENS
  assert final.usage_metadata is not None
  assert final.usage_metadata.total_token_count == 7
  assert final.error_message is not None
  assert 'truncated' in final.error_message
  assert 'max_output_tokens' in final.error_message


@pytest.mark.parametrize('incomplete_index', [0, 2])
def test_length_drops_entire_interleaved_tool_call_batch(
    incomplete_index: int,
) -> None:
  """A good call must not escape when another indexed call is incomplete."""
  handler = ChatCompletionsResponseHandler()
  responses: list[LlmResponse] = []
  for index in (0, 2):
    responses.extend(
        handler.process_chunk(_tool_argument_chunk('{"city":', index))
    )
  for index in (2, 0):
    arguments = '"Par' if index == incomplete_index else '"Paris"}'
    responses.extend(
        handler.process_chunk(_tool_argument_chunk(arguments, index))
    )
  responses.extend(
      handler.process_chunk(
          {'choices': [{'delta': {}, 'finish_reason': 'length'}]}
      )
  )

  assert [r.get_function_calls()[0].id for r in responses[:4]] == [
      'call_0',
      'call_2',
      'call_2',
      'call_0',
  ]
  assert all(r.partial is True for r in responses[:-1])
  assert responses[-1].content is None
  assert responses[-1].error_code == types.FinishReason.MAX_TOKENS
  assert responses[-1].finish_reason == types.FinishReason.MAX_TOKENS


@pytest.mark.parametrize('finish_reason', ['length', 'tool_calls', 'stop'])
@pytest.mark.parametrize(
    ('arguments', 'expected_args'),
    [
        (['{}'], {}),
        (['{"city":', '"Paris"}'], {'city': 'Paris'}),
        (
            ['{"city":"Paris"}', '{"country":"FR"}'],
            {'city': 'Paris', 'country': 'FR'},
        ),
    ],
)
def test_completed_streaming_tool_arguments_are_preserved(
    finish_reason: str, arguments: list[str], expected_args: dict[str, str]
) -> None:
  """Complete arguments, including complete JSON deltas, stay executable."""
  handler = ChatCompletionsResponseHandler()
  responses: list[LlmResponse] = []
  for fragment in arguments:
    responses.extend(handler.process_chunk(_tool_argument_chunk(fragment)))
  responses.extend(
      handler.process_chunk(
          {'choices': [{'delta': {}, 'finish_reason': finish_reason}]}
      )
  )

  assert len(responses) == len(arguments) + 2
  assert all(r.partial is True for r in responses[:-1])
  final = responses[-1]
  assert not final.partial
  assert final.error_code is None
  assert final.error_message is None
  assert final.finish_reason == (
      types.FinishReason.MAX_TOKENS
      if finish_reason == 'length'
      else types.FinishReason.STOP
  )
  assert final.content is not None
  assert final.content.parts is not None
  assert len(final.content.parts) == 1
  (call,) = final.get_function_calls()
  assert call.args == expected_args
  assert call.id == 'call_0'
  assert call.name == 'weather'
  assert not call.will_continue


@pytest.mark.parametrize('finish_reason', ['tool_calls', 'stop', 'unknown'])
def test_non_length_malformed_tool_arguments_still_raise(
    finish_reason: str,
) -> None:
  """The token-limit guard does not change other termination paths."""
  handler = ChatCompletionsResponseHandler()
  list(handler.process_chunk(_tool_argument_chunk('{"city": "Par')))
  terminal = handler.process_chunk(
      {'choices': [{'delta': {}, 'finish_reason': finish_reason}]}
  )
  assert next(terminal).partial is True
  with pytest.raises(ValueError, match='Invalid JSON in tool call arguments'):
    list(terminal)


@pytest.mark.parametrize('finish_reason', ['tool_calls', 'stop', 'unknown'])
def test_non_length_argument_merge_fallback_is_unchanged(
    finish_reason: str,
) -> None:
  """Only length termination rejects a bad delta in the legacy fallback."""
  handler = ChatCompletionsResponseHandler()
  for fragment in ['{"city":"Paris"}', '{"country":']:
    list(handler.process_chunk(_tool_argument_chunk(fragment)))
  responses = list(
      handler.process_chunk(
          {'choices': [{'delta': {}, 'finish_reason': finish_reason}]}
      )
  )
  assert responses[0].partial is True
  (call,) = responses[-1].get_function_calls()
  assert call.args == {'city': 'Paris'}
  assert responses[-1].error_code is None


@pytest.mark.parametrize(
    ('finish_reason', 'expected'),
    [
        ('length', types.FinishReason.MAX_TOKENS),
        ('stop', types.FinishReason.STOP),
    ],
)
def test_text_only_streaming_finish_behavior_is_unchanged(
    finish_reason: str, expected: types.FinishReason
) -> None:
  """Text without tool calls keeps the existing partial and final responses."""
  handler = ChatCompletionsResponseHandler()
  chunks: list[dict[str, Any]] = [
      {'choices': [{'delta': {'content': 'Hello'}}]},
      {'choices': [{'delta': {}, 'finish_reason': finish_reason}]},
  ]
  responses = [r for chunk in chunks for r in handler.process_chunk(chunk)]
  assert len(responses) == 3
  assert responses[0].partial is True
  assert responses[1].partial is True
  assert not responses[-1].partial
  assert responses[-1].content is not None
  assert responses[-1].content.parts == [types.Part.from_text(text='Hello')]
  assert responses[-1].finish_reason == expected
  assert responses[-1].error_code is None
  assert responses[-1].error_message is None


@pytest.mark.parametrize(
    'argument_deltas',
    [
        ['{"city": "Par'],
        [''],
        ['{"city": "Paris"}', '{"country":'],
    ],
)
@pytest.mark.parametrize('same_chunk', [False, True])
async def test_apigee_streaming_truncation_with_mocked_sse(
    same_chunk: bool, argument_deltas: list[str]
) -> None:
  """The adapter keeps text and metadata but emits no truncated final call."""
  chunks = [_tool_argument_chunk(fragment) for fragment in argument_deltas]
  chunks[0]['choices'][0]['delta']['content'] = 'Checking weather.'
  if same_chunk:
    chunks[-1]['choices'][0]['finish_reason'] = 'length'
  else:
    chunks.append({'choices': [{'delta': {}, 'finish_reason': 'length'}]})
  chunks[-1].update(
      model='gpt-4o',
      id='completion_1',
      usage={'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7},
  )
  chunks[-1]['choices'][0]['logprobs'] = {
      'content': [{'token': 'Checking', 'logprob': -0.1, 'top_logprobs': []}]
  }
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    body = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks)
    return httpx.Response(
        200,
        headers={'Content-Type': 'text/event-stream'},
        content=body + 'data: [DONE]\n\n',
    )

  llm = ApigeeLlm(
      model='apigee/openai/gpt-4o', proxy_url='https://apigee.test/v1'
  )
  async with httpx.AsyncClient(
      transport=httpx.MockTransport(respond)
  ) as client:
    with mock.patch(
        'google.adk.models.apigee_llm.httpx.AsyncClient', return_value=client
    ):
      responses = [
          r
          async for r in llm.generate_content_async(
              LlmRequest(
                  model=llm.model,
                  contents=[
                      types.Content(
                          role='user',
                          parts=[types.Part.from_text(text='Weather?')],
                      )
                  ],
              ),
              stream=True,
          )
      ]
      await llm._completions_http_client.aclose()

  assert len(requests) == 1
  assert requests[0].method == 'POST'
  assert str(requests[0].url) == 'https://apigee.test/v1/chat/completions'
  assert json.loads(requests[0].content)['stream'] is True
  assert len(responses) == len(chunks) + 2
  assert all(r.partial is True for r in responses[:-2])
  text, error = responses[-2:]
  assert not text.partial and not error.partial
  assert text.content is not None
  assert text.content.parts == [types.Part.from_text(text='Checking weather.')]
  assert text.error_message == 'Maximum tokens reached'
  assert error.content is None
  assert error.error_message is not None
  assert 'truncated' in error.error_message
  assert text.usage_metadata is not None
  assert text.usage_metadata.total_token_count == 7
  assert error.usage_metadata is None
  assert (
      sum(
          response.usage_metadata.total_token_count or 0
          for response in responses
          if not response.partial and response.usage_metadata
      )
      == 7
  )
  for response in (text, error):
    assert response.finish_reason == types.FinishReason.MAX_TOKENS
    assert response.error_code == types.FinishReason.MAX_TOKENS
    assert response.model_version == 'gpt-4o'
    assert response.custom_metadata is not None
    assert response.custom_metadata['id'] == 'completion_1'
    assert response.logprobs_result is not None
    assert response.logprobs_result.chosen_candidates is not None
    assert response.logprobs_result.chosen_candidates[0].token == 'Checking'
    assert not response.get_function_calls()


def _non_streaming_tool_response(
    arguments: list[str | None],
    finish_reason: str = 'length',
    text: str | None = None,
) -> dict[str, Any]:
  calls = []
  for index, args in enumerate(arguments):
    call = _tool_argument_chunk(args, index)['choices'][0]['delta'][
        'tool_calls'
    ][0]
    calls.append(call)
  return {
      'id': 'completion_1',
      'model': 'gpt-4o',
      'choices': [{
          'finish_reason': finish_reason,
          'message': {
              'role': 'assistant',
              'content': text,
              'tool_calls': calls,
          },
          'logprobs': {
              'content': [
                  {'token': 'Checking', 'logprob': -0.1, 'top_logprobs': []}
              ]
          },
      }],
      'usage': {
          'prompt_tokens': 3,
          'completion_tokens': 4,
          'total_tokens': 7,
          'completion_tokens_details': {'reasoning_tokens': 2},
      },
  }


@pytest.mark.parametrize(
    'arguments', ['{"city": "Pa', '', None, '   ', '[1, 2]', 'null']
)
def test_non_streaming_length_incomplete_arguments_report_error(
    arguments: str | None,
) -> None:
  """Token-limited complete responses do not expose incomplete calls."""
  response = ChatCompletionsResponseHandler().process_response(
      _non_streaming_tool_response([arguments])
  )

  assert not response.partial
  assert response.content is None
  assert response.finish_reason == types.FinishReason.MAX_TOKENS
  assert response.error_code == types.FinishReason.MAX_TOKENS
  assert response.error_message is not None
  assert 'truncated' in response.error_message
  assert not response.get_function_calls()


@pytest.mark.parametrize('legacy', [False, True])
def test_non_streaming_length_preserves_text_and_metadata(legacy: bool) -> None:
  """Dropping modern or legacy calls preserves text, usage, and metadata."""
  payload = _non_streaming_tool_response(['{"city": "Pa'], text='Checking.')
  message = payload['choices'][0]['message']
  if legacy:
    message['function_call'] = message.pop('tool_calls')[0]['function']
  response = ChatCompletionsResponseHandler().process_response(payload)

  assert response.content is not None
  assert response.content.parts == [types.Part.from_text(text='Checking.')]
  assert response.content.role == 'model'
  assert not response.get_function_calls()
  assert response.error_code == types.FinishReason.MAX_TOKENS
  assert response.model_version == 'gpt-4o'
  assert response.custom_metadata is not None
  assert response.custom_metadata['id'] == 'completion_1'
  assert response.logprobs_result is not None
  assert response.logprobs_result.chosen_candidates is not None
  assert response.logprobs_result.chosen_candidates[0].token == 'Checking'
  assert response.usage_metadata is not None
  assert response.usage_metadata.prompt_token_count == 3
  assert response.usage_metadata.candidates_token_count == 4
  assert response.usage_metadata.total_token_count == 7
  assert response.usage_metadata.thoughts_token_count == 2
  # The provider payload must remain intact for callers that inspect it.
  assert message.get('tool_calls') or message.get('function_call')


@pytest.mark.parametrize('bad_index', [0, 1])
def test_non_streaming_length_drops_entire_batch(bad_index: int) -> None:
  """A valid call never escapes alongside an incomplete one."""
  arguments: list[str | None] = ['{"city":"Paris"}', '{"city":"Paris"}']
  arguments[bad_index] = '{"city":"Pa'
  response = ChatCompletionsResponseHandler().process_response(
      _non_streaming_tool_response(arguments)
  )

  assert response.error_code == types.FinishReason.MAX_TOKENS
  assert not response.get_function_calls()


@pytest.mark.parametrize('arguments', ['{}', '{"city":"Paris"}'])
@pytest.mark.parametrize('finish_reason', ['length', 'tool_calls', 'stop'])
def test_non_streaming_complete_arguments_preserved(
    arguments: str, finish_reason: str
) -> None:
  """Complete arguments stay executable, even when the token limit is hit."""
  response = ChatCompletionsResponseHandler().process_response(
      _non_streaming_tool_response([arguments], finish_reason)
  )

  assert response.error_code is None
  assert response.finish_reason == (
      types.FinishReason.MAX_TOKENS
      if finish_reason == 'length'
      else types.FinishReason.STOP
  )
  assert response.get_function_calls()[0].args == json.loads(arguments)
  assert response.get_function_calls()[0].name == 'weather'
  assert response.get_function_calls()[0].id == 'call_0'


@pytest.mark.parametrize('finish_reason', ['tool_calls', 'stop', 'unknown'])
def test_non_streaming_non_length_malformed_arguments_still_raise(
    finish_reason: str,
) -> None:
  """Only explicit token exhaustion changes argument parsing failures."""
  with pytest.raises(ValueError, match='Invalid JSON in tool call arguments'):
    ChatCompletionsResponseHandler().process_response(
        _non_streaming_tool_response(['{"city":"Pa'], finish_reason)
    )


def test_non_streaming_length_unsupported_call_type_still_raises() -> None:
  """A token limit does not suppress unsupported-type ValueErrors."""
  payload = _non_streaming_tool_response(['{"city":"Pa'])
  payload['choices'][0]['message']['tool_calls'][0]['type'] = 'custom'
  with pytest.raises(ValueError, match='Unsupported tool_call type'):
    ChatCompletionsResponseHandler().process_response(payload)


def test_non_streaming_length_invalid_signature_still_raises() -> None:
  """Non-JSON ValueErrors remain visible even with a length finish reason."""
  payload = _non_streaming_tool_response(['{}'])
  payload['choices'][0]['message']['tool_calls'][0]['extra_content'] = {
      'google': {'thought_signature': 'a'}
  }
  with pytest.raises(ValueError, match='Invalid base64'):
    ChatCompletionsResponseHandler().process_response(payload)


@pytest.mark.asyncio
async def test_apigee_non_streaming_truncation_with_mocked_transport() -> None:
  """The public non-streaming adapter emits one complete token-limit error."""
  payload = _non_streaming_tool_response(['{"city":"Pa'], text='Checking.')
  requests: list[httpx.Request] = []

  def respond(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return httpx.Response(200, json=payload)

  llm = ApigeeLlm(
      model='apigee/openai/gpt-4o', proxy_url='https://apigee.test/v1'
  )
  async with httpx.AsyncClient(
      transport=httpx.MockTransport(respond)
  ) as client:
    with mock.patch(
        'google.adk.models.apigee_llm.httpx.AsyncClient', return_value=client
    ):
      responses = [
          response
          async for response in llm.generate_content_async(
              LlmRequest(
                  model=llm.model,
                  contents=[
                      types.Content(
                          role='user',
                          parts=[types.Part.from_text(text='Weather?')],
                      )
                  ],
              ),
              stream=False,
          )
      ]
      await llm._completions_http_client.aclose()

  assert len(responses) == 1
  response = responses[0]
  assert json.loads(requests[0].content)['stream'] is False
  assert response.finish_reason == types.FinishReason.MAX_TOKENS
  assert response.error_code == types.FinishReason.MAX_TOKENS
  assert response.content is not None
  assert response.content.parts == [types.Part.from_text(text='Checking.')]
  assert not response.get_function_calls()
  assert response.usage_metadata is not None
  assert response.usage_metadata.total_token_count == 7
  assert response.model_version == 'gpt-4o'

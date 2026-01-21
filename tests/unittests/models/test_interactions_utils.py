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

"""Tests for interactions_utils.py conversion functions."""

import json
from unittest.mock import MagicMock

from google.adk.models import interactions_utils
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


class TestConvertPartToInteractionContent:
  """Tests for convert_part_to_interaction_content."""

  def test_text_part(self):
    """Test converting a text Part."""
    part = types.Part(text='Hello, world!')
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {'type': 'text', 'text': 'Hello, world!'}

  def test_function_call_part(self):
    """Test converting a function call Part."""
    part = types.Part(
        function_call=types.FunctionCall(
            id='call_123',
            name='get_weather',
            args={'city': 'London'},
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'function_call',
        'id': 'call_123',
        'name': 'get_weather',
        'arguments': {'city': 'London'},
    }

  def test_function_call_part_no_id(self):
    """Test converting a function call Part without id."""
    part = types.Part(
        function_call=types.FunctionCall(
            name='get_weather',
            args={'city': 'London'},
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result['id'] == ''
    assert result['name'] == 'get_weather'

  def test_function_response_dict(self):
    """Test converting a function response Part with dict response."""
    part = types.Part(
        function_response=types.FunctionResponse(
            id='call_123',
            name='get_weather',
            response={'temperature': 20, 'condition': 'sunny'},
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result['type'] == 'function_result'
    assert result['call_id'] == 'call_123'
    assert result['name'] == 'get_weather'
    # Dict should be JSON serialized
    assert json.loads(result['result']) == {
        'temperature': 20,
        'condition': 'sunny',
    }

  def test_function_response_simple(self):
    """Test converting a function response Part with simple response."""
    part = types.Part(
        function_response=types.FunctionResponse(
            id='call_123',
            name='check_weather',
            response={'message': 'Weather is sunny'},
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result['type'] == 'function_result'
    assert result['call_id'] == 'call_123'
    assert result['name'] == 'check_weather'
    # Dict should be JSON serialized
    assert json.loads(result['result']) == {'message': 'Weather is sunny'}

  def test_inline_data_image(self):
    """Test converting an inline image Part."""
    part = types.Part(
        inline_data=types.Blob(
            data=b'image_data',
            mime_type='image/png',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'image',
        'data': b'image_data',
        'mime_type': 'image/png',
    }

  def test_inline_data_audio(self):
    """Test converting an inline audio Part."""
    part = types.Part(
        inline_data=types.Blob(
            data=b'audio_data',
            mime_type='audio/mp3',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'audio',
        'data': b'audio_data',
        'mime_type': 'audio/mp3',
    }

  def test_inline_data_video(self):
    """Test converting an inline video Part."""
    part = types.Part(
        inline_data=types.Blob(
            data=b'video_data',
            mime_type='video/mp4',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'video',
        'data': b'video_data',
        'mime_type': 'video/mp4',
    }

  def test_inline_data_document(self):
    """Test converting an inline document Part."""
    part = types.Part(
        inline_data=types.Blob(
            data=b'doc_data',
            mime_type='application/pdf',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'document',
        'data': b'doc_data',
        'mime_type': 'application/pdf',
    }

  def test_file_data_image(self):
    """Test converting a file data image Part."""
    part = types.Part(
        file_data=types.FileData(
            file_uri='gs://bucket/image.png',
            mime_type='image/png',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'image',
        'uri': 'gs://bucket/image.png',
        'mime_type': 'image/png',
    }

  def test_text_with_thought_flag(self):
    """Test converting a text Part with thought=True flag."""
    # In types.Part, thought is a boolean flag on text content
    # When text is present, the convert function returns text type (not thought)
    # because text check comes before thought check in the implementation
    part = types.Part(text='Let me think about this...', thought=True)
    result = interactions_utils.convert_part_to_interaction_content(part)
    # Text content is returned as-is (thought flag not represented in output)
    assert result == {'type': 'text', 'text': 'Let me think about this...'}

  def test_thought_only_part(self):
    """Test converting a thought-only Part with signature."""
    import base64

    signature_bytes = b'test-thought-signature'
    part = types.Part(thought=True, thought_signature=signature_bytes)
    result = interactions_utils.convert_part_to_interaction_content(part)
    expected_signature = base64.b64encode(signature_bytes).decode('utf-8')
    assert result == {'type': 'thought', 'signature': expected_signature}

  def test_thought_only_part_without_signature(self):
    """Test converting a thought-only Part without signature."""
    part = types.Part(thought=True)
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {'type': 'thought'}

  def test_code_execution_result(self):
    """Test converting a code execution result Part."""
    part = types.Part(
        code_execution_result=types.CodeExecutionResult(
            output='Hello from code',
            outcome=types.Outcome.OUTCOME_OK,
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'code_execution_result',
        'call_id': '',
        'result': 'Hello from code',
        'is_error': False,
    }

  def test_code_execution_result_with_error(self):
    """Test converting a failed code execution result Part."""
    part = types.Part(
        code_execution_result=types.CodeExecutionResult(
            output='Error: something went wrong',
            outcome=types.Outcome.OUTCOME_FAILED,
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'code_execution_result',
        'call_id': '',
        'result': 'Error: something went wrong',
        'is_error': True,
    }

  def test_code_execution_result_deadline_exceeded(self):
    """Test converting a deadline exceeded code execution result Part."""
    part = types.Part(
        code_execution_result=types.CodeExecutionResult(
            output='Timeout',
            outcome=types.Outcome.OUTCOME_DEADLINE_EXCEEDED,
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'code_execution_result',
        'call_id': '',
        'result': 'Timeout',
        'is_error': True,
    }

  def test_executable_code(self):
    """Test converting an executable code Part."""
    part = types.Part(
        executable_code=types.ExecutableCode(
            code='print("hello")',
            language='PYTHON',
        )
    )
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result == {
        'type': 'code_execution_call',
        'id': '',
        'arguments': {
            'code': 'print("hello")',
            'language': 'PYTHON',
        },
    }

  def test_empty_part(self):
    """Test converting an empty Part returns None."""
    part = types.Part()
    result = interactions_utils.convert_part_to_interaction_content(part)
    assert result is None


class TestConvertContentToTurn:
  """Tests for convert_content_to_turn."""

  def test_user_content(self):
    """Test converting user content."""
    content = types.Content(
        role='user',
        parts=[types.Part(text='Hello!')],
    )
    result = interactions_utils.convert_content_to_turn(content)
    assert result == {
        'role': 'user',
        'content': [{'type': 'text', 'text': 'Hello!'}],
    }

  def test_model_content(self):
    """Test converting model content."""
    content = types.Content(
        role='model',
        parts=[types.Part(text='Hi there!')],
    )
    result = interactions_utils.convert_content_to_turn(content)
    assert result == {
        'role': 'model',
        'content': [{'type': 'text', 'text': 'Hi there!'}],
    }

  def test_multiple_parts(self):
    """Test converting content with multiple parts."""
    content = types.Content(
        role='user',
        parts=[
            types.Part(text='Look at this:'),
            types.Part(
                inline_data=types.Blob(data=b'img', mime_type='image/png')
            ),
        ],
    )
    result = interactions_utils.convert_content_to_turn(content)
    assert result['role'] == 'user'
    assert len(result['content']) == 2
    assert result['content'][0] == {'type': 'text', 'text': 'Look at this:'}
    assert result['content'][1]['type'] == 'image'

  def test_default_role(self):
    """Test that default role is 'user' when not specified."""
    content = types.Content(parts=[types.Part(text='Hi')])
    result = interactions_utils.convert_content_to_turn(content)
    assert result['role'] == 'user'


class TestConvertContentsToTurns:
  """Tests for convert_contents_to_turns."""

  def test_single_content(self):
    """Test converting a list with single content."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='What is 2+2?')]),
    ]
    result = interactions_utils.convert_contents_to_turns(contents)
    assert len(result) == 1
    assert result[0]['role'] == 'user'
    assert result[0]['content'][0]['text'] == 'What is 2+2?'

  def test_multi_turn_conversation(self):
    """Test converting a multi-turn conversation."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='Hi')]),
        types.Content(role='model', parts=[types.Part(text='Hello!')]),
        types.Content(role='user', parts=[types.Part(text='How are you?')]),
    ]
    result = interactions_utils.convert_contents_to_turns(contents)
    assert len(result) == 3
    assert result[0]['role'] == 'user'
    assert result[1]['role'] == 'model'
    assert result[2]['role'] == 'user'

  def test_empty_content_skipped(self):
    """Test that empty contents are skipped."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='Hi')]),
        types.Content(role='model', parts=[]),  # Empty parts
    ]
    result = interactions_utils.convert_contents_to_turns(contents)
    # Only the first content should be included
    assert len(result) == 1


class TestConvertToolsConfig:
  """Tests for convert_tools_config_to_interactions_format."""

  def test_function_declaration(self):
    """Test converting function declarations."""
    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name='get_weather',
                        description='Get weather for a city',
                        parameters=types.Schema(
                            type='OBJECT',
                            properties={
                                'city': types.Schema(type='STRING'),
                            },
                            required=['city'],
                        ),
                    )
                ]
            )
        ]
    )
    result = interactions_utils.convert_tools_config_to_interactions_format(
        config
    )
    assert len(result) == 1
    assert result[0]['type'] == 'function'
    assert result[0]['name'] == 'get_weather'
    assert result[0]['description'] == 'Get weather for a city'
    assert 'parameters' in result[0]

  def test_google_search_tool(self):
    """Test converting google search tool."""
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    result = interactions_utils.convert_tools_config_to_interactions_format(
        config
    )
    assert result == [{'type': 'google_search'}]

  def test_code_execution_tool(self):
    """Test converting code execution tool."""
    config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )
    result = interactions_utils.convert_tools_config_to_interactions_format(
        config
    )
    assert result == [{'type': 'code_execution'}]

  def test_no_tools(self):
    """Test handling config with no tools."""
    config = types.GenerateContentConfig()
    result = interactions_utils.convert_tools_config_to_interactions_format(
        config
    )
    assert result == []


class TestConvertInteractionOutputToPart:
  """Tests for convert_interaction_output_to_part."""

  def test_text_output(self):
    """Test converting text output."""
    output = MagicMock()
    output.type = 'text'
    output.text = 'Hello!'
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.text == 'Hello!'

  def test_function_call_output(self):
    """Test converting function call output."""
    output = MagicMock()
    output.type = 'function_call'
    output.id = 'call_123'
    output.name = 'get_weather'
    output.arguments = {'city': 'London'}
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.function_call.id == 'call_123'
    assert result.function_call.name == 'get_weather'
    assert result.function_call.args == {'city': 'London'}

  def test_function_result_output_with_items_list(self):
    """Test converting function result output with items list.

    The implementation handles the case where result has an 'items' attribute
    that returns a list-like structure. This test validates that path.
    """
    output = MagicMock()
    output.type = 'function_result'
    output.call_id = 'call_123'
    # Create a mock that has .items returning a dict (for FunctionResponse)
    output.result = MagicMock()
    output.result.items = {'weather': 'Sunny'}  # items attribute returns dict
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.function_response.id == 'call_123'
    assert result.function_response.response == {'weather': 'Sunny'}

  def test_image_output_with_data(self):
    """Test converting image output with inline data."""
    output = MagicMock()
    output.type = 'image'
    output.data = b'image_bytes'
    output.uri = None
    output.mime_type = 'image/png'
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.inline_data.data == b'image_bytes'
    assert result.inline_data.mime_type == 'image/png'

  def test_image_output_with_uri(self):
    """Test converting image output with URI."""
    output = MagicMock()
    output.type = 'image'
    output.data = None
    output.uri = 'gs://bucket/image.png'
    output.mime_type = 'image/png'
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.file_data.file_uri == 'gs://bucket/image.png'
    assert result.file_data.mime_type == 'image/png'

  def test_code_execution_result_output(self):
    """Test converting code execution result output."""
    output = MagicMock()
    output.type = 'code_execution_result'
    output.result = 'Output from code'
    output.is_error = False  # Indicate successful execution
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.code_execution_result.output == 'Output from code'
    assert result.code_execution_result.outcome == types.Outcome.OUTCOME_OK

  def test_code_execution_result_error_output(self):
    """Test converting code execution result output with error."""
    output = MagicMock()
    output.type = 'code_execution_result'
    output.result = 'Error: division by zero'
    output.is_error = True  # Indicate failed execution
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result.code_execution_result.output == 'Error: division by zero'
    assert result.code_execution_result.outcome == types.Outcome.OUTCOME_FAILED

  def test_thought_output_returns_none(self):
    """Test that thought output returns None (not exposed as Part)."""
    output = MagicMock()
    output.type = 'thought'
    output.signature = 'thinking...'
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result is None

  def test_no_type_attribute(self):
    """Test handling output without type attribute."""
    output = MagicMock(spec=[])  # No 'type' attribute
    result = interactions_utils.convert_interaction_output_to_part(output)
    assert result is None


class TestConvertInteractionToLlmResponse:
  """Tests for convert_interaction_to_llm_response."""

  def test_successful_text_response(self):
    """Test converting a successful text response."""
    interaction = MagicMock()
    interaction.id = 'interaction_123'
    interaction.status = 'completed'
    text_output = MagicMock()
    text_output.type = 'text'
    text_output.text = 'The answer is 4.'
    interaction.outputs = [text_output]
    interaction.usage = MagicMock()
    interaction.usage.total_input_tokens = 10
    interaction.usage.total_output_tokens = 5
    interaction.error = None

    result = interactions_utils.convert_interaction_to_llm_response(interaction)

    assert result.interaction_id == 'interaction_123'
    assert result.content.parts[0].text == 'The answer is 4.'
    assert result.usage_metadata.prompt_token_count == 10
    assert result.usage_metadata.candidates_token_count == 5
    assert result.finish_reason == types.FinishReason.STOP
    assert result.turn_complete is True

  def test_failed_response(self):
    """Test converting a failed response."""
    interaction = MagicMock()
    interaction.id = 'interaction_123'
    interaction.status = 'failed'
    interaction.outputs = []
    interaction.error = MagicMock()
    interaction.error.code = 'INVALID_REQUEST'
    interaction.error.message = 'Bad request'

    result = interactions_utils.convert_interaction_to_llm_response(interaction)

    assert result.interaction_id == 'interaction_123'
    assert result.error_code == 'INVALID_REQUEST'
    assert result.error_message == 'Bad request'

  def test_requires_action_response(self):
    """Test converting a requires_action response (function call)."""
    interaction = MagicMock()
    interaction.id = 'interaction_123'
    interaction.status = 'requires_action'
    fc_output = MagicMock()
    fc_output.type = 'function_call'
    fc_output.id = 'call_1'
    fc_output.name = 'get_weather'
    fc_output.arguments = {'city': 'Paris'}
    interaction.outputs = [fc_output]
    interaction.usage = None
    interaction.error = None

    result = interactions_utils.convert_interaction_to_llm_response(interaction)

    assert result.interaction_id == 'interaction_123'
    assert result.content.parts[0].function_call.name == 'get_weather'
    assert result.finish_reason == types.FinishReason.STOP
    assert result.turn_complete is True


class TestBuildGenerationConfig:
  """Tests for build_generation_config."""

  def test_all_parameters(self):
    """Test building config with all parameters."""
    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=100,
        stop_sequences=['END'],
        presence_penalty=0.5,
        frequency_penalty=0.3,
    )
    result = interactions_utils.build_generation_config(config)
    assert result == {
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'max_output_tokens': 100,
        'stop_sequences': ['END'],
        'presence_penalty': 0.5,
        'frequency_penalty': 0.3,
    }

  def test_partial_parameters(self):
    """Test building config with partial parameters."""
    config = types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=50,
    )
    result = interactions_utils.build_generation_config(config)
    assert result == {
        'temperature': 0.5,
        'max_output_tokens': 50,
    }

  def test_empty_config(self):
    """Test building config with no parameters."""
    config = types.GenerateContentConfig()
    result = interactions_utils.build_generation_config(config)
    assert result == {}


class TestExtractSystemInstruction:
  """Tests for extract_system_instruction."""

  def test_string_instruction(self):
    """Test extracting string system instruction."""
    config = types.GenerateContentConfig(
        system_instruction='You are a helpful assistant.'
    )
    result = interactions_utils.extract_system_instruction(config)
    assert result == 'You are a helpful assistant.'

  def test_content_instruction(self):
    """Test extracting Content system instruction."""
    config = types.GenerateContentConfig(
        system_instruction=types.Content(
            parts=[
                types.Part(text='Be helpful.'),
                types.Part(text='Be concise.'),
            ]
        )
    )
    result = interactions_utils.extract_system_instruction(config)
    assert result == 'Be helpful.\nBe concise.'

  def test_no_instruction(self):
    """Test extracting when no system instruction."""
    config = types.GenerateContentConfig()
    result = interactions_utils.extract_system_instruction(config)
    assert result is None


class TestLlmRequestPreviousInteractionId:
  """Tests for previous_interaction_id field in LlmRequest."""

  def test_previous_interaction_id_default_none(self):
    """Test that previous_interaction_id defaults to None."""
    request = LlmRequest(model='gemini-2.5-flash', contents=[])
    assert request.previous_interaction_id is None

  def test_previous_interaction_id_can_be_set(self):
    """Test that previous_interaction_id can be set."""
    request = LlmRequest(
        model='gemini-2.5-flash',
        contents=[],
        previous_interaction_id='interaction_abc',
    )
    assert request.previous_interaction_id == 'interaction_abc'


class TestLlmResponseInteractionId:
  """Tests for interaction_id field in LlmResponse."""

  def test_interaction_id_in_response(self):
    """Test that interaction_id is properly set in LlmResponse."""
    from google.adk.models.llm_response import LlmResponse

    response = LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text='Hi')]),
        interaction_id='interaction_xyz',
    )
    assert response.interaction_id == 'interaction_xyz'

  def test_interaction_id_default_none(self):
    """Test that interaction_id defaults to None."""
    from google.adk.models.llm_response import LlmResponse

    response = LlmResponse(
        content=types.Content(role='model', parts=[types.Part(text='Hi')]),
    )
    assert response.interaction_id is None


class TestGetLatestUserContents:
  """Tests for _get_latest_user_contents."""

  def test_empty_contents(self):
    """Test with empty contents list."""
    result = interactions_utils._get_latest_user_contents([])
    assert result == []

  def test_single_user_message(self):
    """Test with a single user message."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='Hello')]),
    ]
    result = interactions_utils._get_latest_user_contents(contents)
    assert len(result) == 1
    assert result[0].parts[0].text == 'Hello'

  def test_consecutive_user_messages(self):
    """Test with multiple consecutive user messages at the end."""
    contents = [
        types.Content(role='model', parts=[types.Part(text='Response')]),
        types.Content(role='user', parts=[types.Part(text='First')]),
        types.Content(role='user', parts=[types.Part(text='Second')]),
    ]
    result = interactions_utils._get_latest_user_contents(contents)
    assert len(result) == 2
    assert result[0].parts[0].text == 'First'
    assert result[1].parts[0].text == 'Second'

  def test_stops_at_model_message(self):
    """Test that it stops when encountering a model message."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='First user')]),
        types.Content(role='model', parts=[types.Part(text='Model response')]),
        types.Content(role='user', parts=[types.Part(text='Second user')]),
    ]
    result = interactions_utils._get_latest_user_contents(contents)
    assert len(result) == 1
    assert result[0].parts[0].text == 'Second user'

  def test_all_model_messages(self):
    """Test with only model messages returns empty list."""
    contents = [
        types.Content(role='model', parts=[types.Part(text='Response 1')]),
        types.Content(role='model', parts=[types.Part(text='Response 2')]),
    ]
    result = interactions_utils._get_latest_user_contents(contents)
    assert result == []

  def test_full_conversation(self):
    """Test with a full conversation, returns only latest user turn."""
    contents = [
        types.Content(role='user', parts=[types.Part(text='Hi')]),
        types.Content(role='model', parts=[types.Part(text='Hello!')]),
        types.Content(role='user', parts=[types.Part(text='How are you?')]),
        types.Content(role='model', parts=[types.Part(text='I am fine.')]),
        types.Content(role='user', parts=[types.Part(text='Great')]),
        types.Content(role='user', parts=[types.Part(text='Tell me more')]),
    ]
    result = interactions_utils._get_latest_user_contents(contents)
    assert len(result) == 2
    assert result[0].parts[0].text == 'Great'
    assert result[1].parts[0].text == 'Tell me more'

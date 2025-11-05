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

"""Unit tests for Code Execution logic."""

import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import Agent
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.flows.llm_flows._code_execution import response_processor
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
@patch('google.adk.flows.llm_flows._code_execution.datetime')
async def test_builtin_code_executor_image_artifact_creation(mock_datetime):
  """Test BuiltInCodeExecutor creates artifacts for images in response."""
  mock_now = datetime.datetime(2025, 1, 1, 12, 0, 0)
  mock_datetime.datetime.now.return_value.astimezone.return_value = mock_now
  code_executor = BuiltInCodeExecutor()
  agent = Agent(name='test_agent', code_executor=code_executor)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.artifact_service = MagicMock()
  invocation_context.artifact_service.save_artifact = AsyncMock(
      return_value='v1'
  )
  llm_response = LlmResponse(
      content=types.Content(
          parts=[
              types.Part(
                  inline_data=types.Blob(
                      mime_type='image/png',
                      data=b'image1',
                      display_name='image_1.png',
                  )
              ),
              types.Part(text='this is text'),
              types.Part(
                  inline_data=types.Blob(mime_type='image/jpeg', data=b'image2')
              ),
          ]
      )
  )

  events = []
  async for event in response_processor.run_async(
      invocation_context, llm_response
  ):
    events.append(event)

  expected_timestamp = mock_now.strftime('%Y%m%d_%H%M%S')
  expected_filename2 = f'{expected_timestamp}.jpeg'

  assert invocation_context.artifact_service.save_artifact.call_count == 2
  invocation_context.artifact_service.save_artifact.assert_any_call(
      app_name=invocation_context.app_name,
      user_id=invocation_context.user_id,
      session_id=invocation_context.session.id,
      filename='image_1.png',
      artifact=types.Part.from_bytes(data=b'image1', mime_type='image/png'),
  )
  invocation_context.artifact_service.save_artifact.assert_any_call(
      app_name=invocation_context.app_name,
      user_id=invocation_context.user_id,
      session_id=invocation_context.session.id,
      filename=expected_filename2,
      artifact=types.Part.from_bytes(data=b'image2', mime_type='image/jpeg'),
  )

  assert len(events) == 1
  assert events[0].actions.artifact_delta == {
      'image_1.png': 'v1',
      expected_filename2: 'v1',
  }
  assert not events[0].content
  assert llm_response.content is not None
  assert len(llm_response.content.parts) == 3
  assert (
      llm_response.content.parts[0].text == 'Saved as artifact: image_1.png. '
  )
  assert not llm_response.content.parts[0].inline_data
  assert llm_response.content.parts[1].text == 'this is text'
  assert (
      llm_response.content.parts[2].text
      == f'Saved as artifact: {expected_filename2}. '
  )
  assert not llm_response.content.parts[2].inline_data


@pytest.mark.asyncio
@patch('google.adk.flows.llm_flows._code_execution.logger')
async def test_logs_executed_code(mock_logger):
  """Test that the response processor logs the code it executes."""
  mock_code_executor = MagicMock(spec=BaseCodeExecutor)
  mock_code_executor.code_block_delimiters = [('```python\n', '\n```')]
  mock_code_executor.error_retry_attempts = 2
  mock_code_executor.stateful = False
  mock_code_executor.execute_code.return_value = CodeExecutionResult(
      stdout='hello'
  )

  agent = Agent(name='test_agent', code_executor=mock_code_executor)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.artifact_service = MagicMock()
  invocation_context.artifact_service.save_artifact = AsyncMock()

  llm_response = LlmResponse(
      content=types.Content(
          parts=[
              types.Part(text='Here is some code:'),
              types.Part(text='```python\nprint("hello")\n```'),
          ]
      )
  )

  _ = [
      event
      async for event in response_processor.run_async(
          invocation_context, llm_response
      )
  ]

  mock_code_executor.execute_code.assert_called_once()
  mock_logger.debug.assert_called_once_with(
      'Executed code:\n```\n%s\n```', 'print("hello")'
  )

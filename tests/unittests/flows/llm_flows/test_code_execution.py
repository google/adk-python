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

"""Unit tests for Code Execution logic."""

import ast
import asyncio
import datetime
import threading
from typing import Any
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.llm_agent import Agent
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import File
from google.adk.flows.llm_flows._code_execution import _DATA_FILE_HELPER_LIB
from google.adk.flows.llm_flows._code_execution import _get_data_file_preprocessing_code
from google.adk.flows.llm_flows._code_execution import request_processor
from google.adk.flows.llm_flows._code_execution import response_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

from ... import testing_utils


class _ExecutionRecord:
  """Captures how the executor ran, for cross-thread inspection in tests."""

  def __init__(self):
    self.thread: Optional[Any] = None
    self.released: Optional[bool] = None


class _RecordingCodeExecutor(BaseCodeExecutor):
  """A code executor that records the thread it runs on.

  `execute_code` blocks on `release` so tests can verify it is offloaded from
  the event loop: it records the running thread, signals `started`, then waits
  for `release` before returning.
  """

  model_config = {'arbitrary_types_allowed': True}

  started: threading.Event
  release: threading.Event
  record: _ExecutionRecord

  def execute_code(
      self,
      invocation_context,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    self.record.thread = threading.current_thread()
    self.started.set()
    self.record.released = self.release.wait(timeout=2)
    return CodeExecutionResult(stdout='ok')


@pytest.mark.asyncio
@patch('google.adk.flows.llm_flows._code_execution.datetime')
async def test_builtin_code_executor_image_artifact_creation(mock_datetime):
  """Test BuiltInCodeExecutor creates artifacts for images in response."""
  mock_now = datetime.datetime(2025, 1, 1, 12, 0, 0)
  mock_datetime.datetime.fromtimestamp.return_value.astimezone.return_value = (
      mock_now
  )
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


def test_data_file_helper_lib_defines_crop():
  """`explore_df` in the injected helper lib calls `crop`, which must exist."""
  pd = pytest.importorskip('pandas')
  namespace = {}
  exec(_DATA_FILE_HELPER_LIB, namespace)  # pylint: disable=exec-used

  crop = namespace['crop']
  assert crop('short') == 'short'
  assert crop('x' * 100, max_chars=10) == 'x' * 7 + '...'
  assert crop('abcdef', max_chars=2) == 'ab'

  # Regression for #4011: explore_df raised NameError when crop was undefined.
  namespace['explore_df'](pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}))


def test_get_data_file_preprocessing_code_injection_reproduction():
  """Test that filenames with injection payloads are safely escaped."""
  bad_filename = "'); print('PWNED')#"
  file = File(name=bad_filename, mime_type='text/csv', content=b'')
  code = _get_data_file_preprocessing_code(file)

  tree = ast.parse(code)
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name) and node.func.id == 'print':
        if (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 'PWNED'
        ):
          pytest.fail(
              "Vulnerability reproduction: print('PWNED') was parsed as"
              ' executable code!'
          )

  # Check that read_csv was called with bad_filename as a safe string literal.
  read_csv_arg = None
  for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'read_csv'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'pd'
    ):
      assert len(node.args) == 1
      assert isinstance(node.args[0], ast.Constant)
      read_csv_arg = node.args[0].value
      break

  assert read_csv_arg == bad_filename


@pytest.mark.asyncio
async def test_post_processor_does_not_block_event_loop():
  """Response processor offloads blocking execute_code off the event loop."""
  started = threading.Event()
  release = threading.Event()
  record = _ExecutionRecord()
  loop_ran = False
  code_executor = _RecordingCodeExecutor(
      started=started, release=release, record=record
  )
  agent = Agent(name='test_agent', code_executor=code_executor)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  invocation_context.artifact_service = MagicMock()
  invocation_context.artifact_service.save_artifact = AsyncMock()

  llm_response = LlmResponse(
      content=types.Content(
          parts=[types.Part(text='```python\nprint("hello")\n```')]
      )
  )

  async def _release_when_started():
    nonlocal loop_ran
    while not started.is_set():
      await asyncio.sleep(0.001)
    loop_ran = True
    release.set()

  releaser = asyncio.create_task(_release_when_started())
  _ = [
      event
      async for event in response_processor.run_async(
          invocation_context, llm_response
      )
  ]
  await releaser

  assert record.thread is not threading.main_thread()
  assert record.released is True
  assert loop_ran is True


@pytest.mark.asyncio
async def test_pre_processor_runs_execute_code_off_the_loop():
  """Request processor offloads blocking execute_code off the event loop."""
  started = threading.Event()
  release = threading.Event()
  release.set()
  record = _ExecutionRecord()
  code_executor = _RecordingCodeExecutor(
      started=started,
      release=release,
      record=record,
      optimize_data_file=True,
  )
  agent = Agent(name='test_agent', code_executor=code_executor)
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )

  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      inline_data=types.Blob(
                          mime_type='text/csv',
                          data=b'col1,col2\n1,2\n',
                      )
                  )
              ],
          )
      ]
  )

  _ = [
      event
      async for event in request_processor.run_async(
          invocation_context, llm_request
      )
  ]

  assert record.thread is not threading.main_thread()

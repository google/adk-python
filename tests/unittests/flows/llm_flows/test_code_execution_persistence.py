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

"""Tests for optimize_data_file persistence across turns."""

import copy
from typing import Dict

from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.code_executors.code_execution_utils import File
from google.adk.code_executors.code_executor_context import CodeExecutorContext
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
from pydantic import Field
import pytest


class MockCodeExecutor(BaseCodeExecutor):
  """Mock executor for testing."""

  # Define as Pydantic fields
  injected_files: Dict[str, str] = Field(default_factory=dict)
  execution_count: int = Field(default=0)

  def execute_code(self, invocation_context, code_input: CodeExecutionInput):
    """Mock code execution."""
    self.execution_count += 1
    # Store files if they're new
    for file in code_input.input_files:
      if file.name not in self.injected_files:
        self.injected_files[file.name] = file.content

    return CodeExecutionResult(
        stdout=f'Executed: {len(code_input.input_files)} files available',
        stderr='',
        output_files=[],
    )


@pytest.mark.asyncio
async def test_inline_data_replacement_in_extract_function():
  """Test that _extract_and_replace_inline_files modifies both request and session."""
  from google.adk.flows.llm_flows._code_execution import _extract_and_replace_inline_files

  # Create CSV data
  csv_data = b'name,value\nA,100\nB,200'

  # Create a mock session with events
  session = Session(
      id='test_session',
      app_name='test_app',
      user_id='test_user',
      state={},
      events=[],
  )

  # Create user content with inline_data
  user_content = types.Content(
      role='user',
      parts=[
          types.Part(text='Process this'),
          types.Part(
              inline_data=types.Blob(mime_type='text/csv', data=csv_data)
          ),
      ],
  )

  # Add to session events
  from google.adk.events.event import Event

  user_event = Event(
      invocation_id='test_inv',
      author='user',
      content=user_content,
  )
  session.events.append(user_event)

  # Create LLM request
  llm_request = LlmRequest(contents=[copy.deepcopy(user_content)])

  # Create code executor context
  code_executor_context = CodeExecutorContext(session.state)

  # Create mock invocation context
  from unittest.mock import Mock

  invocation_context = Mock()
  invocation_context.session = session

  # Call the function we're testing
  result_files = _extract_and_replace_inline_files(
      code_executor_context, llm_request, invocation_context
  )

  # Check LLM request was modified
  has_inline_in_request = any(
      p.inline_data for content in llm_request.contents for p in content.parts
  )
  has_placeholder_in_request = any(
      'Available file:' in (p.text or '')
      for content in llm_request.contents
      for p in content.parts
  )

  # Check session events were modified
  user_events = [
      e for e in session.events if e.content and e.content.role == 'user'
  ]
  assert len(user_events) > 0, 'Should have user events'

  first_user_event = user_events[0]
  has_inline_in_session = any(
      p.inline_data for p in first_user_event.content.parts
  )
  has_placeholder_in_session = any(
      'Available file:' in (p.text or '')
      for p in first_user_event.content.parts
  )

  # Assertions for LLM request
  assert (
      not has_inline_in_request
  ), 'inline_data should be replaced in LLM request'
  assert (
      has_placeholder_in_request
  ), 'Placeholder should be present in LLM request'

  # Critical assertions for session events
  assert (
      not has_inline_in_session
  ), 'inline_data should be replaced in session.events'
  assert (
      has_placeholder_in_session
  ), 'Placeholder should be present in session.events'

  # Test that files were extracted
  assert len(result_files) >= 1, 'At least one file should be extracted'


@pytest.mark.asyncio
async def test_persistence_across_simulated_turns():
  """Test that on a second 'turn', inline_data doesn't reappear."""
  from google.adk.flows.llm_flows._code_execution import _extract_and_replace_inline_files

  # Create CSV data
  csv_data = b'name,value\nA,100\nB,200'

  # Create a session
  session = Session(
      id='test_session',
      app_name='test_app',
      user_id='test_user',
      state={},
      events=[],
  )

  # Turn 1: User sends CSV
  user_content_1 = types.Content(
      role='user',
      parts=[
          types.Part(text='Process this'),
          types.Part(
              inline_data=types.Blob(mime_type='text/csv', data=csv_data)
          ),
      ],
  )

  from google.adk.events.event import Event

  user_event_1 = Event(
      invocation_id='test_inv_1',
      author='user',
      content=user_content_1,
  )
  session.events.append(user_event_1)

  # Create code executor context
  code_executor_context = CodeExecutorContext(session.state)

  # Mock invocation context
  from unittest.mock import Mock

  invocation_context = Mock()
  invocation_context.session = session

  # Process turn 1
  llm_request_1 = LlmRequest(contents=[copy.deepcopy(user_content_1)])
  files_1 = _extract_and_replace_inline_files(
      code_executor_context, llm_request_1, invocation_context
  )

  initial_file_count = len(files_1)

  # Verify session was modified
  user_events = [
      e for e in session.events if e.content and e.content.role == 'user'
  ]
  has_inline_after_turn1 = any(
      p.inline_data for e in user_events for p in e.content.parts
  )

  # Turn 2: User sends follow-up
  user_content_2 = types.Content(
      role='user', parts=[types.Part(text='What is the sum?')]
  )

  user_event_2 = Event(
      invocation_id='test_inv_2',
      author='user',
      content=user_content_2,
  )
  session.events.append(user_event_2)

  # Create new LLM request with ALL session events (simulating real flow)
  all_contents = []
  for event in session.events:
    if event.content:
      all_contents.append(copy.deepcopy(event.content))

  llm_request_2 = LlmRequest(contents=all_contents)

  # Check if inline_data reappeared
  has_inline_before_process = any(
      p.inline_data for content in llm_request_2.contents for p in content.parts
  )

  # Process turn 2
  files_2 = _extract_and_replace_inline_files(
      code_executor_context, llm_request_2, invocation_context
  )

  final_file_count = len(code_executor_context.get_input_files())

  # Critical assertion
  assert not has_inline_before_process, (
      'inline_data should NOT reappear in turn 2 if session.events were'
      ' properly modified'
  )


if __name__ == '__main__':
  pytest.main([__file__, '-v', '-s'])

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

"""Tests for empty model response retry logic in BaseLlmFlow.run_async."""

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.base_llm_flow import _has_meaningful_content
from google.adk.flows.llm_flows.base_llm_flow import _MAX_EMPTY_RESPONSE_RETRIES
from google.genai import types
import pytest


class TestHasMeaningfulContent:
  """Tests for the _has_meaningful_content helper function."""

  def test_no_content(self):
    """Event with no content is not meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=None,
    )
    assert not _has_meaningful_content(event)

  def test_empty_parts(self):
    """Event with empty parts list is not meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(role='model', parts=[]),
    )
    assert not _has_meaningful_content(event)

  def test_only_empty_text_part(self):
    """Event with only an empty text part is not meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='')]
        ),
    )
    assert not _has_meaningful_content(event)

  def test_only_whitespace_text_part(self):
    """Event with only whitespace text is not meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='   \n  ')]
        ),
    )
    assert not _has_meaningful_content(event)

  def test_thought_only_parts(self):
    """Event with only thought parts is not meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[types.Part(text='Let me think...', thought=True)],
        ),
    )
    assert not _has_meaningful_content(event)

  def test_text_content_is_meaningful(self):
    """Event with non-empty text is meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text='Here is the answer.')],
        ),
    )
    assert _has_meaningful_content(event)

  def test_function_call_is_meaningful(self):
    """Event with a function call is meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name='my_tool', args={})
                )
            ],
        ),
    )
    assert _has_meaningful_content(event)

  def test_function_response_is_meaningful(self):
    """Event with a function response is meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name='my_tool', response={'result': 'ok'}
                    )
                )
            ],
        ),
    )
    assert _has_meaningful_content(event)

  def test_thought_plus_text_is_meaningful(self):
    """Event with thought AND real text is meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[
                types.Part(text='Thinking...', thought=True),
                types.Part.from_text(text='The answer is 42.'),
            ],
        ),
    )
    assert _has_meaningful_content(event)

  def test_inline_data_is_meaningful(self):
    """Event with inline data is meaningful."""
    event = Event(
        invocation_id='test',
        author='model',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        mime_type='image/png', data=b'\x89PNG'
                    )
                )
            ],
        ),
    )
    assert _has_meaningful_content(event)


class TestMaxEmptyResponseRetries:
  """Verify the retry constant is sensible."""

  def test_retry_limit_is_positive(self):
    assert _MAX_EMPTY_RESPONSE_RETRIES > 0

  def test_retry_limit_is_small(self):
    """Retry limit should be small to avoid excessive re-prompts."""
    assert _MAX_EMPTY_RESPONSE_RETRIES <= 5

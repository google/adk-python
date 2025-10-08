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

from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import get_all_tool_calls_with_responses
from google.adk.evaluation.eval_case import get_all_tool_responses
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.genai import types as genai_types
import pytest


def test_get_all_tool_calls_with_none_input():
  """Tests that an empty list is returned when intermediate_data is None."""
  assert get_all_tool_calls(None) == []


def test_get_all_tool_calls_with_intermediate_data_no_tools():
  """Tests IntermediateData with no tool calls."""
  intermediate_data = IntermediateData(tool_uses=[])
  assert get_all_tool_calls(intermediate_data) == []


def test_get_all_tool_calls_with_intermediate_data():
  """Tests that tool calls are correctly extracted from IntermediateData."""
  tool_call1 = genai_types.FunctionCall(
      name='search', args={'query': 'weather'}
  )
  tool_call2 = genai_types.FunctionCall(name='lookup', args={'id': '123'})
  intermediate_data = IntermediateData(tool_uses=[tool_call1, tool_call2])
  assert get_all_tool_calls(intermediate_data) == [tool_call1, tool_call2]


def test_get_all_tool_calls_with_empty_invocation_events():
  """Tests InvocationEvents with an empty list of invocation events."""
  intermediate_data = InvocationEvents(invocation_events=[])
  assert get_all_tool_calls(intermediate_data) == []


def test_get_all_tool_calls_with_invocation_events_no_tools():
  """Tests InvocationEvents containing events without any tool calls."""
  invocation_event = InvocationEvent(
      author='agent',
      content=genai_types.Content(
          parts=[genai_types.Part(text='Thinking...')], role='model'
      ),
  )
  intermediate_data = InvocationEvents(invocation_events=[invocation_event])
  assert get_all_tool_calls(intermediate_data) == []


def test_get_all_tool_calls_with_invocation_events():
  """Tests that tool calls are correctly extracted from a InvocationSteps object."""
  tool_call1 = genai_types.FunctionCall(
      name='search', args={'query': 'weather'}
  )
  tool_call2 = genai_types.FunctionCall(name='lookup', args={'id': '123'})

  invocation_event1 = InvocationEvent(
      author='agent1',
      content=genai_types.Content(
          parts=[genai_types.Part(function_call=tool_call1)],
          role='model',
      ),
  )
  invocation_event2 = InvocationEvent(
      author='agent2',
      content=genai_types.Content(
          parts=[
              genai_types.Part(text='Found something.'),
              genai_types.Part(function_call=tool_call2),
          ],
          role='model',
      ),
  )
  intermediate_data = InvocationEvents(
      invocation_events=[invocation_event1, invocation_event2]
  )
  assert get_all_tool_calls(intermediate_data) == [tool_call1, tool_call2]


def test_get_all_tool_calls_with_unsupported_type():
  """Tests that a ValueError is raised for unsupported intermediate_data types."""
  with pytest.raises(
      ValueError, match='Unsupported type for intermediate_data'
  ):
    get_all_tool_calls('this is not a valid type')


def test_get_all_tool_responses_with_none_input():
  """Tests that an empty list is returned when intermediate_data is None."""
  assert get_all_tool_responses(None) == []


def test_get_all_tool_responses_with_empty_invocation_events():
  """Tests InvocationEvents with an empty list of events."""
  intermediate_data = InvocationEvents(invocation_events=[])
  assert get_all_tool_responses(intermediate_data) == []


def test_get_all_tool_responses_with_invocation_events_no_tools():
  """Tests InvocationEvents containing events without any tool responses."""
  invocation_event = InvocationEvent(
      author='agent',
      content=genai_types.Content(
          parts=[genai_types.Part(text='Thinking...')], role='model'
      ),
  )
  intermediate_data = InvocationEvents(invocation_events=[invocation_event])
  assert get_all_tool_responses(intermediate_data) == []


def test_get_all_tool_responses_with_invocation_events():
  """Tests that tool responses are correctly extracted from a InvocationEvents object."""
  tool_response1 = genai_types.FunctionResponse(
      name='search', response={'result': 'weather is good'}
  )
  tool_response2 = genai_types.FunctionResponse(
      name='lookup', response={'id': '123'}
  )
  invocation_event1 = InvocationEvent(
      author='agent1',
      content=genai_types.Content(
          parts=[genai_types.Part(function_response=tool_response1)],
          role='model',
      ),
  )
  invocation_event2 = InvocationEvent(
      author='agent2',
      content=genai_types.Content(
          parts=[
              genai_types.Part(text='Found something.'),
              genai_types.Part(function_response=tool_response2),
          ],
          role='model',
      ),
  )
  intermediate_data = InvocationEvents(
      invocation_events=[invocation_event1, invocation_event2]
  )
  assert get_all_tool_responses(intermediate_data) == [
      tool_response1,
      tool_response2,
  ]


def test_get_all_tool_responses_with_unsupported_type():
  """Tests that a ValueError is raised for unsupported intermediate_data types."""
  with pytest.raises(
      ValueError, match='Unsupported type for intermediate_data'
  ):
    get_all_tool_responses('this is not a valid type')


def test_get_all_tool_calls_with_responses_with_none_input():
  """Tests that an empty list is returned when intermediate_data is None."""
  assert get_all_tool_calls_with_responses(None) == []


def test_get_all_tool_calls_with_responses_with_intermediate_data_no_tool_calls():
  """Tests get_all_tool_calls_with_responses with IntermediateData with no tool calls."""
  # No tool calls
  intermediate_data = IntermediateData(tool_uses=[], tool_responses=[])
  assert get_all_tool_calls_with_responses(intermediate_data) == []


def test_get_all_tool_calls_with_responses_with_intermediate_data_with_tool_calls():
  """Tests get_all_tool_calls_with_responses with IntermediateData with tools."""
  # With matching and non-matching tool calls
  tool_call1 = genai_types.FunctionCall(
      name='search', args={'query': 'weather'}, id='call1'
  )
  tool_response1 = genai_types.FunctionResponse(
      name='search', response={'result': 'sunny'}, id='call1'
  )
  tool_call2 = genai_types.FunctionCall(
      name='lookup', args={'id': '123'}, id='call2'
  )
  intermediate_data = IntermediateData(
      tool_uses=[tool_call1, tool_call2], tool_responses=[tool_response1]
  )
  assert get_all_tool_calls_with_responses(intermediate_data) == [
      (tool_call1, tool_response1),
      (tool_call2, None),
  ]


def test_get_all_tool_calls_with_responses_with_steps_no_tool_calls():
  """Tests get_all_tool_calls_with_responses with Steps that don't have tool calls."""
  # No tool calls
  intermediate_data = InvocationEvents(invocation_events=[])
  assert get_all_tool_calls_with_responses(intermediate_data) == []


def test_get_all_tool_calls_with_responses_with_invocation_events():
  """Tests get_all_tool_calls_with_responses with InvocationEvents."""
  # No tools
  intermediate_data = InvocationEvents(invocation_events=[])
  assert get_all_tool_calls_with_responses(intermediate_data) == []

  # With matching and non-matching tool calls
  tool_call1 = genai_types.FunctionCall(
      name='search', args={'query': 'weather'}, id='call1'
  )
  tool_response1 = genai_types.FunctionResponse(
      name='search', response={'result': 'sunny'}, id='call1'
  )
  tool_call2 = genai_types.FunctionCall(
      name='lookup', args={'id': '123'}, id='call2'
  )
  invocation_event1 = InvocationEvent(
      author='agent',
      content=genai_types.Content(
          parts=[
              genai_types.Part(function_call=tool_call1),
              genai_types.Part(function_call=tool_call2),
          ],
          role='model',
      ),
  )
  invocation_event2 = InvocationEvent(
      author='tool',
      content=genai_types.Content(
          parts=[genai_types.Part(function_response=tool_response1)],
          role='tool',
      ),
  )
  intermediate_data = InvocationEvents(
      invocation_events=[invocation_event1, invocation_event2]
  )
  assert get_all_tool_calls_with_responses(intermediate_data) == [
      (tool_call1, tool_response1),
      (tool_call2, None),
  ]

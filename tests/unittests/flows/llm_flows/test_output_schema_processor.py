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

"""Tests for output schema processor functionality."""

import json

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from pydantic import BaseModel
from pydantic import Field
import pytest


class PersonSchema(BaseModel):
  """Test schema for structured output."""

  name: str = Field(description="A person's name")
  age: int = Field(description="A person's age")
  city: str = Field(description='The city they live in')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Searched for: {query}'


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


@pytest.mark.asyncio
async def test_output_schema_with_tools_validation_removed():
  """Test that LlmAgent now allows output_schema with tools."""
  # This should not raise an error anymore
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  assert agent.output_schema == PersonSchema
  assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_basic_processor_skips_output_schema_with_tools():
  """Test that basic processor doesn't set output_schema when tools are present."""
  from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should not have set response_schema since agent has tools
  assert llm_request.config.response_schema is None
  assert llm_request.config.response_mime_type != 'application/json'


@pytest.mark.asyncio
async def test_basic_processor_sets_output_schema_without_tools():
  """Test that basic processor still sets output_schema when no tools are present."""
  from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[],  # No tools
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _BasicLlmRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should have set response_schema since agent has no tools
  assert llm_request.config.response_schema == PersonSchema
  assert llm_request.config.response_mime_type == 'application/json'


@pytest.mark.asyncio
async def test_output_schema_request_processor():
  """Test that output schema processor adds set_model_response tool."""
  from google.adk.flows.llm_flows._output_schema_processor import _OutputSchemaRequestProcessor

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  processor = _OutputSchemaRequestProcessor()

  # Process the request
  events = []
  async for event in processor.run_async(invocation_context, llm_request):
    events.append(event)

  # Should have added set_model_response tool
  assert 'set_model_response' in llm_request.tools_dict

  # Should have added instruction about using set_model_response
  assert 'set_model_response' in llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_set_model_response_tool():
  """Test the set_model_response tool functionality."""
  from google.adk.tools.set_model_response_tool import MODEL_JSON_RESPONSE_KEY
  from google.adk.tools.set_model_response_tool import SetModelResponseTool
  from google.adk.tools.tool_context import ToolContext

  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Call the tool with valid data
  result = await tool.run_async(
      args={'name': 'John Doe', 'age': 30, 'city': 'New York'},
      tool_context=tool_context,
  )

  # Verify the tool now returns dict directly
  assert result is not None
  assert result['name'] == 'John Doe'
  assert result['age'] == 30
  assert result['city'] == 'New York'

  # Check that the response is no longer stored in session state
  stored_response = invocation_context.session.state.get(
      MODEL_JSON_RESPONSE_KEY
  )
  assert stored_response is None


@pytest.mark.asyncio
async def test_output_schema_helper_functions():
  """Test the helper functions for handling set_model_response."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows._output_schema_processor import create_final_model_response_event
  from google.adk.flows.llm_flows._output_schema_processor import get_structured_model_response
  from google.genai import types

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  # Test get_structured_model_response with a function response event
  test_dict = {'name': 'Jane Smith', 'age': 25, 'city': 'Los Angeles'}
  test_json = '{"name": "Jane Smith", "age": 25, "city": "Los Angeles"}'

  # Create a function response event with set_model_response
  function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='set_model_response', response=test_dict
                  )
              )
          ],
      ),
  )

  # Test get_structured_model_response function
  extracted_json = get_structured_model_response(function_response_event)
  assert extracted_json == test_json

  # Test create_final_model_response_event function
  final_event = create_final_model_response_event(invocation_context, test_json)
  assert final_event.author == 'test_agent'
  assert final_event.content.role == 'model'
  assert final_event.content.parts[0].text == test_json

  # Test get_structured_model_response with non-set_model_response function
  other_function_response_event = Event(
      author='test_agent',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='other_tool', response={'result': 'other response'}
                  )
              )
          ],
      ),
  )

  extracted_json = get_structured_model_response(other_function_response_event)
  assert extracted_json is None


@pytest.mark.asyncio
async def test_end_to_end_integration():
  """Test the complete output schema with tools integration."""
  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)

  # Create a flow and test the processors
  flow = SingleFlow()
  llm_request = LlmRequest()

  # Run all request processors
  async for event in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify set_model_response tool was added
  assert 'set_model_response' in llm_request.tools_dict

  # Verify instruction was added
  assert 'set_model_response' in llm_request.config.system_instruction

  # Verify output_schema was NOT set on the model config
  assert llm_request.config.response_schema is None


@pytest.mark.asyncio
async def test_flow_yields_both_events_for_set_model_response():
  """Test that the flow yields both function response and final model response events."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
  from google.adk.tools.set_model_response_tool import SetModelResponseTool
  from google.genai import types

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      output_schema=PersonSchema,
      tools=[],
  )

  invocation_context = await _create_invocation_context(agent)
  flow = BaseLlmFlow()

  # Create a set_model_response tool and add it to the tools dict
  set_response_tool = SetModelResponseTool(PersonSchema)
  llm_request = LlmRequest()
  llm_request.tools_dict['set_model_response'] = set_response_tool

  # Create a function call event (model calling the function)
  function_call_event = Event(
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name='set_model_response',
                      args={
                          'name': 'Test User',
                          'age': 30,
                          'city': 'Test City',
                      },
                  )
              )
          ],
      ),
  )

  # Test the postprocess function handling
  events = []
  async for event in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(event)

  # Should yield exactly 2 events: function response + final model response
  assert len(events) == 2

  # First event should be the function response
  first_event = events[0]
  assert first_event.get_function_responses()[0].name == 'set_model_response'
  # The response should be the dict returned by the tool
  assert first_event.get_function_responses()[0].response == {
      'name': 'Test User',
      'age': 30,
      'city': 'Test City',
  }

  # Second event should be the final model response with JSON
  second_event = events[1]
  assert second_event.author == 'test_agent'
  assert second_event.content.role == 'model'
  assert (
      second_event.content.parts[0].text
      == '{"name": "Test User", "age": 30, "city": "Test City"}'
  )


@pytest.mark.asyncio
async def test_flow_yields_only_function_response_for_normal_tools():
  """Test that the flow yields only function response event for non-set_model_response tools."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
  from google.genai import types

  agent = LlmAgent(
      name='test_agent',
      model='gemini-1.5-flash',
      tools=[FunctionTool(func=dummy_tool)],
  )

  invocation_context = await _create_invocation_context(agent)
  flow = BaseLlmFlow()

  # Create a dummy tool and add it to the tools dict
  dummy_function_tool = FunctionTool(func=dummy_tool)
  llm_request = LlmRequest()
  llm_request.tools_dict['dummy_tool'] = dummy_function_tool

  # Create a function call event (model calling the dummy tool)
  function_call_event = Event(
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name='dummy_tool', args={'query': 'test query'}
                  )
              )
          ],
      ),
  )

  # Test the postprocess function handling
  events = []
  async for event in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(event)

  # Should yield exactly 1 event: just the function response
  assert len(events) == 1

  # Should be the function response from dummy_tool
  first_event = events[0]
  assert first_event.get_function_responses()[0].name == 'dummy_tool'
  assert first_event.get_function_responses()[0].response == {
      'result': 'Searched for: test query'
  }

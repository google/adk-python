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

"""Tests for SetModelResponseTool."""


from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.set_model_response_tool import MODEL_JSON_RESPONSE_KEY
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
import pytest


class PersonSchema(BaseModel):
  """Test schema for structured output."""

  name: str = Field(description="A person's name")
  age: int = Field(description="A person's age")
  city: str = Field(description='The city they live in')


class ComplexSchema(BaseModel):
  """More complex test schema."""

  id: int
  title: str
  tags: list[str] = Field(default_factory=list)
  metadata: dict[str, str] = Field(default_factory=dict)
  is_active: bool = True


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


def test_tool_initialization_simple_schema():
  """Test tool initialization with a simple schema."""
  tool = SetModelResponseTool(PersonSchema)

  assert tool.output_schema == PersonSchema
  assert tool.name == 'set_model_response'
  assert 'Set your final response' in tool.description
  assert tool.func is not None


def test_tool_initialization_complex_schema():
  """Test tool initialization with a complex schema."""
  tool = SetModelResponseTool(ComplexSchema)

  assert tool.output_schema == ComplexSchema
  assert tool.name == 'set_model_response'
  assert tool.func is not None


def test_function_signature_generation():
  """Test that function signature is correctly generated from schema."""
  tool = SetModelResponseTool(PersonSchema)

  import inspect

  sig = inspect.signature(tool.func)

  # Check that parameters match schema fields
  assert 'name' in sig.parameters
  assert 'age' in sig.parameters
  assert 'city' in sig.parameters

  # All parameters should be keyword-only
  for param in sig.parameters.values():
    assert param.kind == inspect.Parameter.KEYWORD_ONLY


def test_get_declaration():
  """Test that tool declaration is properly generated."""
  tool = SetModelResponseTool(PersonSchema)

  declaration = tool._get_declaration()

  assert declaration is not None
  assert declaration.name == 'set_model_response'
  assert declaration.description is not None


@pytest.mark.asyncio
async def test_run_async_valid_data():
  """Test tool execution with valid data."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with valid data
  result = await tool.run_async(
      args={'name': 'Alice', 'age': 25, 'city': 'Seattle'},
      tool_context=tool_context,
  )

  # Verify the tool now returns dict directly
  assert result is not None
  assert result['name'] == 'Alice'
  assert result['age'] == 25
  assert result['city'] == 'Seattle'

  # Verify data is no longer stored in session state (old behavior)
  stored_response = invocation_context.session.state.get(
      MODEL_JSON_RESPONSE_KEY
  )
  assert stored_response is None


@pytest.mark.asyncio
async def test_run_async_complex_schema():
  """Test tool execution with complex schema."""
  tool = SetModelResponseTool(ComplexSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with complex data
  result = await tool.run_async(
      args={
          'id': 123,
          'title': 'Test Item',
          'tags': ['tag1', 'tag2'],
          'metadata': {'key': 'value'},
          'is_active': False,
      },
      tool_context=tool_context,
  )

  # Verify the tool now returns dict directly
  assert result is not None
  assert result['id'] == 123
  assert result['title'] == 'Test Item'
  assert result['tags'] == ['tag1', 'tag2']
  assert result['metadata'] == {'key': 'value'}
  assert result['is_active'] is False

  # Verify data is no longer stored in session state (old behavior)
  stored_response = invocation_context.session.state.get(
      MODEL_JSON_RESPONSE_KEY
  )
  assert stored_response is None


@pytest.mark.asyncio
async def test_run_async_validation_error():
  """Test tool execution with invalid data raises validation error."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with invalid data (wrong type for age)
  with pytest.raises(ValidationError):
    await tool.run_async(
        args={'name': 'Bob', 'age': 'not_a_number', 'city': 'Portland'},
        tool_context=tool_context,
    )


@pytest.mark.asyncio
async def test_run_async_missing_required_field():
  """Test tool execution with missing required field."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # Execute with missing required field
  with pytest.raises(ValidationError):
    await tool.run_async(
        args={'name': 'Charlie', 'city': 'Denver'},  # Missing age
        tool_context=tool_context,
    )


@pytest.mark.asyncio
async def test_session_state_storage_key():
  """Test that response is no longer stored in session state."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  result = await tool.run_async(
      args={'name': 'Diana', 'age': 35, 'city': 'Miami'},
      tool_context=tool_context,
  )

  # Verify response is returned directly, not stored in session state
  assert result is not None
  assert result['name'] == 'Diana'
  assert result['age'] == 35
  assert result['city'] == 'Miami'

  # Verify session state is no longer used
  assert MODEL_JSON_RESPONSE_KEY not in invocation_context.session.state


@pytest.mark.asyncio
async def test_multiple_executions_return_latest():
  """Test that multiple executions return latest response independently."""
  tool = SetModelResponseTool(PersonSchema)

  agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
  invocation_context = await _create_invocation_context(agent)
  tool_context = ToolContext(invocation_context)

  # First execution
  result1 = await tool.run_async(
      args={'name': 'First', 'age': 20, 'city': 'City1'},
      tool_context=tool_context,
  )

  # Second execution should return its own response
  result2 = await tool.run_async(
      args={'name': 'Second', 'age': 30, 'city': 'City2'},
      tool_context=tool_context,
  )

  # Verify each execution returns its own dict
  assert result1['name'] == 'First'
  assert result1['age'] == 20
  assert result1['city'] == 'City1'

  assert result2['name'] == 'Second'
  assert result2['age'] == 30
  assert result2['city'] == 'City2'

  # Verify session state is not used
  assert MODEL_JSON_RESPONSE_KEY not in invocation_context.session.state


def test_function_return_value_consistency():
  """Test that function return value matches run_async return value."""
  tool = SetModelResponseTool(PersonSchema)

  # Direct function call
  direct_result = tool.func()

  # Both should return the same value
  assert direct_result == 'Response set successfully.'

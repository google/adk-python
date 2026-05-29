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

"""Tests for the finish_task_tool module."""

from __future__ import annotations

from typing import Any
from unittest import mock

from google.adk.agents.llm.task._finish_task_tool import FINISH_TASK_TOOL_NAME
from google.adk.agents.llm.task._finish_task_tool import FinishTaskTool
from pydantic import BaseModel
import pytest


class SampleOutputSchema(BaseModel):
  """Sample Pydantic model for testing output schema."""

  result: str
  count: int


class NestedOutputSchema(BaseModel):
  """Nested Pydantic model for testing complex schemas."""

  name: str
  details: dict


def _make_task_agent(
    name='test_agent',
    mode='task',
    output_schema=None,
):
  """Create a mock task agent for FinishTaskTool construction."""
  agent = mock.MagicMock()
  agent.name = name
  agent.mode = mode
  agent.output_schema = output_schema
  return agent


class TestFinishTaskTool:
  """Tests for the FinishTaskTool class."""

  def test_init_without_output_schema(self):
    """Test FinishTaskTool initialization without output schema."""
    agent = _make_task_agent()
    tool = FinishTaskTool(task_agent=agent)
    assert tool.name == FINISH_TASK_TOOL_NAME
    assert 'Signal that this agent has completed' in tool.description
    assert 'output data' not in tool.description

  def test_init_with_output_schema(self):
    """Test FinishTaskTool initialization with output schema."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    assert tool.name == FINISH_TASK_TOOL_NAME
    assert tool.output_schema is SampleOutputSchema
    assert 'Signal that this agent has completed' in tool.description
    assert 'output data' in tool.description

  def test_get_declaration_without_output_schema(self):
    """Test function declaration generation without output schema."""
    agent = _make_task_agent()
    tool = FinishTaskTool(task_agent=agent)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == FINISH_TASK_TOOL_NAME
    assert declaration.parameters_json_schema is not None
    schema = declaration.parameters_json_schema
    assert 'result' in schema.get('properties', {})

  def test_get_declaration_with_output_schema(self):
    """Test function declaration generation with output schema."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == FINISH_TASK_TOOL_NAME
    assert declaration.parameters_json_schema is not None
    schema = declaration.parameters_json_schema
    assert 'result' in schema.get('properties', {})
    assert 'count' in schema.get('properties', {})

  @pytest.mark.asyncio
  async def test_run_async_returns_confirmation(self):
    """Test that run_async returns a confirmation message."""
    agent = _make_task_agent()
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    result = await tool.run_async(
        args={'result': 'done'}, tool_context=mock_tool_context
    )

    assert result == 'Task completed.'

  @pytest.mark.asyncio
  async def test_run_async_with_args(self):
    """Test that run_async validates args and returns confirmation."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    result = await tool.run_async(
        args={'result': 'success', 'count': 42},
        tool_context=mock_tool_context,
    )

    assert result == 'Task completed.'


class TestBuildInstruction:
  """Tests for the _build_instruction method."""

  def test_instruction_content(self):
    """Test instruction generation contains expected content."""
    agent = _make_task_agent()
    tool = FinishTaskTool(task_agent=agent)
    instruction = tool._build_instruction()

    assert 'finish_task' in instruction
    assert 'Do NOT call `finish_task` prematurely' in instruction
    assert 'call `finish_task` by itself with' in instruction


class TestProcessLlmRequest:
  """Tests for the process_llm_request method."""

  @pytest.mark.asyncio
  async def test_process_llm_request_adds_tool_and_instruction(self):
    """Test that process_llm_request adds tool and instruction."""
    agent = _make_task_agent()
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()
    mock_tool_context._invocation_context.branch = None
    mock_tool_context.session.events = []
    mock_llm_request = mock.MagicMock()
    mock_llm_request.append_tools = mock.MagicMock()
    mock_llm_request.append_instructions = mock.MagicMock()

    await tool.process_llm_request(
        tool_context=mock_tool_context, llm_request=mock_llm_request
    )

    # Should add tool via parent's process_llm_request
    mock_llm_request.append_tools.assert_called_once_with([tool])
    # Should append instruction (at least the base instruction)
    mock_llm_request.append_instructions.assert_called()
    instruction_arg = mock_llm_request.append_instructions.call_args_list[0][0][
        0
    ]
    assert len(instruction_arg) == 1
    assert 'finish_task' in instruction_arg[0]


class TestFinishTaskToolName:
  """Tests for the FINISH_TASK_TOOL_NAME constant."""

  def test_constant_value(self):
    """Test that the constant has the expected value."""
    assert FINISH_TASK_TOOL_NAME == 'finish_task'


class TestFinishTaskToolValidation:
  """Tests for the FinishTaskTool argument validation."""

  @pytest.mark.asyncio
  async def test_run_async_validation_error_missing_required_field(self):
    """Test that validation error is returned when required fields are missing."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    # Missing 'count' field which is required
    result = await tool.run_async(
        args={'result': 'success'},
        tool_context=mock_tool_context,
    )

    assert isinstance(result, dict)
    assert 'error' in result
    assert 'finish_task' in result['error']
    assert 'validation errors' in result['error']
    assert 'count' in result['error']

  @pytest.mark.asyncio
  async def test_run_async_validation_error_wrong_type(self):
    """Test that validation error is returned when types are wrong."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    # 'count' should be int, not string
    result = await tool.run_async(
        args={'result': 'success', 'count': 'not_an_int'},
        tool_context=mock_tool_context,
    )

    assert isinstance(result, dict)
    assert 'error' in result
    assert 'validation errors' in result['error']

  @pytest.mark.asyncio
  async def test_run_async_validation_passes_with_valid_args(self):
    """Test that validation passes with valid args."""
    agent = _make_task_agent(output_schema=SampleOutputSchema)
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    result = await tool.run_async(
        args={'result': 'success', 'count': 42},
        tool_context=mock_tool_context,
    )

    assert result == 'Task completed.'


class TestFinishTaskToolAllSchemaTypes:
  """Tests for FinishTaskTool across all supported SchemaType variants.

  Object schemas (BaseModel, dict) use parameters directly.
  Non-object schemas (str, int, bool, float, list) are wrapped in a
  JSON object with a 'result' key so they can be used as function
  parameters.
  """

  @pytest.mark.parametrize(
      'output_schema, expected_wrapper_key',
      [
          (SampleOutputSchema, None),
          (dict[str, Any], None),
          (str, 'result'),
          (int, 'result'),
          (bool, 'result'),
          (float, 'result'),
          (list[str], 'result'),
          (list[int], 'result'),
          (list[SampleOutputSchema], 'result'),
      ],
      ids=[
          'BaseModel',
          'dict',
          'str',
          'int',
          'bool',
          'float',
          'list_str',
          'list_int',
          'list_BaseModel',
      ],
  )
  def test_wrapper_key(self, output_schema, expected_wrapper_key):
    """Verify wrapper key is set correctly for each schema type."""
    agent = _make_task_agent(output_schema=output_schema)
    tool = FinishTaskTool(task_agent=agent)
    assert tool._wrapper_key == expected_wrapper_key

  @pytest.mark.parametrize(
      'output_schema',
      [str, int, bool, float, list[str], list[int], list[SampleOutputSchema]],
      ids=[
          'str',
          'int',
          'bool',
          'float',
          'list_str',
          'list_int',
          'list_BaseModel',
      ],
  )
  def test_get_declaration_wrapped_schema(self, output_schema):
    """Non-object schemas produce a declaration with 'result' property."""
    agent = _make_task_agent(output_schema=output_schema)
    tool = FinishTaskTool(task_agent=agent)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == FINISH_TASK_TOOL_NAME
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert 'result' in schema.get('properties', {})

  @pytest.mark.parametrize(
      'output_schema, args, expected_output',
      [
          (
              SampleOutputSchema,
              {'result': 'done', 'count': 5},
              {'result': 'done', 'count': 5},
          ),
          (dict[str, Any], {'key': 'value'}, {'key': 'value'}),
          (str, {'result': 'hello'}, 'hello'),
          (int, {'result': 42}, 42),
          (bool, {'result': True}, True),
          (float, {'result': 3.14}, 3.14),
          (list[str], {'result': ['a', 'b', 'c']}, ['a', 'b', 'c']),
          (list[int], {'result': [1, 2, 3]}, [1, 2, 3]),
          (
              list[SampleOutputSchema],
              {'result': [{'result': 'ok', 'count': 1}]},
              [{'result': 'ok', 'count': 1}],
          ),
          (list[str], {'result': []}, []),
      ],
      ids=[
          'BaseModel',
          'dict',
          'str',
          'int',
          'bool',
          'float',
          'list_str',
          'list_int',
          'list_BaseModel',
          'list_str_empty',
      ],
  )
  @pytest.mark.asyncio
  async def test_run_async(self, output_schema, args, expected_output):
    """Verify run_async validates and extracts output for each schema type."""
    agent = _make_task_agent(output_schema=output_schema)
    tool = FinishTaskTool(task_agent=agent)
    mock_tool_context = mock.MagicMock()

    result = await tool.run_async(
        args=args,
        tool_context=mock_tool_context,
    )

    # Task Delegation API: tool no longer writes actions.finish_task. The
    # LlmAgent task wrapper sniffs the FC's `output` arg directly to
    # set event.output. The tool just performs schema validation.
    assert result == 'Task completed.'
    del expected_output  # validated_output is no longer surfaced via actions

  def test_get_declaration_list_basemodel_defs_at_root(self):
    """Non-object schemas with $defs should hoist $defs to root level."""
    agent = _make_task_agent(output_schema=list[SampleOutputSchema])
    tool = FinishTaskTool(task_agent=agent)
    declaration = tool._get_declaration()

    schema = declaration.parameters_json_schema
    # $defs must be at the root, not nested inside properties.result
    assert '$defs' in schema
    assert 'SampleOutputSchema' in schema['$defs']
    # The nested result schema should not contain $defs
    assert '$defs' not in schema['properties']['result']

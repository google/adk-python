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

"""Tests for TransferToAgentTool enum constraint functionality."""

from unittest.mock import patch

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.genai import types


def test_transfer_to_agent_tool_enum_constraint():
  """Test that TransferToAgentTool adds enum constraint to agent_name."""
  agent_names = ['agent_a', 'agent_b', 'agent_c']
  tool = TransferToAgentTool(agent_names=agent_names)

  decl = tool._get_declaration()

  assert decl is not None
  assert decl.name == 'transfer_to_agent'
  assert decl.parameters is not None
  assert decl.parameters.type == types.Type.OBJECT
  assert 'agent_name' in decl.parameters.properties

  agent_name_schema = decl.parameters.properties['agent_name']
  assert agent_name_schema.type == types.Type.STRING
  assert agent_name_schema.enum == agent_names

  # Verify that agent_name is marked as required
  assert decl.parameters.required == ['agent_name']


def test_transfer_to_agent_tool_single_agent():
  """Test TransferToAgentTool with a single agent."""
  tool = TransferToAgentTool(agent_names=['single_agent'])

  decl = tool._get_declaration()

  assert decl is not None
  agent_name_schema = decl.parameters.properties['agent_name']
  assert agent_name_schema.enum == ['single_agent']


def test_transfer_to_agent_tool_multiple_agents():
  """Test TransferToAgentTool with multiple agents."""
  agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
  tool = TransferToAgentTool(agent_names=agent_names)

  decl = tool._get_declaration()

  assert decl is not None
  agent_name_schema = decl.parameters.properties['agent_name']
  assert agent_name_schema.enum == agent_names
  assert len(agent_name_schema.enum) == 5


def test_transfer_to_agent_tool_empty_list():
  """Test TransferToAgentTool with an empty agent list."""
  tool = TransferToAgentTool(agent_names=[])

  decl = tool._get_declaration()

  assert decl is not None
  agent_name_schema = decl.parameters.properties['agent_name']
  assert agent_name_schema.enum == []


def test_transfer_to_agent_tool_preserves_description():
  """Test that TransferToAgentTool preserves the original description."""
  tool = TransferToAgentTool(agent_names=['agent_a', 'agent_b'])

  decl = tool._get_declaration()

  assert decl is not None
  assert decl.description is not None
  assert 'Transfer the question to another agent' in decl.description


def test_transfer_to_agent_tool_preserves_parameter_type():
  """Test that TransferToAgentTool preserves the parameter type."""
  tool = TransferToAgentTool(agent_names=['agent_a'])

  decl = tool._get_declaration()

  assert decl is not None
  agent_name_schema = decl.parameters.properties['agent_name']
  # Should still be a string type, just with enum constraint
  assert agent_name_schema.type == types.Type.STRING


def test_transfer_to_agent_tool_no_extra_parameters():
  """Test that TransferToAgentTool doesn't add extra parameters."""
  tool = TransferToAgentTool(agent_names=['agent_a'])

  decl = tool._get_declaration()

  assert decl is not None
  # Should only have agent_name parameter (tool_context is ignored)
  assert len(decl.parameters.properties) == 1
  assert 'agent_name' in decl.parameters.properties
  assert 'tool_context' not in decl.parameters.properties


def test_transfer_to_agent_tool_maintains_inheritance():
  """Test that TransferToAgentTool inherits from FunctionTool correctly."""
  tool = TransferToAgentTool(agent_names=['agent_a'])

  assert isinstance(tool, FunctionTool)
  assert hasattr(tool, '_get_declaration')
  assert hasattr(tool, 'process_llm_request')


def test_transfer_to_agent_tool_handles_parameters_json_schema():
  """Test that TransferToAgentTool handles parameters_json_schema format."""
  agent_names = ['agent_x', 'agent_y', 'agent_z']

  # Create a mock FunctionDeclaration with parameters_json_schema
  mock_decl = type('MockDecl', (), {})()
  mock_decl.parameters = None  # No Schema object
  mock_decl.parameters_json_schema = {
      'type': 'object',
      'properties': {
          'agent_name': {
              'type': 'string',
              'description': 'Agent name to transfer to',
          }
      },
      'required': ['agent_name'],
  }

  # Temporarily patch FunctionTool._get_declaration
  with patch.object(
      FunctionTool,
      '_get_declaration',
      return_value=mock_decl,
  ):
    tool = TransferToAgentTool(agent_names=agent_names)
    result = tool._get_declaration()

  # Verify enum was added to parameters_json_schema
  assert result.parameters_json_schema is not None
  assert 'agent_name' in result.parameters_json_schema['properties']
  assert (
      result.parameters_json_schema['properties']['agent_name']['enum']
      == agent_names
  )
  assert (
      result.parameters_json_schema['properties']['agent_name']['type']
      == 'string'
  )
  # Verify required field is preserved
  assert result.parameters_json_schema['required'] == ['agent_name']

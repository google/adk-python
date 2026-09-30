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

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.genai import types
import pytest


class TestTransferToAgentToolLegacy:
  """Tests for TransferToAgentTool when JSON_SCHEMA_FOR_FUNC_DECL is disabled."""

  @pytest.fixture(autouse=True)
  def disable_feature_flag(self):
    """Disable the JSON_SCHEMA_FOR_FUNC_DECL feature flag for legacy tests."""
    with temporary_feature_override(
        FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False
    ):
      yield

  def test_transfer_to_agent_tool_enum_constraint(self):
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

    # By default, transfer_reason is not included in parameters or description
    assert 'transfer_reason' not in decl.parameters.properties
    assert 'transfer_reason' not in decl.description
    assert decl.parameters.required == ['agent_name']

  def test_transfer_to_agent_tool_with_transfer_reason(self):
    """Test TransferToAgentTool when include_transfer_reason=True."""
    agent_names = ['agent_a', 'agent_b', 'agent_c']
    tool = TransferToAgentTool(
        agent_names=agent_names, include_transfer_reason=True
    )

    decl = tool._get_declaration()

    assert decl is not None
    assert 'agent_name' in decl.parameters.properties
    assert 'transfer_reason' in decl.parameters.properties
    assert 'transfer_reason' in decl.description

    transfer_reason_schema = decl.parameters.properties['transfer_reason']
    assert transfer_reason_schema.type == types.Type.STRING
    assert transfer_reason_schema.enum is None

    # Verify that only agent_name is marked as required (transfer_reason is optional)
    assert decl.parameters.required == ['agent_name']

  def test_transfer_to_agent_tool_single_agent(self):
    """Test TransferToAgentTool with a single agent."""
    tool = TransferToAgentTool(agent_names=['single_agent'])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters.properties['agent_name']
    assert agent_name_schema.enum == ['single_agent']

  def test_transfer_to_agent_tool_multiple_agents(self):
    """Test TransferToAgentTool with multiple agents."""
    agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
    tool = TransferToAgentTool(agent_names=agent_names)

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters.properties['agent_name']
    assert agent_name_schema.enum == agent_names
    assert len(agent_name_schema.enum) == 5

  def test_transfer_to_agent_tool_empty_list(self):
    """Test TransferToAgentTool with an empty agent list."""
    tool = TransferToAgentTool(agent_names=[])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters.properties['agent_name']
    assert agent_name_schema.enum == []

  def test_transfer_to_agent_tool_preserves_parameter_type(self):
    """Test that TransferToAgentTool preserves the parameter type."""
    tool = TransferToAgentTool(agent_names=['agent_a'])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters.properties['agent_name']
    # Should still be a string type, just with enum constraint
    assert agent_name_schema.type == types.Type.STRING

  def test_transfer_to_agent_tool_no_extra_parameters(self):
    """Test that TransferToAgentTool doesn't add extra parameters."""
    tool = TransferToAgentTool(agent_names=['agent_a'])

    decl = tool._get_declaration()

    assert decl is not None
    # Should only have agent_name parameter by default (tool_context and transfer_reason are ignored)
    assert len(decl.parameters.properties) == 1
    assert 'agent_name' in decl.parameters.properties
    assert 'transfer_reason' not in decl.parameters.properties
    assert 'tool_context' not in decl.parameters.properties

    tool_with_reason = TransferToAgentTool(
        agent_names=['agent_a'], include_transfer_reason=True
    )
    decl_with_reason = tool_with_reason._get_declaration()
    assert decl_with_reason is not None
    assert len(decl_with_reason.parameters.properties) == 2
    assert 'agent_name' in decl_with_reason.parameters.properties
    assert 'transfer_reason' in decl_with_reason.parameters.properties


# Shared/Common tests at module level
def test_transfer_to_agent_tool_preserves_description():
  """Test that TransferToAgentTool preserves the original description."""
  tool = TransferToAgentTool(agent_names=['agent_a', 'agent_b'])

  decl = tool._get_declaration()

  assert decl is not None
  assert decl.description is not None
  assert 'Transfer the query to another agent' in decl.description
  assert 'transfer_reason' not in decl.description
  assert 'transfer_reason' not in tool.description
  assert decl.description == tool.description
  assert '\n  Use this tool' not in decl.description
  assert '\nUse this tool' in decl.description


def test_transfer_to_agent_tool_with_reason_includes_reason_in_description():
  """Test that TransferToAgentTool includes transfer_reason in description when enabled."""
  tool = TransferToAgentTool(
      agent_names=['agent_a', 'agent_b'], include_transfer_reason=True
  )

  decl = tool._get_declaration()

  assert decl is not None
  assert decl.description is not None
  assert 'Transfer the query to another agent' in decl.description
  assert 'transfer_reason' in decl.description
  assert 'transfer_reason' in tool.description
  assert decl.description == tool.description


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


class TestTransferToAgentToolWithJsonSchema:
  """Tests for TransferToAgentTool when JSON_SCHEMA_FOR_FUNC_DECL is enabled."""

  @pytest.fixture(autouse=True)
  def enable_feature_flag(self):
    """Enable the JSON_SCHEMA_FOR_FUNC_DECL feature flag."""
    with temporary_feature_override(
        FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True
    ):
      yield

  def test_transfer_to_agent_tool_enum_constraint(self):
    """Test that TransferToAgentTool adds enum constraint to parameters_json_schema."""
    agent_names = ['agent_a', 'agent_b', 'agent_c']
    tool = TransferToAgentTool(agent_names=agent_names)

    decl = tool._get_declaration()

    assert decl is not None
    assert decl.name == 'transfer_to_agent'
    assert decl.parameters_json_schema is not None
    assert 'agent_name' in decl.parameters_json_schema['properties']

    agent_name_schema = decl.parameters_json_schema['properties']['agent_name']
    assert agent_name_schema['type'] == 'string'
    assert agent_name_schema['enum'] == agent_names

    # By default, transfer_reason is not included in parameters_json_schema or description
    assert 'transfer_reason' not in decl.parameters_json_schema['properties']
    assert 'transfer_reason' not in decl.description

    assert decl.parameters_json_schema['required'] == [
        'agent_name',
    ]

  def test_transfer_to_agent_tool_with_transfer_reason(self):
    """Test TransferToAgentTool with include_transfer_reason=True in JSON schema mode."""
    agent_names = ['agent_a', 'agent_b', 'agent_c']
    tool = TransferToAgentTool(
        agent_names=agent_names, include_transfer_reason=True
    )

    decl = tool._get_declaration()

    assert decl is not None
    assert 'agent_name' in decl.parameters_json_schema['properties']
    assert 'transfer_reason' in decl.parameters_json_schema['properties']
    assert 'transfer_reason' in decl.description

    transfer_reason_schema = decl.parameters_json_schema['properties'][
        'transfer_reason'
    ]
    assert transfer_reason_schema['type'] == 'string'
    assert 'enum' not in transfer_reason_schema

    assert decl.parameters_json_schema['required'] == [
        'agent_name',
    ]

  def test_transfer_to_agent_tool_single_agent(self):
    """Test TransferToAgentTool with a single agent."""
    tool = TransferToAgentTool(agent_names=['single_agent'])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters_json_schema['properties']['agent_name']
    assert agent_name_schema['enum'] == ['single_agent']

  def test_transfer_to_agent_tool_multiple_agents(self):
    """Test TransferToAgentTool with multiple agents."""
    agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']
    tool = TransferToAgentTool(agent_names=agent_names)

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters_json_schema['properties']['agent_name']
    assert agent_name_schema['enum'] == agent_names
    assert len(agent_name_schema['enum']) == 5

  def test_transfer_to_agent_tool_empty_list(self):
    """Test TransferToAgentTool with an empty agent list."""
    tool = TransferToAgentTool(agent_names=[])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters_json_schema['properties']['agent_name']
    assert agent_name_schema['enum'] == []

  def test_transfer_to_agent_tool_preserves_parameter_type(self):
    """Test that TransferToAgentTool preserves the parameter type."""
    tool = TransferToAgentTool(agent_names=['agent_a'])

    decl = tool._get_declaration()

    assert decl is not None
    agent_name_schema = decl.parameters_json_schema['properties']['agent_name']
    assert agent_name_schema['type'] == 'string'

  def test_transfer_to_agent_tool_no_extra_parameters(self):
    """Test that TransferToAgentTool doesn't add extra parameters."""
    tool = TransferToAgentTool(agent_names=['agent_a'])

    decl = tool._get_declaration()

    assert decl is not None
    assert len(decl.parameters_json_schema['properties']) == 1
    assert 'agent_name' in decl.parameters_json_schema['properties']
    assert 'transfer_reason' not in decl.parameters_json_schema['properties']
    assert 'tool_context' not in decl.parameters_json_schema['properties']

    tool_with_reason = TransferToAgentTool(
        agent_names=['agent_a'], include_transfer_reason=True
    )
    decl_with_reason = tool_with_reason._get_declaration()
    assert decl_with_reason is not None
    assert len(decl_with_reason.parameters_json_schema['properties']) == 2
    assert 'agent_name' in decl_with_reason.parameters_json_schema['properties']
    assert (
        'transfer_reason'
        in decl_with_reason.parameters_json_schema['properties']
    )


def test_transfer_to_agent_function_sets_reason():
  from unittest.mock import MagicMock

  from google.adk.events.event_actions import EventActions
  from google.adk.tools.transfer_to_agent_tool import transfer_to_agent

  mock_context = MagicMock()
  mock_context.actions = EventActions()

  transfer_to_agent('target_agent', mock_context, transfer_reason='because')

  assert mock_context.actions.transfer_to_agent == 'target_agent'
  assert mock_context.actions.transfer_reason == 'because'


def test_transfer_to_agent_function_without_reason():
  from unittest.mock import MagicMock

  from google.adk.events.event_actions import EventActions
  from google.adk.tools.transfer_to_agent_tool import transfer_to_agent

  mock_context = MagicMock()
  mock_context.actions = EventActions()

  transfer_to_agent('target_agent', mock_context)

  assert mock_context.actions.transfer_to_agent == 'target_agent'
  assert mock_context.actions.transfer_reason is None


async def test_transfer_to_agent_tool_run_async_default():
  from unittest.mock import MagicMock

  from google.adk.events.event_actions import EventActions

  mock_context = MagicMock()
  mock_context.actions = EventActions()
  mock_context._invocation_context = None

  tool = TransferToAgentTool(agent_names=['target_agent'])
  await tool.run_async(
      args={'agent_name': 'target_agent'}, tool_context=mock_context
  )

  assert mock_context.actions.transfer_to_agent == 'target_agent'
  assert mock_context.actions.transfer_reason is None


async def test_transfer_to_agent_tool_run_async_with_reason():
  from unittest.mock import MagicMock

  from google.adk.events.event_actions import EventActions

  mock_context = MagicMock()
  mock_context.actions = EventActions()
  mock_context._invocation_context = None

  tool = TransferToAgentTool(
      agent_names=['target_agent'], include_transfer_reason=True
  )
  await tool.run_async(
      args={'agent_name': 'target_agent', 'transfer_reason': 'escalation'},
      tool_context=mock_context,
  )

  assert mock_context.actions.transfer_to_agent == 'target_agent'
  assert mock_context.actions.transfer_reason == 'escalation'

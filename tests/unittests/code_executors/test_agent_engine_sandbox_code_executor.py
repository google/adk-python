# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.agent_engine_sandbox_code_executor import AgentEngineSandboxCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.sessions.session import Session
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Fixture for a mock InvocationContext."""
  mock = MagicMock(spec=InvocationContext)
  mock.invocation_id = "test-invocation-123"
  session = MagicMock(spec=Session)
  mock.session = session
  session.state = {}

  return mock


class TestAgentEngineSandboxCodeExecutor:
  """Unit tests for the AgentEngineSandboxCodeExecutor."""

  def test_init_with_sandbox_overrides(self):
    """Tests that class attributes can be overridden at instantiation."""
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789",
    )
    assert executor.sandbox_resource_name == (
        "projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )

  def test_init_with_sandbox_overrides_throws_error(self):
    """Tests that class attributes can be overridden at instantiation."""
    with pytest.raises(ValueError):
      AgentEngineSandboxCodeExecutor(
          sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxes/789",
      )

  def test_init_with_agent_engine_overrides_throws_error(self):
    """Tests that class attributes can be overridden at instantiation."""
    with pytest.raises(ValueError):
      AgentEngineSandboxCodeExecutor(
          agent_engine_resource_name=(
              "projects/123/locations/us-central1/reason/456"
          ),
      )

  @patch("vertexai.Client")
  def test_execute_code_success(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"msg_out": "hello world", "msg_err": ""}
    ).encode("utf-8")
    mock_json_output.metadata = None

    mock_file_output = MagicMock()
    mock_file_output.mime_type = "text/plain"
    mock_file_output.data = b"file content"
    mock_file_output.metadata = MagicMock()
    mock_file_output.metadata.attributes = {"file_name": b"file.txt"}

    mock_png_file_output = MagicMock()
    mock_png_file_output.mime_type = "image/png"
    sample_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    mock_png_file_output.data = sample_png_bytes
    mock_png_file_output.metadata = MagicMock()
    mock_png_file_output.metadata.attributes = {"file_name": b"file.png"}

    mock_response.outputs = [
        mock_json_output,
        mock_file_output,
        mock_png_file_output,
    ]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert
    assert result.stdout == "hello world"
    assert not result.stderr
    assert result.output_files[0].mime_type == "text/plain"
    assert result.output_files[0].content == b"file content"

    assert result.output_files[0].name == "file.txt"
    assert result.output_files[1].mime_type == "image/png"
    assert result.output_files[1].name == "file.png"
    assert result.output_files[1].content == sample_png_bytes
    mock_api_client.agent_engines.sandboxes.execute_code.assert_called_once_with(
        name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789",
        input_data={"code": 'print("hello world")'},
    )

  @patch("vertexai.Client")
  def test_execute_code_recreates_sandbox_when_get_returns_none(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client

    # Existing sandbox name stored in session, but get() will return None
    existing_sandbox_name = "projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/old"
    mock_invocation_context.session.state = {
        "sandbox_name": existing_sandbox_name
    }

    # Mock get to return None (simulating missing/expired sandbox)
    mock_api_client.agent_engines.sandboxes.get.return_value = None

    # Mock create operation to return a new sandbox resource name
    operation_mock = MagicMock()
    created_sandbox_name = "projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    operation_mock.response.name = created_sandbox_name
    mock_api_client.agent_engines.sandboxes.create.return_value = operation_mock

    # Mock execute_code response
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"stdout": "recreated sandbox run", "stderr": ""}
    ).encode("utf-8")
    mock_json_output.metadata = None
    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute using agent_engine_resource_name so a sandbox can be created
    executor = AgentEngineSandboxCodeExecutor(
        agent_engine_resource_name=(
            "projects/123/locations/us-central1/reasoningEngines/456"
        )
    )
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert get was called for the existing sandbox
    mock_api_client.agent_engines.sandboxes.get.assert_called_once_with(
        name=existing_sandbox_name
    )

    # Assert create was called and session updated with new sandbox
    mock_api_client.agent_engines.sandboxes.create.assert_called_once()
    assert (
        mock_invocation_context.session.state["sandbox_name"]
        == created_sandbox_name
    )

    # Assert execute_code used the created sandbox name
    mock_api_client.agent_engines.sandboxes.execute_code.assert_called_once_with(
        name=created_sandbox_name,
        input_data={"code": 'print("hello world")'},
    )

  @patch("vertexai.Client")
  def test_execute_code_creates_sandbox_if_missing(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client

    # Mock create operation to return a sandbox resource name
    operation_mock = MagicMock()
    created_sandbox_name = "projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    operation_mock.response.name = created_sandbox_name
    mock_api_client.agent_engines.sandboxes.create.return_value = operation_mock

    # Mock execute_code response
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"stdout": "created sandbox run", "stderr": ""}
    ).encode("utf-8")
    mock_json_output.metadata = None
    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Ensure session.state behaves like a dict for storing sandbox_name
    mock_invocation_context.session.state = {}

    # Execute using agent_engine_resource_name so a sandbox will be created
    executor = AgentEngineSandboxCodeExecutor(
        agent_engine_resource_name=(
            "projects/123/locations/us-central1/reasoningEngines/456"
        ),
        sandbox_resource_name=None,
    )
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert sandbox creation was called and session state updated
    mock_api_client.agent_engines.sandboxes.create.assert_called_once()
    create_call_kwargs = (
        mock_api_client.agent_engines.sandboxes.create.call_args.kwargs
    )
    assert create_call_kwargs["name"] == (
        "projects/123/locations/us-central1/reasoningEngines/456"
    )
    assert (
        mock_invocation_context.session.state["sandbox_name"]
        == created_sandbox_name
    )

    # Assert execute_code used the created sandbox name
    mock_api_client.agent_engines.sandboxes.execute_code.assert_called_once_with(
        name=created_sandbox_name,
        input_data={"code": 'print("hello world")'},
    )

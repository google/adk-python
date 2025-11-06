# Copyright 2025 Google LLC
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
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Fixture for a mock InvocationContext."""
  mock = MagicMock(spec=InvocationContext)
  mock.invocation_id = "test-invocation-123"
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
        {"stdout": "hello world", "stderr": ""}
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

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
from google.adk.code_executors.agent_engine_sandbox_code_executor import (
    AgentEngineSandboxCodeExecutor,
)
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import File
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

  @patch("vertexai.Client")
  def test_execute_code_with_msg_out_msg_err(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that msg_out and msg_err fields from API response are parsed correctly."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"msg_out": "hello from msg_out", "msg_err": "error from msg_err"}
    ).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - msg_out/msg_err should be mapped to stdout/stderr
    assert result.stdout == "hello from msg_out"
    assert result.stderr == "error from msg_err"

  @patch("vertexai.Client")
  def test_execute_code_fallback_to_stdout_stderr(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests fallback to stdout/stderr when msg_out/msg_err are not present."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"stdout": "fallback stdout", "stderr": "fallback stderr"}
    ).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - should fall back to stdout/stderr
    assert result.stdout == "fallback stdout"
    assert result.stderr == "fallback stderr"

  @patch("vertexai.Client")
  def test_execute_code_msg_out_takes_precedence_over_stdout(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that msg_out takes precedence over stdout when both are present."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    # Both msg_out and stdout present - msg_out should win
    mock_json_output.data = json.dumps({
        "msg_out": "primary output",
        "msg_err": "primary error",
        "stdout": "fallback output",
        "stderr": "fallback error",
    }).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - msg_out/msg_err take precedence
    assert result.stdout == "primary output"
    assert result.stderr == "primary error"

  @patch("vertexai.Client")
  def test_execute_code_msg_out_null_ignores_stdout(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that msg_out=None does not fall back to stdout."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"msg_out": None, "stdout": "fallback output"}
    ).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - msg_out is authoritative even when null
    assert result.stdout == ""

  @patch("vertexai.Client")
  def test_execute_code_msg_err_null_ignores_stderr(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that msg_err=None does not fall back to stderr."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps(
        {"msg_err": None, "stderr": "fallback error"}
    ).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - msg_err is authoritative even when null
    assert result.stderr == ""

  @patch("vertexai.Client")
  def test_execute_code_partial_response_only_msg_out(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests handling when only msg_out is present (no msg_err)."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({"msg_out": "only output"}).encode(
        "utf-8"
    )
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - stdout populated, stderr empty string (not None)
    assert result.stdout == "only output"
    assert result.stderr == ""

  @patch("vertexai.Client")
  def test_execute_code_partial_response_only_msg_err(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests handling when only msg_err is present (no msg_out)."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({"msg_err": "only error"}).encode(
        "utf-8"
    )
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - stderr populated, stdout empty string (not None)
    assert result.stdout == ""
    assert result.stderr == "only error"

  @patch("vertexai.Client")
  def test_execute_code_empty_response(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests handling when response has no stdout/stderr fields."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({}).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - both should be empty strings
    assert result.stdout == ""
    assert result.stderr == ""

  @patch("vertexai.Client")
  def test_execute_code_invalid_json_does_not_raise(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that invalid JSON is handled without raising."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = b"{not json}"
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    try:
      result = executor.execute_code(mock_invocation_context, code_input)
    except json.JSONDecodeError as exc:
      pytest.fail(f"Expected invalid JSON to be handled: {exc}")

    # Assert - invalid JSON yields empty stdout/stderr
    assert result.stdout == ""
    assert result.stderr == ""

  @patch("vertexai.Client")
  def test_execute_code_with_input_files_uses_correct_payload_keys(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that input files use content and mime_type keys (not contents/mimeType)."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({"msg_out": "ok", "msg_err": ""}).encode(
        "utf-8"
    )
    mock_json_output.metadata = None
    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute with input files
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    input_files = [
        File(name="data.csv", content=b"col1,col2\n1,2", mime_type="text/csv"),
        File(
            name="config.json",
            content=b'{"key": "value"}',
            mime_type="application/json",
        ),
    ]
    code_input = CodeExecutionInput(
        code='import pandas as pd; df = pd.read_csv("data.csv")',
        input_files=input_files,
    )
    executor.execute_code(mock_invocation_context, code_input)

    # Assert - verify the payload uses correct keys
    call_args = mock_api_client.agent_engines.sandboxes.execute_code.call_args
    input_data = call_args.kwargs["input_data"]

    assert "files" in input_data
    assert len(input_data["files"]) == 2

    # Check first file
    file1 = input_data["files"][0]
    assert file1["name"] == "data.csv"
    assert file1["content"] == b"col1,col2\n1,2"
    assert file1["mime_type"] == "text/csv"
    # Ensure old keys are NOT present
    assert "contents" not in file1
    assert "mimeType" not in file1

    # Check second file
    file2 = input_data["files"][1]
    assert file2["name"] == "config.json"
    assert file2["content"] == b'{"key": "value"}'
    assert file2["mime_type"] == "application/json"
    assert "contents" not in file2
    assert "mimeType" not in file2

  @patch("vertexai.Client")
  def test_execute_code_without_input_files_no_files_key(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that files key is not present when no input files are provided."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({"msg_out": "ok"}).encode("utf-8")
    mock_json_output.metadata = None
    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute without input files
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    executor.execute_code(mock_invocation_context, code_input)

    # Assert - verify no files key in payload
    call_args = mock_api_client.agent_engines.sandboxes.execute_code.call_args
    input_data = call_args.kwargs["input_data"]
    assert "files" not in input_data
    assert input_data == {"code": 'print("hello")'}

  @patch("vertexai.Client")
  def test_execute_code_output_files_metadata_preserved(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that output file metadata is correctly preserved."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()

    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    mock_json_output.data = json.dumps({"msg_out": "done"}).encode("utf-8")
    mock_json_output.metadata = None

    mock_csv_output = MagicMock()
    mock_csv_output.mime_type = "text/csv"
    mock_csv_output.data = b"a,b,c\n1,2,3"
    mock_csv_output.metadata = MagicMock()
    mock_csv_output.metadata.attributes = {"file_name": b"output.csv"}

    mock_response.outputs = [mock_json_output, mock_csv_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='df.to_csv("output.csv")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert
    assert result.stdout == "done"
    assert len(result.output_files) == 1
    assert result.output_files[0].name == "output.csv"
    assert result.output_files[0].content == b"a,b,c\n1,2,3"
    assert result.output_files[0].mime_type == "text/csv"

  @patch("vertexai.Client")
  def test_execute_code_non_dict_json_response_logs_warning(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that non-dict JSON responses are handled gracefully with a warning."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    # Return a list instead of a dict - this is valid JSON but not expected
    mock_json_output.data = json.dumps(["unexpected", "list"]).encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - should handle gracefully with empty stdout/stderr
    assert result.stdout == ""
    assert result.stderr == ""

  @patch("vertexai.Client")
  def test_execute_code_string_json_response_logs_warning(
      self,
      mock_vertexai_client,
      mock_invocation_context,
  ):
    """Tests that string JSON responses are handled gracefully."""
    # Setup Mocks
    mock_api_client = MagicMock()
    mock_vertexai_client.return_value = mock_api_client
    mock_response = MagicMock()
    mock_json_output = MagicMock()
    mock_json_output.mime_type = "application/json"
    # Return a string instead of a dict
    mock_json_output.data = json.dumps("just a string").encode("utf-8")
    mock_json_output.metadata = None

    mock_response.outputs = [mock_json_output]
    mock_api_client.agent_engines.sandboxes.execute_code.return_value = (
        mock_response
    )

    # Execute
    executor = AgentEngineSandboxCodeExecutor(
        sandbox_resource_name="projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789"
    )
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert - should handle gracefully with empty stdout/stderr
    assert result.stdout == ""
    assert result.stderr == ""

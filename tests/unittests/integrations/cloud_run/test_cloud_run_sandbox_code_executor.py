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

import sys
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.integrations.cloud_run import CloudRunSandboxCodeExecutor
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Provides a mock InvocationContext."""
  mock_agent = MagicMock(spec=BaseAgent)
  mock_session = MagicMock(spec=Session)
  mock_session_service = MagicMock(spec=BaseSessionService)
  return InvocationContext(
      invocation_id="test_invocation",
      agent=mock_agent,
      session=mock_session,
      session_service=mock_session_service,
  )


class TestCloudRunSandboxCodeExecutor:

  def test_init_default(self):
    executor = CloudRunSandboxCodeExecutor()
    assert not executor.stateful
    assert not executor.optimize_data_file
    assert executor.sandbox_bin == "/usr/local/gcp/bin/sandbox"
    assert not executor.allow_egress

  def test_init_stateful_raises_error(self):
    with pytest.raises(
        ValueError,
        match="Cannot set `stateful=True` in CloudRunSandboxCodeExecutor.",
    ):
      CloudRunSandboxCodeExecutor(stateful=True)

  def test_init_optimize_data_file_raises_error(self):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot set `optimize_data_file=True` in"
            " CloudRunSandboxCodeExecutor."
        ),
    ):
      CloudRunSandboxCodeExecutor(optimize_data_file=True)

  @patch("subprocess.run")
  def test_execute_code_success(
      self, mock_run, mock_invocation_context: InvocationContext
  ):
    # Setup mock subprocess.run response
    mock_response = MagicMock()
    mock_response.stdout = "hello world\n"
    mock_response.stderr = ""
    mock_response.returncode = 0
    mock_run.return_value = mock_response

    executor = CloudRunSandboxCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == "hello world\n"
    assert result.stderr == ""
    assert result.output_files == []

    # Verify subprocess.run was called with correct arguments
    expected_python = sys.executable or "python3"
    mock_run.assert_called_once_with(
        ["/usr/local/gcp/bin/sandbox", "do", expected_python],
        input='print("hello world")',
        capture_output=True,
        text=True,
        timeout=None,
        check=False,
    )

  @patch("subprocess.run")
  def test_execute_code_with_egress_and_custom_bin(
      self, mock_run, mock_invocation_context: InvocationContext
  ):
    mock_response = MagicMock()
    mock_response.stdout = "egress success\n"
    mock_response.stderr = ""
    mock_response.returncode = 0
    mock_run.return_value = mock_response

    executor = CloudRunSandboxCodeExecutor(
        sandbox_bin="/usr/bin/custom-sandbox",
        allow_egress=True,
        timeout_seconds=10,
    )
    code_input = CodeExecutionInput(code="import requests; print('ok')")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "egress success\n"

    expected_python = sys.executable or "python3"
    mock_run.assert_called_once_with(
        ["/usr/bin/custom-sandbox", "do", "--allow-egress", expected_python],
        input="import requests; print('ok')",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

  @patch("subprocess.run")
  def test_execute_code_with_error(
      self, mock_run, mock_invocation_context: InvocationContext
  ):
    mock_response = MagicMock()
    mock_response.stdout = ""
    mock_response.stderr = "Traceback ... ValueError: Test error\n"
    mock_response.returncode = 1
    mock_run.return_value = mock_response

    executor = CloudRunSandboxCodeExecutor()
    code_input = CodeExecutionInput(code='raise ValueError("Test error")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert "ValueError: Test error" in result.stderr

  @patch("subprocess.run")
  def test_execute_code_timeout(
      self, mock_run, mock_invocation_context: InvocationContext
  ):
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["sandbox", "do", "python3"],
        timeout=5,
        output="partial stdout",
        stderr="partial stderr",
    )

    executor = CloudRunSandboxCodeExecutor(timeout_seconds=5)
    code_input = CodeExecutionInput(code="import time\ntime.sleep(10)")
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"

  @patch("subprocess.run")
  def test_execute_code_binary_not_found(
      self, mock_run, mock_invocation_context: InvocationContext
  ):
    mock_run.side_effect = FileNotFoundError(
        "[Errno 2] No such file or directory: 'sandbox'"
    )

    executor = CloudRunSandboxCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == ""
    assert (
        'Sandbox binary "/usr/local/gcp/bin/sandbox" not found' in result.stderr
    )

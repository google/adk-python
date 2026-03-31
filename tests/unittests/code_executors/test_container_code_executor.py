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

"""Tests for ContainerCodeExecutor timeout enforcement."""

import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

pytest.importorskip('docker', reason='docker SDK not installed')

from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.container_code_executor import ContainerCodeExecutor


@pytest.fixture
def mock_docker():
  with patch('google.adk.code_executors.container_code_executor.docker') as m:
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = 'test-container-123'
    mock_client.containers.run.return_value = mock_container
    # verify_python_installation needs exit_code=0
    mock_container.exec_run.return_value = MagicMock(exit_code=0)
    m.from_env.return_value = mock_client
    yield mock_client, mock_container


@pytest.fixture
def mock_invocation_context():
  return MagicMock()


class TestContainerTimeout:

  def test_timeout_returns_error(self, mock_docker, mock_invocation_context):
    """exec_start that blocks is killed after timeout_seconds."""
    client, container = mock_docker

    def slow_exec_start(exec_id, **kwargs):
      time.sleep(10)
      return (b'output', b'')

    client.api.exec_create.return_value = {'Id': 'exec-123'}
    client.api.exec_start.side_effect = slow_exec_start
    client.api.exec_inspect.return_value = {'Pid': 42}

    executor = ContainerCodeExecutor(image='test:latest', timeout_seconds=1)
    code_input = CodeExecutionInput(code='time.sleep(999)')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert 'timed out' in result.stderr.lower()
    assert result.stdout == ''

    # Verify the best-effort kill cleanup path was exercised
    client.api.exec_inspect.assert_called_once_with('exec-123')
    container.exec_run.assert_called_with(['kill', '-9', '42'])

  def test_no_timeout_waits_indefinitely(
      self, mock_docker, mock_invocation_context
  ):
    """When timeout_seconds is None, wait for completion."""
    client, container = mock_docker
    client.api.exec_create.return_value = {'Id': 'exec-456'}
    client.api.exec_start.return_value = (b'hello\n', b'')

    executor = ContainerCodeExecutor(image='test:latest')
    code_input = CodeExecutionInput(code='print("hello")')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == 'hello\n'
    assert result.stderr == ''

  def test_normal_execution_within_timeout(
      self, mock_docker, mock_invocation_context
  ):
    """Fast command completes within timeout."""
    client, container = mock_docker
    client.api.exec_create.return_value = {'Id': 'exec-789'}
    client.api.exec_start.return_value = (b'42\n', b'')

    executor = ContainerCodeExecutor(image='test:latest', timeout_seconds=30)
    code_input = CodeExecutionInput(code='print(42)')
    result = executor.execute_code(mock_invocation_context, code_input)

    assert result.stdout == '42\n'
    assert result.stderr == ''

  def test_exec_start_exception_propagated(
      self, mock_docker, mock_invocation_context
  ):
    """Docker API error in exec_start is re-raised, not swallowed."""
    client, container = mock_docker
    client.api.exec_create.return_value = {'Id': 'exec-err'}
    client.api.exec_start.side_effect = RuntimeError(
        'Docker API connection lost'
    )

    executor = ContainerCodeExecutor(image='test:latest', timeout_seconds=30)
    code_input = CodeExecutionInput(code='print(1)')
    with pytest.raises(RuntimeError, match='Docker API connection'):
      executor.execute_code(mock_invocation_context, code_input)

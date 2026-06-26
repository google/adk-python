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

"""Tests for the ContainerCodeExecutor container hardening defaults."""

from unittest import mock

from google.adk.code_executors.container_code_executor import ContainerCodeExecutor


def _mock_docker_client():
  """Returns a mock Docker client whose container passes python verification."""
  client = mock.MagicMock()
  container = mock.MagicMock()
  # `_verify_python_installation` runs `exec_run(['which', 'python3'])` and
  # checks `exit_code == 0`.
  container.exec_run.return_value = mock.MagicMock(exit_code=0)
  client.containers.run.return_value = container
  return client


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_container_is_hardened_by_default(mock_docker):
  """Networking is disabled and privileges are dropped by default."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client

  ContainerCodeExecutor(image='test-image')

  _, kwargs = client.containers.run.call_args
  # Untrusted model-generated code must not be able to reach the network
  # (e.g. the cloud metadata endpoint) or escalate privileges by default.
  assert kwargs['network_disabled']
  assert kwargs['cap_drop'] == ['ALL']
  assert kwargs['security_opt'] == ['no-new-privileges']


@mock.patch('google.adk.code_executors.container_code_executor.docker')
def test_container_network_can_be_explicitly_enabled(mock_docker):
  """Networking is left enabled when the caller opts in."""
  client = _mock_docker_client()
  mock_docker.from_env.return_value = client

  ContainerCodeExecutor(image='test-image', network_enabled=True)

  _, kwargs = client.containers.run.call_args
  assert not kwargs['network_disabled']

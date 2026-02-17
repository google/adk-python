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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.gke_code_executor import GkeCodeExecutor
from kubernetes import client
from kubernetes import config
from kubernetes.client.rest import ApiException
import pytest


@pytest.fixture
def mock_invocation_context() -> InvocationContext:
  """Fixture for a mock InvocationContext."""
  mock = MagicMock(spec=InvocationContext)
  mock.invocation_id = "test-invocation-123"
  return mock


@pytest.fixture(autouse=True)
def mock_k8s_config():
  """Fixture for auto-mocking Kubernetes config loading."""
  with patch(
      "google.adk.code_executors.gke_code_executor.config"
  ) as mock_config:
    # Simulate fallback from in-cluster to kubeconfig
    mock_config.ConfigException = config.ConfigException
    mock_config.load_incluster_config.side_effect = config.ConfigException
    yield mock_config


@pytest.fixture
def mock_k8s_clients():
  """Fixture for mock Kubernetes API clients."""
  with patch(
      "google.adk.code_executors.gke_code_executor.client"
  ) as mock_client_class:
    mock_batch_v1 = MagicMock(spec=client.BatchV1Api)
    mock_core_v1 = MagicMock(spec=client.CoreV1Api)
    mock_client_class.BatchV1Api.return_value = mock_batch_v1
    mock_client_class.CoreV1Api.return_value = mock_core_v1
    yield {
        "batch_v1": mock_batch_v1,
        "core_v1": mock_core_v1,
    }


class TestGkeCodeExecutor:
  """Unit tests for the GkeCodeExecutor."""

  def test_init_defaults(self):
    """Tests that the executor initializes with correct default values."""
    executor = GkeCodeExecutor()
    assert executor.namespace == "default"
    assert executor.image == "python:3.11-slim"
    assert executor.timeout_seconds == 300
    assert executor.cpu_requested == "200m"
    assert executor.mem_limit == "512Mi"

  def test_init_with_overrides(self):
    """Tests that class attributes can be overridden at instantiation."""
    executor = GkeCodeExecutor(
        namespace="test-ns",
        image="custom-python:latest",
        timeout_seconds=60,
        cpu_limit="1000m",
    )
    assert executor.namespace == "test-ns"
    assert executor.image == "custom-python:latest"
    assert executor.timeout_seconds == 60
    assert executor.cpu_limit == "1000m"

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_success(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the happy path for successful code execution."""
    # Setup Mocks
    mock_job = MagicMock()
    mock_job.status.succeeded = True
    mock_job.status.failed = None
    mock_watch.return_value.stream.return_value = [{"object": mock_job}]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [MagicMock()]
    mock_pod_list.items[0].metadata.name = "test-pod-name"
    mock_k8s_clients["core_v1"].list_namespaced_pod.return_value = mock_pod_list
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "hello world"
    )

    # Execute
    executor = GkeCodeExecutor()
    code_input = CodeExecutionInput(code='print("hello world")')
    result = executor.execute_code(mock_invocation_context, code_input)

    # Assert
    assert result.stdout == "hello world"
    assert result.stderr == ""
    mock_k8s_clients[
        "core_v1"
    ].create_namespaced_config_map.assert_called_once()
    mock_k8s_clients["batch_v1"].create_namespaced_job.assert_called_once()
    mock_k8s_clients["core_v1"].patch_namespaced_config_map.assert_called_once()
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.assert_called_once()

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_job_failed(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the path where the Kubernetes Job fails."""
    mock_job = MagicMock()
    mock_job.status.succeeded = None
    mock_job.status.failed = True
    mock_watch.return_value.stream.return_value = [{"object": mock_job}]
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "Traceback...\nValueError: failure"
    )

    executor = GkeCodeExecutor()
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="fail")
    )

    assert result.stdout == ""
    assert "Job failed. Logs:" in result.stderr
    assert "ValueError: failure" in result.stderr

  def test_execute_code_api_exception(
      self, mock_k8s_clients, mock_invocation_context
  ):
    """Tests handling of an ApiException from the K8s client."""
    mock_k8s_clients["core_v1"].create_namespaced_config_map.side_effect = (
        ApiException(reason="Test API Error")
    )
    executor = GkeCodeExecutor()
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="...")
    )

    assert result.stdout == ""
    assert "Kubernetes API error: Test API Error" in result.stderr

  @patch("google.adk.code_executors.gke_code_executor.Watch")
  def test_execute_code_timeout(
      self,
      mock_watch,
      mock_k8s_clients,
      mock_invocation_context,
  ):
    """Tests the case where the job watch times out."""
    mock_watch.return_value.stream.return_value = (
        []
    )  # Empty stream simulates timeout
    mock_k8s_clients["core_v1"].read_namespaced_pod_log.return_value = (
        "Still running..."
    )

    executor = GkeCodeExecutor(timeout_seconds=1)
    result = executor.execute_code(
        mock_invocation_context, CodeExecutionInput(code="...")
    )

    assert result.stdout == ""
    assert "Executor timed out" in result.stderr
    assert "did not complete within 1s" in result.stderr
    assert "Pod Logs:\nStill running..." in result.stderr

  def test_create_job_manifest_structure(self, mock_invocation_context):
    """Tests the correctness of the generated Job manifest."""
    executor = GkeCodeExecutor(namespace="test-ns", image="test-img:v1")
    job = executor._create_job_manifest(
        "test-job", "test-cm", mock_invocation_context
    )

    # Check top-level properties
    assert isinstance(job, client.V1Job)
    assert job.api_version == "batch/v1"
    assert job.kind == "Job"
    assert job.metadata.name == "test-job"
    assert job.spec.backoff_limit == 0
    assert job.spec.ttl_seconds_after_finished == 600

    # Check pod template properties
    pod_spec = job.spec.template.spec
    assert pod_spec.restart_policy == "Never"
    assert pod_spec.runtime_class_name == "gvisor"
    assert len(pod_spec.tolerations) == 1
    assert pod_spec.tolerations[0].value == "gvisor"
    assert len(pod_spec.volumes) == 1
    assert pod_spec.volumes[0].name == "code-volume"
    assert pod_spec.volumes[0].config_map.name == "test-cm"

    # Check container properties
    container = pod_spec.containers[0]
    assert container.name == "code-runner"
    assert container.image == "test-img:v1"
    assert container.command == ["python3", "/app/code.py"]

    # Check security context
    sec_context = container.security_context
    assert sec_context.run_as_non_root is True
    assert sec_context.run_as_user == 1001
    assert sec_context.allow_privilege_escalation is False
    assert sec_context.read_only_root_filesystem is True
    assert sec_context.capabilities.drop == ["ALL"]

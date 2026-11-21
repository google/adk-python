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

from google.adk import version
from google.adk.utils import _client_labels_utils
import pytest


def test_get_client_labels_default():
  """Test get_client_labels returns default labels."""
  labels = _client_labels_utils.get_client_labels()
  assert len(labels) == 2
  assert f"google-adk/{version.__version__}" == labels[0]
  assert f"gl-python/{sys.version.split()[0]}" == labels[1]


_RUNTIME_ENV_VARS = (
    "GOOGLE_CLOUD_AGENT_ENGINE_ID",
    "K_SERVICE",
    "CLOUD_RUN_JOB",
    "KUBERNETES_SERVICE_HOST",
)


# The tags are literals because they are a wire format parsed server-side.
@pytest.mark.parametrize(
    "env_vars, expected_tag",
    [
        (["GOOGLE_CLOUD_AGENT_ENGINE_ID"], "remote_reasoning_engine"),
        (["K_SERVICE"], "cloud_run"),
        (["CLOUD_RUN_JOB"], "cloud_run"),
        (["K_SERVICE", "CLOUD_RUN_JOB"], "cloud_run"),
        (["KUBERNETES_SERVICE_HOST"], "k8s"),
        # Knative and Cloud Run on GKE set both; cloud_run wins.
        (["K_SERVICE", "KUBERNETES_SERVICE_HOST"], "cloud_run"),
        (list(_RUNTIME_ENV_VARS), "remote_reasoning_engine"),
    ],
)
def test_get_client_labels_runtime_tag(monkeypatch, env_vars, expected_tag):
  """Each runtime gets its tag, most specific runtime first."""
  for name in env_vars:
    monkeypatch.setenv(name, "value")
  labels = _client_labels_utils.get_client_labels()
  assert labels[0] == f"google-adk/{version.__version__}+{expected_tag}"


@pytest.mark.parametrize("env_var_name", _RUNTIME_ENV_VARS)
def test_empty_runtime_env_var_is_ignored(monkeypatch, env_var_name):
  """A runtime env var set to the empty string does not tag the runtime."""
  monkeypatch.setenv(env_var_name, "")
  labels = _client_labels_utils.get_client_labels()
  assert labels[0] == f"google-adk/{version.__version__}"


@pytest.mark.parametrize("env_var_name", _RUNTIME_ENV_VARS)
def test_get_client_labels_without_any_runtime_tag(monkeypatch, env_var_name):
  """exclude_runtime_tag=True drops every runtime suffix."""
  monkeypatch.setenv(env_var_name, "value")
  labels = _client_labels_utils.get_client_labels(exclude_runtime_tag=True)
  assert labels[0] == f"google-adk/{version.__version__}"


def test_runtime_detection_is_not_cached(monkeypatch):
  """Env changes after the first call are picked up by the next call."""
  assert f"google-adk/{version.__version__}" == (
      _client_labels_utils.get_client_labels()[0]
  )
  monkeypatch.setenv("CLOUD_RUN_JOB", "my-job")
  assert f"google-adk/{version.__version__}+cloud_run" == (
      _client_labels_utils.get_client_labels()[0]
  )


def test_get_client_labels_with_agent_engine_id(monkeypatch):
  """Test get_client_labels returns agent engine tag when env var is set."""
  monkeypatch.setenv(
      _client_labels_utils._AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME,
      "test-agent-id",
  )
  labels = _client_labels_utils.get_client_labels()
  assert len(labels) == 2
  assert (
      f"google-adk/{version.__version__}+{_client_labels_utils._AGENT_ENGINE_TELEMETRY_TAG}"
      == labels[0]
  )
  assert f"gl-python/{sys.version.split()[0]}" == labels[1]


def test_framework_label_kept_without_runtime_tag(monkeypatch):
  """exclude_runtime_tag=True still keeps framework_label."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "test-agent-id")
  labels = _client_labels_utils.get_client_labels(
      framework_label="managed_agent", exclude_runtime_tag=True
  )
  assert labels[0] == f"google-adk/{version.__version__}+managed_agent"


def test_get_client_labels_with_context():
  """Test get_client_labels includes label from context."""
  with _client_labels_utils.client_label_context("my-label/1.0"):
    labels = _client_labels_utils.get_client_labels()
    assert len(labels) == 3
    assert f"google-adk/{version.__version__}" == labels[0]
    assert f"gl-python/{sys.version.split()[0]}" == labels[1]
    assert "my-label/1.0" == labels[2]


def test_client_label_context_nested_error():
  """Test client_label_context raises error when nested."""
  with pytest.raises(ValueError, match="Client label already exists"):
    with _client_labels_utils.client_label_context("my-label/1.0"):
      with _client_labels_utils.client_label_context("another-label/1.0"):
        pass


def test_eval_client_label():
  """Test EVAL_CLIENT_LABEL has correct format."""
  assert (
      f"google-adk-eval/{version.__version__}"
      == _client_labels_utils.EVAL_CLIENT_LABEL
  )


def test_model_consult_client_label():
  """Test MODEL_CONSULT_CLIENT_LABEL has correct format."""
  assert (
      f"google-adk-model-consult/{version.__version__}"
      == _client_labels_utils.MODEL_CONSULT_CLIENT_LABEL
  )


def test_cascade_live_client_label():
  """Test CASCADE_LIVE_CLIENT_LABEL has correct format."""
  assert (
      f"google-adk-cascade-live/{version.__version__}"
      == _client_labels_utils.CASCADE_LIVE_CLIENT_LABEL
  )


def test_classifier_client_label():
  """classifier_client_label names the classifier kind as build metadata."""
  assert (
      f"google-adk-classifier/{version.__version__}+llm"
      == _client_labels_utils.classifier_client_label("llm")
  )


def test_get_client_labels_with_framework_label():
  """An explicit framework_label appends a +suffix to the google-adk token."""
  labels = _client_labels_utils.get_client_labels(
      framework_label="managed_agent"
  )
  assert len(labels) == 2
  assert f"google-adk/{version.__version__}+managed_agent" == labels[0]
  assert f"gl-python/{sys.version.split()[0]}" == labels[1]


def test_framework_label_takes_precedence_over_agent_engine(monkeypatch):
  """An explicit framework_label wins over the Agent Engine env-var suffix."""
  monkeypatch.setenv(
      _client_labels_utils._AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME,
      "test-agent-id",
  )
  labels = _client_labels_utils.get_client_labels(
      framework_label="managed_agent"
  )
  assert f"google-adk/{version.__version__}+managed_agent" == labels[0]

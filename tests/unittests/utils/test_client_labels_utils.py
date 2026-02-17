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

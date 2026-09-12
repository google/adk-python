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

import pytest
from unittest.mock import MagicMock

from google.adk.cli.api_server import ApiServer


def _make_server(**kwargs):
  defaults = {
      "agent_loader": MagicMock(),
      "session_service": MagicMock(),
      "memory_service": MagicMock(),
      "artifact_service": MagicMock(),
      "credential_service": MagicMock(),
      "eval_sets_manager": MagicMock(),
      "eval_set_results_manager": MagicMock(),
      "agents_dir": "/tmp",
  }
  defaults.update(kwargs)
  return ApiServer(**defaults)


def test_trigger_sources_requires_oidc_audience_or_verifier():
  with pytest.raises(ValueError, match="trigger_sources requires trigger_oidc_audience"):
    _make_server(trigger_sources=["pubsub"])


def test_trigger_sources_ok_with_oidc_audience():
  server = _make_server(
      trigger_sources=["pubsub"],
      trigger_oidc_audience="my-audience",
  )
  assert server.trigger_sources == ["pubsub"]
  assert server.trigger_oidc_audience == "my-audience"


def test_trigger_sources_ok_with_auth_verifier():
  server = _make_server(
      trigger_sources=["eventarc"],
      trigger_auth_verifier=lambda req: None,
  )
  assert server.trigger_sources == ["eventarc"]
  assert server.trigger_auth_verifier is not None

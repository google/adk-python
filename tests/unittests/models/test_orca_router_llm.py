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

"""Tests for OrcaRouterLlm."""

import os
from unittest import mock

from google.adk.models.orca_router_llm import OrcaRouterLlm
import pytest


def test_supported_models():
  assert OrcaRouterLlm.supported_models() == [r"orcarouter/.*"]


def test_default_model():
  assert OrcaRouterLlm().model == "anthropic/claude-haiku-4.5"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("orcarouter/anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
        ("anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5"),
    ],
)
def test_resolve_model_name_strips_orca_router_prefix(raw, expected):
  llm = OrcaRouterLlm()
  assert llm._resolve_model_name(raw) == expected


def test_anthropic_client_creation_uses_orca_router_credentials():
  with mock.patch.dict(
      os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test-key"}, clear=True
  ):
    model = OrcaRouterLlm()
    with mock.patch(
        "google.adk.models.orca_router_llm.AsyncAnthropic", autospec=True
    ) as mock_client_class:
      _ = model._anthropic_client
      mock_client_class.assert_called_once()
      _, kwargs = mock_client_class.call_args
      assert kwargs["api_key"] == "sk-orca-test-key"
      assert kwargs["base_url"] == "https://api.orcarouter.ai"


def test_anthropic_client_creation_without_credential_raises():
  with mock.patch.dict(os.environ, {}, clear=True):
    model = OrcaRouterLlm()
    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
      _ = model._anthropic_client

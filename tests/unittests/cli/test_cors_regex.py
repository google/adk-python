# Copyright 2025 Google LLC
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

"""Tests for CORS configuration with regex prefix support."""

from unittest import mock

from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
from google.adk.cli.adk_web_server import _parse_cors_origins
from google.adk.cli.adk_web_server import AdkWebServer
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.sessions.base_session_service import BaseSessionService
import pytest


class MockAgentLoader:
  """Mock agent loader for testing."""

  def __init__(self):
    pass

  def load_agent(self, app_name):
    del self, app_name
    return mock.MagicMock()

  def list_agents(self):
    del self
    return ["test_app"]

  def list_agents_detailed(self):
    del self
    return []


def create_adk_web_server():
  """Create an AdkWebServer instance for testing."""
  return AdkWebServer(
      agent_loader=MockAgentLoader(),
      session_service=mock.create_autospec(BaseSessionService, instance=True),
      memory_service=mock.create_autospec(BaseMemoryService, instance=True),
      artifact_service=mock.create_autospec(BaseArtifactService, instance=True),
      credential_service=mock.create_autospec(
          BaseCredentialService, instance=True
      ),
      eval_sets_manager=mock.create_autospec(EvalSetsManager, instance=True),
      eval_set_results_manager=mock.create_autospec(
          EvalSetResultsManager, instance=True
      ),
      agents_dir=".",
  )


def _get_cors_middleware(app):
  """Extract CORSMiddleware from app's middleware stack.

  Returns:
    The CORSMiddleware instance, or None if not found.
  """
  for middleware in app.user_middleware:
    if middleware.cls.__name__ == "CORSMiddleware":
      return middleware
  return None


CORS_ORIGINS_TEST_CASES = [
    # Literal origins only
    (
        ["https://example.com", "https://test.com"],
        ["https://example.com", "https://test.com"],
        None,
    ),
    # Regex patterns only
    (
        [
            "regex:https://.*\\.example\\.com",
            "regex:https://.*\\.test\\.com",
        ],
        [],
        "https://.*\\.example\\.com|https://.*\\.test\\.com",
    ),
    # Mixed literal and regex
    (
        [
            "https://example.com",
            "regex:https://.*\\.subdomain\\.com",
            "https://test.com",
            "regex:https://tenant-.*\\.myapp\\.com",
        ],
        ["https://example.com", "https://test.com"],
        "https://.*\\.subdomain\\.com|https://tenant-.*\\.myapp\\.com",
    ),
    # Wildcard origin
    (["*"], ["*"], None),
    # Single regex
    (
        ["regex:https://.*\\.example\\.com"],
        [],
        "https://.*\\.example\\.com",
    ),
]

CORS_ORIGINS_TEST_IDS = [
    "literal_only",
    "regex_only",
    "mixed",
    "wildcard",
    "single_regex",
]


class TestParseCorsOrigins:
  """Tests for the _parse_cors_origins helper function."""

  @pytest.mark.parametrize(
      "allow_origins,expected_literal,expected_regex",
      CORS_ORIGINS_TEST_CASES,
      ids=CORS_ORIGINS_TEST_IDS,
  )
  def test_parse_cors_origins(
      self, allow_origins, expected_literal, expected_regex
  ):
    """Test parsing of allow_origins into literal and regex components."""
    literal_origins, combined_regex = _parse_cors_origins(allow_origins)
    assert literal_origins == expected_literal
    assert combined_regex == expected_regex


class TestCorsMiddlewareConfiguration:
  """Tests for CORS middleware configuration in AdkWebServer."""

  @pytest.mark.parametrize(
      "allow_origins,expected_literal,expected_regex",
      CORS_ORIGINS_TEST_CASES,
      ids=CORS_ORIGINS_TEST_IDS,
  )
  def test_cors_middleware_configuration(
      self, allow_origins, expected_literal, expected_regex
  ):
    """Test CORS middleware is configured correctly with various origin types."""
    server = create_adk_web_server()
    app = server.get_fast_api_app(
        allow_origins=allow_origins,
        setup_observer=lambda _o, _s: None,
        tear_down_observer=lambda _o, _s: None,
    )

    cors_middleware = _get_cors_middleware(app)
    assert cors_middleware is not None
    assert cors_middleware.kwargs["allow_origins"] == expected_literal
    assert cors_middleware.kwargs["allow_origin_regex"] == expected_regex

  @pytest.mark.parametrize(
      "allow_origins",
      [None, []],
      ids=["none", "empty_list"],
  )
  def test_cors_middleware_not_added_when_no_origins(self, allow_origins):
    """Test that no CORS middleware is added when allow_origins is None or empty."""
    server = create_adk_web_server()
    app = server.get_fast_api_app(
        allow_origins=allow_origins,
        setup_observer=lambda _o, _s: None,
        tear_down_observer=lambda _o, _s: None,
    )

    cors_middleware = _get_cors_middleware(app)
    assert cors_middleware is None

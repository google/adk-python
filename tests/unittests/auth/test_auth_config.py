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

import os
from pathlib import Path
import subprocess
import sys

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
import pytest


class TestAuthConfig:
  """Tests for the AuthConfig method."""


@pytest.fixture
def oauth2_auth_scheme():
  """Create an OAuth2 auth scheme for testing."""
  # Create the OAuthFlows object first
  flows = OAuthFlows(
      authorizationCode=OAuthFlowAuthorizationCode(
          authorizationUrl="https://example.com/oauth2/authorize",
          tokenUrl="https://example.com/oauth2/token",
          scopes={"read": "Read access", "write": "Write access"},
      )
  )

  # Then create the OAuth2 object with the flows
  return OAuth2(flows=flows)


@pytest.fixture
def oauth2_credentials():
  """Create OAuth2 credentials for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="mock_client_id",
          client_secret="mock_client_secret",
          redirect_uri="https://example.com/callback",
      ),
  )


@pytest.fixture
def auth_config(oauth2_auth_scheme, oauth2_credentials):
  """Create an AuthConfig for testing."""
  # Create a copy of the credentials for the exchanged_auth_credential
  exchanged_credential = oauth2_credentials.model_copy(deep=True)

  return AuthConfig(
      auth_scheme=oauth2_auth_scheme,
      raw_auth_credential=oauth2_credentials,
      exchanged_auth_credential=exchanged_credential,
  )


@pytest.fixture
def auth_config_with_key(oauth2_auth_scheme, oauth2_credentials):
  """Create an AuthConfig for testing."""

  return AuthConfig(
      auth_scheme=oauth2_auth_scheme,
      raw_auth_credential=oauth2_credentials,
      credential_key="test_key",
  )


def test_custom_credential_key(auth_config_with_key):
  """Test using custom credential key."""

  key = auth_config_with_key.credential_key
  assert key == "test_key"


def test_credential_key(auth_config):
  """Test generating a unique credential key."""

  key = auth_config.credential_key
  assert key.startswith("adk_oauth2_")
  assert "_oauth2_" in key


def test_get_credential_key_with_extras(auth_config):
  """Test generating a key when model_extra exists."""
  # Add model_extra to test cleanup

  original_key = auth_config.credential_key
  key = auth_config.credential_key

  auth_config.auth_scheme.model_extra["extra_field"] = "value"
  auth_config.raw_auth_credential.model_extra["extra_field"] = "value"

  assert original_key == key
  assert "extra_field" in auth_config.auth_scheme.model_extra
  assert "extra_field" in auth_config.raw_auth_credential.model_extra


def test_credential_key_is_stable_across_python_hash_seed():
  """Test AuthConfig key generation does not depend on PYTHONHASHSEED."""
  repo_root = Path(__file__).resolve().parents[3]
  pythonpath = str(repo_root / "src")
  code = "\n".join([
      "from fastapi.openapi.models import OAuth2",
      "from fastapi.openapi.models import OAuthFlowAuthorizationCode",
      "from fastapi.openapi.models import OAuthFlows",
      "from google.adk.auth.auth_credential import AuthCredential",
      "from google.adk.auth.auth_credential import AuthCredentialTypes",
      "from google.adk.auth.auth_credential import OAuth2Auth",
      "from google.adk.auth.auth_tool import AuthConfig",
      "",
      "auth_scheme = OAuth2(",
      "  flows=OAuthFlows(",
      "    authorizationCode=OAuthFlowAuthorizationCode(",
      "      authorizationUrl='https://example.com/oauth2/authorize',",
      "      tokenUrl='https://example.com/oauth2/token',",
      "      scopes={'read': 'Read access'},",
      "    )",
      "  )",
      ")",
      "auth_cred = AuthCredential(",
      "  auth_type=AuthCredentialTypes.OAUTH2,",
      "  oauth2=OAuth2Auth(",
      "    client_id='mock_client_id',",
      "    client_secret='mock_client_secret',",
      "  ),",
      ")",
      "print(AuthConfig(",
      "  auth_scheme=auth_scheme,",
      "  raw_auth_credential=auth_cred,",
      ").credential_key)",
  ])

  def _run_with_seed(seed: str) -> str:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join(
        [pythonpath, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
    ).strip()

  assert _run_with_seed("0") == _run_with_seed("1")

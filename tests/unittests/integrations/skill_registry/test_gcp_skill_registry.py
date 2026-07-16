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

"""Tests for GCP Skill Registry."""

import io
import os
from unittest import mock
import zipfile

from google.adk.integrations.skill_registry import gcp_skill_registry
import pytest


@pytest.fixture(autouse=True)
def mock_env():
  """Fixture to mock environment variables."""
  with mock.patch.dict(
      os.environ,
      {
          "GOOGLE_CLOUD_PROJECT": "test-project",
          "GOOGLE_CLOUD_LOCATION": "us-central1",
      },
  ):
    yield


@pytest.fixture(autouse=True)
def mock_google_auth():
  """Fixture to mock google.auth.default."""
  mock_creds = mock.MagicMock()
  mock_creds.valid = True
  mock_creds.token = "fake-token"
  mock_creds.quota_project_id = None
  with mock.patch(
      "google.auth.default", return_value=(mock_creds, "test-project")
  ):
    yield mock_creds


@pytest.fixture(autouse=True)
def disable_mtls_by_default():
  """Fixture to disable mTLS by default for unit tests."""
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=False,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=False,
      ),
  ):
    yield


def _create_fake_zip_bytes():
  """Creates a fake zip file in memory and returns its bytes."""
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  return zip_buffer.getvalue()


@pytest.mark.asyncio
async def test_get_skill_success():
  """Verifies that get_skill successfully fetches and loads a skill in memory."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  fake_zip = _create_fake_zip_bytes()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch(
      "httpx.AsyncClient.get", side_effect=mock_get
  ) as mock_get_called:
    skill = await registry.get_skill(name="my-skill")

  assert skill.frontmatter.name == "my-skill"
  assert skill.frontmatter.description == "test"
  assert skill.instructions == "# My Skill"

  mock_get_called.assert_has_calls([
      mock.call(
          "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills/my-skill",
          headers={
              "Authorization": "Bearer fake-token",
              "Content-Type": "application/json",
              "x-goog-user-project": "test-project",
          },
          params=None,
      ),
      mock.call(
          "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123",
          headers={
              "Authorization": "Bearer fake-token",
              "Content-Type": "application/json",
              "x-goog-user-project": "test-project",
          },
          params={"alt": "media"},
      ),
  ])


@pytest.mark.asyncio
async def test_search_skills_success():
  """Verifies that search_skills successfully returns frontmatter list."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {
      "skills": [
          {
              "name": (
                  "projects/test-project/locations/us-central1/skills/skill1"
              ),
              "description": "Description 1",
          },
          {
              "name": (
                  "projects/test-project/locations/us-central1/skills/skill2"
              ),
              "description": "Description 2",
          },
      ]
  }

  with mock.patch(
      "httpx.AsyncClient.get", return_value=mock_response
  ) as mock_get_called:
    results = await registry.search_skills(query="query")

  assert len(results) == 2
  assert results[0].name == "skill1"
  assert results[0].description == "Description 1"
  assert results[1].name == "skill2"
  assert results[1].description == "Description 2"

  mock_get_called.assert_called_once_with(
      "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills:search",
      headers={
          "Authorization": "Bearer fake-token",
          "Content-Type": "application/json",
          "x-goog-user-project": "test-project",
      },
      params={"search_string": "query"},
  )


@pytest.mark.asyncio
async def test_get_skill_raises_on_missing_zip():
  """Verifies that get_skill raises error if zip filesystem is missing."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
  }

  with mock.patch("httpx.AsyncClient.get", return_value=mock_response):
    with pytest.raises(ValueError, match="does not contain default revision"):
      await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_zip_slip():
  """Verifies that get_skill raises error if zip contains dangerous paths."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr("../evil.txt", "malicious content")
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
    with pytest.raises(ValueError, match="Dangerous zip entry ignored"):
      await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_invalid_skill_name():
  """Verifies that get_skill raises error if skill name is invalid."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: ../evil\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
    with pytest.raises(ValueError, match="Invalid skill name in SKILL.md"):
      await registry.get_skill(name="my-skill")


def test_constructor_configures_base_url():
  """Verifies that constructor configures base URL from environment."""
  # Case 1: Environment variable fallback
  with mock.patch.dict(
      os.environ, {"AGENT_REGISTRY_ENDPOINT": "https://staging.endpoint.com"}
  ):
    registry = gcp_skill_registry.GCPSkillRegistry()
    assert registry.base_url == "https://staging.endpoint.com"

  # Case 2: Default fallback
  registry = gcp_skill_registry.GCPSkillRegistry()
  assert registry.base_url == "https://agentregistry.googleapis.com/v1alpha"


# pylint: disable=protected-access
def test_lazy_load_credentials():
  """Verifies that google.auth.default is not called in constructor."""
  with mock.patch("google.auth.default") as mock_auth:
    registry = gcp_skill_registry.GCPSkillRegistry()
    mock_auth.assert_not_called()
    assert registry._credentials is None


def test_constructor_configures_mtls_base_url():
  """Verifies that constructor configures base URL when mTLS is enabled."""
  mock_cert_source = mock.MagicMock(return_value=(b"fake-cert", b"fake-key"))
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.default_client_cert_source",
          return_value=mock_cert_source,
      ),
      mock.patch("ssl.create_default_context") as mock_create_ssl_context,
  ):
    registry = gcp_skill_registry.GCPSkillRegistry()
    assert (
        registry.base_url == "https://agentregistry.mtls.googleapis.com/v1alpha"
    )
    assert registry._ssl_context is not None
    mock_create_ssl_context.assert_called_once()


@pytest.mark.asyncio
async def test_get_skill_with_mtls():
  """Verifies that get_skill works correctly and passes ssl context when mTLS is enabled."""
  mock_cert_source = mock.MagicMock(return_value=(b"fake-cert", b"fake-key"))
  fake_zip = _create_fake_zip_bytes()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.default_client_cert_source",
          return_value=mock_cert_source,
      ),
      mock.patch("ssl.create_default_context") as mock_create_ssl_context,
  ):
    # Set up mock SSL context
    mock_ssl_context = mock_create_ssl_context.return_value
    registry = gcp_skill_registry.GCPSkillRegistry()

    with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
      mock_client = mock_client_class.return_value
      mock_client.__aenter__.return_value = mock_client
      mock_client.get = mock.AsyncMock(side_effect=mock_get)

      skill = await registry.get_skill(name="my-skill")

      # Verify AsyncClient was instantiated with verify=mock_ssl_context
      mock_client_class.assert_called_with(verify=mock_ssl_context)

  assert skill.frontmatter.name == "my-skill"


# pylint: enable=protected-access


@pytest.mark.asyncio
async def test_use_custom_credentials():
  """Verifies that custom credentials are used when provided."""
  mock_creds = mock.MagicMock()
  mock_creds.valid = True
  mock_creds.token = "custom-token"
  mock_creds.quota_project_id = "custom-quota-project"

  registry = gcp_skill_registry.GCPSkillRegistry(credentials=mock_creds)

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {"skills": []}

  with mock.patch(
      "httpx.AsyncClient.get", return_value=mock_response
  ) as mock_get_called:
    await registry.search_skills(query="query")

  mock_get_called.assert_called_once_with(
      "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills:search",
      headers={
          "Authorization": "Bearer custom-token",
          "Content-Type": "application/json",
          "x-goog-user-project": "custom-quota-project",
      },
      params={"search_string": "query"},
  )

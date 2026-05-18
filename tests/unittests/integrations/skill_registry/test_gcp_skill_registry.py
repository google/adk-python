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

import base64
import os
from unittest import mock
import zipfile

from google.adk.integrations.skill_registry.gcp_skill_registry import GCPSkillRegistry
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


@pytest.fixture
def mock_vertex_client():
  """Fixture to mock vertexai.Client."""
  with mock.patch("vertexai.Client") as mock_client_class:
    mock_client = mock_client_class.return_value
    yield mock_client


def _create_fake_zip_bytes():
  """Creates a fake zip file in memory and returns its bytes."""
  import io

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  return zip_buffer.getvalue()


@pytest.mark.asyncio
async def test_get_skill_success(mock_vertex_client):
  """Verifies that get_skill successfully fetches and loads a skill in memory."""
  registry = GCPSkillRegistry()

  fake_zip = _create_fake_zip_bytes()
  fake_zip_base64 = base64.b64encode(fake_zip).decode("utf-8")

  mock_skill_resource = mock.MagicMock()
  mock_skill_resource.zipped_filesystem = fake_zip_base64

  mock_vertex_client.aio.skills.get = mock.AsyncMock(
      return_value=mock_skill_resource
  )

  skill = await registry.get_skill(name="my-skill")

  assert skill.frontmatter.name == "my-skill"
  assert skill.frontmatter.description == "test"
  assert skill.instructions == "# My Skill"
  mock_vertex_client.aio.skills.get.assert_called_once_with(
      name="projects/test-project/locations/us-central1/skills/my-skill"
  )


@pytest.mark.asyncio
async def test_search_skills_success(mock_vertex_client):
  """Verifies that search_skills successfully returns frontmatter list."""
  registry = GCPSkillRegistry()

  mock_skill1 = mock.MagicMock()
  mock_skill1.skill_name = (
      "projects/test-project/locations/us-central1/skills/skill1"
  )
  mock_skill1.description = "Description 1"

  mock_skill2 = mock.MagicMock()
  mock_skill2.skill_name = (
      "projects/test-project/locations/us-central1/skills/skill2"
  )
  mock_skill2.description = "Description 2"

  mock_response = mock.MagicMock()
  mock_response.retrieved_skills = [mock_skill1, mock_skill2]

  mock_vertex_client.aio.skills.retrieve = mock.AsyncMock(
      return_value=mock_response
  )

  results = await registry.search_skills(query="query")

  assert len(results) == 2
  assert results[0].name == "skill1"
  assert results[0].description == "Description 1"
  assert results[1].name == "skill2"
  assert results[1].description == "Description 2"
  mock_vertex_client.aio.skills.retrieve.assert_called_once_with(query="query")


@pytest.mark.asyncio
async def test_get_skill_raises_on_missing_zip(mock_vertex_client):
  """Verifies that get_skill raises error if zip filesystem is missing."""
  registry = GCPSkillRegistry()

  mock_skill_resource = mock.MagicMock()
  mock_skill_resource.zipped_filesystem = None

  mock_vertex_client.aio.skills.get = mock.AsyncMock(
      return_value=mock_skill_resource
  )

  with pytest.raises(ValueError, match="does not contain zipped filesystem"):
    await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_zip_slip(mock_vertex_client):
  """Verifies that get_skill raises error if zip contains dangerous paths."""
  registry = GCPSkillRegistry()

  import io

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr("../evil.txt", "malicious content")
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()
  fake_zip_base64 = base64.b64encode(fake_zip).decode("utf-8")

  mock_skill_resource = mock.MagicMock()
  mock_skill_resource.zipped_filesystem = fake_zip_base64

  mock_vertex_client.aio.skills.get = mock.AsyncMock(
      return_value=mock_skill_resource
  )

  with pytest.raises(ValueError, match="Dangerous zip entry ignored"):
    await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_invalid_skill_name(mock_vertex_client):
  """Verifies that get_skill raises error if skill name is invalid."""
  registry = GCPSkillRegistry()

  import io

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: ../evil\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()
  fake_zip_base64 = base64.b64encode(fake_zip).decode("utf-8")

  mock_skill_resource = mock.MagicMock()
  mock_skill_resource.zipped_filesystem = fake_zip_base64

  mock_vertex_client.aio.skills.get = mock.AsyncMock(
      return_value=mock_skill_resource
  )

  with pytest.raises(ValueError, match="Invalid skill name in SKILL.md"):
    await registry.get_skill(name="my-skill")

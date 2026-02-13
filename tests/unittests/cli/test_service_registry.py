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

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_services():
  """Mock all service implementation classes to avoid real instantiation."""
  with (
      patch(
          "google.adk.sessions.vertex_ai_session_service.VertexAiSessionService"
      ) as mock_vertex_session,
      patch(
          "google.adk.sessions.database_session_service.DatabaseSessionService"
      ) as mock_db_session,
      patch(
          "google.adk.sessions.sqlite_session_service.SqliteSessionService"
      ) as mock_sqlite_session,
      patch(
          "google.adk.artifacts.gcs_artifact_service.GcsArtifactService"
      ) as mock_gcs_artifact,
      patch(
          "google.adk.memory.vertex_ai_rag_memory_service.VertexAiRagMemoryService"
      ) as mock_rag_memory,
      patch(
          "google.adk.memory.vertex_ai_memory_bank_service.VertexAiMemoryBankService"
      ) as mock_agentengine_memory,
  ):
    yield {
        "vertex_session": mock_vertex_session,
        "db_session": mock_db_session,
        "sqlite_session": mock_sqlite_session,
        "gcs_artifact": mock_gcs_artifact,
        "rag_memory": mock_rag_memory,
        "agentengine_memory": mock_agentengine_memory,
    }


@pytest.fixture
def registry():
  from google.adk.cli.service_registry import get_service_registry

  return get_service_registry()


# Session Service Tests
def test_create_session_service_sqlite(registry, mock_services):
  registry.create_session_service("sqlite:///test.db")
  mock_services["sqlite_session"].assert_called_once_with(db_path="test.db")


def test_create_session_service_sqlite_ignores_unsupported_kwargs(
    registry, mock_services, caplog
):
  """Test that SqliteSessionService ignores unsupported kwargs and logs warning."""
  import logging

  with caplog.at_level(logging.WARNING):
    registry.create_session_service(
        "sqlite:///test.db", pool_size=10, agents_dir="foo"
    )

  # SqliteSessionService should only receive db_path, not pool_size
  mock_services["sqlite_session"].assert_called_once_with(db_path="test.db")

  # Verify warning was logged about ignored kwargs
  assert (
      "SqliteSessionService does not support additional kwargs" in caplog.text
  )
  assert "pool_size" in caplog.text


def test_create_session_service_postgresql(registry, mock_services):
  registry.create_session_service("postgresql://user:pass@host/db")
  mock_services["db_session"].assert_called_once_with(
      db_url="postgresql://user:pass@host/db"
  )


@patch("google.adk.cli.utils.envs.load_dotenv_for_agent")
def test_create_session_service_agentengine_short(
    mock_load_dotenv, registry, mock_services, monkeypatch
):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
  registry.create_session_service(
      "agentengine://123", agents_dir="/path/to/agents"
  )
  mock_services["vertex_session"].assert_called_once_with(
      project="test-project", location="us-central1", agent_engine_id="123"
  )
  mock_load_dotenv.assert_called_once_with("", "/path/to/agents")


def test_create_session_service_agentengine_full(registry, mock_services):
  uri = "agentengine://projects/p/locations/l/reasoningEngines/123"
  registry.create_session_service(uri, agents_dir="/path/to/agents")
  mock_services["vertex_session"].assert_called_once_with(
      project="p", location="l", agent_engine_id="123"
  )


# Artifact Service Tests
def test_create_artifact_service_gcs(registry, mock_services):
  registry.create_artifact_service(
      "gs://my-bucket/path/prefix", agents_dir="foo", other_kwarg="bar"
  )
  mock_services["gcs_artifact"].assert_called_once_with(
      bucket_name="my-bucket", other_kwarg="bar"
  )


# Memory Service Tests
@patch("google.adk.cli.utils.envs.load_dotenv_for_agent")
def test_create_memory_service_rag(
    mock_load_dotenv, registry, mock_services, monkeypatch
):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
  registry.create_memory_service(
      "rag://corpus-123", agents_dir="/path/to/agents"
  )
  mock_services["rag_memory"].assert_called_once_with(
      rag_corpus=(
          "projects/test-project/locations/us-central1/ragCorpora/corpus-123"
      )
  )
  mock_load_dotenv.assert_called_once_with("", "/path/to/agents")


@patch("google.adk.cli.utils.envs.load_dotenv_for_agent")
def test_create_memory_service_agentengine_short(
    mock_load_dotenv, registry, mock_services, monkeypatch
):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
  registry.create_memory_service(
      "agentengine://456", agents_dir="/path/to/agents"
  )
  mock_services["agentengine_memory"].assert_called_once_with(
      project="test-project", location="us-central1", agent_engine_id="456"
  )
  mock_load_dotenv.assert_called_once_with("", "/path/to/agents")


def test_create_memory_service_agentengine_full(registry, mock_services):
  uri = "agentengine://projects/p/locations/l/reasoningEngines/456"
  registry.create_memory_service(uri, agents_dir="/path/to/agents")
  mock_services["agentengine_memory"].assert_called_once_with(
      project="p", location="l", agent_engine_id="456"
  )


# General Tests
def test_unsupported_scheme(registry, mock_services):
  session_service = registry.create_session_service("unsupported://foo")
  artifact_service = registry.create_artifact_service("unsupported://foo")
  memory_service = registry.create_memory_service("unsupported://foo")
  assert session_service is None
  assert artifact_service is None
  assert memory_service is None
  for service in [
      "vertex_session",
      "db_session",
      "gcs_artifact",
      "rag_memory",
      "agentengine_memory",
  ]:
    mock_services[service].assert_not_called()

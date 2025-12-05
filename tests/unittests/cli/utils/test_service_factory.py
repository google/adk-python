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

"""Tests for service factory helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from google.adk.cli.utils.local_storage import PerAgentDatabaseSessionService
import google.adk.cli.utils.service_factory as service_factory
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.database_session_service import DatabaseSessionService
import pytest


def test_create_session_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = Mock()
  expected = object()
  registry.create_session_service.return_value = expected
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  result = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      session_service_uri="sqlite:///test.db",
  )

  assert result is expected
  registry.create_session_service.assert_called_once_with(
      "sqlite:///test.db",
      agents_dir=str(tmp_path),
  )


@pytest.mark.asyncio
async def test_create_session_service_defaults_to_per_agent_sqlite(
    tmp_path: Path,
) -> None:
  agent_dir = tmp_path / "agent_a"
  agent_dir.mkdir()
  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
  )

  assert isinstance(service, PerAgentDatabaseSessionService)
  session = await service.create_session(app_name="agent_a", user_id="user")
  assert session.app_name == "agent_a"
  assert (agent_dir / ".adk" / "session.db").exists()


@pytest.mark.asyncio
async def test_create_session_service_respects_app_name_mapping(
    tmp_path: Path,
) -> None:
  agent_dir = tmp_path / "agent_folder"
  logical_name = "custom_app"
  agent_dir.mkdir()

  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      app_name_to_dir={logical_name: "agent_folder"},
  )

  assert isinstance(service, PerAgentDatabaseSessionService)
  session = await service.create_session(app_name=logical_name, user_id="user")
  assert session.app_name == logical_name
  assert (agent_dir / ".adk" / "session.db").exists()


def test_create_session_service_fallbacks_to_database(
    tmp_path: Path, monkeypatch
):
  registry = Mock()
  registry.create_session_service.return_value = None
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      session_service_uri="sqlite+aiosqlite:///:memory:",
      session_db_kwargs={"echo": True},
  )

  assert isinstance(service, DatabaseSessionService)
  assert service.db_engine.url.drivername == "sqlite+aiosqlite"
  assert service.db_engine.echo is True
  registry.create_session_service.assert_called_once_with(
      "sqlite+aiosqlite:///:memory:",
      agents_dir=str(tmp_path),
      echo=True,
  )


def test_create_artifact_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = Mock()
  expected = object()
  registry.create_artifact_service.return_value = expected
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  result = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      artifact_service_uri="gs://bucket/path",
  )

  assert result is expected
  registry.create_artifact_service.assert_called_once_with(
      "gs://bucket/path",
      agents_dir=str(tmp_path),
  )


def test_create_artifact_service_raises_on_unknown_scheme_when_strict(
    tmp_path: Path, monkeypatch
):
  registry = Mock()
  registry.create_artifact_service.return_value = None
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  with pytest.raises(ValueError):
    service_factory.create_artifact_service_from_options(
        base_dir=tmp_path,
        artifact_service_uri="unknown://foo",
        strict_uri=True,
    )


def test_create_memory_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = Mock()
  expected = object()
  registry.create_memory_service.return_value = expected
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  result = service_factory.create_memory_service_from_options(
      base_dir=tmp_path,
      memory_service_uri="rag://my-corpus",
  )

  assert result is expected
  registry.create_memory_service.assert_called_once_with(
      "rag://my-corpus",
      agents_dir=str(tmp_path),
  )


def test_create_memory_service_defaults_to_in_memory(tmp_path: Path):
  service = service_factory.create_memory_service_from_options(
      base_dir=tmp_path
  )

  assert isinstance(service, InMemoryMemoryService)


def test_create_memory_service_raises_on_unknown_scheme(
    tmp_path: Path, monkeypatch
):
  registry = Mock()
  registry.create_memory_service.return_value = None
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  with pytest.raises(ValueError):
    service_factory.create_memory_service_from_options(
        base_dir=tmp_path,
        memory_service_uri="unknown://foo",
    )

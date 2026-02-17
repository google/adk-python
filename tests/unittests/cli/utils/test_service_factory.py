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

"""Tests for service factory helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest import mock

from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.service_registry import ServiceRegistry
from google.adk.cli.utils.local_storage import PerAgentDatabaseSessionService
import google.adk.cli.utils.service_factory as service_factory
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


def test_create_session_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
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


def test_create_session_service_logs_redacted_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
  registry.create_session_service.return_value = object()
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  session_service_uri = (
      "postgresql://user:supersecret@localhost:5432/dbname?sslmode=require"
  )
  caplog.set_level(logging.INFO, logger=service_factory.logger.name)

  service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      session_service_uri=session_service_uri,
  )

  assert "supersecret" not in caplog.text
  assert "sslmode=require" not in caplog.text
  assert "localhost:5432" in caplog.text


def test_redact_uri_for_log_removes_credentials_with_at_in_password() -> None:
  uri = "postgresql://user:super@secret@localhost:5432/dbname"

  assert (
      service_factory._redact_uri_for_log(uri)
      == "postgresql://localhost:5432/dbname"
  )


def test_redact_uri_for_log_preserves_host_when_no_credentials() -> None:
  uri = "postgresql://localhost:5432/dbname?sslmode=require&password=secret"

  redacted = service_factory._redact_uri_for_log(uri)

  assert redacted.startswith("postgresql://localhost:5432/dbname?")
  assert "require" not in redacted
  assert "secret" not in redacted
  assert "sslmode=<redacted>" in redacted
  assert "password=<redacted>" in redacted


def test_redact_uri_for_log_redacts_when_parse_qsl_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def _raise_value_error(*_args, **_kwargs):
    raise ValueError("bad query")

  monkeypatch.setattr(service_factory, "parse_qsl", _raise_value_error)

  uri = "postgresql://user:pass@localhost:5432/dbname?sslmode=require"
  redacted = service_factory._redact_uri_for_log(uri)

  assert "pass" not in redacted
  assert "require" not in redacted
  assert redacted.endswith("?<redacted>")


def test_redact_uri_for_log_escapes_crlf() -> None:
  uri = (
      "postgresql://user:pass@localhost:5432/dbname\rINJECT\nINJECT"
      "?sslmode=require"
  )

  redacted = service_factory._redact_uri_for_log(uri)

  assert "\r" not in redacted
  assert "\n" not in redacted
  assert "\\rINJECT\\nINJECT" in redacted


def test_redact_uri_for_log_returns_scheme_missing_without_separator() -> None:
  assert (
      service_factory._redact_uri_for_log("user:pass@localhost:5432/dbname")
      == "<scheme-missing>"
  )


@pytest.mark.asyncio
async def test_create_session_service_defaults_to_per_agent_sqlite(
    tmp_path: Path,
) -> None:
  agent_dir = tmp_path / "agent_a"
  agent_dir.mkdir()
  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
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
      use_local_storage=True,
  )

  assert isinstance(service, PerAgentDatabaseSessionService)
  session = await service.create_session(app_name=logical_name, user_id="user")
  assert session.app_name == logical_name
  assert (agent_dir / ".adk" / "session.db").exists()


@pytest.mark.asyncio
async def test_create_session_service_fallbacks_to_database(
    tmp_path: Path, monkeypatch
):
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
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
  await service.close()


def test_create_artifact_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
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
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
  registry.create_artifact_service.return_value = None
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  with pytest.raises(ValueError):
    service_factory.create_artifact_service_from_options(
        base_dir=tmp_path,
        artifact_service_uri="unknown://foo",
        strict_uri=True,
    )


def test_create_memory_service_uses_registry(tmp_path: Path, monkeypatch):
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
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
  registry = mock.create_autospec(ServiceRegistry, instance=True, spec_set=True)
  registry.create_memory_service.return_value = None
  monkeypatch.setattr(service_factory, "get_service_registry", lambda: registry)

  with pytest.raises(ValueError):
    service_factory.create_memory_service_from_options(
        base_dir=tmp_path,
        memory_service_uri="unknown://foo",
    )


@pytest.mark.asyncio
async def test_create_session_service_defaults_to_in_memory_when_disabled(
    tmp_path: Path,
) -> None:
  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=False,
  )

  assert isinstance(service, InMemorySessionService)
  session = await service.create_session(app_name="agent_a", user_id="user")
  assert session.app_name == "agent_a"
  assert not (tmp_path / "agent_a" / ".adk").exists()


def test_create_artifact_service_defaults_to_in_memory_when_disabled(
    tmp_path: Path,
) -> None:
  service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=False,
  )

  assert isinstance(service, InMemoryArtifactService)
  assert not (tmp_path / ".adk").exists()


def test_create_session_service_fallbacks_to_in_memory_on_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def _raise_permission_error(*_args, **_kwargs):
    raise PermissionError("nope")

  monkeypatch.setattr(
      service_factory, "create_local_session_service", _raise_permission_error
  )

  service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )

  assert isinstance(service, InMemorySessionService)


@pytest.mark.skipif(os.name == "nt", reason="chmod behavior differs on Windows")
def test_create_services_default_to_in_memory_when_agents_dir_unwritable(
    tmp_path: Path,
) -> None:
  agents_dir = tmp_path / "agents"
  agents_dir.mkdir()
  try:
    agents_dir.chmod(0o555)
    if os.access(agents_dir, os.W_OK | os.X_OK):
      pytest.skip("Test cannot make directory unwritable in this environment.")

    session_service = service_factory.create_session_service_from_options(
        base_dir=agents_dir,
        use_local_storage=True,
    )
    assert isinstance(session_service, InMemorySessionService)

    artifact_service = service_factory.create_artifact_service_from_options(
        base_dir=agents_dir,
        use_local_storage=True,
    )
    assert isinstance(artifact_service, InMemoryArtifactService)
  finally:
    agents_dir.chmod(0o755)


def test_adk_disable_local_storage_env_forces_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("ADK_DISABLE_LOCAL_STORAGE", "1")

  session_service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(session_service, InMemorySessionService)

  artifact_service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(artifact_service, InMemoryArtifactService)


def test_cloud_run_env_defaults_to_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("K_SERVICE", "adk-service")

  session_service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(session_service, InMemorySessionService)

  artifact_service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(artifact_service, InMemoryArtifactService)


def test_kubernetes_env_defaults_to_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")

  session_service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(session_service, InMemorySessionService)

  artifact_service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(artifact_service, InMemoryArtifactService)


@pytest.mark.asyncio
async def test_adk_force_local_storage_env_overrides_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("ADK_FORCE_LOCAL_STORAGE", "1")
  agent_dir = tmp_path / "agent_a"
  agent_dir.mkdir()

  session_service = service_factory.create_session_service_from_options(
      base_dir=tmp_path,
      use_local_storage=False,
  )
  assert isinstance(session_service, PerAgentDatabaseSessionService)
  await session_service.create_session(app_name="agent_a", user_id="user")
  assert (agent_dir / ".adk" / "session.db").exists()

  artifact_service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=False,
  )
  assert isinstance(artifact_service, FileArtifactService)


def test_create_artifact_service_fallbacks_to_in_memory_on_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def _raise_permission_error(*_args, **_kwargs):
    raise PermissionError("nope")

  monkeypatch.setattr(
      service_factory, "create_local_artifact_service", _raise_permission_error
  )

  service = service_factory.create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )

  assert isinstance(service, InMemoryArtifactService)

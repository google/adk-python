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

from __future__ import annotations

from pathlib import Path

from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.cli.utils.local_storage import create_local_artifact_service
from google.adk.cli.utils.local_storage import create_local_database_session_service
from google.adk.cli.utils.local_storage import create_local_session_service
from google.adk.cli.utils.local_storage import PerAgentDatabaseSessionService
from google.adk.cli.utils.local_storage import PerAgentFileArtifactService
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_per_agent_session_service_creates_scoped_dot_adk(
    tmp_path: Path,
) -> None:
  agent_a = tmp_path / "agent_a"
  agent_b = tmp_path / "agent_b"
  agent_a.mkdir()
  agent_b.mkdir()

  async with PerAgentDatabaseSessionService(agents_root=tmp_path) as service:
    await service.create_session(app_name="agent_a", user_id="user_a")
    await service.create_session(app_name="agent_b", user_id="user_b")

    assert (agent_a / ".adk" / "session.db").exists()
    assert (agent_b / ".adk" / "session.db").exists()

    agent_a_sessions = await service.list_sessions(app_name="agent_a")
    agent_b_sessions = await service.list_sessions(app_name="agent_b")

    assert len(agent_a_sessions.sessions) == 1
    assert agent_a_sessions.sessions[0].app_name == "agent_a"
    assert len(agent_b_sessions.sessions) == 1
    assert agent_b_sessions.sessions[0].app_name == "agent_b"


@pytest.mark.asyncio
async def test_per_agent_session_service_respects_app_name_alias(
    tmp_path: Path,
) -> None:
  folder_name = "agent_folder"
  logical_name = "custom_app"
  (tmp_path / folder_name).mkdir()

  service = create_local_session_service(
      base_dir=tmp_path,
      per_agent=True,
      app_name_to_dir={logical_name: folder_name},
  )
  try:
    session = await service.create_session(
        app_name=logical_name,
        user_id="user",
    )

    assert session.app_name == logical_name
    assert (tmp_path / folder_name / ".adk" / "session.db").exists()
  finally:
    if isinstance(service, PerAgentDatabaseSessionService):
      await service.close()


@pytest.mark.asyncio
async def test_per_agent_session_service_routes_built_in_agents_to_root_dot_adk(
    tmp_path: Path,
) -> None:
  async with PerAgentDatabaseSessionService(agents_root=tmp_path) as service:
    await service.create_session(app_name="__helper", user_id="user")

    assert not (tmp_path / "__helper").exists()
    assert (tmp_path / ".adk" / "session.db").exists()


def test_create_local_database_session_service_returns_sqlite(
    tmp_path: Path,
) -> None:
  service = create_local_database_session_service(base_dir=tmp_path)

  assert isinstance(service, SqliteSessionService)


@pytest.mark.asyncio
async def test_per_agent_session_service_get_user_state(tmp_path: Path) -> None:
  """Verifies get_user_state routes to correct agent and returns correct state."""
  agent_a = tmp_path / "agent_a"
  agent_b = tmp_path / "agent_b"
  agent_a.mkdir()
  agent_b.mkdir()

  async with PerAgentDatabaseSessionService(agents_root=tmp_path) as service:
    session_a = await service.create_session(
        app_name="agent_a", user_id="user_a"
    )
    await service.append_event(
        session_a,
        Event(
            author="system",
            actions=EventActions(
                state_delta={"user:profile": {"name": "Alice"}}
            ),
        ),
    )

    state_a = await service.get_user_state(app_name="agent_a", user_id="user_a")
    state_b = await service.get_user_state(app_name="agent_b", user_id="user_b")

    assert state_a == {"profile": {"name": "Alice"}}
    assert not state_b


@pytest.mark.asyncio
async def test_per_agent_artifact_service_creates_scoped_dot_adk(
    tmp_path: Path,
) -> None:
  agent_a = tmp_path / "agent_a"
  agent_b = tmp_path / "agent_b"
  agent_a.mkdir()
  agent_b.mkdir()

  service = PerAgentFileArtifactService(agents_root=tmp_path)
  artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")

  await service.save_artifact(
      app_name="agent_a",
      user_id="user_a",
      session_id="session_a",
      filename="file.txt",
      artifact=artifact,
  )
  await service.save_artifact(
      app_name="agent_b",
      user_id="user_b",
      session_id="session_b",
      filename="file.txt",
      artifact=artifact,
  )

  assert (agent_a / ".adk" / "artifacts").exists()
  assert (agent_b / ".adk" / "artifacts").exists()
  assert not (tmp_path / ".adk").exists()

  keys_a = await service.list_artifact_keys(
      app_name="agent_a", user_id="user_a", session_id="session_a"
  )
  assert keys_a == ["file.txt"]
  # agent_b's store doesn't see agent_a's artifact, even at the same scope.
  keys_from_other_agent = await service.list_artifact_keys(
      app_name="agent_b", user_id="user_a", session_id="session_a"
  )
  assert keys_from_other_agent == []


@pytest.mark.asyncio
async def test_per_agent_artifact_service_respects_app_name_alias(
    tmp_path: Path,
) -> None:
  folder_name = "agent_folder"
  logical_name = "custom_app"
  (tmp_path / folder_name).mkdir()

  service = create_local_artifact_service(
      base_dir=tmp_path,
      per_agent=True,
      app_name_to_dir={logical_name: folder_name},
  )

  await service.save_artifact(
      app_name=logical_name,
      user_id="user",
      session_id="session",
      filename="file.txt",
      artifact=types.Part.from_bytes(data=b"data", mime_type="text/plain"),
  )

  assert (tmp_path / folder_name / ".adk" / "artifacts").exists()
  assert not (tmp_path / logical_name).exists()


@pytest.mark.asyncio
async def test_per_agent_artifact_service_routes_built_in_agents_to_root_dot_adk(
    tmp_path: Path,
) -> None:
  service = PerAgentFileArtifactService(agents_root=tmp_path)

  await service.save_artifact(
      app_name="__helper",
      user_id="user",
      session_id="session",
      filename="file.txt",
      artifact=types.Part.from_bytes(data=b"data", mime_type="text/plain"),
  )

  assert not (tmp_path / "__helper").exists()
  assert (tmp_path / ".adk" / "artifacts").exists()


def test_create_local_artifact_service_returns_file_service(
    tmp_path: Path,
) -> None:
  service = create_local_artifact_service(base_dir=tmp_path)

  assert isinstance(service, FileArtifactService)


@pytest.mark.asyncio
async def test_per_agent_artifact_service_delegates_all_operations(
    tmp_path: Path,
) -> None:
  (tmp_path / "agent_a").mkdir()
  service = PerAgentFileArtifactService(agents_root=tmp_path)
  artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")
  scope = {"app_name": "agent_a", "user_id": "user", "session_id": "session"}

  version = await service.save_artifact(
      filename="file.txt", artifact=artifact, **scope
  )
  assert version == 0

  loaded = await service.load_artifact(filename="file.txt", **scope)
  assert loaded is not None
  assert loaded.inline_data.data == b"data"
  assert await service.list_versions(filename="file.txt", **scope) == [0]
  assert (
      len(await service.list_artifact_versions(filename="file.txt", **scope))
      == 1
  )
  assert (
      await service.get_artifact_version(filename="file.txt", **scope)
  ) is not None

  await service.delete_artifact(filename="file.txt", **scope)
  assert await service.list_artifact_keys(**scope) == []


@pytest.mark.asyncio
async def test_per_agent_artifact_service_reads_legacy_shared_root(
    tmp_path: Path,
) -> None:
  scope = {"app_name": "agent_a", "user_id": "user", "session_id": "session"}
  # Seed an artifact in the pre-per-agent shared <root>/.adk/artifacts store.
  legacy = FileArtifactService(root_dir=tmp_path / ".adk" / "artifacts")
  await legacy.save_artifact(
      filename="legacy.txt",
      artifact=types.Part.from_bytes(data=b"old", mime_type="text/plain"),
      **scope,
  )

  service = PerAgentFileArtifactService(agents_root=tmp_path)

  loaded = await service.load_artifact(filename="legacy.txt", **scope)
  assert loaded is not None
  assert loaded.inline_data.data == b"old"
  assert await service.list_artifact_keys(**scope) == ["legacy.txt"]
  assert await service.list_versions(filename="legacy.txt", **scope) == [0]
  assert (
      await service.get_artifact_version(filename="legacy.txt", **scope)
  ) is not None


@pytest.mark.asyncio
async def test_per_agent_artifact_service_writes_do_not_touch_legacy_root(
    tmp_path: Path,
) -> None:
  scope = {"app_name": "agent_a", "user_id": "user", "session_id": "session"}
  legacy = FileArtifactService(root_dir=tmp_path / ".adk" / "artifacts")
  await legacy.save_artifact(
      filename="legacy.txt",
      artifact=types.Part.from_bytes(data=b"old", mime_type="text/plain"),
      **scope,
  )

  service = PerAgentFileArtifactService(agents_root=tmp_path)
  await service.save_artifact(
      filename="new.txt",
      artifact=types.Part.from_bytes(data=b"new", mime_type="text/plain"),
      **scope,
  )

  # New write lands per-agent, not in the legacy shared root.
  assert (tmp_path / "agent_a" / ".adk" / "artifacts").exists()
  assert await legacy.list_artifact_keys(**scope) == ["legacy.txt"]
  # Reads union the new per-agent key with the legacy one.
  assert await service.list_artifact_keys(**scope) == ["legacy.txt", "new.txt"]


@pytest.mark.asyncio
async def test_per_agent_artifact_service_no_fallback_without_legacy_dir(
    tmp_path: Path,
) -> None:
  service = PerAgentFileArtifactService(agents_root=tmp_path)

  result = await service.load_artifact(
      app_name="agent_a",
      user_id="user",
      session_id="session",
      filename="missing.txt",
  )

  assert result is None
  assert not (tmp_path / ".adk").exists()


@pytest.mark.asyncio
async def test_per_agent_artifact_service_delete_removes_legacy_copy(
    tmp_path: Path,
) -> None:
  scope = {"app_name": "agent_a", "user_id": "user", "session_id": "session"}
  legacy = FileArtifactService(root_dir=tmp_path / ".adk" / "artifacts")
  await legacy.save_artifact(
      filename="legacy.txt",
      artifact=types.Part.from_bytes(data=b"old", mime_type="text/plain"),
      **scope,
  )

  service = PerAgentFileArtifactService(agents_root=tmp_path)
  await service.delete_artifact(filename="legacy.txt", **scope)

  # Deleted artifact must not reappear through the legacy read fallback.
  assert await service.load_artifact(filename="legacy.txt", **scope) is None
  assert await service.list_artifact_keys(**scope) == []
  assert await legacy.list_artifact_keys(**scope) == []

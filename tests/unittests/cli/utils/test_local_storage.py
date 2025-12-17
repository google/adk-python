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

from __future__ import annotations

from pathlib import Path

from google.adk.cli.utils.local_storage import create_local_database_session_service
from google.adk.cli.utils.local_storage import create_local_session_service
from google.adk.cli.utils.local_storage import PerAgentDatabaseSessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
import pytest


@pytest.mark.asyncio
async def test_per_agent_session_service_creates_scoped_dot_adk(
    tmp_path: Path,
) -> None:
  agent_a = tmp_path / "agent_a"
  agent_b = tmp_path / "agent_b"
  agent_a.mkdir()
  agent_b.mkdir()

  service = PerAgentDatabaseSessionService(agents_root=tmp_path)

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

  session = await service.create_session(
      app_name=logical_name,
      user_id="user",
  )

  assert session.app_name == logical_name
  assert (tmp_path / folder_name / ".adk" / "session.db").exists()


@pytest.mark.asyncio
async def test_per_agent_session_service_routes_built_in_agents_to_root_dot_adk(
    tmp_path: Path,
) -> None:
  service = PerAgentDatabaseSessionService(agents_root=tmp_path)

  await service.create_session(app_name="__helper", user_id="user")

  assert not (tmp_path / "__helper").exists()
  assert (tmp_path / ".adk" / "session.db").exists()


def test_create_local_database_session_service_returns_sqlite(
    tmp_path: Path,
) -> None:
  service = create_local_database_session_service(base_dir=tmp_path)

  assert isinstance(service, SqliteSessionService)

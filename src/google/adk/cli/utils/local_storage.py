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
"""Utilities for local .adk folder persistence."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Mapping
from typing import Optional

from typing_extensions import override

from ...artifacts.base_artifact_service import BaseArtifactService
from ...artifacts.file_artifact_service import FileArtifactService
from ...events.event import Event
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from .dot_adk_folder import dot_adk_folder_for_agent
from .dot_adk_folder import DotAdkFolder

logger = logging.getLogger("google_adk." + __name__)


def create_local_database_session_service(
    *,
    base_dir: Path | str,
) -> BaseSessionService:
  """Creates a SQLite-backed session service at .adk/session.db.

  Args:
    base_dir: The base directory for the agent (parent of .adk folder).

  Returns:
    A SqliteSessionService instance.
  """
  from ...sessions.sqlite_session_service import SqliteSessionService

  manager = DotAdkFolder(base_dir)
  manager.dot_adk_dir.mkdir(parents=True, exist_ok=True)

  session_db_path = manager.session_db_path

  logger.info("Creating local session service at %s", session_db_path)
  return SqliteSessionService(db_path=str(session_db_path))


def create_local_session_service(
    *,
    base_dir: Path | str,
    per_agent: bool = False,
    app_name_to_dir: Optional[Mapping[str, str]] = None,
) -> BaseSessionService:
  """Creates a local SQLite-backed session service.

  Args:
    base_dir: The base directory for the agent(s).
    per_agent: If True, creates a PerAgentDatabaseSessionService that stores
      sessions in each agent's .adk folder. If False, creates a single
      SqliteSessionService at base_dir/.adk/session.db.
    app_name_to_dir: Optional mapping from logical app name to on-disk agent
      folder name. Only used when per_agent is True; defaults to identity.

  Returns:
    A BaseSessionService instance backed by SQLite.
  """
  if per_agent:
    logger.info(
        "Using per-agent session storage rooted at %s",
        base_dir,
    )
    return PerAgentDatabaseSessionService(
        agents_root=base_dir,
        app_name_to_dir=app_name_to_dir,
    )

  return create_local_database_session_service(base_dir=base_dir)


def create_local_artifact_service(
    *, base_dir: Path | str
) -> BaseArtifactService:
  """Creates a file-backed artifact service rooted in `.adk/artifacts`.

  Args:
    base_dir: Directory whose `.adk` folder will store artifacts.

  Returns:
    A `FileArtifactService` scoped to the derived root directory.
  """
  manager = DotAdkFolder(base_dir)
  artifact_root = manager.artifacts_dir
  artifact_root.mkdir(parents=True, exist_ok=True)
  logger.info("Using file artifact service at %s", artifact_root)
  return FileArtifactService(root_dir=artifact_root)


class PerAgentDatabaseSessionService(BaseSessionService):
  """Routes session storage to per-agent `.adk/session.db` files."""

  def __init__(
      self,
      *,
      agents_root: Path | str,
      app_name_to_dir: Optional[Mapping[str, str]] = None,
  ):
    self._agents_root = Path(agents_root).resolve()
    self._app_name_to_dir = dict(app_name_to_dir or {})
    self._services: dict[str, BaseSessionService] = {}
    self._service_lock = asyncio.Lock()

  async def _get_service(self, app_name: str) -> BaseSessionService:
    async with self._service_lock:
      storage_name = self._app_name_to_dir.get(app_name, app_name)
      service = self._services.get(storage_name)
      if service is not None:
        return service
      folder = dot_adk_folder_for_agent(
          agents_root=self._agents_root, app_name=storage_name
      )
      service = create_local_database_session_service(
          base_dir=folder.agent_dir,
      )
      self._services[storage_name] = service
      return service

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, object]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    service = await self._get_service(app_name)
    return await service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    service = await self._get_service(app_name)
    return await service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        config=config,
    )

  @override
  async def list_sessions(
      self,
      *,
      app_name: str,
      user_id: Optional[str] = None,
  ) -> ListSessionsResponse:
    service = await self._get_service(app_name)
    return await service.list_sessions(app_name=app_name, user_id=user_id)

  @override
  async def delete_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> None:
    service = await self._get_service(app_name)
    await service.delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    service = await self._get_service(session.app_name)
    return await service.append_event(session, event)

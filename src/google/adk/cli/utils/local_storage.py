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
"""Utilities for local .adk folder persistence."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional

from google.genai import types
from typing_extensions import override

from ...artifacts.base_artifact_service import ArtifactVersion
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

_BUILT_IN_SESSION_SERVICE_KEY = "__adk_built_in_session_service__"
_BUILT_IN_ARTIFACT_SERVICE_KEY = "__adk_built_in_artifact_service__"


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
    *,
    base_dir: Path | str,
    per_agent: bool = False,
    app_name_to_dir: Optional[Mapping[str, str]] = None,
) -> BaseArtifactService:
  """Creates a file-backed artifact service that persists data in `.adk/artifacts` folders.

  Args:
    base_dir: Directory whose `.adk` folder will store artifacts.
    per_agent: If True, creates a PerAgentFileArtifactService that stores
      artifacts in each agent's `.adk/artifacts` folder. If False, creates a
      single FileArtifactService at base_dir/.adk/artifacts.
    app_name_to_dir: Optional mapping from logical app name to on-disk agent
      folder name. Only used when per_agent is True; defaults to identity.

  Returns:
    A `BaseArtifactService` backed by the local filesystem.
  """
  if per_agent:
    logger.info("Using per-agent artifact storage rooted at %s", base_dir)
    return PerAgentFileArtifactService(
        agents_root=base_dir,
        app_name_to_dir=app_name_to_dir,
    )

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
      if app_name.startswith("__"):
        storage_key = _BUILT_IN_SESSION_SERVICE_KEY
        base_dir = self._agents_root
      else:
        storage_key = self._app_name_to_dir.get(app_name, app_name)
        folder = dot_adk_folder_for_agent(
            agents_root=self._agents_root, app_name=storage_key
        )
        base_dir = folder.agent_dir

      service = self._services.get(storage_key)
      if service is not None:
        return service

      service = create_local_database_session_service(
          base_dir=base_dir,
      )

      self._services[storage_key] = service
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
  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    service = await self._get_service(app_name)
    return await service.get_user_state(app_name=app_name, user_id=user_id)

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    service = await self._get_service(session.app_name)
    return await service.append_event(session, event)

  async def close(self) -> None:
    """Closes all underlying session services."""
    for service in self._services.values():
      if hasattr(service, "close"):
        await service.close()
    self._services.clear()

  async def __aenter__(self) -> PerAgentDatabaseSessionService:
    """Enters the async context manager."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exits the async context manager and closes the service."""
    await self.close()


class PerAgentFileArtifactService(BaseArtifactService):
  """Routes artifact storage to per-agent `.adk/artifacts` folders."""

  def __init__(
      self,
      *,
      agents_root: Path | str,
      app_name_to_dir: Optional[Mapping[str, str]] = None,
  ):
    self._agents_root = Path(agents_root).resolve()
    self._app_name_to_dir = dict(app_name_to_dir or {})
    self._services: dict[str, BaseArtifactService] = {}
    self._legacy_service: Optional[BaseArtifactService] = None
    self._service_lock = asyncio.Lock()

  async def _get_service(self, app_name: str) -> BaseArtifactService:
    async with self._service_lock:
      if app_name.startswith("__"):
        storage_key = _BUILT_IN_ARTIFACT_SERVICE_KEY
        base_dir = self._agents_root
      else:
        storage_key = self._app_name_to_dir.get(app_name, app_name)
        folder = dot_adk_folder_for_agent(
            agents_root=self._agents_root, app_name=storage_key
        )
        base_dir = folder.agent_dir

      service = self._services.get(storage_key)
      if service is not None:
        return service

      service = create_local_artifact_service(base_dir=base_dir)
      self._services[storage_key] = service
      return service

  async def _get_legacy_service(
      self, app_name: str
  ) -> Optional[BaseArtifactService]:
    """Returns a reader for the pre-per-agent shared `.adk/artifacts` root.

    Returns None for built-in agents (which already use that root) and when
    no legacy directory exists, so reads fall back only when there is legacy
    data to find. Never creates the legacy directory.
    """
    if app_name.startswith("__"):
      return None
    if self._legacy_service is not None:
      return self._legacy_service
    legacy_dir = DotAdkFolder(self._agents_root).artifacts_dir
    if not legacy_dir.exists():
      return None
    async with self._service_lock:
      if self._legacy_service is None:
        self._legacy_service = FileArtifactService(root_dir=legacy_dir)
      return self._legacy_service

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: types.Part | dict[str, Any],
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    service = await self._get_service(app_name)
    return await service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        artifact=artifact,
        session_id=session_id,
        custom_metadata=custom_metadata,
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    service = await self._get_service(app_name)
    result = await service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
        version=version,
    )
    if result is not None:
      return result
    legacy = await self._get_legacy_service(app_name)
    if legacy is None:
      return None
    return await legacy.load_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
        version=version,
    )

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    service = await self._get_service(app_name)
    keys = await service.list_artifact_keys(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    legacy = await self._get_legacy_service(app_name)
    if legacy is None:
      return keys
    legacy_keys = await legacy.list_artifact_keys(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    return sorted(set(keys) | set(legacy_keys))

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    service = await self._get_service(app_name)
    await service.delete_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )
    # Also delete any legacy copy so a deleted artifact can't reappear via the
    # read fallback.
    legacy = await self._get_legacy_service(app_name)
    if legacy is not None:
      await legacy.delete_artifact(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    service = await self._get_service(app_name)
    versions = await service.list_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )
    if versions:
      return versions
    legacy = await self._get_legacy_service(app_name)
    if legacy is None:
      return versions
    return await legacy.list_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    service = await self._get_service(app_name)
    versions = await service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )
    if versions:
      return versions
    legacy = await self._get_legacy_service(app_name)
    if legacy is None:
      return versions
    return await legacy.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    service = await self._get_service(app_name)
    result = await service.get_artifact_version(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
        version=version,
    )
    if result is not None:
      return result
    legacy = await self._get_legacy_service(app_name)
    if legacy is None:
      return None
    return await legacy.get_artifact_version(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
        version=version,
    )

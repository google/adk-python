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

import logging
from pathlib import Path
from typing import Any
from typing import Optional

from ...artifacts.base_artifact_service import BaseArtifactService
from ...memory.base_memory_service import BaseMemoryService
from ...sessions.base_session_service import BaseSessionService
from ..service_registry import get_service_registry
from .local_storage import create_local_artifact_service
from .local_storage import create_local_session_service

logger = logging.getLogger("google_adk." + __name__)


def create_session_service_from_options(
    *,
    base_dir: Path | str,
    session_service_uri: Optional[str] = None,
    session_db_kwargs: Optional[dict[str, Any]] = None,
    app_name_to_dir: Optional[dict[str, str]] = None,
) -> BaseSessionService:
  """Creates a session service based on CLI/web options."""
  base_path = Path(base_dir)
  registry = get_service_registry()

  kwargs: dict[str, Any] = {
      "agents_dir": str(base_path),
  }
  if session_db_kwargs:
    kwargs.update(session_db_kwargs)

  if session_service_uri:
    logger.info("Using session service URI: %s", session_service_uri)
    service = registry.create_session_service(session_service_uri, **kwargs)
    if service is not None:
      return service

    # Fallback to DatabaseSessionService if the registry doesn't support the
    # session service URI scheme. This keeps support for SQLAlchemy-compatible
    # databases like AlloyDB or Cloud Spanner without explicit registration.
    from ...sessions.database_session_service import DatabaseSessionService

    fallback_kwargs = dict(kwargs)
    fallback_kwargs.pop("agents_dir", None)
    logger.info(
        "Falling back to DatabaseSessionService for URI: %s",
        session_service_uri,
    )
    return DatabaseSessionService(db_url=session_service_uri, **fallback_kwargs)

  # Default to per-agent local SQLite storage in <agents_root>/<agent>/.adk/.
  return create_local_session_service(
      base_dir=base_path,
      per_agent=True,
      app_name_to_dir=app_name_to_dir,
  )


def create_memory_service_from_options(
    *,
    base_dir: Path | str,
    memory_service_uri: Optional[str] = None,
) -> BaseMemoryService:
  """Creates a memory service based on CLI/web options."""
  base_path = Path(base_dir)
  registry = get_service_registry()

  if memory_service_uri:
    logger.info("Using memory service URI: %s", memory_service_uri)
    service = registry.create_memory_service(
        memory_service_uri,
        agents_dir=str(base_path),
    )
    if service is None:
      raise ValueError(f"Unsupported memory service URI: {memory_service_uri}")
    return service

  logger.info("Using in-memory memory service")
  from ...memory.in_memory_memory_service import InMemoryMemoryService

  return InMemoryMemoryService()


def create_artifact_service_from_options(
    *,
    base_dir: Path | str,
    artifact_service_uri: Optional[str] = None,
    strict_uri: bool = False,
) -> BaseArtifactService:
  """Creates an artifact service based on CLI/web options."""
  base_path = Path(base_dir)
  registry = get_service_registry()

  if artifact_service_uri:
    logger.info("Using artifact service URI: %s", artifact_service_uri)
    service = registry.create_artifact_service(
        artifact_service_uri,
        agents_dir=str(base_path),
    )
    if service is None:
      if strict_uri:
        raise ValueError(
            f"Unsupported artifact service URI: {artifact_service_uri}"
        )
      logger.warning(
          "Unsupported artifact service URI: %s, falling back to in-memory",
          artifact_service_uri,
      )
      from ...artifacts.in_memory_artifact_service import InMemoryArtifactService

      return InMemoryArtifactService()
    return service

  return create_local_artifact_service(base_dir=base_path)

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

import errno
import logging
import os
from pathlib import Path
from typing import Any
from typing import Optional
from urllib.parse import parse_qsl
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from ...artifacts.base_artifact_service import BaseArtifactService
from ...memory.base_memory_service import BaseMemoryService
from ...sessions.base_session_service import BaseSessionService
from ...utils.env_utils import is_env_enabled
from ..service_registry import get_service_registry
from .local_storage import create_local_artifact_service
from .local_storage import create_local_session_service

logger = logging.getLogger("google_adk." + __name__)

_DISABLE_LOCAL_STORAGE_ENV = "ADK_DISABLE_LOCAL_STORAGE"
_FORCE_LOCAL_STORAGE_ENV = "ADK_FORCE_LOCAL_STORAGE"
_LOCAL_STORAGE_ERRNOS = frozenset({
    errno.EACCES,
    errno.EPERM,
    errno.EROFS,
})

_CLOUD_RUN_SERVICE_ENV = "K_SERVICE"
_KUBERNETES_HOST_ENV = "KUBERNETES_SERVICE_HOST"


def _redact_uri_for_log(uri: str) -> str:
  """Returns a safe-to-log representation of a URI.

  Redacts user info (username/password) and query parameter values.
  """
  if not uri or not uri.strip():
    return "<empty>"
  sanitized = uri.replace("\r", "\\r").replace("\n", "\\n")
  if "://" not in sanitized:
    return "<scheme-missing>"
  try:
    parsed = urlsplit(sanitized)
  except ValueError:
    return "<unparseable>"

  if not parsed.scheme:
    return "<scheme-missing>"

  netloc = parsed.netloc
  if "@" in netloc:
    _, netloc = netloc.rsplit("@", 1)

  if parsed.query:
    try:
      redacted_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
      query = "<redacted>"
    else:
      query = "&".join(f"{key}=<redacted>" for key, _ in redacted_pairs)
  else:
    query = ""

  return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def _is_cloud_run() -> bool:
  """Returns True when running in Cloud Run."""
  return bool(os.environ.get(_CLOUD_RUN_SERVICE_ENV))


def _is_kubernetes() -> bool:
  """Returns True when running in Kubernetes (including GKE)."""
  return bool(os.environ.get(_KUBERNETES_HOST_ENV))


def _is_dir_writable(path: Path) -> bool:
  """Returns True if the directory exists and is writable/executable."""
  try:
    if not path.exists() or not path.is_dir():
      return False
  except OSError:
    return False
  return os.access(path, os.W_OK | os.X_OK)


def _resolve_use_local_storage(
    *,
    base_path: Path,
    requested: bool,
) -> tuple[bool, str | None]:
  """Resolves effective local storage setting with safe defaults."""
  if is_env_enabled(_DISABLE_LOCAL_STORAGE_ENV):
    warning_message = (
        "Local storage is disabled by %s; using in-memory services. "
        "Set --session_service_uri/--artifact_service_uri for production "
        "deployments."
    ) % _DISABLE_LOCAL_STORAGE_ENV
    return False, warning_message

  if is_env_enabled(_FORCE_LOCAL_STORAGE_ENV):
    if not _is_dir_writable(base_path):
      warning_message = (
          "Local storage is forced by %s, but %s is not writable; "
          "using in-memory services."
      ) % (_FORCE_LOCAL_STORAGE_ENV, base_path)
      return False, warning_message
    return True, None

  if not requested:
    return False, None

  if _is_cloud_run() or _is_kubernetes():
    warning_message = (
        "Detected Cloud Run/Kubernetes runtime; using in-memory services "
        "instead of local .adk storage. Set %s=1 to force local storage."
    ) % _FORCE_LOCAL_STORAGE_ENV
    return False, warning_message

  if not _is_dir_writable(base_path):
    warning_message = (
        "Agents directory %s is not writable; using in-memory services "
        "instead of local .adk storage. Set %s=1 to force local storage."
    ) % (base_path, _FORCE_LOCAL_STORAGE_ENV)
    return False, warning_message

  return True, None


def _create_in_memory_session_service(
    warning_message: str | None = None,
    *warning_args: object,
) -> BaseSessionService:
  """Creates an in-memory session service, optionally logging a warning."""
  if warning_message is not None:
    logger.warning(warning_message, *warning_args)
  from ...sessions.in_memory_session_service import InMemorySessionService

  return InMemorySessionService()


def _create_in_memory_artifact_service(
    warning_message: str | None = None,
    *warning_args: object,
) -> BaseArtifactService:
  """Creates an in-memory artifact service, optionally logging a warning."""
  if warning_message is not None:
    logger.warning(warning_message, *warning_args)
  from ...artifacts.in_memory_artifact_service import InMemoryArtifactService

  return InMemoryArtifactService()


def create_session_service_from_options(
    *,
    base_dir: Path | str,
    session_service_uri: Optional[str] = None,
    session_db_kwargs: Optional[dict[str, Any]] = None,
    app_name_to_dir: Optional[dict[str, str]] = None,
    use_local_storage: bool = True,
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
    logger.info(
        "Using session service URI: %s",
        _redact_uri_for_log(session_service_uri),
    )
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
        _redact_uri_for_log(session_service_uri),
    )
    return DatabaseSessionService(db_url=session_service_uri, **fallback_kwargs)

  effective_use_local_storage, auto_warning = _resolve_use_local_storage(
      base_path=base_path,
      requested=use_local_storage,
  )
  if not effective_use_local_storage:
    if auto_warning is not None:
      return _create_in_memory_session_service(auto_warning)
    return _create_in_memory_session_service(
        "Local session storage is disabled; using in-memory session service. "
        "Set --session_service_uri for production deployments."
    )

  # Default to per-agent local SQLite storage in <agents_root>/<agent>/.adk/.
  try:
    return create_local_session_service(
        base_dir=base_path,
        per_agent=True,
        app_name_to_dir=app_name_to_dir,
    )
  except OSError as exc:
    if exc.errno not in _LOCAL_STORAGE_ERRNOS and not isinstance(
        exc, PermissionError
    ):
      raise
    return _create_in_memory_session_service(
        "Failed to initialize local session storage under %s (%r); "
        "falling back to in-memory session service.",
        base_path,
        exc,
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
    logger.info(
        "Using memory service URI: %s", _redact_uri_for_log(memory_service_uri)
    )
    service = registry.create_memory_service(
        memory_service_uri,
        agents_dir=str(base_path),
    )
    if service is None:
      raise ValueError(
          "Unsupported memory service URI: %s"
          % _redact_uri_for_log(memory_service_uri)
      )
    return service

  logger.info("Using in-memory memory service")
  from ...memory.in_memory_memory_service import InMemoryMemoryService

  return InMemoryMemoryService()


def create_artifact_service_from_options(
    *,
    base_dir: Path | str,
    artifact_service_uri: Optional[str] = None,
    strict_uri: bool = False,
    use_local_storage: bool = True,
) -> BaseArtifactService:
  """Creates an artifact service based on CLI/web options."""
  base_path = Path(base_dir)
  registry = get_service_registry()

  if artifact_service_uri:
    logger.info(
        "Using artifact service URI: %s",
        _redact_uri_for_log(artifact_service_uri),
    )
    service = registry.create_artifact_service(
        artifact_service_uri,
        agents_dir=str(base_path),
    )
    if service is None:
      if strict_uri:
        raise ValueError(
            "Unsupported artifact service URI: %s"
            % _redact_uri_for_log(artifact_service_uri)
        )
      return _create_in_memory_artifact_service(
          "Unsupported artifact service URI: %s, falling back to in-memory",
          _redact_uri_for_log(artifact_service_uri),
      )
    return service

  effective_use_local_storage, auto_warning = _resolve_use_local_storage(
      base_path=base_path,
      requested=use_local_storage,
  )
  if not effective_use_local_storage:
    if auto_warning is not None:
      return _create_in_memory_artifact_service(auto_warning)
    return _create_in_memory_artifact_service(
        "Local artifact storage is disabled; using in-memory artifact service. "
        "Set --artifact_service_uri for production deployments."
    )

  try:
    return create_local_artifact_service(base_dir=base_path)
  except OSError as exc:
    if exc.errno not in _LOCAL_STORAGE_ERRNOS and not isinstance(
        exc, PermissionError
    ):
      raise
    return _create_in_memory_artifact_service(
        "Failed to initialize local artifact storage under %s (%r); "
        "falling back to in-memory artifact service.",
        base_path,
        exc,
    )

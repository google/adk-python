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

"""Service wrappers that pin the ``app_name`` used for persisted storage.

When agents are served via ``adk web`` / ``adk api_server`` by pointing at a
directory, the ``app_name`` used to key sessions and artifacts is derived from
the agent's *folder name*. Two servers that load an identically-named agent
folder against a shared backend therefore write to the same ``app_name`` and
share each other's sessions.

These thin decorators let a server pin a distinct ``app_name`` for its
persisted data without renaming the folder (so the loader, the REST routes and
the web UI keep using the folder name, while storage is namespaced by the
override). They are wired up from :func:`get_fast_api_app` when an override is
supplied via the ``app_name`` argument, the ``--app_name`` CLI flag, or the
``ADK_APP_NAME`` environment variable.

The wrappers only rewrite the ``app_name`` keyword argument; every other
argument and any implementation-specific attribute is delegated unchanged.
"""

from __future__ import annotations

from typing import Any

from ...artifacts.base_artifact_service import BaseArtifactService
from ...events.event import Event
from ...sessions.base_session_service import BaseSessionService
from ...sessions.session import Session


class _AppNameOverrideBase:
  """Holds the wrapped service and the override, delegating unknown attrs."""

  def __init__(self, delegate: Any, app_name: str) -> None:
    self._delegate = delegate
    self._app_name = app_name

  def __getattr__(self, name: str) -> Any:
    # __getattr__ only runs for attributes not found normally. Guard the two
    # private attributes so a lookup before __init__ completes (e.g. during
    # copy/pickle) raises AttributeError instead of recursing forever.
    if name in ("_delegate", "_app_name"):
      raise AttributeError(name)
    return getattr(self._delegate, name)


class AppNameOverrideSessionService(_AppNameOverrideBase, BaseSessionService):
  """A :class:`BaseSessionService` that forces a fixed ``app_name``."""

  async def create_session(self, **kwargs: Any) -> Session:
    kwargs["app_name"] = self._app_name
    return await self._delegate.create_session(**kwargs)

  async def get_session(self, **kwargs: Any):
    kwargs["app_name"] = self._app_name
    return await self._delegate.get_session(**kwargs)

  async def list_sessions(self, **kwargs: Any):
    kwargs["app_name"] = self._app_name
    return await self._delegate.list_sessions(**kwargs)

  async def delete_session(self, **kwargs: Any) -> None:
    kwargs["app_name"] = self._app_name
    return await self._delegate.delete_session(**kwargs)

  async def get_user_state(self, **kwargs: Any) -> dict[str, Any]:
    kwargs["app_name"] = self._app_name
    return await self._delegate.get_user_state(**kwargs)

  async def append_event(self, session: Session, event: Event) -> Event:
    # ``session.app_name`` already carries the override, because the session
    # was created or fetched through this wrapper, so no rewrite is needed.
    return await self._delegate.append_event(session=session, event=event)

  async def flush(self) -> None:
    return await self._delegate.flush()


class AppNameOverrideArtifactService(_AppNameOverrideBase, BaseArtifactService):
  """A :class:`BaseArtifactService` that forces a fixed ``app_name``."""

  async def save_artifact(self, **kwargs: Any) -> int:
    kwargs["app_name"] = self._app_name
    return await self._delegate.save_artifact(**kwargs)

  async def load_artifact(self, **kwargs: Any):
    kwargs["app_name"] = self._app_name
    return await self._delegate.load_artifact(**kwargs)

  async def list_artifact_keys(self, **kwargs: Any) -> list[str]:
    kwargs["app_name"] = self._app_name
    return await self._delegate.list_artifact_keys(**kwargs)

  async def delete_artifact(self, **kwargs: Any) -> None:
    kwargs["app_name"] = self._app_name
    return await self._delegate.delete_artifact(**kwargs)

  async def list_versions(self, **kwargs: Any) -> list[int]:
    kwargs["app_name"] = self._app_name
    return await self._delegate.list_versions(**kwargs)

  async def list_artifact_versions(self, **kwargs: Any):
    kwargs["app_name"] = self._app_name
    return await self._delegate.list_artifact_versions(**kwargs)

  async def get_artifact_version(self, **kwargs: Any):
    kwargs["app_name"] = self._app_name
    return await self._delegate.get_artifact_version(**kwargs)


def maybe_override_app_name(
    app_name: str | None,
    *,
    session_service: BaseSessionService,
    artifact_service: BaseArtifactService,
) -> tuple[BaseSessionService, BaseArtifactService]:
  """Wraps the services to pin ``app_name`` when an override is provided.

  Args:
    app_name: The override application name. When falsy the services are
      returned unchanged.
    session_service: The session service to wrap.
    artifact_service: The artifact service to wrap.

  Returns:
    A ``(session_service, artifact_service)`` tuple, wrapped when an override
    is supplied and unchanged otherwise.
  """
  if not app_name:
    return session_service, artifact_service
  return (
      AppNameOverrideSessionService(session_service, app_name),
      AppNameOverrideArtifactService(artifact_service, app_name),
  )

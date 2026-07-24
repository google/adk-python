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

from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils.app_name_override import AppNameOverrideArtifactService
from google.adk.cli.utils.app_name_override import AppNameOverrideSessionService
from google.adk.cli.utils.app_name_override import maybe_override_app_name
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

# The "folder name" a caller/route uses; the override is what should be stored.
FOLDER = "assistant"
OVERRIDE = "tenant_a"
USER = "u1"


@pytest.mark.asyncio
async def test_session_override_stores_under_override_name():
  delegate = InMemorySessionService()
  service = AppNameOverrideSessionService(delegate, OVERRIDE)

  created = await service.create_session(
      app_name=FOLDER, user_id=USER, session_id="s1"
  )
  # The returned session is keyed by the override, not the folder name.
  assert created.app_name == OVERRIDE

  # It is reachable through the override on the underlying service ...
  assert (
      await delegate.get_session(
          app_name=OVERRIDE, user_id=USER, session_id="s1"
      )
      is not None
  )
  # ... and NOT under the original folder name.
  assert (
      await delegate.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is None
  )


@pytest.mark.asyncio
async def test_session_override_get_list_delete_roundtrip():
  service = AppNameOverrideSessionService(InMemorySessionService(), OVERRIDE)

  await service.create_session(app_name=FOLDER, user_id=USER, session_id="s1")

  # get/list/delete all accept the folder name but operate on the override.
  assert (
      await service.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is not None
  )
  listed = await service.list_sessions(app_name=FOLDER, user_id=USER)
  assert [s.id for s in listed.sessions] == ["s1"]

  await service.delete_session(app_name=FOLDER, user_id=USER, session_id="s1")
  assert (
      await service.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is None
  )


@pytest.mark.asyncio
async def test_two_overrides_do_not_share_a_backend():
  """Two servers, same folder + backend, distinct overrides -> isolated."""
  delegate = InMemorySessionService()
  service_a = AppNameOverrideSessionService(delegate, "tenant_a")
  service_b = AppNameOverrideSessionService(delegate, "tenant_b")

  await service_a.create_session(app_name=FOLDER, user_id=USER, session_id="s1")

  # B uses the same folder name but a different override -> cannot see A's.
  assert (
      await service_b.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is None
  )
  # A second server with the SAME override does share, on purpose.
  service_a2 = AppNameOverrideSessionService(delegate, "tenant_a")
  assert (
      await service_a2.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is not None
  )


@pytest.mark.asyncio
async def test_artifact_override_stores_under_override_name():
  delegate = InMemoryArtifactService()
  service = AppNameOverrideArtifactService(delegate, OVERRIDE)
  part = types.Part.from_text(text="hello")

  await service.save_artifact(
      app_name=FOLDER,
      user_id=USER,
      session_id="s1",
      filename="a.txt",
      artifact=part,
  )

  # Stored under the override, not the folder name.
  assert (
      await delegate.load_artifact(
          app_name=OVERRIDE, user_id=USER, session_id="s1", filename="a.txt"
      )
      is not None
  )
  assert (
      await delegate.load_artifact(
          app_name=FOLDER, user_id=USER, session_id="s1", filename="a.txt"
      )
      is None
  )


def test_maybe_override_is_noop_without_app_name():
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()

  s, a = maybe_override_app_name(
      None, session_service=session_service, artifact_service=artifact_service
  )
  assert s is session_service
  assert a is artifact_service


def test_maybe_override_wraps_when_app_name_given():
  s, a = maybe_override_app_name(
      OVERRIDE,
      session_service=InMemorySessionService(),
      artifact_service=InMemoryArtifactService(),
  )
  assert isinstance(s, AppNameOverrideSessionService)
  assert isinstance(a, AppNameOverrideArtifactService)


def test_unknown_attributes_delegate_through():
  delegate = InMemorySessionService()
  service = AppNameOverrideSessionService(delegate, OVERRIDE)
  # sessions dict is an impl detail of InMemorySessionService; it must be
  # reachable via delegation rather than raising AttributeError.
  assert service.sessions is delegate.sessions


def _make_session_backend(name: str, tmp_path):
  """Builds a concrete session backend, skipping when drivers are absent.

  The override wrapper operates at the ``BaseSessionService`` abstraction, so it
  must behave identically regardless of the underlying storage. ``postgresql``
  and ``mysql`` use the very same ``DatabaseSessionService`` class as the
  ``sqlite+aiosqlite`` case below (only the connection URL differs), so covering
  the SQLAlchemy path with sqlite exercises the same code that runs on Postgres.
  """
  if name == "in_memory":
    return InMemorySessionService()
  if name == "database_sqlite":
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    from google.adk.sessions.database_session_service import (
        DatabaseSessionService,
    )

    db_path = (tmp_path / "database.sqlite").as_posix()
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_path}")
  if name == "sqlite_service":
    pytest.importorskip("aiosqlite")
    from google.adk.sessions.sqlite_session_service import SqliteSessionService

    return SqliteSessionService(db_path=(tmp_path / "sqlite.db").as_posix())
  raise ValueError(name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_name",
    ["in_memory", "database_sqlite", "sqlite_service"],
)
async def test_override_is_backend_agnostic(backend_name, tmp_path):
  """The override must isolate storage identically on every backend.

  This is the guard that keeps the feature non-breaking across databases: the
  wrapper only rewrites the ``app_name`` string, so create/get/list/delete must
  round-trip under the override (and never under the folder name) whether the
  backend is in-memory, a SQLAlchemy database, or the sqlite service.
  """
  delegate = _make_session_backend(backend_name, tmp_path)
  service = AppNameOverrideSessionService(delegate, OVERRIDE)

  created = await service.create_session(
      app_name=FOLDER, user_id=USER, session_id="s1"
  )
  assert created.app_name == OVERRIDE

  fetched = await service.get_session(
      app_name=FOLDER, user_id=USER, session_id="s1"
  )
  assert fetched is not None and fetched.app_name == OVERRIDE

  listed = await service.list_sessions(app_name=FOLDER, user_id=USER)
  assert [s.id for s in listed.sessions] == ["s1"]

  # Persisted under the override, never under the caller-supplied folder name.
  assert (
      await delegate.get_session(
          app_name=OVERRIDE, user_id=USER, session_id="s1"
      )
      is not None
  )
  assert (
      await delegate.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is None
  )

  await service.delete_session(app_name=FOLDER, user_id=USER, session_id="s1")
  assert (
      await service.get_session(
          app_name=FOLDER, user_id=USER, session_id="s1"
      )
      is None
  )

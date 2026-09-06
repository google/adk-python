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

"""Backend registry behind the shared session service contract tests.

Every test that takes the ``session_service`` fixture states a behavior all
``BaseSessionService`` implementations owe their callers. A backend is only
held to those behaviors once it is registered here, so one left out of this
list can drift from the contract with no test disagreeing.

A backend that fails a contract test has to record it in ``divergences`` with
a written reason. The test is then marked ``xfail(strict=True)``, so the entry
becomes a defect anyone can pick up, and whoever fixes the backend has to
delete the entry in the same change.

The Vertex AI and Firestore backends are still missing from the list. Each
needs a stateful in-memory stand-in for its storage API first: the Firestore
tests drive a call-by-call mock that holds no state, and the Agent Engine fake
keys sessions by id alone rather than by app and user.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Mapping
import contextlib
import dataclasses
import pathlib

from google.adk.cli.utils.local_storage import PerAgentDatabaseSessionService
from google.adk.features import FeatureName
from google.adk.features import override_feature_enabled
from google.adk.integrations.redis._config import RedisSessionServiceConfig
from google.adk.integrations.redis._redis_session_service import RedisSessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
import pytest

from ..integrations.redis._fake_redis import FakeRedisAsync

_MakeService = Callable[
    [pathlib.Path], contextlib.AbstractAsyncContextManager[BaseSessionService]
]


@dataclasses.dataclass(frozen=True)
class _Backend:
  """A session service implementation held to the shared contract."""

  name: str
  make: _MakeService
  divergences: Mapping[str, str] = dataclasses.field(default_factory=dict)
  """Contract test name -> the written reason this backend fails it today."""


@contextlib.asynccontextmanager
async def _make_in_memory(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  del tmp_path
  yield InMemorySessionService()


@contextlib.asynccontextmanager
async def _make_in_memory_light_copy(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  del tmp_path
  override_feature_enabled(
      FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, True
  )
  try:
    yield InMemorySessionService()
  finally:
    override_feature_enabled(
        FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, False
    )


@contextlib.asynccontextmanager
async def _make_database(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  del tmp_path
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  try:
    yield service
  finally:
    await service.close()


@contextlib.asynccontextmanager
async def _make_sqlite(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  yield SqliteSessionService(str(tmp_path / 'sqlite.db'))


@contextlib.asynccontextmanager
async def _make_redis(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  del tmp_path
  yield RedisSessionService(
      config=RedisSessionServiceConfig(key_prefix='conformance:session:'),
      redis_client=FakeRedisAsync(),
  )


@contextlib.asynccontextmanager
async def _make_per_agent_database(
    tmp_path: pathlib.Path,
) -> AsyncIterator[BaseSessionService]:
  service = PerAgentDatabaseSessionService(agents_root=tmp_path)
  try:
    yield service
  finally:
    await service.close()


BACKENDS = [
    _Backend('in_memory', _make_in_memory),
    _Backend('in_memory_light_copy', _make_in_memory_light_copy),
    _Backend('database', _make_database),
    _Backend('sqlite', _make_sqlite),
    # One more Redis divergence has no contract test to hang an xfail on yet:
    # it builds its key scan pattern from a truthiness check on the user id, so
    # an empty one lists every user's sessions.
    _Backend(
        'redis',
        _make_redis,
        divergences={
            'test_list_sessions_ordered_by_last_update_time': (
                'Redis sorts sessions newest first, while the base class'
                ' documents oldest first.'
            ),
            'test_session_last_update_time_updates_on_event': (
                'Redis stamps the session with the wall clock instead of the'
                " appended event's timestamp."
            ),
            'test_append_event_to_unknown_session_raises_session_not_found': (
                'Redis writes the session key unconditionally on append, so'
                ' appending to a session it has never stored creates one'
                ' instead of raising.'
            ),
        },
    ),
    _Backend('per_agent_database', _make_per_agent_database),
]


@pytest.fixture(params=BACKENDS, ids=lambda backend: backend.name)
async def session_service(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> AsyncIterator[BaseSessionService]:
  """Yields each registered backend in turn, xfailing its known divergences."""
  backend: _Backend = request.param
  divergence = backend.divergences.get(request.node.originalname)
  if divergence is not None:
    request.node.add_marker(pytest.mark.xfail(strict=True, reason=divergence))
  async with backend.make(tmp_path) as service:
    yield service

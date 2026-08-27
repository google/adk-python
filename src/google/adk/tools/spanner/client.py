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

from contextlib import AbstractContextManager
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Protocol
from typing import Sequence

from google.auth.credentials import Credentials
from google.cloud import spanner
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect

from ... import version

USER_AGENT = f"adk-spanner-tool google-adk/{version.__version__}"


class _SpannerResultSet(Protocol):
  """Typed view of the row operations ADK uses from a Spanner result set."""

  def __iter__(self) -> Iterator[Sequence[object]]:
    ...

  def one(self) -> Sequence[object]:
    ...

  def to_dict_list(self) -> Sequence[dict[str, object]]:
    ...


class _SpannerSnapshot(Protocol):
  """Typed view of the Spanner snapshot operations used by ADK."""

  def execute_sql(
      self,
      sql: str,
      params: Mapping[str, object] | None = None,
      param_types: Mapping[str, object] | None = None,
  ) -> _SpannerResultSet:
    ...


class _SpannerOperation(Protocol):
  """Typed view of a synchronous long-running Spanner operation."""

  def result(self, timeout: float | None = None) -> object:
    ...


class _SpannerBatch(Protocol):
  """Typed view of the Spanner mutation batch used by the vector store."""

  def insert_or_update(
      self,
      *,
      table: str,
      columns: Sequence[str],
      values: Iterable[Sequence[object]],
  ) -> None:
    ...


class _SpannerTable(Protocol):
  """Typed view of table metadata returned by ``Database.list_tables``."""

  table_id: str


class _SpannerDatabase(Protocol):
  """Typed subset of ``google.cloud.spanner_v1.database.Database``."""

  database_dialect: DatabaseDialect

  def batch(self) -> AbstractContextManager[_SpannerBatch]:
    ...

  def exists(self) -> bool:
    ...

  def list_tables(self, *, schema: str) -> Iterable[_SpannerTable]:
    ...

  def reload(self) -> None:
    ...

  def snapshot(
      self, *, multi_use: bool = False
  ) -> AbstractContextManager[_SpannerSnapshot]:
    ...

  def update_ddl(self, ddl_statements: Sequence[str]) -> _SpannerOperation:
    ...


class _SpannerInstance(Protocol):
  """Typed subset of ``google.cloud.spanner_v1.instance.Instance``."""

  def database(
      self, database_id: str, *, database_role: str | None = None
  ) -> _SpannerDatabase:
    ...

  def exists(self) -> bool:
    ...


class _SpannerClientInfo(Protocol):
  """Client metadata used to compose ADK's user-agent string."""

  user_agent: str | None


class _SpannerClient(Protocol):
  """Typed subset of the legacy, unannotated Spanner data client."""

  _client_info: _SpannerClientInfo

  def instance(self, instance_id: str) -> _SpannerInstance:
    ...


def get_spanner_client(
    *, project: str, credentials: Credentials | None
) -> spanner.Client:
  """Get a Spanner client."""

  spanner_client = spanner.Client(project=project, credentials=credentials)
  spanner_client._client_info.user_agent = USER_AGENT
  return spanner_client


def _get_typed_spanner_client(
    *, project: str, credentials: Credentials | None
) -> _SpannerClient:
  """Get a Spanner client as the subset of operations ADK uses."""

  # The high-level Spanner client is not annotated, so contain that SDK gap at
  # construction and expose only the operations ADK uses through a protocol.
  return cast(
      _SpannerClient,
      get_spanner_client(project=project, credentials=credentials),
  )

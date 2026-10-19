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

from typing import Any
from typing import TYPE_CHECKING

from ... import version

if TYPE_CHECKING:
  from pymongo import MongoClient


def get_mongo_client(
    connection_string: str, timeout_ms: int | None = None
) -> MongoClient:
  """Creates a PyMongo client for the given connection string.

  PyMongo connects lazily, so the returned client does not perform any
  network I/O until the first operation.

  Args:
      connection_string: The MongoDB connection string (URI), e.g.
        "mongodb+srv://user:password@cluster.mongodb.net/".
      timeout_ms: Time limit in milliseconds for each operation run through
        this client. Unset leaves PyMongo's default, which is no limit.

  Returns:
      A PyMongo client configured with ADK driver metadata.

  Raises:
      ImportError: If the `pymongo` package is not installed.
  """
  try:
    from pymongo import MongoClient  # pylint: disable=import-outside-toplevel
    from pymongo.driver_info import DriverInfo  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError(
        "MongoDB tools require the 'pymongo' package. "
        "Please install it using `pip install google-adk[mongodb]`."
    ) from exc

  options: dict[str, Any] = {}
  if timeout_ms is not None:
    options["timeoutMS"] = timeout_ms
  return MongoClient(
      connection_string,
      driver=DriverInfo(name="adk-mongodb-tool", version=version.__version__),
      **options,
  )


def drop_client_for_pickle(
    state: dict[str, Any], *, owns_client: bool, owner: str
) -> dict[str, Any]:
  """Returns a picklable copy of an instance dict holding a MongoClient.

  A MongoClient wraps sockets, locks and background threads, none of which
  survive pickling — and pickling is how Agent Engine ships an app to the
  runtime (cloudpickle). The client is dropped here and rebuilt from the
  stored `_connection_string` when the object is restored on the destination.

  Args:
      state: The instance `__dict__` to make picklable.
      owns_client: Whether the instance created its client from
        `connection_string` (True) or received it via `mongo_client` (False).
      owner: Class name used in the error message.

  Returns:
      A copy of `state` with `_client` set to None.

  Raises:
      TypeError: If the client was caller-provided (`mongo_client=`), because
        it cannot be rebuilt on the destination.
  """
  if not owns_client:
    raise TypeError(
        f"{owner} cannot be pickled when constructed with `mongo_client`: a"
        " caller-owned client cannot be rebuilt on the destination. Construct"
        " it with `connection_string` when it must cross a pickle boundary"
        " (e.g. Agent Engine deployments)."
    )
  state = dict(state)
  state["_client"] = None
  return state

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

from typing import TYPE_CHECKING

from ... import version

if TYPE_CHECKING:
  from pymongo import MongoClient


def get_mongo_client(connection_string: str) -> MongoClient:
  """Creates a PyMongo client for the given connection string.

  PyMongo connects lazily, so the returned client does not perform any
  network I/O until the first operation.

  Args:
      connection_string: The MongoDB connection string (URI), e.g.
        "mongodb+srv://user:password@cluster.mongodb.net/".

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

  return MongoClient(
      connection_string,
      driver=DriverInfo(name="adk-mongodb-tool", version=version.__version__),
  )

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

"""Tests for the MongoDB client factory."""

import sys
from types import ModuleType
from unittest import mock

from google.adk.integrations.mongodb._client import get_mongo_client
import pytest


def test_get_mongo_client_raises_import_error_without_pymongo(monkeypatch):
  """get_mongo_client raises a helpful ImportError when pymongo is missing."""
  monkeypatch.setitem(sys.modules, "pymongo", None)

  with pytest.raises(ImportError, match=r"google-adk\[mongodb\]"):
    get_mongo_client("mongodb://localhost:27017")


@pytest.fixture(name="mongo_client_cls")
def _fake_pymongo(monkeypatch):
  """Installs a fake pymongo and returns its MongoClient mock."""
  fake_pymongo = ModuleType("pymongo")
  fake_driver_info = ModuleType("pymongo.driver_info")
  mongo_client_cls = mock.MagicMock()
  fake_pymongo.MongoClient = mongo_client_cls
  fake_driver_info.DriverInfo = mock.MagicMock()
  monkeypatch.setitem(sys.modules, "pymongo", fake_pymongo)
  monkeypatch.setitem(sys.modules, "pymongo.driver_info", fake_driver_info)
  return mongo_client_cls


def test_get_mongo_client_creates_client_with_driver_metadata(mongo_client_cls):
  """get_mongo_client builds a MongoClient from the connection string."""
  driver_info_cls = sys.modules["pymongo.driver_info"].DriverInfo

  result = get_mongo_client("mongodb://localhost:27017")

  assert result is mongo_client_cls.return_value
  mongo_client_cls.assert_called_once()
  assert mongo_client_cls.call_args.args[0] == "mongodb://localhost:27017"
  assert (
      mongo_client_cls.call_args.kwargs["driver"]
      is driver_info_cls.return_value
  )
  assert driver_info_cls.call_args.kwargs["name"] == "adk-mongodb-tool"


def test_get_mongo_client_applies_the_timeout(mongo_client_cls):
  """A timeout bounds every operation run through the client."""
  get_mongo_client("mongodb://localhost:27017", timeout_ms=5000)

  assert mongo_client_cls.call_args.kwargs["timeoutMS"] == 5000


def test_get_mongo_client_omits_an_unset_timeout(mongo_client_cls):
  """Without a timeout, PyMongo's own default is left in place."""
  get_mongo_client("mongodb://localhost:27017")

  assert "timeoutMS" not in mongo_client_cls.call_args.kwargs

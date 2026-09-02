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


def test_get_mongo_client_creates_client_with_driver_metadata(monkeypatch):
  """get_mongo_client builds a MongoClient from the connection string."""
  fake_pymongo = ModuleType("pymongo")
  fake_driver_info = ModuleType("pymongo.driver_info")
  mongo_client_cls = mock.MagicMock()
  driver_info_cls = mock.MagicMock()
  fake_pymongo.MongoClient = mongo_client_cls
  fake_driver_info.DriverInfo = driver_info_cls
  monkeypatch.setitem(sys.modules, "pymongo", fake_pymongo)
  monkeypatch.setitem(sys.modules, "pymongo.driver_info", fake_driver_info)

  result = get_mongo_client("mongodb://localhost:27017")

  assert result is mongo_client_cls.return_value
  mongo_client_cls.assert_called_once()
  assert mongo_client_cls.call_args.args[0] == "mongodb://localhost:27017"
  assert (
      mongo_client_cls.call_args.kwargs["driver"]
      is driver_info_cls.return_value
  )
  assert driver_info_cls.call_args.kwargs["name"] == "adk-mongodb-tool"

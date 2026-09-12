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

"""Tests for MongoDbToolset.

Verifies that the toolset exposes prefixed, filterable search tools and
injects the bound client, database and settings at run time.
"""

from unittest import mock

from google.adk.integrations.mongodb import MongoDbToolset
from google.adk.integrations.mongodb import MongoDbToolSettings
from google.adk.integrations.mongodb._mongodb_toolset import DEFAULT_MONGODB_TOOL_NAME_PREFIX
import pytest


def _make_toolset(**kwargs):
  return MongoDbToolset(
      database_name="test_db", mongo_client=mock.MagicMock(), **kwargs
  )


def test_mongodb_toolset_name_prefix():
  """MongoDbToolset prefixes its tool names with 'mongodb'."""
  toolset = _make_toolset()
  assert toolset.tool_name_prefix == DEFAULT_MONGODB_TOOL_NAME_PREFIX


async def test_mongodb_toolset_tools_default():
  """The default toolset exposes the vector and hybrid search tools."""
  toolset = _make_toolset()

  tools = await toolset.get_tools()

  assert set([tool.name for tool in tools]) == {
      "vector_search",
      "hybrid_search",
  }


async def test_mongodb_toolset_tools_prefixed():
  """Tools are returned with the 'mongodb' name prefix applied."""
  toolset = _make_toolset()

  tools = await toolset.get_tools_with_prefix()

  assert set([tool.name for tool in tools]) == {
      "mongodb_vector_search",
      "mongodb_hybrid_search",
  }


async def test_mongodb_toolset_tools_selective():
  """tool_filter restricts the exposed tools to the listed names."""
  toolset = _make_toolset(tool_filter=["vector_search"])

  tools = await toolset.get_tools()

  assert [tool.name for tool in tools] == ["vector_search"]


async def test_mongodb_toolset_unknown_tool_filtered_out():
  """Unknown names in tool_filter yield no tools."""
  toolset = _make_toolset(tool_filter=["unknown"])

  tools = await toolset.get_tools()

  assert tools == []


def test_mongodb_toolset_requires_client_or_connection_string():
  """Constructing without a client or connection string raises ValueError."""
  with pytest.raises(ValueError, match="must be provided"):
    MongoDbToolset(database_name="test_db")


def test_mongodb_toolset_rejects_client_and_connection_string():
  """Constructing with both a client and a connection string raises ValueError."""
  with pytest.raises(ValueError, match="Only one of"):
    MongoDbToolset(
        database_name="test_db",
        connection_string="mongodb://localhost:27017",
        mongo_client=mock.MagicMock(),
    )


async def test_mongodb_tool_injects_client_database_and_settings():
  """Running a tool injects the bound client, database and settings."""
  client = mock.MagicMock()
  client["test_db"]["test_coll"].aggregate.return_value = iter(
      [{"_id": 1, "title": "Doc"}]
  )
  toolset = MongoDbToolset(
      database_name="test_db",
      mongo_client=client,
      settings=MongoDbToolSettings(default_limit=9),
  )
  tools = await toolset.get_tools()
  tool = next(tool for tool in tools if tool.name == "vector_search")

  result = await tool.run_async(
      args={"collection_name": "test_coll", "query_embedding": [0.1]},
      tool_context=mock.MagicMock(),
  )

  assert result["status"] == "SUCCESS"
  assert result["rows"] == [{"_id": 1, "title": "Doc"}]
  pipeline = client["test_db"]["test_coll"].aggregate.call_args[0][0]
  assert pipeline[0]["$vectorSearch"]["queryVector"] == [0.1]
  # The custom settings flow through to the tool.
  assert pipeline[0]["$vectorSearch"]["limit"] == 9


async def test_mongodb_tool_declaration_hides_injected_parameters():
  """The generated function schema only exposes search parameters to the model."""
  toolset = _make_toolset()
  tools = await toolset.get_tools()

  declaration = next(
      tool for tool in tools if tool.name == "vector_search"
  )._get_declaration()

  properties = declaration.parameters_json_schema["properties"]
  assert "collection_name" in properties
  assert "query_embedding" in properties
  for injected in ("client", "database_name", "settings"):
    assert injected not in properties


async def test_close_does_not_close_injected_client():
  """close() leaves a caller-owned client open."""
  injected_client = mock.MagicMock()
  toolset = MongoDbToolset(
      database_name="test_db", mongo_client=injected_client
  )

  await toolset.close()

  injected_client.close.assert_not_called()


async def test_close_closes_client_created_from_connection_string(monkeypatch):
  """close() closes the client the toolset created from a connection string."""
  created_client = mock.MagicMock()
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._mongodb_toolset._client.get_mongo_client",
      lambda connection_string: created_client,
  )
  toolset = MongoDbToolset(
      database_name="test_db", connection_string="mongodb://localhost:27017"
  )

  await toolset.close()

  created_client.close.assert_called_once()

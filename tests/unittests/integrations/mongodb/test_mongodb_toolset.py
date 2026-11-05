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

import pickle
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
  with pytest.raises(ValueError, match="Exactly one of"):
    MongoDbToolset(
        database_name="test_db",
        connection_string="mongodb://localhost:27017",
        mongo_client=mock.MagicMock(),
    )


async def test_mongodb_tool_injects_client_database_and_settings(monkeypatch):
  """Running a tool injects the bound client, database and settings."""
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._search_tool._embed_query",
      mock.AsyncMock(return_value=[0.1]),
  )
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
      args={"collection_name": "test_coll", "query": "test query"},
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

  vector_declaration = next(
      tool for tool in tools if tool.name == "vector_search"
  )._get_declaration()

  properties = vector_declaration.parameters_json_schema["properties"]
  assert "collection_name" in properties
  assert "query" in properties
  assert "query_embedding" not in properties
  for injected in ("client", "database_name", "settings", "genai_client"):
    assert injected not in properties

  hybrid_declaration = next(
      tool for tool in tools if tool.name == "hybrid_search"
  )._get_declaration()
  hybrid_properties = hybrid_declaration.parameters_json_schema["properties"]
  assert "collection_name" in hybrid_properties
  assert "query" in hybrid_properties
  assert "text_search_field" in hybrid_properties
  assert "query_embedding" not in hybrid_properties
  for injected in ("client", "database_name", "settings", "genai_client"):
    assert injected not in hybrid_properties


async def test_mongodb_tool_detects_error_in_response():
  """_MongoDbTool detects status=ERROR as TOOL_ERROR for telemetry."""
  toolset = _make_toolset()
  tools = await toolset.get_tools()
  tool = tools[0]

  assert (
      tool._detect_error_in_response(
          {"status": "ERROR", "error_details": "failed"}
      )
      == "TOOL_ERROR"
  )
  assert (
      tool._detect_error_in_response({"status": "SUCCESS", "rows": []}) is None
  )
  assert tool._detect_error_in_response({"error": "fallback"}) == "TOOL_ERROR"
  assert tool._detect_error_in_response("plain string") is None
  assert tool._detect_error_in_response(None) is None


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
      lambda connection_string, timeout_ms: created_client,
  )
  toolset = MongoDbToolset(
      database_name="test_db", connection_string="mongodb://localhost:27017"
  )

  await toolset.close()

  created_client.close.assert_called_once()


def test_toolset_passes_the_timeout_to_the_client_it_owns(monkeypatch):
  """A client the toolset creates carries the configured timeout."""
  get_mongo_client = mock.MagicMock()
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._mongodb_toolset._client.get_mongo_client",
      get_mongo_client,
  )

  MongoDbToolset(
      database_name="test_db",
      connection_string="mongodb://localhost:27017",
      settings=MongoDbToolSettings(timeout_ms=7000),
  )

  assert get_mongo_client.call_args.kwargs["timeout_ms"] == 7000


async def test_mongodb_tool_injects_the_genai_client(monkeypatch):
  """A genai client given to the toolset reaches the search tool."""
  embed_query = mock.AsyncMock(return_value=[0.1])
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._search_tool._embed_query", embed_query
  )
  genai_client = mock.MagicMock()
  toolset = MongoDbToolset(
      database_name="test_db",
      mongo_client=mock.MagicMock(),
      genai_client=genai_client,
  )
  tools = await toolset.get_tools()
  tool = next(tool for tool in tools if tool.name == "vector_search")

  await tool.run_async(
      args={"collection_name": "test_coll", "query": "test query"},
      tool_context=mock.MagicMock(),
  )

  assert embed_query.await_args.kwargs["genai_client"] is genai_client


def test_toolset_pickles_when_client_comes_from_connection_string(monkeypatch):
  """A connection-string toolset drops and rebuilds its client on pickle."""
  get_mongo_client = mock.MagicMock(
      side_effect=lambda *args, **kwargs: mock.MagicMock()
  )
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._client.get_mongo_client",
      get_mongo_client,
  )
  toolset = MongoDbToolset(
      database_name="test_db",
      connection_string="mongodb://localhost:27017",
      settings=MongoDbToolSettings(timeout_ms=7000),
  )

  restored = pickle.loads(pickle.dumps(toolset))

  assert get_mongo_client.call_count == 2  # constructor + restore
  assert get_mongo_client.call_args.args == ("mongodb://localhost:27017",)
  assert get_mongo_client.call_args.kwargs["timeout_ms"] == 7000
  assert restored._database_name == "test_db"
  assert restored._owns_client is True


async def test_toolset_pickle_drops_cached_tools(monkeypatch):
  """Cached tool instances referencing the old client do not get pickled."""
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._client.get_mongo_client",
      lambda *args, **kwargs: mock.MagicMock(),
  )
  toolset = MongoDbToolset(
      database_name="test_db", connection_string="mongodb://localhost:27017"
  )
  await toolset.get_tools_with_prefix(mock.MagicMock(invocation_id="inv-1"))
  assert toolset._cached_invocation_id == "inv-1"
  assert toolset._cached_prefixed_tools is not None

  restored = pickle.loads(pickle.dumps(toolset))

  assert restored._cached_invocation_id is None
  assert restored._cached_prefixed_tools is None


def test_toolset_with_caller_owned_client_cannot_be_pickled():
  """A caller-owned client cannot be rebuilt on the destination."""
  with pytest.raises(TypeError, match="connection_string"):
    pickle.dumps(_make_toolset())


def test_toolset_with_caller_owned_genai_client_cannot_be_pickled(monkeypatch):
  """A caller-owned genai_client cannot be rebuilt on the destination."""
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._client.get_mongo_client",
      lambda *args, **kwargs: mock.MagicMock(),
  )
  toolset = MongoDbToolset(
      database_name="test_db",
      connection_string="mongodb://localhost:27017",
      genai_client=mock.MagicMock(),
  )
  with pytest.raises(TypeError, match="genai_client"):
    pickle.dumps(toolset)


def test_toolset_env_var_rejects_a_second_connection_source():
  """connection_string_env_var cannot be combined with another source."""
  with pytest.raises(ValueError, match="Exactly one of"):
    MongoDbToolset(
        database_name="test_db",
        connection_string="mongodb://localhost:27017",
        connection_string_env_var="MONGODB_URI",
    )
  with pytest.raises(ValueError, match="Exactly one of"):
    MongoDbToolset(
        database_name="test_db",
        connection_string_env_var="MONGODB_URI",
        mongo_client=mock.MagicMock(),
    )


def test_toolset_env_var_mode_resolves_the_connection_string(monkeypatch):
  """connection_string_env_var builds the client from the env var's value."""
  monkeypatch.setenv("ITEST_MONGODB_URI", "mongodb://localhost:27017")
  get_mongo_client = mock.MagicMock(
      side_effect=lambda *args, **kwargs: mock.MagicMock()
  )
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._client.get_mongo_client",
      get_mongo_client,
  )

  toolset = MongoDbToolset(
      database_name="test_db", connection_string_env_var="ITEST_MONGODB_URI"
  )

  assert get_mongo_client.call_args.args == ("mongodb://localhost:27017",)
  # The credential itself is never stored on the toolset...
  assert toolset._connection_string is None
  assert toolset._connection_string_env_var == "ITEST_MONGODB_URI"


def test_toolset_env_var_mode_missing_env_var_raises(monkeypatch):
  """A missing environment variable fails with its name in the error."""
  monkeypatch.delenv("ITEST_MONGODB_URI", raising=False)
  with pytest.raises(ValueError, match="ITEST_MONGODB_URI"):
    MongoDbToolset(
        database_name="test_db",
        connection_string_env_var="ITEST_MONGODB_URI",
    )


def test_toolset_env_var_mode_pickle_carries_no_credential(monkeypatch):
  """Pickling keeps only the env var name; the URI never enters the package."""
  uri = "mongodb://user:s3cret@localhost:27017"
  monkeypatch.setenv("ITEST_MONGODB_URI", uri)
  monkeypatch.setattr(
      "google.adk.integrations.mongodb._client.get_mongo_client",
      lambda *args, **kwargs: mock.MagicMock(),
  )
  toolset = MongoDbToolset(
      database_name="test_db", connection_string_env_var="ITEST_MONGODB_URI"
  )

  payload = pickle.dumps(toolset)

  assert uri.encode() not in payload
  assert b"s3cret" not in payload
  assert b"ITEST_MONGODB_URI" in payload
  # ...and restoring re-resolves from the environment on the destination.
  restored = pickle.loads(payload)
  assert restored._client is not None
  assert restored._connection_string is None

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

"""Tests for MongoDB search tools.

Verifies that vector_search and hybrid_search build the expected MongoDB
aggregation pipelines and return JSON-safe results.
"""

from unittest import mock

from google.adk.integrations.mongodb import _search_tool
from google.adk.integrations.mongodb import MongoDbToolSettings

_EMBEDDING = [0.1, 0.2, 0.3]


def _make_client(documents=None):
  """Returns a mock MongoClient whose aggregate() yields the given documents."""
  client = mock.MagicMock()
  client["test_db"]["test_coll"].aggregate.return_value = iter(documents or [])
  return client


def _aggregate_pipeline(client):
  """Returns the pipeline passed to aggregate() on the mock client."""
  return client["test_db"]["test_coll"].aggregate.call_args[0][0]


async def test_vector_search_uses_settings_defaults():
  """Vector search queries the collection with index, field and limits from settings."""
  client = _make_client()

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "SUCCESS", "rows": []}
  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"] == {
      "index": "vector_index",
      "path": "embedding",
      "queryVector": _EMBEDDING,
      "numCandidates": 100,
      "limit": 4,
  }


async def test_vector_search_applies_explicit_arguments():
  """Explicit index, field, filter and limits override the settings defaults."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      filter={"category": "kitchen"},
      limit=7,
      num_candidates=42,
      index_name="my_index",
      embedding_field="text_embedding",
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"] == {
      "index": "my_index",
      "path": "text_embedding",
      "queryVector": _EMBEDDING,
      "filter": {"category": "kitchen"},
      "numCandidates": 42,
      "limit": 7,
  }


async def test_vector_search_caps_limit_at_max_results():
  """A limit above settings.max_results is capped."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(max_results=10),
      limit=50,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["limit"] == 10


async def test_vector_search_raises_num_candidates_to_limit():
  """numCandidates below the limit is raised, as $vectorSearch requires it."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      limit=8,
      num_candidates=5,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["limit"] == 8
  assert pipeline[0]["$vectorSearch"]["numCandidates"] == 8


async def test_vector_search_excludes_embedding_field_from_results():
  """The default projection hides the raw embedding vector and adds the score."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[1] == {
      "$project": {
          "embedding": 0,
          "search_score": {"$meta": "vectorSearchScore"},
      }
  }


async def test_vector_search_projects_output_fields_when_given():
  """output_fields switches the projection to inclusion mode."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      output_fields=["title", "price"],
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[1] == {
      "$project": {
          "title": 1,
          "price": 1,
          "search_score": {"$meta": "vectorSearchScore"},
      }
  }


async def test_vector_search_returns_json_safe_rows():
  """Non-JSON-serializable values in result documents are converted to strings."""
  object_id = object()
  client = _make_client(
      [{"_id": object_id, "title": "Doc", "search_score": 0.9}]
  )

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result["status"] == "SUCCESS"
  assert result["rows"] == [
      {"_id": str(object_id), "title": "Doc", "search_score": 0.9}
  ]


async def test_vector_search_returns_error_on_failure():
  """A failing aggregation returns an ERROR result instead of raising."""
  client = _make_client()
  client["test_db"]["test_coll"].aggregate.side_effect = RuntimeError("boom")

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query_embedding=_EMBEDDING,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "ERROR", "error_details": "boom"}


async def test_hybrid_search_builds_rank_fusion_pipeline():
  """Hybrid search fuses vector and full-text rankings via $rankFusion."""
  client = _make_client()

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      query_embedding=_EMBEDDING,
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "SUCCESS", "rows": []}
  pipeline = _aggregate_pipeline(client)
  rank_fusion = pipeline[0]["$rankFusion"]
  pipelines = rank_fusion["input"]["pipelines"]
  assert pipelines["vector"] == [{
      "$vectorSearch": {
          "index": "vector_index",
          "path": "embedding",
          "queryVector": _EMBEDDING,
          "numCandidates": 100,
          "limit": 100,
      }
  }]
  assert pipelines["full_text"] == [
      {
          "$search": {
              "index": "default",
              "text": {"query": "cordless vacuum", "path": "description"},
          }
      },
      {"$limit": 100},
  ]
  assert rank_fusion["combination"]["weights"] == {
      "vector": 1.0,
      "full_text": 1.0,
  }
  assert rank_fusion["scoreDetails"] is False
  assert pipeline[1] == {"$limit": 4}
  assert pipeline[2] == {
      "$project": {"embedding": 0, "search_score": {"$meta": "score"}}
  }


async def test_hybrid_search_applies_weights_filter_and_index_names():
  """Explicit weights, filter and index names are applied to the pipeline."""
  client = _make_client()

  await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      query_embedding=_EMBEDDING,
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      filter={"in_stock": True},
      limit=5,
      num_candidates=25,
      vector_index_name="v_idx",
      search_index_name="s_idx",
      embedding_field="vec",
      vector_weight=2.0,
      text_weight=0.5,
  )

  pipeline = _aggregate_pipeline(client)
  rank_fusion = pipeline[0]["$rankFusion"]
  assert rank_fusion["input"]["pipelines"]["vector"] == [{
      "$vectorSearch": {
          "index": "v_idx",
          "path": "vec",
          "queryVector": _EMBEDDING,
          "filter": {"in_stock": True},
          "numCandidates": 25,
          "limit": 25,
      }
  }]
  full_text = rank_fusion["input"]["pipelines"]["full_text"]
  assert full_text[0]["$search"]["index"] == "s_idx"
  assert full_text[1] == {"$limit": 25}
  assert rank_fusion["combination"]["weights"] == {
      "vector": 2.0,
      "full_text": 0.5,
  }
  assert pipeline[1] == {"$limit": 5}


async def test_hybrid_search_returns_error_on_failure():
  """A failing aggregation returns an ERROR result instead of raising."""
  client = _make_client()
  client["test_db"]["test_coll"].aggregate.side_effect = RuntimeError("boom")

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      query_embedding=_EMBEDDING,
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "ERROR", "error_details": "boom"}

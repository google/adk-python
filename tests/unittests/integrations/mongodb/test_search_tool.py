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
import pytest

_QUERY = "test query"
_EMBEDDING = [0.1, 0.2, 0.3]
_real_embed_query = _search_tool._embed_query


@pytest.fixture(autouse=True)
def mock_embed_query(monkeypatch):
  mock_embed = mock.AsyncMock(return_value=_EMBEDDING)
  monkeypatch.setattr(_search_tool, "_embed_query", mock_embed)
  return mock_embed


def _make_client(documents=None):
  """Returns a mock MongoClient whose aggregate() yields the given documents."""
  client = mock.MagicMock()
  client["test_db"]["test_coll"].aggregate.return_value = iter(documents or [])
  return client


def _aggregate_pipeline(client):
  """Returns the pipeline passed to aggregate() on the mock client."""
  return client["test_db"]["test_coll"].aggregate.call_args[0][0]


def _aggregate_options(client):
  """Returns the keyword options passed to aggregate() on the mock client."""
  return client["test_db"]["test_coll"].aggregate.call_args.kwargs


def _assert_embedded(mock_embed, query, model_name, output_dimensionality=None):
  """Asserts the query was embedded with the given model configuration."""
  mock_embed.assert_awaited_once_with(
      query=query,
      model_name=model_name,
      output_dimensionality=output_dimensionality,
      genai_client=None,
  )


def _assert_projections_are_valid(pipeline):
  """Fails if a $project excludes a field and also includes or computes one.

  MongoDB rejects that combination, and a MagicMock collection will not.
  """
  for stage in pipeline:
    projection = stage.get("$project")
    if not projection:
      continue
    spec = {
        field: value for field, value in projection.items() if field != "_id"
    }
    excludes = any(value == 0 or value is False for value in spec.values())
    includes = any(value != 0 and value is not False for value in spec.values())
    assert not (
        excludes and includes
    ), f"$project mixes exclusion with inclusion or a computed field: {spec}"


async def test_vector_search_uses_settings_defaults(mock_embed_query):
  """Vector search queries the collection with index, field and limits from settings."""
  client = _make_client()

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "SUCCESS", "rows": []}
  _assert_embedded(mock_embed_query, _QUERY, "text-embedding-005")
  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"] == {
      "index": "vector_index",
      "path": "embedding",
      "queryVector": _EMBEDDING,
      "numCandidates": 100,
      "limit": 4,
  }


async def test_vector_search_applies_explicit_arguments(mock_embed_query):
  """Explicit index, field, filter and limits override the settings defaults."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(
          vertex_ai_embedding_model_name="custom-model",
          vertex_ai_embedding_output_dimensionality=256,
      ),
      filter={"category": "kitchen"},
      limit=7,
      num_candidates=42,
      index_name="my_index",
      embedding_field="text_embedding",
  )

  _assert_embedded(
      mock_embed_query, _QUERY, "custom-model", output_dimensionality=256
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
      query=_QUERY,
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
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      limit=8,
      num_candidates=5,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["limit"] == 8
  assert pipeline[0]["$vectorSearch"]["numCandidates"] == 8


@pytest.mark.parametrize("limit", [0, -3])
async def test_vector_search_replaces_non_positive_limit(limit):
  """A limit MongoDB would reject falls back to the settings default."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      limit=limit,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["limit"] == 4


async def test_vector_search_caps_num_candidates_at_mongodb_maximum():
  """numCandidates above the $vectorSearch maximum of 10000 is capped."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      num_candidates=999_999,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["numCandidates"] == 10000


async def test_vector_search_caps_derived_num_candidates():
  """A large max_results does not push the derived numCandidates over the cap."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(max_results=5000),
      limit=5000,
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[0]["$vectorSearch"]["numCandidates"] == 10000


@pytest.mark.parametrize(
    "unsafe_filter",
    [
        {"$where": "function() { return true; }"},
        {"$expr": {"$function": {"body": "f", "args": [], "lang": "js"}}},
        {"$and": [{"category": "kitchen"}, {"$where": "true"}]},
        {"nested": {"$expr": {"$eq": ["$a", "$b"]}}},
        # $regex is not a $vectorSearch filter operator, but $match accepts it,
        # so it would otherwise probe an unprojected field character by
        # character through the hybrid full-text arm.
        {"ssn": {"$regex": "^123-45"}},
        {"$or": [{"category": "kitchen"}, {"ssn": {"$regex": "^9"}}]},
        {"description": {"$text": {"$search": "x"}}},
        {"loc": {"$near": [0, 0]}},
    ],
)
async def test_vector_search_rejects_filter_operators_outside_allowlist(
    unsafe_filter,
):
  """Only the operators $vectorSearch.filter accepts reach the server."""
  client = _make_client()

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      filter=unsafe_filter,
  )

  assert result["status"] == "ERROR"
  assert "not allowed in a search filter" in result["error_details"]
  client["test_db"]["test_coll"].aggregate.assert_not_called()


@pytest.mark.parametrize(
    "unsafe_filter",
    [
        {"$where": "function() { return true; }"},
        {"ssn": {"$regex": "^123-45"}},
    ],
)
async def test_hybrid_search_rejects_filter_operators_outside_allowlist(
    unsafe_filter,
):
  """hybrid_search applies the same filter check before building a pipeline.

  The $match arm accepts all of MQL, so the check is what stops the two arms
  disagreeing on which filters they honor.
  """
  client = _make_client()

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      filter=unsafe_filter,
  )

  assert result["status"] == "ERROR"
  assert "not allowed in a search filter" in result["error_details"]
  client["test_db"]["test_coll"].aggregate.assert_not_called()


@pytest.mark.parametrize(
    "safe_filter",
    [
        {"category": "kitchen"},
        {"price": {"$gte": 10, "$lt": 100}},
        {"tags": {"$in": ["a", "b"]}, "archived": {"$exists": False}},
        {
            "$and": [
                {"a": {"$ne": 1}},
                {"$or": [{"b": 2}, {"c": {"$nin": [3]}}]},
            ]
        },
        {"d": {"$not": {"$gt": 5}}},
        {"$nor": [{"e": 1}]},
        {"address": {"city": "NY"}},
    ],
)
async def test_vector_search_accepts_allowlisted_filters(safe_filter):
  """Every operator $vectorSearch.filter supports is passed through as given."""
  client = _make_client()

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      filter=safe_filter,
  )

  assert result["status"] == "SUCCESS"
  assert (
      _aggregate_pipeline(client)[0]["$vectorSearch"]["filter"] == safe_filter
  )


async def test_vector_search_excludes_embedding_field_from_results():
  """The default projection hides the raw embedding vector and adds the score."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[1] == {
      "$addFields": {"search_score": {"$meta": "vectorSearchScore"}}
  }
  assert pipeline[2] == {"$project": {"embedding": 0}}
  _assert_projections_are_valid(pipeline)


async def test_vector_search_projects_output_fields_when_given():
  """output_fields switches the projection to inclusion mode."""
  client = _make_client()

  await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      output_fields=["title", "price"],
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[1] == {
      "$addFields": {"search_score": {"$meta": "vectorSearchScore"}}
  }
  assert pipeline[2] == {
      "$project": {"title": 1, "price": 1, "search_score": 1}
  }
  _assert_projections_are_valid(pipeline)


async def test_vector_search_returns_json_safe_rows():
  """Non-JSON-serializable values in result documents are converted to strings."""
  object_id = object()
  client = _make_client(
      [{"_id": object_id, "title": "Doc", "search_score": 0.9}]
  )

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
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
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "ERROR", "error_details": "boom"}


async def test_vector_search_returns_error_when_embedding_fails(
    mock_embed_query,
):
  """A failure during embedding returns an ERROR result instead of raising."""
  mock_embed_query.side_effect = RuntimeError("embedding failed")
  client = _make_client()

  result = await _search_tool.vector_search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {
      "status": "ERROR",
      "error_details": "embedding failed",
  }


async def test_hybrid_search_builds_rank_fusion_pipeline(mock_embed_query):
  """Hybrid search fuses vector and full-text rankings via $rankFusion."""
  client = _make_client()

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "SUCCESS", "rows": []}
  _assert_embedded(mock_embed_query, "cordless vacuum", "text-embedding-005")
  pipeline = _aggregate_pipeline(client)
  rank_fusion = pipeline[0]["$rankFusion"]
  pipelines = rank_fusion["input"]["pipelines"]
  # Each arm returns 5x the final limit of 4, so rank fusion has something to
  # promote from; numCandidates is untouched at the settings default.
  assert pipelines["vector"] == [{
      "$vectorSearch": {
          "index": "vector_index",
          "path": "embedding",
          "queryVector": _EMBEDDING,
          "numCandidates": 100,
          "limit": 20,
      }
  }]
  assert pipelines["full_text"] == [
      {
          "$search": {
              "index": "default",
              "text": {"query": "cordless vacuum", "path": "description"},
          }
      },
      {"$limit": 20},
  ]
  assert rank_fusion["combination"]["weights"] == {
      "vector": 1.0,
      "full_text": 1.0,
  }
  assert rank_fusion["scoreDetails"] is False
  assert pipeline[1] == {"$limit": 4}
  assert pipeline[2] == {"$addFields": {"search_score": {"$meta": "score"}}}
  assert pipeline[3] == {"$project": {"embedding": 0}}
  _assert_projections_are_valid(pipeline)


async def test_hybrid_search_applies_weights_filter_and_index_names():
  """Explicit weights, filter and index names are applied to the pipeline."""
  client = _make_client()

  await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
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
  # 5x the final limit of 5 is above num_candidates, so the arms stop at 25:
  # $vectorSearch rejects a limit above numCandidates.
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
  assert full_text[1] == {"$match": {"in_stock": True}}
  assert full_text[2] == {"$limit": 25}
  assert rank_fusion["combination"]["weights"] == {
      "vector": 2.0,
      "full_text": 0.5,
  }
  assert pipeline[1] == {"$limit": 5}


async def test_hybrid_search_projects_output_fields_when_given():
  """output_fields switches the shared projection stages to inclusion mode."""
  client = _make_client()

  await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      output_fields=["title", "price"],
  )

  pipeline = _aggregate_pipeline(client)
  assert pipeline[2] == {"$addFields": {"search_score": {"$meta": "score"}}}
  assert pipeline[3] == {
      "$project": {"title": 1, "price": 1, "search_score": 1}
  }
  _assert_projections_are_valid(pipeline)


@pytest.mark.parametrize(
    "weight,expected",
    [(-1.0, 0.0), (0.0, 0.0), (3, 3.0)],
)
async def test_hybrid_search_clamps_negative_weights(weight, expected):
  """A weight $rankFusion would reject is clamped rather than passed through."""
  client = _make_client()

  await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
      vector_weight=weight,
      text_weight=weight,
  )

  weights = _aggregate_pipeline(client)[0]["$rankFusion"]["combination"][
      "weights"
  ]
  assert weights == {"vector": expected, "full_text": expected}


@pytest.mark.parametrize(
    "search,extra_kwargs",
    [
        (_search_tool.vector_search, {}),
        (_search_tool.hybrid_search, {"text_search_field": "description"}),
    ],
)
async def test_search_bounds_the_aggregate_with_a_time_limit(
    search, extra_kwargs
):
  """Every aggregate carries maxTimeMS, since to_thread cannot be cancelled."""
  client = _make_client()

  await search(
      collection_name="test_coll",
      query=_QUERY,
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(timeout_ms=1234),
      **extra_kwargs,
  )

  assert _aggregate_options(client)["maxTimeMS"] == 1234


async def test_hybrid_search_returns_error_on_failure():
  """A failing aggregation returns an ERROR result instead of raising."""
  client = _make_client()
  client["test_db"]["test_coll"].aggregate.side_effect = RuntimeError("boom")

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {"status": "ERROR", "error_details": "boom"}


async def test_hybrid_search_returns_error_when_embedding_fails(
    mock_embed_query,
):
  """A failure during embedding in hybrid search returns an ERROR result."""
  mock_embed_query.side_effect = RuntimeError("embedding failed")
  client = _make_client()

  result = await _search_tool.hybrid_search(
      collection_name="test_coll",
      query="cordless vacuum",
      text_search_field="description",
      client=client,
      database_name="test_db",
      settings=MongoDbToolSettings(),
  )

  assert result == {
      "status": "ERROR",
      "error_details": "embedding failed",
  }


def _make_genai_client(values=(0.1, 0.2, 0.3)):
  """Returns a mock genai client whose embed_content returns `values`."""
  genai_client = mock.MagicMock()
  embedding = mock.MagicMock(values=list(values))
  genai_client.aio.models.embed_content = mock.AsyncMock(
      return_value=mock.MagicMock(embeddings=[embedding])
  )
  return genai_client


async def test_embed_query_uses_the_given_genai_client():
  """An injected client is used instead of one built from the environment."""
  genai_client = _make_genai_client()

  result = await _real_embed_query(
      "search text", "text-embedding-005", genai_client=genai_client
  )

  assert result == [0.1, 0.2, 0.3]
  genai_client.aio.models.embed_content.assert_awaited_once()
  call = genai_client.aio.models.embed_content.call_args.kwargs
  assert call["model"] == "text-embedding-005"
  assert call["contents"] == ["search text"]
  assert call["config"].output_dimensionality is None


async def test_embed_query_passes_output_dimensionality():
  """The configured dimensionality reaches the embedding call."""
  genai_client = _make_genai_client()

  await _real_embed_query(
      "search text",
      "text-embedding-005",
      output_dimensionality=256,
      genai_client=genai_client,
  )

  config = genai_client.aio.models.embed_content.call_args.kwargs["config"]
  assert config.output_dimensionality == 256


async def test_embed_query_defaults_to_an_ambient_client(monkeypatch):
  """Without an injected client, _embed_query builds a google.genai Client."""
  genai_client = _make_genai_client()
  client_cls = mock.MagicMock(return_value=genai_client)
  monkeypatch.setattr("google.genai.Client", client_cls)

  result = await _real_embed_query("search text", "text-embedding-005")

  assert result == [0.1, 0.2, 0.3]
  client_cls.assert_called_once_with()


async def test_embed_query_failure():
  """_embed_query wraps embedding call failures in RuntimeError."""
  genai_client = mock.MagicMock()
  genai_client.aio.models.embed_content = mock.AsyncMock(
      side_effect=ValueError("quota exceeded")
  )

  with pytest.raises(RuntimeError, match="Failed to embed query"):
    await _real_embed_query(
        "search text", "text-embedding-005", genai_client=genai_client
    )


async def test_embed_query_reports_an_empty_embedding_as_itself():
  """An empty response raises on its own terms, not as a wrapped failure."""
  genai_client = _make_genai_client(values=[])

  with pytest.raises(ValueError, match="returned no embedding"):
    await _real_embed_query(
        "search text", "text-embedding-005", genai_client=genai_client
    )

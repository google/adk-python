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

"""Tools to run vector and hybrid search against MongoDB collections."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types as genai_types

from ._settings import MongoDbToolSettings

if TYPE_CHECKING:
  from google.genai import Client

logger = logging.getLogger("google_adk." + __name__)

_SEARCH_SCORE_ALIAS = "search_score"
_VECTOR_PIPELINE_NAME = "vector"
_FULL_TEXT_PIPELINE_NAME = "full_text"

# $vectorSearch rejects a numCandidates above this.
_MAX_NUM_CANDIDATES = 10000

# How many documents each $rankFusion arm returns, as a multiple of the final
# limit. Rank fusion can only rank what the arms hand it, so an arm that stops
# at the final limit hides every document the other arm would have promoted.
_RANK_FUSION_ARM_OVERSAMPLE = 5

# The only operators allowed in a caller-supplied `filter`: the set
# $vectorSearch.filter accepts. The same dict also reaches $match in
# hybrid_search, which accepts all of MQL, so an allowlist is what keeps the
# two arms agreeing on what a filter may express. A denylist does not: it
# leaves the $match arm richer than the vector arm, so the same filter would
# mean one thing to vector_search and another to hybrid_search, and it lets
# through $where and $expr, which run server-side expressions the vector arm
# cannot express at all.
_ALLOWED_FILTER_OPERATORS = frozenset([
    "$eq",
    "$ne",
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$in",
    "$nin",
    "$exists",
    "$and",
    "$or",
    "$not",
    "$nor",
])


def _json_safe(value: Any) -> Any:
  """Returns the value unchanged if JSON-serializable, else its string form."""
  try:
    json.dumps(value)
    return value
  except (TypeError, ValueError, OverflowError):
    return str(value)


def _check_filter(filter: Any) -> None:
  """Raises ValueError if the filter uses an operator outside the allowlist.

  The filter comes from the model, so every nesting level is checked, not just
  the top. Field names cannot start with `$` in a query predicate, so any
  `$`-prefixed key here is an operator.
  """
  if isinstance(filter, list):
    for item in filter:
      _check_filter(item)
    return
  if not isinstance(filter, dict):
    return
  for key, value in filter.items():
    if key.startswith("$") and key not in _ALLOWED_FILTER_OPERATORS:
      raise ValueError(f"Operator {key} is not allowed in a search filter.")
    _check_filter(value)


def _resolve_limits(
    limit: int | None, num_candidates: int | None, settings: MongoDbToolSettings
) -> tuple[int, int]:
  """Resolves the result limit and the candidate count for a search operation.

  Both arguments reach us from the model, so they are clamped to the range
  MongoDB accepts instead of being passed through into a pipeline the server
  would reject.
  """
  # A limit MongoDB would reject is treated as if it had not been given.
  resolved_limit = min(
      limit if limit and limit > 0 else settings.default_limit,
      settings.max_results,
      _MAX_NUM_CANDIDATES,
  )
  if num_candidates is None:
    resolved_num_candidates = max(
        resolved_limit * 10, settings.default_num_candidates
    )
  else:
    resolved_num_candidates = num_candidates
  # $vectorSearch requires numCandidates to be at least the limit, and rejects
  # anything above _MAX_NUM_CANDIDATES.
  resolved_num_candidates = min(
      max(resolved_num_candidates, resolved_limit), _MAX_NUM_CANDIDATES
  )
  return resolved_limit, resolved_num_candidates


def _resolve_weight(weight: float | None) -> float:
  """Resolves a $rankFusion combination weight.

  The weight reaches us from the model, and $rankFusion rejects a negative
  one, so it is clamped for the same reason the limits are.
  """
  if weight is None:
    return 1.0
  return max(float(weight), 0.0)


def _build_result_stages(
    embedding_field: str, output_fields: list[str] | None, score_meta: str
) -> list[dict[str, Any]]:
  """Builds the stages that shape search results.

  The raw embedding vector is excluded by default to keep results compact;
  callers can opt into exact fields via `output_fields`. The search score is
  always added under the `search_score` field.

  The score gets its own `$addFields` stage because `$project` rejects a
  computed field alongside an exclusion, which is what the default case needs.
  """
  stages: list[dict[str, Any]] = [
      {"$addFields": {_SEARCH_SCORE_ALIAS: {"$meta": score_meta}}}
  ]
  if output_fields:
    projection: dict[str, Any] = {field: 1 for field in output_fields}
    projection[_SEARCH_SCORE_ALIAS] = 1
  else:
    projection = {embedding_field: 0}
  stages.append({"$project": projection})
  return stages


def _aggregate_documents(
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    collection_name: str,
    pipeline: list[dict[str, Any]],
    timeout_ms: int,
) -> dict[str, Any]:
  """Runs an aggregation pipeline and returns JSON-safe rows.

  `maxTimeMS` bounds the call on the server. Callers run this through
  `asyncio.to_thread`, which cannot be cancelled, so an unbounded aggregate
  would hold a worker of the default executor for as long as the server takes.
  """
  cursor = client[database_name][collection_name].aggregate(
      pipeline, maxTimeMS=timeout_ms
  )
  rows = [
      {key: _json_safe(value) for key, value in document.items()}
      for document in cursor
  ]
  return {"status": "SUCCESS", "rows": rows}


async def _embed_query(
    query: str,
    model_name: str,
    output_dimensionality: int | None = None,
    genai_client: Client | None = None,
) -> list[float]:
  """Embeds query text into a vector using the Vertex AI embedding model."""
  # google.genai is imported lazily here, and only `types` at module level:
  # CheckGoogleGenaiLazyImport keeps the rest off the import path.
  from google.genai import Client  # pylint: disable=import-outside-toplevel

  config = genai_types.EmbedContentConfig()
  if output_dimensionality:
    config.output_dimensionality = output_dimensionality
  contents: list[genai_types.PartUnion] = [query]
  try:
    response = await (genai_client or Client()).aio.models.embed_content(
        model=model_name,
        contents=contents,
        config=config,
    )
  except Exception as ex:
    raise RuntimeError(f"Failed to embed query: {ex!r}") from ex
  values = response.embeddings[0].values if response.embeddings else None
  if not values:
    raise ValueError(f"Model {model_name} returned no embedding.")
  return [float(v) for v in values]


async def vector_search(
    collection_name: str,
    query: str,
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    settings: MongoDbToolSettings,
    # google.genai.Client; kept as Any because the tool declaration resolves
    # annotations at run time and Client is only imported lazily.
    genai_client: Any = None,
    filter: dict[str, Any] | None = None,
    limit: int | None = None,
    num_candidates: int | None = None,
    index_name: str | None = None,
    embedding_field: str | None = None,
    output_fields: list[str] | None = None,
) -> dict[str, Any]:
  """Runs an Atlas Vector Search query against a MongoDB collection.

  Finds documents whose embedding is most similar to the query text using the
  `$vectorSearch` aggregation stage. Requires a vector search index on the
  collection, and a deployment that supports `$vectorSearch` (MongoDB Atlas
  6.0.11+/7.0.2+, or self-managed MongoDB 8.2+).

  Args:
      collection_name (str): The name of the collection to search.
      query (str): The search query text to find similar documents for.
      filter (dict): An optional MongoDB query filter to pre-filter documents
        before searching, e.g. {"category": "kitchen"}. Only fields indexed as
        filter fields in the vector search index can be used, and only the
        operators $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $and,
        $or, $not and $nor are accepted.
      limit (int): The maximum number of documents to return. Capped by the
        toolset settings.
      num_candidates (int): The number of nearest neighbors to consider during
        the search. Higher values improve recall at the cost of latency.
      index_name (str): The name of the vector search index to query. Defaults
        to the toolset settings.
      embedding_field (str): The document field that stores the embedding
        vectors. Defaults to the toolset settings.
      output_fields (list[str]): The document fields to return in the results.
        By default all fields except the embedding vector are returned. The
        `_id` and the `search_score` are always included.

  Returns:
      dict: A dictionary with the search results.
        On success: {"status": "SUCCESS", "rows": [...]}, where each row is a
        matching document with a "search_score" field.
        On error: {"status": "ERROR", "error_details": "..."}.

  Examples:
      Find the two products most similar to a query, restricted to
      a category:
        >>> await vector_search(
        ...   collection_name="products",
        ...   query="cordless vacuum cleaner",
        ...   filter={"category": "kitchen"},
        ...   limit=2,
        ... )
        {
          "status": "SUCCESS",
          "rows": [
            {"_id": "...", "name": "Robot Vacuum", "search_score": 0.93},
            {"_id": "...", "name": "Steam Mop", "search_score": 0.88},
          ],
        }
  """
  try:
    _check_filter(filter)
    query_embedding = await _embed_query(
        query=query,
        model_name=settings.vertex_ai_embedding_model_name,
        output_dimensionality=(
            settings.vertex_ai_embedding_output_dimensionality
        ),
        genai_client=genai_client,
    )
    resolved_index_name = index_name or settings.default_vector_index_name
    resolved_embedding_field = (
        embedding_field or settings.default_embedding_field
    )
    resolved_limit, resolved_num_candidates = _resolve_limits(
        limit, num_candidates, settings
    )

    vector_search_stage: dict[str, Any] = {
        "index": resolved_index_name,
        "path": resolved_embedding_field,
        "queryVector": query_embedding,
        "numCandidates": resolved_num_candidates,
        "limit": resolved_limit,
    }
    if filter:
      vector_search_stage["filter"] = filter

    pipeline: list[dict[str, Any]] = [
        {"$vectorSearch": vector_search_stage},
        *_build_result_stages(
            resolved_embedding_field, output_fields, "vectorSearchScore"
        ),
    ]

    return await asyncio.to_thread(
        _aggregate_documents,
        client,
        database_name,
        collection_name,
        pipeline,
        settings.timeout_ms,
    )
  except Exception as ex:
    logger.exception("MongoDB vector search failed")
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


async def hybrid_search(
    collection_name: str,
    query: str,
    text_search_field: str,
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    settings: MongoDbToolSettings,
    # google.genai.Client; kept as Any because the tool declaration resolves
    # annotations at run time and Client is only imported lazily.
    genai_client: Any = None,
    filter: dict[str, Any] | None = None,
    limit: int | None = None,
    num_candidates: int | None = None,
    vector_index_name: str | None = None,
    search_index_name: str | None = None,
    embedding_field: str | None = None,
    vector_weight: float | None = None,
    text_weight: float | None = None,
    output_fields: list[str] | None = None,
) -> dict[str, Any]:
  """Runs a hybrid (full-text + vector) search against a MongoDB collection.

  Combines full-text search and vector search with reciprocal rank fusion
  using the `$rankFusion` aggregation stage, so documents matching either the
  text query or the embedding similarity are ranked together. Requires a
  full-text search index and a vector search index on the collection, and a
  deployment that supports both `$search`/`$vectorSearch` and `$rankFusion`
  (MongoDB Atlas 8.0+).

  Args:
      collection_name (str): The name of the collection to search.
      query (str): The text query for full-text search and vector search.
      text_search_field (str): The document field to run the full-text search
        against.
      filter (dict): An optional MongoDB query filter to filter documents,
        e.g. {"category": "kitchen"}. In vector search, only fields indexed as
        filter fields in the vector search index can be used. Both arms accept
        only the operators $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists,
        $and, $or, $not and $nor.
      limit (int): The maximum number of documents to return. Capped by the
        toolset settings.
      num_candidates (int): The number of nearest neighbors to consider during
        the vector search. Higher values improve recall at the cost of
        latency.
      vector_index_name (str): The name of the vector search index to query.
        Defaults to the toolset settings.
      search_index_name (str): The name of the full-text search index to
        query. Defaults to the toolset settings.
      embedding_field (str): The document field that stores the embedding
        vectors. Defaults to the toolset settings.
      vector_weight (float): The weight of the vector search ranking in the
        fused score. Defaults to 1.0. Must not be negative.
      text_weight (float): The weight of the full-text search ranking in the
        fused score. Defaults to 1.0. Must not be negative.
      output_fields (list[str]): The document fields to return in the results.
        By default all fields except the embedding vector are returned. The
        `_id` and the `search_score` are always included.

  Returns:
      dict: A dictionary with the search results.
        On success: {"status": "SUCCESS", "rows": [...]}, where each row is a
        matching document with a "search_score" field holding the fused
        reciprocal rank fusion score.
        On error: {"status": "ERROR", "error_details": "..."}.

  Examples:
      Find products relevant to "cordless vacuum for pet hair", weighing
      vector matches twice as much as text matches:
        >>> await hybrid_search(
        ...   collection_name="products",
        ...   query="cordless vacuum for pet hair",
        ...   text_search_field="description",
        ...   vector_weight=2.0,
        ...   limit=3,
        ... )
        {
          "status": "SUCCESS",
          "rows": [
            {"_id": "...", "name": "Pet Hair Vacuum", "search_score": 0.032},
            ...
          ],
        }
  """
  try:
    _check_filter(filter)
    query_embedding = await _embed_query(
        query=query,
        model_name=settings.vertex_ai_embedding_model_name,
        output_dimensionality=(
            settings.vertex_ai_embedding_output_dimensionality
        ),
        genai_client=genai_client,
    )
    resolved_vector_index_name = (
        vector_index_name or settings.default_vector_index_name
    )
    resolved_search_index_name = (
        search_index_name or settings.default_search_index_name
    )
    resolved_embedding_field = (
        embedding_field or settings.default_embedding_field
    )
    resolved_limit, resolved_num_candidates = _resolve_limits(
        limit, num_candidates, settings
    )
    # Each arm feeds $rankFusion a ranked list this long, and both arms use the
    # same length so neither is favored by depth alone. It stays above the
    # final limit, because a document the fusion should promote has to be in
    # the arm's list to be promoted, and below numCandidates, which is the
    # $vectorSearch ceiling on how much the ANN search actually looked at.
    arm_limit = min(
        resolved_limit * _RANK_FUSION_ARM_OVERSAMPLE, resolved_num_candidates
    )

    vector_search_stage: dict[str, Any] = {
        "index": resolved_vector_index_name,
        "path": resolved_embedding_field,
        "queryVector": query_embedding,
        "numCandidates": resolved_num_candidates,
        "limit": arm_limit,
    }
    if filter:
      vector_search_stage["filter"] = filter

    full_text_pipeline: list[dict[str, Any]] = [
        {
            "$search": {
                "index": resolved_search_index_name,
                "text": {
                    "query": query,
                    "path": text_search_field,
                },
            }
        },
    ]
    if filter:
      full_text_pipeline.append({"$match": filter})
    full_text_pipeline.append({"$limit": arm_limit})

    pipeline: list[dict[str, Any]] = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        _VECTOR_PIPELINE_NAME: [
                            {"$vectorSearch": vector_search_stage}
                        ],
                        _FULL_TEXT_PIPELINE_NAME: full_text_pipeline,
                    }
                },
                "combination": {
                    "weights": {
                        _VECTOR_PIPELINE_NAME: _resolve_weight(vector_weight),
                        _FULL_TEXT_PIPELINE_NAME: _resolve_weight(text_weight),
                    }
                },
                "scoreDetails": False,
            }
        },
        {"$limit": resolved_limit},
        *_build_result_stages(resolved_embedding_field, output_fields, "score"),
    ]

    return await asyncio.to_thread(
        _aggregate_documents,
        client,
        database_name,
        collection_name,
        pipeline,
        settings.timeout_ms,
    )
  except Exception as ex:
    logger.exception("MongoDB hybrid search failed")
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

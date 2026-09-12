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

"""Tools to run vector and hybrid search against MongoDB collections."""

import asyncio
import json
import logging
from typing import Any

from ._settings import MongoDbToolSettings

logger = logging.getLogger("google_adk." + __name__)

_SEARCH_SCORE_ALIAS = "search_score"
_VECTOR_PIPELINE_NAME = "vector"
_FULL_TEXT_PIPELINE_NAME = "full_text"


def _json_safe(value: Any) -> Any:
  """Returns the value unchanged if JSON-serializable, else its string form."""
  try:
    json.dumps(value)
    return value
  except (TypeError, ValueError, OverflowError):
    return str(value)


def _resolve_limits(
    limit: int | None, num_candidates: int | None, settings: MongoDbToolSettings
) -> tuple[int, int]:
  """Resolves the result limit and the candidate count for a search operation."""
  resolved_limit = min(limit or settings.default_limit, settings.max_results)
  if num_candidates is None:
    resolved_num_candidates = max(
        resolved_limit * 10, settings.default_num_candidates
    )
  else:
    # $vectorSearch requires numCandidates to be at least the limit.
    resolved_num_candidates = max(num_candidates, resolved_limit)
  return resolved_limit, resolved_num_candidates


def _build_result_projection(
    embedding_field: str, output_fields: list[str] | None, score_meta: str
) -> dict[str, Any]:
  """Builds the projection stage applied to search results.

  The raw embedding vector is excluded by default to keep results compact;
  callers can opt into exact fields via `output_fields`. The search score is
  always added under the `search_score` field.
  """
  if output_fields:
    projection: dict[str, Any] = {field: 1 for field in output_fields}
  else:
    projection = {embedding_field: 0}
  projection[_SEARCH_SCORE_ALIAS] = {"$meta": score_meta}
  return {"$project": projection}


def _aggregate_documents(
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    collection_name: str,
    pipeline: list[dict[str, Any]],
) -> dict[str, Any]:
  """Runs an aggregation pipeline and returns JSON-safe rows."""
  cursor = client[database_name][collection_name].aggregate(pipeline)
  rows = [
      {key: _json_safe(value) for key, value in document.items()}
      for document in cursor
  ]
  return {"status": "SUCCESS", "rows": rows}


async def vector_search(
    collection_name: str,
    query_embedding: list[float],
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    settings: MongoDbToolSettings,
    filter: dict[str, Any] | None = None,
    limit: int | None = None,
    num_candidates: int | None = None,
    index_name: str | None = None,
    embedding_field: str | None = None,
    output_fields: list[str] | None = None,
) -> dict[str, Any]:
  """Runs an Atlas Vector Search query against a MongoDB collection.

  Finds documents whose embedding is most similar to `query_embedding` using
  the `$vectorSearch` aggregation stage. Requires a vector search index on the
  collection (available on MongoDB Atlas and MongoDB 8.0+).

  Args:
      collection_name (str): The name of the collection to search.
      query_embedding (list[float]): The embedding vector of the query, e.g.
        produced by an embedding model for the user's text.
      filter (dict): An optional MongoDB query filter to pre-filter documents
        before searching, e.g. {"category": "kitchen"}. Only fields indexed as
        filter fields in the vector search index can be used.
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
      Find the two products most similar to a query embedding, restricted to
      a category:
        >>> await vector_search(
        ...   collection_name="products",
        ...   query_embedding=[0.12, -0.03, ...],
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

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        _build_result_projection(
            resolved_embedding_field, output_fields, "vectorSearchScore"
        ),
    ]

    return await asyncio.to_thread(
        _aggregate_documents, client, database_name, collection_name, pipeline
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
    query_embedding: list[float],
    text_search_field: str,
    client: Any,  # pymongo.MongoClient; kept as Any so pymongo stays optional.
    database_name: str,
    settings: MongoDbToolSettings,
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
  deployment that supports `$rankFusion` (MongoDB 8.0+, or MongoDB Atlas).

  Args:
      collection_name (str): The name of the collection to search.
      query (str): The text query for full-text search.
      query_embedding (list[float]): The embedding vector of the query, e.g.
        produced by an embedding model for the user's text.
      text_search_field (str): The document field to run the full-text search
        against.
      filter (dict): An optional MongoDB query filter to pre-filter documents
        before the vector search, e.g. {"category": "kitchen"}. Only fields
        indexed as filter fields in the vector search index can be used.
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
        fused score. Defaults to 1.0.
      text_weight (float): The weight of the full-text search ranking in the
        fused score. Defaults to 1.0.
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
        ...   query_embedding=[0.12, -0.03, ...],
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

    vector_search_stage: dict[str, Any] = {
        "index": resolved_vector_index_name,
        "path": resolved_embedding_field,
        "queryVector": query_embedding,
        "numCandidates": resolved_num_candidates,
        "limit": resolved_num_candidates,
    }
    if filter:
      vector_search_stage["filter"] = filter

    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        _VECTOR_PIPELINE_NAME: [
                            {"$vectorSearch": vector_search_stage}
                        ],
                        _FULL_TEXT_PIPELINE_NAME: [
                            {
                                "$search": {
                                    "index": resolved_search_index_name,
                                    "text": {
                                        "query": query,
                                        "path": text_search_field,
                                    },
                                }
                            },
                            {"$limit": resolved_num_candidates},
                        ],
                    }
                },
                "combination": {
                    "weights": {
                        _VECTOR_PIPELINE_NAME: (
                            vector_weight if vector_weight is not None else 1.0
                        ),
                        _FULL_TEXT_PIPELINE_NAME: (
                            text_weight if text_weight is not None else 1.0
                        ),
                    }
                },
                "scoreDetails": False,
            }
        },
        {"$limit": resolved_limit},
        _build_result_projection(
            resolved_embedding_field, output_fields, "score"
        ),
    ]

    return await asyncio.to_thread(
        _aggregate_documents, client, database_name, collection_name, pipeline
    )
  except Exception as ex:
    logger.exception("MongoDB hybrid search failed")
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

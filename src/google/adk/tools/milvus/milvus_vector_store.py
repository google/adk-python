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

"""Milvus vector store utility class for data ingestion and retrieval."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable
from typing import Optional

from ...features import experimental
from ...features import FeatureName
from .settings import MilvusToolSettings

logger = logging.getLogger("google_adk." + __name__)

try:
  from pymilvus import DataType
  from pymilvus import MilvusClient
except ImportError:
  MilvusClient = None
  DataType = None


@experimental(FeatureName.MILVUS_VECTOR_STORE)
class MilvusVectorStore:
  """A utility class for managing a Milvus vector store.

  This class provides methods for setting up a Milvus collection,
  adding content with embeddings, and performing similarity search.
  """

  def __init__(
      self,
      settings: MilvusToolSettings,
      embedding_fn: Callable[[list[str]], list[list[float]]],
  ):
    """Initializes the MilvusVectorStore with settings and embedding function.

    Args:
      settings: The Milvus tool settings containing vector store configuration.
      embedding_fn: A function that takes a list of texts and returns a list
        of embedding vectors. Signature: ``(list[str]) -> list[list[float]]``.
        For example, using Google GenAI::

          from google.genai import Client
          client = Client()
          def embedding_fn(texts):
              resp = client.models.embed_content(
                  model="text-embedding-004", contents=texts)
              return [list(e.values) for e in resp.embeddings]

    Raises:
      ValueError: If vector_store_settings is not set.
      ImportError: If pymilvus is not installed.
    """
    if not settings.vector_store_settings:
      raise ValueError("Milvus vector store settings are not set.")

    if MilvusClient is None:
      raise ImportError(
          "pymilvus package not found. "
          'Please install with: pip install "google-adk[milvus]"'
      )

    self._settings = settings.vector_store_settings
    self._embedding_fn = embedding_fn

    self._client = MilvusClient(
        uri=self._settings.uri,
        token=self._settings.token,
        db_name=self._settings.db_name,
    )

  def setup(self) -> None:
    """Creates the Milvus collection and index if they do not exist.

    The collection schema includes:
    - A primary key field (auto-generated int64 id).
    - A text content field (VARCHAR).
    - A vector embedding field (FLOAT_VECTOR).

    The vector index is created with the configured metric type and
    index type.
    """
    if self._client.has_collection(self._settings.collection_name):
      logger.info(
          "Collection '%s' already exists, skipping setup.",
          self._settings.collection_name,
      )
      return

    schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(
        field_name=self._settings.primary_field,
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name=self._settings.content_field,
        datatype=DataType.VARCHAR,
        max_length=65535,
    )
    schema.add_field(
        field_name=self._settings.embedding_field,
        datatype=DataType.FLOAT_VECTOR,
        dim=self._settings.dimension,
    )

    index_params = self._client.prepare_index_params()
    index_params.add_index(
        field_name=self._settings.embedding_field,
        index_type=self._settings.index_type,
        metric_type=self._settings.metric_type,
    )

    self._client.create_collection(
        collection_name=self._settings.collection_name,
        schema=schema,
        index_params=index_params,
    )

    logger.info(
        "Created collection '%s' with dimension=%d, metric=%s, index=%s.",
        self._settings.collection_name,
        self._settings.dimension,
        self._settings.metric_type,
        self._settings.index_type,
    )

  def add_contents(
      self,
      contents: list[str],
      *,
      additional_fields: Optional[list[dict]] = None,
      batch_size: int = 200,
  ) -> None:
    """Adds text contents to the vector store.

    Performs batch embedding generation and insertion into the Milvus
    collection.

    Args:
      contents: An iterable of text contents to add.
      additional_fields: Optional list of dicts with extra field values
        for each content row.
      batch_size: Maximum number of items per batch. Defaults to 200.

    Raises:
      ValueError: If additional_fields length does not match contents length.
    """
    if additional_fields and len(additional_fields) != len(contents):
      raise ValueError(
          "The number of additional_fields must match the number of contents."
      )

    total_rows = 0
    for i in range(0, len(contents), batch_size):
      batch_contents = contents[i : i + batch_size]
      batch_extra = (
          additional_fields[i : i + batch_size] if additional_fields else None
      )

      logger.debug(
          "Embedding batch %d to %d (size: %d)...",
          i,
          i + len(batch_contents),
          len(batch_contents),
      )
      embeddings = self._embedding_fn(batch_contents)

      data = []
      for j, (content, embedding) in enumerate(zip(batch_contents, embeddings)):
        row = {
            self._settings.content_field: content,
            self._settings.embedding_field: embedding,
        }
        if batch_extra and j < len(batch_extra):
          row.update(batch_extra[j])
        data.append(row)

      self._client.insert(
          collection_name=self._settings.collection_name,
          data=data,
      )
      total_rows += len(data)

    logger.info(
        "Added %d contents to collection '%s'.",
        total_rows,
        self._settings.collection_name,
    )

  async def add_contents_async(
      self,
      contents: list[str],
      *,
      additional_fields: Optional[list[dict]] = None,
      batch_size: int = 200,
  ) -> None:
    """Asynchronously adds text contents to the vector store.

    Args:
      contents: An iterable of text contents to add.
      additional_fields: Optional list of dicts with extra field values
        for each content row.
      batch_size: Maximum number of items per batch. Defaults to 200.
    """
    await asyncio.to_thread(
        self.add_contents,
        contents,
        additional_fields=additional_fields,
        batch_size=batch_size,
    )

  def search(
      self,
      query: str,
      *,
      top_k: Optional[int] = None,
      filter_expr: Optional[str] = None,
  ) -> list[dict]:
    """Performs vector similarity search.

    Args:
      query: The search query text. It will be embedded using the
        configured embedding function.
      top_k: Number of results to return. Overrides the default from
        settings if provided.
      filter_expr: Optional Milvus filter expression for pre-filtering.

    Returns:
      A list of dicts, each containing the matched content and any
      configured output fields, along with the distance score.
    """
    top_k = top_k or self._settings.top_k

    query_embedding = self._embedding_fn([query])[0]

    output_fields = self._settings.output_fields or [
        self._settings.content_field
    ]

    search_params = {"metric_type": self._settings.metric_type}

    results = self._client.search(
        collection_name=self._settings.collection_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=output_fields,
        search_params=search_params,
        filter=filter_expr or "",
    )

    if not results or not results[0]:
      return []

    return [
        {**hit["entity"], "distance": hit["distance"]} for hit in results[0]
    ]

  def close(self) -> None:
    """Closes the Milvus client connection."""
    self._client.close()

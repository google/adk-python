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

"""Settings for Milvus vector store and toolset."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import model_validator


class MilvusVectorStoreSettings(BaseModel):
  """Settings for Milvus vector store.

  This is used for vector similarity search in a Milvus collection.
  Provide the collection and connection settings to use with the
  ``similarity_search`` tool.
  """

  uri: str = "http://localhost:19530"
  """The Milvus server URI.

  Can be a remote address like ``http://localhost:19530`` or a local
  file path for Milvus Lite (e.g. ``./milvus.db``).
  """

  token: Optional[str] = None
  """Optional authentication token (e.g. for Zilliz Cloud)."""

  db_name: str = "default"
  """The Milvus database name."""

  collection_name: str
  """Required. The name of the Milvus collection."""

  dimension: int = 768
  """The dimension of the embedding vectors."""

  metric_type: str = "COSINE"
  """The distance metric for similarity search.

  Supported values: ``COSINE``, ``L2``, ``IP``.
  """

  index_type: str = "AUTOINDEX"
  """The index type for the vector field.

  Supported values: ``AUTOINDEX``, ``IVF_FLAT``, ``HNSW``, etc.
  """

  content_field: str = "content"
  """The name of the text content field in the collection."""

  embedding_field: str = "embedding"
  """The name of the vector embedding field in the collection."""

  primary_field: str = "id"
  """The name of the primary key field in the collection."""

  top_k: int = 5
  """The default number of results to return from similarity search."""

  output_fields: Optional[list[str]] = None
  """Optional additional fields to return in search results.

  If ``None``, only the ``content_field`` is returned.
  """

  @model_validator(mode="after")
  def _validate_settings(self):
    """Validate the vector store settings."""
    if self.dimension <= 0:
      raise ValueError(
          f"Invalid dimension: {self.dimension}. Must be positive."
      )
    if self.top_k <= 0:
      raise ValueError(f"Invalid top_k: {self.top_k}. Must be positive.")
    return self


class MilvusToolSettings(BaseModel):
  """Settings for Milvus toolset."""

  vector_store_settings: Optional[MilvusVectorStoreSettings] = None
  """Settings for Milvus vector store and vector similarity search."""

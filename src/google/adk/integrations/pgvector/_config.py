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

"""Configuration for the PostgreSQL/pgvector integrations."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class PgVectorMemoryServiceConfig(BaseModel):
  """Configuration for PgVectorMemoryService.

  The embedding model is called through the ``google-genai`` client, but the
  vectors themselves live in a PostgreSQL database you control, so the memory
  store does not depend on any managed Google Cloud service.
  """

  dsn: Optional[str] = Field(
      default=None,
      description=(
          "PostgreSQL connection string, e.g."
          " postgresql://user:password@host:5432/dbname. Required unless a"
          " pre-configured connection pool is passed to the service."
      ),
  )
  table_name: str = Field(
      default="adk_memory_entries",
      description="Name of the table that stores memory entries.",
  )
  embedding_model: str = Field(
      default="gemini-embedding-001",
      description=(
          "google-genai embedding model used to embed events and queries."
      ),
  )
  embedding_dimension: int = Field(
      default=768,
      description=(
          "Dimension of the stored embedding vectors. Must match the output"
          " dimension of embedding_model and stays fixed for the life of the"
          " table."
      ),
  )
  top_k: int = Field(
      default=10,
      description="Maximum number of memories returned by a search.",
  )
  hnsw_m: int = Field(
      default=16,
      description="pgvector HNSW index parameter m (graph degree).",
  )
  hnsw_ef_construction: int = Field(
      default=64,
      description=(
          "pgvector HNSW index parameter ef_construction (build effort)."
      ),
  )
  distance_threshold: Optional[float] = Field(
      default=None,
      description=(
          "Optional cosine-distance ceiling in [0, 2]. When set, memories whose"
          " distance to the query is greater than this are dropped from the"
          " results. When None, the closest top_k memories are always returned."
      ),
  )

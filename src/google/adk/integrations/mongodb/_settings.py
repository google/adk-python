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

from pydantic import BaseModel
from pydantic import Field

from ...features import experimental
from ...features import FeatureName


@experimental(FeatureName.MONGODB_TOOL_SETTINGS)
class MongoDbToolSettings(BaseModel):
  """Settings for MongoDB tools."""

  vertex_ai_embedding_model_name: str = "text-embedding-005"
  """The Vertex AI embedding model name used to generate query embeddings.

  The query vector is compared against vectors already stored in the
  collection, so this has to be the model those were produced with. A
  different model, or the same model at a different output dimensionality,
  ranks by a similarity that means nothing.
  """

  vertex_ai_embedding_output_dimensionality: int | None = Field(
      default=None, gt=0
  )
  """Length of the query embedding, for models that support shortening it.

  Leave it unset to get the model's native length. Set it to the length of the
  stored vectors when those were generated with a truncated embedding.
  """

  default_vector_index_name: str = "vector_index"
  """Default name of the vector search index to query."""

  default_search_index_name: str = "default"
  """Default name of the full-text search index used by hybrid search."""

  default_embedding_field: str = "embedding"
  """Default document field that stores embedding vectors."""

  default_limit: int = Field(default=4, gt=0)
  """Default number of documents returned by a search operation."""

  max_results: int = Field(default=50, gt=0)
  """Maximum number of documents a search operation may return."""

  default_num_candidates: int = Field(default=100, gt=0)
  """Default number of nearest neighbors considered by vector search."""

  timeout_ms: int = Field(default=60_000, gt=0)
  """Time limit in milliseconds for a single search operation.

  A search runs on a worker thread that cannot be cancelled, and the model
  steers how much work it asks the server for, so every search is bounded.
  """

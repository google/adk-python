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

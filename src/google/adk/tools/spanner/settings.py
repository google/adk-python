# Copyright 2025 Google LLC
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

from enum import Enum
from typing import List
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import model_validator

from ...features import experimental
from ...features import FeatureName

# Vector similarity search nearest neighbors search algorithms.
EXACT_NEAREST_NEIGHBORS = "EXACT_NEAREST_NEIGHBORS"
APPROXIMATE_NEAREST_NEIGHBORS = "APPROXIMATE_NEAREST_NEIGHBORS"
NearestNeighborsAlgorithm = Literal[
    EXACT_NEAREST_NEIGHBORS,
    APPROXIMATE_NEAREST_NEIGHBORS,
]


class Capabilities(Enum):
  """Capabilities indicating what type of operation tools are allowed to be performed on Spanner."""

  DATA_READ = "data_read"
  """Read only data operations tools are allowed."""


class QueryResultMode(Enum):
  """Settings for Spanner execute sql query result."""

  DEFAULT = "default"
  """Return the result of a query as a list of rows data."""

  DICT_LIST = "dict_list"
  """Return the result of a query as a list of dictionaries.

  In each dictionary the key is the column name and the value is the value of
  the that column in a given row.
  """


class SpannerVectorStoreSettings(BaseModel):
  """Settings for Spanner Vector Store.

  This is used for vector similarity search in a Spanner vector store table.
  Provide the vector store table and the embedding model settings to use with
  the `vector_store_similarity_search` tool.
  """

  project_id: str
  """Required. The GCP project id in which the Spanner database resides."""

  instance_id: str
  """Required. The instance id of the Spanner database."""

  database_id: str
  """Required. The database id of the Spanner database."""

  table_name: str
  """Required. The name of the vector store table to use for vector similarity search."""

  content_column: str
  """Required. The name of the content column in the vector store table. By default, this column value is also returned as part of the vector similarity search result."""

  embedding_column: str
  """Required. The name of the embedding column to search in the vector store table."""

  vector_length: int
  """Required. The the dimension of the vectors in the `embedding_column`."""

  vertex_ai_embedding_model_name: str
  """Required. The Vertex AI embedding model name, which is used to generate embeddings for vector store and vector similarity search.
    For example, 'text-embedding-005'.

    Note: the output dimensionality of the embedding model should be the same as the value specified in the `vector_length` field.
    Otherwise, a runtime error might be raised during a query.
  """

  selected_columns: List[str] = []
  """Required. The vector store table columns to return in the vector similarity search result.

    By default, only the `content_column` value and the distance value are returned.
    If sepecified, the list of selected columns and the distance value are returned.
    For example, if `selected_columns` is ['col1', 'col2'], then the result will contain the values of 'col1' and 'col2' columns and the distance value.
  """

  nearest_neighbors_algorithm: NearestNeighborsAlgorithm = (
      "EXACT_NEAREST_NEIGHBORS"
  )
  """The algorithm used to perform vector similarity search. This value can be EXACT_NEAREST_NEIGHBORS or APPROXIMATE_NEAREST_NEIGHBORS.

    For more details about EXACT_NEAREST_NEIGHBORS, see https://docs.cloud.google.com/spanner/docs/find-k-nearest-neighbors
    For more details about APPROXIMATE_NEAREST_NEIGHBORS, see https://docs.cloud.google.com/spanner/docs/find-approximate-nearest-neighbors
  """

  top_k: int = 4
  """Required. The number of neighbors to return for each vector similarity search query. The default value is 4."""

  distance_type: str = "COSINE"
  """Required. The distance metric used to build the vector index or perform vector similarity search. This value can be COSINE, DOT_PRODUCT, or EUCLIDEAN."""

  num_leaves_to_search: Optional[int] = None
  """Optional. This option specifies how many leaf nodes of the index are searched.

    Note: this option is only used when the nearest neighbors search algorithm (`nearest_neighbors_algorithm`) is APPROXIMATE_NEAREST_NEIGHBORS.
    For more details, see https://docs.cloud.google.com/spanner/docs/vector-index-best-practices
  """

  additional_filter: Optional[str] = None
  """Optional. An optional filter to apply to the search query. If provided, this will be added to the WHERE clause of the final query."""

  @model_validator(mode="after")
  def __post_init__(self):
    """Validate the embedding settings."""
    if not self.vector_length or self.vector_length <= 0:
      raise ValueError(
          "Invalid vector length in the Spanner vector store settings."
      )

    if not self.selected_columns:
      self.selected_columns = [self.content_column]

    return self


@experimental(FeatureName.SPANNER_TOOL_SETTINGS)
class SpannerToolSettings(BaseModel):
  """Settings for Spanner tools."""

  capabilities: List[Capabilities] = [
      Capabilities.DATA_READ,
  ]
  """Allowed capabilities for Spanner tools.

  By default, the tool will allow only read operations. This behaviour may
  change in future versions.
  """

  max_executed_query_result_rows: int = 50
  """Maximum number of rows to return from a query result."""

  query_result_mode: QueryResultMode = QueryResultMode.DEFAULT
  """Mode for Spanner execute sql query result."""

  vector_store_settings: Optional[SpannerVectorStoreSettings] = None
  """Settings for Spanner vector store and vector similarity search."""

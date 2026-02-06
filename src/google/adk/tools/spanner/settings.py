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


class TableColumn(BaseModel):
  """Represents column configuration, to be used as part of create DDL statement for a new vector store table set up."""

  name: str
  """Required. The name of the column."""

  type: str
  """Required. The type of the column.

  For example,

  - GoogleSQL: 'STRING(MAX)', 'INT64', 'FLOAT64', 'BOOL', etc.
  - PostgreSQL: 'text', 'int8', 'float8', 'boolean', etc.
  """

  is_nullable: bool = True
  """Optional. Whether the column is nullable. By default, the column is nullable."""


class VectorSearchIndexSettings(BaseModel):
  """Settings for the index for use with Approximate Nearest Neighbor (ANN) vector similarity search."""

  index_name: str
  """Required. The name of the vector similarity search index."""

  additional_key_columns: Optional[list[str]] = None
  """Optional. The list of the additional key column names in the vector similarity search index.

  To further speed up filtering for highly selective filtering columns, organize
  them as additional keys in the vector index after the embedding column.
  For example: `category` as additional key column.
  `CREATE VECTOR INDEX ON documents(embedding, category);`
  """

  additional_storing_columns: Optional[list[str]] = None
  """Optional. The list of the storing column names in the vector similarity search index.

  This enables filtering while walking the vector index, removing unqualified
  rows early.
  For example: `category` as storing column.
  `CREATE VECTOR INDEX ON documents(embedding) STORING (category);`
  """

  tree_depth: int = 2
  """Required. The tree depth (level). This value can be either 2 or 3.

  A tree with 2 levels only has leaves (num_leaves) as nodes.
  If the dataset has more than 100 million rows,
  then you can use a tree with 3 levels and add branches (num_branches) to
  further partition the dataset.
  """

  num_leaves: int = 1000
  """Required. The number of leaves (i.e. potential partitions) for the vector data.

  You can designate num_leaves for trees with 2 or 3 levels.
  We recommend that the number of leaves is number_of_rows_in_dataset/1000.
  """

  num_branches: Optional[int] = None
  """Optional. The number of branches to further partition the vector data.

  You can only designate num_branches for trees with 3 levels.
  The number of branches must be fewer than the number of leaves
  We recommend that the number of leaves is between 1000 and sqrt(number_of_rows_in_dataset).
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
  """Required. The dimension of the vectors in the `embedding_column`."""

  vertex_ai_embedding_model_name: str
  """Required. The Vertex AI embedding model name, which is used to generate embeddings for vector store and vector similarity search.

  For example, 'text-embedding-005'.

  Note: the output dimensionality of the embedding model should be the same as the value specified in the `vector_length` field.
  Otherwise, a runtime error might be raised during a query.
  """

  selected_columns: list[str] = []
  """Required. The vector store table columns to return in the vector similarity search result.

  By default, only the `content_column` value and the distance value are returned.
  If specified, the list of selected columns and the distance value are returned.
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

  Note: This option is only used when the nearest neighbors search algorithm (`nearest_neighbors_algorithm`) is APPROXIMATE_NEAREST_NEIGHBORS.
  For more details, see https://docs.cloud.google.com/spanner/docs/vector-index-best-practices
  """

  additional_filter: Optional[str] = None
  """Optional. An optional filter to apply to the search query. If provided, this will be added to the WHERE clause of the final query."""

  vector_search_index_settings: Optional[VectorSearchIndexSettings] = None
  """Optional. Settings for the index for use with Approximate Nearest Neighbor (ANN) in the vector store.

  Note: This option is only required when the nearest neighbors search algorithm (`nearest_neighbors_algorithm`) is APPROXIMATE_NEAREST_NEIGHBORS.
  For more details, see https://docs.cloud.google.com/spanner/docs/vector-indexes
  """

  additional_columns_to_setup: Optional[list[TableColumn]] = None
  """Optional. A list of supplemental columns to be created when initializing a new vector store table or inserting content rows.

  Note: This configuration is only utilized during the initial table setup
  or when inserting content rows.
  """

  primary_key_columns: Optional[list[str]] = None
  """Optional. Specifies the column names to be used as the primary key for a new vector store table.

  If provided, every column name listed here must be defined within
  `additional_columns_to_setup`. If this field is omitted (set to `None`),
  defaults to a single primary key column named `id` which automatically
  generates UUIDs for each entry.

  Note: This field is only used during the creation phase of a new vector store.
  """

  @model_validator(mode="after")
  def __post_init__(self):
    """Validate the vector store settings."""
    if not self.vector_length or self.vector_length <= 0:
      raise ValueError(
          "Invalid vector length in the Spanner vector store settings."
      )

    if not self.selected_columns:
      self.selected_columns = [self.content_column]

    if self.primary_key_columns:
      cols = {self.content_column, self.embedding_column}
      if self.additional_columns_to_setup:
        cols.update({c.name for c in self.additional_columns_to_setup})

      for pk in self.primary_key_columns:
        if pk not in cols:
          raise ValueError(
              f"Primary key column '{pk}' not found in column definitions."
          )

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

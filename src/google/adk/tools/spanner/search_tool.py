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

import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.adk.tools.spanner import client
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.tool_context import ToolContext
from google.auth.credentials import Credentials
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1.database import Database

# Embedding options
_SPANNER_EMBEDDING_MODEL_NAME = "spanner_embedding_model_name"
_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT = "vertex_ai_embedding_model_endpoint"

# Search options
_TOP_K = "top_k"
_DISTANCE_TYPE = "distance_type"
_NEAREST_NEIGHBORS_ALGORITHM = "nearest_neighbors_algorithm"
_EXACT_NEAREST_NEIGHBORS = "EXACT_NEAREST_NEIGHBORS"
_APPROXIMATE_NEAREST_NEIGHBORS = "APPROXIMATE_NEAREST_NEIGHBORS"
_NUM_LEAVES_TO_SEARCH = "num_leaves_to_search"

# Constants
_DISTANCE_ALIAS = "distance"
_GOOGLESQL_PARAMETER_TEXT_QUERY = "query"
_POSTGRESQL_PARAMETER_TEXT_QUERY = "1"
_GOOGLESQL_PARAMETER_QUERY_EMBEDDING = "embedding"
_POSTGRESQL_PARAMETER_QUERY_EMBEDDING = "1"


def _generate_googlesql_for_embedding_query(
    spanner_embedding_model_name: str,
) -> str:
  return f"""
    SELECT embeddings.values
    FROM ML.PREDICT(
      MODEL {spanner_embedding_model_name},
      (SELECT CAST(@{_GOOGLESQL_PARAMETER_TEXT_QUERY} AS STRING) as content)
    )
  """


def _generate_postgresql_for_embedding_query(
    vertex_ai_embedding_model_endpoint: str,
) -> str:
  return f"""
    SELECT spanner.FLOAT32_ARRAY( spanner.ML_PREDICT_ROW(
      '{vertex_ai_embedding_model_endpoint}',
      JSONB_BUILD_OBJECT(
        'instances',
        JSONB_BUILD_ARRAY( JSONB_BUILD_OBJECT(
          'content',
          ${_POSTGRESQL_PARAMETER_TEXT_QUERY}::TEXT
        ))
      )
    ) -> 'predictions'->0->'embeddings'->'values' )
  """


def _get_embedding_for_query(
    database: Database,
    dialect: DatabaseDialect,
    spanner_embedding_model_name: Optional[str],
    vertex_ai_embedding_model_endpoint: Optional[str],
    query: str,
) -> List[float]:
  """Gets the embedding for the query."""
  if dialect == DatabaseDialect.POSTGRESQL:
    embedding_query = _generate_postgresql_for_embedding_query(
        vertex_ai_embedding_model_endpoint
    )
    params = {f"p{_POSTGRESQL_PARAMETER_TEXT_QUERY}": query}
  else:
    embedding_query = _generate_googlesql_for_embedding_query(
        spanner_embedding_model_name
    )
    params = {_GOOGLESQL_PARAMETER_TEXT_QUERY: query}
  with database.snapshot() as snapshot:
    result_set = snapshot.execute_sql(embedding_query, params=params)
    return result_set.one()[0]


def _get_postgresql_distance_function(distance_type: str) -> str:
  return {
      "COSINE_DISTANCE": "spanner.cosine_distance",
      "EUCLIDEAN_DISTANCE": "spanner.euclidean_distance",
      "DOT_PRODUCT": "spanner.dot_product",
  }[distance_type]


def _get_googlesql_distance_function(distance_type: str, ann: bool) -> str:
  if ann:
    return {
        "COSINE_DISTANCE": "APPROX_COSINE_DISTANCE",
        "EUCLIDEAN_DISTANCE": "APPROX_EUCLIDEAN_DISTANCE",
        "DOT_PRODUCT": "APPROX_DOT_PRODUCT",
    }[distance_type]
  return {
      "COSINE_DISTANCE": "COSINE_DISTANCE",
      "EUCLIDEAN_DISTANCE": "EUCLIDEAN_DISTANCE",
      "DOT_PRODUCT": "DOT_PRODUCT",
  }[distance_type]


def _generate_sql_for_knn(
    dialect: DatabaseDialect,
    table_name: str,
    embedding_column_to_search: str,
    columns,
    additional_filter: Optional[str],
    distance_type: str,
    top_k: int,
) -> str:
  """Generates a SQL query for kNN search."""
  if dialect == DatabaseDialect.POSTGRESQL:
    distance_function = _get_postgresql_distance_function(distance_type)
    embedding_parameter = f"${_POSTGRESQL_PARAMETER_QUERY_EMBEDDING}"
  else:
    distance_function = _get_googlesql_distance_function(
        distance_type, ann=False
    )
    embedding_parameter = f"@{_GOOGLESQL_PARAMETER_QUERY_EMBEDDING}"
  columns = columns + [f"""{distance_function}(
      {embedding_column_to_search},
      {embedding_parameter}) AS {_DISTANCE_ALIAS}
  """]
  columns = ", ".join(columns)
  if additional_filter is None:
    additional_filter = "1=1"

  optional_limit_clause = ""
  if top_k > 0:
    optional_limit_clause = f"""LIMIT {top_k}"""
  return f"""
    SELECT {columns}
    FROM {table_name}
    WHERE {additional_filter}
    ORDER BY {_DISTANCE_ALIAS}
    {optional_limit_clause}
  """


def _generate_sql_for_ann(
    dialect: DatabaseDialect,
    table_name: str,
    embedding_column_to_search: str,
    columns,
    additional_filter: Optional[str],
    distance_type: str,
    top_k: int,
    num_leaves_to_search: int,
):
  """Generates a SQL query for ANN search."""
  if dialect == DatabaseDialect.POSTGRESQL:
    raise NotImplementedError(
        f"{_APPROXIMATE_NEAREST_NEIGHBORS} is not supported for PostgreSQL"
        " dialect."
    )
  distance_function = _get_googlesql_distance_function(distance_type, ann=True)
  columns = columns + [f"""{distance_function}(
      {embedding_column_to_search},
      @{_GOOGLESQL_PARAMETER_QUERY_EMBEDDING},
      options => JSON '{{"num_leaves_to_search": {num_leaves_to_search}}}'
  ) AS {_DISTANCE_ALIAS}
  """]
  columns = ", ".join(columns)
  query_filter = f"{embedding_column_to_search} IS NOT NULL"
  if additional_filter is not None:
    query_filter = f"{query_filter} AND {additional_filter}"

  return f"""
    SELECT {columns}
    FROM {table_name}
    WHERE {query_filter}
    ORDER BY {_DISTANCE_ALIAS}
    LIMIT {top_k}
  """


def similarity_search(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_name: str,
    query: str,
    embedding_column_to_search: str,
    columns: List[str],
    embedding_options: Dict[str, str],
    credentials: Credentials,
    settings: SpannerToolSettings,
    tool_context: ToolContext,
    additional_filter: Optional[str] = None,
    search_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  # fmt: off
  """Similarity search in Spanner using a text query.

  The function will use embedding service (provided from options) to embed
  the text query automatically, then use the embedding vector to do similarity
  search and to return requested data. This is suitable when the Spanner table
  contains a column that stores the embeddings of the data that we want to
  search the `query` against.

  Args:
      project_id (str): The GCP project id in which the spanner database
        resides.
      instance_id (str): The instance id of the spanner database.
      database_id (str): The database id of the spanner database.
      table_name (str): The name of the table used for vector search.
      query (str): The user query for which the tool will find the top similar
        content. The query will be embedded and used for vector search.
      embedding_column_to_search (str): The name of the column that contains the
        embeddings of the documents. The tool will do similarity search on this
        column.
      columns (List[str]): A list of column names, representing the additional
        columns to return in the search results.
      embedding_options (Dict[str, str]): A dictionary of options to use for
        the embedding service. The following options are supported:
        - spanner_embedding_model_name: (For GoogleSQL dialect) The
          name of the embedding model that is registered in Spanner via a
          `CREATE MODEL` statement. For more details, see
          https://cloud.google.com/spanner/docs/ml-tutorial-embeddings#generate_and_store_text_embeddings
        - vertex_ai_embedding_model_endpoint: (For PostgreSQL dialect)
          The fully qualified endpoint of the Vertex AI embedding model,
          in the format of
          `projects/$project/locations/$location/publishers/google/models/$model_name`,
          where $project is the project hosting the Vertex AI endpoint,
          $location is the location of the endpoint, and $model_name is
          the name of the text embedding model.
      credentials (Credentials): The credentials to use for the request.
      settings (SpannerToolSettings): The configuration for the tool.
      tool_context (ToolContext): The context for the tool.
      additional_filter (Optional[str]): An optional filter to apply to the
        search query. If provided, this will be added to the WHERE clause of the
        final query.
      search_options (Optional[Dict[str, Any]]): A dictionary of options to use
        for the similarity search. The following options are supported:
        - top_k: The number of most similar documents to return. The
          default value is 4.
        - distance_type: The distance type to use to perform the
          similarity search. Valid values include "COSINE_DISTANCE",
          "EUCLIDEAN_DISTANCE", and "DOT_PRODUCT". Default value is
          "COSINE_DISTANCE".
        - nearest_neighbors_algorithm: The nearest neighbors search
          algorithm to use. Valid values include "EXACT_NEAREST_NEIGHBORS"
          and "APPROXIMATE_NEAREST_NEIGHBORS". Default value is
          "EXACT_NEAREST_NEIGHBORS".
        - num_leaves_to_search: (Only applies when the
          nearest_neighbors_algorithm is APPROXIMATE_NEAREST_NEIGHBORS.)
          The number of leaves to search in the vector index.

  Returns:
      Dict[str, Any]: A dictionary representing the result of the search.
        On success, it contains {"status": "SUCCESS", "rows": [...]}. The last
        column of each row is the distance between the query and the column
        embedding (i.e. the embedding_column_to_search).
        On error, it contains {"status": "ERROR", "error_details": "..."}.

  Examples:
      Search for relevant products given a user's text description and a filter
      on the price:
        >>> similarity_search(
        ...   project_id="my-project",
        ...   instance_id="my-instance",
        ...   database_id="my-database",
        ...   table_name="my-product-table",
        ...   query="Tools that can help me clean my house.",
        ...   embedding_column_to_search="product_description_embedding",
        ...   columns=["product_name", "product_description", "price_in_cents"],
        ...   credentials=credentials,
        ...   settings=settings,
        ...   tool_context=tool_context,
        ...   additional_filter="price_in_cents < 100000",
        ...   embedding_options={
        ...     "spanner_embedding_model_name": "my_embedding_model"
        ...   },
        ...   search_options={
        ...     "top_k": 2,
        ...     "distance_type": "COSINE_DISTANCE"
        ...   }
        ... )
        {
          "status": "SUCCESS",
          "rows": [
            (
              "Powerful Robot Vacuum",
              "This is a powerful robot vacuum that can clean carpets and wood floors.",
              99999,
              0.31,
            ),
            (
              "Nice Mop",
              "Great for cleaning different surfaces.",
              5099,
              0.45,
            ),
          ],
        }
  """
  # fmt: on
  try:
    # Get Spanner client
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    assert database.database_dialect in [
        DatabaseDialect.GOOGLE_STANDARD_SQL,
        DatabaseDialect.POSTGRESQL,
    ], (
        "Unsupported database dialect: %s" % database.database_dialect
    )

    if embedding_options is None:
      embedding_options = {}
    if search_options is None:
      search_options = {}
    spanner_embedding_model_name = embedding_options.get(
        _SPANNER_EMBEDDING_MODEL_NAME
    )
    if (
        database.database_dialect == DatabaseDialect.GOOGLE_STANDARD_SQL
        and spanner_embedding_model_name is None
    ):
      raise ValueError(
          f"embedding_options['{_SPANNER_EMBEDDING_MODEL_NAME}']"
          " must be specified for GoogleSQL dialect."
      )
    vertex_ai_embedding_model_endpoint = embedding_options.get(
        _VERTEX_AI_EMBEDDING_MODEL_ENDPOINT
    )
    if (
        database.database_dialect == DatabaseDialect.POSTGRESQL
        and vertex_ai_embedding_model_endpoint is None
    ):
      raise ValueError(
          f"embedding_options['{_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT}']"
          " must be specified for PostgreSQL dialect."
      )

    # Use cosine distance by default.
    distance_type = search_options.get(_DISTANCE_TYPE)
    if distance_type is None:
      distance_type = "COSINE_DISTANCE"

    top_k = search_options.get(_TOP_K)
    if top_k is None:
      top_k = 4

    # Use EXACT_NEAREST_NEIGHBORS (i.e. kNN) by default.
    nearest_neighbors_algorithm = search_options.get(
        _NEAREST_NEIGHBORS_ALGORITHM, _EXACT_NEAREST_NEIGHBORS
    )
    if nearest_neighbors_algorithm not in (
        _EXACT_NEAREST_NEIGHBORS,
        _APPROXIMATE_NEAREST_NEIGHBORS,
    ):
      raise NotImplementedError(
          f"Unsupported search_options['{_NEAREST_NEIGHBORS_ALGORITHM}']:"
          f" {nearest_neighbors_algorithm}"
      )

    embedding = _get_embedding_for_query(
        database,
        database.database_dialect,
        spanner_embedding_model_name,
        vertex_ai_embedding_model_endpoint,
        query,
    )

    if nearest_neighbors_algorithm == _EXACT_NEAREST_NEIGHBORS:
      sql = _generate_sql_for_knn(
          database.database_dialect,
          table_name,
          embedding_column_to_search,
          columns,
          additional_filter,
          distance_type,
          top_k,
      )
    else:
      num_leaves_to_search = search_options.get(_NUM_LEAVES_TO_SEARCH)
      if num_leaves_to_search is None:
        num_leaves_to_search = 1000
      sql = _generate_sql_for_ann(
          database.database_dialect,
          table_name,
          embedding_column_to_search,
          columns,
          additional_filter,
          distance_type,
          top_k,
          num_leaves_to_search,
      )

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      params = {f"p{_POSTGRESQL_PARAMETER_QUERY_EMBEDDING}": embedding}
    else:
      params = {_GOOGLESQL_PARAMETER_QUERY_EMBEDDING: embedding}

    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(sql, params=params)
      rows = []
      result = {}
      for row in result_set:
        try:
          # if the json serialization of the row succeeds, use it as is
          json.dumps(row)
        except:
          row = str(row)

        rows.append(row)

      result["status"] = "SUCCESS"
      result["rows"] = rows
      return result
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

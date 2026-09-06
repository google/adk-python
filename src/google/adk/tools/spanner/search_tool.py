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

import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
import json
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from google.auth.credentials import Credentials
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from pydantic import TypeAdapter
from pydantic import ValidationError

from . import client
from . import utils
from .settings import APPROXIMATE_NEAREST_NEIGHBORS
from .settings import EXACT_NEAREST_NEIGHBORS
from .settings import SpannerToolSettings

if TYPE_CHECKING:
  from google.cloud import spanner

# Pattern for valid SQL identifiers: alphanumeric, underscores,
# dots (for schema-qualified names), and backtick/double-quote quoting.
# Supports per-part quoting for schema-qualified names.
_IDENTIFIER_PART_PATTERN = r'(?:[A-Za-z_][A-Za-z0-9_]*|`[^`\\]+`|"[^"\\]+")'
_SAFE_IDENTIFIER_RE = re.compile(
    rf"^{_IDENTIFIER_PART_PATTERN}(?:\.{_IDENTIFIER_PART_PATTERN})*$"
)

# Operator allowlist for additional_filter
_ALLOWED_OPERATORS = r"(?:=|!=|<=|>=|<|>|(?i:\bLIKE\b|\bIS\s+NOT\b|\bIS\b))"

# Value allowlist for additional_filter: numbers, single-quoted strings (no backslashes), booleans, NULL
_ALLOWED_VALUES = (
    r"(?:[+-]?\d+(?:\.\d+)?|'[^'\\]*'|(?i:\bTRUE\b|\bFALSE\b|\bNULL\b))"
)

# IN operator support
_IN_OPERATOR = r"(?i:\bNOT\s+IN\b|\bIN\b)"
_IN_VALUES = rf"\(\s*{_ALLOWED_VALUES}(?:\s*,\s*{_ALLOWED_VALUES})*\s*\)"

# BETWEEN operator support
_BETWEEN_OPERATOR = r"(?i:\bBETWEEN\b)"
_BETWEEN_VALUE = rf"{_ALLOWED_VALUES}\s+(?i:\bAND\b)\s+{_ALLOWED_VALUES}"

# A single condition (without paren)
_BASE_COND = (
    rf"(?:"
    rf"(?:{_IDENTIFIER_PART_PATTERN}(?:\.{_IDENTIFIER_PART_PATTERN})*)\s*{_ALLOWED_OPERATORS}\s*{_ALLOWED_VALUES}"
    rf"|(?:{_IDENTIFIER_PART_PATTERN}(?:\.{_IDENTIFIER_PART_PATTERN})*)\s*{_IN_OPERATOR}\s*{_IN_VALUES}"
    rf"|(?:{_IDENTIFIER_PART_PATTERN}(?:\.{_IDENTIFIER_PART_PATTERN})*)\s*{_BETWEEN_OPERATOR}\s*{_BETWEEN_VALUE}"
    rf"|{_IDENTIFIER_PART_PATTERN}(?:\.{_IDENTIFIER_PART_PATTERN})*"  # Just identifier (e.g. boolean col)
    rf"|1\s*=\s*1"  # dummy filter
    rf")"
)

_BLOCK_0 = rf"{_BASE_COND}(?:\s+(?i:\bAND\b|\bOR\b)\s+{_BASE_COND})*"
_COND_1 = rf"(?:{_BASE_COND}|\(\s*{_BLOCK_0}\s*\))"
_BLOCK_1 = rf"{_COND_1}(?:\s+(?i:\bAND\b|\bOR\b)\s+{_COND_1})*"
_COND_2 = rf"(?:{_BASE_COND}|\(\s*{_BLOCK_1}\s*\))"

# Full filter pattern: conditions joined by AND/OR, supporting up to 2 levels of nested parens
_SAFE_FILTER_RE = re.compile(
    rf"^\s*{_COND_2}(?:\s+(?i:\bAND\b|\bOR\b)\s+{_COND_2})*\s*$",
    re.IGNORECASE,
)


def _validate_identifier(value: str, param_name: str) -> str:
  """Validate that a value is a safe SQL identifier.

  Args:
    value: The identifier string to validate.
    param_name: Name of the parameter (for error messages).

  Returns:
    The validated identifier string.

  Raises:
    ValueError: If the identifier contains unsafe characters.
  """
  if not value or not _SAFE_IDENTIFIER_RE.match(value.strip()):
    raise ValueError(
        f"Invalid SQL identifier for {param_name}: {value!r}. "
        "Identifiers must contain only alphanumeric characters, underscores, "
        "and dots, or be quoted with backticks or double quotes."
    )
  return value.strip()


def _validate_column_list(columns: List[str], param_name: str) -> List[str]:
  """Validate that each column name in a list is a safe SQL identifier."""
  validated = []
  for col in columns:
    _validate_identifier(col, param_name)
    validated.append(col)
  return validated


def _validate_additional_filter(
    filter_value: Optional[str],
) -> Optional[str]:
  """Validate that an additional_filter does not contain injection patterns.

  This is a defense-in-depth measure. The additional_filter field is
  documented as a developer-trusted value, but since it can be populated
  by the LLM at runtime via tool calls, we restrict it to an allow-listed
  grammar.

  Args:
    filter_value: The filter string to validate.

  Returns:
    The validated filter string, or None.

  Raises:
    ValueError: If the filter contains dangerous patterns.
  """
  if filter_value is None:
    return None
  if not _SAFE_FILTER_RE.match(filter_value):
    raise ValueError(
        "additional_filter contains unsafe or unsupported patterns: "
        f"{filter_value!r}. Only simple filters using =, !=, <=, >=, <, >, "
        "LIKE, IS, IS NOT, IN, BETWEEN joined by AND or OR (with up to 2 "
        "levels of nested parentheses) are allowed."
    )
  return filter_value


# Embedding model settings.
# Only for Spanner GoogleSQL dialect database, and use Spanner ML.PREDICT
# function.
_SPANNER_GSQL_EMBEDDING_MODEL_NAME = "spanner_googlesql_embedding_model_name"
# Only for Spanner PostgreSQL dialect database, and use spanner.ML_PREDICT_ROW
# to inferencing with Vertex AI embedding model endpoint.
_SPANNER_PG_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT = (
    "spanner_postgresql_vertex_ai_embedding_model_endpoint"
)
# For both Spanner GoogleSQL and PostgreSQL dialects, use Vertex AI embedding
# model to generate embeddings for vector similarity search.
_VERTEX_AI_EMBEDDING_MODEL_NAME = "vertex_ai_embedding_model_name"
_OUTPUT_DIMENSIONALITY = "output_dimensionality"

# Search options
_TOP_K = "top_k"
_DISTANCE_TYPE = "distance_type"
_NEAREST_NEIGHBORS_ALGORITHM = "nearest_neighbors_algorithm"
_NUM_LEAVES_TO_SEARCH = "num_leaves_to_search"

# Constants
_DISTANCE_ALIAS = "distance"
_GOOGLESQL_PARAMETER_TEXT_QUERY = "query"
_POSTGRESQL_PARAMETER_TEXT_QUERY = "1"
_GOOGLESQL_PARAMETER_QUERY_EMBEDDING = "embedding"
_POSTGRESQL_PARAMETER_QUERY_EMBEDDING = "1"


# The options arrive from a model-generated tool call, so their values are
# parsed rather than trusted: they end up in the generated SQL, and pydantic
# accepts the numeric strings a model tends to emit while rejecting anything
# that is not a number.
_OPTIONAL_STR: TypeAdapter[Optional[str]] = TypeAdapter(Optional[str])
_OPTIONAL_INT: TypeAdapter[Optional[int]] = TypeAdapter(Optional[int])


def _optional_str_option(options: Mapping[str, object], key: str) -> str | None:
  value = options.get(key)
  try:
    return _OPTIONAL_STR.validate_python(value)
  except ValidationError as ex:
    raise ValueError(f"Option {key!r} must be a string, got {value!r}.") from ex


def _optional_int_option(options: Mapping[str, object], key: str) -> int | None:
  value = options.get(key)
  try:
    return _OPTIONAL_INT.validate_python(value)
  except ValidationError as ex:
    raise ValueError(
        f"Option {key!r} must be an integer, got {value!r}."
    ) from ex


def _generate_googlesql_for_embedding_query(
    spanner_gsql_embedding_model_name: str,
) -> str:
  return f"""
    SELECT embeddings.values
    FROM ML.PREDICT(
      MODEL {spanner_gsql_embedding_model_name},
      (SELECT CAST(@{_GOOGLESQL_PARAMETER_TEXT_QUERY} AS STRING) as content)
    )
  """


def _generate_postgresql_for_embedding_query(
    vertex_ai_embedding_model_endpoint: str,
    output_dimensionality: int | None,
) -> str:
  if output_dimensionality is not None:
    output_dimensionality = int(output_dimensionality)
  instances_json = f"""
      'instances',
      JSONB_BUILD_ARRAY(
          JSONB_BUILD_OBJECT(
              'content',
              ${_POSTGRESQL_PARAMETER_TEXT_QUERY}::TEXT
          )
      )
  """

  params_list = []
  if output_dimensionality is not None:
    params_list.append(f"""
        'parameters',
        JSONB_BUILD_OBJECT(
            'outputDimensionality',
            {output_dimensionality}
        )
    """)

  jsonb_build_args = ",\n".join([instances_json] + params_list)

  return f"""
      SELECT spanner.FLOAT32_ARRAY(
          spanner.ML_PREDICT_ROW(
              '{vertex_ai_embedding_model_endpoint}',
              JSONB_BUILD_OBJECT(
                  {jsonb_build_args}
              )
          ) -> 'predictions' -> 0 -> 'embeddings' -> 'values'
      )
  """


def _get_embedding_for_query(
    database: client._SpannerDatabase,
    dialect: DatabaseDialect,
    spanner_gsql_embedding_model_name: str | None,
    spanner_pg_vertex_ai_embedding_model_endpoint: str | None,
    query: str,
    output_dimensionality: int | None = None,
) -> list[float]:
  """Gets the embedding for the query."""
  if dialect == DatabaseDialect.POSTGRESQL:
    if spanner_pg_vertex_ai_embedding_model_endpoint is None:
      raise ValueError("A PostgreSQL embedding model endpoint is required.")
    embedding_query = _generate_postgresql_for_embedding_query(
        spanner_pg_vertex_ai_embedding_model_endpoint,
        output_dimensionality,
    )
    params = {f"p{_POSTGRESQL_PARAMETER_TEXT_QUERY}": query}
  else:
    if spanner_gsql_embedding_model_name is None:
      raise ValueError("A GoogleSQL embedding model name is required.")
    embedding_query = _generate_googlesql_for_embedding_query(
        spanner_gsql_embedding_model_name
    )
    params = {_GOOGLESQL_PARAMETER_TEXT_QUERY: query}
  with database.snapshot() as snapshot:
    result_set = snapshot.execute_sql(embedding_query, params=params)
    return cast("list[float]", result_set.one()[0])


def _get_postgresql_distance_function(distance_type: str) -> str:
  return {
      "COSINE": "spanner.cosine_distance",
      "EUCLIDEAN": "spanner.euclidean_distance",
      "DOT_PRODUCT": "spanner.dot_product",
  }[distance_type]


def _get_googlesql_distance_function(distance_type: str, ann: bool) -> str:
  if ann:
    return {
        "COSINE": "APPROX_COSINE_DISTANCE",
        "EUCLIDEAN": "APPROX_EUCLIDEAN_DISTANCE",
        "DOT_PRODUCT": "APPROX_DOT_PRODUCT",
    }[distance_type]
  return {
      "COSINE": "COSINE_DISTANCE",
      "EUCLIDEAN": "EUCLIDEAN_DISTANCE",
      "DOT_PRODUCT": "DOT_PRODUCT",
  }[distance_type]


def _generate_sql_for_knn(
    dialect: DatabaseDialect,
    table_name: str,
    embedding_column_to_search: str,
    columns: Sequence[str],
    additional_filter: str | None,
    distance_type: str,
    top_k: int,
) -> str:
  """Generates a SQL query for kNN search."""
  top_k = int(top_k)
  if dialect == DatabaseDialect.POSTGRESQL:
    distance_function = _get_postgresql_distance_function(distance_type)
    embedding_parameter = f"${_POSTGRESQL_PARAMETER_QUERY_EMBEDDING}"
  else:
    distance_function = _get_googlesql_distance_function(
        distance_type, ann=False
    )
    embedding_parameter = f"@{_GOOGLESQL_PARAMETER_QUERY_EMBEDDING}"
  selected_columns = [
      *columns,
      f"""{distance_function}(
      {embedding_column_to_search},
      {embedding_parameter}) AS {_DISTANCE_ALIAS}
  """,
  ]
  columns_sql = ", ".join(selected_columns)
  if additional_filter is None:
    additional_filter = "1=1"

  optional_limit_clause = ""
  if top_k > 0:
    optional_limit_clause = f"""LIMIT {top_k}"""
  return f"""
    SELECT {columns_sql}
    FROM {table_name}
    WHERE {additional_filter}
    ORDER BY {_DISTANCE_ALIAS}
    {optional_limit_clause}
  """


def _generate_sql_for_ann(
    dialect: DatabaseDialect,
    table_name: str,
    embedding_column_to_search: str,
    columns: Sequence[str],
    additional_filter: str | None,
    distance_type: str,
    top_k: int,
    num_leaves_to_search: int,
) -> str:
  """Generates a SQL query for ANN search."""
  top_k = int(top_k)
  num_leaves_to_search = int(num_leaves_to_search)
  if dialect == DatabaseDialect.POSTGRESQL:
    raise NotImplementedError(
        f"{APPROXIMATE_NEAREST_NEIGHBORS} is not supported for PostgreSQL"
        " dialect."
    )
  distance_function = _get_googlesql_distance_function(distance_type, ann=True)
  selected_columns = [
      *columns,
      f"""{distance_function}(
      {embedding_column_to_search},
      @{_GOOGLESQL_PARAMETER_QUERY_EMBEDDING},
      options => JSON '{{"num_leaves_to_search": {num_leaves_to_search}}}'
  ) AS {_DISTANCE_ALIAS}
  """,
  ]
  columns_sql = ", ".join(selected_columns)
  query_filter = f"{embedding_column_to_search} IS NOT NULL"
  if additional_filter is not None:
    query_filter = f"{query_filter} AND {additional_filter}"

  return f"""
    SELECT {columns_sql}
    FROM {table_name}
    WHERE {query_filter}
    ORDER BY {_DISTANCE_ALIAS}
    LIMIT {top_k}
  """


async def similarity_search(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_name: str,
    query: str,
    embedding_column_to_search: str,
    columns: List[str],
    embedding_options: Dict[str, str],
    credentials: Credentials,
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
        the embedding service. **Exactly one of the following three keys
        MUST be present in this dictionary**:
        `vertex_ai_embedding_model_name`, `spanner_googlesql_embedding_model_name`,
        or `spanner_postgresql_vertex_ai_embedding_model_endpoint`.
        - vertex_ai_embedding_model_name (str): (Supported both **GoogleSQL and
            PostgreSQL** dialects Spanner database) The name of a
            public Vertex AI embedding model (e.g., `'text-embedding-005'`).
            If specified, the tool generates embeddings client-side using the
            Vertex AI embedding model.
        - spanner_googlesql_embedding_model_name (str): (For GoogleSQL dialect) The
          name of the embedding model that is registered in Spanner via a
          `CREATE MODEL` statement. For more details, see
          https://cloud.google.com/spanner/docs/ml-tutorial-embeddings#generate_and_store_text_embeddings
          If specified, embedding generation is performed using Spanner's
          `ML.PREDICT` function.
        - spanner_postgresql_vertex_ai_embedding_model_endpoint (str):
          (For PostgreSQL dialect) The fully qualified endpoint of the Vertex AI
          embedding model, in the format of
          `projects/$project/locations/$location/publishers/google/models/$model_name`,
          where $project is the project hosting the Vertex AI endpoint,
          $location is the location of the endpoint, and $model_name is
          the name of the text embedding model.
          If specified, embedding generation is performed using Spanner's
          `spanner.ML_PREDICT_ROW` function.
        - output_dimensionality: Optional. An integer. The output
          dimensionality of the embedding. If not specified, the embedding
          model's default output dimensionality will be used.
      credentials (Credentials): The credentials to use for the request.
      additional_filter (Optional[str]): An optional filter to apply to the
        search query. If provided, this will be added to the WHERE clause of the
        final query. Only simple filters are allowed. Supported grammar:
        - Columns and values compared with: =, !=, <, >, <=, >=, LIKE, IS, IS NOT
        - Set membership: IN, NOT IN (e.g., col IN (val1, val2))
        - Range checks: BETWEEN ... AND ... (e.g., col BETWEEN val1 AND val2)
        - Boolean columns (e.g., col_name) or dummy filter '1=1'
        - Logical operators: AND, OR (case-insensitive)
        - Parentheses: up to 2 levels of nesting (e.g., (col1 = val1 OR col2 = val2) AND col3 = val3)
        Values must be numbers, single-quoted strings (without backslashes), booleans, or NULL.
      search_options (Optional[Dict[str, Any]]): A dictionary of options to use
        for the similarity search. The following options are supported:
        - top_k: An integer. The number of most similar documents to return.
          The default value is 4.
        - distance_type: The distance type to use to perform the
          similarity search. Valid values include "COSINE",
          "EUCLIDEAN", and "DOT_PRODUCT". Default value is
          "COSINE".
        - nearest_neighbors_algorithm: The nearest neighbors search
          algorithm to use. Valid values include "EXACT_NEAREST_NEIGHBORS"
          and "APPROXIMATE_NEAREST_NEIGHBORS". Default value is
          "EXACT_NEAREST_NEIGHBORS".
        - num_leaves_to_search: An integer. (Only applies when the
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
        ...   table_name="my_product_table",
        ...   query="Tools that can help me clean my house.",
        ...   embedding_column_to_search="product_description_embedding",
        ...   columns=["product_name", "product_description", "price_in_cents"],
        ...   credentials=credentials,
        ...   additional_filter="price_in_cents < 100000",
        ...   embedding_options={
        ...     "vertex_ai_embedding_model_name": "text-embedding-005"
        ...   },
        ...   search_options={
        ...     "top_k": 2,
        ...     "distance_type": "COSINE"
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
    # Validate input arguments to prevent SQL injection
    _validate_identifier(table_name, "table_name")
    _validate_identifier(
        embedding_column_to_search, "embedding_column_to_search"
    )
    _validate_column_list(columns, "columns")
    if additional_filter:
      _validate_additional_filter(additional_filter)

    opts = embedding_options or {}
    gsql_model = opts.get(_SPANNER_GSQL_EMBEDDING_MODEL_NAME)
    if gsql_model:
      _validate_identifier(gsql_model, _SPANNER_GSQL_EMBEDDING_MODEL_NAME)

    pg_endpoint = opts.get(_SPANNER_PG_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT)
    if pg_endpoint:
      if not re.match(
          r"^projects/[\w-]+/locations/[\w-]+/publishers/[\w-]+/models/[\w.-]+$",
          pg_endpoint,
      ):
        raise ValueError(
            "Invalid Vertex AI endpoint format: "
            f"{pg_endpoint!r}. Expected format: "
            "projects/$project/locations/$location/publishers/google/models/$model"
        )

    return await _similarity_search_internal(
        project_id=project_id,
        instance_id=instance_id,
        database_id=database_id,
        table_name=table_name,
        query=query,
        embedding_column_to_search=embedding_column_to_search,
        columns=columns,
        embedding_options=embedding_options,
        credentials=credentials,
        additional_filter=additional_filter,
        search_options=search_options,
    )
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


async def _similarity_search_internal(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_name: str,
    query: str,
    embedding_column_to_search: str,
    columns: List[str],
    embedding_options: Dict[str, str],
    credentials: Credentials,
    additional_filter: Optional[str] = None,
    search_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  spanner_client: spanner.Client | None = None
  database: Database | None = None
  try:

    # Get Spanner client
    spanner_client = client._get_typed_spanner_client(
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

    embedding_options = embedding_options or {}
    search_options = search_options or {}

    exclusive_embedding_model_keys = {
        _VERTEX_AI_EMBEDDING_MODEL_NAME,
        _SPANNER_GSQL_EMBEDDING_MODEL_NAME,
        _SPANNER_PG_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT,
    }
    if (
        len(
            exclusive_embedding_model_keys.intersection(
                embedding_options.keys()
            )
        )
        != 1
    ):
      raise ValueError("Exactly one embedding model option must be specified.")

    vertex_ai_embedding_model_name = _optional_str_option(
        embedding_options, _VERTEX_AI_EMBEDDING_MODEL_NAME
    )
    spanner_gsql_embedding_model_name = _optional_str_option(
        embedding_options, _SPANNER_GSQL_EMBEDDING_MODEL_NAME
    )
    spanner_pg_vertex_ai_embedding_model_endpoint = _optional_str_option(
        embedding_options, _SPANNER_PG_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT
    )
    if (
        database.database_dialect == DatabaseDialect.GOOGLE_STANDARD_SQL
        and vertex_ai_embedding_model_name is None
        and spanner_gsql_embedding_model_name is None
    ):
      raise ValueError(
          f"embedding_options['{_VERTEX_AI_EMBEDDING_MODEL_NAME}'] or"
          f" embedding_options['{_SPANNER_GSQL_EMBEDDING_MODEL_NAME}'] must be"
          " specified for GoogleSQL dialect Spanner database."
      )
    if (
        database.database_dialect == DatabaseDialect.POSTGRESQL
        and vertex_ai_embedding_model_name is None
        and spanner_pg_vertex_ai_embedding_model_endpoint is None
    ):
      raise ValueError(
          f"embedding_options['{_VERTEX_AI_EMBEDDING_MODEL_NAME}'] or"
          f" embedding_options['{_SPANNER_PG_VERTEX_AI_EMBEDDING_MODEL_ENDPOINT}']"
          " must be specified for PostgreSQL dialect Spanner database."
      )
    output_dimensionality = _optional_int_option(
        embedding_options, _OUTPUT_DIMENSIONALITY
    )
    if (
        output_dimensionality is not None
        and spanner_gsql_embedding_model_name is not None
    ):
      # Currently, Spanner GSQL Model ML.PREDICT does not support
      # output_dimensionality parameter for inference embedding models.
      raise ValueError(
          f"embedding_options[{_OUTPUT_DIMENSIONALITY}] is not supported when"
          f" embedding_options['{_SPANNER_GSQL_EMBEDDING_MODEL_NAME}'] is"
          " specified."
      )

    # Use cosine distance by default.
    distance_type = (
        _optional_str_option(search_options, _DISTANCE_TYPE) or "COSINE"
    )

    top_k = _optional_int_option(search_options, _TOP_K)
    if top_k is None:
      top_k = 4

    # Use EXACT_NEAREST_NEIGHBORS (i.e. kNN) by default.
    nearest_neighbors_algorithm = (
        _optional_str_option(search_options, _NEAREST_NEIGHBORS_ALGORITHM)
        or EXACT_NEAREST_NEIGHBORS
    )
    if nearest_neighbors_algorithm not in (
        EXACT_NEAREST_NEIGHBORS,
        APPROXIMATE_NEAREST_NEIGHBORS,
    ):
      raise NotImplementedError(
          f"Unsupported search_options['{_NEAREST_NEIGHBORS_ALGORITHM}']:"
          f" {nearest_neighbors_algorithm}"
      )

    # Generate embedding for the query according to the embedding options.
    if vertex_ai_embedding_model_name:
      embedding = (
          await utils.embed_contents_async(
              vertex_ai_embedding_model_name,
              [query],
              output_dimensionality,
          )
      )[0]
    else:
      embedding = await asyncio.to_thread(
          _get_embedding_for_query,
          database,
          database.database_dialect,
          spanner_gsql_embedding_model_name,
          spanner_pg_vertex_ai_embedding_model_endpoint,
          query,
          output_dimensionality,
      )

    if nearest_neighbors_algorithm == EXACT_NEAREST_NEIGHBORS:
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
      num_leaves_to_search = _optional_int_option(
          search_options, _NUM_LEAVES_TO_SEARCH
      )
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

    def _execute_sql() -> dict[str, Any]:
      with database.snapshot() as snapshot:
        result_set = snapshot.execute_sql(sql, params=params)
        rows: list[object] = []
        for row in result_set:
          try:
            # If the json serialization of the row succeeds, use it as is
            json.dumps(row)
          except (TypeError, ValueError, OverflowError):
            row = str(row)
          rows.append(row)
        return {"status": "SUCCESS", "rows": rows}

    return await asyncio.to_thread(_execute_sql)
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }
  finally:
    if spanner_client is not None:
      # Shielded so cancelling the search still runs the cleanup to completion.
      await asyncio.shield(
          asyncio.to_thread(
              client._close_spanner_resources, spanner_client, database
          )
      )


async def vector_store_similarity_search(
    query: str,
    credentials: Credentials,
    settings: SpannerToolSettings,
) -> Dict[str, Any]:
  """Performs a semantic similarity search to retrieve relevant context from the Spanner vector store.

  This function performs vector similarity search directly on a vector store
  table in Spanner database and returns the relevant data.

  Args:
      query (str): The search string based on the user's question.
      credentials (Credentials): The credentials to use for the request.
      settings (SpannerToolSettings): The configuration for the tool.

  Returns:
      Dict[str, Any]: A dictionary representing the result of the search.
        On success, it contains {"status": "SUCCESS", "rows": [...]}. The last
        column of each row is the distance between the query and the row result.
        On error, it contains {"status": "ERROR", "error_details": "..."}.

  Examples:
        >>> vector_store_similarity_search(
        ...   query="Spanner database optimization techniques for high QPS",
        ...   credentials=credentials,
        ...   settings=settings
        ... )
        {
          "status": "SUCCESS",
          "rows": [
            (
              "Optimizing Query Performance",
              0.12,
            ),
            (
              "Schema Design Best Practices",
              0.25,
            ),
            (
              "Using Secondary Indexes Effectively",
              0.31,
            ),
            ...
          ],
        }
  """

  try:
    if not settings or not settings.vector_store_settings:
      raise ValueError("Spanner vector store settings are not set.")

    # Get the embedding model settings. The output dimensionality is an
    # integer, so this is wider than the string values `similarity_search`
    # declares; it is passed as declared rather than reshaped.
    embedding_options: Dict[str, Any] = {
        _VERTEX_AI_EMBEDDING_MODEL_NAME: (
            settings.vector_store_settings.vertex_ai_embedding_model_name
        ),
        _OUTPUT_DIMENSIONALITY: settings.vector_store_settings.vector_length,
    }

    # Get the search settings.
    search_options: Dict[str, Any] = {
        _TOP_K: settings.vector_store_settings.top_k,
        _DISTANCE_TYPE: settings.vector_store_settings.distance_type,
        _NEAREST_NEIGHBORS_ALGORITHM: (
            settings.vector_store_settings.nearest_neighbors_algorithm
        ),
    }
    if (
        settings.vector_store_settings.nearest_neighbors_algorithm
        == APPROXIMATE_NEAREST_NEIGHBORS
    ):
      search_options[_NUM_LEAVES_TO_SEARCH] = (
          settings.vector_store_settings.num_leaves_to_search
      )

    return await _similarity_search_internal(
        project_id=settings.vector_store_settings.project_id,
        instance_id=settings.vector_store_settings.instance_id,
        database_id=settings.vector_store_settings.database_id,
        table_name=settings.vector_store_settings.table_name,
        query=query,
        embedding_column_to_search=settings.vector_store_settings.embedding_column,
        columns=settings.vector_store_settings.selected_columns,
        embedding_options=embedding_options,
        credentials=credentials,
        additional_filter=settings.vector_store_settings.additional_filter,
        search_options=search_options,
    )
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }

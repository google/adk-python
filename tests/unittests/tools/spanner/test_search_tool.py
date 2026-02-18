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

from unittest import mock
from unittest.mock import MagicMock

from google.adk.tools.spanner import client
from google.adk.tools.spanner import search_tool
from google.adk.tools.spanner import utils
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
import pytest


@pytest.fixture
def mock_credentials():
  return MagicMock()


@pytest.fixture
def mock_spanner_ids():
  return {
      "project_id": "test-project",
      "instance_id": "test-instance",
      "database_id": "test-database",
      "table_name": "test-table",
  }


@pytest.mark.parametrize(
    ("embedding_option_key", "embedding_option_value", "expected_embedding"),
    [
        pytest.param(
            "spanner_googlesql_embedding_model_name",
            "EmbeddingsModel",
            [0.1, 0.2, 0.3],
            id="spanner_googlesql_embedding_model",
        ),
        pytest.param(
            "vertex_ai_embedding_model_name",
            "text-embedding-005",
            [0.4, 0.5, 0.6],
            id="vertex_ai_embedding_model",
        ),
    ],
)
@mock.patch.object(utils, "embed_contents")
@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_knn_success(
    mock_get_spanner_client,
    mock_embed_contents,
    mock_spanner_ids,
    mock_credentials,
    embedding_option_key,
    embedding_option_value,
    expected_embedding,
):
  """Test similarity_search function with kNN success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  if embedding_option_key == "vertex_ai_embedding_model_name":
    mock_embed_contents.return_value = [expected_embedding]
    # execute_sql is called once for the kNN search
    mock_snapshot.execute_sql.return_value = iter([("result1",), ("result2",)])
  else:
    mock_embedding_result = MagicMock()
    mock_embedding_result.one.return_value = (expected_embedding,)
    # First call to execute_sql is for getting the embedding,
    # second call is for the kNN search
    mock_snapshot.execute_sql.side_effect = [
        mock_embedding_result,
        iter([("result1",), ("result2",)]),
    ]

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={embedding_option_key: embedding_option_value},
      credentials=mock_credentials,
  )
  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [("result1",), ("result2",)]

  # Check the generated SQL for kNN search
  call_args = mock_snapshot.execute_sql.call_args
  sql = call_args.args[0]
  assert "COSINE_DISTANCE" in sql
  assert "@embedding" in sql
  assert call_args.kwargs == {"params": {"embedding": expected_embedding}}
  if embedding_option_key == "vertex_ai_embedding_model_name":
    mock_embed_contents.assert_called_once_with(
        embedding_option_value, ["test query"], None
    )


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_ann_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search function with ANN success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_embedding_result = MagicMock()
  mock_embedding_result.one.return_value = ([0.1, 0.2, 0.3],)
  # First call to execute_sql is for getting the embedding
  # Second call is for the ANN search
  mock_snapshot.execute_sql.side_effect = [
      mock_embedding_result,
      iter([("ann_result1",), ("ann_result2",)]),
  ]
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_googlesql_embedding_model_name": "test_model"
      },
      credentials=mock_credentials,
      search_options={
          "nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS"
      },
  )
  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [("ann_result1",), ("ann_result2",)]
  call_args = mock_snapshot.execute_sql.call_args
  sql = call_args.args[0]
  assert "APPROX_COSINE_DISTANCE" in sql
  assert "@embedding" in sql
  assert call_args.kwargs == {"params": {"embedding": [0.1, 0.2, 0.3]}}


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search function with a generic error."""
  mock_get_spanner_client.side_effect = Exception("Test Exception")
  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      embedding_options={
          "spanner_googlesql_embedding_model_name": "test_model"
      },
      columns=["col1"],
      credentials=mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert "Test Exception" in result["error_details"]


@mock.patch.object(utils, "embed_contents")
@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_circular_row_fallback_to_string(
    mock_get_spanner_client,
    mock_embed_contents,
    mock_spanner_ids,
    mock_credentials,
):
  """Test similarity_search stringifies rows with circular references."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  circular_row = []
  circular_row.append(circular_row)
  mock_embed_contents.return_value = [[0.1, 0.2, 0.3]]
  mock_snapshot.execute_sql.return_value = iter([circular_row])
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "vertex_ai_embedding_model_name": "text-embedding-005"
      },
      credentials=mock_credentials,
  )

  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [str(circular_row)]


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_postgresql_knn_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with PostgreSQL dialect for kNN."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_embedding_result = MagicMock()
  mock_embedding_result.one.return_value = ([0.1, 0.2, 0.3],)
  mock_snapshot.execute_sql.side_effect = [
      mock_embedding_result,
      iter([("pg_result",)]),
  ]
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.POSTGRESQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
              "test_endpoint"
          )
      },
      credentials=mock_credentials,
  )
  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [("pg_result",)]
  call_args = mock_snapshot.execute_sql.call_args
  sql = call_args.args[0]
  assert "spanner.cosine_distance" in sql
  assert "$1" in sql
  assert call_args.kwargs == {"params": {"p1": [0.1, 0.2, 0.3]}}


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_postgresql_ann_unsupported(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with unsupported ANN for PostgreSQL dialect."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.POSTGRESQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
              "test_endpoint"
          )
      },
      credentials=mock_credentials,
      search_options={
          "nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS"
      },
  )
  assert result["status"] == "ERROR"
  assert (
      "APPROXIMATE_NEAREST_NEIGHBORS is not supported for PostgreSQL dialect."
      in result["error_details"]
  )


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_gsql_missing_embedding_model_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with missing embedding_options for GoogleSQL dialect."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
              "test_endpoint"
          )
      },
      credentials=mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert (
      "embedding_options['vertex_ai_embedding_model_name'] or"
      " embedding_options['spanner_googlesql_embedding_model_name'] must be"
      " specified for GoogleSQL dialect Spanner database."
      in result["error_details"]
  )


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_pg_missing_embedding_model_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with missing embedding_options for PostgreSQL dialect."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.POSTGRESQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_googlesql_embedding_model_name": "EmbeddingsModel"
      },
      credentials=mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert (
      "embedding_options['vertex_ai_embedding_model_name'] or"
      " embedding_options['spanner_postgresql_vertex_ai_embedding_model_endpoint']"
      " must be specified for PostgreSQL dialect Spanner database."
      in result["error_details"]
  )


@pytest.mark.parametrize(
    "embedding_options",
    [
        pytest.param(
            {
                "vertex_ai_embedding_model_name": "test-model",
                "spanner_googlesql_embedding_model_name": "test-model-2",
            },
            id="vertex_ai_and_googlesql",
        ),
        pytest.param(
            {
                "vertex_ai_embedding_model_name": "test-model",
                "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
                    "test-endpoint"
                ),
            },
            id="vertex_ai_and_postgresql",
        ),
        pytest.param(
            {
                "spanner_googlesql_embedding_model_name": "test-model",
                "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
                    "test-endpoint"
                ),
            },
            id="googlesql_and_postgresql",
        ),
        pytest.param(
            {
                "vertex_ai_embedding_model_name": "test-model",
                "spanner_googlesql_embedding_model_name": "test-model-2",
                "spanner_postgresql_vertex_ai_embedding_model_endpoint": (
                    "test-endpoint"
                ),
            },
            id="all_three_models",
        ),
        pytest.param(
            {},
            id="no_models",
        ),
    ],
)
@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_multiple_embedding_options_error(
    mock_get_spanner_client,
    mock_spanner_ids,
    mock_credentials,
    embedding_options,
):
  """Test similarity_search with multiple embedding models."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options=embedding_options,
      credentials=mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert (
      "Exactly one embedding model option must be specified."
      in result["error_details"]
  )


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_output_dimensionality_gsql_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with output_dimensionality and spanner_googlesql_embedding_model_name."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={
          "spanner_googlesql_embedding_model_name": "EmbeddingsModel",
          "output_dimensionality": 128,
      },
      credentials=mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert "is not supported when" in result["error_details"]


@mock.patch.object(client, "get_spanner_client")
def test_similarity_search_unsupported_algorithm_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with an unsupported nearest neighbors algorithm."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = search_tool.similarity_search(
      project_id=mock_spanner_ids["project_id"],
      instance_id=mock_spanner_ids["instance_id"],
      database_id=mock_spanner_ids["database_id"],
      table_name=mock_spanner_ids["table_name"],
      query="test query",
      embedding_column_to_search="embedding_col",
      columns=["col1"],
      embedding_options={"vertex_ai_embedding_model_name": "test-model"},
      credentials=mock_credentials,
      search_options={"nearest_neighbors_algorithm": "INVALID_ALGORITHM"},
  )
  assert result["status"] == "ERROR"
  assert "Unsupported search_options" in result["error_details"]

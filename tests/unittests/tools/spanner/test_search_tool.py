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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.spanner import search_tool
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


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_similarity_search_knn_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search function with kNN success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_embedding_result = MagicMock()
  mock_embedding_result.one.return_value = ([0.1, 0.2, 0.3],)
  # First call to execute_sql is for getting the embedding
  # Second call is for the kNN search
  mock_snapshot.execute_sql.side_effect = [
      mock_embedding_result,
      iter([("result1",), ("result2",)]),
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
      embedding_options={"spanner_embedding_model_name": "test_model"},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
  )
  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [("result1",), ("result2",)]

  # Check the generated SQL for kNN search
  call_args = mock_snapshot.execute_sql.call_args
  sql = call_args.args[0]
  assert "COSINE_DISTANCE" in sql
  assert "@embedding" in sql
  assert call_args.kwargs == {"params": {"embedding": [0.1, 0.2, 0.3]}}


@patch("google.adk.tools.spanner.client.get_spanner_client")
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
      embedding_options={"spanner_embedding_model_name": "test_model"},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
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


@patch("google.adk.tools.spanner.client.get_spanner_client")
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
      embedding_options={"spanner_embedding_model_name": "test_model"},
      columns=["col1"],
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
  )
  assert result["status"] == "ERROR"
  assert result["error_details"] == "Test Exception"


@patch("google.adk.tools.spanner.client.get_spanner_client")
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
      embedding_options={"vertex_ai_embedding_model_endpoint": "test_endpoint"},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
  )
  assert result["status"] == "SUCCESS", result
  assert result["rows"] == [("pg_result",)]
  call_args = mock_snapshot.execute_sql.call_args
  sql = call_args.args[0]
  assert "spanner.cosine_distance" in sql
  assert "$1" in sql
  assert call_args.kwargs == {"params": {"p1": [0.1, 0.2, 0.3]}}


@patch("google.adk.tools.spanner.client.get_spanner_client")
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
      embedding_options={"vertex_ai_embedding_model_endpoint": "test_endpoint"},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
      search_options={
          "nearest_neighbors_algorithm": "APPROXIMATE_NEAREST_NEIGHBORS"
      },
  )
  assert result["status"] == "ERROR"
  assert (
      result["error_details"]
      == "APPROXIMATE_NEAREST_NEIGHBORS is not supported for PostgreSQL"
      " dialect."
  )


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_similarity_search_missing_spanner_embedding_model_name_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with missing spanner_embedding_model_name."""
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
      embedding_options={},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
  )
  assert result["status"] == "ERROR"
  assert (
      "embedding_options['spanner_embedding_model_name'] must be"
      " specified for GoogleSQL dialect."
      in result["error_details"]
  )


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_similarity_search_missing_vertex_ai_embedding_model_endpoint_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test similarity_search with missing vertex_ai_embedding_model_endpoint."""
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
      embedding_options={},
      credentials=mock_credentials,
      settings=MagicMock(),
      tool_context=MagicMock(),
  )
  assert result["status"] == "ERROR"
  assert (
      "embedding_options['vertex_ai_embedding_model_endpoint'] must "
      "be specified for PostgreSQL dialect."
      in result["error_details"]
  )

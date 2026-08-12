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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.spanner import metadata_tool
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
def test_list_table_names_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_table_names function with success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_table = MagicMock()
  mock_table.table_id = "table1"
  mock_database.list_tables.return_value = [mock_table]
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.list_table_names(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_credentials,
  )
  assert result["status"] == "SUCCESS"
  assert result["results"] == ["table1"]


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_list_table_names_error(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_table_names function with error."""
  mock_get_spanner_client.side_effect = Exception("Test Exception")
  result = metadata_tool.list_table_names(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_credentials,
  )
  assert result["status"] == "ERROR"
  assert result["error_details"] == "Test Exception"


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_get_table_schema_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test get_table_schema function with success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()

  mock_columns_result = [(
      "col1",  # COLUMN_NAME
      "",  # TABLE_SCHEMA
      "STRING(MAX)",  # SPANNER_TYPE
      1,  # ORDINAL_POSITION
      None,  # COLUMN_DEFAULT
      "NO",  # IS_NULLABLE
      "NEVER",  # IS_GENERATED
      None,  # GENERATION_EXPRESSION
      None,  # IS_STORED
  )]

  mock_key_columns_result = [(
      "col1",  # COLUMN_NAME
      "PK_Table",  # CONSTRAINT_NAME
      1,  # ORDINAL_POSITION
      None,  # POSITION_IN_UNIQUE_CONSTRAINT
  )]

  mock_table_metadata_result = [(
      "",  # TABLE_SCHEMA
      "test_table",  # TABLE_NAME
      "BASE TABLE",  # TABLE_TYPE
      None,  # PARENT_TABLE_NAME
      None,  # ON_DELETE_ACTION
      "COMMITTED",  # SPANNER_STATE
      None,  # INTERLEAVE_TYPE
      "OLDER_THAN(CreatedAt, INTERVAL 1 DAY)",  # ROW_DELETION_POLICY_EXPRESSION
  )]

  mock_snapshot.execute_sql.side_effect = [
      mock_columns_result,
      mock_key_columns_result,
      mock_table_metadata_result,
  ]

  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.get_table_schema(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_spanner_ids["table_name"],
      mock_credentials,
  )

  assert result["status"] == "SUCCESS"
  assert "col1" in result["results"]["schema"]
  assert result["results"]["schema"]["col1"]["SPANNER_TYPE"] == "STRING(MAX)"
  assert "KEY_COLUMN_USAGE" in result["results"]["schema"]["col1"]
  assert (
      result["results"]["schema"]["col1"]["KEY_COLUMN_USAGE"][0][
          "CONSTRAINT_NAME"
      ]
      == "PK_Table"
  )
  assert "metadata" in result["results"]
  assert result["results"]["metadata"][0]["TABLE_NAME"] == "test_table"
  assert (
      result["results"]["metadata"][0]["ROW_DELETION_POLICY_EXPRESSION"]
      == "OLDER_THAN(CreatedAt, INTERVAL 1 DAY)"
  )


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_list_table_indexes_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_table_indexes function with success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_result_set = MagicMock()
  mock_result_set.__iter__.return_value = iter([(
      "PRIMARY_KEY",
      "",
      "PRIMARY_KEY",
      "",
      True,
      False,
      None,
  )])
  mock_snapshot.execute_sql.return_value = mock_result_set
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.list_table_indexes(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_spanner_ids["table_name"],
      mock_credentials,
  )
  assert result["status"] == "SUCCESS"
  assert len(result["results"]) == 1
  assert result["results"][0]["INDEX_NAME"] == "PRIMARY_KEY"


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_list_table_indexes_circular_row_fallback_to_string(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_table_indexes stringifies rows with circular references."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  circular_value = []
  circular_value.append(circular_value)
  mock_result_set = MagicMock()
  mock_result_set.__iter__.return_value = iter([(
      circular_value,
      "",
      "PRIMARY_KEY",
      "",
      True,
      False,
      None,
  )])
  mock_snapshot.execute_sql.return_value = mock_result_set
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.list_table_indexes(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_spanner_ids["table_name"],
      mock_credentials,
  )
  assert result["status"] == "SUCCESS"
  assert isinstance(result["results"][0], str)


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_list_table_index_columns_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_table_index_columns function with success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_result_set = MagicMock()
  mock_result_set.__iter__.return_value = iter([(
      "PRIMARY_KEY",
      "",
      "col1",
      1,
      "NO",
      "STRING(MAX)",
  )])
  mock_snapshot.execute_sql.return_value = mock_result_set
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.list_table_index_columns(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_spanner_ids["table_name"],
      mock_credentials,
  )
  assert result["status"] == "SUCCESS"
  assert len(result["results"]) == 1
  assert result["results"][0]["COLUMN_NAME"] == "col1"


@patch("google.adk.tools.spanner.client.get_spanner_client")
def test_list_named_schemas_success(
    mock_get_spanner_client, mock_spanner_ids, mock_credentials
):
  """Test list_named_schemas function with success."""
  mock_spanner_client = MagicMock()
  mock_instance = MagicMock()
  mock_database = MagicMock()
  mock_snapshot = MagicMock()
  mock_result_set = MagicMock()
  mock_result_set.__iter__.return_value = iter([("schema1",), ("schema2",)])
  mock_snapshot.execute_sql.return_value = mock_result_set
  mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
  mock_database.database_dialect = DatabaseDialect.GOOGLE_STANDARD_SQL
  mock_instance.database.return_value = mock_database
  mock_spanner_client.instance.return_value = mock_instance
  mock_get_spanner_client.return_value = mock_spanner_client

  result = metadata_tool.list_named_schemas(
      mock_spanner_ids["project_id"],
      mock_spanner_ids["instance_id"],
      mock_spanner_ids["database_id"],
      mock_credentials,
  )
  assert result["status"] == "SUCCESS"
  assert result["results"] == ["schema1", "schema2"]

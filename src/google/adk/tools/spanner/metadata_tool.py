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

from google.auth.credentials import Credentials
from google.cloud.spanner_admin_database_v1.types import DatabaseDialect
from google.cloud.spanner_v1 import param_types as spanner_param_types

from . import client


def list_table_names(
    project_id: str,
    instance_id: str,
    database_id: str,
    credentials: Credentials,
    named_schema: str = "",
) -> dict:
  """List tables within the database.

  Args:
      project_id (str): The Google Cloud project id.
      instance_id (str): The Spanner instance id.
      database_id (str): The Spanner database id.
      credentials (Credentials): The credentials to use for the request.
      named_schema (str): The named schema to list tables in. Default is empty
        string "" to search for tables in the default schema of the database.

  Returns:
      dict: Dictionary with a list of the Spanner table names.

  Examples:
      >>> list_tables("my_project", "my_instance", "my_database")
      {
        "status": "SUCCESS",
        "results": [
          "table_1",
          "table_2"
        ]
      }
  """
  try:
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    tables = []
    named_schema = named_schema if named_schema else "_default"
    for table in database.list_tables(schema=named_schema):
      tables.append(table.table_id)

    return {"status": "SUCCESS", "results": tables}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_table_schema(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_name: str,
    credentials: Credentials,
    named_schema: str = "",
) -> dict:
  """Get schema and metadata information about a Spanner table.

  Args:
      project_id (str): The Google Cloud project id.
      instance_id (str): The Spanner instance id.
      database_id (str): The Spanner database id.
      table_id (str): The Spanner table id.
      credentials (Credentials): The credentials to use for the request.
      named_schema (str): The named schema to list tables in. Default is empty
        string "" to search for tables in the default schema of the database.

  Returns:
      dict: Dictionary with the Spanner table schema information.

  Examples:
      >>> get_table_schema("my_project", "my_instance", "my_database",
      ... "my_table")
      {
        "status": "SUCCESS",
        "results":
          {
            "schema":  {
              'colA': {
                'SPANNER_TYPE': 'STRING(1024)',
                'TABLE_SCHEMA': '',
                'ORDINAL_POSITION': 1,
                'COLUMN_DEFAULT': None,
                'IS_NULLABLE': 'NO',
                'IS_GENERATED': 'NEVER',
                'GENERATION_EXPRESSION': None,
                'IS_STORED': None,
                'KEY_COLUMN_USAGE': [
                  # This part is added if it's a key column
                  {
                    'CONSTRAINT_NAME': 'PK_Table1',
                    'ORDINAL_POSITION': 1,
                    'POSITION_IN_UNIQUE_CONSTRAINT': None
                  }
                ]
              },
              'colB': { ... },
              ...
            },
            "metadata": [
              {
                'TABLE_SCHEMA': '',
                'TABLE_NAME': 'MyTable',
                'TABLE_TYPE': 'BASE TABLE',
                'PARENT_TABLE_NAME': NULL,
                'ON_DELETE_ACTION': NULL,
                'SPANNER_STATE': 'COMMITTED',
                'INTERLEAVE_TYPE': NULL,
                'ROW_DELETION_POLICY_EXPRESSION':
                  'OLDER_THAN(CreatedAt, INTERVAL 1 DAY)',
              }
            ]
          }
  """

  columns_query = """
      SELECT
          COLUMN_NAME,
          TABLE_SCHEMA,
          SPANNER_TYPE,
          ORDINAL_POSITION,
          COLUMN_DEFAULT,
          IS_NULLABLE,
          IS_GENERATED,
          GENERATION_EXPRESSION,
          IS_STORED
      FROM
          INFORMATION_SCHEMA.COLUMNS
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema
      ORDER BY
          ORDINAL_POSITION
  """

  key_column_usage_query = """
      SELECT
          COLUMN_NAME,
          CONSTRAINT_NAME,
          ORDINAL_POSITION,
          POSITION_IN_UNIQUE_CONSTRAINT
      FROM
          INFORMATION_SCHEMA.KEY_COLUMN_USAGE
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema
  """
  params = {"table_name": table_name, "named_schema": named_schema}
  param_types = {
      "table_name": spanner_param_types.STRING,
      "named_schema": spanner_param_types.STRING,
  }

  table_metadata_query = """
      SELECT
          TABLE_SCHEMA,
          TABLE_NAME,
          TABLE_TYPE,
          PARENT_TABLE_NAME,
          ON_DELETE_ACTION,
          SPANNER_STATE,
          INTERLEAVE_TYPE,
          ROW_DELETION_POLICY_EXPRESSION
      FROM
          INFORMATION_SCHEMA.TABLES
      WHERE
          TABLE_NAME = @table_name
          AND TABLE_SCHEMA = @named_schema;
  """

  results = {"schema": {}, "metadata": []}
  try:
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      return {
          "status": "ERROR",
          "error_details": "PostgreSQL dialect is not supported",
      }

    with database.snapshot(multi_use=True) as snapshot:
      result_set = snapshot.execute_sql(
          columns_query, params=params, param_types=param_types
      )
      for row in result_set:
        (
            column_name,
            table_schema,
            spanner_type,
            ordinal_position,
            column_default,
            is_nullable,
            is_generated,
            generation_expression,
            is_stored,
        ) = row
        column_metadata = {
            "SPANNER_TYPE": spanner_type,
            "TABLE_SCHEMA": table_schema,
            "ORDINAL_POSITION": ordinal_position,
            "COLUMN_DEFAULT": column_default,
            "IS_NULLABLE": is_nullable,
            "IS_GENERATED": is_generated,
            "GENERATION_EXPRESSION": generation_expression,
            "IS_STORED": is_stored,
        }
        results["schema"][column_name] = column_metadata

      key_column_result_set = snapshot.execute_sql(
          key_column_usage_query, params=params, param_types=param_types
      )
      for row in key_column_result_set:
        (
            column_name,
            constraint_name,
            ordinal_position,
            position_in_unique_constraint,
        ) = row

        key_column_properties = {
            "CONSTRAINT_NAME": constraint_name,
            "ORDINAL_POSITION": ordinal_position,
            "POSITION_IN_UNIQUE_CONSTRAINT": position_in_unique_constraint,
        }
        # Attach key column info to the existing column schema entry
        if column_name in results["schema"]:
          results["schema"][column_name].setdefault(
              "KEY_COLUMN_USAGE", []
          ).append(key_column_properties)

      table_metadata_result_set = snapshot.execute_sql(
          table_metadata_query, params=params, param_types=param_types
      )
      for row in table_metadata_result_set:
        metadata_result = {
            "TABLE_SCHEMA": row[0],
            "TABLE_NAME": row[1],
            "TABLE_TYPE": row[2],
            "PARENT_TABLE_NAME": row[3],
            "ON_DELETE_ACTION": row[4],
            "SPANNER_STATE": row[5],
            "INTERLEAVE_TYPE": row[6],
            "ROW_DELETION_POLICY_EXPRESSION": row[7],
        }
        results["metadata"].append(metadata_result)

    try:
      json.dumps(results)
    except:
      results = str(results)

    return {"status": "SUCCESS", "results": results}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_table_indexes(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_id: str,
    credentials: Credentials,
) -> dict:
  """Get index information about a Spanner table.

  Args:
      project_id (str): The Google Cloud project id.
      instance_id (str): The Spanner instance id.
      database_id (str): The Spanner database id.
      table_id (str): The Spanner table id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of the Spanner table index information.

  Examples:
      >>> list_table_indexes("my_project", "my_instance", "my_database",
      ... "my_table")
      {
        "status": "SUCCESS",
        "results": [
          {
            'INDEX_NAME': 'IDX_MyTable_Column_FC70CD41F3A5FD3A',
            'TABLE_SCHEMA': '',
            'INDEX_TYPE': 'INDEX',
            'PARENT_TABLE_NAME': '',
            'IS_UNIQUE': False,
            'IS_NULL_FILTERED': False,
            'INDEX_STATE': 'READ_WRITE'
          },
          {
            'INDEX_NAME': 'PRIMARY_KEY',
            'TABLE_SCHEMA': '',
            'INDEX_TYPE': 'PRIMARY_KEY',
            'PARENT_TABLE_NAME': '',
            'IS_UNIQUE': True,
            'IS_NULL_FILTERED': False,
            'INDEX_STATE': None
          }
        ]
      }
  """
  try:
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      return {
          "status": "ERROR",
          "error_details": "PostgreSQL dialect is not supported.",
      }

    # Using query parameters is best practice to prevent SQL injection,
    # even if table_id is typically from a controlled source here.
    sql_query = (
        "SELECT INDEX_NAME, TABLE_SCHEMA, INDEX_TYPE,"
        " PARENT_TABLE_NAME, IS_UNIQUE, IS_NULL_FILTERED, INDEX_STATE "
        "FROM INFORMATION_SCHEMA.INDEXES "
        "WHERE TABLE_NAME = @table_id "  # Use query parameter
    )
    params = {"table_id": table_id}
    param_types = {"table_id": spanner_param_types.STRING}

    indexes = []
    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(
          sql_query, params=params, param_types=param_types
      )
      for row in result_set:
        index_info = {}
        index_info["INDEX_NAME"] = row[0]
        index_info["TABLE_SCHEMA"] = row[1]
        index_info["INDEX_TYPE"] = row[2]
        index_info["PARENT_TABLE_NAME"] = row[3]
        index_info["IS_UNIQUE"] = row[4]
        index_info["IS_NULL_FILTERED"] = row[5]
        index_info["INDEX_STATE"] = row[6]

        try:
          json.dumps(index_info)
        except:
          index_info = str(index_info)

        indexes.append(index_info)

    return {"status": "SUCCESS", "results": indexes}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_table_index_columns(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_id: str,
    credentials: Credentials,
) -> dict:
  """Get the columns in each index of a Spanner table.

  Args:
      project_id (str): The Google Cloud project id.
      instance_id (str): The Spanner instance id.
      database_id (str): The Spanner database id.
      table_id (str): The Spanner table id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of Spanner table index column
      information.

  Examples:
      >>> get_table_index_columns("my_project", "my_instance", "my_database",
      ... "my_table")
      {
        "status": "SUCCESS",
        "results": [
          {
            'INDEX_NAME': 'PRIMARY_KEY',
            'TABLE_SCHEMA': '',
            'COLUMN_NAME': 'ColumnKey1',
            'ORDINAL_POSITION': 1,
            'IS_NULLABLE': 'NO',
            'SPANNER_TYPE': 'STRING(MAX)'
          },
          {
            'INDEX_NAME': 'PRIMARY_KEY',
            'TABLE_SCHEMA': '',
            'COLUMN_NAME': 'ColumnKey2',
            'ORDINAL_POSITION': 2,
            'IS_NULLABLE': 'NO',
            'SPANNER_TYPE': 'INT64'
          },
          {
              'INDEX_NAME': 'IDX_MyTable_Column_FC70CD41F3A5FD3A',
              'TABLE_SCHEMA': '',
              'COLUMN_NAME': 'Column',
              'ORDINAL_POSITION': 3,
              'IS_NULLABLE': 'NO',
              'SPANNER_TYPE': 'STRING(MAX)'
          }
        ]
      }
  """
  try:
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      return {
          "status": "ERROR",
          "error_details": "PostgreSQL dialect is not supported.",
      }

    sql_query = (
        "SELECT INDEX_NAME, TABLE_SCHEMA, COLUMN_NAME,"
        " ORDINAL_POSITION, IS_NULLABLE, SPANNER_TYPE "
        "FROM INFORMATION_SCHEMA.INDEX_COLUMNS "
        "WHERE TABLE_NAME = @table_id "  # Use query parameter
    )
    params = {"table_id": table_id}
    param_types = {"table_id": spanner_param_types.STRING}

    index_columns = []
    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(
          sql_query, params=params, param_types=param_types
      )
      for row in result_set:
        index_column_info = {}
        index_column_info["INDEX_NAME"] = row[0]
        index_column_info["TABLE_SCHEMA"] = row[1]
        index_column_info["COLUMN_NAME"] = row[2]
        index_column_info["ORDINAL_POSITION"] = row[3]
        index_column_info["IS_NULLABLE"] = row[4]
        index_column_info["SPANNER_TYPE"] = row[5]

        try:
          json.dumps(index_column_info)
        except:
          index_column_info = str(index_column_info)

        index_columns.append(index_column_info)

    return {"status": "SUCCESS", "results": index_columns}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def list_named_schemas(
    project_id: str,
    instance_id: str,
    database_id: str,
    credentials: Credentials,
) -> dict:
  """Get the named schemas in the Spanner database.

  Args:
      project_id (str): The Google Cloud project id.
      instance_id (str): The Spanner instance id.
      database_id (str): The Spanner database id.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary with a list of named schemas information in the Spanner
      database.

  Examples:
      >>> list_named_schemas("my_project", "my_instance", "my_database")
      {
        "status": "SUCCESS",
        "results": [
            "schema_1",
            "schema_2"
          ]
      }
  """
  try:
    spanner_client = client.get_spanner_client(
        project=project_id, credentials=credentials
    )
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    if database.database_dialect == DatabaseDialect.POSTGRESQL:
      return {
          "status": "ERROR",
          "error_details": "PostgreSQL dialect is not supported.",
      }

    sql_query = """
    SELECT
        SCHEMA_NAME
    FROM
        INFORMATION_SCHEMA.SCHEMATA
    WHERE
        SCHEMA_NAME NOT IN ('', 'INFORMATION_SCHEMA', 'SPANNER_SYS');
    """

    named_schemas = []
    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(sql_query)
      for row in result_set:
        named_schemas.append(row[0])

    return {"status": "SUCCESS", "results": named_schemas}
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

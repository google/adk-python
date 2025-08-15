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

from . import client
from ..tool_context import ToolContext
from .settings import SpannerToolSettings

DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS = 50


def execute_sql(
    project_id: str,
    instance_id: str,
    database_id: str,
    query: str,
    credentials: Credentials,
    settings: SpannerToolSettings,
    tool_context: ToolContext,
) -> dict:
  """Run a Spanner Read-Only query in the spanner database and return the result.

  Args:
      project_id (str): The GCP project id in which the spanner database
        resides.
      instance_id (str): The instance id of the spanner database.
      database_id (str): The database id of the spanner database.
      query (str): The Spanner SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (SpannerToolSettings): The settings for the tool.
      tool_context (ToolContext): The context for the tool.

  Returns:
      dict: Dictionary with the result of the query.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project", "my_instance", "my_database",
          ... "SELECT COUNT(*) AS count FROM my_table")
          {
            "status": "SUCCESS",
            "rows": [
              [100]
            ]
          }

  Note:
    This is running with Read-Only Transaction for query that only read data.
  """

  try:
    # Get Spanner client
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

    with database.snapshot() as snapshot:
      result_set = snapshot.execute_sql(query)
      rows = []
      counter = (
          settings.max_executed_query_result_rows
          if settings and settings.max_executed_query_result_rows > 0
          else DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS
      )
      for row in result_set:
        try:
          # if the json serialization of the row succeeds, use it as is
          json.dumps(row)
        except:
          row = str(row)

        rows.append(row)
        counter -= 1
        if counter <= 0:
          break

      result = {"status": "SUCCESS", "rows": rows}
      if counter <= 0:
        result["result_is_likely_truncated"] = True
      return result
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

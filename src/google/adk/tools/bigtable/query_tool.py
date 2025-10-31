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

"""Tool to execute SQL queries against Bigtable."""
import json
from typing import Any
from typing import Dict
from typing import List

from google.auth.credentials import Credentials
from google.cloud import bigtable

from . import client
from ..tool_context import ToolContext
from .settings import BigtableToolSettings

DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS = 50


def execute_sql(
    project_id: str,
    instance_id: str,
    query: str,
    credentials: Credentials,
    settings: BigtableToolSettings,
    tool_context: ToolContext,
) -> dict:
  """Execute a GoogleSQL query from a Bigtable table.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      instance_id (str): The instance id of the Bigtable database.
      query (str): The Bigtable SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.
      settings (BigtableToolSettings): The configuration for the tool.
      tool_context (ToolContext): The context for the tool.
  Returns:
      dict: Dictionary containing the status and the rows read.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.

  Examples:
      Fetch data or insights from a table:

          >>> execute_sql("my_project", "my_instance",
          ... "SELECT * from mytable", credentials, config, tool_context)
          {
            "status": "SUCCESS",
            "rows": [
                {
                    "user_id": 1,
                    "user_name": "Alice"
                }
            ]
          }
  """
  del tool_context  # Unused for now

  try:
    bt_client = client.get_bigtable_data_client(
        project=project_id, credentials=credentials
    )
    eqi = bt_client.execute_query(
        query=query,
        instance_id=instance_id,
    )

    rows: List[Dict[str, Any]] = []
    max_rows = (
        settings.max_query_result_rows
        if settings and settings.max_query_result_rows > 0
        else DEFAULT_MAX_EXECUTED_QUERY_RESULT_ROWS
    )
    counter = max_rows
    truncated = False
    try:
      for row in eqi:
        if counter <= 0:
          truncated = True
          break
        row_values = {}
        for key, val in dict(row.fields).items():
          try:
            # if the json serialization of the value succeeds, use it as is
            json.dumps(val)
          except:
            val = str(val)
          row_values[key] = val
        rows.append(row_values)
        counter -= 1
    finally:
      eqi.close()

    result = {"status": "SUCCESS", "rows": rows}
    if truncated:
      result["result_is_likely_truncated"] = True
    return result

  except Exception as ex:
    print(ex)
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

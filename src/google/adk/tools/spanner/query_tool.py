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

import functools
import textwrap
import types
from typing import Callable

from google.auth.credentials import Credentials

from . import utils
from ..tool_context import ToolContext
from .settings import QueryResultMode
from .settings import SpannerToolSettings


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
      <Example>
        >>> execute_sql("my_project", "my_instance", "my_database",
        ... "SELECT COUNT(*) AS count FROM my_table")
        {
          "status": "SUCCESS",
          "rows": [
            [100]
          ]
        }
      </Example>

      <Example>
        >>> execute_sql("my_project", "my_instance", "my_database",
        ... "SELECT name, rating, description FROM hotels_table")
        {
          "status": "SUCCESS",
          "rows": [
            ["The Hotel", 4.1, "Modern hotel."],
            ["Park Inn", 4.5, "Cozy hotel."],
            ...
          ]
        }
      </Example>

  Note:
    This is running with Read-Only Transaction for query that only read data.
  """
  return utils.execute_sql(
      project_id,
      instance_id,
      database_id,
      query,
      credentials,
      settings,
      tool_context,
  )


_EXECUTE_SQL_DICT_LIST_MODE_DOCSTRING = textwrap.dedent("""\
Run a Spanner Read-Only query in the spanner database and return the result.

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
    <Example>
      >>> execute_sql("my_project", "my_instance", "my_database",
      ... "SELECT COUNT(*) AS count FROM my_table")
      {
        "status": "SUCCESS",
        "rows": [
          {
            "count": 100
          }
        ]
      }
    </Example>

    <Example>
      >>> execute_sql("my_project", "my_instance", "my_database",
      ... "SELECT COUNT(*) FROM my_table")
      {
        "status": "SUCCESS",
        "rows": [
          {
            "": 100
          }
        ]
      }
    </Example>

    <Example>
      >>> execute_sql("my_project", "my_instance", "my_database",
      ... "SELECT name, rating, description FROM hotels_table")
      {
        "status": "SUCCESS",
        "rows": [
          {
            "name": "The Hotel",
            "rating": 4.1,
            "description": "Modern hotel."
          },
          {
            "name": "Park Inn",
            "rating": 4.5,
            "description": "Cozy hotel."
          },
          ...
        ]
      }
    </Example>

Note:
  This is running with Read-Only Transaction for query that only read data.
""")


def get_execute_sql(settings: SpannerToolSettings) -> Callable[..., dict]:
  """Get the execute_sql tool customized as per the given tool settings.

  Args:
      settings: Spanner tool settings indicating the behavior of the execute_sql
        tool.

  Returns:
      callable[..., dict]: A version of the execute_sql tool respecting the tool
      settings.
  """

  if settings and settings.query_result_mode is QueryResultMode.DICT_LIST:

    execute_sql_wrapper = types.FunctionType(
        execute_sql.__code__,
        execute_sql.__globals__,
        execute_sql.__name__,
        execute_sql.__defaults__,
        execute_sql.__closure__,
    )
    functools.update_wrapper(execute_sql_wrapper, execute_sql)
    # Update with the new docstring
    execute_sql_wrapper.__doc__ = _EXECUTE_SQL_DICT_LIST_MODE_DOCSTRING
    return execute_sql_wrapper

  # Return the default execute_sql function.
  return execute_sql

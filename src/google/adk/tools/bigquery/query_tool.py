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

from google.oauth2.credentials import Credentials

from ...tools.bigquery import client

MAX_DOWNLOADED_QUERY_RESULT_ROWS = 50


def execute_sql(project_id: str, query: str, credentials: Credentials) -> dict:
  """Run a BigQuery SQL query in the project and return the result.

  Args:
      project_id (str): The GCP project id in which the query should be
        executed.
      query (str): The BigQuery SQL query to be executed.
      credentials (Credentials): The credentials to use for the request.

  Returns:
      dict: Dictionary representing the result of the query.
            If the result contains the key "result_is_likely_truncated" with
            value True, it means that there may be additional rows matching the
            query not returned in the result.

  Examples:
      >>> execute_sql("bigframes-dev",
      ... "SELECT island, COUNT(*) AS population "
      ... "FROM bigquery-public-data.ml_datasets.penguins GROUP BY island")
      {
        "rows": [
            {
                "island": "Dream",
                "population": 124
            },
            {
                "island": "Biscoe",
                "population": 168
            },
            {
                "island": "Torgersen",
                "population": 52
            }
        ]
      }
  """

  try:
    bq_client = client.get_bigquery_client(credentials=credentials)
    row_iterator = bq_client.query_and_wait(
        query, project=project_id, max_results=MAX_DOWNLOADED_QUERY_RESULT_ROWS
    )
    rows = [{key: val for key, val in row.items()} for row in row_iterator]
    result = {"rows": rows}
    if (
        MAX_DOWNLOADED_QUERY_RESULT_ROWS is not None
        and len(rows) == MAX_DOWNLOADED_QUERY_RESULT_ROWS
    ):
      result["result_is_likely_truncated"] = True
    return result
  except Exception as ex:
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

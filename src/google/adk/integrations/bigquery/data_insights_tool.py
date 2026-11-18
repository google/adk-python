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

from typing import Any

from google.adk.tools import _gda_stream_util
from google.auth.credentials import Credentials

from .config import BigQueryToolConfig

_GDA_CLIENT_ID = "GOOGLE_ADK"

# Per-list limit enforced by the Conversational Analytics API:
# https://cloud.google.com/gemini/docs/conversational-analytics-api/troubleshoot-ca-errors
_MAX_SEARCH_SCOPE_ENTRIES = 100

# Maps each accepted spelling of a `search_scope` key to its REST JSON name.
_SEARCH_SCOPE_KEYS = {
    "includeProjectIds": "includeProjectIds",
    "include_project_ids": "includeProjectIds",
    "includeDatasetReferences": "includeDatasetReferences",
    "include_dataset_references": "includeDatasetReferences",
}

# Maps each accepted spelling of a dataset reference key to its REST JSON name.
_DATASET_REFERENCE_KEYS = {
    "projectId": "projectId",
    "project_id": "projectId",
    "datasetId": "datasetId",
    "dataset_id": "datasetId",
}


def ask_data_insights(
    project_id: str,
    user_query_with_context: str,
    table_references: list[dict[str, str]] | None = None,
    credentials: Credentials | None = None,
    settings: BigQueryToolConfig | None = None,
    search_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Answers questions about structured data in BigQuery tables using natural language.

  This function takes a user's question (which can include conversational
  history for context) and the BigQuery data to answer it from, and sends them
  to a stateless conversational API. The data is given as references to
  specific BigQuery tables, as a search scope of BigQuery projects or datasets
  in which the API finds the tables relevant to the question, or as both.

  The API uses a GenAI agent to understand the question, generate and execute
  SQL queries and Python code, and formulate an answer. This function returns a
  detailed, sequential log of this entire process, which includes any generated
  SQL or Python code, the data retrieved, and the final text answer. The final
  answer is always in plain text, as the underlying API is instructed not to
  generate any charts, graphs, images, or other visualizations.

  Use this tool to perform data analysis, get insights, or answer complex
  questions about the contents of BigQuery tables. Pass `table_references` when
  the exact tables are known, and `search_scope` when only the projects or
  datasets that hold the relevant tables are known.

  Args:
      project_id (str): The project that the inquiry is performed in.
      user_query_with_context (str): The user's original request, enriched with
        relevant context from the conversation history. The user's core intent
        should be preserved, but context should be added to resolve ambiguities
        in follow-up questions.
      table_references (list[dict[str, str]] | None): A list of dictionaries,
        each specifying a BigQuery table to be used as context for the question.
        Optional when `search_scope` is provided.
      credentials (Credentials | None): The credentials to use for the request.
        Required.
      settings (BigQueryToolConfig | None): The settings for the tool. Defaults
        to `BigQueryToolConfig()`.
      search_scope (dict[str, Any] | None): The BigQuery projects or datasets in
        which the API finds the tables relevant to the question, for use when
        the exact tables are not known. Optional when `table_references` is
        provided. Supports two keys, at least one of which must be a non-empty
        list:
        - `includeProjectIds`: up to 100 project IDs, for example
          `["my-gcp-project"]`.
        - `includeDatasetReferences`: up to 100 datasets, each a dictionary
          with a `projectId` and a `datasetId`, for example
          `[{"projectId": "my-gcp-project", "datasetId": "sales_data"}]`.
        The snake_case spellings of these keys (`include_project_ids`,
        `include_dataset_references`, `project_id` and `dataset_id`) are also
        accepted.

  Returns:
      A dictionary with two keys:
      - 'status': A string indicating the final status (e.g., "SUCCESS").
      - 'response': A list of dictionaries, where each dictionary
        represents a step in the API's execution process (e.g., SQL
        generation, data retrieval, final answer).

  Examples:
      A query joining multiple tables, showing the full return structure.
      The original question: "Which customer from New York spent the most last
      month?"

      >>> ask_data_insights(
      ...     project_id="some-project-id",
      ...     user_query_with_context=(
      ...         "Which customer from New York spent the most last month?"
      ...         "Context: The 'customers' table joins with the 'orders' table"
      ...         " on the 'customer_id' column."
      ...         ""
      ...     ),
      ...     table_references=[
      ...         {
      ...             "projectId": "my-gcp-project",
      ...             "datasetId": "sales_data",
      ...             "tableId": "customers"
      ...         },
      ...         {
      ...             "projectId": "my-gcp-project",
      ...             "datasetId": "sales_data",
      ...             "tableId": "orders"
      ...         }
      ...     ]
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "SQL Generated": "SELECT t1.customer_name, SUM(t2.order_total) ... "
          },
          {
            "Data Retrieved": {
              "headers": ["customer_name", "total_spent"],
              "rows": [["Jane Doe", 1234.56]],
              "summary": "Showing all 1 rows."
            }
          },
          {
            "Answer": "The customer who spent the most was Jane Doe."
          }
        ]
      }

      A question about the tables of a project, for when the exact tables are
      not known in advance:

      >>> ask_data_insights(
      ...     project_id="some-project-id",
      ...     user_query_with_context=(
      ...         "Which product category had the most returns last quarter?"
      ...     ),
      ...     search_scope={"includeProjectIds": ["my-gcp-project"]},
      ... )
  """
  try:
    bq_datasource_references = _build_bq_datasource_references(
        table_references, search_scope
    )
    if credentials is None:
      raise ValueError(
          "credentials are required to call the Conversational Analytics API."
      )
    if settings is None:
      settings = BigQueryToolConfig()
    location = "global"
    session, endpoint = _gda_stream_util.get_gda_session(credentials)
    with session:
      headers = {
          "Content-Type": "application/json",
          "X-Goog-API-Client": _GDA_CLIENT_ID,
      }
      ca_url = f"{endpoint}/v1/projects/{project_id}/locations/{location}:chat"

      instructions = """**INSTRUCTIONS - FOLLOW THESE RULES:**
    1.  **CONTENT:** Your answer should present the supporting data and then provide a conclusion based on that data, including relevant details and observations where possible.
    2.  **ANALYSIS DEPTH:** Your analysis must go beyond surface-level observations. Crucially, you must prioritize metrics that measure impact or outcomes over metrics that simply measure volume or raw counts. For open-ended questions, explore the topic from multiple perspectives to provide a holistic view.
    3.  **OUTPUT FORMAT:** Your entire response MUST be in plain text format ONLY.
    4.  **NO CHARTS:** You are STRICTLY FORBIDDEN from generating any charts, graphs, images, or any other form of visualization.
    """

      ca_payload = {
          "messages": [{"userMessage": {"text": user_query_with_context}}],
          "inlineContext": {
              "datasourceReferences": {"bq": bq_datasource_references},
              "systemInstruction": instructions,
          },
          "clientIdEnum": _GDA_CLIENT_ID,
      }

      resp = _gda_stream_util.get_stream(
          session, ca_url, ca_payload, headers, settings.max_query_result_rows
      )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
  return {"status": "SUCCESS", "response": resp}


def _build_bq_datasource_references(
    table_references: list[dict[str, str]] | None,
    search_scope: dict[str, Any] | None,
) -> dict[str, Any]:
  """Builds the BigQuery `datasourceReferences` of the chat request.

  Args:
      table_references: The `table_references` argument of `ask_data_insights`.
      search_scope: The `search_scope` argument of `ask_data_insights`.

  Returns:
      The `bq` datasource references, holding `tableReferences`,
      `searchScope`, or both.

  Raises:
      ValueError: If neither argument is provided or `search_scope` is
        invalid.
  """
  if table_references is None and search_scope is None:
    raise ValueError(
        "At least one of table_references or search_scope must be provided."
    )
  bq_references: dict[str, Any] = {}
  if table_references is not None:
    bq_references["tableReferences"] = table_references
  if search_scope is not None:
    bq_references["searchScope"] = _normalize_search_scope(search_scope)
  return bq_references


def _normalize_search_scope(search_scope: object) -> dict[str, Any]:
  """Validates a search scope and converts it to its REST JSON form.

  The scope is checked here rather than left to the API, because the API's
  validation errors reach the model only as an HTTP status code, which gives
  it nothing to correct its next call with.

  Args:
      search_scope: The `search_scope` argument of `ask_data_insights`. It is
        typed loosely because model-generated arguments can have any shape.

  Returns:
      The search scope keyed by camelCase REST JSON names, without empty
      lists.

  Raises:
      ValueError: If `search_scope` is malformed, exceeds 100 entries in a
        list, or names no project or dataset.
  """
  if not isinstance(search_scope, dict):
    raise ValueError("search_scope must be a dictionary.")
  scope = _to_rest_keys(search_scope, _SEARCH_SCOPE_KEYS, "search_scope")
  normalized: dict[str, Any] = {}

  project_ids = scope.get("includeProjectIds")
  if project_ids is not None:
    if not isinstance(project_ids, list) or not all(
        _is_non_blank_string(project_id) for project_id in project_ids
    ):
      raise ValueError(
          "includeProjectIds must be a list of non-blank project IDs."
      )
    if len(project_ids) > _MAX_SEARCH_SCOPE_ENTRIES:
      raise ValueError(
          "includeProjectIds must not contain more than"
          f" {_MAX_SEARCH_SCOPE_ENTRIES} project IDs."
      )
    if project_ids:
      normalized["includeProjectIds"] = [
          project_id.strip() for project_id in project_ids
      ]

  dataset_references = scope.get("includeDatasetReferences")
  if dataset_references is not None:
    if not isinstance(dataset_references, list) or not all(
        isinstance(reference, dict) for reference in dataset_references
    ):
      raise ValueError(
          "includeDatasetReferences must be a list of dictionaries, each with"
          " a projectId and a datasetId."
      )
    if len(dataset_references) > _MAX_SEARCH_SCOPE_ENTRIES:
      raise ValueError(
          "includeDatasetReferences must not contain more than"
          f" {_MAX_SEARCH_SCOPE_ENTRIES} dataset references."
      )
    rest_references = []
    for reference in dataset_references:
      rest_reference = _to_rest_keys(
          reference, _DATASET_REFERENCE_KEYS, "includeDatasetReferences entry"
      )
      if not all(
          _is_non_blank_string(rest_reference.get(key))
          for key in ("projectId", "datasetId")
      ):
        raise ValueError(
            "Each includeDatasetReferences entry must have a non-blank"
            " projectId and datasetId."
        )
      rest_references.append({
          "projectId": rest_reference["projectId"].strip(),
          "datasetId": rest_reference["datasetId"].strip(),
      })
    if rest_references:
      normalized["includeDatasetReferences"] = rest_references

  if not normalized:
    raise ValueError(
        "search_scope must contain non-empty includeProjectIds or"
        " includeDatasetReferences."
    )
  return normalized


def _to_rest_keys(
    value: dict[Any, Any], rest_keys: dict[str, str], label: str
) -> dict[str, Any]:
  """Renames camelCase or snake_case keys to their camelCase REST JSON names.

  Args:
      value: The dictionary whose keys are renamed.
      rest_keys: Maps each accepted key spelling to its REST JSON name.
      label: The name of `value` in error messages.

  Returns:
      A copy of `value` keyed by REST JSON names.

  Raises:
      ValueError: If `value` has an unsupported key, or sets a key under both
        of its spellings.
  """
  renamed: dict[str, Any] = {}
  for key, item in value.items():
    rest_key = rest_keys.get(key)
    if rest_key is None:
      supported = ", ".join(sorted(set(rest_keys.values())))
      raise ValueError(
          f"{label} has unsupported key {key!r}; supported keys: {supported}."
      )
    if rest_key in renamed:
      raise ValueError(f"{label} sets {rest_key} more than once.")
    renamed[rest_key] = item
  return renamed


def _is_non_blank_string(value: object) -> bool:
  """Returns whether `value` is a string with a non-whitespace character."""
  return isinstance(value, str) and bool(value.strip())

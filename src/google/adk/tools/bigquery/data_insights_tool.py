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
from typing import Any
from typing import Dict
from typing import List

from google.auth.credentials import Credentials
from google.cloud import bigquery
import requests

from . import client
from .config import BigQueryToolConfig


def ask_data_insights(
    project_id: str,
    user_query_with_context: str,
    table_references: List[Dict[str, str]],
    credentials: Credentials,
    settings: BigQueryToolConfig,
) -> Dict[str, Any]:
  """Answers questions about structured data in BigQuery tables using natural language.

  This function takes a user's question (which can include conversational
  history for context) and references to specific BigQuery tables, and sends
  them to a stateless conversational API.

  The API uses a GenAI agent to understand the question, generate and execute
  SQL queries and Python code, and formulate an answer. This function returns a
  detailed, sequential log of this entire process, which includes any generated
  SQL or Python code, the data retrieved, and the final text answer. The final
  answer is always in plain text, as the underlying API is instructed not to
  generate any charts, graphs, images, or other visualizations.

  Use this tool to perform data analysis, get insights, or answer complex
  questions about the contents of specific BigQuery tables.

  Args:
      project_id (str): The project that the inquiry is performed in.
      user_query_with_context (str): The user's original request, enriched with
        relevant context from the conversation history. The user's core intent
        should be preserved, but context should be added to resolve ambiguities
        in follow-up questions.
      table_references (List[Dict[str, str]]): A list of dictionaries, each
        specifying a BigQuery table to be used as context for the question.
      credentials (Credentials): The credentials to use for the request.
      settings (BigQueryToolConfig): The settings for the tool.

  Returns:
      A dictionary with two keys:
      - 'status': A string indicating the final status (e.g., "SUCCESS").
      - 'response': A list of dictionaries, where each dictionary
        represents a step in the API's execution process (e.g., SQL
        generation, data retrieval, final answer).

  Example:
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
  """
  try:
    location = "global"
    if not credentials.token:
      error_message = (
          "Error: The provided credentials object does not have a valid access"
          " token.\n\nThis is often because the credentials need to be"
          " refreshed or require specific API scopes. Please ensure the"
          " credentials are prepared correctly before calling this"
          " function.\n\nThere may be other underlying causes as well."
      )
      return {
          "status": "ERROR",
          "error_details": "ask_data_insights requires a valid access token.",
      }
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    ca_url = f"https://geminidataanalytics.googleapis.com/v1alpha/projects/{project_id}/locations/{location}:chat"

    instructions = """**INSTRUCTIONS - FOLLOW THESE RULES:**
    1.  **CONTENT:** Your answer should present the supporting data and then provide a conclusion based on that data, including relevant details and observations where possible.
    2.  **ANALYSIS DEPTH:** Your analysis must go beyond surface-level observations. Crucially, you must prioritize metrics that measure impact or outcomes over metrics that simply measure volume or raw counts. For open-ended questions, explore the topic from multiple perspectives to provide a holistic view.
    3.  **OUTPUT FORMAT:** Your entire response MUST be in plain text format ONLY.
    4.  **NO CHARTS:** You are STRICTLY FORBIDDEN from generating any charts, graphs, images, or any other form of visualization.
    """

    ca_payload = {
        "project": f"projects/{project_id}",
        "messages": [{"userMessage": {"text": user_query_with_context}}],
        "inlineContext": {
            "datasourceReferences": {
                "bq": {"tableReferences": table_references}
            },
            "systemInstruction": instructions,
            "options": {"chart": {"image": {"noImage": {}}}},
        },
        "clientIdEnum": "GOOGLE_ADK",
    }

    resp = _get_stream(
        ca_url, ca_payload, headers, settings.max_query_result_rows
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }
  return {"status": "SUCCESS", "response": resp}


def _get_stream(
    url: str,
    ca_payload: Dict[str, Any],
    headers: Dict[str, str],
    max_query_result_rows: int,
) -> List[Dict[str, Any]]:
  """Sends a JSON request to a streaming API and returns a list of messages."""
  s = requests.Session()

  accumulator = ""
  messages = []

  with s.post(url, json=ca_payload, headers=headers, stream=True) as resp:
    for line in resp.iter_lines():
      if not line:
        continue

      decoded_line = str(line, encoding="utf-8")

      if decoded_line == "[{":
        accumulator = "{"
      elif decoded_line == "}]":
        accumulator += "}"
      elif decoded_line == ",":
        continue
      else:
        accumulator += decoded_line

      if not _is_json(accumulator):
        continue

      data_json = json.loads(accumulator)
      if "systemMessage" not in data_json:
        if "error" in data_json:
          _append_message(messages, _handle_error(data_json["error"]))
        continue

      system_message = data_json["systemMessage"]
      if "text" in system_message:
        _append_message(messages, _handle_text_response(system_message["text"]))
      elif "schema" in system_message:
        _append_message(
            messages,
            _handle_schema_response(system_message["schema"]),
        )
      elif "data" in system_message:
        _append_message(
            messages,
            _handle_data_response(
                system_message["data"], max_query_result_rows
            ),
        )
      accumulator = ""
  return messages


def _is_json(s: str) -> bool:
  """Checks if a string is a valid JSON object."""
  try:
    json.loads(s)
  except ValueError:
    return False
  return True


def _get_property(
    data: Dict[str, Any], field_name: str, default: Any = ""
) -> Any:
  """Safely gets a property from a dictionary."""
  return data.get(field_name, default)


def _format_bq_table_ref(table_ref: Dict[str, str]) -> str:
  """Formats a BigQuery table reference dictionary into a string."""
  return f"{table_ref.get('projectId')}.{table_ref.get('datasetId')}.{table_ref.get('tableId')}"


def _format_schema_as_dict(
    data: Dict[str, Any],
) -> Dict[str, List[Any]]:
  """Extracts schema fields into a dictionary."""
  fields = data.get("fields", [])
  if not fields:
    return {"columns": []}

  column_details = []
  headers = ["Column", "Type", "Description", "Mode"]
  rows: List[List[str, str, str, str]] = []
  for field in fields:
    row_list = [
        _get_property(field, "name"),
        _get_property(field, "type"),
        _get_property(field, "description", ""),
        _get_property(field, "mode"),
    ]
    rows.append(row_list)

  return {"headers": headers, "rows": rows}


def _format_datasource_as_dict(datasource: Dict[str, Any]) -> Dict[str, Any]:
  """Formats a full datasource object into a dictionary with its name and schema."""
  source_name = _format_bq_table_ref(datasource["bigqueryTableReference"])

  schema = _format_schema_as_dict(datasource["schema"])
  return {"source_name": source_name, "schema": schema}


def _handle_text_response(resp: Dict[str, Any]) -> Dict[str, str]:
  """Formats a text response into a dictionary."""
  parts = resp.get("parts", [])
  return {"Answer": "".join(parts)}


def _handle_schema_response(resp: Dict[str, Any]) -> Dict[str, Any]:
  """Formats a schema response into a dictionary."""
  if "query" in resp:
    return {"Question": resp["query"].get("question", "")}
  elif "result" in resp:
    datasources = resp["result"].get("datasources", [])
    # Format each datasource and join them with newlines
    formatted_sources = [_format_datasource_as_dict(ds) for ds in datasources]
    return {"Schema Resolved": formatted_sources}
  return {}


def _handle_data_response(
    resp: Dict[str, Any], max_query_result_rows: int
) -> Dict[str, Any]:
  """Formats a data response into a dictionary."""
  if "query" in resp:
    query = resp["query"]
    return {
        "Retrieval Query": {
            "Query Name": query.get("name", "N/A"),
            "Question": query.get("question", "N/A"),
        }
    }
  elif "generatedSql" in resp:
    return {"SQL Generated": resp["generatedSql"]}
  elif "result" in resp:
    schema = resp["result"]["schema"]
    headers = [field.get("name") for field in schema.get("fields", [])]

    all_rows = resp["result"].get("data", [])
    total_rows = len(all_rows)

    compact_rows = []
    for row_dict in all_rows[:max_query_result_rows]:
      row_values = [row_dict.get(header) for header in headers]
      compact_rows.append(row_values)

    summary_string = f"Showing all {total_rows} rows."
    if total_rows > max_query_result_rows:
      summary_string = (
          f"Showing the first {len(compact_rows)} of {total_rows} total rows."
      )

    return {
        "Data Retrieved": {
            "headers": headers,
            "rows": compact_rows,
            "summary": summary_string,
        }
    }

  return {}


def _handle_error(resp: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
  """Formats an error response into a dictionary."""
  return {
      "Error": {
          "Code": resp.get("code", "N/A"),
          "Message": resp.get("message", "No message provided."),
      }
  }


def _append_message(
    messages: List[Dict[str, Any]], new_message: Dict[str, Any]
):
  if not new_message:
    return

  if messages and ("Data Retrieved" in messages[-1]):
    messages.pop()

  messages.append(new_message)

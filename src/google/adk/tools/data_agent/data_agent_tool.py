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

import json
from typing import Any

from google.auth.credentials import Credentials
import requests

from ..tool_context import ToolContext
from .config import DataAgentToolConfig

BASE_URL = "https://geminidataanalytics.googleapis.com/v1beta"


def _get_http_headers(
    credentials: Credentials,
) -> dict[str, str]:
  """Prepares headers for HTTP requests."""
  if not credentials.token:
    error_details = (
        "The provided credentials object does not have a valid access"
        " token.\n\nThis is often because the credentials need to be"
        " refreshed or require specific API scopes. Please ensure the"
        " credentials are prepared correctly before calling this"
        " function.\n\nThere may be other underlying causes as well."
    )
    raise ValueError(error_details)
  return {
      "Authorization": f"Bearer {credentials.token}",
      "Content-Type": "application/json",
  }


def list_accessible_data_agents(
    project_id: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """Lists accessible data agents in a project.

  Args:
      project_id: The project to list agents in.
      credentials: The credentials to use for the request.

  Returns:
      A dictionary containing the status and a list of data agents with their
      detailed information, including name, display_name, description (if
      available), create_time, update_time, and data_analytics_agent context,
      or error details if the request fails.

  Examples:
      >>> list_accessible_data_agents(
      ...     project_id="my-gcp-project",
      ...     credentials=credentials,
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "name": "projects/my-project/locations/global/dataAgents/agent1",
            "displayName": "My Test Agent",
            "createTime": "2025-10-01T22:44:22.473927629Z",
            "updateTime": "2025-10-01T22:44:23.094541325Z",
            "dataAnalyticsAgent": {
              "publishedContext": {
                "datasourceReferences": [{
                  "bq": {
                    "tableReferences": [{
                      "projectId": "my-project",
                      "datasetId": "dataset1",
                      "tableId": "table1"
                    }]
                  }
                }]
              }
            }
          },
          {
            "name": "projects/my-project/locations/global/dataAgents/agent2",
            "displayName": "",
            "description": "Description for Agent 2.",
            "createTime": "2025-06-23T20:23:48.650597312Z",
            "updateTime": "2025-06-23T20:23:49.437095391Z",
            "dataAnalyticsAgent": {
              "publishedContext": {
                "datasourceReferences": [{
                  "bq": {
                    "tableReferences": [{
                      "projectId": "another-project",
                      "datasetId": "dataset2",
                      "tableId": "table2"
                    }]
                  }
                }],
                "systemInstruction": "You are a helpful assistant.",
                "options": {"analysis": {"python": {"enabled": True}}}
              }
            }
          }
        ]
      }
  """
  try:
    headers = _get_http_headers(credentials)
    list_url = f"{BASE_URL}/projects/{project_id}/locations/global/dataAgents:listAccessible"
    resp = requests.get(
        list_url,
        headers=headers,
    )
    resp.raise_for_status()
    return {
        "status": "SUCCESS",
        "response": resp.json().get("dataAgents", []),
    }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
) -> dict[str, Any]:
  """Gets a data agent by name.

  Args:
      data_agent_name: The name of the agent to get, in format
        projects/{project}/locations/{location}/dataAgents/{agent}.
      credentials: The credentials to use for the request.

  Returns:
      A dictionary containing the status and details of a data agent,
      including name, display_name, description (if available),
      create_time, update_time, and data_analytics_agent context,
      or error details if the request fails.

  Examples:
      >>> get_data_agent_info(
      ...
      data_agent_name="projects/my-project/locations/global/dataAgents/agent-1",
      ...     credentials=credentials,
      ... )
      {
          "status": "SUCCESS",
          "response": {
              "name": "projects/my-project/locations/global/dataAgents/agent-1",
              "description": "Description for Agent 1.",
              "createTime": "2025-06-23T20:23:48.650597312Z",
              "updateTime": "2025-06-23T20:23:49.437095391Z",
              "dataAnalyticsAgent": {
                  "publishedContext": {
                      "systemInstruction": "You are a helpful assistant.",
                      "options": {"analysis": {"python": {"enabled": True}}},
                      "datasourceReferences": {
                          "bq": {
                              "tableReferences": [{
                                  "projectId": "my-gcp-project",
                                  "datasetId": "dataset1",
                                  "tableId": "table1"
                              }]
                          }
                      },
                  }
              }
          }
      }
  """
  try:
    headers = _get_http_headers(credentials)
    get_url = f"{BASE_URL}/{data_agent_name}"
    resp = requests.get(
        get_url,
        headers=headers,
    )
    resp.raise_for_status()
    return {
        "status": "SUCCESS",
        "response": resp.json(),
    }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def ask_data_agent(
    data_agent_name: str,
    query: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Asks a question to a data agent.

  Args:
      data_agent_name: The resource name of an existing data agent to ask, in
        format projects/{project}/locations/{location}/dataAgents/{agent}.
      query: The question to ask the agent.
      credentials: The credentials to use for the request.
      tool_context: The context for the tool.

  Returns:
      A dictionary with two keys:
      - 'status': A string indicating the final status (e.g., "SUCCESS").
      - 'response': A list of dictionaries, where each dictionary
        represents a step in the agent's execution process (e.g., SQL
        generation, data retrieval, final answer). Note that the 'Answer'
        step contains a text response which may summarize findings or refer
        to previous steps of agent execution, such as 'Data Retrieved', in
        which cases, the 'Answer' step does not include the result data.

  Examples:
      A query to a data agent, showing the full return structure.
      The original question: "Which customer from New York spent the most last
      month?"

      >>> ask_data_agent(
      ...
      data_agent_name="projects/my-project/locations/global/dataAgents/sales-agent",
      ...     query="Which customer from New York spent the most last month?",
      ...     credentials=credentials,
      ...     tool_context=tool_context,
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "Question": "Which customer from New York spent the most last
            month?"
          },
          {
            "Schema Resolved": [
              {
                "source_name": "my-gcp-project.sales_data.customers",
                "schema": {
                  "headers": ["Column", "Type", "Description", "Mode"],
                  "rows": [
                    ["customer_id", "INT64", "Customer ID", "REQUIRED"],
                    ["customer_name", "STRING", "Customer Name", "NULLABLE"],
                  ]
                }
              }
            ]
          },
          {
            "Retrieval Query": {
              "Query Name": "top_spender",
              "Question": "Find top spending customer from New York in the last
              month."
            }
          },
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
            "Answer": "The customer who spent the most last month was Jane Doe."
          }
        ]
      }
  """
  try:
    headers = _get_http_headers(credentials)

    agent_info = get_data_agent_info(data_agent_name, credentials)
    if agent_info.get("status") == "ERROR":
      return agent_info
    parent = data_agent_name.rsplit("/", 2)[0]
    chat_url = f"{BASE_URL}/{parent}:chat"
    chat_payload = {
        "messages": [{"userMessage": {"text": query}}],
        "dataAgentContext": {
            "dataAgent": data_agent_name,
        },
        "clientIdEnum": "GOOGLE_ADK",
    }
    resp = _get_stream(
        chat_url,
        chat_payload,
        headers=headers,
        max_query_result_rows=settings.max_query_result_rows,
    )
    return {"status": "SUCCESS", "response": resp}
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": repr(ex),
    }


def _get_stream(
    url: str,
    ca_payload: dict[str, Any],
    *,
    headers: dict[str, str],
    max_query_result_rows: int,
) -> list[dict[str, Any]]:
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

      try:
        data_json = json.loads(accumulator)
      except ValueError:
        continue
      if "systemMessage" not in data_json:
        if "error" in data_json:
          _append_message(
              messages,
              _handle_error(data_json["error"]),
          )
        continue

      system_message = data_json["systemMessage"]
      if "text" in system_message:
        _append_message(
            messages,
            _handle_text_response(system_message["text"]),
        )
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


def _format_bq_table_ref(table_ref: dict[str, str]) -> str:
  """Formats a BigQuery table reference dictionary into a string."""
  return f"{table_ref.get('projectId')}.{table_ref.get('datasetId')}.{table_ref.get('tableId')}"


def _format_schema_as_dict(
    data: dict[str, Any],
) -> dict[str, list[Any]]:
  """Extracts schema fields into a dictionary."""
  fields = data.get("fields", [])
  if not fields:
    return {"columns": []}

  column_details = []
  headers = ["Column", "Type", "Description", "Mode"]
  rows: list[list[str, str, str, str]] = []
  for field in fields:
    row_list = [
        field.get("name", ""),
        field.get("type", ""),
        field.get("description", ""),
        field.get("mode", ""),
    ]
    rows.append(row_list)

  return {"headers": headers, "rows": rows}


def _format_datasource_as_dict(datasource: dict[str, Any]) -> dict[str, Any]:
  """Formats a full datasource object into a dictionary with its name and schema."""
  source_name = _format_bq_table_ref(datasource["bigqueryTableReference"])

  schema = _format_schema_as_dict(datasource["schema"])
  return {"source_name": source_name, "schema": schema}


def _handle_text_response(resp: dict[str, Any]) -> dict[str, str]:
  """Formats a text response into a dictionary."""
  parts = resp.get("parts", [])
  return {"Answer": "".join(parts)}


def _handle_schema_response(resp: dict[str, Any]) -> dict[str, Any]:
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
    resp: dict[str, Any], max_query_result_rows: int
) -> dict[str, Any]:
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


def _handle_error(resp: dict[str, Any]) -> dict[str, dict[str, Any]]:
  """Formats an error response into a dictionary."""
  return {
      "Error": {
          "Code": resp.get("code", "N/A"),
          "Message": resp.get("message", "No message provided."),
      }
  }


def _append_message(
    messages: list[dict[str, Any]],
    new_message: dict[str, Any],
):
  """Appends a message to the list."""
  if not new_message:
    return

  messages.append(new_message)

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

from .. import _gda_stream_util
from ..tool_context import ToolContext
from .config import DataAgentToolConfig

BASE_URL = "https://geminidataanalytics.googleapis.com/v1beta"
_GDA_CLIENT_ID = "GOOGLE_ADK"


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
      "X-Goog-API-Client": _GDA_CLIENT_ID,
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
        "error_details": str(ex),
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
        "error_details": str(ex),
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
        represents a step in the agent's execution process and can
        contain keys like 'text', 'data', or 'Data Retrieved' indicating
        thought process, SQL generation, data retrieval, or final answer.

  Examples:
      A query to a data agent, showing the full return structure.
      The original question: "What is the average tree height in San
      Francisco?"

      >>> ask_data_agent(
      ...
      data_agent_name="projects/my-project/locations/global/dataAgents/sf-trees-agent",
      ...     query="What is the average tree height in San Francisco?",
      ...     credentials=credentials,
      ...     tool_context=tool_context,
      ... )
      {
        "status": "SUCCESS",
        "response": [
          {
            "text": {
              "parts": [
                "Analyzing context",
                "Retrieved context for 1 table."
              ],
              "textType": "THOUGHT"
            }
          },
          {
            "data": {
              "generatedSql": "SELECT\n AVG(SAFE_CAST(street_trees.dbh AS FLOAT64)) AS average_height\nFROM\n bigquery-public-data.san_francisco.street_trees AS street_trees;"
            }
          },
          {
            "Data Retrieved": {
              "headers": [
                "average_height"
              ],
              "rows": [
                [
                  10.073475670972512
                ]
              ],
              "summary": "Showing all 1 rows."
            }
          },
          {
            "text": {
              "parts": [
                "### Summary\nBased on the street tree data for San Francisco, the average height (recorded in the dbh column) is approximately 10.07."
              ],
              "textType": "FINAL_RESPONSE"
            }
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
        "clientIdEnum": _GDA_CLIENT_ID,
    }
    resp = _gda_stream_util.get_stream(
        chat_url,
        chat_payload,
        headers,
        settings.max_query_result_rows,
    )

    return {"status": "SUCCESS", "response": resp}
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

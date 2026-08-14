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

import asyncio
import json
import time
from typing import Any

from google.auth.credentials import Credentials
import requests

from .. import _gda_stream_util
from ..tool_context import ToolContext
from .config import DataAgentToolConfig

_GDA_CLIENT_ID = "GOOGLE_ADK"
_GDA_REQUEST_TIMEOUT_SECONDS = 30


def _extract_location_from_resource_name(resource_name: str) -> str | None:
  """Extracts the location segment from a resource name if present."""
  parts = resource_name.split("/")
  for i, part in enumerate(parts[:-1]):
    if part == "locations" and i + 1 < len(parts):
      return parts[i + 1]
  return None


def list_accessible_data_agents(
    project_id: str,
    credentials: Credentials,
    settings: DataAgentToolConfig | None = None,
) -> dict[str, Any]:
  """Lists accessible data agents in a project.

  Args:
      project_id: The project to list agents in.
      credentials: The credentials to use for the request.
      settings: Optional tool settings containing location or custom endpoint.

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
    location = (
        settings.location
        if settings and isinstance(settings.location, str)
        else None
    )
    api_endpoint = (
        settings.api_endpoint
        if settings and isinstance(settings.api_endpoint, str)
        else None
    )

    kwargs: dict[str, str] = {}
    if location:
      kwargs["location"] = location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
    base_url = f"{endpoint}/v1"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-API-Client": _GDA_CLIENT_ID,
    }
    target_location = location or "global"
    list_url = f"{base_url}/projects/{project_id}/locations/{target_location}/dataAgents:listAccessible"
    with session:
      resp = session.get(
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


def _get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
    session: requests.Session | None = None,
    settings: DataAgentToolConfig | None = None,
) -> dict[str, Any]:
  try:
    real_session: requests.Session | None = session
    real_settings: DataAgentToolConfig | None = settings

    location = (
        real_settings.location
        if real_settings and isinstance(real_settings.location, str)
        else None
    )
    api_endpoint = (
        real_settings.api_endpoint
        if real_settings and isinstance(real_settings.api_endpoint, str)
        else None
    )
    if not location and not api_endpoint and data_agent_name:
      location = _extract_location_from_resource_name(data_agent_name)

    kwargs: dict[str, str] = {}
    if location:
      kwargs["location"] = location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    endpoint = _gda_stream_util.get_gda_endpoint(**kwargs)
    base_url = f"{endpoint}/v1"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-API-Client": _GDA_CLIENT_ID,
    }
    get_url = f"{base_url}/{data_agent_name}"

    if real_session:
      resp = real_session.get(
          get_url,
          headers=headers,
      )
    else:
      local_session, _ = _gda_stream_util.get_gda_session(credentials, **kwargs)
      with local_session:
        resp = local_session.get(
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


def get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
    settings: DataAgentToolConfig | None = None,
) -> dict[str, Any]:
  """Gets a data agent by name.

  Args:
      data_agent_name: The name of the agent to get, in format
        projects/{project}/locations/{location}/dataAgents/{agent}.
      credentials: The credentials to use for the request.
      settings: Optional tool settings containing location or custom endpoint.

  Returns:
      A dictionary containing the status and details of a data agent,
      including name, display_name, description (if available),
      create_time, update_time, and data_analytics_agent context,
      or error details if the request fails.

  Examples:
      >>> get_data_agent_info(
      ...     data_agent_name="projects/p/locations/g/dataAgents/agent-1",
      ...     credentials=credentials,
      ... )
      {
          "status": "SUCCESS",
          "response": {
              "name": "projects/p/locations/g/dataAgents/agent-1",
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
  return _get_data_agent_info(data_agent_name, credentials, settings=settings)


def ask_data_agent(
    data_agent_name: str,
    query: str,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
    tool_context: ToolContext,
) -> dict[str, Any]:
  r"""Asks a question to a data agent.

  Args:
      data_agent_name: The resource name of an existing data agent to ask, in
        format projects/{project}/locations/{location}/dataAgents/{agent}.
      query: The question to ask the agent.
      credentials: The credentials to use for the request.
      settings: Tool configuration including max rows and optional endpoint.
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
      ...     data_agent_name="projects/p/locations/g/dataAgents/agent-1",
      ...     query="What is the average tree height in San Francisco?",
      ...     credentials=credentials,
      ...     settings=settings,
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
              "generatedSql": "SELECT\n AVG(SAFE_CAST(street_trees.dbh AS\n
              FLOAT64)) AS average_height\nFROM\n
              bigquery-public-data.san_francisco.street_trees AS street_trees;"
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
                "### Summary\nBased on the street tree data for San Francisco,\n
                the average height (recorded in the dbh column) is
                approximately\n                10.07."
              ],
              "textType": "FINAL_RESPONSE"
            }
          }
        ]
      }
  """
  try:
    location = (
        settings.location
        if settings and isinstance(settings.location, str)
        else None
    )
    api_endpoint = (
        settings.api_endpoint
        if settings and isinstance(settings.api_endpoint, str)
        else None
    )

    if not location and not api_endpoint and data_agent_name:
      location = _extract_location_from_resource_name(data_agent_name)

    kwargs: dict[str, str] = {}
    if location:
      kwargs["location"] = location
    if api_endpoint:
      kwargs["api_endpoint"] = api_endpoint

    session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
    with session:
      base_url = f"{endpoint}/v1"
      headers = {
          "Content-Type": "application/json",
          "X-Goog-API-Client": _GDA_CLIENT_ID,
      }

      agent_info = _get_data_agent_info(
          data_agent_name, credentials, session=session
      )

      if agent_info.get("status") == "ERROR":
        return agent_info
      parent = data_agent_name.rsplit("/", 2)[0]
      chat_url = f"{base_url}/{parent}:chat"
      chat_payload = {
          "messages": [{"userMessage": {"text": query}}],
          "dataAgentContext": {
              "dataAgent": data_agent_name,
          },
          "clientIdEnum": _GDA_CLIENT_ID,
      }
      resp = _gda_stream_util.get_stream(
          session,
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


async def create_data_agent(
    project_id: str,
    data_agent_id: str,
    agent_config: str,
    location: str | None = None,
    *,
    credentials: Credentials,
    settings: DataAgentToolConfig,
) -> dict[str, Any]:
  r"""Creates a new data agent.

  Args:
      project_id: The project in which to create the agent.
      data_agent_id: The ID to use for the new data agent.
      agent_config: A JSON string representing the DataAgent resource to create.
        For detailed REST resource schema and create documentation, see:
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents#DataAgent
        https://docs.cloud.google.com/gemini/data-agents/reference/rest/v1/projects.locations.dataAgents/create
      location: The Google Cloud location for data agent creation. If omitted,
        uses the toolset's configured location, falling back to "global". Only
        specify this when the user explicitly asks for a different region.
      credentials: The credentials to use for the request.
      settings: The configuration for the tool.

  Returns:
      A dictionary containing the status and the newly created data agent's
      details, or error details if the request fails.
      The tool waits for the create operation to finish, polling for up to
      `DataAgentToolConfig.data_agent_modification_timeout_seconds` (60s by
      default) in total. A timeout does not necessarily mean the creation
      failed; the operation may still be processing in the background, and
      `operation_name` is returned so the caller can check status later.

  Examples:
      >>> await create_data_agent(
      ...     project_id="my-gcp-project",
      ...     data_agent_id="my-new-agent",
      ...     agent_config='{"displayName": "My New Agent", "description":'
      ...     ' "An agent that helps with my-new-agent tasks",'
      ...     ' "dataAnalyticsAgent": {"publishedContext":'
      ...     ' {"datasourceReferences": {"bq": {"tableReferences":'
      ...     ' [{"projectId": "my-gcp-project", "datasetId": "dataset1",'
      ...     ' "tableId": "table1"}]}}, "systemInstruction": "You are a'
      ...     ' helpful assistant.", "options": {"analysis": {"python":'
      ...     ' {"enabled": True}}}}}}',
      ...     location="global",
      ...     credentials=credentials,
      ...     settings=DataAgentToolConfig(enable_data_agent_modification=True),
      ... )
      {
        "status": "SUCCESS",
        "response": {
          "@type":
          "type.googleapis.com/google.cloud.geminidataanalytics.v1.DataAgent",
          "name":
          "projects/my-gcp-project/locations/global/dataAgents/my-new-agent",
          "displayName": "My New Agent",
          "description": "An agent that helps with my-new-agent tasks",
          "createTime": "2025-10-01T22:44:22.473927629Z",
          "updateTime": "2025-10-01T22:44:22.473927629Z",
          "dataAnalyticsAgent": {
            "publishedContext": {
              "datasourceReferences": {
                "bq": {
                  "tableReferences": [{
                    "projectId": "my-gcp-project",
                    "datasetId": "dataset1",
                    "tableId": "table1"
                  }]
                }
              },
              "systemInstruction": "You are a helpful assistant.",
              "options": {"analysis": {"python": {"enabled": True}}}
            }
          }
        }
      }

      Example showing an error response if the Gemini Data Analytics API is
      disabled:
      >>> await create_data_agent(
      ...     project_id="my-gcp-project",
      ...     data_agent_id="my-new-agent",
      ...     agent_config={"displayName": "My New Agent"},
      ...     credentials=credentials,
      ...     settings=DataAgentToolConfig(enable_data_agent_modification=True),
      ... )
      {
        "status": "ERROR",
        "error_details": "API returned error status: 403 {\n  \"error\": {\n
        \"code\": 403,\n    \"message\": \"Data Analytics API with Gemini has
        not been used in project my-gcp-project before or it is disabled.\",\n
        \"status\": \"PERMISSION_DENIED\"\n  }\n}"
      }
  """
  try:
    if not settings.enable_data_agent_modification:
      return {
          "status": "ERROR",
          "error_details": (
              "Data agent mutation is disabled. Enable it by setting "
              "`enable_data_agent_modification=True` in DataAgentToolConfig."
          ),
      }

    try:
      # The public tool signatures annotate agent_config as `str` so the
      # generated function-calling schema stays a plain string without `anyOf`.
      # We accept `dict` here for Python programmatic callers and AI middleware.
      parsed_config = (
          agent_config
          if isinstance(agent_config, dict)
          else json.loads(agent_config)
      )
      if not isinstance(parsed_config, dict):
        raise TypeError(
            "agent_config must be a dictionary or a JSON string representing a"
            f" dictionary, got {type(parsed_config).__name__}"
        )
    except (ValueError, TypeError) as ex:
      return {
          "status": "ERROR",
          "error_details": f"Invalid agent_config: {ex}",
      }

    config_location = (
        settings.location
        if settings and isinstance(getattr(settings, "location", None), str)
        else None
    )
    effective_location = location or config_location or "global"
    kwargs = {}
    if effective_location:
      kwargs["location"] = effective_location
    api_endpoint = getattr(settings, "api_endpoint", None)
    if isinstance(api_endpoint, str):
      kwargs["api_endpoint"] = api_endpoint
    session, endpoint = _gda_stream_util.get_gda_session(credentials, **kwargs)
    base_url = f"{endpoint}/v1"
    url = f"{base_url}/projects/{project_id}/locations/{effective_location}/dataAgents"
    params = {"dataAgentId": data_agent_id}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-API-Client": _GDA_CLIENT_ID,
    }

    total_timeout = settings.data_agent_modification_timeout_seconds
    poll_interval = settings.data_agent_modification_poll_interval_seconds
    deadline = time.monotonic() + total_timeout

    with session:
      resp = await asyncio.to_thread(
          session.post,
          url,
          params=params,
          json=parsed_config,
          headers=headers,
          timeout=min(
              _GDA_REQUEST_TIMEOUT_SECONDS,
              max(0.0, deadline - time.monotonic()),
          ),
      )

      if not resp.ok:
        return {
            "status": "ERROR",
            "error_details": (
                f"API returned error status: {resp.status_code} {resp.text}"
            ),
        }

      operation = resp.json()
      if operation.get("done"):
        if "error" in operation:
          return {
              "status": "ERROR",
              "error_details": json.dumps(operation["error"]),
          }
        return {
            "status": "SUCCESS",
            "response": operation.get("response", operation),
        }

      operation_name = operation.get("name")
      if not operation_name or "/operations/" not in operation_name:
        return {"status": "SUCCESS", "response": operation}

      poll_url = f"{base_url}/{operation_name}"

      while True:
        remaining_budget = deadline - time.monotonic()
        if remaining_budget <= 0.1:
          break

        request_timeout = min(_GDA_REQUEST_TIMEOUT_SECONDS, remaining_budget)
        poll_resp = await asyncio.to_thread(
            session.get,
            poll_url,
            headers=headers,
            timeout=request_timeout,
        )
        if not poll_resp.ok:
          return {
              "status": "ERROR",
              "error_details": (
                  f"Polling failed with status: {poll_resp.status_code} "
                  f"{poll_resp.text}"
              ),
              "operation_name": operation_name,
          }
        poll_op = poll_resp.json()
        if poll_op.get("done"):
          if "error" in poll_op:
            return {
                "status": "ERROR",
                "error_details": json.dumps(poll_op["error"]),
                "operation_name": operation_name,
            }
          return {
              "status": "SUCCESS",
              "response": poll_op.get("response", poll_op),
          }

        sleep_duration = min(poll_interval, deadline - time.monotonic())
        if sleep_duration <= 0.1:
          break
        await asyncio.sleep(sleep_duration)

      return {
          "status": "ERROR",
          "error_details": (
              f"Operation {operation_name} did not complete within"
              f" {total_timeout} seconds."
          ),
          "operation_name": operation_name,
      }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

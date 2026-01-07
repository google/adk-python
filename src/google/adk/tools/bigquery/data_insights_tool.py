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
from google.cloud import geminidataanalytics
import requests

from . import client
from ..tool_context import ToolContext
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
  # TODO(huanc): replace this with official client library.
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


def list_accessible_data_agents(
    project_id: str,
    credentials: Credentials,
) -> Dict[str, Any]:
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
            "display_name": "My Test Agent",
            "create_time": {"seconds": 1759358662, "nanos": 473927629},
            "update_time": {"seconds": 1759358663, "nanos": 94541325},
            "data_analytics_agent": {
              "published_context": {
                "datasource_references": [{
                  "bq": {
                    "table_references": [{
                      "project_id": "my-project",
                      "dataset_id": "dataset1",
                      "table_id": "table1"
                    }]
                  }
                }]
              }
            }
          },
          {
            "name": "projects/my-project/locations/global/dataAgents/agent2",
            "display_name": "",
            "description": "Description for Agent 2.",
            "create_time": {"seconds": 1750710228, "nanos": 650597312},
            "update_time": {"seconds": 1750710229, "nanos": 437095391},
            "data_analytics_agent": {
              "published_context": {
                "datasource_references": [{
                  "bq": {
                    "table_references": [{
                      "project_id": "another-project",
                      "dataset_id": "dataset2",
                      "table_id": "table2"
                    }]
                  }
                }],
                "system_instruction": "You are a helpful assistant.",
                "options": {"analysis": {"python": {"enabled": True}}}
              }
            }
          }
        ]
      }
  """
  try:
    client = geminidataanalytics.DataAgentServiceClient(credentials=credentials)
    request = geminidataanalytics.ListAccessibleDataAgentsRequest(
        parent=f"projects/{project_id}/locations/global",
    )
    page_result = client.list_accessible_data_agents(request=request)
    return {
        "status": "SUCCESS",
        "response": [str(agent) for agent in page_result],
    }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def get_data_agent_info(
    data_agent_name: str,
    credentials: Credentials,
) -> Dict[str, Any]:
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
      ...     data_agent_name="projects/my-project/locations/global/dataAgents/agent-1",
      ...     credentials=credentials,
      ... )
      {
          "status": "SUCCESS",
          "response": {
              "name": "projects/my-project/locations/global/dataAgents/agent-1",
              "display_name": "My Agent 1",
              "description": "Description for Agent 1.",
              "create_time": {"seconds": 1750710228, "nanos": 650597312},
              "update_time": {"seconds": 1750710229, "nanos": 437095391},
              "data_analytics_agent": {
                  "published_context": {
                      "datasource_references": [{
                          "bq": {
                              "table_references": [{
                                  "project_id": "my-gcp-project",
                                  "dataset_id": "dataset1",
                                  "table_id": "table1"
                              }]
                          }
                      }],
                      "system_instruction": "You are a helpful assistant.",
                      "options": {"analysis": {"python": {"enabled": True}}}
                  }
              }
          }
      }
  """
  try:
    client = geminidataanalytics.DataAgentServiceClient(credentials=credentials)
    request = geminidataanalytics.GetDataAgentRequest(
        name=data_agent_name,
    )
    response = client.get_data_agent(request=request)
    return {
        "status": "SUCCESS",
        "response": str(response),
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
    tool_context: ToolContext,
) -> Dict[str, Any]:
  """Asks a question to a data agent.

  Args:
      data_agent_name: The resource name of an existing data agent to ask,
        in format projects/{project}/locations/{location}/dataAgents/{agent}.
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
        to previous steps of agent execution, such as 'Data Retrieved'， in
        which cases, the 'Answer' step does not include the result data.

  Examples:
      A query to a data agent, showing the full return structure.
      The original question: "Which customer from New York spent the most last
      month?"

      >>> ask_data_agent(
      ...     data_agent_name="projects/my-project/locations/global/dataAgents/sales-agent",
      ...     query="Which customer from New York spent the most last month?",
      ...     credentials=credentials,
      ...     tool_context=tool_context,
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
    agent_info = get_data_agent_info(data_agent_name, credentials)
    if agent_info.get("status") == "ERROR":
      return agent_info
    client = geminidataanalytics.DataChatServiceClient(credentials=credentials)
    parent = data_agent_name.rsplit("/", 2)[0]
    conversation_name = None

    if (
        tool_context.state.get("bigquery_data_agent_conv_agent")
        == data_agent_name
    ):
      conversation_name = tool_context.state.get(
          "bigquery_data_agent_conv_name"
      )
    else:
      conversation = geminidataanalytics.Conversation()
      conversation.agents = [data_agent_name]
      request = geminidataanalytics.CreateConversationRequest(
          parent=parent,
          conversation=conversation,
      )
      response = client.create_conversation(request=request)
      conversation_name = response.name
      tool_context.state["bigquery_data_agent_conv_agent"] = data_agent_name
      tool_context.state["bigquery_data_agent_conv_name"] = conversation_name

    new_user_message = geminidataanalytics.Message()
    new_user_message.user_message.text = query
    messages = [new_user_message]

    if conversation_name:
      conversation_reference = geminidataanalytics.ConversationReference()
      conversation_reference.conversation = conversation_name
      conversation_reference.data_agent_context.data_agent = data_agent_name
      request = geminidataanalytics.ChatRequest(
          parent=parent,
          messages=messages,
          conversation_reference=conversation_reference,
      )
    else:
      data_agent_context = geminidataanalytics.DataAgentContext()
      data_agent_context.data_agent = data_agent_name
      request = geminidataanalytics.ChatRequest(
          parent=parent,
          messages=messages,
          data_agent_context=data_agent_context,
      )
    stream = client.chat(request=request)
    responses = list(stream)
    print({
        "status": "SUCCESS",
        "response": _process_data_agent_stream(responses),
    })
    return {
        "status": "SUCCESS",
        "response": _process_data_agent_stream(responses),
    }
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }


def _process_result_message(result, max_rows: int) -> dict[str, Any]:
  """Processes result message from data agent chat."""
  headers = [f.name for f in result.schema.fields]
  all_rows_structs = result.data
  total_rows = len(all_rows_structs)

  summary_string = f"Showing all {total_rows} rows."
  if total_rows > max_rows:
    summary_string = f"Showing the first {max_rows} of {total_rows} total rows."
  rows = []
  i = 0
  for row in all_rows_structs:
    if i >= max_rows:
      break
    rows.append([row.get(h) for h in headers])
    i += 1
  return {
      "headers": headers,
      "rows": rows,
      "summary": summary_string,
  }


def _process_data_agent_stream(
    stream: list[geminidataanalytics.Message], max_rows: int = 1000
) -> list[dict[str, Any]]:
  """Processes stream from data agent chat."""
  processed_responses = []
  for i, msg in enumerate(stream):
    if msg.system_message:
      message = msg.system_message
      if message.text.parts:
        processed_responses.append({"Answer": "".join(message.text.parts)})
      elif message.data.generated_sql:
        processed_responses.append(
            {"SQL Generated": message.data.generated_sql}
        )
      elif message.data.result and message.data.result.data:
        processed_responses.append({
            "Data Retrieved": _process_result_message(
                message.data.result, max_rows
            )
        })
      elif message.error:
        processed_responses.append({"Error": message.error.text})
  return processed_responses


def _get_stream(
    url: str,
    ca_payload: Dict[str, Any],
    *,
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

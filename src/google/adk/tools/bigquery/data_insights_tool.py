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
from typing import Dict
from typing import List

from google.auth.credentials import Credentials
from google.cloud import bigquery
import requests

from . import client
from .. import _gda_stream_util
from .config import BigQueryToolConfig

_GDA_CLIENT_ID = "GOOGLE_ADK"


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
        represents a step in the agent's execution process and can
        contain keys like 'text', 'data', or 'Data Retrieved' indicating
        thought process, SQL generation, data retrieval, or final answer.

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
            "text": {
              "parts": [
                "Analyzing context",
                "Retrieved context for 2 tables."
              ],
              "textType": "THOUGHT"
            }
          },
          {
            "data": {
              "generatedSql": "SELECT t1.customer_name, SUM(t2.order_total) ..."
            }
          },
          {
            "Data Retrieved": {
              "headers": ["customer_name", "total_spent"],
              "rows": [["Jane Doe", 1234.56]],
              "summary": "Showing all 1 rows."
            }
          },
          {
            "text": {
              "parts": [
                "The customer who spent the most was Jane Doe."
              ],
              "textType": "FINAL_RESPONSE"
            }
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
        "X-Goog-API-Client": _GDA_CLIENT_ID,
    }
    ca_url = f"https://geminidataanalytics.googleapis.com/v1beta/projects/{project_id}/locations/{location}:chat"

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
        "clientIdEnum": _GDA_CLIENT_ID,
    }

    resp = _gda_stream_util.get_stream(
        ca_url, ca_payload, headers, settings.max_query_result_rows
    )
  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }

  return {"status": "SUCCESS", "response": resp}

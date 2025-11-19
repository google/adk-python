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

"""ADK knowledge search tool."""
from __future__ import annotations

from typing import Any
import uuid

import requests

KNOWLEDGE_SERVICE_APP_URL = "https://adk-agent-builder-knowledge-service-654646711756.us-central1.run.app"
KNOWLEDGE_SERVICE_APP_NAME = "adk_knowledge_agent"
KNOWLEDGE_SERVICE_APP_USER_NAME = "agent_builder_assistant"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def search_adk_knowledge(
    query: str,
) -> dict[str, Any]:
  """Searches ADK knowledge base for relevant information.

  Args:
    query: The query to search in ADK knowledge base.

  Returns:
    A dict with status and the response from the knowledge service.
  """
  # Create a new session
  session_id = uuid.uuid4()
  create_session_url = f"{KNOWLEDGE_SERVICE_APP_URL}/apps/{KNOWLEDGE_SERVICE_APP_NAME}/users/{KNOWLEDGE_SERVICE_APP_USER_NAME}/sessions/{session_id}"

  try:
    create_session_response = post_request(
        create_session_url,
        {},
    )
  except requests.exceptions.RequestException as e:
    return error_response(f"Failed to create session: {e}")
  session_id = create_session_response["id"]

  # Search ADK knowledge base
  search_url = f"{KNOWLEDGE_SERVICE_APP_URL}/run"
  try:
    search_response = post_request(
        search_url,
        {
            "app_name": KNOWLEDGE_SERVICE_APP_NAME,
            "user_id": KNOWLEDGE_SERVICE_APP_USER_NAME,
            "session_id": session_id,
            "new_message": {"role": "user", "parts": [{"text": query}]},
        },
    )
  except requests.exceptions.RequestException as e:
    return error_response(f"Failed to search ADK knowledge base: {e}")
  return {
      "status": "success",
      "response": search_response,
  }


def error_response(error_message: str) -> dict[str, Any]:
  """Returns an error response."""
  return {"status": "error", "error_message": error_message}


def post_request(url: str, payload: dict[str, Any]) -> dict[str, Any]:
  """Executes a POST request."""
  response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()

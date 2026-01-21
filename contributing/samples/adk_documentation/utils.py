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

import re
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from adk_documentation.settings import GITHUB_TOKEN
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.genai import types
import requests

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def error_response(error_message: str) -> Dict[str, Any]:
  return {"status": "error", "error_message": error_message}


def get_request(
    url: str,
    headers: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Dict[str, Any]:
  """Executes a GET request."""
  if headers is None:
    headers = HEADERS
  if params is None:
    params = {}
  response = requests.get(url, headers=headers, params=params, timeout=60)
  response.raise_for_status()
  return response.json()


def get_paginated_request(
    url: str, headers: dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
  """Executes GET requests and follows 'next' pagination links to fetch all results."""
  if headers is None:
    headers = HEADERS

  results = []
  while url:
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    results.extend(response.json())
    url = response.links.get("next", {}).get("url")
  return results


def post_request(url: str, payload: Any) -> Dict[str, Any]:
  response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


def patch_request(url: str, payload: Any) -> Dict[str, Any]:
  response = requests.patch(url, headers=HEADERS, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


async def call_agent_async(
    runner: Runner, user_id: str, session_id: str, prompt: str
) -> str:
  """Call the agent asynchronously with the user's prompt."""
  content = types.Content(
      role="user", parts=[types.Part.from_text(text=prompt)]
  )

  final_response_text = ""
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(save_input_blobs_as_artifacts=False),
  ):
    if event.content and event.content.parts:
      if text := "".join(part.text or "" for part in event.content.parts):
        if event.author != "user":
          final_response_text += text

  return final_response_text


def parse_suggestions(issue_body: str) -> List[Tuple[int, str]]:
  """Parse numbered suggestions from issue body.

  Supports multiple formats:
  - Format A (markdown headers): "### 1. Title"
  - Format B (numbered list with bold): "1. **Title**"

  Args:
      issue_body: The body text of the GitHub issue.

  Returns:
      A list of tuples, where each tuple contains:
      - The suggestion number (1-based)
      - The full text of that suggestion
  """
  # Try different patterns in order of preference
  patterns = [
      # Format A: "### 1. Title" (markdown header with number)
      (r"(?=^###\s+\d+\.)", r"^###\s+(\d+)\."),
      # Format B: "1. **Title**" (numbered list with bold)
      (r"(?=^\d+\.\s+\*\*)", r"^(\d+)\.\s+\*\*"),
  ]

  for split_pattern, match_pattern in patterns:
    parts = re.split(split_pattern, issue_body, flags=re.MULTILINE)

    suggestions = []
    for part in parts:
      part = part.strip()
      if not part:
        continue

      match = re.match(match_pattern, part)
      if match:
        suggestion_num = int(match.group(1))
        suggestions.append((suggestion_num, part))

    # If we found suggestions with this pattern, return them
    if suggestions:
      return suggestions

  return []

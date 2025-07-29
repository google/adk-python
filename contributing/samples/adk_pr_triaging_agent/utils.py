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

from typing import Any

from adk_pr_triaging_agent.settings import GITHUB_GRAPHQL_URL
from adk_pr_triaging_agent.settings import GITHUB_TOKEN
import requests

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

diff_headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}


def run_graphql_query(query: str, variables: dict[str, Any]) -> dict[str, Any]:
  """Executes a GraphQL query."""
  payload = {"query": query, "variables": variables}
  response = requests.post(
      GITHUB_GRAPHQL_URL, headers=headers, json=payload, timeout=60
  )
  response.raise_for_status()
  return response.json()


def get_request(url: str, params: dict[str, Any] | None = None) -> Any:
  """Executes a GET request."""
  if params is None:
    params = {}
  response = requests.get(url, headers=headers, params=params, timeout=60)
  response.raise_for_status()
  return response.json()


def get_diff(url: str) -> str:
  """Executes a GET request for a diff."""
  response = requests.get(url, headers=diff_headers)
  response.raise_for_status()
  return response.text


def post_request(url: str, payload: Any) -> dict[str, Any]:
  """Executes a POST request."""
  response = requests.post(url, headers=headers, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


def error_response(error_message: str) -> dict[str, Any]:
  """Returns an error response."""
  return {"status": "error", "error_message": error_message}


def read_file(file_path: str) -> str:
  """Read the content of the given file."""
  try:
    with open(file_path, "r") as f:
      return f.read()
  except FileNotFoundError:
    print(f"Error: File not found: {file_path}.")
    return ""

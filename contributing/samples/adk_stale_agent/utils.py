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

from adk_stale_agent.settings import GITHUB_TOKEN
import requests

_session = requests.Session()
_session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
})


def get_request(url: str, params: dict[str, Any] | None = None) -> Any:
  """Sends a GET request to the GitHub API."""
  response = _session.get(url, params=params or {}, timeout=60)
  response.raise_for_status()
  return response.json()


def post_request(url: str, payload: Any) -> Any:
  """Sends a POST request to the GitHub API."""
  response = _session.post(url, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


def patch_request(url: str, payload: Any) -> Any:
  """Sends a PATCH request to the GitHub API."""
  response = _session.patch(url, json=payload, timeout=60)
  response.raise_for_status()
  return response.json()


def delete_request(url: str) -> Any:
  """Sends a DELETE request to the GitHub API."""
  response = _session.delete(url, timeout=60)
  response.raise_for_status()
  if response.status_code == 204:
    return {"status": "success"}
  return response.json()


def error_response(error_message: str) -> dict[str, Any]:
  """Creates a standardized error dictionary for the agent."""
  return {"status": "error", "message": error_message}

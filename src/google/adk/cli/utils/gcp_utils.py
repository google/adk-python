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

"""Utilities for GCP authentication and Vertex AI Express Mode."""

from __future__ import annotations

import subprocess
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import google.auth
import google.auth.exceptions
from google.auth.transport.requests import AuthorizedSession
from google.auth.transport.requests import Request
from google.cloud import resourcemanager_v3
import requests

_VERTEX_AI_ENDPOINT = "https://{location}-aiplatform.googleapis.com/v1beta1"


def check_adc() -> bool:
  """Checks if Application Default Credentials exist."""
  try:
    google.auth.default()
    return True
  except google.auth.exceptions.DefaultCredentialsError:
    return False


def login_adc() -> None:
  """Prompts user to login via gcloud ADC."""
  try:
    subprocess.run(
        ["gcloud", "auth", "application-default", "login"], check=True
    )
  except (subprocess.CalledProcessError, FileNotFoundError):
    raise RuntimeError(
        "gcloud is not installed or failed to run. "
        "Please install gcloud to login to Application Default Credentials."
    )


def get_access_token() -> str:
  """Gets the ADC access token."""
  try:
    credentials, _ = google.auth.default()
    if not credentials.valid:
      credentials.refresh(Request())
    return credentials.token or ""
  except google.auth.exceptions.DefaultCredentialsError:
    raise RuntimeError("Application Default Credentials not found.")


def _call_vertex_express_api(
    method: str,
    action: str,
    location: str = "us-central1",
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Calls a Vertex AI Express API."""
  credentials, _ = google.auth.default()
  session = AuthorizedSession(credentials)
  url = f"{_VERTEX_AI_ENDPOINT.format(location=location)}/vertexExpress{action}"
  headers = {
      "Content-Type": "application/json",
  }

  if method == "GET":
    response = session.get(url, headers=headers, params=params)
  elif method == "POST":
    response = session.post(url, headers=headers, json=data, params=params)
  else:
    raise ValueError(f"Unsupported method: {method}")

  response.raise_for_status()
  return response.json()


def retrieve_express_project(
    location: str = "us-central1",
) -> Optional[Dict[str, Any]]:
  """Retrieves existing Express project info."""
  try:
    response = _call_vertex_express_api(
        "GET",
        ":retrieveExpressProject",
        location=location,
        params={"get_default_api_key": True},
    )
    project = response.get("expressProject")
    if not project:
      return None

    return {
        "project_id": project.get("projectId"),
        "api_key": project.get("defaultApiKey"),
        "region": project.get("region", location),
    }
  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      return None
    raise


def check_express_eligibility(
    location: str = "us-central1",
) -> bool:
  """Checks if user is eligible for Express Mode."""
  try:
    result = _call_vertex_express_api(
        "GET", "/Eligibility:check", location=location
    )
    return result.get("eligibility") == "IN_SCOPE"
  except (requests.exceptions.HTTPError, KeyError) as e:
    return False


def sign_up_express(
    location: str = "us-central1",
) -> Dict[str, Any]:
  """Signs up for Express Mode."""
  project = _call_vertex_express_api(
      "POST",
      ":signUp",
      location=location,
      data={"region": location, "tos_accepted": True},
  )
  return {
      "project_id": project.get("projectId"),
      "api_key": project.get("defaultApiKey"),
      "region": project.get("region", location),
  }


def list_gcp_projects(limit: int = 20) -> List[Tuple[str, str]]:
  """Lists GCP projects available to the user.

  Args:
    limit: The maximum number of projects to return.

  Returns:
    A list of (project_id, name) tuples.
  """
  try:
    client = resourcemanager_v3.ProjectsClient()
    search_results = client.search_projects()

    projects = []
    for project in search_results:
      if len(projects) >= limit:
        break
      projects.append(
          (project.project_id, project.display_name or project.project_id)
      )
    return projects
  except Exception:
    return []

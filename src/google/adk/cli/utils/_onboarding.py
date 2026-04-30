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

"""Utilities for ADK CLI onboarding flow."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import click
from pydantic import BaseModel

from . import gcp_utils

_GOOGLE_API_MSG = """
Don't have API Key? Create one in AI Studio: https://aistudio.google.com/apikey
"""

_GOOGLE_CLOUD_SETUP_MSG = """
You need an existing Google Cloud account and project, check out this link for details:
https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai
"""

_EXPRESS_TOS_MSG = """
Google Cloud Express Mode Terms of Service: https://cloud.google.com/terms/google-cloud-express
By using this application, you agree to the Google Cloud Express Mode terms of service and any
applicable services and APIs: https://console.cloud.google.com/terms. You also agree to only use
this application for your trade, business, craft, or profession.
"""

_NOT_ELIGIBLE_MSG = """
You are not eligible for Express Mode.
Please follow these instructions to set up a full Google Cloud project:
https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai
"""


class GoogleAIAuth(BaseModel):
  api_key: str


class VertexAIAuth(BaseModel):
  project_id: str
  region: str


class ExpressModeAuth(BaseModel):
  api_key: str
  project_id: str
  region: str


def get_gcp_project_from_gcloud() -> str:
  """Uses gcloud to get default project."""
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return ""


def get_gcp_region_from_gcloud() -> str:
  """Uses gcloud to get default region."""
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "compute/region"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return ""


def prompt_str(
    prompt_prefix: str,
    *,
    prior_msg: Optional[str] = None,
    default_value: Optional[str] = None,
) -> str:
  if prior_msg:
    click.secho(prior_msg, fg="green")
  while True:
    value: str = click.prompt(
        prompt_prefix, default=default_value or None, type=str
    )
    if value and value.strip():
      return value.strip()


def prompt_for_google_cloud(
    google_cloud_project: Optional[str],
) -> str:
  """Prompts user for Google Cloud project ID."""
  google_cloud_project = (
      google_cloud_project
      or os.environ.get("GOOGLE_CLOUD_PROJECT", None)
      or get_gcp_project_from_gcloud()
  )

  google_cloud_project = prompt_str(
      "Enter Google Cloud project ID", default_value=google_cloud_project
  )

  return google_cloud_project


def prompt_for_google_cloud_region(
    google_cloud_region: Optional[str],
) -> str:
  """Prompts user for Google Cloud region."""
  google_cloud_region = (
      google_cloud_region
      or os.environ.get("GOOGLE_CLOUD_LOCATION", None)
      or get_gcp_region_from_gcloud()
  )

  google_cloud_region = prompt_str(
      "Enter Google Cloud region",
      default_value=google_cloud_region or "us-central1",
  )
  return google_cloud_region


def prompt_for_google_api_key(
    google_api_key: Optional[str],
) -> str:
  """Prompts user for Google API key."""
  google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", None)

  google_api_key = prompt_str(
      "Enter Google API key",
      prior_msg=_GOOGLE_API_MSG,
      default_value=google_api_key,
  )
  return google_api_key


def handle_login_with_google() -> VertexAIAuth | ExpressModeAuth:
  """Handles the "Login with Google" flow."""
  if not gcp_utils.check_adc():
    click.secho(
        "No Application Default Credentials found. "
        "Opening browser for login...",
        fg="yellow",
    )
    try:
      gcp_utils.login_adc()
    except RuntimeError as e:
      click.secho(str(e), fg="red")
      raise click.Abort()

  # Check for existing Express project
  express_project = gcp_utils.retrieve_express_project()
  if express_project:
    api_key = express_project.get("api_key")
    project_id = express_project.get("project_id")
    region = express_project.get("region", "us-central1")
    if project_id:
      click.secho(f"Using existing Express project: {project_id}", fg="green")
      return ExpressModeAuth(
          api_key=api_key, project_id=project_id, region=region
      )

  # Check for existing full GCP projects
  projects = gcp_utils.list_gcp_projects(limit=20)
  if projects:
    click.secho("Recently created Google Cloud projects found:", fg="green")
    click.echo("0. Enter project ID manually")
    for i, (p_id, p_name) in enumerate(projects, 1):
      click.echo(f"{i}. {p_name} ({p_id})")

    project_index = click.prompt(
        "Select a project",
        type=click.IntRange(0, len(projects)),
    )
    if project_index == 0:
      selected_project_id = prompt_for_google_cloud(None)
    else:
      selected_project_id = projects[project_index - 1][0]
    region = prompt_for_google_cloud_region(None)
    return VertexAIAuth(project_id=selected_project_id, region=region)

  click.secho(
      "A Google Cloud project is required to continue. You can enter an"
      " existing project ID or create an Express Mode project. Learn more:"
      " https://cloud.google.com/resources/cloud-express-faqs",
      fg="green",
  )
  action = click.prompt(
      "1. Enter an existing Google Cloud project ID\n"
      "2. Create a new project (Express Mode)\n"
      "3. Abandon\n"
      "Choose an action",
      type=click.Choice(["1", "2", "3"]),
  )

  if action == "3":
    raise click.Abort()

  if action == "1":
    google_cloud_project = prompt_for_google_cloud(None)
    google_cloud_region = prompt_for_google_cloud_region(None)
    return VertexAIAuth(
        project_id=google_cloud_project, region=google_cloud_region
    )

  elif action == "2":
    if gcp_utils.check_express_eligibility():
      click.secho(_EXPRESS_TOS_MSG, fg="yellow")
      if click.confirm("Do you accept the Terms of Service?", default=False):
        selected_region = click.prompt(
            """\
Choose a region for Express Mode:
1. us-central1
2. europe-west1
3. asia-southeast1
Choose region""",
            type=click.Choice(["1", "2", "3"]),
            default="1",
        )
        region_map = {
            "1": "us-central1",
            "2": "europe-west1",
            "3": "asia-southeast1",
        }
        region = region_map[selected_region]
        express_info = gcp_utils.sign_up_express(location=region)
        api_key = express_info.get("api_key")
        project_id = express_info.get("project_id")
        region = express_info.get("region", region)
        click.secho(
            f"Express Mode project created: {project_id}",
            fg="green",
        )
        current_proj = get_gcp_project_from_gcloud()
        if current_proj and current_proj != project_id:
          click.secho(
              "Warning: Your default gcloud project is set to"
              f" '{current_proj}'. This might conflict with or override your"
              f" Express Mode project '{project_id}'. We recommend"
              " unsetting it.",
              fg="yellow",
          )
          if click.confirm("Run 'gcloud config unset project'?", default=True):
            try:
              subprocess.run(
                  ["gcloud", "config", "unset", "project"],
                  check=True,
                  capture_output=True,
              )
              click.secho("Unset default gcloud project.", fg="green")
            except Exception:
              click.secho(
                  "Failed to unset project. Please do it manually.", fg="red"
              )
        return ExpressModeAuth(
            api_key=api_key, project_id=project_id, region=region
        )

    click.secho(_NOT_ELIGIBLE_MSG, fg="red")
    raise click.Abort()


def prompt_to_choose_backend(
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
) -> GoogleAIAuth | VertexAIAuth | ExpressModeAuth:
  """Prompts user to choose backend.

  Returns:
    A tuple of (google_api_key, google_cloud_project, google_cloud_region).
  """
  backend_choice = click.prompt(
      "1. Google AI\n2. Vertex AI\n3. Login with Google\nChoose a backend",
      type=click.Choice(["1", "2", "3"]),
  )
  if backend_choice == "1":
    google_api_key = prompt_for_google_api_key(google_api_key)
    return GoogleAIAuth(api_key=google_api_key)
  elif backend_choice == "2":
    click.secho(_GOOGLE_CLOUD_SETUP_MSG, fg="green")
    google_cloud_project = prompt_for_google_cloud(google_cloud_project)
    google_cloud_region = prompt_for_google_cloud_region(google_cloud_region)
    return VertexAIAuth(
        project_id=google_cloud_project, region=google_cloud_region
    )
  elif backend_choice == "3":
    return handle_login_with_google()

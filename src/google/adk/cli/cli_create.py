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

import os
import subprocess
from typing import Optional
from typing import Tuple

import click

from ..apps.app import validate_app_name
from .utils import gcp_utils

_INIT_PY_TEMPLATE = """\
from . import agent
"""

_AGENT_PY_TEMPLATE = """\
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model='{model_name}',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
"""

_AGENT_CONFIG_TEMPLATE = """\
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: root_agent
description: A helpful assistant for user questions.
instruction: Answer user questions to the best of your knowledge
model: {model_name}
"""


_GOOGLE_API_MSG = """
Don't have API Key? Create one in AI Studio: https://aistudio.google.com/apikey
"""

_GOOGLE_CLOUD_SETUP_MSG = """
You need an existing Google Cloud account and project, check out this link for details:
https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai
"""

_OTHER_MODEL_MSG = """
Please see below guide to configure other models:
https://google.github.io/adk-docs/agents/models
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

_SUCCESS_MSG_CODE = """
Agent created in {agent_folder}:
- .env
- __init__.py
- agent.py

⚠️  WARNING: Secrets (like GOOGLE_API_KEY) are stored in .env.
Please ensure .env is added to your .gitignore to avoid committing secrets to version control.
"""

_SUCCESS_MSG_CONFIG = """
Agent created in {agent_folder}:
- .env
- __init__.py
- root_agent.yaml

⚠️  WARNING: Secrets (like GOOGLE_API_KEY) are stored in .env.
Please ensure .env is added to your .gitignore to avoid committing secrets to version control.
"""


def _get_gcp_project_from_gcloud() -> str:
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


def _get_gcp_region_from_gcloud() -> str:
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


def _prompt_str(
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


def _prompt_for_google_cloud(
    google_cloud_project: Optional[str],
) -> str:
  """Prompts user for Google Cloud project ID."""
  google_cloud_project = (
      google_cloud_project
      or os.environ.get("GOOGLE_CLOUD_PROJECT", None)
      or _get_gcp_project_from_gcloud()
  )

  google_cloud_project = _prompt_str(
      "Enter Google Cloud project ID", default_value=google_cloud_project
  )

  return google_cloud_project


def _prompt_for_google_cloud_region(
    google_cloud_region: Optional[str],
) -> str:
  """Prompts user for Google Cloud region."""
  google_cloud_region = (
      google_cloud_region
      or os.environ.get("GOOGLE_CLOUD_LOCATION", None)
      or _get_gcp_region_from_gcloud()
  )

  google_cloud_region = _prompt_str(
      "Enter Google Cloud region",
      default_value=google_cloud_region or "us-central1",
  )
  return google_cloud_region


def _prompt_for_google_api_key(
    google_api_key: Optional[str],
) -> str:
  """Prompts user for Google API key."""
  google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", None)

  google_api_key = _prompt_str(
      "Enter Google API key",
      prior_msg=_GOOGLE_API_MSG,
      default_value=google_api_key,
  )
  return google_api_key


def _generate_files(
    agent_folder: str,
    *,
    google_api_key: Optional[str] = None,
    google_cloud_project: Optional[str] = None,
    google_cloud_region: Optional[str] = None,
    model: Optional[str] = None,
    type: str,
):
  """Generates a folder name for the agent."""
  os.makedirs(agent_folder, exist_ok=True)

  dotenv_file_path = os.path.join(agent_folder, ".env")
  init_file_path = os.path.join(agent_folder, "__init__.py")
  agent_py_file_path = os.path.join(agent_folder, "agent.py")
  agent_config_file_path = os.path.join(agent_folder, "root_agent.yaml")

  with open(dotenv_file_path, "w", encoding="utf-8") as f:
    lines = []
    if google_cloud_project and google_cloud_region:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=1")
    elif google_api_key:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=0")
    if google_api_key:
      lines.append(f"GOOGLE_API_KEY={google_api_key}")
    if google_cloud_project:
      lines.append(f"GOOGLE_CLOUD_PROJECT={google_cloud_project}")
    if google_cloud_region:
      lines.append(f"GOOGLE_CLOUD_LOCATION={google_cloud_region}")
    f.write("\n".join(lines))

  if type == "config":
    with open(agent_config_file_path, "w", encoding="utf-8") as f:
      f.write(_AGENT_CONFIG_TEMPLATE.format(model_name=model))
    with open(init_file_path, "w", encoding="utf-8") as f:
      f.write("")
    click.secho(
        _SUCCESS_MSG_CONFIG.format(agent_folder=agent_folder),
        fg="green",
    )
  else:
    with open(init_file_path, "w", encoding="utf-8") as f:
      f.write(_INIT_PY_TEMPLATE)

    with open(agent_py_file_path, "w", encoding="utf-8") as f:
      f.write(_AGENT_PY_TEMPLATE.format(model_name=model))
    click.secho(
        _SUCCESS_MSG_CODE.format(agent_folder=agent_folder),
        fg="green",
    )


def _prompt_for_model() -> str:
  model_choice = click.prompt(
      """\
Choose a model for the root agent:
1. gemini-2.5-flash
2. Other models (fill later)
Choose model""",
      type=click.Choice(["1", "2"]),
  )
  if model_choice == "1":
    return "gemini-2.5-flash"
  else:
    click.secho(_OTHER_MODEL_MSG, fg="green")
    return "<FILL_IN_MODEL>"


def _prompt_to_choose_backend(
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  """Prompts user to choose backend.

  Returns:
    A tuple of (google_api_key, google_cloud_project, google_cloud_region).
  """
  backend_choice = click.prompt(
      "1. Google AI\n2. Vertex AI\n3. Login with Google\nChoose a backend",
      type=click.Choice(["1", "2", "3"]),
  )
  if backend_choice == "1":
    google_api_key = _prompt_for_google_api_key(google_api_key)
  elif backend_choice == "2":
    click.secho(_GOOGLE_CLOUD_SETUP_MSG, fg="green")
    google_cloud_project = _prompt_for_google_cloud(google_cloud_project)
    google_cloud_region = _prompt_for_google_cloud_region(google_cloud_region)
  elif backend_choice == "3":
    google_api_key, google_cloud_project, google_cloud_region = (
        _handle_login_with_google()
    )
  return google_api_key, google_cloud_project, google_cloud_region


def _handle_login_with_google() -> (
    Tuple[Optional[str], Optional[str], Optional[str]]
):
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
      return api_key, project_id, region

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
      selected_project_id = _prompt_for_google_cloud(None)
    else:
      selected_project_id = projects[project_index - 1][0]
    region = _prompt_for_google_cloud_region(None)
    return None, selected_project_id, region
  else:
    if click.confirm(
        "No projects found automatically. Would you like to enter one"
        " manually?",
        default=False,
    ):
      selected_project_id = _prompt_for_google_cloud(None)
      region = _prompt_for_google_cloud_region(None)
      return None, selected_project_id, region

  # Check Express eligibility
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
      return api_key, project_id, region

  click.secho(_NOT_ELIGIBLE_MSG, fg="red")
  raise click.Abort()


def _prompt_to_choose_type() -> str:
  """Prompts user to choose type of agent to create."""
  type_choice = click.prompt(
      """\
Choose a type for the root agent:
1. YAML config (experimental, may change without notice)
2. Code
Choose type""",
      type=click.Choice(["1", "2"]),
  )
  if type_choice == "1":
    return "CONFIG"
  else:
    return "CODE"


def run_cmd(
    agent_name: str,
    *,
    model: Optional[str],
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
    type: Optional[str],
):
  """Runs `adk create` command to create agent template.

  Args:
    agent_name: str, The name of the agent.
    google_api_key: Optional[str], The Google API key for using Google AI as
      backend.
    google_cloud_project: Optional[str], The Google Cloud project for using
      VertexAI as backend.
    google_cloud_region: Optional[str], The Google Cloud region for using
      VertexAI as backend.
    type: Optional[str], Whether to define agent with config file or code.
  """
  app_name = os.path.basename(os.path.normpath(agent_name))
  try:
    validate_app_name(app_name)
  except ValueError as exc:
    raise click.BadParameter(str(exc)) from exc

  agent_folder = os.path.join(os.getcwd(), agent_name)
  # check folder doesn't exist or it's empty. Otherwise, throw
  if os.path.exists(agent_folder) and os.listdir(agent_folder):
    # Prompt user whether to override existing files using click
    if not click.confirm(
        f"Non-empty folder already exist: '{agent_folder}'\n"
        "Override existing content?",
        default=False,
    ):
      raise click.Abort()

  if not model:
    model = _prompt_for_model()

  if not google_api_key and not (google_cloud_project and google_cloud_region):
    if model.startswith("gemini"):
      google_api_key, google_cloud_project, google_cloud_region = (
          _prompt_to_choose_backend(
              google_api_key, google_cloud_project, google_cloud_region
          )
      )

  if not type:
    type = _prompt_to_choose_type()

  _generate_files(
      agent_folder,
      google_api_key=google_api_key,
      google_cloud_project=google_cloud_project,
      google_cloud_region=google_cloud_region,
      model=model,
      type=type.lower(),
  )

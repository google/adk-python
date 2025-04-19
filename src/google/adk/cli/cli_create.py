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

import os
import subprocess
from typing import Optional
from typing import Tuple

import click

_INIT_PY_TEMPLATE = """\
from . import agent
"""

_AGENT_PY_TEMPLATE = """\
from google.adk.agents import Agent

root_agent = Agent(
    model='{model_name}',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
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

_SUCCESS_MSG = """
Agent created in {agent_folder}:
- .env
- __init__.py
- agent.py
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
):
  """Generates a folder name for the agent."""
  os.makedirs(agent_folder, exist_ok=True)

  dotenv_file_path = os.path.join(agent_folder, ".env")
  init_file_path = os.path.join(agent_folder, "__init__.py")
  agent_file_path = os.path.join(agent_folder, "agent.py")

  with open(dotenv_file_path, "w", encoding="utf-8") as f:
    lines = []
    if google_api_key:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=0")
    elif google_cloud_project and google_cloud_region:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=1")
    if google_api_key:
      lines.append(f"GOOGLE_API_KEY={google_api_key}")
    if google_cloud_project:
      lines.append(f"GOOGLE_CLOUD_PROJECT={google_cloud_project}")
    if google_cloud_region:
      lines.append(f"GOOGLE_CLOUD_LOCATION={google_cloud_region}")
    f.write("\n".join(lines))

  with open(init_file_path, "w", encoding="utf-8") as f:
    f.write(_INIT_PY_TEMPLATE)

  with open(agent_file_path, "w", encoding="utf-8") as f:
    f.write(_AGENT_PY_TEMPLATE.format(model_name=model))

  click.secho(
      _SUCCESS_MSG.format(agent_folder=agent_folder),
      fg="green",
  )


def _prompt_for_model() -> str:
  model_choice = click.prompt(
      """\
Choose a model for the root agent:
1. gemini-2.0-flash-001
2. Other models (fill later)
Choose model""",
      type=click.Choice(["1", "2"]),
  )
  if model_choice == "1":
    return "gemini-2.0-flash-001"
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
      "1. Google AI\n2. Vertex AI\nChoose a backend",
      type=click.Choice(["1", "2"]),
  )
  if backend_choice == "1":
    google_api_key = _prompt_for_google_api_key(google_api_key)
  elif backend_choice == "2":
    click.secho(_GOOGLE_CLOUD_SETUP_MSG, fg="green")
    google_cloud_project = _prompt_for_google_cloud(google_cloud_project)
    google_cloud_region = _prompt_for_google_cloud_region(google_cloud_region)
  return google_api_key, google_cloud_project, google_cloud_region


def run_cmd(
    agent_name: str,
    *,
    model: Optional[str],
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
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
  """
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

  _generate_files(
      agent_folder,
      google_api_key=google_api_key,
      google_cloud_project=google_cloud_project,
      google_cloud_region=google_cloud_region,
      model=model,
  )

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
from typing import Optional

import click

from ..apps.app import validate_app_name
from .utils import _onboarding

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


_OTHER_MODEL_MSG = """
Please see below guide to configure other models:
https://google.github.io/adk-docs/agents/models
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
      auth_info = _onboarding.prompt_to_choose_backend(
          google_api_key, google_cloud_project, google_cloud_region
      )
      if isinstance(auth_info, _onboarding.GoogleAIAuth):
        google_api_key = auth_info.api_key
      elif isinstance(auth_info, _onboarding.VertexAIAuth):
        google_cloud_project = auth_info.project_id
        google_cloud_region = auth_info.region
      elif isinstance(auth_info, _onboarding.ExpressModeAuth):
        google_api_key = auth_info.api_key
        google_cloud_project = auth_info.project_id
        google_cloud_region = auth_info.region

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

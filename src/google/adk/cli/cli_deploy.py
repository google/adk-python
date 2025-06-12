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
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import click

_DOCKERFILE_TEMPLATE = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Change ownership of /app to myuser
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT={gcp_project_id}
ENV GOOGLE_CLOUD_LOCATION={gcp_region}

# Set up environment variables - End

# Install ADK - Start
RUN pip install google-adk=={adk_version}
# Install ADK - End

# Copy agent - Start

COPY "agents/{app_name}/" "/app/agents/{app_name}/"
{install_agent_deps}

# Copy agent - End

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {service_option} {trace_to_cloud_option} "/app/agents"
"""

_AGENT_ENGINE_APP_TEMPLATE = """
from agent import root_agent
from vertexai.preview.reasoning_engines import AdkApp

adk_app = AdkApp(
  agent=root_agent,
  enable_tracing={trace_to_cloud_option},
)
"""


def _resolve_project(project_in_option: Optional[str]) -> str:
  if project_in_option:
    return project_in_option

  result = subprocess.run(
      ['gcloud', 'config', 'get-value', 'project'],
      check=True,
      capture_output=True,
      text=True,
  )
  project = result.stdout.strip()
  click.echo(f'Use default project: {project}')
  return project


def _get_service_option_by_adk_version(
    adk_version: str,
    session_uri: Optional[str],
    artifact_uri: Optional[str],
    memory_uri: Optional[str],
) -> str:
  """Returns service option string based on adk_version."""
  if adk_version >= '1.3.0':
    session_option = (
        f'--session_service_uri={session_uri}' if session_uri else ''
    )
    artifact_option = (
        f'--artifact_service_uri={artifact_uri}' if artifact_uri else ''
    )
    memory_option = f'--memory_service_uri={memory_uri}' if memory_uri else ''
    return f'{session_option} {artifact_option} {memory_option}'
  elif adk_version >= '1.2.0':
    session_option = f'--session_db_url={session_uri}' if session_uri else ''
    artifact_option = (
        f'--artifact_storage_uri={artifact_uri}' if artifact_uri else ''
    )
    return f'{session_option} {artifact_option}'
  else:
    return f'--session_db_url={session_uri}' if session_uri else ''


def to_cloud_run(
    *,
    agent_folder: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    verbosity: str,
    adk_version: str,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
):
  """Deploys an agent to Google Cloud Run.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - requirements.txt (optional, for additional dependencies)
  - ... (other required source files)

  The folder structure of temp_folder will be

  * dist/[google_adk wheel file]
  * agents/[app_name]/
    * agent source code from `agent_folder`

  Args:
    agent_folder: The folder (absolute path) containing the agent source code.
    project: Google Cloud project id.
    region: Google Cloud region.
    service_name: The service name in Cloud Run.
    app_name: The name of the app, by default, it's basename of `agent_folder`.
    temp_folder: The temp folder for the generated Cloud Run source files.
    port: The port of the ADK api server.
    trace_to_cloud: Whether to enable Cloud Trace.
    with_ui: Whether to deploy with UI.
    verbosity: The verbosity level of the CLI.
    adk_version: The ADK version to use in Cloud Run.
    session_service_uri: The URI of the session service.
    artifact_service_uri: The URI of the artifact service.
    memory_service_uri: The URI of the memory service.
  """
  app_name = app_name or os.path.basename(agent_folder)

  click.echo(f'Start generating Cloud Run source files in {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    shutil.copytree(agent_folder, agent_src_path)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = (
        f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
        if os.path.exists(requirements_txt_path)
        else ''
    )
    click.echo('Copying agent source code complete.')

    # create Dockerfile
    click.echo('Creating Dockerfile...')
    host_option = '--host=0.0.0.0' if adk_version > '0.5.0' else ''
    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=port,
        command='web' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        service_option=_get_service_option_by_adk_version(
            adk_version,
            session_service_uri,
            artifact_service_uri,
            memory_service_uri,
        ),
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
        adk_version=adk_version,
        host_option=host_option,
    )
    dockerfile_path = os.path.join(temp_folder, 'Dockerfile')
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
      f.write(
          dockerfile_content,
      )
    click.echo(f'Creating Dockerfile complete: {dockerfile_path}')

    # Deploy to Cloud Run
    click.echo('Deploying to Cloud Run...')
    region_options = ['--region', region] if region else []
    project = _resolve_project(project)
    subprocess.run(
        [
            'gcloud',
            'run',
            'deploy',
            service_name,
            '--source',
            temp_folder,
            '--project',
            project,
            *region_options,
            '--port',
            str(port),
            '--verbosity',
            verbosity,
            '--labels',
            'created-by=adk',
        ],
        check=True,
    )
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)


def to_agent_engine(
    *,
    agent_folder: str,
    temp_folder: str,
    adk_app: str,
    staging_bucket: str,
    trace_to_cloud: bool,
    project: Optional[str] = None,
    region: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    requirements_file: Optional[str] = None,
    env_file: Optional[str] = None,
):
  """Deploys an agent to Vertex AI Agent Engine.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - <adk_app>.py (optional, for customization; will be autogenerated otherwise)
  - requirements.txt (optional, for additional dependencies)
  - .env (optional, for environment variables)
  - ... (other required source files)

  The contents of `adk_app` should look something like:

  ```
  from agent import root_agent
  from vertexai.preview.reasoning_engines import AdkApp

  adk_app = AdkApp(
    agent=root_agent,
    enable_tracing=True,
  )
  ```

  Args:
    agent_folder (str): The folder (absolute path) containing the agent source
      code.
    temp_folder (str): The temp folder for the generated Agent Engine source
      files. It will be replaced with the generated files if it already exists.
    project (str): Google Cloud project id.
    region (str): Google Cloud region.
    staging_bucket (str): The GCS bucket for staging the deployment artifacts.
    trace_to_cloud (bool): Whether to enable Cloud Trace.
    requirements_file (str): The filepath to the `requirements.txt` file to use.
      If not specified, the `requirements.txt` file in the `agent_folder` will
      be used.
    env_file (str): The filepath to the `.env` file for environment variables.
      If not specified, the `.env` file in the `agent_folder` will be used. The
      values of `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` will be
      overridden by `project` and `region` if they are specified.
  """
  # remove temp_folder if it exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder)

  try:
    click.echo('Copying agent source code...')
    shutil.copytree(agent_folder, temp_folder)
    click.echo('Copying agent source code complete.')

    click.echo('Initializing Vertex AI...')
    import sys

    import vertexai
    from vertexai import agent_engines

    sys.path.append(temp_folder)
    project = _resolve_project(project)

    click.echo('Resolving files and dependencies...')
    if not requirements_file:
      # Attempt to read requirements from requirements.txt in the dir (if any).
      requirements_txt_path = os.path.join(temp_folder, 'requirements.txt')
      if not os.path.exists(requirements_txt_path):
        click.echo(f'Creating {requirements_txt_path}...')
        with open(requirements_txt_path, 'w', encoding='utf-8') as f:
          f.write('google-cloud-aiplatform[adk,agent_engines]')
        click.echo(f'Created {requirements_txt_path}')
      requirements_file = requirements_txt_path
    env_vars = None
    if not env_file:
      # Attempt to read the env variables from .env in the dir (if any).
      env_file = os.path.join(temp_folder, '.env')
    if os.path.exists(env_file):
      from dotenv import dotenv_values

      click.echo(f'Reading environment variables from {env_file}')
      env_vars = dotenv_values(env_file)
      if 'GOOGLE_CLOUD_PROJECT' in env_vars:
        env_project = env_vars.pop('GOOGLE_CLOUD_PROJECT')
        if env_project:
          if project:
            click.secho(
                'Ignoring GOOGLE_CLOUD_PROJECT in .env as `--project` was'
                ' explicitly passed and takes precedence',
                fg='yellow',
            )
          else:
            project = env_project
            click.echo(f'{project=} set by GOOGLE_CLOUD_PROJECT in {env_file}')
      if 'GOOGLE_CLOUD_LOCATION' in env_vars:
        env_region = env_vars.pop('GOOGLE_CLOUD_LOCATION')
        if env_region:
          if region:
            click.secho(
                'Ignoring GOOGLE_CLOUD_LOCATION in .env as `--region` was'
                ' explicitly passed and takes precedence',
                fg='yellow',
            )
          else:
            region = env_region
            click.echo(f'{region=} set by GOOGLE_CLOUD_LOCATION in {env_file}')

    vertexai.init(
        project=project,
        location=region,
        staging_bucket=staging_bucket,
    )
    click.echo('Vertex AI initialized.')

    adk_app_file = f'{adk_app}.py'
    with open(
        os.path.join(temp_folder, adk_app_file), 'w', encoding='utf-8'
    ) as f:
      f.write(
          _AGENT_ENGINE_APP_TEMPLATE.format(
              trace_to_cloud_option=trace_to_cloud
          )
      )
    click.echo(f'Created {os.path.join(temp_folder, adk_app_file)}')
    click.echo('Files and dependencies resolved')

    click.echo('Deploying to agent engine...')
    agent_engine = agent_engines.ModuleAgent(
        module_name=adk_app,
        agent_name='adk_app',
        register_operations={
            '': [
                'get_session',
                'list_sessions',
                'create_session',
                'delete_session',
            ],
            'async': [
                'async_get_session',
                'async_list_sessions',
                'async_create_session',
                'async_delete_session',
            ],
            'async_stream': ['async_stream_query'],
            'stream': ['stream_query', 'streaming_agent_run_with_events'],
        },
        sys_paths=[temp_folder[1:]],
    )

    agent_engines.create(
        agent_engine=agent_engine,
        requirements=requirements_file,
        display_name=display_name,
        description=description,
        env_vars=env_vars,
        extra_packages=[temp_folder],
    )
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)

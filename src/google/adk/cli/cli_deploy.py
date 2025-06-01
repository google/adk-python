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
from typing import Optional
from typing import Tuple

import click

from .config.dockerfile_template import _DOCKERFILE_TEMPLATE
from .deployers.deployer_factory import DeployerFactory


def run(
    *,
    agent_folder: str,
    provider: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    verbosity: str,
    session_db_url: str,
    artifact_storage_uri: Optional[str],
    adk_version: str,
    provider_args: Tuple[str],
    env: Tuple[str],
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
    provider: Target deployment platform (cloud_run, docker, etc).
    project: Google Cloud project id.
    region: Google Cloud region.
    service_name: The service name in Cloud Run.
    app_name: The name of the app, by default, it's basename of `agent_folder`.
    temp_folder: The temp folder for the generated Cloud Run source files.
    port: The port of the ADK api server.
    trace_to_cloud: Whether to enable Cloud Trace.
    with_ui: Whether to deploy with UI.
    verbosity: The verbosity level of the CLI.
    session_db_url: The database URL to connect the session.
    artifact_storage_uri: The artifact storage URI to store the artifacts.
    adk_version: The ADK version to use in Cloud Run.
    provider_args: The arguments specific to cloud provider
    env: The environment valriables provided

  """
  app_name = app_name or os.path.basename(agent_folder)
  mode = 'web' if with_ui else 'api_server'
  trace_to_cloud_option = '--trace_to_cloud' if trace_to_cloud else ''

  click.echo(f'Start generating deployment files in {temp_folder}')

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
        command=mode,
        install_agent_deps=install_agent_deps,
        session_db_option=f'--session_db_url={session_db_url}'
        if session_db_url
        else '',
        trace_to_cloud_option=trace_to_cloud_option,
        artifact_storage_option=f'--artifact_storage_uri={artifact_storage_uri}'
        if artifact_storage_uri
        else '',
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
    click.echo(f'Deploying to {provider}...')
    deployer = DeployerFactory.get_deployer(provider)
    deployer.deploy(
        agent_folder=agent_folder,
        temp_folder=temp_folder,
        service_name=service_name,
        provider_args=provider_args,
        env_vars=env,
        project=project,
        region=region,
        port=port,
        verbosity=verbosity,
    )

    click.echo(f'Deployment to {provider} complete.')

  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)

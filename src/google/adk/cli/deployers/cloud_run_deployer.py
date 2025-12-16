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
import subprocess
from typing import Optional
from typing import Tuple

import click

from ..deployers.base_deployer import Deployer

_IS_WINDOWS = os.name == 'nt'
_GCLOUD_CMD = 'gcloud.cmd' if _IS_WINDOWS else 'gcloud'

class CloudRunDeployer(Deployer):

  def deploy(
      self,
      agent_folder: str,
      temp_folder: str,
      service_name: str,
      provider_args: Tuple[str],  # optional for Deployer
      env_vars: Tuple[str],
      **kwargs,
  ):
    project = self._resolve_project(kwargs.get('project'))
    region = kwargs.get('region', 'us-central1')
    port = kwargs.get('port', 8000)
    verbosity = kwargs.get('verbosity', 'info')
    extra_gcloud_args = kwargs.get('extra_gcloud_args')
    log_level = kwargs.get('log_level')
    region_options = ['--region', region] if region else []

    # Build the set of args that ADK will manage
    adk_managed_args = {'--source', '--project', '--port', '--verbosity'}
    if region:
      adk_managed_args.add('--region')

    # Validate that extra gcloud args don't conflict with ADK-managed args
    self._validate_gcloud_extra_args(extra_gcloud_args, adk_managed_args)

    # Add environment variables
    env_vars_str = self.build_env_vars_string(env_vars)
    env_file_str = self.build_env_file_arg(agent_folder)
    if env_vars_str and env_file_str:
      env_vars_str += ',' + env_file_str
    elif not env_vars_str:
      env_vars_str = env_file_str

    env_vars_str = self.add_gcp_env_vars(env_vars_str, project, region)

    # Build the command with extra gcloud args
    gcloud_cmd = [
        _GCLOUD_CMD,
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
        '--set-env-vars',
        env_vars_str,
        '--verbosity',
        log_level.lower() if log_level else verbosity,
    ]

    # Handle labels specially - merge user labels with ADK label
    user_labels = []
    extra_args_without_labels = []

    if extra_gcloud_args:
      for arg in extra_gcloud_args:
        if arg.startswith('--labels='):
          # Extract user-provided labels
          user_labels_value = arg[9:]  # Remove '--labels=' prefix
          user_labels.append(user_labels_value)
        else:
          extra_args_without_labels.append(arg)

    # Combine ADK label with user labels
    all_labels = ['created-by=adk']
    all_labels.extend(user_labels)
    labels_arg = ','.join(all_labels)

    gcloud_cmd.extend(['--labels', labels_arg])

    # Add any remaining extra passthrough args
    gcloud_cmd.extend(extra_args_without_labels)

    subprocess.run(gcloud_cmd, check=True)

  def _resolve_project(self, project_in_option: str = None) -> str:
    """
    Resolves the Google Cloud project ID. If a project is provided in the options, it will use that.
    Otherwise, it retrieves the default project from the active gcloud configuration.

    Args:
        project_in_option: Optional project ID to override the default.

    Returns:
        str: The resolved project ID.
    """
    if project_in_option:
      return project_in_option

    try:
      result = subprocess.run(
          [_GCLOUD_CMD, 'config', 'get-value', 'project'],
          check=True,
          capture_output=True,
          text=True,
      )
      project = result.stdout.strip()
      if not project:
        raise click.ClickException('No project ID found in gcloud config.')

      click.echo(f'Using default project: {project}')
      return project
    except subprocess.CalledProcessError as e:
      raise click.ClickException(f'Failed to get project from gcloud: {e}')

  def _validate_gcloud_extra_args(
      self,
      extra_gcloud_args: Optional[tuple[str, ...]],
      adk_managed_args: set[str],
  ) -> None:
    """Validates that extra gcloud args don't conflict with ADK-managed args.

    This function dynamically checks for conflicts based on the actual args
    that ADK will set, rather than using a hardcoded list.

    Args:
      extra_gcloud_args: User-provided extra arguments for gcloud.
      adk_managed_args: Set of argument names that ADK will set automatically.
                      Should include '--' prefix (e.g., '--project').

    Raises:
      click.ClickException: If any conflicts are found.
    """
    if not extra_gcloud_args:
      return

    # Parse user arguments into a set of argument names for faster lookup
    user_arg_names = set()
    for arg in extra_gcloud_args:
      if arg.startswith('--'):
        # Handle both '--arg=value' and '--arg value' formats
        arg_name = arg.split('=')[0]
        user_arg_names.add(arg_name)

    # Check for conflicts with ADK-managed args
    conflicts = user_arg_names.intersection(adk_managed_args)

    if conflicts:
      conflict_list = ', '.join(f"'{arg}'" for arg in sorted(conflicts))
      if len(conflicts) == 1:
        raise click.ClickException(
            f"The argument {conflict_list} conflicts with ADK's automatic"
            ' configuration. ADK will set this argument automatically, so'
            ' please remove it from your command.'
        )
      else:
        raise click.ClickException(
            f"The arguments {conflict_list} conflict with ADK's automatic"
            ' configuration. ADK will set these arguments automatically, so'
            ' please remove them from your command.'
        )

  def build_env_vars_string(self, env_vars: Tuple[str]) -> str:
    """
    Returns a comma-separated string of 'KEY=value' entries
    from a tuple of environment variable strings.
    """
    valid_pairs = [item.strip() for item in env_vars if '=' in item]
    return ','.join(valid_pairs)

  def build_env_file_arg(self, agent_folder: str) -> str:
    """
    Reads the `.env` file (if present) and returns a comma-separated `KEY=VALUE` string
    for use with `--set-env-vars` in `gcloud run deploy`.
    """
    env_file_path = os.path.join(agent_folder, '.env')
    env_vars_str = ''

    if os.path.exists(env_file_path):
      with open(env_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

      env_vars = []
      for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
          key, value = line.split('=', 1)
          env_vars.append(f'{key}={value}')

      env_vars_str = ','.join(env_vars)

    return env_vars_str

  def add_gcp_env_vars(
      self, env_vars_str: str, project: str, region: str
  ) -> str:
    """
    Appends required Google-specific environment variables to the existing env var string.
    """
    extra_envs = [
        f'GOOGLE_GENAI_USE_VERTEXAI=1',
        f'GOOGLE_CLOUD_PROJECT={project}',
        f'GOOGLE_CLOUD_LOCATION={region}',
    ]

    if env_vars_str:
      return env_vars_str + ',' + ','.join(extra_envs)
    return ','.join(extra_envs)

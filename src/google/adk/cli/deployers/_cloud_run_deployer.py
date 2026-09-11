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

import click

from ._base_deployer import Deployer

_IS_WINDOWS = os.name == 'nt'
_GCLOUD_CMD = 'gcloud.cmd' if _IS_WINDOWS else 'gcloud'


class CloudRunDeployer(Deployer):

  def deploy(
      self,
      *,
      agent_folder: str,
      temp_folder: str,
      service_name: str,
      provider_args: tuple[str, ...] = (),
      env_vars: tuple[str, ...] = (),
      project: str | None = None,
      region: str | None = None,
      port: int = 8000,
      verbosity: str = 'info',
      extra_gcloud_args: tuple[str, ...] | None = None,
      log_level: str | None = None,
      with_cloud_run_sandbox: bool = False,
  ) -> None:

    # Load and process .env file
    env_file = os.path.join(agent_folder, '.env')
    file_env_vars = self._load_env_file(agent_folder)

    if 'GOOGLE_CLOUD_PROJECT' in file_env_vars:
      env_project = file_env_vars.pop('GOOGLE_CLOUD_PROJECT')
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

    if 'GOOGLE_CLOUD_LOCATION' in file_env_vars:
      env_region = file_env_vars.pop('GOOGLE_CLOUD_LOCATION')
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

    project = self._resolve_project(project)
    region_options = ['--region', region] if region else []

    # Build the set of args that ADK will manage
    adk_managed_args = {
        '--source',
        '--project',
        '--port',
        '--verbosity',
        '--update-env-vars',
    }
    if region:
      adk_managed_args.add('--region')
    if with_cloud_run_sandbox:
      adk_managed_args.add('--sandbox-launcher')

    # Combine and validate extra gcloud args & provider args
    all_extra_args = list(extra_gcloud_args or ()) + list(provider_args or ())
    self._validate_gcloud_extra_args(
        tuple(all_extra_args) if all_extra_args else None, adk_managed_args
    )

    # Add environment variables (precedence: defaults < --env flags)
    cli_env_str = self._build_env_vars_string(env_vars)
    env_vars_str = self._add_gcp_env_vars(cli_env_str, project, region)

    # Build the command with extra gcloud args
    gcloud_cmd = [_GCLOUD_CMD]
    if with_cloud_run_sandbox:
      # --sandbox-launcher is only supported on the beta release track.
      gcloud_cmd.append('beta')
    gcloud_cmd += [
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
        '--update-env-vars',
        env_vars_str,
        '--verbosity',
        log_level.lower() if log_level else verbosity,
    ]
    if with_cloud_run_sandbox:
      gcloud_cmd.append('--sandbox-launcher')

    # Handle labels specially - merge user labels with ADK label
    user_labels = []
    extra_args_without_labels = []

    if all_extra_args:
      for arg in all_extra_args:
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

  def _load_env_file(self, agent_folder: str) -> dict[str, str]:
    """Reads the `.env` file (if present) and returns a dictionary of key-value pairs."""
    env_file_path = os.path.join(agent_folder, '.env')
    if os.path.exists(env_file_path):
      from dotenv import dotenv_values

      parsed = dotenv_values(env_file_path)
      return {
          k: str(v)
          for k, v in parsed.items()
          if k is not None and v is not None
      }
    return {}

  def _resolve_project(self, project_in_option: str | None = None) -> str:
    """Resolves the Google Cloud project ID.

    If a project is provided in the options, it will use that.
    Otherwise, it retrieves the default project from the active gcloud
    configuration.

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
      extra_gcloud_args: tuple[str, ...] | None,
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

  def _build_env_vars_string(self, env_vars: tuple[str, ...] | None) -> str:
    """Returns a comma-separated string of 'KEY=value' entries from a tuple of environment variable strings."""
    if not env_vars:
      return ''
    valid_pairs = [item.strip() for item in env_vars if '=' in item]
    return ','.join(valid_pairs)

  def _add_gcp_env_vars(
      self,
      env_vars_str: str,
      project: str | None = None,
      region: str | None = None,
  ) -> str:
    """Combines default Google Cloud environment variables with the user env var string, with user values taking precedence."""
    env_dict = {
        'GOOGLE_GENAI_USE_ENTERPRISE': '1',
    }
    if project:
      env_dict['GOOGLE_CLOUD_PROJECT'] = project
    if region:
      env_dict['GOOGLE_CLOUD_LOCATION'] = region

    if env_vars_str:
      for pair in env_vars_str.split(','):
        pair = pair.strip()
        if '=' in pair:
          k, v = pair.split('=', 1)
          env_dict[k.strip()] = v.strip()

    return ','.join(f'{k}={v}' for k, v in env_dict.items())

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


class DockerDeployer(Deployer):

  def deploy(
      self,
      *,
      agent_folder: str,
      temp_folder: str,
      service_name: str,
      provider_args: tuple[str, ...] = (),
      env_vars: tuple[str, ...] = (),
      port: int = 8000,
      project: str | None = None,
      region: str | None = None,
      verbosity: str = 'info',
      extra_gcloud_args: tuple[str, ...] | None = None,
      log_level: str | None = None,
      with_cloud_run_sandbox: bool = False,
  ) -> None:
    del (
        project,
        region,
        verbosity,
        extra_gcloud_args,
        log_level,
        with_cloud_run_sandbox,
    )
    image_name = f'adk-python-{service_name.lower()}'

    click.echo('Deploying to Local Docker')

    # Build Docker image
    subprocess.run(
        ['docker', 'build', '-t', image_name, temp_folder],
        check=True,
    )

    env_args = self._get_env_file_arg(agent_folder)
    env_args.extend(self._get_cli_env_args(env_vars))

    provider_args_list = list(provider_args) if provider_args else []

    # Run Docker container
    subprocess.run(
        [
            'docker',
            'run',
            '-d',
            '-p',
            f'{port}:{port}',
            *provider_args_list,
            *env_args,
            image_name,
        ],
        check=True,
    )
    click.echo(f'Container running locally at http://localhost:{port}')

  def _get_cli_env_args(self, env_vars: tuple[str, ...] | None) -> list[str]:
    """Converts tuple of environment variable strings into Docker -e arguments."""
    if not env_vars:
      return []
    env_args = []
    for item in env_vars:
      item = item.strip()
      if item:
        env_args.extend(['-e', item])
    return env_args

  def _get_env_file_arg(self, agent_folder: str) -> list[str]:
    """Returns Docker `--env-file` argument if .env file exists in agent_folder."""
    env_file_path = os.path.join(agent_folder, '.env')
    if os.path.exists(env_file_path):
      return ['--env-file', env_file_path]
    return []

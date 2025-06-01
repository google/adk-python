import os
import subprocess
from typing import List
from typing import Tuple

import click

from ..deployers.base_deployer import Deployer


class DockerDeployer(Deployer):

  def deploy(
      self,
      agent_folder: str,
      temp_folder: str,
      service_name: str,
      provider_args: Tuple[str],  # optional for Deployer
      env_vars: Tuple[str],
      **kwargs,
  ):
    port = kwargs.get('port', 8000)
    image_name = f'adk-python-{service_name.lower()}'

    click.echo('Deploying to Local Docker')

    # Build Docker image
    subprocess.run(
        ['docker', 'build', '-t', image_name, temp_folder],
        check=True,
    )

    env_args = self.get_cli_env_args(env_vars)
    env_args.extend(self.get_env_file_arg(agent_folder))

    # Run Docker container
    subprocess.run(
        ['docker', 'run', '-d', '-p', f'{port}:{port}', *env_args, image_name],
        check=True,
    )
    click.echo(f'Container running locally at http://localhost:{port}')

  def get_cli_env_args(self, env_vars: Tuple[str]) -> List[str]:
    """Converts tuple of 'KEY=value' strings into Docker -e arguments."""
    env_args = []
    for item in env_vars:
      if '=' in item:
        key, value = item.split('=', 1)
        env_args.extend(['-e', f'{key}={value}'])
    return env_args

  def get_env_file_arg(self, agent_folder: str) -> List[str]:
    """Returns Docker `--env-file` argument if .env file exists in agent_folder."""
    env_args = []
    env_file_path = os.path.join(agent_folder, '.env')
    if os.path.exists(env_file_path):
      env_args.extend(['--env-file', env_file_path])
    return env_args

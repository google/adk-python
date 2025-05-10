# deployers/gcp_deployer.py

import subprocess
import os
import click
from typing import Tuple
from ..deployers.base_deployer import Deployer

class GCPDeployer(Deployer):
    def deploy(self, 
               agent_folder: str, 
               temp_folder: str, 
               service_name: str, 
               provider_args: Tuple[str],        # optional for Deployer
               env_vars: Tuple[str], 
               **kwargs):
        project = self._resolve_project(kwargs.get('project'))
        region = kwargs.get('region', 'us-central1')
        port = kwargs.get('port', 8000)
        verbosity = kwargs.get('verbosity', 'info')
        region_options = ['--region', region] if region else []

        # Add environment variables
        env_vars_str = self.build_env_vars_string(env_vars)
        env_file_str = self.build_env_file_arg(agent_folder)
        if env_vars_str and env_file_str:
            env_vars_str += "," + env_file_str
        elif not env_vars_str:
            env_vars_str = env_file_str

        env_vars_str = self.add_required_env_vars(env_vars_str, project, region)

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
                '--set-env-vars',
                env_vars_str,
                '--verbosity',
                verbosity,
                '--labels',
                'created-by=adk',
            ],
            check=True,
        )

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
                ['gcloud', 'config', 'get-value', 'project'],
                check=True,
                capture_output=True,
                text=True,
            )
            project = result.stdout.strip()
            if not project:
                raise ValueError("No project ID found in gcloud config.")
            click.echo(f'Using default project: {project}')
            return project
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Failed to get project from gcloud: {e}")
        except ValueError as e:
            raise click.ClickException(str(e))
        
    def build_env_vars_string(self, env_vars: Tuple[str]) -> str:
        """
        Returns a comma-separated string of 'KEY=value' entries 
        from a tuple of environment variable strings.
        """
        valid_pairs = [item for item in env_vars if "=" in item]
        return ",".join(valid_pairs)

    def build_env_file_arg(self, agent_folder: str) -> str:
        """
        Reads the `.env` file (if present) and returns a comma-separated `KEY=VALUE` string
        for use with `--set-env-vars` in `gcloud run deploy`.
        """
        env_file_path = os.path.join(agent_folder, ".env")
        env_vars_str = ""

        if os.path.exists(env_file_path):
            with open(env_file_path, "r") as f:
                lines = f.readlines()

            env_vars = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars.append(f"{key}={value}")

            env_vars_str = ",".join(env_vars)

        return env_vars_str
    
    def add_required_env_vars(self, env_vars_str: str, project: str, region: str) -> str:
        """
        Appends required Google-specific environment variables to the existing env var string.
        """
        extra_envs = [
            f"GOOGLE_CLOUD_PROJECT={project}",
            f"GOOGLE_CLOUD_LOCATION={region}",
        ]

        if env_vars_str:
            return env_vars_str + "," + ",".join(extra_envs)
        return ",".join(extra_envs)


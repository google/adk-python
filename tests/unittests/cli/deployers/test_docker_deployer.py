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

"""Tests for run functionality in docker_deployer."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.cli.deployers._docker_deployer import DockerDeployer
import pytest


@pytest.fixture
def docker_deployer():
  return DockerDeployer()


@patch('subprocess.run')
def test_deploy_success(mock_run, docker_deployer):
  agent_folder = 'path/to/agent'
  temp_folder = 'path/to/temp'
  service_name = 'test-service'
  provider_args = ()
  env_vars = ('ENV_VAR1=value1', 'ENV_VAR2=value2')
  port = 8080

  docker_deployer.deploy(
      agent_folder=agent_folder,
      temp_folder=temp_folder,
      service_name=service_name,
      provider_args=provider_args,
      env_vars=env_vars,
      port=port,
  )

  expected_build_cmd = [
      'docker',
      'build',
      '-t',
      'adk-python-test-service',
      temp_folder,
  ]
  mock_run.assert_any_call(expected_build_cmd, check=True)

  expected_run_cmd = [
      'docker',
      'run',
      '-d',
      '-p',
      f'{port}:{port}',
      '-e',
      'ENV_VAR1=value1',
      '-e',
      'ENV_VAR2=value2',
      'adk-python-test-service',
  ]
  mock_run.assert_any_call(expected_run_cmd, check=True)


@patch('subprocess.run')
def test_deploy_with_provider_args(mock_run, docker_deployer):
  agent_folder = 'path/to/agent'
  temp_folder = 'path/to/temp'
  service_name = 'test-service'
  provider_args = ('--restart=always', '--cpus=2')
  env_vars = ('ENV_VAR1=value1',)
  port = 8080

  docker_deployer.deploy(
      agent_folder=agent_folder,
      temp_folder=temp_folder,
      service_name=service_name,
      provider_args=provider_args,
      env_vars=env_vars,
      port=port,
  )

  expected_run_cmd = [
      'docker',
      'run',
      '-d',
      '-p',
      f'{port}:{port}',
      '--restart=always',
      '--cpus=2',
      '-e',
      'ENV_VAR1=value1',
      'adk-python-test-service',
  ]
  mock_run.assert_any_call(expected_run_cmd, check=True)


@patch('subprocess.run')
def test_deploy_with_env_file(mock_run, docker_deployer, tmp_path):
  # Create a .env file for testing
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write('ENV_VAR1=value1\nENV_VAR2=value2\n')

  agent_folder = str(tmp_path)
  temp_folder = 'path/to/temp'
  service_name = 'test-service'
  provider_args = ()
  env_vars = ()
  port = 8080

  docker_deployer.deploy(
      agent_folder=agent_folder,
      temp_folder=temp_folder,
      service_name=service_name,
      provider_args=provider_args,
      env_vars=env_vars,
      port=port,
  )

  # Check that subprocess.run was called to run the Docker container with --env-file
  expected_run_cmd = [
      'docker',
      'run',
      '-d',
      '-p',
      f'{port}:{port}',
      '--env-file',
      str(env_file_path),
      'adk-python-test-service',
  ]
  mock_run.assert_any_call(expected_run_cmd, check=True)


@patch('subprocess.run')
def test_deploy_without_env_file(mock_run, docker_deployer, tmp_path):
  agent_folder = str(tmp_path)
  temp_folder = 'path/to/temp'
  service_name = 'test-service'
  provider_args = ()
  env_vars = ('ENV_VAR1=value1',)
  port = 8080

  docker_deployer.deploy(
      agent_folder=agent_folder,
      temp_folder=temp_folder,
      service_name=service_name,
      provider_args=provider_args,
      env_vars=env_vars,
      port=port,
  )

  # Check that subprocess.run was called to run the Docker container without --env-file
  expected_run_cmd = [
      'docker',
      'run',
      '-d',
      '-p',
      f'{port}:{port}',
      '-e',
      'ENV_VAR1=value1',
      'adk-python-test-service',
  ]
  mock_run.assert_any_call(expected_run_cmd, check=True)


# Test helper functions
def test_get_cli_env_args(docker_deployer):
  env_vars = ('ENV_VAR1=value1', 'ENV_VAR2=value2')
  result = docker_deployer._get_cli_env_args(env_vars)
  assert result == ['-e', 'ENV_VAR1=value1', '-e', 'ENV_VAR2=value2']


def test_get_cli_env_args_bare_var(docker_deployer):
  env_vars = ('HOST_VAR', 'KEY=value')
  result = docker_deployer._get_cli_env_args(env_vars)
  assert result == ['-e', 'HOST_VAR', '-e', 'KEY=value']


def test_get_env_file_arg_with_env_file(docker_deployer, tmp_path):
  # Create a .env file for testing
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write('ENV_VAR1=value1\nENV_VAR2=value2\n')

  result = docker_deployer._get_env_file_arg(str(tmp_path))
  assert result == ['--env-file', str(env_file_path)]


def test_get_env_file_arg_without_env_file(docker_deployer, tmp_path):
  result = docker_deployer._get_env_file_arg(str(tmp_path))
  assert result == []


def test_get_cli_env_args_none(docker_deployer):
  result = docker_deployer._get_cli_env_args(None)
  assert result == []


@patch('subprocess.run')
def test_deploy_with_env_file_and_cli_env_vars_order(
    mock_run, docker_deployer, tmp_path
):
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write('FOO=from_file\n')

  agent_folder = str(tmp_path)
  temp_folder = 'path/to/temp'
  service_name = 'test-service'
  provider_args = ()
  env_vars = ('FOO=from_cli',)
  port = 8080

  docker_deployer.deploy(
      agent_folder=agent_folder,
      temp_folder=temp_folder,
      service_name=service_name,
      provider_args=provider_args,
      env_vars=env_vars,
      port=port,
  )

  expected_run_cmd = [
      'docker',
      'run',
      '-d',
      '-p',
      f'{port}:{port}',
      '--env-file',
      str(env_file_path),
      '-e',
      'FOO=from_cli',
      'adk-python-test-service',
  ]
  mock_run.assert_any_call(expected_run_cmd, check=True)


def test_deploy_rejects_unknown_kwarg(docker_deployer):
  kwargs = {'projct': 'misspelled-project'}
  with pytest.raises(TypeError):
    docker_deployer.deploy(
        agent_folder='path/to/agent',
        temp_folder='path/to/temp',
        service_name='test-service',
        **kwargs,
    )


def test_deploy_accepts_deployer_kwargs(docker_deployer):
  with patch('subprocess.run'):
    docker_deployer.deploy(
        agent_folder='path/to/agent',
        temp_folder='path/to/temp',
        service_name='test-service',
        extra_gcloud_args=('--flag',),
        with_cloud_run_sandbox=True,
        project='my-proj',
        region='us-central1',
        verbosity='info',
        log_level='info',
    )

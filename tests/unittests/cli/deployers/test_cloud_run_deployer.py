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

"""Tests for run functionality in cloud_run_deployer."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.google.adk.cli.deployers.cloud_run_deployer import CloudRunDeployer


@pytest.fixture
def cloud_run_deployer():
  return CloudRunDeployer()


@patch('subprocess.run')
def test_deploy_success(mock_run, cloud_run_deployer):
  cloud_run_deployer.deploy(
      agent_folder='path/to/agent',
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=('ENV_VAR1=value1', 'ENV_VAR2=value2'),
      project='test-project',
      region='us-central1',
      port=8080,
      log_level='info',
  )

  # Check that subprocess.run was called with the expected command
  expected_cmd = [
      'gcloud',
      'run',
      'deploy',
      'test-service',
      '--source',
      'path/to/temp',
      '--project',
      'test-project',
      '--region',
      'us-central1',
      '--port',
      '8080',
      '--set-env-vars',
      'ENV_VAR1=value1,ENV_VAR2=value2,GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=test-project,GOOGLE_CLOUD_LOCATION=us-central1',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


# Test helper functions
def test_build_env_vars_string(cloud_run_deployer):
  env_vars = ('ENV_VAR1=value1', 'ENV_VAR2=value2')
  result = cloud_run_deployer.build_env_vars_string(env_vars)
  assert result == 'ENV_VAR1=value1,ENV_VAR2=value2'


def test_build_env_file_arg_without_env_file(cloud_run_deployer, tmp_path):
  result = cloud_run_deployer.build_env_file_arg(str(tmp_path))
  assert result == ''


def test_build_env_file_arg_with_env_file(cloud_run_deployer, tmp_path):
  # Create a .env file for testing
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write('ENV_VAR1=value1,ENV_VAR2=value2\n')

  result = cloud_run_deployer.build_env_file_arg(str(tmp_path))
  assert result == 'ENV_VAR1=value1,ENV_VAR2=value2'


def test_validate_gcloud_extra_args_no_conflicts(cloud_run_deployer):
  extra_gcloud_args = ['--timeout=600']
  adk_managed_args = {'--project', '--region'}
  try:
    cloud_run_deployer._validate_gcloud_extra_args(
        extra_gcloud_args, adk_managed_args
    )
  except Exception:
    pytest.fail('Unexpected exception raised')


def test_validate_gcloud_extra_args_with_conflicts(cloud_run_deployer):
  extra_gcloud_args = ['--project=test-project']
  adk_managed_args = {'--project', '--region'}
  with pytest.raises(Exception) as excinfo:
    cloud_run_deployer._validate_gcloud_extra_args(
        extra_gcloud_args, adk_managed_args
    )
  assert "conflicts with ADK's automatic configuration" in str(excinfo.value)


def test_resolve_project_with_provided_project(cloud_run_deployer):
  project = cloud_run_deployer._resolve_project('test-project')
  assert project == 'test-project'


@patch('subprocess.run')
def test_resolve_project_without_provided_project(mock_run, cloud_run_deployer):
  mock_run.return_value.stdout = 'default-project\n'
  project = cloud_run_deployer._resolve_project()
  assert project == 'default-project'


@patch('subprocess.run')
def test_resolve_project_error(mock_run, cloud_run_deployer):
  mock_run.side_effect = subprocess.CalledProcessError(1, 'gcloud')
  with pytest.raises(Exception) as excinfo:
    cloud_run_deployer._resolve_project()
  assert 'Failed to get project from gcloud' in str(excinfo.value)

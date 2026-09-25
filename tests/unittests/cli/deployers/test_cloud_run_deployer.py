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

import click
from google.adk.cli.deployers._cloud_run_deployer import CloudRunDeployer
import pytest


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
      '--update-env-vars',
      'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=test-project,GOOGLE_CLOUD_LOCATION=us-central1,ENV_VAR1=value1,ENV_VAR2=value2',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


@patch('subprocess.run')
def test_deploy_region_none(mock_run, cloud_run_deployer):
  cloud_run_deployer.deploy(
      agent_folder='path/to/agent',
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=(),
      project='test-project',
      region=None,
      port=8080,
      log_level='info',
  )

  expected_cmd = [
      'gcloud',
      'run',
      'deploy',
      'test-service',
      '--source',
      'path/to/temp',
      '--project',
      'test-project',
      '--port',
      '8080',
      '--update-env-vars',
      'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=test-project',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


@patch('subprocess.run')
def test_deploy_with_provider_args(mock_run, cloud_run_deployer):
  cloud_run_deployer.deploy(
      agent_folder='path/to/agent',
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=('--min-instances=1', '--max-instances=5'),
      env_vars=(),
      project='test-project',
      region='us-central1',
      port=8080,
      log_level='info',
  )

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
      '--update-env-vars',
      'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=test-project,GOOGLE_CLOUD_LOCATION=us-central1',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
      '--min-instances=1',
      '--max-instances=5',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


@patch('subprocess.run')
def test_deploy_with_env_file_resolves_project_and_region(
    mock_run, cloud_run_deployer, tmp_path
):
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write(
        'GOOGLE_CLOUD_PROJECT=env-project\n'
        'GOOGLE_CLOUD_LOCATION=europe-west1\n'
        'CUSTOM_VAR=val\n'
    )

  cloud_run_deployer.deploy(
      agent_folder=str(tmp_path),
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=(),
      project=None,
      region=None,
      port=8080,
      log_level='info',
  )

  expected_cmd = [
      'gcloud',
      'run',
      'deploy',
      'test-service',
      '--source',
      'path/to/temp',
      '--project',
      'env-project',
      '--region',
      'europe-west1',
      '--port',
      '8080',
      '--update-env-vars',
      'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=env-project,GOOGLE_CLOUD_LOCATION=europe-west1',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


@patch('subprocess.run')
def test_deploy_cli_options_take_precedence_over_env_file(
    mock_run, cloud_run_deployer, tmp_path
):
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write(
        'GOOGLE_CLOUD_PROJECT=env-project\nGOOGLE_CLOUD_LOCATION=europe-west1\n'
    )

  cloud_run_deployer.deploy(
      agent_folder=str(tmp_path),
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=(),
      project='cli-project',
      region='us-central1',
      port=8080,
      log_level='info',
  )

  expected_cmd = [
      'gcloud',
      'run',
      'deploy',
      'test-service',
      '--source',
      'path/to/temp',
      '--project',
      'cli-project',
      '--region',
      'us-central1',
      '--port',
      '8080',
      '--update-env-vars',
      'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=cli-project,GOOGLE_CLOUD_LOCATION=us-central1',
      '--verbosity',
      'info',
      '--labels',
      'created-by=adk',
  ]
  mock_run.assert_called_once_with(expected_cmd, check=True)


@patch('subprocess.run')
def test_deploy_user_env_overrides_enterprise_default(
    mock_run, cloud_run_deployer
):
  cloud_run_deployer.deploy(
      agent_folder='path/to/agent',
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=('GOOGLE_GENAI_USE_ENTERPRISE=0',),
      project='test-project',
      region='us-central1',
      port=8080,
      log_level='info',
  )
  cmd = mock_run.call_args[0][0]
  env_vars_str = cmd[cmd.index('--update-env-vars') + 1]
  assert (
      env_vars_str
      == 'GOOGLE_GENAI_USE_ENTERPRISE=0,GOOGLE_CLOUD_PROJECT=test-project,GOOGLE_CLOUD_LOCATION=us-central1'
  )


# Test helper functions
def test_build_env_vars_string(cloud_run_deployer):
  env_vars = ('ENV_VAR1=value1', 'ENV_VAR2=value2')
  result = cloud_run_deployer._build_env_vars_string(env_vars)
  assert result == 'ENV_VAR1=value1,ENV_VAR2=value2'


def test_add_gcp_env_vars_with_region(cloud_run_deployer):
  result = cloud_run_deployer._add_gcp_env_vars(
      'FOO=bar', project='my-proj', region='us-central1'
  )
  assert (
      result
      == 'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=my-proj,GOOGLE_CLOUD_LOCATION=us-central1,FOO=bar'
  )


def test_add_gcp_env_vars_without_region(cloud_run_deployer):
  result = cloud_run_deployer._add_gcp_env_vars(
      'FOO=bar', project='my-proj', region=None
  )
  assert (
      result
      == 'GOOGLE_GENAI_USE_ENTERPRISE=1,GOOGLE_CLOUD_PROJECT=my-proj,FOO=bar'
  )
  assert 'GOOGLE_CLOUD_LOCATION' not in result


def test_add_gcp_env_vars_user_override(cloud_run_deployer):
  result = cloud_run_deployer._add_gcp_env_vars(
      'GOOGLE_GENAI_USE_ENTERPRISE=0', project='my-proj', region='us-central1'
  )
  assert (
      result
      == 'GOOGLE_GENAI_USE_ENTERPRISE=0,GOOGLE_CLOUD_PROJECT=my-proj,GOOGLE_CLOUD_LOCATION=us-central1'
  )


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


def test_validate_gcloud_extra_args_update_env_vars_conflict(
    cloud_run_deployer,
):
  extra_gcloud_args = ['--update-env-vars=FOO=bar']
  adk_managed_args = {
      '--source',
      '--project',
      '--port',
      '--verbosity',
      '--update-env-vars',
  }
  with pytest.raises(click.ClickException) as excinfo:
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


@patch('subprocess.run')
def test_deploy_with_env_file_location_when_region_omitted(
    mock_run, cloud_run_deployer, tmp_path
):
  env_file_path = tmp_path / '.env'
  with open(env_file_path, 'w') as f:
    f.write('GOOGLE_CLOUD_LOCATION=europe-west1\n')

  cloud_run_deployer.deploy(
      agent_folder=str(tmp_path),
      temp_folder='path/to/temp',
      service_name='test-service',
      provider_args=(),
      env_vars=(),
      project='test-project',
      port=8080,
      log_level='info',
  )

  assert mock_run.call_count == 1
  cmd = mock_run.call_args[0][0]
  assert '--region' in cmd
  assert cmd[cmd.index('--region') + 1] == 'europe-west1'


def test_build_env_vars_string_none(cloud_run_deployer):
  result = cloud_run_deployer._build_env_vars_string(None)
  assert result == ''


def test_deploy_rejects_unknown_kwarg(cloud_run_deployer):
  kwargs = {'projct': 'misspelled-project'}
  with pytest.raises(TypeError):
    cloud_run_deployer.deploy(
        agent_folder='path/to/agent',
        temp_folder='path/to/temp',
        service_name='test-service',
        **kwargs,
    )

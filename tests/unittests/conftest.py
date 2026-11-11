# Copyright 2026 Google LLC
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

import asyncio
import os
import subprocess
import sys
from typing import Any

import pytest

pytest.register_assert_rewrite('google.adk.cli.agent_test_runner')

# autospec of a lazily imported name specs its proxy instead of the class.
if sys.version_info[:2] == (3, 15):
  sys.set_lazy_imports_filter(lambda *_: False)

from google.adk.utils import _gcp_metadata
from pytest import fixture
from pytest import FixtureRequest
from pytest import hookimpl
from pytest import Metafunc

_ENV_VARS = {
    'GOOGLE_API_KEY': 'fake_google_api_key',
    'GOOGLE_CLOUD_PROJECT': 'fake_google_cloud_project',
    'GOOGLE_CLOUD_LOCATION': 'fake_google_cloud_location',
    'ADK_ALLOW_WIP_FEATURES': 'true',
    'ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS': 'true',
}

ENV_SETUPS = {
    'GOOGLE_AI': {
        'GOOGLE_GENAI_USE_ENTERPRISE': '0',
        **_ENV_VARS,
    },
    'VERTEX': {
        'GOOGLE_GENAI_USE_ENTERPRISE': '1',
        **_ENV_VARS,
    },
}


@fixture
def env_variables(request: FixtureRequest):
  # Set up the environment
  env_name: str = request.param
  envs = ENV_SETUPS[env_name]
  original_env = {key: os.environ.get(key) for key in envs}
  os.environ.update(envs)

  yield  # Run the test

  # Restore the environment
  for key in envs:
    if (original_val := original_env.get(key)) is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = original_val


# Store original environment variables to restore later
_original_env = {}


@hookimpl(tryfirst=True)
def pytest_sessionstart(session):
  """Set up environment variables at the beginning of the test session."""
  if not ENV_SETUPS:
    return
  # Use the first env setup to initialize environment for module-level imports
  env_name = next(iter(ENV_SETUPS.keys()))
  envs = ENV_SETUPS[env_name]
  global _original_env
  _original_env = {key: os.environ.get(key) for key in envs}
  os.environ.update(envs)


@hookimpl(trylast=True)
def pytest_sessionfinish(session):
  """Restore original environment variables at the end of the test session."""
  global _original_env
  for key, original_val in _original_env.items():
    if original_val is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = original_val
  _original_env = {}


@hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: Metafunc):
  """Generate test cases for each environment setup."""
  if env_variables.__name__ in metafunc.fixturenames:
    if not _is_explicitly_marked(env_variables.__name__, metafunc):
      metafunc.parametrize(
          env_variables.__name__, ENV_SETUPS.keys(), indirect=True
      )


def _is_explicitly_marked(mark_name: str, metafunc: Metafunc) -> bool:
  if hasattr(metafunc.function, 'pytestmark'):
    for mark in metafunc.function.pytestmark:
      if mark.name == 'parametrize' and mark.args[0] == mark_name:
        return True
  return False


@pytest.fixture(autouse=True)
def mock_gcp_metadata_off(request):
  """Ensures GCP metadata defaults are not applied during most tests."""
  if 'test_gcp_metadata' in request.module.__name__:
    yield
    return

  from google.auth import _cloud_sdk

  orig_get_project_id = _gcp_metadata.get_project_id_from_metadata
  orig_cached_project_id = _gcp_metadata._cached_project_id
  orig_get_adc_path = _cloud_sdk.get_application_default_credentials_path

  _gcp_metadata.get_project_id_from_metadata = lambda: None
  _gcp_metadata._cached_project_id = None
  _cloud_sdk.get_application_default_credentials_path = lambda: ''
  try:
    yield
  finally:
    _gcp_metadata.get_project_id_from_metadata = orig_get_project_id
    _gcp_metadata._cached_project_id = orig_cached_project_id
    _cloud_sdk.get_application_default_credentials_path = orig_get_adc_path


_GCLOUD_PROGRAMS = frozenset({'gcloud', 'gcloud.cmd'})


def _runs_gcloud(args: Any) -> bool:
  """Returns whether a subprocess call would run the gcloud CLI."""
  if isinstance(args, (str, bytes, os.PathLike)):
    args = os.fsdecode(args).split()
  if not args:
    return False
  return os.path.basename(os.fsdecode(args[0])) in _GCLOUD_PROGRAMS


@pytest.fixture(autouse=True)
def _no_real_gcloud():
  """Fails any test that would run the developer's real gcloud CLI.

  The CLI onboarding and deploy code shells out to gcloud. A test that reaches
  it unmocked opens a browser sign-in or changes the developer's gcloud
  config, so it fails here instead. Tests that mock subprocess themselves
  replace this guard for their own duration.

  It calls pytest.fail, whose exception derives from BaseException, so code
  under test that catches Exception around the gcloud call can't swallow it.

  The guard uses its own MonkeyPatch rather than the monkeypatch fixture, so
  it doesn't change the order in which other tests' fixtures are torn down.
  """
  real_run = subprocess.run
  real_create_subprocess_exec = asyncio.create_subprocess_exec

  def run(args: Any, *rest: Any, **kwargs: Any) -> Any:
    if _runs_gcloud(args):
      pytest.fail(f'Test ran the real gcloud CLI: {args}')
    return real_run(args, *rest, **kwargs)

  async def create_subprocess_exec(
      program: Any, *rest: Any, **kwargs: Any
  ) -> Any:
    if _runs_gcloud([program]):
      pytest.fail(f'Test ran the real gcloud CLI: {[program, *rest]}')
    return await real_create_subprocess_exec(program, *rest, **kwargs)

  with pytest.MonkeyPatch.context() as patcher:
    patcher.setattr(subprocess, 'run', run)
    patcher.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    yield

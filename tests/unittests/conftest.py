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

import os

from pytest import fixture
from pytest import FixtureRequest
from pytest import hookimpl
from pytest import Metafunc

_ENV_VARS = {
    'GOOGLE_API_KEY': 'fake_google_api_key',
    'GOOGLE_CLOUD_PROJECT': 'fake_google_cloud_project',
    'GOOGLE_CLOUD_LOCATION': 'fake_google_cloud_location',
    'ADK_ALLOW_WIP_FEATURES': 'true',
}

ENV_SETUPS = {
    'GOOGLE_AI': {
        'GOOGLE_GENAI_USE_VERTEXAI': '0',
        **_ENV_VARS,
    },
    'VERTEX': {
        'GOOGLE_GENAI_USE_VERTEXAI': '1',
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

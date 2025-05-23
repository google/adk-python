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

import logging
import os
from typing import Literal
import warnings

from dotenv import load_dotenv
from google.adk import Agent
from pytest import fixture
from pytest import FixtureRequest
from pytest import hookimpl
from pytest import Metafunc

from .utils import TestRunner

logger = logging.getLogger('google_adk.' + __name__)


def load_env_for_tests():
  dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
  if not os.path.exists(dotenv_path):
    warnings.warn(
        f'Missing .env file at {dotenv_path}. See dotenv.sample for an example.'
    )
  else:
    load_dotenv(dotenv_path, override=True, verbose=True)
  if 'GOOGLE_API_KEY' not in os.environ:
    warnings.warn(
        'Missing GOOGLE_API_KEY in the environment variables. GOOGLE_AI backend'
        ' integration tests will fail.'
    )
  for env_var in [
      'GOOGLE_CLOUD_PROJECT',
      'GOOGLE_CLOUD_LOCATION',
  ]:
    if env_var not in os.environ:
      warnings.warn(
          f'Missing {env_var} in the environment variables. Vertex backend'
          ' integration tests will fail.'
      )


load_env_for_tests()

BackendType = Literal['GOOGLE_AI', 'VERTEX']


@fixture
def agent_runner(request: FixtureRequest) -> TestRunner:
  assert isinstance(request.param, dict)

  if 'agent' in request.param:
    assert isinstance(request.param['agent'], Agent)
    return TestRunner(request.param['agent'])
  elif 'agent_name' in request.param:
    assert isinstance(request.param['agent_name'], str)
    return TestRunner.from_agent_name(request.param['agent_name'])

  raise NotImplementedError('Must provide agent or agent_name.')


@fixture(autouse=True)
def llm_backend(request: FixtureRequest):
  # Set backend environment value.
  original_val = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')
  backend_type = request.param
  if backend_type == 'GOOGLE_AI':
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = '0'
  else:
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = '1'

  yield  # Run the test

  # Restore the environment
  if original_val is None:
    os.environ.pop('GOOGLE_GENAI_USE_VERTEXAI', None)
  else:
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = original_val


@hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: Metafunc):
  if llm_backend.__name__ in metafunc.fixturenames:
    if not _is_explicitly_marked(llm_backend.__name__, metafunc):
      test_backend = os.environ.get('TEST_BACKEND', 'BOTH')
      if test_backend == 'GOOGLE_AI_ONLY':
        metafunc.parametrize(llm_backend.__name__, ['GOOGLE_AI'], indirect=True)
      elif test_backend == 'VERTEX_ONLY':
        metafunc.parametrize(llm_backend.__name__, ['VERTEX'], indirect=True)
      elif test_backend == 'BOTH':
        metafunc.parametrize(
            llm_backend.__name__, ['GOOGLE_AI', 'VERTEX'], indirect=True
        )
      else:
        raise ValueError(
            f'Invalid TEST_BACKEND value: {test_backend}, should be one of'
            ' [GOOGLE_AI_ONLY, VERTEX_ONLY, BOTH]'
        )


def _is_explicitly_marked(mark_name: str, metafunc: Metafunc) -> bool:
  if hasattr(metafunc.function, 'pytestmark'):
    for mark in metafunc.function.pytestmark:
      if mark.name == 'parametrize' and mark.args[0] == mark_name:
        return True
  return False

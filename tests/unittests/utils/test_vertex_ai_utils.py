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

"""Tests for vertex_utils."""

from unittest import mock

from google.adk.utils import vertex_ai_utils
import pytest


def test_get_express_mode_api_key_value_error():
  with pytest.raises(ValueError) as excinfo:
    vertex_ai_utils.get_express_mode_api_key(
        project='test-project', location=None, express_mode_api_key='key'
    )
  assert (
      'Cannot specify project or location and express_mode_api_key. Either use'
      ' project and location, or just the express_mode_api_key.'
      in str(excinfo.value)
  )
  with pytest.raises(ValueError) as excinfo:
    vertex_ai_utils.get_express_mode_api_key(
        project=None, location='test-location', express_mode_api_key='key'
    )
  assert (
      'Cannot specify project or location and express_mode_api_key. Either use'
      ' project and location, or just the express_mode_api_key.'
      in str(excinfo.value)
  )
  with pytest.raises(ValueError) as excinfo:
    vertex_ai_utils.get_express_mode_api_key(
        project='test-project',
        location='test-location',
        express_mode_api_key='key',
    )
  assert (
      'Cannot specify project or location and express_mode_api_key. Either use'
      ' project and location, or just the express_mode_api_key.'
      in str(excinfo.value)
  )


@pytest.mark.parametrize(
    (
        'use_vertexai_env',
        'google_api_key_env',
        'express_mode_api_key',
        'expected',
    ),
    [
        ('true', None, 'express_key', 'express_key'),
        ('1', 'google_key', 'express_key', 'express_key'),
        ('true', 'google_key', None, 'google_key'),
        ('1', None, None, None),
        ('false', 'google_key', 'express_key', None),
        ('0', 'google_key', None, None),
        (None, 'google_key', 'express_key', None),
    ],
)
def test_get_express_mode_api_key(
    use_vertexai_env,
    google_api_key_env,
    express_mode_api_key,
    expected,
):
  env_vars = {}
  if use_vertexai_env:
    env_vars['GOOGLE_GENAI_USE_VERTEXAI'] = use_vertexai_env
  if google_api_key_env:
    env_vars['GOOGLE_API_KEY'] = google_api_key_env
  with mock.patch.dict('os.environ', env_vars, clear=True):
    assert (
        vertex_ai_utils.get_express_mode_api_key(
            project=None,
            location=None,
            express_mode_api_key=express_mode_api_key,
        )
        == expected
    )

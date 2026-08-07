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

"""Tests for the shared base model behind the auth credential models."""

from __future__ import annotations

from google.adk.auth.auth_credential import BaseModelWithConfig


class _Sample(BaseModelWithConfig):
  access_token: str


def test_base_model_with_config_accepts_camel_case_alias():
  """Credentials arrive as JSON using the camelCase wire names."""
  model = _Sample.model_validate({'accessToken': 'abc'})
  assert model.access_token == 'abc'


def test_base_model_with_config_accepts_the_python_field_name():
  """Python callers construct with the snake_case field name."""
  model = _Sample(access_token='abc')
  assert model.access_token == 'abc'


def test_base_model_with_config_keeps_unknown_fields():
  # Provider-specific keys are not modelled here, but dropping them would
  # lose data on a load/dump round trip.
  model = _Sample.model_validate({'accessToken': 'abc', 'tenantId': 'xyz'})
  assert model.model_dump()['tenantId'] == 'xyz'


def test_base_model_with_config_dumps_camel_case_only_when_asked():
  model = _Sample(access_token='abc')
  assert model.model_dump()['access_token'] == 'abc'
  assert model.model_dump(by_alias=True)['accessToken'] == 'abc'

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

"""Tests for ModelArmorConfig."""

from __future__ import annotations

from google.adk.integrations.model_armor import ModelArmorConfig
import pytest


def test_defaults_to_blocking_on_failure():
  """Screening that cannot produce a verdict blocks unless opted out."""
  config = ModelArmorConfig(prompt_template_name='test-prompt-template')

  assert config.block_on_screening_failure is True
  assert config.input_blocked_message
  assert config.output_blocked_message


def test_missing_both_templates_raises():
  """A config with neither template configured is rejected."""
  with pytest.raises(ValueError, match='At least one of'):
    ModelArmorConfig()


def test_only_response_template_is_allowed():
  """Configuring only the response template is allowed."""
  config = ModelArmorConfig(response_template_name='test-response-template')

  assert config.response_template_name == 'test-response-template'
  assert config.prompt_template_name is None


def test_only_prompt_template_is_allowed():
  """Configuring only the prompt template is allowed."""
  config = ModelArmorConfig(prompt_template_name='test-prompt-template')

  assert config.prompt_template_name == 'test-prompt-template'
  assert config.response_template_name is None

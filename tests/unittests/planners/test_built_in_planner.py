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

"""Tests for the BuiltInPlanner."""

import logging
from unittest import mock

from google.adk.models.llm_request import LlmRequest
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.genai import types
import pytest


@pytest.fixture
def thinking_config():
  return types.ThinkingConfig(thinking_budget=1024)


@pytest.fixture
def planner(thinking_config):
  return BuiltInPlanner(thinking_config=thinking_config)


def test_init_stores_thinking_config(thinking_config):
  planner = BuiltInPlanner(thinking_config=thinking_config)
  assert planner.thinking_config == thinking_config


def test_apply_thinking_config_sets_config(planner):
  llm_request = LlmRequest()

  planner.apply_thinking_config(llm_request)

  assert llm_request.config is not None
  assert llm_request.config.thinking_config == planner.thinking_config


def test_apply_thinking_config_preserves_default_config(planner):
  llm_request = LlmRequest()
  assert llm_request.config is not None

  planner.apply_thinking_config(llm_request)

  assert llm_request.config.thinking_config == planner.thinking_config


def test_apply_thinking_config_overwrites_existing(planner, caplog):
  existing_thinking_config = types.ThinkingConfig(thinking_budget=512)
  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          thinking_config=existing_thinking_config,
      )
  )

  with caplog.at_level(logging.DEBUG):
    planner.apply_thinking_config(llm_request)

  assert llm_request.config.thinking_config == planner.thinking_config
  assert 'Overwriting' in caplog.text


def test_apply_thinking_config_preserves_other_config(planner):
  llm_request = LlmRequest(
      config=types.GenerateContentConfig(
          temperature=0.7,
      )
  )

  planner.apply_thinking_config(llm_request)

  assert llm_request.config.temperature == 0.7
  assert llm_request.config.thinking_config == planner.thinking_config


def test_build_planning_instruction_returns_none(planner):
  readonly_context = mock.MagicMock()
  llm_request = LlmRequest()

  result = planner.build_planning_instruction(readonly_context, llm_request)

  assert result is None


def test_process_planning_response_returns_none(planner):
  callback_context = mock.MagicMock()
  response_parts = [types.Part(text='some text')]

  result = planner.process_planning_response(callback_context, response_parts)

  assert result is None

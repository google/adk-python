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

"""Tests for the BasePlanner."""

from unittest import mock

from google.adk.models.llm_request import LlmRequest
from google.adk.planners.base_planner import BasePlanner
from google.genai import types
import pytest


class _CompletePlanner(BasePlanner):
  """Planner that implements all abstract methods."""

  def build_planning_instruction(self, readonly_context, llm_request):
    return 'test instruction'

  def process_planning_response(self, callback_context, response_parts):
    return response_parts


def test_cannot_instantiate_base_planner():
  with pytest.raises(TypeError):
    BasePlanner()


def test_can_instantiate_complete_planner():
  planner = _CompletePlanner()
  assert isinstance(planner, BasePlanner)


def test_build_planning_instruction_interface():
  planner = _CompletePlanner()
  readonly_context = mock.MagicMock()
  llm_request = LlmRequest()

  result = planner.build_planning_instruction(readonly_context, llm_request)

  assert result == 'test instruction'


def test_process_planning_response_interface():
  planner = _CompletePlanner()
  callback_context = mock.MagicMock()
  parts = [types.Part(text='response')]

  result = planner.process_planning_response(callback_context, parts)

  assert result == parts

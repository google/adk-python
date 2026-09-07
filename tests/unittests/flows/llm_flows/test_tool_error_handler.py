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

"""Unit tests for _tool_error_handler."""

from __future__ import annotations

from typing import Any
from unittest import mock

from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import _tool_error_handler
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
import pytest


def test_detect_error_type_for_telemetry_no_hook() -> None:
  tool = BaseTool(name='test_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()

  assert (
      _tool_error_handler.detect_error_type_for_telemetry(
          tool, tool_context, {'status': 'ok'}
      )
      is None
  )


def test_detect_error_type_for_telemetry_with_hook() -> None:
  class CustomTool(BaseTool):

    def _detect_error_in_response(self, response: Any) -> str | None:
      if 'error' in response:
        return 'CustomError'
      return None

  tool = CustomTool(name='custom_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()

  assert (
      _tool_error_handler.detect_error_type_for_telemetry(
          tool, tool_context, {'error': 'failed'}
      )
      == 'CustomError'
  )
  assert (
      _tool_error_handler.detect_error_type_for_telemetry(
          tool, tool_context, {'result': 'ok'}
      )
      is None
  )


def test_detect_error_type_for_telemetry_swallows_exception() -> None:
  class ExplodingTool(BaseTool):

    def _detect_error_in_response(self, response: Any) -> str | None:
      raise RuntimeError('explosion')

  tool = ExplodingTool(name='exploding_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()

  # Should not raise exception
  assert (
      _tool_error_handler.detect_error_type_for_telemetry(
          tool, tool_context, {'result': 'ok'}
      )
      is None
  )


def test_build_tool_not_found_response() -> None:
  tools = {
      'tool_a': BaseTool(name='tool_a', description='a'),
      'tool_b': BaseTool(name='tool_b', description='b'),
  }
  res = _tool_error_handler.build_tool_not_found_response('unknown_tool', tools)
  assert 'error' in res
  assert 'unknown_tool()' in res['error']
  assert 'tool_a, tool_b' in res['error']


def test_build_tool_not_found_response_empty() -> None:
  res = _tool_error_handler.build_tool_not_found_response('unknown_tool', {})
  assert 'error' in res
  assert 'none' in res['error']

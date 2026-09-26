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

"""Tests for ToolOutputModelArmorPlugin sample."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.tools.base_tool import BaseTool
from google.cloud import modelarmor_v1
import pytest

_SAMPLE_DIR = (
    Path(__file__).resolve().parents[4]
    / 'contributing/samples/integrations/model_armor_tool_output'
)
_SPEC = importlib.util.spec_from_file_location(
    'model_armor_tool_output_sample',
    _SAMPLE_DIR / 'tool_output_plugin.py',
)
assert _SPEC and _SPEC.loader
_SAMPLE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SAMPLE
_SPEC.loader.exec_module(_SAMPLE)

ToolOutputModelArmorPlugin = _SAMPLE.ToolOutputModelArmorPlugin
stringify_tool_result = _SAMPLE.stringify_tool_result

_PROMPT_TEMPLATE = (
    'projects/test-project/locations/us-central1/templates/test-prompt'
)
_BLOCKED = 'tool output blocked'


def _sanitization_result(
    *,
    match: bool = False,
    invocation_result=modelarmor_v1.InvocationResult.SUCCESS,
):
  return modelarmor_v1.SanitizationResult(
      filter_match_state=(
          modelarmor_v1.FilterMatchState.MATCH_FOUND
          if match
          else modelarmor_v1.FilterMatchState.NO_MATCH_FOUND
      ),
      invocation_result=invocation_result,
  )


def _sdk_client(*, result=None, raises: bool = False) -> mock.Mock:
  result = _sanitization_result() if result is None else result
  client = mock.Mock()
  client.sanitize_user_prompt = mock.AsyncMock(
      return_value=modelarmor_v1.SanitizeUserPromptResponse(
          sanitization_result=result
      )
  )
  if raises:
    client.sanitize_user_prompt.side_effect = RuntimeError('unreachable')
  return client


def _config(**overrides) -> ModelArmorConfig:
  defaults = dict(
      prompt_template_name=_PROMPT_TEMPLATE,
      input_blocked_message=_BLOCKED,
  )
  defaults.update(overrides)
  return ModelArmorConfig(**defaults)


def _plugin(*, result=None, raises: bool = False, **config_overrides):
  client = _sdk_client(result=result, raises=raises)
  plugin = ToolOutputModelArmorPlugin(config=_config(**config_overrides), client=client)
  return plugin, client


def _tool() -> mock.Mock:
  tool = mock.Mock(spec=BaseTool)
  tool.name = 'fetch_external_text'
  return tool


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        ({'text': 'hello'}, 'hello'),
        ({'text': 42}, '42'),
        ({'a': 1, 'b': 2}, '{"a": 1, "b": 2}'),
        ({}, None),
    ],
)
def test_stringify_tool_result(result, expected):
  assert stringify_tool_result(result) == expected


@pytest.mark.asyncio
async def test_clean_tool_output_passes_through():
  plugin, client = _plugin()

  out = await plugin.after_tool_callback(
      tool=_tool(),
      tool_args={},
      tool_context=mock.Mock(),
      result={'text': 'safe payload'},
  )

  assert out is None
  client.sanitize_user_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_matched_tool_output_returns_error_dict():
  plugin, client = _plugin(result=_sanitization_result(match=True))

  out = await plugin.after_tool_callback(
      tool=_tool(),
      tool_args={},
      tool_context=mock.Mock(),
      result={'text': 'hostile payload'},
  )

  assert out == {'error': _BLOCKED}
  request = client.sanitize_user_prompt.await_args.kwargs['request']
  assert request.user_prompt_data.text == 'hostile payload'
  assert request.name == _PROMPT_TEMPLATE


@pytest.mark.asyncio
async def test_screening_failure_blocks_by_default():
  plugin, _ = _plugin(
      result=_sanitization_result(
          invocation_result=modelarmor_v1.InvocationResult.PARTIAL
      )
  )

  out = await plugin.after_tool_callback(
      tool=_tool(),
      tool_args={},
      tool_context=mock.Mock(),
      result={'text': 'payload'},
  )

  assert out == {'error': _BLOCKED}


@pytest.mark.asyncio
async def test_screening_failure_can_fail_open():
  plugin, _ = _plugin(
      result=_sanitization_result(
          invocation_result=modelarmor_v1.InvocationResult.PARTIAL
      ),
      block_on_screening_failure=False,
  )

  out = await plugin.after_tool_callback(
      tool=_tool(),
      tool_args={},
      tool_context=mock.Mock(),
      result={'text': 'payload'},
  )

  assert out is None


@pytest.mark.asyncio
async def test_api_error_blocks_when_fail_closed():
  plugin, _ = _plugin(raises=True)

  out = await plugin.after_tool_callback(
      tool=_tool(),
      tool_args={},
      tool_context=mock.Mock(),
      result={'text': 'payload'},
  )

  assert out == {'error': _BLOCKED}


def test_requires_prompt_template_name():
  with pytest.raises(ValueError, match='prompt_template_name'):
    ToolOutputModelArmorPlugin(
        config=ModelArmorConfig(response_template_name=_PROMPT_TEMPLATE)
    )

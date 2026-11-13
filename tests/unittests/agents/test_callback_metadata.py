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

import copy
from dataclasses import FrozenInstanceError
from unittest import mock

from google.adk.agents import CallbackInfo
from google.adk.agents import CallbackType
from google.adk.agents.callback_context import CallbackContext
from google.adk.utils._callback_pipeline import _run_callbacks
from google.adk.utils._callback_pipeline import _stop_on_truthy
import pytest


def test_callback_type_values_are_stable():
  """Callback type values match the documented lifecycle names."""
  assert [callback_type.value for callback_type in CallbackType] == [
      'before_agent',
      'after_agent',
      'before_model',
      'after_model',
      'before_tool',
      'after_tool',
      'on_model_error',
      'on_tool_error',
      'on_agent_error',
  ]


def test_callback_info_is_immutable():
  """CallbackInfo cannot be changed by consumers."""
  callback_info = CallbackInfo(type=CallbackType.BEFORE_MODEL)

  with pytest.raises(FrozenInstanceError):
    callback_info.type = CallbackType.AFTER_MODEL


@pytest.mark.asyncio
async def test_context_is_deepcopyable_with_callback_info():
  """Context supports deepcopy before, during, and after a callback."""
  mock_context = mock.MagicMock()
  mock_context.session.state = {}
  mock_context._state_schema = None
  context = CallbackContext(mock_context)

  copied_before = copy.deepcopy(context)
  assert copied_before.callback_info is None

  copied_during: CallbackContext | None = None

  def callback(callback_context: CallbackContext) -> None:
    nonlocal copied_during
    assert callback_context.callback_info == CallbackInfo(
        type=CallbackType.BEFORE_AGENT
    )
    copied_during = copy.deepcopy(callback_context)
    assert copied_during.callback_info == CallbackInfo(
        type=CallbackType.BEFORE_AGENT
    )

  await _run_callbacks(
      [callback],
      _stop_on_truthy,
      callback_context=context,
      callback_type=CallbackType.BEFORE_AGENT,
  )

  assert copied_during is not None
  assert copied_during.callback_info == CallbackInfo(
      type=CallbackType.BEFORE_AGENT
  )
  assert context.callback_info is None
  assert copied_before.callback_info is None


@pytest.mark.asyncio
async def test_nested_context_does_not_inherit_outer_callback_info():
  """Nested Context instances do not inherit the outer callback type."""
  outer_inv_ctx = mock.MagicMock()
  outer_inv_ctx.session.state = {}
  outer_inv_ctx._state_schema = None
  outer_context = CallbackContext(outer_inv_ctx)

  nested_inv_ctx = mock.MagicMock()
  nested_inv_ctx.session.state = {}
  nested_inv_ctx._state_schema = None
  nested_context = CallbackContext(nested_inv_ctx)
  observations: list[CallbackInfo | None] = []

  async def outer_callback(callback_context: CallbackContext) -> None:
    observations.append(callback_context.callback_info)
    observations.append(nested_context.callback_info)

    def nested_callback(callback_context: CallbackContext) -> None:
      observations.append(callback_context.callback_info)
      observations.append(outer_context.callback_info)

    await _run_callbacks(
        [nested_callback],
        _stop_on_truthy,
        callback_context=nested_context,
        callback_type=CallbackType.BEFORE_AGENT,
    )
    observations.append(nested_context.callback_info)
    observations.append(callback_context.callback_info)

  await _run_callbacks(
      [outer_callback],
      _stop_on_truthy,
      callback_context=outer_context,
      callback_type=CallbackType.BEFORE_MODEL,
  )

  assert observations == [
      CallbackInfo(type=CallbackType.BEFORE_MODEL),
      None,
      CallbackInfo(type=CallbackType.BEFORE_AGENT),
      CallbackInfo(type=CallbackType.BEFORE_MODEL),
      None,
      CallbackInfo(type=CallbackType.BEFORE_MODEL),
  ]
  assert outer_context.callback_info is None
  assert nested_context.callback_info is None


@pytest.mark.asyncio
async def test_reentrant_callback_scope_restores_previous_callback_info():
  """Nested callback scopes on the same Context restore the previous type."""
  inv_ctx = mock.MagicMock()
  inv_ctx.session.state = {}
  inv_ctx._state_schema = None
  context = CallbackContext(inv_ctx)
  observations: list[CallbackInfo | None] = []

  async def outer_callback(callback_context: CallbackContext) -> None:
    observations.append(callback_context.callback_info)

    def inner_callback(callback_context: CallbackContext) -> None:
      observations.append(callback_context.callback_info)

    await _run_callbacks(
        [inner_callback],
        _stop_on_truthy,
        callback_context=callback_context,
        callback_type=CallbackType.AFTER_MODEL,
    )
    observations.append(callback_context.callback_info)

  await _run_callbacks(
      [outer_callback],
      _stop_on_truthy,
      callback_context=context,
      callback_type=CallbackType.BEFORE_MODEL,
  )

  assert observations == [
      CallbackInfo(type=CallbackType.BEFORE_MODEL),
      CallbackInfo(type=CallbackType.AFTER_MODEL),
      CallbackInfo(type=CallbackType.BEFORE_MODEL),
  ]
  assert context.callback_info is None

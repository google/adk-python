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

"""Unit tests for _runner_utils module in utils package."""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.utils._runner_utils import _notify_run_error
from google.adk.utils._runner_utils import _with_caller_context
from opentelemetry import context
import pytest


@pytest.mark.asyncio
async def test_with_caller_context():
  caller_ctx = context.get_current()

  async def dummy_generator():
    yield 1
    yield 2

  items = []
  async for item in _with_caller_context(dummy_generator(), caller_ctx):
    items.append(item)

  assert items == [1, 2]


@pytest.mark.asyncio
async def test_notify_run_error_calls_callback():
  pm = MagicMock(spec=PluginManager)
  pm.run_on_run_error_callback = AsyncMock()
  ic = MagicMock(spec=InvocationContext)
  error = RuntimeError("test error")

  await _notify_run_error(pm, ic, error)
  pm.run_on_run_error_callback.assert_awaited_once_with(
      invocation_context=ic, error=error
  )


@pytest.mark.asyncio
async def test_notify_run_error_suppresses_callback_exception():
  pm = MagicMock(spec=PluginManager)
  pm.run_on_run_error_callback = AsyncMock(
      side_effect=Exception("callback failed")
  )
  ic = MagicMock(spec=InvocationContext)
  error = RuntimeError("test error")

  # Should not raise
  await _notify_run_error(pm, ic, error)

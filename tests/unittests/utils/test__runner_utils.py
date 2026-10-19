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

import asyncio
import contextvars
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.utils._runner_utils import _notify_run_error
from google.adk.utils._runner_utils import _stream_in_caller_scope
import pytest

_VAR = contextvars.ContextVar("_VAR", default="caller")


async def _set_var_across_yields(seen_in_finally: list[str]):
  """Holds a token across yields, like `start_as_current_span`."""
  token = _VAR.set("gen")
  try:
    yield 1
    yield 2
  finally:
    seen_in_finally.append(_VAR.get())
    _VAR.reset(token)  # ValueError if run in another context than set()


@pytest.mark.asyncio
async def test_stream_in_caller_scope_does_not_leak_into_consumer():
  seen_in_finally = []
  seen_by_consumer = []

  async for _ in _stream_in_caller_scope(
      _set_var_across_yields(seen_in_finally)
  ):
    seen_by_consumer.append(_VAR.get())

  assert seen_by_consumer == ["caller", "caller"]
  assert seen_in_finally == ["gen"]
  assert _VAR.get() == "caller"


@pytest.mark.asyncio
async def test_stream_in_caller_scope_closes_from_another_task():
  seen_in_finally = []
  agen = _stream_in_caller_scope(_set_var_across_yields(seen_in_finally))

  await asyncio.create_task(agen.__anext__())
  await asyncio.create_task(agen.aclose())  # would raise ValueError unwrapped

  assert seen_in_finally == ["gen"]


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

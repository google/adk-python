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

"""Tests for the tool-result buffer and its Antigravity SDK hook binding.

Two concerns: that the buffer keeps and hands back the results the converter
needs, and that the hook the SDK is handed is one the SDK will accept.
"""

from __future__ import annotations

import copy

from google.adk.labs.antigravity import _tool_result_capture
from google.antigravity import types as sdk_types
from google.antigravity.hooks import hook_runner as sdk_hook_runner
from google.antigravity.hooks import hooks as sdk_hooks
import pytest


def _result(call_id: str | None, value=None, error=None, name='reviewer'):
  """Builds an SDK ToolResult as the post-tool-call hook would receive one."""
  return sdk_types.ToolResult(name=name, id=call_id, result=value, error=error)


class _ToolFailure(RuntimeError):
  """Stands in for the SDK's ``ToolExecutionError``.

  Structural, not the SDK class, for the same reason the capture reads a
  ``ToolError`` protocol rather than that class: a structurally identical copy
  of it exists, and ``google-antigravity`` on PyPI does not export one at all
  yet. Testing against the shape is what the production code actually depends
  on.
  """

  def __init__(self, message: str, tool_name: str, call_id: str | None = None):
    super().__init__(message)
    self.tool_name = tool_name
    self.call_id = call_id


def _failure(
    call_id: str | None, message='child agent exploded', name='reviewer'
):
  """Builds the error the on-tool-error hook would receive."""
  return _ToolFailure(message, name, call_id=call_id)


@pytest.mark.skipif(
    not hasattr(sdk_types, 'ToolExecutionError'),
    reason='The pinned open-source google-antigravity does not export it.',
)
def test_the_real_sdk_error_satisfies_the_protocol():
  """The stand-in above is only honest if the real class matches it."""
  error = sdk_types.ToolExecutionError('boom', 'reviewer', call_id='c1')

  assert isinstance(error, _tool_result_capture.ToolError)


def test_the_capture_is_accepted_by_the_sdk_hook_runner():
  """A mis-based hook raises `Unknown hook type` from inside `__aenter__`.

  Every other test in this package fakes the SDK `Agent`, so this is the only
  place the base class is checked at all.
  """
  capture = _tool_result_capture.ToolResultCapture()
  runner = sdk_hook_runner.HookRunner()

  runner.register_hook(capture)

  assert isinstance(capture, sdk_hooks.PostToolCallHook)
  assert runner.post_tool_call_hooks == (capture,)


def test_the_error_capture_is_accepted_by_the_sdk_hook_runner():
  """Same check as above for the failure half of the pair."""
  capture = _tool_result_capture.ToolErrorCapture(
      _tool_result_capture.ToolResultBuffer()
  )
  runner = sdk_hook_runner.HookRunner()

  runner.register_hook(capture)

  assert isinstance(capture, sdk_hooks.OnToolErrorHook)
  assert runner.on_tool_error_hooks == (capture,)


def test_the_two_captures_register_on_separate_hook_lists():
  """`register_hook` appends to every list an object matches.

  One object claiming both interfaces would have its single `run` handed a
  ToolResult and an exception alike, which is why these are two classes.
  """
  buffer = _tool_result_capture.ToolResultCapture()
  errors = _tool_result_capture.ToolErrorCapture(buffer)
  runner = sdk_hook_runner.HookRunner()

  runner.register_hook(buffer)
  runner.register_hook(errors)

  assert runner.post_tool_call_hooks == (buffer,)
  assert runner.on_tool_error_hooks == (errors,)


def test_the_bare_buffer_is_not_a_hook():
  """A second SDK binding subclasses it, so it must not already be a hook."""
  bare = _tool_result_capture.ToolResultBuffer()

  assert not isinstance(bare, sdk_hooks.PostToolCallHook)
  assert not isinstance(bare, sdk_hooks.OnToolErrorHook)


@pytest.mark.asyncio
async def test_a_result_is_buffered_under_its_call_id():
  """The hook's only job: put the result where the converter can find it."""
  capture = _tool_result_capture.ToolResultCapture()
  result = _result('call_3', value='{"result": "good name"}')

  await capture.run(None, result)

  assert capture.take({'call_3'}) == [('call_3', result)]


@pytest.mark.asyncio
async def test_a_result_without_an_id_is_dropped():
  """`id` is the only correlator with the emitted function call."""
  capture = _tool_result_capture.ToolResultCapture()

  await capture.run(None, _result(None, value='orphan'))

  assert not capture


def test_take_returns_only_the_requested_ids_and_removes_them():
  """Draining is by id and is one-shot: a result answers exactly one call."""
  buffer = _tool_result_capture.ToolResultBuffer()
  first, second = _result('c1', value='a'), _result('c2', value='b')
  buffer.record(first)
  buffer.record(second)

  assert buffer.take({'c1'}) == [('c1', first)]
  assert not buffer.take({'c1'})
  assert buffer.take({'c1', 'c2'}) == [('c2', second)]


def test_take_preserves_arrival_order():
  """Several results drained at once keep the order the tools finished in."""
  buffer = _tool_result_capture.ToolResultBuffer()
  buffer.record(_result('c2', value='second'))
  buffer.record(_result('c1', value='first'))

  assert [call_id for call_id, _ in buffer.take({'c1', 'c2'})] == ['c2', 'c1']


@pytest.mark.asyncio
async def test_a_failure_is_buffered_under_its_call_id():
  """The harness reports a failed tool here and nowhere else.

  `post_tool_call` never fires for one, so without this hook the call keeps
  its function_call and never gets a function_response.
  """
  buffer = _tool_result_capture.ToolResultBuffer()
  capture = _tool_result_capture.ToolErrorCapture(buffer)

  await capture.run(None, _failure('call_3'))

  call_id, result = buffer.take({'call_3'})[0]
  assert call_id == 'call_3'
  assert result.name == 'reviewer'
  assert result.error == 'child agent exploded'
  assert result.result is None


@pytest.mark.asyncio
async def test_a_failure_without_a_call_id_is_dropped():
  """`call_id` is the only correlator, exactly as for a result."""
  buffer = _tool_result_capture.ToolResultBuffer()
  capture = _tool_result_capture.ToolErrorCapture(buffer)

  await capture.run(None, _failure(None))

  assert not buffer


@pytest.mark.asyncio
async def test_the_error_hook_returns_none_so_the_harness_message_stands():
  """`OnToolErrorHook` rewrites what the model sees if it returns a string."""
  buffer = _tool_result_capture.ToolResultBuffer()
  capture = _tool_result_capture.ToolErrorCapture(buffer)

  assert await capture.run(None, _failure('call_3')) is None


@pytest.mark.asyncio
async def test_both_hooks_feed_one_buffer():
  """The converter drains one buffer, so a turn's two hooks must share it."""
  results = _tool_result_capture.ToolResultCapture()
  errors = _tool_result_capture.ToolErrorCapture(results)

  await results.run(None, _result('c1', value='ok'))
  await errors.run(None, _failure('c2'))

  assert [call_id for call_id, _ in results.take({'c1', 'c2'})] == ['c1', 'c2']


def test_clear_empties_the_buffer():
  """End-of-turn housekeeping, for results that were never owed."""
  buffer = _tool_result_capture.ToolResultBuffer()
  buffer.record(_result('c1', value='a'))

  buffer.clear()

  assert not buffer


def test_a_later_result_for_one_call_id_replaces_the_earlier():
  """One call gets one answer; a repeat is a correction, not a second entry."""
  buffer = _tool_result_capture.ToolResultBuffer()
  buffer.record(_result('c1', value='stale'))
  final = _result('c1', value='fresh')
  buffer.record(final)

  assert buffer.take({'c1'}) == [('c1', final)]


def test_the_capture_survives_a_deep_copy_of_the_config():
  """`_build_sdk_config` deep-copies the config with the hook still on it."""
  capture = _tool_result_capture.ToolResultCapture()
  capture.record(_result('c1', value='a'))

  clone = copy.deepcopy(capture)

  assert clone is not capture
  assert clone.take({'c1'})

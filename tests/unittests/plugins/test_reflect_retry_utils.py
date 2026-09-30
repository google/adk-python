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

"""Unit tests for the shared reflect-and-retry scope/failure bookkeeping."""

from __future__ import annotations

import enum

from google.adk.plugins import _reflect_retry_utils
import pytest


def test_resolve_scope_key_invocation_scope_uses_invocation_id():
  key = _reflect_retry_utils.resolve_scope_key(
      _reflect_retry_utils.TrackingScope.INVOCATION, 'invocation-1'
  )

  assert key == 'invocation-1'


@pytest.mark.parametrize('invocation_id', [None, ''])
def test_resolve_scope_key_invocation_scope_requires_invocation_id(
    invocation_id,
):
  with pytest.raises(ValueError, match='invocation_id must be provided'):
    _reflect_retry_utils.resolve_scope_key(
        _reflect_retry_utils.TrackingScope.INVOCATION, invocation_id
    )


@pytest.mark.parametrize('invocation_id', [None, 'invocation-1'])
def test_resolve_scope_key_global_scope_ignores_invocation_id(invocation_id):
  key = _reflect_retry_utils.resolve_scope_key(
      _reflect_retry_utils.TrackingScope.GLOBAL, invocation_id
  )

  assert key == _reflect_retry_utils.GLOBAL_SCOPE_KEY


def test_resolve_scope_key_rejects_unknown_scope():
  class _OtherScope(enum.Enum):
    SOMETHING_ELSE = 'something_else'

  with pytest.raises(ValueError, match='Unknown scope'):
    _reflect_retry_utils.resolve_scope_key(
        _OtherScope.SOMETHING_ELSE, 'invocation-1'
    )


async def test_tracker_increment_returns_running_count_per_item():
  tracker = _reflect_retry_utils.ScopedFailureTracker()

  first = await tracker.increment('scope', 'tool_a')
  second = await tracker.increment('scope', 'tool_a')
  other_tool = await tracker.increment('scope', 'tool_b')
  third = await tracker.increment('scope', 'tool_a')

  assert [first, second, third] == [1, 2, 3]
  # A sibling item in the same scope keeps its own count.
  assert other_tool == 1


async def test_tracker_keeps_scopes_independent():
  tracker = _reflect_retry_utils.ScopedFailureTracker()

  await tracker.increment('scope_a', 'tool')
  await tracker.increment('scope_a', 'tool')

  assert await tracker.increment('scope_b', 'tool') == 1
  assert await tracker.increment('scope_a', 'tool') == 3


async def test_tracker_reset_clears_only_the_named_item():
  tracker = _reflect_retry_utils.ScopedFailureTracker()
  await tracker.increment('scope', 'tool_a')
  await tracker.increment('scope', 'tool_b')
  await tracker.increment('scope', 'tool_b')

  await tracker.reset('scope', 'tool_a')

  assert await tracker.increment('scope', 'tool_a') == 1
  assert await tracker.increment('scope', 'tool_b') == 3


async def test_tracker_reset_of_unseen_scope_is_a_noop():
  tracker = _reflect_retry_utils.ScopedFailureTracker()

  await tracker.reset('never-seen', 'tool')

  assert await tracker.increment('never-seen', 'tool') == 1

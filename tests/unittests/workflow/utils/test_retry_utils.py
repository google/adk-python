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

from __future__ import annotations

from google.adk.workflow._node_state import NodeState
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow.utils._retry_utils import _get_retry_delay
import pytest


class TestGetRetryDelay:

  def test_returns_default_delay_without_config(self):
    """Returns default delay of 1.0 second when config is missing."""
    state = NodeState(attempt_count=1)

    result = _get_retry_delay(None, state)

    assert result == 1.0

  def test_returns_initial_delay_on_first_failure(self):
    """Returns initial delay on the first failure attempt."""
    config = RetryConfig(initial_delay=2.0, jitter=0.0)
    state = NodeState(attempt_count=1)

    result = _get_retry_delay(config, state)

    assert result == 2.0

  def test_applies_exponential_backoff(self):
    """Applies exponential backoff for subsequent attempts."""
    config = RetryConfig(initial_delay=2.0, backoff_factor=2.0, jitter=0.0)
    state = NodeState(attempt_count=2)

    result = _get_retry_delay(config, state)

    assert result == 4.0

  def test_caps_at_max_delay(self):
    """Caps calculated delay at the specified maximum delay."""
    config = RetryConfig(
        initial_delay=2.0, backoff_factor=10.0, max_delay=15.0, jitter=0.0
    )
    state = NodeState(attempt_count=2)

    result = _get_retry_delay(config, state)

    assert result == 15.0

  def test_adds_jitter_when_enabled(self):
    """Adds random jitter to the calculated delay."""
    config = RetryConfig(initial_delay=10.0, backoff_factor=1.0, jitter=0.5)
    state = NodeState(attempt_count=1)

    delays = [_get_retry_delay(config, state) for _ in range(10)]

    assert all(5.0 <= d <= 15.0 for d in delays)
    assert len(set(delays)) > 1

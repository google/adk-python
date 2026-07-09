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

"""Utility functions for retrying nodes in a workflow."""

import random

from .._node_state import NodeState
from .._retry_config import RetryConfig


def _should_retry_node(
    exception: BaseException,
    retry_config: RetryConfig | None,
    node_state: NodeState,
) -> bool:
  """Checks if a failed node should be retried based on its retry_config."""
  if not retry_config:
    return False

  attempt_count = node_state.attempt_count
  max_attempts = retry_config.max_attempts or 5

  # attempt_count starts at 1 for the original request.
  # So if attempt_count >= max_attempts, we have reached the limit.
  if attempt_count >= max_attempts:
    return False

  if retry_config.exceptions is not None:
    ex_name = type(exception).__name__
    if ex_name not in retry_config.exceptions:
      return False

  return True


def _get_retry_delay(
    retry_config: RetryConfig | None,
    node_state: NodeState,
) -> float:
  """Calculates the delay before retrying a node."""
  # Default delay is 1.0 second.
  if not retry_config:
    return 1.0

  initial_delay = (
      retry_config.initial_delay
      if retry_config.initial_delay is not None
      else 1.0
  )
  max_delay = (
      retry_config.max_delay if retry_config.max_delay is not None else 60.0
  )
  backoff_factor = (
      retry_config.backoff_factor
      if retry_config.backoff_factor is not None
      else 2.0
  )
  jitter = retry_config.jitter if retry_config.jitter is not None else 1.0

  attempt_count = node_state.attempt_count or 1
  # attempt_count is the attempt number that just failed (1-based).
  # For the first failure (attempt 1), the exponent should be 0.
  attempt_for_calc = max(0, attempt_count - 1)

  delay = initial_delay * (backoff_factor**attempt_for_calc)
  delay = min(delay, max_delay)

  if jitter > 0.0:
    random_offset = random.uniform(-jitter * delay, jitter * delay)
    delay = max(0.0, delay + random_offset)

  return delay

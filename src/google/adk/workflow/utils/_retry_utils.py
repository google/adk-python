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

"""Utility functions for retrying nodes in a workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...platform import _random as platform_random
from .._errors import WorkflowConfigurationError
from .._errors import WorkflowInvariantError
from .._node_state import NodeState
from .._retry_config import RetryConfig

if TYPE_CHECKING:
  from .._base_node import BaseNode


def _should_retry_node(
    exception: BaseException,
    retry_config: RetryConfig | None,
    node_state: NodeState,
) -> bool:
  """Checks if a failed node should be retried.

  ``WorkflowConfigurationError`` and ``WorkflowInvariantError`` are never
  retried regardless of ``retry_config``.
  """
  # Deterministic errors: retrying cannot fix them.
  if not retry_config or isinstance(
      exception, (WorkflowConfigurationError, WorkflowInvariantError)
  ):
    return False

  attempt_count = node_state.attempt_count
  max_attempts = (
      retry_config.max_attempts if retry_config.max_attempts is not None else 5
  )

  # attempt_count starts at 1 for the original request.
  # So if attempt_count >= max_attempts, we have reached the limit.
  if attempt_count >= max_attempts:
    return False

  if retry_config.exceptions is not None:
    # Match against every class the exception inherits from, so a configured
    # exception also covers its subclasses. object is excluded so that naming
    # it does not silently mean "retry everything".
    ex_names = {
        cls.__name__ for cls in type(exception).__mro__ if cls is not object
    }
    if ex_names.isdisjoint(retry_config.exceptions):
      return False

  return True


def _delegates_item_retry(node: BaseNode) -> bool:
  """Returns whether a node leaves its item failures to the item node.

  A ``_ParallelWorker`` runs each item of its input as a dynamic child, and a
  ``Node`` with ``parallel_worker=True`` runs its items through an inner
  ``_ParallelWorker``. Either way the item node applies the retry policy to
  each item, so the node itself does not retry an item failure: a retry of the
  whole node would cancel the items still running and give the failed item a
  fresh budget.

  Args:
    node: The node to check.

  Returns:
    True if ``node`` is a ``_ParallelWorker`` or a ``Node`` with
    ``parallel_worker=True``.
  """
  # Imported here to avoid import cycles with the node modules.
  from .._node import Node
  from .._parallel_worker import _ParallelWorker

  return isinstance(node, _ParallelWorker) or (
      isinstance(node, Node) and node.parallel_worker
  )


def _node_retry_config(node: BaseNode) -> RetryConfig | None:
  """Returns the retry policy configured on ``node``.

  A ``_ParallelWorker`` moves its ``retry_config`` onto the wrapped item node
  and leaves its own ``retry_config`` as ``None``, so its wrapped node's policy
  is returned when ``node.retry_config`` is ``None``.

  Args:
    node: The node to inspect.

  Returns:
    The ``RetryConfig`` on ``node`` (or on its wrapped item node for a
    ``_ParallelWorker``), or ``None`` if none is configured.
  """
  from .._parallel_worker import _ParallelWorker

  if isinstance(node, _ParallelWorker) and node.retry_config is None:
    return _node_retry_config(node._node)
  return node.retry_config


def _has_own_retry_config(node: BaseNode) -> bool:
  """Returns whether a node defines its own retry policy.

  A ``_ParallelWorker`` moves its ``retry_config`` onto the wrapped item node
  and leaves its own ``retry_config`` as ``None``, so its wrapped node is
  checked as well.

  Args:
    node: The node to check.

  Returns:
    True if ``node`` (or, for a ``_ParallelWorker``, its wrapped item node)
    has a ``retry_config``.
  """
  return _node_retry_config(node) is not None


def _with_retry_config(
    node: BaseNode, retry_config: RetryConfig | None
) -> BaseNode:
  """Returns a copy of ``node`` that uses ``retry_config``.

  Args:
    node: The node to copy. It is left unchanged.
    retry_config: The retry policy for the copy, replacing the node's own, or
      ``None`` to clear it.

  Returns:
    The copy, of the same class as ``node``.
  """
  from ...agents.base_agent import BaseAgent

  copied: BaseNode
  # An agent is cloned rather than copied, so that it gets its own sub-agents
  # and its callbacks are rebound to the copy instead of the original.
  if isinstance(node, BaseAgent):
    copied = node.clone(update={"retry_config": retry_config})
    # BaseAgent.clone() resets parent_agent to None; restore it.
    copied.parent_agent = node.parent_agent
  else:
    copied = node.model_copy(update={"retry_config": retry_config})
  return copied


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

  if jitter > 0.0:
    # Cap the delay before jittering, so that even the widest positive offset
    # lands on max_delay. Capping the jittered result instead would hold the
    # bound but collapse every overshooting draw onto exactly max_delay,
    # firing the retries jitter exists to spread out at the same instant.
    delay = min(delay, max_delay / (1.0 + jitter))
    random_offset = platform_random.get_random().uniform(
        -jitter * delay, jitter * delay
    )
    delay = max(0.0, delay + random_offset)

  return min(delay, max_delay)

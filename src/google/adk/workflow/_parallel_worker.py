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

"""Parallel worker node for workflows."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Iterator
from collections.abc import Mapping
import logging
from typing import Any

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.context import Context
from ..utils._schema_utils import _unwrap_content_payload
from ._base_node import BaseNode
from ._errors import NodeInterruptedError
from ._errors import WorkflowConfigurationError
from ._graph import NodeLike
from ._retry_config import RetryConfig
from .utils._retry_utils import _with_retry_config
from .utils._workflow_graph_utils import build_node

logger = logging.getLogger('google_adk.' + __name__)

# How long to wait for cancelled items to actually stop before giving up on
# them, so an item that swallows cancellation cannot hang this node forever.
_CANCELLED_ITEM_DRAIN_TIMEOUT_SECONDS = 5.0


def _get_node_input_annotation(node: BaseNode) -> Any:
  """Returns the input type annotation for the wrapped node, if any."""
  schema = getattr(node, 'input_schema', None)
  if schema not in (None, Any, object) and not isinstance(schema, dict):
    return schema
  type_hints = getattr(node, '_type_hints', None)
  return type_hints.get('node_input') if isinstance(type_hints, dict) else None


def _unwrap_content_input(
    node_input: types.Content, inner_node: BaseNode
) -> Any:
  """Unwraps text/JSON payload from types.Content for parallel fan-out."""
  return _unwrap_content_payload(
      node_input,
      _get_node_input_annotation(inner_node),
      fan_out_list=True,
  )


class _ParallelWorker(BaseNode):
  """A node that runs a wrapped node in parallel for each item in the input list.

  ``retry_config`` applies to each item, never to the worker as a whole. The
  worker moves it onto the wrapped node, replacing any policy the wrapped node
  had, so a failed item is retried in place while its siblings keep running,
  and the worker's own ``retry_config`` stays None. ``timeout`` applies to the
  worker as a whole, so a timeout of the whole worker is not retried.

  Attributes:
    max_parallel_workers: The maximum number of parallel tasks to run. If None,
      there is no limit on concurrency.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  max_parallel_workers: int | None = Field(default=None)

  _node: BaseNode = PrivateAttr()

  def __init__(
      self,
      *,
      node: NodeLike,
      max_parallel_workers: int | None = None,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    """Initializes the worker.

    Args:
      node: The node to run once per item: a node, an agent, a tool or a
        callable, built into a node with ``build_node``.
      max_parallel_workers: The maximum number of items to run at once. None
        means no limit.
      retry_config: The retry policy for each item. When set, it is moved onto
        the wrapped node, replacing the node's own policy if it has one, just as
        ``node(..., retry_config=...)`` overrides a node's policy. If ``None``,
        the wrapped node keeps its own policy.
      timeout: The time limit for the whole worker, in seconds, covering all
        of its items. ``None`` means no time limit.

    Raises:
      WorkflowConfigurationError: If ``node`` is START, or if
        ``max_parallel_workers`` is less than 1.
    """
    if node == 'START':
      raise WorkflowConfigurationError(
          'ParallelWorker cannot wrap a START node.'
      )
    built_node = build_node(node)
    if retry_config is not None:
      # Moved rather than copied, as node(fn, parallel_worker=True,
      # retry_config=...) does: the policy applies to each item, and a failure
      # of the worker as a whole, such as its timeout, is not retried.
      built_node = _with_retry_config(built_node, retry_config)
    super().__init__(
        name=built_node.name,
        rerun_on_resume=True,
        timeout=timeout,
    )
    if max_parallel_workers is not None and max_parallel_workers < 1:
      raise WorkflowConfigurationError(
          'max_parallel_workers must be greater than or equal to 1.'
      )
    self._node = built_node
    self.max_parallel_workers = max_parallel_workers

  @override
  def _iter_static_children(self) -> Iterator[BaseNode]:
    """Yields the wrapped node."""
    yield self._node

  @override
  def model_copy(
      self, *, update: Mapping[str, Any] | None = None, deep: bool = False
  ) -> _ParallelWorker:
    """Copies the worker, moving any ``retry_config`` onto the wrapped node.

    ``node(worker, retry_config=...)`` reaches the worker through this method,
    so the policy is handled as in ``__init__``: it replaces the wrapped node's
    own and is cleared on the copy, leaving it to apply to each item.

    Args:
      update: Values for fields of the copy, replacing the worker's. If
        ``None``, no field values are replaced.
      deep: Whether to make a deep copy.

    Returns:
      The copy of the worker.
    """
    copied = super().model_copy(update=update, deep=deep)
    if copied.retry_config is not None or (
        update is not None and 'retry_config' in update
    ):
      copied._node = _with_retry_config(copied._node, copied.retry_config)
      copied.retry_config = None
    return copied

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if isinstance(node_input, types.Content):
      node_input = _unwrap_content_input(node_input, self._node)

    if not isinstance(node_input, list):
      # Wrap the single input in a list to allow processing.
      # This handles cases where the input is a single item.
      node_input = [node_input]

    if not node_input:
      yield []
      return

    results = [None] * len(node_input)
    pending_tasks: set[asyncio.Task[Any]] = set()
    input_index = 0
    interrupted: NodeInterruptedError | None = None

    try:
      while (
          not interrupted and input_index < len(node_input)
      ) or pending_tasks:
        # Check for any inputs waiting to be processed. Once a worker has
        # interrupted, stop scheduling new items and wait for in-flight
        # sibling workers to finish.
        while (
            not interrupted
            and input_index < len(node_input)
            and (
                self.max_parallel_workers is None
                or len(pending_tasks) < self.max_parallel_workers
            )
        ):
          item = node_input[input_index]
          task = asyncio.create_task(
              ctx.run_node(
                  self._node,
                  node_input=item,
                  use_sub_branch=True,
              )
          )
          # Store index on task so we can place result correctly when done
          setattr(task, '_worker_index', input_index)
          pending_tasks.add(task)
          input_index += 1

        # If there are pending tasks, wait for first one to complete.
        # We only wait for the first one, because after it completes, we want
        # to check if any new items are waiting to be processed.
        if pending_tasks:
          done, pending_tasks = await asyncio.wait(
              pending_tasks, return_when=asyncio.FIRST_COMPLETED
          )
          # asyncio.wait returns completed tasks as an unordered set; iterate in
          # input order so that, when several tasks fail in the same wake-up,
          # the surfaced exception is deterministic across runs and replays.
          failure: BaseException | None = None
          for task in sorted(done, key=lambda t: getattr(t, '_worker_index')):
            # Ask every completed task for its exception, not just the one we
            # surface, otherwise asyncio reports the rest as never retrieved.
            exc = task.exception()
            if exc is not None:
              if isinstance(exc, NodeInterruptedError):
                if interrupted is None:
                  interrupted = exc
              elif failure is None:
                failure = exc
              continue

            index = getattr(task, '_worker_index')
            results[index] = task.result()
          if failure is not None:
            # Items still in flight are cancelled and drained below.
            raise failure
      if interrupted is not None:
        raise interrupted
    finally:
      await self._drain_pending_items(pending_tasks)

    yield results

  async def _drain_pending_items(
      self, pending_tasks: set[asyncio.Task[Any]]
  ) -> None:
    """Cancels the items still in flight and waits for them to stop."""
    if not pending_tasks:
      return
    # asyncio.wait does not own the tasks it waits on, so an item left running
    # when this node is cancelled keeps going and then blocks forever emitting
    # into a run nobody consumes any more.
    for task in pending_tasks:
      task.cancel()
    drain = asyncio.gather(*pending_tasks, return_exceptions=True)
    try:
      # This normally runs while a CancelledError is being unwound, so the
      # drain is shielded: a second cancellation would otherwise abandon it
      # part-way and leave the remaining exceptions unretrieved. The timeout is
      # a separate concern: an item that swallows cancellation would otherwise
      # hold this node here forever.
      await asyncio.wait_for(
          asyncio.shield(drain),
          timeout=_CANCELLED_ITEM_DRAIN_TIMEOUT_SECONDS,
      )
    except asyncio.TimeoutError:
      logger.warning(
          'Node %s: %d item(s) did not stop within %ss of being cancelled;'
          ' abandoning them.',
          self.name,
          sum(1 for task in pending_tasks if not task.done()),
          _CANCELLED_ITEM_DRAIN_TIMEOUT_SECONDS,
      )
    except asyncio.CancelledError:
      logger.warning(
          'Node %s: cancelled again while stopping %d item(s).',
          self.name,
          sum(1 for task in pending_tasks if not task.done()),
      )
      raise

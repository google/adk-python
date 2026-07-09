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
from typing import Any

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.context import Context
from ._base_node import BaseNode
from ._graph import NodeLike
from ._retry_config import RetryConfig
from .utils._workflow_graph_utils import build_node


class _ParallelWorker(BaseNode):
  """A node that runs a wrapped node in parallel for each item in the input list.

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
    if node == 'START':
      raise ValueError('ParallelWorker cannot wrap a START node.')
    built_node = build_node(node)
    super().__init__(
        name=built_node.name,
        rerun_on_resume=True,
        retry_config=retry_config,
        timeout=timeout,
    )
    if max_parallel_workers is not None and max_parallel_workers < 1:
      raise ValueError(
          'max_parallel_workers must be greater than or equal to 1.'
      )
    self._node = built_node
    self.max_parallel_workers = max_parallel_workers

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
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

    while input_index < len(node_input) or pending_tasks:
      # Check for any inputs waiting to be processed.
      while input_index < len(node_input) and (
          self.max_parallel_workers is None
          or len(pending_tasks) < self.max_parallel_workers
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
        done, pending = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
          exc = task.exception()
          if exc is not None:
            # If a task failed, cancel all other pending tasks.
            for p_task in pending:
              p_task.cancel()
            # Wait for all pending tasks to be cancelled
            if pending:
              await asyncio.wait(pending)
            # Raise the exception from the failed task
            raise exc

          index = getattr(task, '_worker_index')
          results[index] = task.result()
        pending_tasks = pending

    yield results

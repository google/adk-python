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

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._workflow import Workflow
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context


@pytest.mark.asyncio
async def test_max_concurrency_limits_running_nodes(
    request: pytest.FixtureRequest,
):
  """Max concurrency limits the number of parallel graph-scheduled nodes.

  Setup:
    Workflow with 4 parallel nodes and max_concurrency=2.
  Act:
    - Start workflow in background.
    - Release nodes one by one.
  Assert:
    - Initially only 2 nodes start.
    - Releasing one node allows another to start.
    - All nodes eventually complete.
  """

  class ConcurrencyWorkerNode(BaseNode):
    """A node that signals when it starts and waits for a signal to finish."""

    started_event: asyncio.Event
    finish_event: asyncio.Event

    @override
    async def _run_impl(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      self.started_event.set()
      await self.finish_event.wait()
      yield f'{self.name}_done'

  class TerminalNode(BaseNode):

    @override
    async def _run_impl(
        self, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'workflow_done'

  # Given a workflow with 4 parallel nodes and max_concurrency=2
  num_nodes = 4
  max_concurrency = 2
  started_events = [asyncio.Event() for _ in range(num_nodes)]
  finish_events = [asyncio.Event() for _ in range(num_nodes)]

  nodes = [
      ConcurrencyWorkerNode(
          name=f'Node{i}',
          started_event=started_events[i],
          finish_event=finish_events[i],
      )
      for i in range(num_nodes)
  ]

  terminal_node = TerminalNode(name='Terminal')
  edges = [(START, tuple(nodes), terminal_node)]

  agent = Workflow(
      name='concurrency_agent',
      max_concurrency=max_concurrency,
      edges=edges,
  )

  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  # When the workflow is run in background
  async def run_agent():
    return await runner.run_async(testing_utils.get_user_content('start'))

  run_task = asyncio.create_task(run_agent())

  # Then initially only max_concurrency nodes should start
  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency

  # When one node is released
  for i in range(num_nodes):
    if started_events[i].is_set():
      finish_events[i].set()
      break

  # Then another node should start, bringing total to max_concurrency + 1
  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency + 1

  # When all remaining nodes are released
  for e in finish_events:
    e.set()

  # Then all nodes should eventually complete
  await run_task
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == num_nodes

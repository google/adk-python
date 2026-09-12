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

"""First-match fan-out: race several branches, cancel the losers.

``JoinNode`` is fan-in that waits for *all* predecessors. ``FirstMatchNode`` is
its opposite: it fans out over a set of branch nodes, runs them concurrently,
and the instant one returns a result the ``match`` predicate accepts, it
*cancels the still-running siblings* and yields the winner.

This is the "search Recall@k" pattern: retrieve k candidate sources, read them
in parallel, and answer the moment any one of them contains the answer -- you
neither wait for the slow branches nor keep paying to generate their output.
Cancellation propagates cooperatively down ``ctx.run_node`` into each branch's
in-flight work (e.g. a ``StreamingRouterNode``'s SSE model call is torn down via
its ``aclosing`` chain), so the losers stop *decoding* immediately.

What it does *not* do: it cannot un-send input already prefilled. Branches that
have started reading their source still paid that read; ``FirstMatchNode`` saves
the losers' generation and wall-clock, not their prefill. Pair it with a cheap
relevance gate (or rank-ordered ``max_parallel``) if you also need to avoid
reading low-ranked sources at all.

Example::

    FirstMatchNode(
        name='first_source_with_answer',
        nodes=[read_sharepoint, read_wiki, read_crm, read_docs, read_drive],
        match=lambda r: isinstance(r, dict) and r.get('found'),
    )

Each branch is handed the same ``node_input`` (broadcast); encapsulate each
source inside its own branch node. If no branch matches, the node yields
``no_match_output`` (``None`` by default).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable
import logging
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

logger = logging.getLogger('google_adk.' + __name__)


def _default_match(result: Any) -> bool:
  """Accepts any non-``None`` result as a match."""
  return result is not None


class FirstMatchNode(BaseNode):
  """Races branch nodes and returns the first matching result, cancelling the rest.

  Attributes:
    max_parallel: Maximum branches to run at once. ``None`` runs them all
      concurrently. Set this (with branches supplied in priority order) to read
      higher-ranked sources first and avoid ever starting lower-ranked ones once
      an earlier branch wins.
    no_match_output: The node's output when no branch produces a matching
      result. Defaults to ``None``.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  max_parallel: int | None = Field(default=None)
  no_match_output: Any = Field(default=None)

  _nodes: list[BaseNode] = PrivateAttr()
  _match: Callable[[Any], bool] = PrivateAttr()

  def __init__(
      self,
      *,
      name: str,
      nodes: list[NodeLike],
      match: Callable[[Any], bool] | None = None,
      max_parallel: int | None = None,
      no_match_output: Any = None,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    if not nodes:
      raise ValueError('FirstMatchNode requires at least one branch node.')
    if max_parallel is not None and max_parallel < 1:
      raise ValueError('max_parallel must be >= 1.')
    built = [build_node(n) for n in nodes]
    super().__init__(
        name=name,
        rerun_on_resume=True,
        retry_config=retry_config,
        timeout=timeout,
        max_parallel=max_parallel,
        no_match_output=no_match_output,
    )
    self._nodes = built
    self._match = match or _default_match

  async def _run_one(self, ctx: Context, node: BaseNode, node_input: Any) -> Any:
    return await ctx.run_node(node, node_input=node_input, use_sub_branch=True)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    pending: set[asyncio.Task[Any]] = set()
    remaining = list(self._nodes)
    winner: Any = self.no_match_output
    found = False

    def _launch_next() -> None:
      while remaining and (
          self.max_parallel is None or len(pending) < self.max_parallel
      ):
        node = remaining.pop(0)
        pending.add(asyncio.create_task(self._run_one(ctx, node, node_input)))

    try:
      _launch_next()
      while pending and not found:
        done, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
          # A failed branch must not sink the whole race; log and move on so a
          # single flaky source can't deny an answer another source can give.
          if task.cancelled():
            continue
          exc = task.exception()
          if exc is not None:
            logger.warning('FirstMatchNode %s: branch failed: %s', self.name, exc)
            continue
          result = task.result()
          if self._match(result):
            winner = result
            found = True
            break
        if not found:
          _launch_next()
    finally:
      # Cancel every still-running loser and wait for their teardown so the
      # in-flight reads are actually torn down before we advance the graph.
      for task in pending:
        task.cancel()
      if pending:
        await asyncio.wait(pending)

    yield winner

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

"""JoinNode implementation for workflow orchestration."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import logging
from typing import Any

from typing_extensions import override

from ..agents.context import Context
from ..events.event import Event
from ._base_node import BaseNode

logger = logging.getLogger('google_adk.' + __name__)


class JoinNode(BaseNode):
  """Waits for every predecessor, then emits their outputs keyed by name.

  The predecessors are the nodes with an edge into the join, such as ``a``
  and ``b`` in ``((a, b), join, c)``. The output is a dict that maps each
  predecessor's name to its latest output.

  In a loop, each firing waits for the predecessors inside the loop to run
  again: those the join can reach by following edges, including routes the
  run never takes, so the steps of an outer loop count as inside too. A
  predecessor the join cannot reach, such as a setup step that runs once
  before the loop starts, keeps counting in later iterations.
  """

  @property
  @override
  def _requires_all_predecessors(self) -> bool:
    return True

  @override
  def _validate_input_data(self, data: Any) -> Any:
    """Validates individual trigger inputs against input_schema."""
    if self.input_schema and isinstance(data, dict):
      return {
          k: self._validate_schema(v, self.input_schema)
          for k, v in data.items()
      }
    return super()._validate_input_data(data)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """JoinNode simply passes through the aggregated inputs provided by the orchestrator."""
    yield Event(
        output=node_input,
        branch=ctx._invocation_context.branch,
    )

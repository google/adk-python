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


def _get_common_branch_prefix(branches: list[str]) -> str:
  """Find the common prefix of dot-separated branch strings."""
  if not branches:
    return ''
  split_branches = [b.split('.') if b else [] for b in branches]

  common = []
  for segments in zip(*split_branches):
    if len(set(segments)) == 1:
      common.append(segments[0])
    else:
      break
  return '.'.join(common)


class JoinNode(BaseNode):
  """A node that waits for all specified predecessors to trigger it before
  outputting."""

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

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

"""Internal protocol for scheduling dynamic nodes with full result."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.context import Context


class ScheduleDynamicNode(Protocol):
  """Protocol for scheduling a dynamic node.

  Implementations handle the lifecycle of dynamically scheduled nodes (e.g.,
  via `ctx.run_node()`). This includes:
  1. Fresh Execution: Running a node for the first time.
  2. Deduplication: Returning cached output if the node already completed in a
     prior turn (based on event history).
  3. Resumption: Rehydrating state from session events when execution is
     resumed after an interrupt, and resolving or propagating remaining
     interrupts.

  Args:
    ctx: The calling node's Context.
    node: The node to execute. Usually a subclass of `BaseNode`.
    node_input: Input data for the node. Must match the node's input schema if
      defined.
    node_name: Deterministic tracking name. If None, uses `node.name`. This is
      critical for matching events on resume.
    use_as_output: If True, the child node's output will replace the calling
      node's output.
    run_id: A unique ID for this specific execution of the node.
    use_sub_branch: Whether the node should execute in an isolated sub-branch
      to prevent message history pollution.
    override_branch: Optional specific branch name to use, overriding defaults.

  Returns:
    Awaitable[Context]: A future that resolves to the child node's Context,
    which will contain the output, routing information, and any active
    interrupt IDs.

  Raises:
    ValueError: If input validation fails or if the node configuration is
      invalid on resume (e.g., waiting for output but called with
      `rerun_on_resume=False`).
    RuntimeError: If the execution reaches an inconsistent state.
  """

  def __call__(
      self,
      ctx: Context,
      node: Any,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
      run_id: str,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
  ) -> Awaitable[Context]:
    ...

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

"""A node that wraps an ADK Tool."""

from collections.abc import AsyncGenerator
from typing import Any
import uuid

from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ..agents.context import Context
from ..events.event import Event
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from ._base_node import BaseNode
from ._retry_config import RetryConfig


class _ToolNode(BaseNode):
  """A node that wraps an ADK Tool."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  tool: BaseTool = Field(...)

  def __init__(
      self,
      *,
      tool: BaseTool,
      name: str | None = None,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    super().__init__(
        tool=tool,
        name=name or tool.name,
        rerun_on_resume=False,
        retry_config=retry_config,
        timeout=timeout,
    )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    tool_context = ToolContext(
        invocation_context=ctx.get_invocation_context(),
        function_call_id=str(uuid.uuid4()),
    )

    args = node_input
    if args is None:
      args = {}
    elif not isinstance(args, dict):
      raise TypeError(
          'The input to ToolNode must be a dictionary of tool arguments or'
          f' None, but got {type(args)}.'
      )

    response = await self.tool.run_async(args=args, tool_context=tool_context)
    state_delta = (
        dict(tool_context.actions.state_delta)
        if tool_context.actions.state_delta
        else None
    )
    if response is not None:
      yield Event(
          output=response,
          state=state_delta,
      )
    elif state_delta:
      yield Event(state=state_delta)

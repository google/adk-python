# Copyright 2025 Google LLC
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

import asyncio
import inspect
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from ..utils.context_utils import Aclosing
from .function_tool import FunctionTool
from .progressive_function_tool import ProgressiveFunctionTool
from .tool_context import ToolContext


class ProgressiveTool(ProgressiveFunctionTool):
  """Wraps a regular async function to emit progress during run_async.

  Usage:
    from google.adk.tools.progressive_tool import ProgressiveTool
    ProgressiveTool(my_async_function)

  Supported function shapes:
    - async generator function: yields are treated as progress; last yielded
      value is treated as the final result.
    - async function with optional `progress` or `progress_callback` parameter:
      the wrapper injects a reporter callable that streams progress; the return
      value of the function is treated as the final result.
    - async function without any progress parameter: no progress is emitted; the
      return value is treated as the final result.
  """

  def __init__(self, func):
    # Initialize as FunctionTool to extract name/description and signature logic
    FunctionTool.__init__(self, func)
    self._results_by_call_id: dict[str, Any] = {}
    # Hide internal progress params from function declaration so the model is
    # never prompted for them and schema parsing doesn't fail.
    ignore_list = list(getattr(self, '_ignore_params', []))

    for p in ('progress', 'progress_callback'):
      if p not in ignore_list:
        ignore_list.append(p)

    self._ignore_params = ignore_list

  def _prepare_args_for_call(
      self, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    """Prepares arguments for the wrapped function call."""
    signature = inspect.signature(self.func)
    valid_params = {param for param in signature.parameters}
    args_to_call = {k: v for k, v in args.items() if k in valid_params}
    if 'tool_context' in valid_params:
      args_to_call['tool_context'] = tool_context
    return args_to_call

  async def progress_stream(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> AsyncGenerator[Any, None]:
    signature = inspect.signature(self.func)
    valid_params = {param for param in signature.parameters}

    # Build args for the wrapped function
    args_to_call = {k: v for k, v in args.items() if k in valid_params}
    if 'tool_context' in valid_params:
      args_to_call['tool_context'] = tool_context

    call_id: Optional[str] = tool_context.function_call_id

    # Async generator function: yield directly and capture last item
    if inspect.isasyncgenfunction(self.func):
      last: Any = None
      async with Aclosing(self.func(**args_to_call)) as agen:
        async for item in agen:
          last = item
          yield item
      if call_id:
        self._results_by_call_id[call_id] = last
      return

    # Coroutine function: run in background, capture progress via callback
    # Determine which progress parameter to use if present
    progress_param: Optional[str] = None
    if 'progress' in valid_params:
      progress_param = 'progress'
    elif 'progress_callback' in valid_params:
      progress_param = 'progress_callback'

    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _report_progress(payload: Any):
      await queue.put(payload)

    if progress_param:
      args_to_call[progress_param] = _report_progress

    result_box: dict[str, Any] = {}

    async def _run_and_capture():
      result_box['value'] = await self.func(**args_to_call)

    task = asyncio.create_task(_run_and_capture())

    # Drain progress while task runs
    try:
      while True:
        if task.done() and queue.empty():
          break
        try:
          item = await asyncio.wait_for(queue.get(), timeout=0.1)
          yield item
        except asyncio.TimeoutError:
          await asyncio.sleep(0)
          continue
    finally:
      # Ensure task completion / propagate exception
      await task

    if call_id:
      self._results_by_call_id[call_id] = result_box.get('value')

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Return final result. If progress_stream already ran, use captured value."""
    call_id: Optional[str] = tool_context.function_call_id
    if call_id and call_id in self._results_by_call_id:
      return self._results_by_call_id.pop(call_id)

    # Fallback: invoke function directly if progress_stream wasn't used
    signature = inspect.signature(self.func)
    valid_params = {param for param in signature.parameters}
    args_to_call = {k: v for k, v in args.items() if k in valid_params}
    if 'tool_context' in valid_params:
      args_to_call['tool_context'] = tool_context

    if inspect.isasyncgenfunction(self.func):
      # Consume generator fully; return last item
      last: Any = None
      async with Aclosing(self.func(**args_to_call)) as agen:
        async for item in agen:
          last = item
      return last

    # Coroutine function
    return await self.func(**args_to_call)

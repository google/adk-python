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

from typing import Any
from typing import AsyncGenerator

from .function_tool import FunctionTool
from .tool_context import ToolContext


class ProgressiveFunctionTool(FunctionTool):
  """A FunctionTool that can stream progress updates during run_async.

  Implement `progress_stream` to yield intermediate progress payloads.
  The final result for model consumption must be returned by `run_async`.
  """

  async def progress_stream(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> AsyncGenerator[Any, None]:
    """Yields progress updates while the tool is executing.

    Subclasses should override this method to emit progress objects. The last
    item yielded here does not need to be the final result; the final result
    should be returned by `run_async`.
    """
    raise NotImplementedError(
        f"{type(self).__name__}.progress_stream is not implemented"
    )

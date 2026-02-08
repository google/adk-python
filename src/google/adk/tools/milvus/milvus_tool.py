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

"""MilvusTool wraps a function and injects the MilvusVectorStore instance."""

from __future__ import annotations

import inspect
from typing import Any
from typing import Callable

from typing_extensions import override

from ..function_tool import FunctionTool
from ..tool_context import ToolContext
from .milvus_vector_store import MilvusVectorStore


class MilvusTool(FunctionTool):
  """A FunctionTool that injects MilvusVectorStore into the wrapped function.

  The ``vector_store`` parameter is hidden from the LLM function
  declaration and automatically injected at runtime.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      vector_store: MilvusVectorStore,
  ):
    super().__init__(func=func)
    self._ignore_params.append("vector_store")
    self._vector_store = vector_store

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if "vector_store" in signature.parameters:
      args_to_call["vector_store"] = self._vector_store
    return await super().run_async(args=args_to_call, tool_context=tool_context)

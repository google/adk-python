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

"""Retrieval tool that wraps a user-provided callable."""

from __future__ import annotations

import inspect
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Union

from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.adk.tools.tool_context import ToolContext
from typing_extensions import override


class CallableRetrieval(BaseRetrievalTool):
  """Retrieval tool backed by a user-provided function.

  Wraps any callable that accepts a query string and returns results,
  making it a first-class retrieval tool in ADK.

  Example:
      >>> def search_docs(query: str) -> list[str]:
      ...     return my_db.search(query)
      >>> tool = CallableRetrieval(
      ...     name="search_docs",
      ...     description="Search the knowledge base.",
      ...     retriever=search_docs,
      ... )

  Args:
      name: Tool name exposed to the LLM.
      description: Tool description exposed to the LLM.
      retriever: A sync or async callable. Must accept a ``query``
          string as its first argument. May optionally accept a
          ``tool_context`` parameter.
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      retriever: Union[
          Callable[[str], Any],
          Callable[[str], Awaitable[Any]],
      ],
  ):
    super().__init__(name=name, description=description)
    self._retriever = retriever
    self._pass_tool_context = (
        "tool_context" in inspect.signature(retriever).parameters
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    query = args["query"]
    kwargs = {"tool_context": tool_context} if self._pass_tool_context else {}
    result = self._retriever(query, **kwargs)
    if inspect.isawaitable(result):
      return await result
    return result

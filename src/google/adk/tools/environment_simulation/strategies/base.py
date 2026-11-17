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

from typing import Any
from typing import Dict
from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.genai import types as genai_types


class MockStrategy:
  """Base class for mock strategies."""

  def __init__(self, *, raise_on_error: bool = False):
    self._raise_on_error = raise_on_error

  def _error_response(
      self, tool: BaseTool, message: str, llm_output: str | None = None
  ) -> Dict[str, Any]:
    """Reports a simulator failure, without model output in strict diagnostics."""
    if self._raise_on_error:
      raise ValueError(
          f"Mock strategy for tool '{tool.name}' failed: {message}"
      )
    response = {"status": "error", "error_message": message}
    if llm_output is not None:
      response["llm_output"] = llm_output
    return response

  async def mock(
      self,
      tool: BaseTool,
      args: Dict[str, Any],
      tool_context: Any,
      tool_connection_map: Optional[ToolConnectionMap],
      state_store: Dict[str, Any],
      environment_data: Optional[str] = None,
      tracing: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Generates a mock response for a tool call."""
    raise NotImplementedError()


class TracingMockStrategy(MockStrategy):

  def __init__(
      self,
      llm_name: str = "",
      llm_config: Optional[genai_types.GenerateContentConfig] = None,
      *,
      raise_on_error: bool = False,
  ):
    super().__init__(raise_on_error=raise_on_error)
    self._llm_name = llm_name
    self._llm_config = llm_config

  async def mock(
      self,
      tool: BaseTool,
      args: Dict[str, Any],
      tool_context: Any,
      tool_connection_map: Optional[ToolConnectionMap],
      state_store: Dict[str, Any],
      environment_data: Optional[str] = None,
      tracing: Optional[str] = None,
  ) -> Dict[str, Any]:
    return self._error_response(tool, "Not implemented")

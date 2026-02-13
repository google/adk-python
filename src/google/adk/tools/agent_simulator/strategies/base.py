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

from google.adk.tools.agent_simulator.tool_connection_map import ToolConnectionMap


class MockStrategy:
  """Base class for mock strategies."""

  async def mock(
      self,
      tool: BaseTool,
      args: Dict[str, Any],
      tool_context: Any,
      tool_connection_map: Optional[ToolConnectionMap],
      state_store: Dict[str, Any],
      environment_data: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Generates a mock response for a tool call."""
    raise NotImplementedError()


class TracingMockStrategy(MockStrategy):
  """Mocks a tool response based on tracing and an LLM."""

  def __init__(
      self, llm_name: str, llm_config: genai_types.GenerateContentConfig
  ):
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
  ) -> Dict[str, Any]:
    # TODO: Implement tracing LLM-based mocking.
    return {"status": "error", "error_message": "Not implemented"}

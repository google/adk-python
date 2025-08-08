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

from typing import Optional

from google.genai import types
from typing_extensions import override

from .function_tool import FunctionTool
from .tool_context import ToolContext


def transfer_to_agent(agent_name: str, tool_context: ToolContext) -> None:
  """Transfer the question to another agent.

  This tool hands off control to another agent when it's more suitable to
  answer the user's question according to the agent's description.

  Args:
    agent_name: the agent name to transfer to.
  """
  tool_context.actions.transfer_to_agent = agent_name


class TransferToAgentTool(FunctionTool):
  """A specialized FunctionTool for agent transfer."""

  def __init__(
      self,
      agent_names: list[str],
  ):
    super().__init__(func=transfer_to_agent)
    self._agent_names = agent_names

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Add enum constraint to the agent_name."""
    function_decl = super()._get_declaration()
    if function_decl and function_decl.parameters:
      agent_name_schema = function_decl.parameters.properties.get("agent_name")
      if agent_name_schema:
        agent_name_schema.enum = self._agent_names
    return function_decl

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

import inspect
from typing import Optional

from google.genai import types
from typing_extensions import override

from .function_tool import FunctionTool
from .tool_context import ToolContext


def _transfer_to_agent_without_reason(
    agent_name: str, tool_context: ToolContext
) -> None:
  """Transfer the query to another agent.

  Use this tool to hand off control to another agent that is more suitable to
  answer the user's query according to the agent's description.

  Args:
    agent_name: the agent name to transfer to.
  """


_DOCSTRING_WITHOUT_REASON = inspect.cleandoc(
    _transfer_to_agent_without_reason.__doc__ or ''
)


# Note:
#   For most use cases, you should use TransferToAgentTool instead of this
#   function directly. TransferToAgentTool provides additional enum constraints
#   that prevent LLMs from hallucinating invalid agent names.
def transfer_to_agent(
    agent_name: str,
    tool_context: ToolContext,
    *,
    transfer_reason: str = '',
) -> None:
  """Transfer the query to another agent.

  Use this tool to hand off control to another agent that is more suitable to
  answer the user's query according to the agent's description.

  Args:
    agent_name: the agent name to transfer to.
    transfer_reason: the reason for transferring to the target agent.
  """
  tool_context.actions.transfer_to_agent = agent_name
  tool_context.actions.transfer_reason = transfer_reason or None


class TransferToAgentTool(FunctionTool):
  """A specialized FunctionTool for agent transfer with enum constraints.

  This tool enhances the base transfer_to_agent function by adding JSON Schema
  enum constraints to the agent_name parameter. This prevents LLMs from
  hallucinating invalid agent names by restricting choices to only valid agents.

  Attributes:
    agent_names: List of valid agent names that can be transferred to.
    include_transfer_reason: Whether to include the transfer_reason parameter
      in the tool declaration.
  """

  def __init__(
      self,
      agent_names: list[str],
      include_transfer_reason: bool = False,
  ):
    """Initialize the TransferToAgentTool.

    Args:
      agent_names: List of valid agent names that can be transferred to.
      include_transfer_reason: Whether to include the transfer_reason parameter
        in the tool declaration. Defaults to False.
    """
    super().__init__(func=transfer_to_agent)
    self._agent_names = agent_names
    self._include_transfer_reason = include_transfer_reason
    if not self._include_transfer_reason:
      self.description = _DOCSTRING_WITHOUT_REASON

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Add enum constraint to the agent_name parameter.

    Returns:
      FunctionDeclaration with enum constraint on agent_name parameter.
    """
    function_decl = super()._get_declaration()
    if not function_decl:
      return function_decl

    if not self._include_transfer_reason:
      function_decl.description = _DOCSTRING_WITHOUT_REASON

    # Handle parameters (types.Schema object)
    parameters = function_decl.parameters
    if parameters is not None and parameters.properties is not None:
      agent_name_schema = parameters.properties.get('agent_name')
      if agent_name_schema:
        agent_name_schema.enum = self._agent_names
      if (
          not self._include_transfer_reason
          and 'transfer_reason' in parameters.properties
      ):
        del parameters.properties['transfer_reason']

    # Handle parameters_json_schema (dict)
    if function_decl.parameters_json_schema:
      properties = function_decl.parameters_json_schema.get('properties', {})
      if 'agent_name' in properties:
        properties['agent_name']['enum'] = self._agent_names
      if not self._include_transfer_reason and 'transfer_reason' in properties:
        del properties['transfer_reason']

    return function_decl

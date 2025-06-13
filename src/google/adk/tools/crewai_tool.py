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

from google.genai import types
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool
from ..approval.approval_policy import FunctionToolPolicy, register_policy_for_tool

try:
  from crewai.tools import BaseTool as CrewaiBaseTool
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "Crewai Tools require Python 3.10+. Please upgrade your Python version."
    ) from e
  else:
    raise ImportError(
        "Crewai Tools require pip install 'google-adk[extensions]'."
    ) from e


class CrewaiTool(FunctionTool):
  """Use this class to wrap a CrewAI tool.

  If the original tool name and description are not suitable, you can override
  them in the constructor.
  """

  tool: CrewaiBaseTool
  """The wrapped CrewAI tool."""

  def __init__(
      self,
      tool: CrewaiBaseTool,
      *,
      name: str,
      description: str,
      policies: list[FunctionToolPolicy] = None,
  ):
    """Initializes a CrewaiTool, wrapping a CrewAI BaseTool, and registers policies.

    Args:
        tool: The CrewAI `BaseTool` instance to wrap.
        name: The name for this tool. If empty, it attempts to use `tool.name`
              (replacing spaces with underscores and lowercasing).
        description: The description for this tool. If empty, it attempts to use
                     `tool.description`.
        policies: An optional list of `FunctionToolPolicy` objects to register for this tool.
                  Each policy defines actions and resource mappings for approval.
    """
    super().__init__(tool.run)
    self.tool = tool
    if name:
      self.name = name
    elif tool.name:
      # Right now, CrewAI tool name contains white spaces. White spaces are
      # not supported in our framework. So we replace them with "_".
      self.name = tool.name.replace(" ", "_").lower()
    if description:
      self.description = description
    elif tool.description:
      self.description = tool.description
    for policy in policies or []:
      register_policy_for_tool(self, policy)

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool."""
    function_declaration = _automatic_function_calling_util.build_function_declaration_for_params_for_crewai(
        False,
        self.name,
        self.description,
        self.func,
        self.tool.args_schema.model_json_schema(),
    )
    return function_declaration

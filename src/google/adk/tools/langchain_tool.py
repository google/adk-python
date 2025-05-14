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

from typing import Any
from typing import Callable

from google.genai import types
from pydantic import model_validator
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool
from ..approval.approval_policy import FunctionToolPolicy, register_policy_for_tool


class LangchainTool(FunctionTool):
  """Use this class to wrap a langchain tool.

  If the original tool name and description are not suitable, you can override
  them in the constructor.
  """

  tool: Any
  """The wrapped langchain tool."""

  def __init__(self, tool: Any, policies: list[FunctionToolPolicy] = None):
    super().__init__(tool._run)
    self.tool = tool
    if tool.name:
      self.name = tool.name
    if tool.description:
      self.description = tool.description
    for policy in (policies or []):
      register_policy_for_tool(self, policy)


  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    # Override this to not use function's signature name as it's
    # mostly "run" or "invoke" for thir-party tools.
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool."""
    from langchain.agents import Tool
    from langchain_core.tools import BaseTool

    # There are two types of tools:
    # 1. BaseTool: the tool is defined in langchain.tools.
    # 2. Other tools: the tool doesn't inherit any class but follow some
    #    conventions, like having a "run" method.
    if isinstance(self.tool, BaseTool):
      tool_wrapper = Tool(
          name=self.name,
          func=self.func,
          description=self.description,
      )
      if self.tool.args_schema:
        tool_wrapper.args_schema = self.tool.args_schema
      function_declaration = _automatic_function_calling_util.build_function_declaration_for_langchain(
          False,
          self.name,
          self.description,
          tool_wrapper.func,
          tool_wrapper.args,
      )
      return function_declaration
    else:
      # Need to provide a way to override the function names and descriptions
      # as the original function names are mostly ".run" and the descriptions
      # may not meet users' needs.
      function_declaration = (
          _automatic_function_calling_util.build_function_declaration(
              func=self.tool.run,
          )
      )
      return function_declaration

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

from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.utils.agent_info import get_tools_info
from google.genai import types
import pytest


class _CountingTool(BaseTool):
  """A tool that records how many times its declaration was requested."""

  def __init__(self, name: str, *, declared: bool = True):
    super().__init__(name=name, description=f'{name} description')
    self.declaration_calls = 0
    self._declared = declared

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    self.declaration_calls += 1
    if not self._declared:
      return None
    return types.FunctionDeclaration(
        name=self.name, description=self.description
    )


class _CountingToolset(BaseToolset):

  def __init__(self, tools: list[BaseTool]):
    super().__init__()
    self._tools = tools

  async def get_tools(self, readonly_context=None) -> list[BaseTool]:
    return self._tools

  async def close(self) -> None:
    pass


@pytest.mark.asyncio
async def test_get_tools_info_calls_get_declaration_once_per_tool():
  declared = _CountingTool('declared_tool')
  undeclared = _CountingTool('undeclared_tool', declared=False)
  in_toolset = _CountingTool('toolset_tool')

  tools_info = await get_tools_info(
      [declared, undeclared, _CountingToolset([in_toolset])]
  )

  assert declared.declaration_calls == 1
  assert undeclared.declaration_calls == 1
  assert in_toolset.declaration_calls == 1
  assert tools_info == [
      types.Tool(
          function_declarations=[
              types.FunctionDeclaration(
                  name='declared_tool', description='declared_tool description'
              )
          ]
      ),
      types.Tool(
          function_declarations=[
              types.FunctionDeclaration(
                  name='toolset_tool', description='toolset_tool description'
              )
          ]
      ),
  ]


@pytest.mark.asyncio
async def test_get_tools_info_wraps_plain_callable():
  def echo(text: str) -> str:
    """Echoes the text."""
    return text

  tools_info = await get_tools_info([echo])

  assert len(tools_info) == 1
  declaration = tools_info[0].function_declarations[0]
  assert declaration.name == 'echo'
  assert declaration.description == 'Echoes the text.'

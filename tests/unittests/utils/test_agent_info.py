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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.utils.agent_info import get_agents_dict
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

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    return self._tools

  async def close(self) -> None:
    pass


def _declaration_names(tools: list[types.Tool]) -> list[str]:
  return [tool.function_declarations[0].name for tool in tools]


def _declared_parameters(
    declaration: types.FunctionDeclaration,
) -> dict[str, object]:
  """Returns the declared parameters whichever schema field is populated."""
  if declaration.parameters_json_schema is not None:
    return declaration.parameters_json_schema['properties']
  return declaration.parameters.properties


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
  # The callable is adapted into a FunctionTool, so its name, docstring and
  # signature become the declaration the model sees.
  assert declaration.name == 'echo'
  assert declaration.description == 'Echoes the text.'
  assert list(_declared_parameters(declaration)) == ['text']


@pytest.mark.asyncio
async def test_get_tools_info_empty_input_returns_empty_list():
  assert await get_tools_info([]) == []


@pytest.mark.asyncio
async def test_get_tools_info_wraps_each_declaration_in_its_own_tool():
  tools_info = await get_tools_info(
      [_CountingTool('alpha'), _CountingTool('beta')]
  )

  # One types.Tool per tool, in input order, each holding exactly one
  # declaration rather than all declarations being merged into one Tool.
  assert _declaration_names(tools_info) == ['alpha', 'beta']
  assert [len(t.function_declarations) for t in tools_info] == [1, 1]


@pytest.mark.asyncio
async def test_get_tools_info_flattens_toolset_into_its_tools():
  toolset = _CountingToolset(
      [_CountingTool('inner_one'), _CountingTool('inner_two')]
  )

  tools_info = await get_tools_info([_CountingTool('outer'), toolset])

  # The toolset itself is never reported; it is replaced in place by the
  # tools it resolves to.
  assert _declaration_names(tools_info) == ['outer', 'inner_one', 'inner_two']


@pytest.mark.asyncio
async def test_get_tools_info_omits_tools_without_a_declaration():
  tools_info = await get_tools_info(
      [_CountingTool('hidden', declared=False), _CountingTool('visible')]
  )

  assert _declaration_names(tools_info) == ['visible']


@pytest.mark.asyncio
async def test_get_agents_dict_single_agent_has_no_sub_agents():
  agent = LlmAgent(
      name='root', description='the root', instruction='be helpful'
  )

  agents = await get_agents_dict(agent)

  assert list(agents) == ['root']
  assert agents['root'].description == 'the root'
  assert agents['root'].instruction == 'be helpful'
  assert agents['root'].sub_agents == []
  assert agents['root'].tools == []


@pytest.mark.asyncio
async def test_get_agents_dict_coerces_callable_instruction():
  def dynamic_instruction(ctx: ReadonlyContext) -> str:
    raise RuntimeError('app-info must not resolve InstructionProvider')

  agent = LlmAgent(
      name='dyn',
      description='Agent with a dynamic (callable) instruction.',
      instruction=dynamic_instruction,
  )

  agents = await get_agents_dict(agent)

  assert agents['dyn'].instruction == (
      '<InstructionProvider: dynamic_instruction>'
  )


@pytest.mark.asyncio
async def test_get_agents_dict_coerces_async_and_subagent_callables():
  async def async_instruction(ctx: ReadonlyContext) -> str:
    return 'async'

  def child_instruction(ctx: ReadonlyContext) -> str:
    return 'child'

  child = LlmAgent(name='child', instruction=child_instruction)
  root = LlmAgent(
      name='root', instruction=async_instruction, sub_agents=[child]
  )

  agents = await get_agents_dict(root)

  assert agents['root'].instruction == (
      '<InstructionProvider: async_instruction>'
  )
  assert agents['child'].instruction == (
      '<InstructionProvider: child_instruction>'
  )


@pytest.mark.asyncio
async def test_get_agents_dict_coerces_instruction_provider_instance():
  class Persona:

    def __call__(self, ctx: ReadonlyContext) -> str:
      return 'persona'

  agent = LlmAgent(name='dyn', instruction=Persona())

  agents = await get_agents_dict(agent)

  assert agents['dyn'].instruction == '<InstructionProvider: Persona>'


@pytest.mark.asyncio
async def test_get_agents_dict_includes_transitively_nested_agents():
  grandchild = LlmAgent(name='grandchild')
  child = LlmAgent(name='child', sub_agents=[grandchild])
  root = LlmAgent(name='root', sub_agents=[child])

  agents = await get_agents_dict(root)

  # Every agent in the tree is keyed by its own name, not just the direct
  # children of the root.
  assert set(agents) == {'root', 'child', 'grandchild'}


@pytest.mark.asyncio
async def test_get_agents_dict_records_only_direct_children_per_agent():
  grandchild = LlmAgent(name='grandchild')
  child = LlmAgent(name='child', sub_agents=[grandchild])
  sibling = LlmAgent(name='sibling')
  root = LlmAgent(name='root', sub_agents=[child, sibling])

  agents = await get_agents_dict(root)

  assert agents['root'].sub_agents == ['child', 'sibling']
  assert agents['child'].sub_agents == ['grandchild']
  assert agents['grandchild'].sub_agents == []


@pytest.mark.asyncio
async def test_get_agents_dict_reports_each_agents_own_tools():
  child = LlmAgent(name='child', tools=[_CountingTool('child_tool')])
  root = LlmAgent(
      name='root',
      tools=[_CountingTool('root_tool')],
      sub_agents=[child],
  )

  agents = await get_agents_dict(root)

  # Tools are per-agent; a parent does not inherit its child's tools.
  assert _declaration_names(agents['root'].tools) == ['root_tool']
  assert _declaration_names(agents['child'].tools) == ['child_tool']

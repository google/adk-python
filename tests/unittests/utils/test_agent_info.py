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

import logging

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools._node_tool import NodeTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.utils.agent_info import get_agents_dict
from google.adk.utils.agent_info import get_tools_info
from google.adk.workflow import Edge
from google.adk.workflow import JoinNode
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
import pytest


class _CountingTool(BaseTool):
  """A tool that records how many times its declaration was requested."""

  def __init__(self, name: str, *, declared: bool = True):
    super().__init__(name=name, description=f'{name} description')
    self.declaration_calls = 0
    self._declared = declared

  def _get_declaration(self) -> types.FunctionDeclaration | None:
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
      self, readonly_context: ReadonlyContext | None = None
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
async def test_get_tools_info_wraps_base_node():
  """Adapts a BaseNode into a NodeTool and extracts its declaration."""

  @node
  def double(x: int) -> int:
    """Doubles x."""
    return x * 2

  tools_info = await get_tools_info([double])

  assert len(tools_info) == 1
  declaration = tools_info[0].function_declarations[0]
  assert declaration.name == 'double'
  assert declaration.description == 'Doubles x.'
  assert list(_declared_parameters(declaration)) == ['x']


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


@pytest.mark.asyncio
async def test_get_agents_dict_collects_llm_agents_from_workflow():
  """Collects LlmAgent nodes from a Workflow while omitting non-LLM nodes."""

  @node
  def preprocess(x: str) -> str:
    return x

  join = JoinNode(name='merge')
  weather = LlmAgent(
      name='weather_agent',
      description='Fetches weather.',
      instruction='Use tool.',
      tools=[_CountingTool('tell_weather')],
  )
  translator = LlmAgent(
      name='translate_agent',
      description='Translates text.',
      instruction='Translate to German.',
  )
  workflow = Workflow(
      name='weather_workflow',
      edges=[
          (START, preprocess, (weather, translator), join),
      ],
  )

  agents = await get_agents_dict(workflow)

  assert set(agents) == {'weather_agent', 'translate_agent'}
  assert _declaration_names(agents['weather_agent'].tools) == ['tell_weather']
  assert agents['weather_agent'].sub_agents == []
  assert agents['translate_agent'].sub_agents == []


@pytest.mark.asyncio
async def test_get_agents_dict_traverses_nested_workflows_and_shell_agents():
  """Finds LlmAgents nested inside Workflows, shell agents, and workers."""

  @node
  def noop() -> str:
    return 'ok'

  seq_leaf = LlmAgent(name='seq_leaf')
  par_leaf = LlmAgent(name='par_leaf')
  loop_leaf = LlmAgent(name='loop_leaf')
  worker_leaf = LlmAgent(name='worker_leaf')
  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, SequentialAgent(name='seq', sub_agents=[seq_leaf]))],
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[
          (
              START,
              inner_wf,
              ParallelAgent(name='par', sub_agents=[par_leaf]),
              LoopAgent(name='loop', sub_agents=[loop_leaf]),
              node(worker_leaf, parallel_worker=True),
              noop,
          ),
      ],
  )

  agents = await get_agents_dict(outer_wf)

  assert set(agents) == {'seq_leaf', 'par_leaf', 'loop_leaf', 'worker_leaf'}


@pytest.mark.asyncio
async def test_get_agents_dict_flattens_non_llm_sub_agent_containers():
  """Links LlmAgent parents to nearest LlmAgent descendants across shells."""
  grandchild = LlmAgent(name='grandchild')
  first = LlmAgent(name='first', sub_agents=[grandchild])
  second = LlmAgent(name='second')
  shell = SequentialAgent(name='shell', sub_agents=[first, second])
  root = LlmAgent(name='root', sub_agents=[shell])

  agents = await get_agents_dict(root)

  assert set(agents) == {'root', 'first', 'second', 'grandchild'}
  assert agents['root'].sub_agents == ['first', 'second']
  assert agents['first'].sub_agents == ['grandchild']


@pytest.mark.asyncio
async def test_get_agents_dict_visits_node_tool_without_adding_sub_agent():
  """Includes LlmAgents inside a NodeTool without listing them as sub_agents."""
  helper = LlmAgent(name='helper', instruction='Help.')
  tool_wf = Workflow(name='tool_wf', edges=[(START, helper)])
  root = LlmAgent(
      name='root',
      tools=[NodeTool(node=tool_wf, name='run_helper')],
  )

  agents = await get_agents_dict(root)

  assert set(agents) == {'root', 'helper'}
  assert agents['root'].sub_agents == []


@pytest.mark.asyncio
async def test_get_agents_dict_does_not_traverse_into_agent_tool():
  """Leaves LlmAgents wrapped in an AgentTool out of the result."""
  wrapped = LlmAgent(name='wrapped', instruction='Help.')
  root = LlmAgent(name='root', tools=[AgentTool(agent=wrapped)])

  agents = await get_agents_dict(root)

  assert set(agents) == {'root'}
  assert agents['root'].sub_agents == []
  assert _declaration_names(agents['root'].tools) == ['wrapped']


@pytest.mark.asyncio
async def test_get_agents_dict_terminates_on_workflow_cycle_and_dag():
  """Terminates cleanly on routed cycles and shared nodes in a Workflow."""
  reviewer = LlmAgent(name='reviewer')
  publisher = LlmAgent(name='publisher')
  shared = LlmAgent(name='shared')
  workflow = Workflow(
      name='cyclic_wf',
      edges=[
          Edge(from_node=START, to_node=reviewer),
          Edge(from_node=reviewer, to_node=shared, route='draft'),
          Edge(from_node=shared, to_node=reviewer, route='retry'),
          Edge(from_node=reviewer, to_node=publisher, route='approved'),
          Edge(from_node=shared, to_node=publisher, route='skip'),
      ],
  )

  agents = await get_agents_dict(workflow)

  assert set(agents) == {'reviewer', 'shared', 'publisher'}


@pytest.mark.asyncio
async def test_get_agents_dict_empty_when_no_llm_agents_exist():
  """Returns an empty dict for trees without any LlmAgent."""

  @node
  def step_one() -> str:
    return 'one'

  workflow = Workflow(name='fn_only', edges=[(START, step_one)])
  base_only = BaseAgent(name='custom_base')

  assert await get_agents_dict(workflow) == {}
  assert await get_agents_dict(base_only) == {}


@pytest.mark.asyncio
async def test_get_agents_dict_keeps_first_entry_on_duplicate_llm_name(
    caplog: pytest.LogCaptureFixture,
):
  """Logs a warning and keeps the first entry when LlmAgent names collide."""
  first = LlmAgent(name='dup', description='first')
  second = LlmAgent(name='dup', description='second')
  inner_wf = Workflow(name='inner_wf', edges=[(START, second)])
  root = LlmAgent(
      name='root',
      tools=[NodeTool(node=inner_wf, name='inner_tool')],
      sub_agents=[first],
  )

  with caplog.at_level(logging.WARNING):
    agents = await get_agents_dict(root)

  assert agents['dup'].description == 'first'
  assert any('Duplicate LlmAgent name' in r.message for r in caplog.records)

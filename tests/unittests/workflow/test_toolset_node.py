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

"""Tests for ToolsetNode, which runs a named tool resolved from a toolset."""

from typing import Any
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events.event import Event
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.workflow import START
from google.adk.workflow import ToolsetNode
from google.adk.workflow._workflow import Workflow
import pytest

from . import workflow_testing_utils
from .. import testing_utils


class _EchoTool(BaseTool):
  """A tool that returns the args it was called with."""

  async def run_async(self, *, args: dict[str, Any], tool_context) -> Any:
    return args


class _RecordingToolset(BaseToolset):
  """A toolset that serves fixed tools and counts how often it is listed."""

  def __init__(self, *tool_names: str, tool_name_prefix: Optional[str] = None):
    super().__init__(tool_name_prefix=tool_name_prefix)
    self._tool_names = tool_names
    self.get_tools_call_count = 0

  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    self.get_tools_call_count += 1
    return [
        _EchoTool(name=name, description=f'Echoes for {name}')
        for name in self._tool_names
    ]


async def _run(wf: Workflow) -> list[Any]:
  """Runs a workflow and returns its simplified events."""
  app_instance = testing_utils.App(name='test_app', root_agent=wf)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async('start')
  return workflow_testing_utils.simplify_events_with_node(events)


@pytest.mark.asyncio
async def test_named_tool_receives_the_node_input_as_arguments():
  """The tool named by the node is resolved and called with the node input."""
  toolset = _RecordingToolset('search', 'fetch')
  args = {'query': 'adk'}

  def start_node():
    return Event(output=args)

  simplified = await _run(
      Workflow(
          name='wf',
          edges=[
              (START, start_node),
              (start_node, ToolsetNode(toolset=toolset, tool_name='search')),
          ],
      )
  )

  assert ('wf@1/search@1', {'output': args}) in simplified


@pytest.mark.asyncio
async def test_unknown_tool_name_reports_the_available_tools():
  """Naming a tool the toolset does not serve fails with a usable message."""
  toolset = _RecordingToolset('search', 'fetch')

  def start_node():
    return Event(output={})

  wf = Workflow(
      name='wf',
      edges=[
          (START, start_node),
          (start_node, ToolsetNode(toolset=toolset, tool_name='missing')),
      ],
  )

  with pytest.raises(
      ValueError, match=r"'missing'.*Available tools: fetch, search"
  ):
    await _run(wf)


@pytest.mark.asyncio
async def test_tool_name_matches_the_toolsets_prefixed_name():
  """A toolset's tool_name_prefix is part of the name the node matches."""
  toolset = _RecordingToolset('search', tool_name_prefix='web')

  def start_node():
    return Event(output={'query': 'adk'})

  simplified = await _run(
      Workflow(
          name='wf',
          edges=[
              (START, start_node),
              (
                  start_node,
                  ToolsetNode(toolset=toolset, tool_name='web_search'),
              ),
          ],
      )
  )

  assert ('wf@1/web_search@1', {'output': {'query': 'adk'}}) in simplified


@pytest.mark.asyncio
async def test_nodes_sharing_a_toolset_list_its_tools_once_per_run():
  """Resolution is cached per invocation, so one run lists tools once."""
  toolset = _RecordingToolset('search', 'fetch')

  def start_node():
    return Event(output={})

  await _run(
      Workflow(
          name='wf',
          edges=[
              (START, start_node),
              (
                  start_node,
                  ToolsetNode(toolset=toolset, tool_name='search'),
                  ToolsetNode(toolset=toolset, tool_name='fetch'),
              ),
          ],
      )
  )

  assert toolset.get_tools_call_count == 1


@pytest.mark.asyncio
async def test_tool_name_that_is_not_an_identifier_becomes_a_valid_node_name():
  """A dashed tool name, common for MCP servers, yields a usable node name."""
  toolset = _RecordingToolset('read-file')

  def start_node():
    return Event(output={'path': '/tmp/x'})

  simplified = await _run(
      Workflow(
          name='wf',
          edges=[
              (START, start_node),
              (start_node, ToolsetNode(toolset=toolset, tool_name='read-file')),
          ],
      )
  )

  assert ('wf@1/read_file@1', {'output': {'path': '/tmp/x'}}) in simplified


def test_explicit_name_overrides_the_derived_node_name():
  """An explicit name wins over the name derived from tool_name."""
  node = ToolsetNode(
      toolset=_RecordingToolset('read-file'),
      tool_name='read-file',
      name='reader',
  )

  assert node.name == 'reader'


@pytest.mark.asyncio
async def test_state_written_by_the_tool_reaches_later_nodes():
  """State the tool sets on its context is persisted for downstream nodes."""

  class _StatefulTool(BaseTool):

    async def run_async(self, *, args, tool_context):
      tool_context.state['tool_key'] = 'tool_value'
      return {'status': 'ok'}

  class _StatefulToolset(BaseToolset):

    async def get_tools(self, readonly_context=None) -> list[BaseTool]:
      return [_StatefulTool(name='stateful', description='Sets state')]

  def start_node():
    return Event(output={})

  def read_state(tool_key: str) -> str:
    return f'tool_key={tool_key}'

  tool_node = ToolsetNode(toolset=_StatefulToolset(), tool_name='stateful')

  simplified = await _run(
      Workflow(
          name='wf',
          edges=[
              (START, start_node),
              (start_node, tool_node),
              (tool_node, read_state),
          ],
      )
  )

  assert ('wf@1/read_state@1', {'output': 'tool_key=tool_value'}) in simplified

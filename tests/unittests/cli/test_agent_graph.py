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

"""Tests for the DOT graph the dev UI renders for an agent tree."""

from __future__ import annotations

import re

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.cli.agent_graph import get_agent_graph
from google.adk.tools.agent_tool import AgentTool
import pytest

_DARK_GREEN = '#0F5223'
_LIGHT_GREEN = '#69CB87'
_LIGHT_GRAY = '#cccccc'

_EDGE_RE = re.compile(
    r'^(?P<src>"[^"]+"|[^\s\[]+) -> (?P<dst>"[^"]+"|[^\s\[]+)'
    r'(?: \[(?P<attrs>.*)\])?$'
)
_NODE_RE = re.compile(r'^(?P<name>"[^"]+"|[^\s\[]+) \[(?P<attrs>.*)\]$')
_ATTR_RE = re.compile(r'(\w+)=("[^"]*"|[^\s\]]+)')

# Graph-level defaults, not agent/tool nodes.
_DOT_KEYWORDS = frozenset({'graph', 'node', 'edge'})


def _unquote(value: str) -> str:
  return value[1:-1] if value.startswith('"') and value.endswith('"') else value


def _attrs(attr_text: str) -> dict[str, str]:
  return {
      key: _unquote(value) for key, value in _ATTR_RE.findall(attr_text or '')
  }


def _parse(source: str) -> tuple[dict[str, dict[str, str]], dict[tuple, dict]]:
  """Splits DOT source into {node_name: attrs} and {(src, dst): attrs}."""
  nodes: dict[str, dict[str, str]] = {}
  edges: dict[tuple[str, str], dict[str, str]] = {}
  for raw_line in source.splitlines():
    line = raw_line.strip()
    edge_match = _EDGE_RE.match(line)
    if edge_match:
      key = (_unquote(edge_match['src']), _unquote(edge_match['dst']))
      edges[key] = _attrs(edge_match['attrs'])
      continue
    node_match = _NODE_RE.match(line)
    if node_match:
      name = _unquote(node_match['name'])
      if name in _DOT_KEYWORDS:
        continue
      nodes[name] = _attrs(node_match['attrs'])
  return nodes, edges


def roll_dice(sides: int) -> int:
  """Rolls a die with the given number of sides."""
  return sides


def check_prime(number: int) -> bool:
  """Checks whether a number is prime."""
  return number == 2


def _tree_with_sub_agent_and_tools() -> LlmAgent:
  """root -> [child -> roll_dice], plus check_prime and an AgentTool."""
  child = LlmAgent(name='child', model='gemini-2.0-flash', tools=[roll_dice])
  quoted = LlmAgent(name='quoted_agent', model='gemini-2.0-flash')
  return LlmAgent(
      name='root',
      model='gemini-2.0-flash',
      sub_agents=[child],
      tools=[check_prime, AgentTool(quoted)],
  )


@pytest.mark.asyncio
async def test_build_graph_llm_tree_has_exactly_the_agent_and_tool_nodes():
  graph = await get_agent_graph(_tree_with_sub_agent_and_tools(), [])

  nodes, edges = _parse(graph.source)

  assert set(nodes) == {
      'root',
      'child',
      'roll_dice',
      'check_prime',
      'quoted_agent',
  }
  assert set(edges) == {
      ('root', 'child'),
      ('child', 'roll_dice'),
      ('root', 'check_prime'),
      ('root', 'quoted_agent'),
  }


@pytest.mark.asyncio
async def test_build_graph_shapes_distinguish_agents_tools_and_agent_tools():
  graph = await get_agent_graph(_tree_with_sub_agent_and_tools(), [])

  nodes, _ = _parse(graph.source)

  # A sub-agent is an ellipse; anything reached as a tool is a box.
  assert nodes['child']['shape'] == 'ellipse'
  assert nodes['child']['label'] == '🤖 child'
  assert nodes['roll_dice']['shape'] == 'box'
  assert nodes['roll_dice']['label'] == '🔧 roll_dice'
  # An AgentTool is drawn as a tool (box) but captioned as an agent.
  assert nodes['quoted_agent']['shape'] == 'box'
  assert nodes['quoted_agent']['label'] == '🤖 quoted_agent'


@pytest.mark.asyncio
async def test_build_graph_sequential_agent_chains_sub_agents_in_a_cluster():
  pipeline = SequentialAgent(
      name='pipeline',
      sub_agents=[
          LlmAgent(name='first', model='gemini-2.0-flash'),
          LlmAgent(name='second', model='gemini-2.0-flash'),
      ],
  )
  root = LlmAgent(name='root', model='gemini-2.0-flash', sub_agents=[pipeline])

  graph = await get_agent_graph(root, [])

  nodes, edges = _parse(graph.source)
  # The workflow agent itself is a cluster, never a node, and the parent
  # connects straight to the first step.
  assert set(nodes) == {'root', 'first', 'second'}
  assert set(edges) == {('root', 'first'), ('first', 'second')}
  assert 'subgraph "cluster_pipeline (Sequential Agent)"' in graph.source


@pytest.mark.asyncio
async def test_build_graph_loop_agent_closes_the_cycle_to_the_first_sub_agent():
  loop = LoopAgent(
      name='looper',
      sub_agents=[
          LlmAgent(name='first', model='gemini-2.0-flash'),
          LlmAgent(name='second', model='gemini-2.0-flash'),
      ],
  )

  graph = await get_agent_graph(loop, [])

  nodes, edges = _parse(graph.source)
  assert set(nodes) == {'first', 'second'}
  # Last step loops back to the first one.
  assert set(edges) == {('first', 'second'), ('second', 'first')}
  assert 'subgraph "cluster_looper (Loop Agent)"' in graph.source


@pytest.mark.asyncio
async def test_build_graph_parallel_agent_fans_out_from_the_parent():
  parallel = ParallelAgent(
      name='fanout',
      sub_agents=[
          LlmAgent(name='first', model='gemini-2.0-flash'),
          LlmAgent(name='second', model='gemini-2.0-flash'),
      ],
  )
  root = LlmAgent(name='root', model='gemini-2.0-flash', sub_agents=[parallel])

  graph = await get_agent_graph(root, [])

  nodes, edges = _parse(graph.source)
  assert set(nodes) == {'root', 'first', 'second'}
  # No edge between the branches: the parent points at each of them.
  assert set(edges) == {('root', 'first'), ('root', 'second')}
  assert 'subgraph "cluster_fanout (Parallel Agent)"' in graph.source


@pytest.mark.asyncio
async def test_build_graph_highlight_pair_fills_both_nodes_and_colors_edge():
  graph = await get_agent_graph(
      _tree_with_sub_agent_and_tools(), [('root', 'check_prime')]
  )

  nodes, edges = _parse(graph.source)

  assert nodes['root']['fillcolor'] == _DARK_GREEN
  assert nodes['root']['style'] == 'filled,rounded'
  assert nodes['check_prime']['fillcolor'] == _DARK_GREEN
  assert edges[('root', 'check_prime')]['color'] == _LIGHT_GREEN
  # Untouched parts of the tree stay gray and unfilled.
  assert 'fillcolor' not in nodes['child']
  assert nodes['child']['color'] == _LIGHT_GRAY
  assert edges[('root', 'child')]['color'] == _LIGHT_GRAY


@pytest.mark.asyncio
async def test_build_graph_reversed_highlight_pair_draws_a_back_edge():
  # The pair is (callee, caller); the drawn edge still runs caller -> callee,
  # so it has to be flipped visually instead of duplicated.
  graph = await get_agent_graph(
      _tree_with_sub_agent_and_tools(), [('check_prime', 'root')]
  )

  _, edges = _parse(graph.source)

  assert edges[('root', 'check_prime')]['color'] == _LIGHT_GREEN
  assert edges[('root', 'check_prime')]['dir'] == 'back'


@pytest.mark.asyncio
async def test_get_agent_graph_dark_mode_selects_the_background_color():
  agent = LlmAgent(name='root', model='gemini-2.0-flash')

  dark = await get_agent_graph(agent, [], dark_mode=True)
  light = await get_agent_graph(agent, [], dark_mode=False)

  assert 'bgcolor="#333537"' in dark.source
  assert 'bgcolor="#ffffff"' in light.source
  assert 'rankdir=LR' in dark.source


@pytest.mark.asyncio
async def test_get_agent_graph_is_strict_so_repeated_edges_collapse():
  # The same tool is attached to a parent and its sub-agent, which makes
  # build_graph emit the child -> tool edge twice.
  shared = LlmAgent(name='child', model='gemini-2.0-flash', tools=[roll_dice])
  root = LlmAgent(
      name='root',
      model='gemini-2.0-flash',
      sub_agents=[shared],
      tools=[roll_dice],
  )

  graph = await get_agent_graph(root, [])

  assert graph.strict
  assert graph.source.count('child -> roll_dice') == 1

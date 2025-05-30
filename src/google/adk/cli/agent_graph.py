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

import logging
from typing import Union

import graphviz

from ..agents import BaseAgent
from ..agents.llm_agent import LlmAgent
from ..tools.agent_tool import AgentTool
from ..tools.base_tool import BaseTool
from ..tools.function_tool import FunctionTool

logger = logging.getLogger('google_adk.' + __name__)

try:
  from ..tools.retrieval.base_retrieval_tool import BaseRetrievalTool
except ModuleNotFoundError:
  retrieval_tool_module_loaded = False
else:
  retrieval_tool_module_loaded = True


async def build_graph(graph, agent: BaseAgent, highlight_pairs):
  dark_green = '#0F5223'
  light_green = '#69CB87'
  light_gray = '#cccccc'

  def get_node_name(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return tool_or_agent.name
    elif isinstance(tool_or_agent, BaseTool):
      return tool_or_agent.name
    else:
      raise ValueError(f'Unsupported tool type: {tool_or_agent}')

  def get_node_caption(tool_or_agent: Union[BaseAgent, BaseTool]):

    if isinstance(tool_or_agent, BaseAgent):
      return '🤖 ' + tool_or_agent.name
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return '🔎 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, FunctionTool):
      return '🔧 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, AgentTool):
      return '🤖 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, BaseTool):
      return '🔧 ' + tool_or_agent.name
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return f'❓ Unsupported tool type: {type(tool_or_agent)}'

  def get_node_shape(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return 'ellipse'
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return 'cylinder'
    elif isinstance(tool_or_agent, FunctionTool):
      return 'box'
    elif isinstance(tool_or_agent, BaseTool):
      return 'box'
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return 'cylinder'

  def draw_node(tool_or_agent: Union[BaseAgent, BaseTool]):
    name = get_node_name(tool_or_agent)
    shape = get_node_shape(tool_or_agent)
    caption = get_node_caption(tool_or_agent)
    if highlight_pairs:
      for highlight_tuple in highlight_pairs:
        if name in highlight_tuple:
          graph.node(
              name,
              caption,
              style='filled,rounded',
              fillcolor=dark_green,
              color=dark_green,
              shape=shape,
              fontcolor=light_gray,
          )
          return
    # if not in highlight, draw non-highliht node
    graph.node(
        name,
        caption,
        shape=shape,
        style='rounded',
        color=light_gray,
        fontcolor=light_gray,
    )

  def draw_edge(from_name, to_name):
    if highlight_pairs:
      for highlight_from, highlight_to in highlight_pairs:
        if from_name == highlight_from and to_name == highlight_to:
          graph.edge(from_name, to_name, color=light_green)
          return
        elif from_name == highlight_to and to_name == highlight_from:
          graph.edge(from_name, to_name, color=light_green, dir='back')
          return
    # if no need to highlight, color gray
    graph.edge(from_name, to_name, arrowhead='none', color=light_gray)

  draw_node(agent)
  for sub_agent in agent.sub_agents:
    await build_graph(graph, sub_agent, highlight_pairs)
    draw_edge(agent.name, sub_agent.name)
  if isinstance(agent, LlmAgent):
    for tool in await agent.canonical_tools():
      draw_node(tool)
      draw_edge(agent.name, get_node_name(tool))


async def get_agent_graph(root_agent, highlights_pairs, image=False):
  print('build graph')
  graph = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'bgcolor': '#333537'})
  await build_graph(graph, root_agent, highlights_pairs)
  if image:
    return graph.pipe(format='png')
  else:
    return graph

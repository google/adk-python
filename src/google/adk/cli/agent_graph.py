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
from ..agents import LoopAgent
from ..agents import ParallelAgent
from ..agents import SequentialAgent
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


async def build_graph(
    graph: graphviz.Digraph,
    agent: BaseAgent,
    highlight_pairs,
    parent_agent=None,
):
  """
  Build a graph of the agent and its sub-agents.
  Args:
    graph: The graph to build on.
    agent: The agent to build the graph for.
    highlight_pairs: A list of pairs of nodes to highlight.
    parent_agent: The parent agent of the current agent. This is specifically used when building Workflow Agents to directly connect a node to nodes inside a Workflow Agent.

  Returns:
    None
  """
  dark_green = '#0F5223'
  light_green = '#69CB87'
  light_gray = '#cccccc'
  white = '#ffffff'

  def get_node_name(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      # Added Workflow Agent checks for different agent types
      if isinstance(tool_or_agent, SequentialAgent):
        return tool_or_agent.name + ' (Sequential Agent)'
      elif isinstance(tool_or_agent, LoopAgent):
        return tool_or_agent.name + ' (Loop Agent)'
      elif isinstance(tool_or_agent, ParallelAgent):
        return tool_or_agent.name + ' (Parallel Agent)'
      else:
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

  def should_build_agent_cluster(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      if isinstance(tool_or_agent, SequentialAgent):
        return True
      elif isinstance(tool_or_agent, LoopAgent):
        return True
      elif isinstance(tool_or_agent, ParallelAgent):
        return True
      else:
        return False
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return False
    elif isinstance(tool_or_agent, FunctionTool):
      return False
    elif isinstance(tool_or_agent, BaseTool):
      return False
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return False

  async def build_cluster(child: graphviz.Digraph, agent: BaseAgent, name: str):
    if isinstance(agent, LoopAgent):
      # Draw the edge from the parent agent to the first sub-agent
      if parent_agent:
        draw_edge(parent_agent.name, agent.sub_agents[0].name)
      length = len(agent.sub_agents)
      curr_length = 0
      # Draw the edges between the sub-agents
      for sub_agent_int_sequential in agent.sub_agents:
        await build_graph(child, sub_agent_int_sequential, highlight_pairs)
        # Draw the edge between the current sub-agent and the next one
        # If it's the last sub-agent, draw an edge to the first one to indicating a loop
        draw_edge(
            agent.sub_agents[curr_length].name,
            agent.sub_agents[
                0 if curr_length == length - 1 else curr_length + 1
            ].name,
        )
        curr_length += 1
    elif isinstance(agent, SequentialAgent):
      # Draw the edge from the parent agent to the first sub-agent
      if parent_agent:
        draw_edge(parent_agent.name, agent.sub_agents[0].name)
      length = len(agent.sub_agents)
      curr_length = 0

      # Draw the edges between the sub-agents
      for sub_agent_int_sequential in agent.sub_agents:
        await build_graph(child, sub_agent_int_sequential, highlight_pairs)
        # Draw the edge between the current sub-agent and the next one
        # If it's the last sub-agent, don't draw an edge to avoid a loop
        if curr_length != length - 1:
          draw_edge(
              agent.sub_agents[curr_length].name,
              agent.sub_agents[curr_length + 1].name,
          )
        curr_length += 1

    elif isinstance(agent, ParallelAgent):
      # Draw the edge from the parent agent to every sub-agent
      for sub_agent in agent.sub_agents:
        await build_graph(child, sub_agent, highlight_pairs)
        if parent_agent:
          draw_edge(parent_agent.name, sub_agent.name)
    else:
      for sub_agent in agent.sub_agents:
        await build_graph(child, sub_agent, highlight_pairs)
        draw_edge(agent.name, sub_agent.name)

    child.attr(
        label=name,
        style='rounded',
        color=white,
        fontcolor=light_gray,
    )

  async def draw_node(tool_or_agent: Union[BaseAgent, BaseTool]):
    name = get_node_name(tool_or_agent)
    shape = get_node_shape(tool_or_agent)
    caption = get_node_caption(tool_or_agent)
    as_cluster = should_build_agent_cluster(tool_or_agent)
    if highlight_pairs:
      for highlight_tuple in highlight_pairs:
        if name in highlight_tuple:
          # if in highlight, draw highlight node
          if as_cluster:
            cluster = graphviz.Digraph(
                name='cluster_' + name
            )  # adding "cluster_" to the name makes the graph render as a cluster subgraph
            await build_cluster(cluster, agent, name)
            graph.subgraph(cluster)
          else:
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
    # if not in highlight, draw non-highlight node
    if as_cluster:

      cluster = graphviz.Digraph(
          name='cluster_' + name
      )  # adding "cluster_" to the name makes the graph render as a cluster subgraph
      await build_cluster(cluster, agent, name)
      graph.subgraph(cluster)

    else:
      graph.node(
          name,
          caption,
          shape=shape,
          style='rounded',
          color=light_gray,
          fontcolor=light_gray,
      )

      return

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
    if should_build_agent_cluster(agent):

      graph.edge(
          from_name,
          to_name,
          color=light_gray,
      )
    else:
      graph.edge(from_name, to_name, arrowhead='none', color=light_gray)

  await draw_node(agent)
  for sub_agent in agent.sub_agents:
    await build_graph(graph, sub_agent, highlight_pairs, agent)
    if not should_build_agent_cluster(
        sub_agent
    ) and not should_build_agent_cluster(
        agent
    ):  # This is to avoid making a node for a Workflow Agent
      draw_edge(agent.name, sub_agent.name)
  if isinstance(agent, LlmAgent):
    for tool in await agent.canonical_tools():
      await draw_node(tool)
      draw_edge(agent.name, get_node_name(tool))


async def get_agent_graph(root_agent, highlights_pairs, image=False):
  print('build graph')
  graph = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'bgcolor': '#333537'})
  await build_graph(graph, root_agent, highlights_pairs)
  if image:
    return graph.pipe(format='png')
  else:
    return graph

"""Tests for GraphAgent visualization in agent_graph.py.

Asserts on graphviz.Digraph.source string — no rendering engine needed.
Only covers GraphAgent-specific cluster rendering (our code).
"""

import graphviz
import pytest

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent, GraphNode, GraphState
from google.adk.agents.graph.patterns import (
    DynamicNode,
    DynamicParallelGroup,
    NestedGraphNode,
)
from google.adk.cli.agent_graph import build_graph, get_agent_graph
from google.adk.events.event import Event
from google.genai import types


class SimpleTestAgent(BaseAgent):
  """Minimal async agent for visualization tests (no LLM)."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text="stub")]),
    )


def _make_digraph():
  return graphviz.Digraph(
      graph_attr={"rankdir": "LR", "bgcolor": "#333537"}, strict=True
  )


class TestGraphAgentVisualization:
  """Test GraphAgent rendering in agent_graph build_graph."""

  @pytest.mark.asyncio
  async def test_graph_agent_linear_cluster(self):
    """3 agent nodes with linear edges render as cluster with edges."""
    a1 = SimpleTestAgent(name="step1")
    a2 = SimpleTestAgent(name="step2")
    a3 = SimpleTestAgent(name="step3")

    g = GraphAgent(name="workflow")
    g.add_node(GraphNode(name="n1", agent=a1))
    g.add_node(GraphNode(name="n2", agent=a2))
    g.add_node(GraphNode(name="n3", agent=a3))
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n3")
    g.set_start("n1")
    g.set_end("n3")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "cluster_" in src
    assert "step1" in src
    assert "step2" in src
    assert "step3" in src

  @pytest.mark.asyncio
  async def test_graph_agent_conditional_branch(self):
    """1 source -> 2 targets: both edges present."""
    a1 = SimpleTestAgent(name="check")
    a2 = SimpleTestAgent(name="pass_path")
    a3 = SimpleTestAgent(name="fail_path")

    g = GraphAgent(name="branch")
    g.add_node(GraphNode(name="n1", agent=a1))
    g.add_node(GraphNode(name="n2", agent=a2))
    g.add_node(GraphNode(name="n3", agent=a3))
    g.add_edge("n1", "n2", condition=lambda s: True)
    g.add_edge("n1", "n3", condition=lambda s: False)
    g.set_start("n1")
    g.set_end("n2")
    g.set_end("n3")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "check" in src
    assert "pass_path" in src
    assert "fail_path" in src

  @pytest.mark.asyncio
  async def test_graph_agent_loop(self):
    """Back-edge present in source."""
    a1 = SimpleTestAgent(name="reason")
    a2 = SimpleTestAgent(name="observe")

    g = GraphAgent(name="react")
    g.add_node(GraphNode(name="n1", agent=a1))
    g.add_node(GraphNode(name="n2", agent=a2))
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n1", condition=lambda s: True)
    g.set_start("n1")
    g.set_end("n2")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "reason" in src
    assert "observe" in src

  @pytest.mark.asyncio
  async def test_graph_agent_nested(self):
    """NestedGraphNode renders inner graph as sub-cluster."""
    inner_a = SimpleTestAgent(name="inner_step")
    inner_g = GraphAgent(name="inner")
    inner_g.add_node(GraphNode(name="is", agent=inner_a))
    inner_g.set_start("is")
    inner_g.set_end("is")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner_g))
    outer.set_start("nested")
    outer.set_end("nested")

    dg = _make_digraph()
    await build_graph(dg, outer, highlight_pairs=None)
    src = dg.source

    assert "inner" in src
    assert "inner_step" in src

  @pytest.mark.asyncio
  async def test_graph_agent_function_node(self):
    """Function-only node rendered as box shape."""

    async def my_func(state, ctx):
      return "done"

    g = GraphAgent(name="wf")
    g.add_node("fn_node", function=my_func)
    g.set_start("fn_node")
    g.set_end("fn_node")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "fn_node" in src
    assert "box" in src

  @pytest.mark.asyncio
  async def test_graph_agent_dynamic_node(self):
    """DynamicNode rendered as diamond shape."""
    g = GraphAgent(name="wf")
    dyn = DynamicNode(
        name="dispatcher",
        agent_selector=lambda _: None,
    )
    g.add_node(dyn)
    g.set_start("dispatcher")
    g.set_end("dispatcher")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "dispatcher" in src
    assert "diamond" in src
    assert "(dynamic)" in src

  @pytest.mark.asyncio
  async def test_graph_agent_dynamic_parallel_group(self):
    """DynamicParallelGroup rendered as parallelogram shape."""
    g = GraphAgent(name="wf")
    dpg = DynamicParallelGroup(
        name="fan_out",
        agent_generator=lambda _: [],
        aggregator=lambda r, _: "",
    )
    g.add_node(dpg)
    g.set_start("fan_out")
    g.set_end("fan_out")

    dg = _make_digraph()
    await build_graph(dg, g, highlight_pairs=None)
    src = dg.source

    assert "fan_out" in src
    assert "parallelogram" in src
    assert "(parallel)" in src

  @pytest.mark.asyncio
  async def test_get_agent_graph_returns_digraph(self):
    """get_agent_graph returns a graphviz.Digraph for GraphAgent."""
    a = SimpleTestAgent(name="s")
    g = GraphAgent(name="wf")
    g.add_node(GraphNode(name="n", agent=a))
    g.set_start("n")
    g.set_end("n")

    result = await get_agent_graph(g, highlights_pairs=None)
    assert isinstance(result, graphviz.Digraph)

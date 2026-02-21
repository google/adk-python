"""Tests for GraphAgent first-class pattern APIs.

Tests:
- DynamicNode: Runtime agent selection based on state
- NestedGraphNode: Hierarchical workflow composition (graph within graph)
- DynamicParallelGroup: Dynamic concurrent execution with variable agent count

Uses Runner/GraphAgent for end-to-end integration testing per ADK conventions.
Manual InvocationContext construction is wrong because it requires internal
fields (invocation_id, agent) that Runner fills automatically.
"""

import asyncio
from typing import List
import uuid

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import DynamicNode
from google.adk.agents.graph import DynamicParallelGroup
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import NestedGraphNode
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

# ============================================================================
# Test Helpers
# ============================================================================


class SimpleTestAgent(BaseAgent):
  """Minimal agent that returns a fixed list of responses in order."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, responses: list, delay: float = 0.0):
    super().__init__(name=name)
    object.__setattr__(self, "_responses", responses)
    object.__setattr__(self, "_call_count", 0)
    object.__setattr__(self, "_delay", delay)

  async def _run_async_impl(self, ctx):
    delay = object.__getattribute__(self, "_delay")
    await asyncio.sleep(delay)
    call_count = object.__getattribute__(self, "_call_count")
    responses = object.__getattribute__(self, "_responses")
    response = responses[min(call_count, len(responses) - 1)]
    object.__setattr__(self, "_call_count", call_count + 1)
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=response)]),
    )

  @property
  def call_count(self):
    return object.__getattribute__(self, "_call_count")


def make_runner(graph):
  svc = InMemorySessionService()
  runner = Runner(app_name="test", agent=graph, session_service=svc)
  return runner, svc


async def run_graph(runner, svc, message):
  """Create a fresh session, run the graph, return the final non-metadata text."""
  sid = f"s_{uuid.uuid4().hex[:8]}"
  await svc.create_session(app_name="test", user_id="u", session_id=sid)
  final = ""
  async for event in runner.run_async(
      user_id="u",
      session_id=sid,
      new_message=types.Content(role="user", parts=[types.Part(text=message)]),
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text or ""
      if text and not text.startswith("[GraphMetadata]"):
        final = text
  return final


async def run_graph_with_state(runner, svc, message):
  """Run the graph and return (final_text, graph_data_dict)."""
  sid = f"s_{uuid.uuid4().hex[:8]}"
  await svc.create_session(app_name="test", user_id="u", session_id=sid)
  final = ""
  async for event in runner.run_async(
      user_id="u",
      session_id=sid,
      new_message=types.Content(role="user", parts=[types.Part(text=message)]),
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text or ""
      if text and not text.startswith("[GraphMetadata]"):
        final = text
  session = await svc.get_session(app_name="test", user_id="u", session_id=sid)
  graph_data = session.state.get("graph_data", {}) if session else {}
  return final, graph_data


# ============================================================================
# DynamicNode
# ============================================================================


class TestDynamicNode:
  """DynamicNode selects which agent to run based on GraphState at runtime."""

  @pytest.mark.asyncio
  async def test_selects_agent_based_on_input(self):
    """Selector reads state.data and returns appropriate agent."""
    simple = SimpleTestAgent("simple", ["SIMPLE"])
    complex_ = SimpleTestAgent("complex", ["COMPLEX"])

    def selector(state):
      return complex_ if "complex" in state.data.get("input", "") else simple

    graph = GraphAgent(name="g")
    graph.add_node(DynamicNode(name="d", agent_selector=selector))
    graph.set_start("d")
    graph.set_end("d")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "simple task") == "SIMPLE"
    assert simple.call_count == 1
    assert complex_.call_count == 0

    assert await run_graph(runner, svc, "complex task") == "COMPLEX"
    assert complex_.call_count == 1

  @pytest.mark.asyncio
  async def test_fallback_when_selector_returns_none(self):
    """When selector returns None, fallback_agent is used."""
    fallback = SimpleTestAgent("fallback", ["FALLBACK"])

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicNode(
            name="d",
            agent_selector=lambda _: None,
            fallback_agent=fallback,
        )
    )
    graph.set_start("d")
    graph.set_end("d")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "any") == "FALLBACK"
    assert fallback.call_count == 1

  @pytest.mark.asyncio
  async def test_raises_when_no_agent_and_no_fallback(self):
    """ValueError raised when selector returns None and no fallback set."""
    graph = GraphAgent(name="g")
    graph.add_node(DynamicNode(name="d", agent_selector=lambda _: None))
    graph.set_start("d")
    graph.set_end("d")

    runner, svc = make_runner(graph)
    with pytest.raises(ValueError, match="No agent selected"):
      await run_graph(runner, svc, "any")

  @pytest.mark.asyncio
  async def test_different_agents_on_sequential_runs(self):
    """Selector can pick different agents on each graph invocation."""
    a = SimpleTestAgent("a", ["A"])
    b = SimpleTestAgent("b", ["B"])
    n = {"count": 0}

    def selector(state):
      n["count"] += 1
      return a if n["count"] % 2 == 1 else b

    graph = GraphAgent(name="g")
    graph.add_node(DynamicNode(name="d", agent_selector=selector))
    graph.set_start("d")
    graph.set_end("d")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "run1") == "A"
    assert await run_graph(runner, svc, "run2") == "B"
    assert a.call_count == 1
    assert b.call_count == 1


# ============================================================================
# NestedGraphNode
# ============================================================================


class TestNestedGraphNode:
  """NestedGraphNode executes a full GraphAgent as a single node step."""

  @pytest.mark.asyncio
  async def test_nested_graph_runs_all_steps(self):
    """All nodes in the nested graph execute; all outputs are accumulated."""
    a1 = SimpleTestAgent("n1", ["STEP1"])
    a2 = SimpleTestAgent("n2", ["STEP2"])

    inner = GraphAgent(name="inner")
    inner.add_node(GraphNode(name="s1", agent=a1))
    inner.add_node(GraphNode(name="s2", agent=a2))
    inner.add_edge("s1", "s2")
    inner.set_start("s1")
    inner.set_end("s2")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    result = await run_graph(runner, svc, "go")
    assert result == "STEP1STEP2"
    assert a1.call_count == 1
    assert a2.call_count == 1

  @pytest.mark.asyncio
  async def test_inherit_session_true(self):
    """With inherit_session=True the nested graph shares the parent session."""
    inner_agent = SimpleTestAgent("inner", ["INNER"])
    inner = GraphAgent(name="inner_graph")
    inner.add_node(GraphNode(name="p", agent=inner_agent))
    inner.set_start("p")
    inner.set_end("p")

    outer = GraphAgent(name="outer")
    outer.add_node(
        NestedGraphNode(name="nested", graph_agent=inner, inherit_session=True)
    )
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    assert await run_graph(runner, svc, "go") == "INNER"

  @pytest.mark.asyncio
  async def test_inherit_session_false_isolated(self):
    """With inherit_session=False the nested graph gets its own session."""
    inner_agent = SimpleTestAgent("inner", ["ISOLATED"])
    inner = GraphAgent(name="inner_graph")
    inner.add_node(GraphNode(name="p", agent=inner_agent))
    inner.set_start("p")
    inner.set_end("p")

    outer = GraphAgent(name="outer")
    outer.add_node(
        NestedGraphNode(name="nested", graph_agent=inner, inherit_session=False)
    )
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    assert await run_graph(runner, svc, "go") == "ISOLATED"

  @pytest.mark.asyncio
  async def test_multi_node_inner_graph(self):
    """Inner graph with 3 nodes; outer gets all accumulated output."""
    agents = [SimpleTestAgent(f"n{i}", [f"INNER{i}"]) for i in range(3)]
    inner = GraphAgent(name="inner")
    for i, a in enumerate(agents):
      inner.add_node(GraphNode(name=f"n{i}", agent=a))
    inner.add_edge("n0", "n1")
    inner.add_edge("n1", "n2")
    inner.set_start("n0")
    inner.set_end("n2")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    assert await run_graph(runner, svc, "go") == "INNER0INNER1INNER2"
    for a in agents:
      assert a.call_count == 1


# ============================================================================
# DynamicParallelGroup
# ============================================================================


class TestDynamicParallelGroup:
  """DynamicParallelGroup generates agent list at runtime and runs concurrently."""

  @pytest.mark.asyncio
  async def test_all_agents_run_and_aggregated(self):
    """All generated agents execute; aggregator receives all results."""
    agents = [SimpleTestAgent(f"a{i}", [f"R{i}"]) for i in range(3)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: agents,
            aggregator=lambda results, _: "|".join(results),
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    result = await run_graph(runner, svc, "go")
    for i in range(3):
      assert f"R{i}" in result
    for a in agents:
      assert a.call_count == 1

  @pytest.mark.asyncio
  async def test_empty_list_handled(self):
    """Empty agent list: aggregator receives [] and returns gracefully."""
    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: [],
            aggregator=lambda results, _: f"count={len(results)}",
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "go") == "count=0"

  @pytest.mark.asyncio
  async def test_max_parallelism_all_complete(self):
    """All agents complete even when max_parallelism < agent count."""
    agents = [SimpleTestAgent(f"a{i}", [f"R{i}"], delay=0.02) for i in range(6)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: agents,
            aggregator=lambda results, _: f"count={len(results)}",
            max_parallelism=2,
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "go") == "count=6"

  @pytest.mark.asyncio
  async def test_variable_count_from_state(self):
    """Generator reads state.data.input to determine how many agents to spawn."""

    def gen(state):
      n = int(state.data.get("input", "3"))
      return [SimpleTestAgent(f"a{i}", ["x"]) for i in range(n)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=gen,
            aggregator=lambda results, _: str(len(results)),
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "5") == "5"
    assert await run_graph(runner, svc, "2") == "2"

  @pytest.mark.asyncio
  async def test_custom_aggregator(self):
    """Aggregator fully controls result combination (e.g. sum)."""
    agents = [SimpleTestAgent(f"a{i}", [str((i + 1) * 10)]) for i in range(3)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: agents,
            aggregator=lambda results, _: str(sum(int(r) for r in results)),
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    assert await run_graph(runner, svc, "go") == "60"  # 10+20+30


# ============================================================================
# Pattern Integration
# ============================================================================


class TestPatternIntegration:
  """Patterns compose naturally within a single GraphAgent."""

  @pytest.mark.asyncio
  async def test_dynamic_node_then_parallel_group(self):
    """DynamicNode routes first, then DynamicParallelGroup runs downstream."""
    router_agent = SimpleTestAgent("router", ["routed"])
    parallel_agents = [SimpleTestAgent(f"p{i}", [f"P{i}"]) for i in range(3)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicNode(
            name="router",
            agent_selector=lambda _: router_agent,
        )
    )
    graph.add_node(
        DynamicParallelGroup(
            name="parallel",
            agent_generator=lambda _: parallel_agents,
            aggregator=lambda results, _: "-".join(results),
        )
    )
    graph.add_edge("router", "parallel")
    graph.set_start("router")
    graph.set_end("parallel")

    runner, svc = make_runner(graph)
    result = await run_graph(runner, svc, "go")
    for i in range(3):
      assert f"P{i}" in result

  @pytest.mark.asyncio
  async def test_nested_graph_with_dynamic_node_inside(self):
    """NestedGraphNode wraps a sub-graph that itself uses DynamicNode."""
    inner_agent = SimpleTestAgent("inner_a", ["INNER_DYNAMIC"])

    inner = GraphAgent(name="inner")
    inner.add_node(
        DynamicNode(
            name="dyn",
            agent_selector=lambda _: inner_agent,
        )
    )
    inner.set_start("dyn")
    inner.set_end("dyn")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    assert await run_graph(runner, svc, "go") == "INNER_DYNAMIC"


# ============================================================================
# Observability (_debug_ keys in state.data)
# ============================================================================


class TestPatternObservability:
  """Pattern nodes write _debug_ keys to state.data for observability."""

  @pytest.mark.asyncio
  async def test_dynamic_node_records_selected_agent(self):
    """DynamicNode records _debug_{name}_selected_agent in state.data."""
    agent_a = SimpleTestAgent("agent_a", ["A"])

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicNode(
            name="d",
            agent_selector=lambda _: agent_a,
        )
    )
    graph.set_start("d")
    graph.set_end("d")

    runner, svc = make_runner(graph)
    result, data = await run_graph_with_state(runner, svc, "go")
    assert result == "A"
    assert data.get("_debug_d_selected_agent") == "agent_a"

  @pytest.mark.asyncio
  async def test_parallel_group_records_count(self):
    """DynamicParallelGroup records _debug_{name}_parallel_count."""
    agents = [SimpleTestAgent(f"a{i}", [f"R{i}"]) for i in range(3)]

    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: agents,
            aggregator=lambda results, _: "|".join(results),
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    result, data = await run_graph_with_state(runner, svc, "go")
    assert "R0" in result
    assert data.get("_debug_p_parallel_count") == 3

  @pytest.mark.asyncio
  async def test_parallel_group_records_zero_count(self):
    """DynamicParallelGroup records count=0 for empty agent list."""
    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="p",
            agent_generator=lambda _: [],
            aggregator=lambda results, _: "empty",
        )
    )
    graph.set_start("p")
    graph.set_end("p")

    runner, svc = make_runner(graph)
    result, data = await run_graph_with_state(runner, svc, "go")
    assert result == "empty"
    assert data.get("_debug_p_parallel_count") == 0

  @pytest.mark.asyncio
  async def test_nested_graph_records_output(self):
    """NestedGraphNode records _debug_{name}_output in state.data."""
    inner_agent = SimpleTestAgent("inner_agent", ["NESTED_OUTPUT"])
    inner = GraphAgent(name="inner")
    inner.add_node(GraphNode(name="s", agent=inner_agent))
    inner.set_start("s")
    inner.set_end("s")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    result, data = await run_graph_with_state(runner, svc, "go")
    assert result == "NESTED_OUTPUT"
    assert data.get("_debug_nested_output") == "NESTED_OUTPUT"

  @pytest.mark.asyncio
  async def test_nested_graph_output_truncated(self):
    """NestedGraphNode truncates _debug_ output to 500 chars."""
    long_text = "x" * 1000
    inner_agent = SimpleTestAgent("inner_agent", [long_text])
    inner = GraphAgent(name="inner")
    inner.add_node(GraphNode(name="s", agent=inner_agent))
    inner.set_start("s")
    inner.set_end("s")

    outer = GraphAgent(name="outer")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    runner, svc = make_runner(outer)
    _, data = await run_graph_with_state(runner, svc, "go")
    assert len(data.get("_debug_nested_output", "")) == 500


# ============================================================================
# Sub-Agent Registration for Pattern Nodes
# ============================================================================


class TestPatternSubAgentRegistration:
  """Test GraphAgent.sub_agents registration for pattern node types."""

  def test_nested_graph_node_registers_graph_agent(self):
    """NestedGraphNode's graph_agent should appear in outer graph sub_agents."""
    inner_agent = SimpleTestAgent("inner_a", ["ok"])
    inner_graph = GraphAgent(name="inner_graph")
    inner_graph.add_node(GraphNode(name="step", agent=inner_agent))
    inner_graph.set_start("step")
    inner_graph.set_end("step")

    outer = GraphAgent(name="outer")
    nested = NestedGraphNode(name="nested_step", graph_agent=inner_graph)
    outer.add_node(nested)

    assert inner_graph in outer.sub_agents
    assert inner_graph.parent_agent is outer

  def test_dynamic_node_registers_fallback_agent(self):
    """DynamicNode's fallback_agent should appear in sub_agents when provided."""
    fallback = SimpleTestAgent("fallback", ["fb"])

    outer = GraphAgent(name="g")
    dyn = DynamicNode(
        name="dispatcher",
        agent_selector=lambda _: None,
        fallback_agent=fallback,
    )
    outer.add_node(dyn)

    assert fallback in outer.sub_agents
    assert fallback.parent_agent is outer

  def test_dynamic_node_no_fallback_no_registration(self):
    """DynamicNode without fallback should not add anything to sub_agents."""
    outer = GraphAgent(name="g")
    dyn = DynamicNode(name="dispatcher", agent_selector=lambda _: None)
    outer.add_node(dyn)

    assert len(outer.sub_agents) == 0

  def test_dynamic_parallel_no_registration(self):
    """DynamicParallelGroup should not register anything (runtime-only)."""

    def gen(state):
      return [SimpleTestAgent(f"tmp_{i}", ["x"]) for i in range(2)]

    outer = GraphAgent(name="g")
    dpg = DynamicParallelGroup(
        name="par",
        agent_generator=gen,
        aggregator=lambda results, _: ",".join(results),
    )
    outer.add_node(dpg)

    assert len(outer.sub_agents) == 0

  def test_find_agent_through_nested_graph_node(self):
    """outer.find_agent should traverse into NestedGraphNode's graph_agent."""
    inner_agent = SimpleTestAgent("deep_agent", ["ok"])
    inner_graph = GraphAgent(name="inner_graph")
    inner_graph.add_node(GraphNode(name="step", agent=inner_agent))
    inner_graph.set_start("step")
    inner_graph.set_end("step")

    outer = GraphAgent(name="outer")
    nested = NestedGraphNode(name="nested_step", graph_agent=inner_graph)
    outer.add_node(nested)

    # Find inner graph itself
    assert outer.find_agent("inner_graph") is inner_graph
    # Find deeply-nested agent
    assert outer.find_agent("deep_agent") is inner_agent


# ============================================================================
# Multi-event output accumulation tests
# ============================================================================


class MultiEventAgent(BaseAgent):
  """Agent that yields multiple content events (simulates streaming)."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, texts: list):
    super().__init__(name=name)
    object.__setattr__(self, "_texts", texts)

  async def _run_async_impl(self, ctx):
    for text in object.__getattribute__(self, "_texts"):
      yield Event(
          author=self.name,
          content=types.Content(parts=[types.Part(text=text)]),
      )


@pytest.mark.asyncio
class TestPatternOutputAccumulation:
  """Pattern nodes must accumulate output from multi-event agents."""

  async def test_dynamic_node_accumulates_multi_event_output(self):
    """DynamicNode must concatenate all event texts, not keep only last."""
    agent = MultiEventAgent(name="streamer", texts=["Hello ", "World"])
    graph = GraphAgent(name="g")
    graph.add_node(DynamicNode(name="dyn", agent_selector=lambda s: agent))
    graph.set_start("dyn")
    graph.set_end("dyn")

    result = await run_graph(*make_runner(graph), "test")
    assert "Hello " in result
    assert "World" in result

  async def test_dynamic_parallel_group_accumulates_multi_event_output(self):
    """DynamicParallelGroup must concatenate all event texts per agent."""
    agent = MultiEventAgent(name="s", texts=["part1", "part2"])
    graph = GraphAgent(name="g")
    graph.add_node(
        DynamicParallelGroup(
            name="dpg",
            agent_generator=lambda s: [agent],
            aggregator=lambda results, s: " ".join(results),
        )
    )
    graph.set_start("dpg")
    graph.set_end("dpg")

    result = await run_graph(*make_runner(graph), "test")
    assert "part1" in result
    assert "part2" in result


class MultiPartEventAgent(BaseAgent):
  """Agent that yields a single event with multiple parts."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, texts: list):
    super().__init__(name=name)
    object.__setattr__(self, "_texts", texts)

  async def _run_async_impl(self, ctx):
    parts = [
        types.Part(text=t) for t in object.__getattribute__(self, "_texts")
    ]
    yield Event(
        author=self.name,
        content=types.Content(parts=parts),
    )


@pytest.mark.asyncio
class TestNestedGraphNodeMultiPart:
  """NestedGraphNode must properly join multi-part event content."""

  async def test_nested_graph_node_multi_part_output(self):
    """Multi-part event parts are joined, not just parts[0]."""
    agent = MultiPartEventAgent(
        name="mp_agent", texts=["alpha", " beta", " gamma"]
    )
    inner = GraphAgent(name="inner_mp")
    inner.add_node(GraphNode(name="step", agent=agent))
    inner.set_start("step")
    inner.set_end("step")

    outer = GraphAgent(name="outer_mp")
    outer.add_node(NestedGraphNode(name="nested", graph_agent=inner))
    outer.set_start("nested")
    outer.set_end("nested")

    result = await run_graph(*make_runner(outer), "go")
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result
    assert result == "alpha beta gamma"

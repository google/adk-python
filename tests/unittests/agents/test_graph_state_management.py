"""Comprehensive state management tests for GraphAgent.

Tests all state reducers and state propagation patterns:
- StateReducer.OVERWRITE
- StateReducer.APPEND
- StateReducer.SUM
- StateReducer.CUSTOM
- State propagation through graph
- State isolation in parallel execution

These are unit tests focusing on state reducer logic, not full integration tests.
Full integration tests are in test_graph_agent.py and test_parallel_execution.py.
"""

from typing import Any
from typing import Dict

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import GraphNode
from google.adk.agents.graph import GraphState
from google.adk.agents.graph import StateReducer
from google.adk.events.event import Event
from google.genai import types
import pytest

# ============================================================================
# Test Agents (Real BaseAgent implementations per ADK guidelines)
# ============================================================================


class TextAgent(BaseAgent):
  """Agent that outputs text."""

  model_config = {"extra": "allow", "arbitrary_types_allowed": True}

  def __init__(self, name: str, text: str):
    super().__init__(name=name)
    object.__setattr__(self, "_text", text)

  async def _run_async_impl(self, ctx):
    """Output text."""
    text = object.__getattribute__(self, "_text")
    yield Event(
        author=self.name, content=types.Content(parts=[types.Part(text=text)])
    )


# ============================================================================
# Test: StateReducer.OVERWRITE
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerOverwrite:
  """Test OVERWRITE reducer - replaces existing value."""

  async def test_overwrite_reducer_basic(self):
    """Test basic OVERWRITE behavior - new value replaces old."""
    node = GraphNode(
        name="test_node",
        agent=TextAgent("agent", "new"),
        reducer=StateReducer.OVERWRITE,
    )

    # Initial state with existing value
    state = GraphState(data={"test_node": "old"})

    # Apply output with OVERWRITE reducer
    new_state = node._default_output_mapper("new", state)

    # Verify value was overwritten
    assert new_state.data["test_node"] == "new"
    assert "old" not in str(new_state.data["test_node"])

  async def test_overwrite_reducer_new_key(self):
    """Test OVERWRITE creates key if it doesn't exist."""
    node = GraphNode(
        name="new_key",
        agent=TextAgent("agent", "value"),
        reducer=StateReducer.OVERWRITE,
    )

    state = GraphState(data={})
    new_state = node._default_output_mapper("value", state)

    assert new_state.data["new_key"] == "value"

  async def test_overwrite_preserves_other_keys(self):
    """Test OVERWRITE doesn't affect other state keys."""
    node = GraphNode(
        name="key1",
        agent=TextAgent("agent", "new"),
        reducer=StateReducer.OVERWRITE,
    )

    state = GraphState(data={"key1": "old", "key2": "preserved"})
    new_state = node._default_output_mapper("new", state)

    assert new_state.data["key1"] == "new"
    assert new_state.data["key2"] == "preserved"


# ============================================================================
# Test: StateReducer.APPEND
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerAppend:
  """Test APPEND reducer - appends to list."""

  async def test_append_reducer_creates_list(self):
    """Test APPEND reducer creates list when key doesn't exist."""
    node = GraphNode(
        name="collector",
        agent=TextAgent("agent", "item"),
        reducer=StateReducer.APPEND,
    )

    state = GraphState(data={})
    new_state = node._default_output_mapper("first_item", state)

    # Verify list was created with first item
    assert "collector" in new_state.data
    assert isinstance(new_state.data["collector"], list)
    assert new_state.data["collector"] == ["first_item"]

  async def test_append_reducer_appends_to_existing_list(self):
    """Test APPEND adds to existing list."""
    node = GraphNode(
        name="collector",
        agent=TextAgent("agent", "item"),
        reducer=StateReducer.APPEND,
    )

    state = GraphState(data={"collector": ["item1", "item2"]})
    new_state = node._default_output_mapper("item3", state)

    assert new_state.data["collector"] == ["item1", "item2", "item3"]

  async def test_append_multiple_values(self):
    """Test APPEND accumulates multiple values."""
    node = GraphNode(
        name="results",
        agent=TextAgent("agent", "item"),
        reducer=StateReducer.APPEND,
    )

    # First append
    state1 = GraphState(data={})
    state2 = node._default_output_mapper("first", state1)

    # Second append
    state3 = node._default_output_mapper("second", state2)

    # Third append
    state4 = node._default_output_mapper("third", state3)

    assert state4.data["results"] == ["first", "second", "third"]


# ============================================================================
# Test: StateReducer.SUM
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerSum:
  """Test SUM reducer - accumulates values via + operator."""

  async def test_sum_reducer_string_concatenation(self):
    """Test SUM reducer concatenates string outputs."""
    node = GraphNode(
        name="log", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    state = GraphState(data={})
    state1 = node._default_output_mapper("hello", state)
    assert state1.data["log"] == "hello"

    state2 = node._default_output_mapper(" world", state1)
    assert state2.data["log"] == "hello world"

  async def test_sum_reducer_with_existing_string(self):
    """Test SUM concatenates onto existing string value."""
    node = GraphNode(
        name="counter", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    state = GraphState(data={"counter": "prefix"})
    state1 = node._default_output_mapper("_suffix", state)
    assert state1.data["counter"] == "prefix_suffix"

  async def test_sum_reducer_numeric_types(self):
    """Test SUM works correctly with int and float values."""
    node = GraphNode(
        name="total", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    state = GraphState(data={})
    # int + int
    state1 = node._default_output_mapper(10, state)
    assert state1.data["total"] == 10

    # int + float
    state2 = node._default_output_mapper(2.5, state1)
    assert state2.data["total"] == 12.5

    # float + int
    state3 = node._default_output_mapper(3, state2)
    assert state3.data["total"] == 15.5

  async def test_sum_reducer_type_mismatch(self):
    """Test SUM raises TypeError for incompatible types."""
    node = GraphNode(
        name="counter", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    # string existing + int output → TypeError
    state = GraphState(data={"counter": "not_a_number"})
    with pytest.raises(TypeError, match="cannot add"):
      node._default_output_mapper(5, state)

  async def test_sum_reducer_list_concatenation(self):
    """Test SUM concatenates list outputs."""
    node = GraphNode(
        name="items", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    state = GraphState(data={})
    state1 = node._default_output_mapper([1, 2], state)
    assert state1.data["items"] == [1, 2]

    state2 = node._default_output_mapper([3, 4], state1)
    assert state2.data["items"] == [1, 2, 3, 4]

  async def test_sum_reducer_agent_string_output(self):
    """Test SUM works natively with agent string outputs (no custom mapper needed)."""
    node = GraphNode(
        name="transcript", agent=TextAgent("agent", "x"), reducer=StateReducer.SUM
    )

    state = GraphState(data={})
    state1 = node._default_output_mapper("Agent says: hello. ", state)
    state2 = node._default_output_mapper("Agent says: goodbye. ", state1)
    assert state2.data["transcript"] == "Agent says: hello. Agent says: goodbye. "


# ============================================================================
# Test: StateReducer.CUSTOM
# ============================================================================


@pytest.mark.asyncio
class TestStateReducerCustom:
  """Test CUSTOM reducer - uses custom reduction function."""

  async def test_custom_reducer_basic(self):
    """Test CUSTOM reducer with simple concatenation."""

    def concat_reducer(existing, new_value):
      if existing is None:
        return new_value
      return f"{existing}|{new_value}"

    node = GraphNode(
        name="custom",
        agent=TextAgent("agent", "test"),
        reducer=StateReducer.CUSTOM,
        custom_reducer=concat_reducer,
    )

    # First call - no existing value
    state1 = GraphState(data={})
    new_state1 = node._default_output_mapper("A", state1)
    assert new_state1.data["custom"] == "A"

    # Second call - merge with existing
    state2 = GraphState(data={"custom": "A"})
    new_state2 = node._default_output_mapper("B", state2)
    assert new_state2.data["custom"] == "A|B"

  async def test_custom_reducer_dict_merge(self):
    """Test CUSTOM reducer for merging dictionaries."""

    def dict_merge_reducer(existing, new_value):
      """Merge dict-like string representations."""
      if existing is None:
        return {"data": [new_value]}
      if isinstance(existing, dict):
        existing["data"].append(new_value)
        return existing
      return {"data": [existing, new_value]}

    node = GraphNode(
        name="merger",
        agent=TextAgent("agent", "test"),
        reducer=StateReducer.CUSTOM,
        custom_reducer=dict_merge_reducer,
    )

    state1 = GraphState(data={})
    new_state1 = node._default_output_mapper("item1", state1)
    assert new_state1.data["merger"] == {"data": ["item1"]}

    new_state2 = node._default_output_mapper("item2", new_state1)
    assert new_state2.data["merger"] == {"data": ["item1", "item2"]}

  async def test_custom_reducer_counter(self):
    """Test CUSTOM reducer for counting."""

    def count_reducer(existing, new_value):
      """Count occurrences."""
      if existing is None:
        return 1
      return existing + 1

    node = GraphNode(
        name="counter",
        agent=TextAgent("agent", "test"),
        reducer=StateReducer.CUSTOM,
        custom_reducer=count_reducer,
    )

    state1 = GraphState(data={})
    new_state1 = node._default_output_mapper("ignored", state1)
    assert new_state1.data["counter"] == 1

    new_state2 = node._default_output_mapper("ignored", new_state1)
    assert new_state2.data["counter"] == 2

    new_state3 = node._default_output_mapper("ignored", new_state2)
    assert new_state3.data["counter"] == 3


# ============================================================================
# Test: State Propagation (Unit Tests)
# ============================================================================


@pytest.mark.asyncio
class TestStatePropagation:
  """Test how state flows through graph nodes (unit tests)."""

  async def test_output_mapper_preserves_existing_state(self):
    """Test that output mapper preserves existing state data."""
    node = GraphNode(name="node1", agent=TextAgent("agent", "new"))

    state = GraphState(data={"existing_key": "existing_value", "meta": "data"})

    new_state = node._default_output_mapper("new_output", state)

    # New output added
    assert new_state.data["node1"] == "new_output"

    # Existing state preserved
    assert new_state.data["existing_key"] == "existing_value"
    assert new_state.data["meta"] == "data"

  async def test_domain_data_preserved_across_state_updates(self):
    """Test that domain data is preserved during state updates."""
    node = GraphNode(name="node", agent=TextAgent("agent", "output"))

    state = GraphState(
        data={"key": "value", "iteration": 1, "path": ["start", "middle"]},
    )

    new_state = node._default_output_mapper("output", state)

    assert new_state.data["iteration"] == 1
    assert new_state.data["path"] == ["start", "middle"]

  async def test_state_isolation_between_nodes(self):
    """Test that each node gets its own state copy."""
    node1 = GraphNode(name="node1", agent=TextAgent("agent1", "output1"))
    node2 = GraphNode(name="node2", agent=TextAgent("agent2", "output2"))

    state = GraphState(data={})

    # Node 1 processes state
    state1 = node1._default_output_mapper("output1", state)

    # Node 2 processes original state (not state1)
    state2 = node2._default_output_mapper("output2", state)

    # Verify isolation - state2 doesn't have node1's output
    assert "node1" in state1.data
    assert "node1" not in state2.data
    assert "node2" in state2.data


# ============================================================================
# Test: Custom Output Mappers
# ============================================================================


@pytest.mark.asyncio
class TestCustomOutputMappers:
  """Test custom output mapper functionality."""

  async def test_custom_output_mapper_override(self):
    """Test custom output mapper completely overrides default."""

    def custom_mapper(output: str, state: GraphState) -> GraphState:
      # Completely custom logic
      new_state = GraphState(
          data={"custom_key": f"CUSTOM_{output}", "custom": True}
      )
      return new_state

    node = GraphNode(
        name="custom",
        agent=TextAgent("agent", "test"),
        output_mapper=custom_mapper,
    )

    state = GraphState(data={"existing": "data"})
    new_state = node.output_mapper("output", state)

    # Custom mapper replaced everything
    assert "custom_key" in new_state.data
    assert new_state.data["custom_key"] == "CUSTOM_output"
    assert new_state.data.get("custom") == True
    # Original state data gone (custom mapper replaced it)
    assert "existing" not in new_state.data

  async def test_custom_output_mapper_with_state_merge(self):
    """Test custom output mapper that merges with existing state."""

    def merging_mapper(output: str, state: GraphState) -> GraphState:
      # Preserve existing state and add new data
      new_state = GraphState(data=state.data.copy())
      new_state.data["processed"] = output.upper()
      new_state.data["processed_count"] = (
          new_state.data.get("processed_count", 0) + 1
      )
      return new_state

    node = GraphNode(
        name="merger",
        agent=TextAgent("agent", "test"),
        output_mapper=merging_mapper,
    )

    state = GraphState(data={"existing": "value", "processed_count": 5})
    new_state = node.output_mapper("hello", state)

    # Existing state preserved
    assert new_state.data["existing"] == "value"
    # New data added
    assert new_state.data["processed"] == "HELLO"
    # Processed count updated in data
    assert new_state.data["processed_count"] == 6


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.asyncio
class TestStateEdgeCases:
  """Test edge cases in state management."""

  async def test_empty_state_initialization(self):
    """Test graph node with empty initial state."""
    node = GraphNode(name="solo", agent=TextAgent("agent", "output"))

    state = GraphState(data={})
    new_state = node._default_output_mapper("output", state)

    assert new_state.data["solo"] == "output"

  async def test_state_copy_safety(self):
    """Test that state copies don't share references for simple types."""
    state1 = GraphState(data={"key": "value", "meta": "data"})

    # GraphNode does .copy() for data
    state2 = GraphState(data=state1.data.copy())

    # Modify state2
    state2.data["key"] = "modified"
    state2.data["meta"] = "modified"

    # State1 unchanged (shallow copy works for simple types)
    assert state1.data["key"] == "value"
    assert state1.data["meta"] == "data"

  async def test_state_nested_dict_deep_copy_isolation(self):
    """Verify _default_output_mapper uses deepcopy for nested state isolation."""
    state = GraphState(data={"nested": {"key": "value"}, "list": [1, 2, 3]})

    node = GraphNode(name="test_node", agent=TextAgent("agent", "output"))
    new_state = node._default_output_mapper("output", state)

    # Modify nested structure in new_state
    new_state.data["nested"]["key"] = "modified"
    new_state.data["list"].append(4)

    # Original state is NOT affected (deepcopy ensures isolation)
    assert state.data["nested"]["key"] == "value"
    assert state.data["list"] == [1, 2, 3]

  async def test_reducer_with_none_output(self):
    """Test reducer behavior with None or empty output."""
    node = GraphNode(
        name="test",
        agent=TextAgent("agent", ""),
        reducer=StateReducer.OVERWRITE,
    )

    state = GraphState(data={})
    new_state = node._default_output_mapper("", state)

    # Empty string is still stored
    assert new_state.data["test"] == ""


# ============================================================================
# GraphState.data_to_json and PydanticJSONEncoder tests
# ============================================================================


def test_data_to_json_simple_values():
  """data_to_json serializes plain dict values to JSON string."""
  import json

  state = GraphState(data={"key": "value", "num": 42})
  result = state.data_to_json()

  parsed = json.loads(result)
  assert parsed["key"] == "value"
  assert parsed["num"] == 42


def test_data_to_json_pydantic_model():
  """data_to_json converts Pydantic BaseModel values via model_dump."""
  import json

  from pydantic import BaseModel

  class Inner(BaseModel):
    x: int
    y: str

  state = GraphState(data={"model": Inner(x=1, y="hello")})
  result = state.data_to_json()

  parsed = json.loads(result)
  assert parsed["model"] == {"x": 1, "y": "hello"}


def test_data_to_json_non_serializable_raises():
  """data_to_json raises TypeError for non-JSON-serializable, non-Pydantic objects."""
  import json

  class Unserializable:
    pass

  state = GraphState(data={"bad": Unserializable()})
  with pytest.raises(TypeError):
    state.data_to_json()

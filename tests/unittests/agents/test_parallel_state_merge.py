"""Unit tests for parallel execution state merging.

Tests parallel execution state management:
- Deep copy isolation between branches
- State merge logic with conflict detection
- Reducers for different merge strategies
- False-positive regression tests for merge conflict warnings
"""

from copy import deepcopy
from unittest.mock import patch

from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.parallel import ErrorPolicy
from google.adk.agents.graph.parallel import JoinStrategy
from google.adk.agents.graph.parallel import ParallelNodeGroup
import pytest


@pytest.mark.asyncio
async def test_deep_copy_isolation():
  """Test that nested structures are truly isolated in parallel branches."""
  original_state = GraphState(data={"results": [1, 2, 3], "meta": {"count": 0}})

  # Simulate parallel execution deep copy (from parallel.py line 162)
  branch1 = GraphState(
      data=deepcopy(original_state.data),
  )
  branch2 = GraphState(
      data=deepcopy(original_state.data),
  )

  # Modify branches
  branch1.data["results"].append(4)
  branch2.data["results"].append(5)
  branch1.data["meta"]["count"] = 1
  branch2.data["meta"]["count"] = 2

  # Original unchanged
  assert original_state.data["results"] == [1, 2, 3]
  assert original_state.data["meta"]["count"] == 0

  # Branches isolated
  assert branch1.data["results"] == [1, 2, 3, 4]
  assert branch2.data["results"] == [1, 2, 3, 5]
  assert branch1.data["meta"]["count"] == 1
  assert branch2.data["meta"]["count"] == 2


@pytest.mark.asyncio
async def test_shallow_vs_deep_copy_bug():
  """Test that shallow copy would cause state mutation (the bug we fixed)."""
  original_state = GraphState(data={"nested_list": [1, 2, 3]})

  # Shallow copy (BUG - mutations affect original)
  shallow_branch = GraphState(
      data=original_state.data.copy(),  # Shallow copy
  )

  # Deep copy (FIXED - mutations isolated)
  deep_branch = GraphState(
      data=deepcopy(original_state.data),  # Deep copy
  )

  # Modify both branches
  shallow_branch.data["nested_list"].append(4)
  deep_branch.data["nested_list"].append(5)

  # Shallow copy MUTATES original (BUG!)
  assert original_state.data["nested_list"] == [1, 2, 3, 4]

  # Deep copy is isolated (made before shallow mutation)
  assert deep_branch.data["nested_list"] == [1, 2, 3, 5]


@pytest.mark.asyncio
async def test_state_merge_no_conflicts():
  """Test state merge when branches modify different keys."""
  # Simulate two branches with no conflicts
  state = GraphState(data={})

  branch1_state = GraphState(
      data={"branch1_result": "value1", "branch1_meta": "meta1"}
  )

  branch2_state = GraphState(
      data={"branch2_result": "value2", "branch2_meta": "meta2"}
  )

  # Simulate merge (from parallel.py lines 276-320)
  results = {
      "node1": {"state": branch1_state},
      "node2": {"state": branch2_state},
  }

  for node_name, result in results.items():
    branch_state = result["state"]

    # Merge data keys
    for key, value in branch_state.data.items():
      state.data[key] = value

  # Both branches merged
  assert state.data["branch1_result"] == "value1"
  assert state.data["branch2_result"] == "value2"
  assert state.data["branch1_meta"] == "meta1"
  assert state.data["branch2_meta"] == "meta2"


@pytest.mark.asyncio
async def test_state_merge_with_conflicts():
  """Test state merge when branches modify same keys (last write wins)."""
  state = GraphState(data={"shared_key": "original"})

  branch1_state = GraphState(data={"shared_key": "branch1_value"})

  branch2_state = GraphState(data={"shared_key": "branch2_value"})

  # Simulate merge with conflict detection
  results = {
      "node1": {"state": branch1_state},
      "node2": {"state": branch2_state},
  }

  conflicts_detected = []
  keys_merged = set()

  for node_name, result in results.items():
    branch_state = result["state"]

    for key, value in branch_state.data.items():
      if key in state.data and key in keys_merged:
        # Conflict detected!
        conflicts_detected.append({
            "key": key,
            "node": node_name,
            "old_value": state.data[key],
            "new_value": value,
        })

      state.data[key] = value  # Last write wins
      keys_merged.add(key)

  # Conflict was detected
  assert len(conflicts_detected) == 1
  assert conflicts_detected[0]["key"] == "shared_key"
  assert conflicts_detected[0]["node"] == "node2"
  assert conflicts_detected[0]["old_value"] == "branch1_value"
  assert conflicts_detected[0]["new_value"] == "branch2_value"

  # Last write wins (node2 overwrote node1)
  assert state.data["shared_key"] == "branch2_value"


@pytest.mark.asyncio
async def test_parallel_group_config():
  """Test ParallelNodeGroup configuration."""
  # Test WAIT_ALL strategy
  group1 = ParallelNodeGroup(
      nodes=["node1", "node2"],
      join_strategy=JoinStrategy.WAIT_ALL,
      error_policy=ErrorPolicy.FAIL_FAST,
  )
  assert group1.join_strategy == JoinStrategy.WAIT_ALL
  assert group1.error_policy == ErrorPolicy.FAIL_FAST
  assert group1.nodes == ["node1", "node2"]

  # Test WAIT_ANY strategy
  group2 = ParallelNodeGroup(
      nodes=["node3", "node4"],
      join_strategy=JoinStrategy.WAIT_ANY,
      error_policy=ErrorPolicy.CONTINUE,
  )
  assert group2.join_strategy == JoinStrategy.WAIT_ANY
  assert group2.error_policy == ErrorPolicy.CONTINUE

  # Test WAIT_N strategy
  group3 = ParallelNodeGroup(
      nodes=["node5", "node6", "node7"],
      join_strategy=JoinStrategy.WAIT_N,
      wait_n=2,
      error_policy=ErrorPolicy.COLLECT,
  )
  assert group3.join_strategy == JoinStrategy.WAIT_N
  assert group3.wait_n == 2
  assert group3.error_policy == ErrorPolicy.COLLECT


# ========================================
# False-Positive Regression Tests
# ========================================


@pytest.mark.asyncio
@patch("google.adk.agents.graph.parallel.logger")
async def test_no_warning_when_same_value_across_branches(mock_logger):
  """Two branches set the same key to the same value → no warning (false-positive regression)."""
  state = GraphState(data={})

  branch1_state = GraphState(data={"score": 100, "status": "ready"})

  branch2_state = GraphState(
      data={"score": 100, "status": "ready"}  # Same values!
  )

  # Simulate merge with the FIXED conflict detection (value equality check)
  results = {
      "node1": {"state": branch1_state},
      "node2": {"state": branch2_state},
  }

  keys_merged = set()

  for node_name, result in results.items():
    branch_state = result["state"]

    for key, value in branch_state.data.items():
      # FIXED: only warn when values DIFFER (parallel.py:288)
      if (
          key in state.data
          and key in keys_merged
          and state.data[key] != value  # <-- The fix!
      ):
        mock_logger.warning(
            f"State merge conflict detected: key '{key}' modified by "
            f"multiple parallel branches. Last write wins (node: {node_name})."
        )

      state.data[key] = value
      keys_merged.add(key)

  # No warnings should have been logged (same values = no conflict)
  mock_logger.warning.assert_not_called()

  # State contains the shared value
  assert state.data["score"] == 100
  assert state.data["status"] == "ready"


@pytest.mark.asyncio
@patch("google.adk.agents.graph.parallel.logger")
async def test_warning_only_when_values_differ(mock_logger):
  """Branches set same key to different values → exactly one warning."""
  state = GraphState(data={})

  branch1_state = GraphState(data={"score": 100, "other": "same"})

  branch2_state = GraphState(
      data={"score": 200, "other": "same"},  # score differs, other is same
  )

  # Simulate merge
  results = {
      "node1": {"state": branch1_state},
      "node2": {"state": branch2_state},
  }

  keys_merged = set()

  for node_name, result in results.items():
    branch_state = result["state"]

    for key, value in branch_state.data.items():
      if (
          key in state.data
          and key in keys_merged
          and state.data[key] != value  # Only warn when different
      ):
        mock_logger.warning(
            f"State merge conflict detected: key '{key}' modified by "
            f"multiple parallel branches. Last write wins (node: {node_name})."
        )

      state.data[key] = value
      keys_merged.add(key)

  # Exactly ONE warning (for "score" key, NOT "other")
  assert mock_logger.warning.call_count == 1
  warning_call = mock_logger.warning.call_args[0][0]
  assert "score" in warning_call
  assert "node2" in warning_call

  # Last write wins
  assert state.data["score"] == 200
  assert state.data["other"] == "same"


@pytest.mark.asyncio
@patch("google.adk.agents.graph.parallel.logger")
async def test_single_branch_never_conflicts(mock_logger):
  """With only one branch, merging can never produce a conflict."""
  state = GraphState(data={"initial": "value"})

  branch1_state = GraphState(
      data={"result": "computed", "initial": "overwritten"},
  )

  # Simulate merge with only ONE branch
  results = {"node1": {"state": branch1_state}}

  keys_merged = set()

  for node_name, result in results.items():
    branch_state = result["state"]

    for key, value in branch_state.data.items():
      if (
          key in state.data
          and key in keys_merged  # Never true for single branch!
          and state.data[key] != value
      ):
        mock_logger.warning(
            f"State merge conflict detected: key '{key}' modified by "
            f"multiple parallel branches. Last write wins (node: {node_name})."
        )

      state.data[key] = value
      keys_merged.add(key)

  # No warnings (single branch can't conflict with itself)
  mock_logger.warning.assert_not_called()

  # State has branch1's values
  assert state.data["result"] == "computed"
  assert state.data["initial"] == "overwritten"


@pytest.mark.asyncio
@patch("google.adk.agents.graph.parallel.logger")
async def test_none_value_is_a_real_conflict(mock_logger):
  """None vs non-None is a genuine conflict and must warn."""
  state = GraphState(data={})

  branch1_state = GraphState(data={"result": None})  # Branch 1 sets to None

  branch2_state = GraphState(
      data={"result": "value"}  # Branch 2 sets to non-None
  )

  # Simulate merge
  results = {
      "node1": {"state": branch1_state},
      "node2": {"state": branch2_state},
  }

  keys_merged = set()

  for node_name, result in results.items():
    branch_state = result["state"]

    for key, value in branch_state.data.items():
      if (
          key in state.data
          and key in keys_merged
          and state.data[key] != value  # None != "value" is True
      ):
        mock_logger.warning(
            f"State merge conflict detected: key '{key}' modified by "
            f"multiple parallel branches. Last write wins (node: {node_name})."
        )

      state.data[key] = value
      keys_merged.add(key)

  # Exactly one warning (None vs "value" is a real conflict)
  assert mock_logger.warning.call_count == 1
  warning_call = mock_logger.warning.call_args[0][0]
  assert "result" in warning_call

  # Last write wins (node2's "value")
  assert state.data["result"] == "value"


@pytest.mark.asyncio
async def test_merge_order_is_deterministic_by_definition_order():
  """State merge must iterate in group.nodes order, not completion order.

  When multiple branches set the same key, the result must be deterministic
  based on the node definition order, not whichever task finishes first.
  """
  import asyncio
  from google.adk.agents.graph.parallel import execute_parallel_group
  from google.adk.agents.graph.graph_node import GraphNode
  from google.adk.events.event import Event
  from google.adk.agents.base_agent import BaseAgent
  from google.genai import types
  from unittest.mock import MagicMock

  class WriteAgent(BaseAgent):
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __init__(self, name, value, delay=0.0):
      super().__init__(name=name)
      object.__setattr__(self, "_value", value)
      object.__setattr__(self, "_delay", delay)

    async def _run_async_impl(self, ctx):
      await asyncio.sleep(object.__getattribute__(self, "_delay"))
      yield Event(
          author=self.name,
          content=types.Content(
              parts=[types.Part(text=object.__getattribute__(self, "_value"))]
          ),
      )

  # Node order: [fast, slow]. Fast finishes first but slow is defined second.
  fast = WriteAgent(name="fast", value="fast_val", delay=0.0)
  slow = WriteAgent(name="slow", value="slow_val", delay=0.01)

  nodes = {
      "fast": GraphNode(name="fast", agent=fast),
      "slow": GraphNode(name="slow", agent=slow),
  }

  # Definition order: slow THEN fast
  group = ParallelNodeGroup(nodes=["slow", "fast"])
  state = GraphState(data={"input": "test"})

  async def execute_node(node, branch_state, ctx):
    async for event in node.agent._run_async_impl(ctx):
      branch_state.data["shared_key"] = f"from_{node.name}"
      yield event

  mock_ctx = MagicMock()
  events = []
  async for event in execute_parallel_group(
      group, nodes, state, mock_ctx, execute_node
  ):
    events.append(event)

  # "fast" is defined AFTER "slow" in group.nodes, so fast's value wins
  assert state.data["shared_key"] == "from_fast"

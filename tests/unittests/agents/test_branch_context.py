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

"""Tests for structural branch tracking.

The structural approach tracks fork points, not accumulated tokens.
This provides O(D) memory complexity (D = max fork depth) vs O(N) for token-set.
"""

from __future__ import annotations

from google.adk.agents.branch import Branch
from google.adk.agents.branch import BranchTokenFactory
import pytest


class TestBranchTokenFactory:
  """Tests for the BranchTokenFactory class."""

  def test_new_token_increments(self):
    """Test that new_token generates unique incrementing tokens."""
    BranchTokenFactory.reset()

    token1 = BranchTokenFactory.new_token()
    token2 = BranchTokenFactory.new_token()
    token3 = BranchTokenFactory.new_token()

    assert token1 < token2 < token3
    assert token2 == token1 + 1
    assert token3 == token2 + 1

  def test_new_token_thread_safe(self):
    """Test that token generation is thread-safe."""
    import threading

    BranchTokenFactory.reset()
    tokens = []

    def generate_tokens():
      for _ in range(100):
        tokens.append(BranchTokenFactory.new_token())

    threads = [threading.Thread(target=generate_tokens) for _ in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    # All tokens should be unique
    assert len(tokens) == len(set(tokens))
    assert len(tokens) == 1000


class TestBranchBasics:
  """Tests for basic Branch functionality."""

  def test_initialization_default(self):
    """Test that default initialization creates root context."""
    ctx = Branch()
    assert ctx.active_forks == {}

  def test_initialization_with_active_forks(self):
    """Test initialization with specific active_forks."""
    ctx = Branch(active_forks={1: (frozenset({0}), 3)})
    assert ctx.active_forks == {1: (frozenset({0}), 3)}

  def test_fork_creates_correct_number_of_children(self):
    """Test that fork(N) creates exactly N children."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)

    assert len(children) == 3
    assert all(isinstance(c, Branch) for c in children)

  def test_fork_children_have_distinct_indices(self):
    """Test that forked children have distinct indices at the fork point."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)

    # All children should have the same fork point ID
    fork_point_ids = list(children[0].active_forks.keys())
    assert len(fork_point_ids) == 1  # One fork point
    fork_id = fork_point_ids[0]
    
    # All children share same fork_id
    for c in children:
      assert fork_id in c.active_forks

    # But distinct branch indices (stored in the frozenset, first element of tuple)
    indices = [list(c.active_forks[fork_id][0])[0] for c in children]
    assert sorted(indices) == [0, 1, 2]
    
    # All should have the same total branch count (second element of tuple)
    totals = [c.active_forks[fork_id][1] for c in children]
    assert totals == [3, 3, 3]

  def test_fork_children_share_parent_active_forks(self):
    """Test that children inherit parent's active forks."""
    BranchTokenFactory.reset()
    
    # Create a nested scenario
    root = Branch()
    level1 = root.fork(2)  # First fork
    level2 = level1[0].fork(2)  # Second fork from first child
    
    # Level2 children should have 2 active forks: the one from level1 and the new one
    for child in level2:
      assert len(child.active_forks) == 2

  def test_join_merges_sibling_branches(self):
    """Test that join merges all sibling indices at the fork point."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)
    
    joined = parent.join(children)
    
    # After join, the fork point should be removed (all siblings merged)
    # The joined branch should look like the parent
    assert joined.active_forks == parent.active_forks

  def test_join_partial_siblings_keeps_fork(self):
    """Test that joining subset of siblings maintains fork tracking."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)  # Creates children with indices 0, 1, 2
    
    # Only join first two children
    partial_join = parent.join([children[0], children[1]])
    
    # Fork point should still exist with merged indices {0, 1}
    assert len(partial_join.active_forks) == 1
    fork_id = list(partial_join.active_forks.keys())[0]
    merged_indices, total = partial_join.active_forks[fork_id]
    assert merged_indices == frozenset({0, 1})


class TestBranchVisibility:
  """Tests for Branch can_see visibility logic."""

  def test_sibling_isolation(self):
    """Test that siblings cannot see each other during parallel execution."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)  # A, B, C
    
    # Siblings cannot see each other
    assert not children[0].can_see(children[1])  # A can't see B
    assert not children[0].can_see(children[2])  # A can't see C
    assert not children[1].can_see(children[0])  # B can't see A
    assert not children[1].can_see(children[2])  # B can't see C
    assert not children[2].can_see(children[0])  # C can't see A
    assert not children[2].can_see(children[1])  # C can't see B

  def test_self_visibility(self):
    """Test that a branch can always see itself."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(2)
    
    assert children[0].can_see(children[0])
    assert children[1].can_see(children[1])
    assert parent.can_see(parent)

  def test_parent_can_see_children_after_join(self):
    """Test that parent (joined) context can see all children."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)
    
    # Before join, parent sees its own context
    # After join, parent context with all merged tokens can see all children
    joined = parent.join(children)
    
    assert joined.can_see(children[0])
    assert joined.can_see(children[1])
    assert joined.can_see(children[2])

  def test_children_see_ancestors(self):
    """Test that children can see events from ancestors."""
    BranchTokenFactory.reset()
    root = Branch()
    
    # Child can see root
    children = root.fork(2)
    assert children[0].can_see(root)
    assert children[1].can_see(root)
    
    # Grandchild can see root and parent
    grandchildren = children[0].fork(2)
    assert grandchildren[0].can_see(root)
    assert grandchildren[0].can_see(children[0])

  def test_children_cannot_see_parent_siblings(self):
    """Test that children cannot see their parent's siblings."""
    BranchTokenFactory.reset()
    root = Branch()
    children = root.fork(2)  # child0, child1
    grandchildren = children[0].fork(2)  # From child0
    
    # Grandchildren should NOT see child1 (parent's sibling)
    assert not grandchildren[0].can_see(children[1])
    assert not grandchildren[1].can_see(children[1])


class TestGitHubIssue3470Scenarios:
  """Tests for scenarios from GitHub issue #3470.

  Issue: https://github.com/google/adk-python/issues/3470
  Two problematic architectures:
  1. Reducer architecture: Sequential[Parallel[A,B,C], Reducer]
  2. Sequence of parallels: Sequential[Parallel1[A,B,C], Parallel2[D,E,F]]
  """

  def test_reducer_architecture_single(self):
    """Test reducer architecture: Sequential[Parallel[A,B,C], Reducer].

    The reducer R1 should be able to see outputs from A, B, and C.
    """
    BranchTokenFactory.reset()

    root = Branch()
    children = root.fork(3)  # A, B, C
    agent_a_ctx, agent_b_ctx, agent_c_ctx = children

    # After parallel execution, join the branches for reducer
    after_parallel1 = root.join(children)
    reducer1_ctx = after_parallel1

    # CRITICAL: Reducer1 should see all outputs from A, B, C
    assert reducer1_ctx.can_see(agent_a_ctx)
    assert reducer1_ctx.can_see(agent_b_ctx)
    assert reducer1_ctx.can_see(agent_c_ctx)

  def test_nested_reducer_architecture(self):
    """Test nested reducer architecture from issue #3470.

    Architecture:
      Sequential[
        Parallel[
          Sequential[Parallel[A,B,C], R1],
          Sequential[Parallel[D,E,F], R2]
        ],
        R3
      ]
    """
    BranchTokenFactory.reset()

    root = Branch()

    # Top-level parallel splits into two sequential branches
    top_children = root.fork(2)
    seq1_ctx, seq2_ctx = top_children

    # === GROUP 1: Sequential[Parallel[A,B,C], R1] ===
    abc_children = seq1_ctx.fork(3)
    agent_a_ctx, agent_b_ctx, agent_c_ctx = abc_children

    # After parallel1, join for R1
    after_parallel1 = seq1_ctx.join(abc_children)
    reducer1_ctx = after_parallel1

    # R1 should see A, B, C
    assert reducer1_ctx.can_see(agent_a_ctx)
    assert reducer1_ctx.can_see(agent_b_ctx)
    assert reducer1_ctx.can_see(agent_c_ctx)

    # === GROUP 2: Sequential[Parallel[D,E,F], R2] ===
    def_children = seq2_ctx.fork(3)
    agent_d_ctx, agent_e_ctx, agent_f_ctx = def_children

    # After parallel2, join for R2
    after_parallel2 = seq2_ctx.join(def_children)
    reducer2_ctx = after_parallel2

    # R2 should see D, E, F
    assert reducer2_ctx.can_see(agent_d_ctx)
    assert reducer2_ctx.can_see(agent_e_ctx)
    assert reducer2_ctx.can_see(agent_f_ctx)

    # === CROSS-GROUP ISOLATION ===
    # R1 should NOT see D, E, F (different top-level branch)
    assert not reducer1_ctx.can_see(agent_d_ctx)
    assert not reducer1_ctx.can_see(agent_e_ctx)
    assert not reducer1_ctx.can_see(agent_f_ctx)

    # R2 should NOT see A, B, C (different top-level branch)
    assert not reducer2_ctx.can_see(agent_a_ctx)
    assert not reducer2_ctx.can_see(agent_b_ctx)
    assert not reducer2_ctx.can_see(agent_c_ctx)

    # === FINAL: Join both groups and run R3 ===
    final_joined = root.join([after_parallel1, after_parallel2])
    reducer3_ctx = final_joined

    # R3 should see R1 and R2's contexts
    assert reducer3_ctx.can_see(reducer1_ctx)
    assert reducer3_ctx.can_see(reducer2_ctx)

    # R3 should also see all original agents transitively
    assert reducer3_ctx.can_see(agent_a_ctx)
    assert reducer3_ctx.can_see(agent_b_ctx)
    assert reducer3_ctx.can_see(agent_c_ctx)
    assert reducer3_ctx.can_see(agent_d_ctx)
    assert reducer3_ctx.can_see(agent_e_ctx)
    assert reducer3_ctx.can_see(agent_f_ctx)

  def test_sequence_of_parallels(self):
    """Test sequence of parallels from issue #3470.

    Architecture:
      Sequential[
        Parallel1[A, B, C],
        Parallel2[D, E, F],
        Parallel3[G, H, I]
      ]
    """
    BranchTokenFactory.reset()

    root = Branch()

    # === PARALLEL GROUP 1: A, B, C ===
    abc_children = root.fork(3)
    agent_a_ctx, agent_b_ctx, agent_c_ctx = abc_children

    # After parallel1, join for sequential continuation
    after_parallel1 = root.join(abc_children)

    # === PARALLEL GROUP 2: D, E, F ===
    def_children = after_parallel1.fork(3)
    agent_d_ctx, agent_e_ctx, agent_f_ctx = def_children

    # CRITICAL: D, E, F should see A, B, C's outputs
    assert agent_d_ctx.can_see(agent_a_ctx)
    assert agent_d_ctx.can_see(agent_b_ctx)
    assert agent_d_ctx.can_see(agent_c_ctx)
    assert agent_e_ctx.can_see(agent_a_ctx)
    assert agent_f_ctx.can_see(agent_a_ctx)

    # But parallel2 siblings can't see each other
    assert not agent_d_ctx.can_see(agent_e_ctx)
    assert not agent_d_ctx.can_see(agent_f_ctx)

    # After parallel2, join for sequential continuation
    after_parallel2 = after_parallel1.join(def_children)

    # === PARALLEL GROUP 3: G, H, I ===
    ghi_children = after_parallel2.fork(3)
    agent_g_ctx, agent_h_ctx, agent_i_ctx = ghi_children

    # CRITICAL: G, H, I should see ALL previous agents' outputs
    # Can see group 1
    assert agent_g_ctx.can_see(agent_a_ctx)
    assert agent_g_ctx.can_see(agent_b_ctx)
    assert agent_g_ctx.can_see(agent_c_ctx)

    # Can see group 2
    assert agent_g_ctx.can_see(agent_d_ctx)
    assert agent_g_ctx.can_see(agent_e_ctx)
    assert agent_g_ctx.can_see(agent_f_ctx)

    # Same for H and I
    assert agent_h_ctx.can_see(agent_a_ctx)
    assert agent_h_ctx.can_see(agent_d_ctx)
    assert agent_i_ctx.can_see(agent_a_ctx)
    assert agent_i_ctx.can_see(agent_d_ctx)

    # But parallel3 siblings can't see each other
    assert not agent_g_ctx.can_see(agent_h_ctx)
    assert not agent_g_ctx.can_see(agent_i_ctx)


class TestBranchPydantic:
  """Tests for Pydantic serialization of Branch."""

  def test_pydantic_serialization(self):
    """Test that Branch can be serialized by Pydantic."""
    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(2)
    ctx = children[0]

    # Test model_dump (Pydantic serialization)
    dumped = ctx.model_dump()
    assert "active_forks" in dumped

    # Test round-trip - dict serializes as dict
    assert isinstance(dumped["active_forks"], dict)

  def test_immutability(self):
    """Test that Branch is immutable (frozen)."""
    ctx = Branch(active_forks={1: (frozenset({0}), 3)})

    # Should not be able to modify active_forks
    with pytest.raises(Exception):
      ctx.active_forks = ()


class TestBranchEquality:
  """Tests for Branch equality and hashing."""

  def test_equality(self):
    """Test equality based on active_forks."""
    ctx1 = Branch(active_forks={1: (frozenset({0}), 2)})
    ctx2 = Branch(active_forks={1: (frozenset({0}), 2)})
    ctx3 = Branch(active_forks={1: (frozenset({0}), 3)})

    assert ctx1 == ctx2
    assert ctx1 != ctx3

  def test_hashable(self):
    """Test that Branch can be used in sets and dicts."""
    ctx1 = Branch(active_forks={1: (frozenset({0}), 2)})
    ctx2 = Branch(active_forks={1: (frozenset({0}), 2)})
    ctx3 = Branch(active_forks={1: (frozenset({0}), 3)})

    # Should be able to add to set
    context_set = {ctx1, ctx2, ctx3}
    assert len(context_set) == 2  # ctx1 and ctx2 are equal

    # Should be able to use as dict key
    context_dict = {ctx1: "first", ctx3: "second"}
    assert context_dict[ctx2] == "first"  # ctx2 == ctx1

  def test_str_representation(self):
    """Test string representation."""
    root = Branch()
    assert str(root) == "Branch(root)"

    BranchTokenFactory.reset()
    parent = Branch()
    children = parent.fork(3)
    # String representation should include fork info
    assert "Branch" in str(children[0])


class TestMemoryBoundedness:
  """Tests to verify O(D) memory complexity (D = max fork depth)."""

  def test_sequential_parallels_bounded_memory(self):
    """Test that sequential parallels have bounded active_forks.
    
    After join, fork entries are removed, keeping memory bounded.
    """
    BranchTokenFactory.reset()
    root = Branch()
    
    # Simulate 100 sequential parallel operations
    current = root
    for _ in range(100):
      children = current.fork(10)
      current = current.join(children)
    
    # After all joins, should be back to root-like state
    assert current.active_forks == root.active_forks

  def test_nested_depth_tracking(self):
    """Test that nested forks track depth correctly."""
    BranchTokenFactory.reset()
    root = Branch()
    
    # Create deeply nested structure
    current = root
    depth = 10
    for _ in range(depth):
      children = current.fork(2)
      current = children[0]  # Always take first child
    
    # Should have exactly 'depth' active forks
    assert len(current.active_forks) == depth

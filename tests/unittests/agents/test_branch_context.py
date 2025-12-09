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

"""Tests for Branch token-set based branch tracking."""

from __future__ import annotations

from google.adk.agents.branch import Branch
from google.adk.agents.branch import BranchTokenFactory
import pytest


class TestTokenFactory:
  """Tests for the TokenFactory class."""

  def test_new_token_increments(self):
    """Test that new_token generates unique incrementing tokens."""
    # Reset the factory
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

    # Reset the factory
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
    # Should have 1000 total tokens
    assert len(tokens) == 1000


class TestBranchContext:
  """Tests for the Branch class."""

  def test_initialization_default(self):
    """Test that default initialization creates root context."""
    ctx = Branch()
    assert ctx.tokens == frozenset()

  def test_initialization_with_tokens(self):
    """Test initialization with specific tokens."""
    ctx = Branch(tokens=frozenset({1, 2, 3}))
    assert ctx.tokens == frozenset({1, 2, 3})

  def test_fork_creates_children(self):
    """Test that fork creates child contexts."""
    BranchTokenFactory.reset()
    parent = Branch()
    child1 = parent.fork()
    child2 = parent.fork()
    child3 = parent.fork()

    assert isinstance(child1, Branch)
    assert isinstance(child2, Branch)
    assert isinstance(child3, Branch)

  def test_fork_children_have_unique_tokens(self):
    """Test that each forked child has a unique token."""
    BranchTokenFactory.reset()
    parent = Branch(tokens=frozenset({0}))
    child1 = parent.fork()
    child2 = parent.fork()
    child3 = parent.fork()

    # Each child should have parent tokens plus one new unique token
    assert len(child1.tokens) == 2
    assert len(child2.tokens) == 2
    assert len(child3.tokens) == 2

    # Extract the new tokens (the ones not in parent)
    new_token1 = list(child1.tokens - parent.tokens)[0]
    new_token2 = list(child2.tokens - parent.tokens)[0]
    new_token3 = list(child3.tokens - parent.tokens)[0]

    # All new tokens should be unique
    assert len({new_token1, new_token2, new_token3}) == 3

  def test_fork_children_inherit_parent_tokens(self):
    """Test that forked children inherit all parent tokens."""
    BranchTokenFactory.reset()
    parent = Branch(tokens=frozenset({10, 20, 30}))
    child1 = parent.fork()
    child2 = parent.fork()

    assert parent.tokens.issubset(child1.tokens)
    assert parent.tokens.issubset(child2.tokens)

  def test_join_unions_all_tokens(self):
    """Test that join creates union of all token sets."""
    BranchTokenFactory.reset()
    parent = Branch(tokens=frozenset({0}))
    child1 = Branch(tokens=frozenset({0, 1}))
    child2 = Branch(tokens=frozenset({0, 2}))
    child3 = Branch(tokens=frozenset({0, 3}))

    joined = parent.join([child1, child2, child3])

    assert joined.tokens == frozenset({0, 1, 2, 3})

  def test_can_see_subset_relationship(self):
    """Test that can_see implements correct subset logic."""
    parent = Branch(tokens=frozenset({1, 2, 3, 4}))
    event1 = Branch(tokens=frozenset({1, 2}))
    event2 = Branch(tokens=frozenset({1, 2, 3}))
    event3 = Branch(tokens=frozenset({1, 2, 3, 4, 5}))

    # Parent can see events whose tokens are subsets
    assert parent.can_see(event1)  # {1,2} ⊆ {1,2,3,4}
    assert parent.can_see(event2)  # {1,2,3} ⊆ {1,2,3,4}

    # Parent cannot see events with tokens it doesn't have
    assert not parent.can_see(event3)  # {1,2,3,4,5} ⊄ {1,2,3,4}

  def test_can_see_empty_context(self):
    """Test visibility with empty (root) contexts."""
    root = Branch()
    child = Branch(tokens=frozenset({1}))

    # Root can see itself
    assert root.can_see(root)

    # Child can see root (empty set is subset of any set)
    assert child.can_see(root)

    # Root cannot see child
    assert not root.can_see(child)

  def test_copy_creates_independent_instance(self):
    """Test that copy creates a new independent instance."""
    original = Branch(tokens=frozenset({1, 2, 3}))
    copied = original.copy()

    assert original.tokens == copied.tokens
    # Since model is frozen, this is actually the same test
    assert original == copied

  def test_equality(self):
    """Test equality based on token sets."""
    ctx1 = Branch(tokens=frozenset({1, 2, 3}))
    ctx2 = Branch(tokens=frozenset({1, 2, 3}))
    ctx3 = Branch(tokens=frozenset({1, 2}))

    assert ctx1 == ctx2
    assert ctx1 != ctx3
    assert ctx2 != ctx3

  def test_hashable(self):
    """Test that Branch can be used in sets and dicts."""
    ctx1 = Branch(tokens=frozenset({1, 2}))
    ctx2 = Branch(tokens=frozenset({1, 2}))
    ctx3 = Branch(tokens=frozenset({3, 4}))

    # Should be able to add to set
    context_set = {ctx1, ctx2, ctx3}
    # ctx1 and ctx2 are equal, so set should have 2 elements
    assert len(context_set) == 2

    # Should be able to use as dict key
    context_dict = {ctx1: "first", ctx3: "second"}
    assert context_dict[ctx2] == "first"  # ctx2 == ctx1

  def test_str_representation(self):
    """Test string representation."""
    root = Branch()
    assert str(root) == "Branch(root)"

    ctx = Branch(tokens=frozenset({3, 1, 2}))
    # Should show sorted tokens
    assert str(ctx) == "Branch([1, 2, 3])"

  def test_parallel_to_sequential_scenario(self):
    """Test the actual bug scenario: parallel → sequential → parallel."""
    BranchTokenFactory.reset()

    # Root context
    root = Branch()

    # First parallel agent forks to 2 children
    agent1_ctx = root.fork()  # tokens={1}
    agent2_ctx = root.fork()  # tokens={2}

    # After parallel execution, join the branches
    after_parallel1 = root.join([agent1_ctx, agent2_ctx])  # tokens={1,2}

    # Sequential agent passes context through (second parallel agent)
    agent3_ctx = after_parallel1.fork()  # tokens={1,2,3}
    agent4_ctx = after_parallel1.fork()  # tokens={1,2,4}

    # THE BUG FIX: agent3 should be able to see agent1's events
    assert agent3_ctx.can_see(agent1_ctx)  # {1} ⊆ {1,2,3} ✓

    # agent3 should also see agent2's events
    assert agent3_ctx.can_see(agent2_ctx)  # {2} ⊆ {1,2,3} ✓

    # agent4 should see both agent1 and agent2
    assert agent4_ctx.can_see(agent1_ctx)  # {1} ⊆ {1,2,4} ✓
    assert agent4_ctx.can_see(agent2_ctx)  # {2} ⊆ {1,2,4} ✓

    # But siblings shouldn't see each other during parallel execution
    assert not agent1_ctx.can_see(agent2_ctx)  # {2} ⊄ {1} ✗
    assert not agent2_ctx.can_see(agent1_ctx)  # {1} ⊄ {2} ✗
    assert not agent3_ctx.can_see(agent4_ctx)  # {1,2,4} ⊄ {1,2,3} ✗
    assert not agent4_ctx.can_see(agent3_ctx)  # {1,2,3} ⊄ {1,2,4} ✗

  def test_pydantic_serialization(self):
    """Test that Branch can be serialized by Pydantic."""
    ctx = Branch(tokens=frozenset({1, 2, 3}))

    # Test model_dump (Pydantic serialization)
    dumped = ctx.model_dump()
    assert "tokens" in dumped
    # Frozenset gets converted to some iterable
    assert set(dumped["tokens"]) == {1, 2, 3}

    # Test round-trip
    restored = Branch(**dumped)
    assert restored.tokens == ctx.tokens

  def test_immutability(self):
    """Test that Branch is immutable (frozen)."""
    ctx = Branch(tokens=frozenset({1, 2, 3}))

    # Should not be able to modify tokens
    with pytest.raises(
        Exception
    ):  # Pydantic raises ValidationError or AttributeError
      ctx.tokens = frozenset({4, 5, 6})


class TestGitHubIssue3470Scenarios:
  """Tests for the exact scenarios described in GitHub issue #3470.

  Issue: https://github.com/google/adk-python/issues/3470
  Two problematic architectures:
  1. Reducer architecture: Sequential[Parallel[A,B,C], Reducer]
  2. Sequence of parallels: Sequential[Parallel1[A,B,C], Parallel2[D,E,F]]
  """

  def test_reducer_architecture_single(self):
    """Test reducer architecture: Sequential[Parallel[A,B,C], Reducer].

    The reducer R1 should be able to see outputs from A, B, and C.
    This is the basic reducer pattern that should work.
    """
    BranchTokenFactory.reset()

    # Root context
    root = Branch()

    # Sequential agent S1 has sub-agents: [Parallel1, Reducer1]
    # Parallel1 forks into A, B, C
    agent_a_ctx = root.fork()  # tokens={1}
    agent_b_ctx = root.fork()  # tokens={2}
    agent_c_ctx = root.fork()  # tokens={3}

    # After parallel execution, join the branches for sequential continuation
    after_parallel1 = root.join(
        [agent_a_ctx, agent_b_ctx, agent_c_ctx]
    )  # tokens={1,2,3}

    # Reducer1 runs in sequential after parallel, uses joined context
    reducer1_ctx = after_parallel1

    # CRITICAL: Reducer1 should see all outputs from A, B, C
    assert reducer1_ctx.can_see(agent_a_ctx)  # {1} ⊆ {1,2,3} ✓
    assert reducer1_ctx.can_see(agent_b_ctx)  # {2} ⊆ {1,2,3} ✓
    assert reducer1_ctx.can_see(agent_c_ctx)  # {3} ⊆ {1,2,3} ✓

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

    This is the failing case where:
    - R1 should see A, B, C
    - R2 should see D, E, F
    - R3 should see R1, R2 (and transitively A-F)
    """
    BranchTokenFactory.reset()

    root = Branch()

    # Top-level parallel splits into two sequential branches
    seq1_ctx = root.fork()  # Group1: tokens={1}
    seq2_ctx = root.fork()  # Group2: tokens={2}

    # === GROUP 1: Sequential[Parallel[A,B,C], R1] ===
    # Parallel1 (ABC) forks from seq1_ctx
    agent_a_ctx = seq1_ctx.fork()  # tokens={1,3}
    agent_b_ctx = seq1_ctx.fork()  # tokens={1,4}
    agent_c_ctx = seq1_ctx.fork()  # tokens={1,5}

    # After parallel1, join for R1
    after_parallel1 = seq1_ctx.join(
        [agent_a_ctx, agent_b_ctx, agent_c_ctx]
    )  # tokens={1,3,4,5}
    reducer1_ctx = after_parallel1

    # R1 should see A, B, C
    assert reducer1_ctx.can_see(agent_a_ctx)  # {1,3} ⊆ {1,3,4,5} ✓
    assert reducer1_ctx.can_see(agent_b_ctx)  # {1,4} ⊆ {1,3,4,5} ✓
    assert reducer1_ctx.can_see(agent_c_ctx)  # {1,5} ⊆ {1,3,4,5} ✓

    # === GROUP 2: Sequential[Parallel[D,E,F], R2] ===
    # Parallel2 (DEF) forks from seq2_ctx
    agent_d_ctx = seq2_ctx.fork()  # tokens={2,6}
    agent_e_ctx = seq2_ctx.fork()  # tokens={2,7}
    agent_f_ctx = seq2_ctx.fork()  # tokens={2,8}

    # After parallel2, join for R2
    after_parallel2 = seq2_ctx.join(
        [agent_d_ctx, agent_e_ctx, agent_f_ctx]
    )  # tokens={2,6,7,8}
    reducer2_ctx = after_parallel2

    # R2 should see D, E, F
    assert reducer2_ctx.can_see(agent_d_ctx)  # {2,6} ⊆ {2,6,7,8} ✓
    assert reducer2_ctx.can_see(agent_e_ctx)  # {2,7} ⊆ {2,6,7,8} ✓
    assert reducer2_ctx.can_see(agent_f_ctx)  # {2,8} ⊆ {2,6,7,8} ✓

    # === FINAL: Join both groups and run R3 ===
    # After top-level parallel completes, join for final reducer
    final_joined = root.join(
        [after_parallel1, after_parallel2]
    )  # tokens={1,2,3,4,5,6,7,8}
    reducer3_ctx = final_joined

    # R3 should see R1 and R2's contexts
    assert reducer3_ctx.can_see(reducer1_ctx)  # {1,3,4,5} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(reducer2_ctx)  # {2,6,7,8} ⊆ {1,2,3,4,5,6,7,8} ✓

    # R3 should also see all original agents transitively
    assert reducer3_ctx.can_see(agent_a_ctx)  # {1,3} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(agent_b_ctx)  # {1,4} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(agent_c_ctx)  # {1,5} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(agent_d_ctx)  # {2,6} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(agent_e_ctx)  # {2,7} ⊆ {1,2,3,4,5,6,7,8} ✓
    assert reducer3_ctx.can_see(agent_f_ctx)  # {2,8} ⊆ {1,2,3,4,5,6,7,8} ✓

    # But groups shouldn't see each other during parallel execution
    assert not agent_a_ctx.can_see(agent_d_ctx)  # {2,6} ⊄ {1,3} ✗
    assert not reducer1_ctx.can_see(reducer2_ctx)  # {2,6,7,8} ⊄ {1,3,4,5} ✗

  def test_sequence_of_parallels(self):
    """Test sequence of parallels from issue #3470.

    Architecture:
      Sequential[
        Parallel1[A, B, C],
        Parallel2[D, E, F],
        Parallel3[G, H, I]
      ]

    The bug: With string-based branches:
    - A, B, C have branches: parallel1.A, parallel1.B, parallel1.C
    - D, E, F have branches: parallel2.D, parallel2.E, parallel2.F
    - G, H, I have branches: parallel3.G, parallel3.H, parallel3.I

    These are NOT prefixes of each other, so D/E/F can't see A/B/C,
    and G/H/I can't see anyone before them.

    With token-sets: Each subsequent parallel group inherits tokens from
    previous groups via join, so visibility works correctly.
    """
    BranchTokenFactory.reset()

    root = Branch()

    # === PARALLEL GROUP 1: A, B, C ===
    agent_a_ctx = root.fork()  # tokens={1}
    agent_b_ctx = root.fork()  # tokens={2}
    agent_c_ctx = root.fork()  # tokens={3}

    # After parallel1, join for sequential continuation
    after_parallel1 = root.join(
        [agent_a_ctx, agent_b_ctx, agent_c_ctx]
    )  # tokens={1,2,3}

    # === PARALLEL GROUP 2: D, E, F ===
    # Fork from joined context, so inherits all previous tokens
    agent_d_ctx = after_parallel1.fork()  # tokens={1,2,3,4}
    agent_e_ctx = after_parallel1.fork()  # tokens={1,2,3,5}
    agent_f_ctx = after_parallel1.fork()  # tokens={1,2,3,6}

    # CRITICAL: D, E, F should see A, B, C's outputs
    assert agent_d_ctx.can_see(agent_a_ctx)  # {1} ⊆ {1,2,3,4} ✓
    assert agent_d_ctx.can_see(agent_b_ctx)  # {2} ⊆ {1,2,3,4} ✓
    assert agent_d_ctx.can_see(agent_c_ctx)  # {3} ⊆ {1,2,3,4} ✓

    assert agent_e_ctx.can_see(agent_a_ctx)  # {1} ⊆ {1,2,3,5} ✓
    assert agent_f_ctx.can_see(agent_a_ctx)  # {1} ⊆ {1,2,3,6} ✓

    # But parallel2 siblings can't see each other
    assert not agent_d_ctx.can_see(agent_e_ctx)  # {1,2,3,5} ⊄ {1,2,3,4} ✗
    assert not agent_d_ctx.can_see(agent_f_ctx)  # {1,2,3,6} ⊄ {1,2,3,4} ✗

    # After parallel2, join for sequential continuation
    after_parallel2 = after_parallel1.join(
        [agent_d_ctx, agent_e_ctx, agent_f_ctx]
    )  # tokens={1,2,3,4,5,6}

    # === PARALLEL GROUP 3: G, H, I ===
    agent_g_ctx = after_parallel2.fork()  # tokens={1,2,3,4,5,6,7}
    agent_h_ctx = after_parallel2.fork()  # tokens={1,2,3,4,5,6,8}
    agent_i_ctx = after_parallel2.fork()  # tokens={1,2,3,4,5,6,9}

    # CRITICAL: G, H, I should see ALL previous agents' outputs
    # Can see group 1
    assert agent_g_ctx.can_see(agent_a_ctx)  # {1} ⊆ {1,2,3,4,5,6,7} ✓
    assert agent_g_ctx.can_see(agent_b_ctx)  # {2} ⊆ {1,2,3,4,5,6,7} ✓
    assert agent_g_ctx.can_see(agent_c_ctx)  # {3} ⊆ {1,2,3,4,5,6,7} ✓

    # Can see group 2
    assert agent_g_ctx.can_see(agent_d_ctx)  # {1,2,3,4} ⊆ {1,2,3,4,5,6,7} ✓
    assert agent_g_ctx.can_see(agent_e_ctx)  # {1,2,3,5} ⊆ {1,2,3,4,5,6,7} ✓
    assert agent_g_ctx.can_see(agent_f_ctx)  # {1,2,3,6} ⊆ {1,2,3,4,5,6,7} ✓

    # Same for H and I
    assert agent_h_ctx.can_see(agent_a_ctx)
    assert agent_h_ctx.can_see(agent_d_ctx)
    assert agent_i_ctx.can_see(agent_a_ctx)
    assert agent_i_ctx.can_see(agent_d_ctx)

    # But parallel3 siblings can't see each other
    assert not agent_g_ctx.can_see(
        agent_h_ctx
    )  # {1,2,3,4,5,6,8} ⊄ {1,2,3,4,5,6,7} ✗
    assert not agent_g_ctx.can_see(
        agent_i_ctx
    )  # {1,2,3,4,5,6,9} ⊄ {1,2,3,4,5,6,7} ✗

  def test_string_based_approach_fails(self):
    """Demonstrate why string-based prefix matching fails for sequence of parallels.

    This test documents the OLD broken behavior to show why token-sets are necessary.
    """
    # With string-based branches (OLD APPROACH - BROKEN):
    # Parallel1: "parallel1.A", "parallel1.B", "parallel1.C"
    # Parallel2: "parallel2.D", "parallel2.E", "parallel2.F"

    # Check if "parallel2.D" starts with "parallel1.A"
    assert not "parallel2.D".startswith("parallel1.A")  # FALSE - Can't see!

    # Check if "parallel1.A" starts with "parallel2.D"
    assert not "parallel1.A".startswith("parallel2.D")  # FALSE - Can't see!

    # Neither direction works with prefix matching for sibling parallel groups!
    # This is why the bug exists in the original implementation.

    # With token-sets (NEW APPROACH - CORRECT):
    # After parallel1, context has tokens {1,2,3}
    # Parallel2 forks from {1,2,3}, so D gets {1,2,3,4}
    # Agent A has tokens {1}
    # Check: {1} ⊆ {1,2,3,4} = TRUE ✓

    # Token-set approach correctly handles this case!

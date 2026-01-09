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

"""Branch context for structural event filtering in parallel agents.

Uses structural fork tracking with O(D) bounded memory (D = parallel nesting depth),
instead of token accumulation which grows O(n) with iterations.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_serializer


class BranchTokenFactory:
  """Thread-safe global counter for fork IDs.

  Each fork operation in a parallel agent execution creates a unique fork ID
  used to track which parallel execution context events belong to.

  The counter resets at the start of each invocation, ensuring fork IDs are
  only used for parallel execution isolation within that invocation. Events
  from previous invocations are always visible (branch filtering only applies
  within current invocation).
  """

  _lock = threading.Lock()
  _next = 0

  @classmethod
  def new_token(cls) -> int:
    """Generate a new unique fork ID.

    Returns:
      A unique integer fork ID.
    """
    with cls._lock:
      cls._next += 1
      return cls._next

  @classmethod
  def reset(cls) -> None:
    """Reset the counter to zero.

    This should be called at the start of each invocation to ensure fork IDs
    are fresh for that invocation's parallel execution tracking.
    """
    with cls._lock:
      cls._next = 0


class Branch(BaseModel):
  """Structural branch tracking for parallel agent execution with bounded memory.

  == PROBLEM ==
  When ParallelAgent runs workers A, B, C in parallel:
  - During execution: A should NOT see B's or C's events (sibling isolation)
  - After join: The parent context should see A, B, C's events (merged history)

  == SOLUTION: Structural Fork Tracking ==
  Instead of accumulating tokens forever (O(n) growth), we track:
  - Which fork points are currently active (not yet fully joined)
  - Which branch index(es) we're on at each fork point

  == DATA STRUCTURE ==
  active_forks: dict[fork_id, (branch_indices, total_branches_in_fork)]

  - fork_id (int): Unique ID for each fork point (ParallelAgent execution)
  - branch_indices (frozenset[int]): Which branch(es) of that fork we're on.
    Starts as {0}, {1}, etc. After partial joins, could be {0, 1}.
  - total_branches_in_fork (int): How many branches were created at this fork.
    Needed to know when ALL branches have joined (so we can delete the entry).

  == OPERATIONS ==
  fork(N): Create N children. Each child gets active_forks[new_fork_id] = ({i}, N)
  join([others]): Merge branch_indices. If all N branches merged, DELETE the entry.
  can_see(event): Check visibility rules below.

  == VISIBILITY RULES ==
  For each fork_id in the event's active_forks:
    1. If fork_id NOT in our active_forks -> VISIBLE (fork was joined, it's history)
    2. If fork_id in our active_forks:
       - If our branch_indices & event's branch_indices is non-empty -> VISIBLE
       - If intersection is empty -> HIDDEN (different branches, same active fork)

  == WHY MEMORY IS BOUNDED ==
  When ALL branches of a fork join together, we DELETE that fork entry.
  After Loop(Parallel(A, B)) runs 1000 times:
  - Token-set approach: {1, 2, 3, ..., 2000} - grows forever
  - Structural approach: {} - empty after each full join!

  Memory is O(D) where D = parallel nesting depth (typically 1-3), not O(iterations).
  """

  model_config = ConfigDict(
      frozen=True,
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  active_forks: dict[int, tuple[frozenset[int], int]] = Field(default_factory=dict)
  """Maps fork_id -> (branch_indices_we_have, total_branches_in_this_fork).

  Example: {1: (frozenset({0}), 3)} means:
  - At fork point 1, we're on branch index 0
  - Fork point 1 has 3 total branches (0, 1, 2)
  - We can only see events that have branch 0 at fork 1

  Example after partial join: {1: (frozenset({0, 1}), 3)} means:
  - At fork point 1, we've merged branches 0 and 1
  - Fork point 1 has 3 total branches
  - We can see events from branch 0 OR branch 1, but NOT branch 2
  """

  @model_serializer
  def serialize_model(self):
    """Custom serializer to convert to JSON-serializable format."""
    return {
        'active_forks': {
            str(k): (list(v[0]), v[1]) for k, v in self.active_forks.items()
        }
    }

  def fork(self, num_children: int) -> list[Branch]:
    """Create N child branches at a new fork point.

    Each child gets labeled with its branch index (0, 1, 2, ..., N-1) at this fork.

    Args:
      num_children: Number of parallel branches to create.

    Returns:
      List of N Branch objects, one per parallel worker.

    Example: parent.fork(3) creates:
      - Child 0: {fork_1: ({0}, 3)}  "I'm branch 0 of fork 1 (which has 3 branches)"
      - Child 1: {fork_1: ({1}, 3)}  "I'm branch 1 of fork 1"
      - Child 2: {fork_1: ({2}, 3)}  "I'm branch 2 of fork 1"
    """
    fork_id = BranchTokenFactory.new_token()
    children = []
    for i in range(num_children):
      new_forks = dict(self.active_forks)
      new_forks[fork_id] = (frozenset({i}), num_children)
      children.append(Branch(active_forks=new_forks))
    return children

  def join(self, others: list[Branch]) -> Branch:
    """Merge branches back together after parallel execution.

    Combines branch indices at each fork point. When ALL branches of a fork
    are merged together, that fork entry is DELETED (this is the key to
    bounded memory - completed forks don't accumulate).

    Args:
      others: List of other Branches to join with self.

    Returns:
      New Branch with merged fork state.

    Example - Partial join (2 of 3 branches):
      {fork_1: ({0}, 3)}.join([{fork_1: ({1}, 3)}])
      -> {fork_1: ({0, 1}, 3)}  # Still active, branch 2 hasn't joined

    Example - Full join (all 3 branches):
      {fork_1: ({0, 1}, 3)}.join([{fork_1: ({2}, 3)}])
      -> {}  # Fork entry DELETED! All branches merged.
    """
    merged_forks = dict(self.active_forks)

    for other in others:
      for fork_id, (other_branches, total) in other.active_forks.items():
        if fork_id in merged_forks:
          my_branches, my_total = merged_forks[fork_id]
          combined = my_branches | other_branches

          # Full join: all branches merged -> DELETE fork entry (bounded memory!)
          if len(combined) >= my_total:
            del merged_forks[fork_id]
          else:
            # Partial join: keep tracking with merged branch set
            merged_forks[fork_id] = (combined, my_total)
        else:
          # Fork not in self yet, add it (unless already fully merged)
          if len(other_branches) < total:
            merged_forks[fork_id] = (other_branches, total)

    return Branch(active_forks=merged_forks)

  def can_see(self, event_branch: Branch) -> bool:
    """Check if this context can see an event with the given branch.

    THE ONE RULE:
    You can only see an event if you're NOT on a different active branch
    of the same fork.

    For each fork in the event:
    - If that fork is NOT in our active_forks -> VISIBLE
      (The fork was fully joined, it's history now)
    - If that fork IS in our active_forks -> check branch indices
      - If we share at least one branch index -> VISIBLE (same path)
      - If no overlap -> HIDDEN (different branch of same active fork)

    Args:
      event_branch: The Branch of the event to check visibility for.

    Returns:
      True if the event is visible, False otherwise.

    Examples:
      Context {1: ({0}, 2)} checking event {1: ({0}, 2)} -> same branch -> VISIBLE
      Context {1: ({0}, 2)} checking event {1: ({1}, 2)} -> different branch -> HIDDEN
      Context {} checking event {1: ({0}, 2)} -> fork 1 not active -> VISIBLE (history)
      Context {1: ({0, 1}, 2)} checking event {1: ({0}, 2)} -> overlapping -> VISIBLE
    """
    for fork_id, (event_branches, _) in event_branch.active_forks.items():
      if fork_id in self.active_forks:
        my_branches, _ = self.active_forks[fork_id]
        # Same active fork - must share at least one branch index
        if not (my_branches & event_branches):
          return False  # Different branch of same active fork -> HIDDEN
      # Fork not in self -> it's been joined -> visible (history)
    return True

  def __hash__(self) -> int:
    """Hash based on active_forks content."""
    return hash(
        tuple(
            sorted(
                (k, tuple(sorted(v[0])), v[1])
                for k, v in self.active_forks.items()
            )
        )
    )

  def __str__(self) -> str:
    """Human-readable string representation.

    Returns:
      String showing fork state or "root" if empty.
    """
    if not self.active_forks:
      return 'Branch(root)'
    parts = [
        f'{fid}:{sorted(br)}/{tot}'
        for fid, (br, tot) in sorted(self.active_forks.items())
    ]
    return f"Branch({', '.join(parts)})"

  def __repr__(self) -> str:
    """Developer representation.

    Returns:
      String representation for debugging.
    """
    return str(self)

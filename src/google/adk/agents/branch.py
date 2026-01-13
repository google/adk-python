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

"""Branch context for provenance-based event filtering in parallel agents."""

from __future__ import annotations

import threading
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_serializer
from pydantic import PrivateAttr


class BranchTokenFactory:
  """Thread-safe global counter for branch tokens.

  Each fork operation in a parallel agent execution creates new unique tokens
  that are used to track provenance and determine event visibility across
  branches WITHIN a single invocation.

  The counter resets at the start of each invocation, ensuring tokens are
  only used for parallel execution isolation within that invocation. Events
  from previous invocations are always visible (branch filtering only applies
  within current invocation).
  """

  _lock = threading.Lock()
  _next = 0

  @classmethod
  def new_token(cls) -> int:
    """Generate a new unique token.

    Returns:
      A unique integer token.
    """
    with cls._lock:
      cls._next += 1
      return cls._next

  @classmethod
  def reset(cls) -> None:
    """Reset the counter to zero.

    This should be called at the start of each invocation to ensure tokens
    are fresh for that invocation's parallel execution tracking.
    """
    with cls._lock:
      cls._next = 0


class Branch(BaseModel):
  """Branch tracking using token sets for parallel agent execution.

  Tracks event provenance across parallel and sequential agent execution.
  Event visibility is determined by subset relationships: an event is visible
  to a context if all the event's tokens are present in the context's token set.

  Example:
    Root context: {}
    After fork(): child_0 has {1}, child_1 has {2}
    After join: parent has {1, 2}

    Events from child_0 (tokens={1}) are visible to parent (tokens={1,2})
    because {1} ⊆ {1,2}.
  """

  model_config = ConfigDict(
      frozen=True,  # Make instances immutable for hashing
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  tokens: frozenset[int] = Field(default_factory=frozenset)
  """Set of integer tokens representing branch provenance.
  
  If empty, represents the root context. Use frozenset for immutability
  and to enable hashing for use in sets/dicts.
  """

  @model_serializer
  def serialize_model(self):
    """Custom serializer to convert frozenset to list for JSON serialization."""
    return {'tokens': list(self.tokens)}

  def fork(self) -> Branch:
    """Create a child context for parallel execution.

    The child gets a unique new token added to the parent's token set.
    This ensures:
    1. Child can see parent's events (parent tokens ⊆ child tokens)
    2. Siblings cannot see each other's events (sibling tokens are disjoint)

    Returns:
      A new Branch with parent.tokens ∪ {new_token}.
    """
    new_token = BranchTokenFactory.new_token()
    return Branch(tokens=self.tokens | {new_token})

  def join(self, others: list[Branch]) -> Branch:
    """Merge token sets from parallel branches.

    This is called when parallel execution completes and we need to merge
    the provenance from all branches. The result contains the union of all
    token sets, ensuring subsequent agents can see events from all branches.

    Args:
      others: List of other Branches to join with self.

    Returns:
      New Branch with union of all token sets.
    """
    combined = set(self.tokens)
    for ctx in others:
      combined |= ctx.tokens
    return Branch(tokens=frozenset(combined))

  def can_see(self, event_ctx: Branch) -> bool:
    """Check if an event is visible from this context.

    An event is visible if all of its tokens are present in the current
    context's token set (subset relationship).

    Args:
      event_ctx: The Branch of the event to check.

    Returns:
      True if the event is visible, False otherwise.
    """
    return event_ctx.tokens.issubset(self.tokens)

  def __str__(self) -> str:
    """Human-readable string representation.

    Returns:
      String showing token set or "root" if empty.
    """
    if not self.tokens:
      return 'Branch(root)'
    return f'Branch({sorted(self.tokens)})'

  def __repr__(self) -> str:
    """Developer representation.

    Returns:
      String representation for debugging.
    """
    return str(self)

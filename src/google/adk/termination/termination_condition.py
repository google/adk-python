# Copyright 2026 Google LLC
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

"""Base termination condition and compound combinators."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional
from typing import Sequence

from ..events.event import Event


@dataclass
class TerminationResult:
  """The result returned by a termination condition when the conversation should stop."""

  reason: str
  """A human-readable description of why the conversation was terminated."""


class TerminationCondition(abc.ABC):
  """Abstract base class for all termination conditions.

  A termination condition is evaluated after each event in the agent loop.
  When ``check()`` returns a ``TerminationResult``, the loop stops and the
  ``reason`` is surfaced in the final event's ``actions.termination_reason``.

  Conditions are stateful but reset automatically at the start of each run.
  They can be combined with ``.and_()`` and ``.or_()`` to create compound
  logic.

  Example::

      condition = MaxIterationsTermination(10).or_(
          TextMentionTermination('TERMINATE')
      )
  """

  @property
  @abc.abstractmethod
  def terminated(self) -> bool:
    """Whether this termination condition has been reached."""

  @abc.abstractmethod
  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    """Checks whether the termination condition is met.

    Called after each event emitted by the agent. Returns a
    ``TerminationResult`` if the loop should stop, or ``None`` to continue.

    Args:
      events: The delta sequence of events since the last check.
    """

  @abc.abstractmethod
  async def reset(self) -> None:
    """Resets this condition to its initial state.

    Called automatically at the start of each run so the same instance can
    be reused across multiple runs.
    """

  def and_(self, other: TerminationCondition) -> TerminationCondition:
    """Returns a new condition that terminates only when BOTH conditions are met.

    Args:
      other: The other termination condition.
    """
    return AndTerminationCondition(self, other)

  def or_(self, other: TerminationCondition) -> TerminationCondition:
    """Returns a new condition that terminates when EITHER condition is met.

    Args:
      other: The other termination condition.
    """
    return OrTerminationCondition(self, other)

  def __and__(self, other: TerminationCondition) -> TerminationCondition:
    """Supports ``condition_a & condition_b`` syntax."""
    return self.and_(other)

  def __or__(self, other: TerminationCondition) -> TerminationCondition:
    """Supports ``condition_a | condition_b`` syntax."""
    return self.or_(other)


class AndTerminationCondition(TerminationCondition):
  """A compound condition that terminates only when ALL children have fired."""

  def __init__(
      self,
      left: TerminationCondition,
      right: TerminationCondition,
  ) -> None:
    self._left = left
    self._right = right
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return None
    # Forward to both children so each accumulates its own state.
    await self._left.check(events)
    await self._right.check(events)

    if self._left.terminated and self._right.terminated:
      self._terminated = True
      return TerminationResult(reason='All termination conditions met')
    return None

  async def reset(self) -> None:
    self._terminated = False
    await self._left.reset()
    await self._right.reset()


class OrTerminationCondition(TerminationCondition):
  """A compound condition that terminates when ANY child fires first."""

  def __init__(
      self,
      left: TerminationCondition,
      right: TerminationCondition,
  ) -> None:
    self._left = left
    self._right = right
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return None

    left_result = await self._left.check(events)
    if left_result:
      self._terminated = True
      return left_result

    right_result = await self._right.check(events)
    if right_result:
      self._terminated = True
      return right_result

    return None

  async def reset(self) -> None:
    self._terminated = False
    await self._left.reset()
    await self._right.reset()

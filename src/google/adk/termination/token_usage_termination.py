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

"""Terminates when cumulative token usage exceeds a limit."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


class TokenUsageTermination(TerminationCondition):
  """Terminates when cumulative token usage exceeds configured limits.

  At least one of the token limits must be provided.

  Example::

      # Stop after 10000 total tokens
      condition = TokenUsageTermination(max_total_tokens=10_000)

      # Stop after 5000 prompt tokens OR 2000 completion tokens
      condition = TokenUsageTermination(
          max_prompt_tokens=5_000,
          max_completion_tokens=2_000,
      )
  """

  def __init__(
      self,
      *,
      max_total_tokens: Optional[int] = None,
      max_prompt_tokens: Optional[int] = None,
      max_completion_tokens: Optional[int] = None,
  ) -> None:
    if (
        max_total_tokens is None
        and max_prompt_tokens is None
        and max_completion_tokens is None
    ):
      raise ValueError(
          'At least one of max_total_tokens, max_prompt_tokens, or'
          ' max_completion_tokens must be provided.'
      )
    self._max_total_tokens = max_total_tokens
    self._max_prompt_tokens = max_prompt_tokens
    self._max_completion_tokens = max_completion_tokens
    self._total_tokens = 0
    self._prompt_tokens = 0
    self._completion_tokens = 0
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(self, events: Sequence[Event]) -> Optional[TerminationResult]:
    if self._terminated:
      return None

    for event in events:
      if not event.usage_metadata:
        continue

      self._total_tokens += event.usage_metadata.total_token_count or 0
      self._prompt_tokens += event.usage_metadata.prompt_token_count or 0
      self._completion_tokens += (
          event.usage_metadata.candidates_token_count or 0
      )

      if (
          self._max_total_tokens is not None
          and self._total_tokens >= self._max_total_tokens
      ):
        self._terminated = True
        return TerminationResult(
            reason=(
                f'Token limit exceeded: total_tokens={self._total_tokens}'
                f' >= max_total_tokens={self._max_total_tokens}'
            )
        )

      if (
          self._max_prompt_tokens is not None
          and self._prompt_tokens >= self._max_prompt_tokens
      ):
        self._terminated = True
        return TerminationResult(
            reason=(
                f'Token limit exceeded: prompt_tokens={self._prompt_tokens}'
                f' >= max_prompt_tokens={self._max_prompt_tokens}'
            )
        )

      if (
          self._max_completion_tokens is not None
          and self._completion_tokens >= self._max_completion_tokens
      ):
        self._terminated = True
        return TerminationResult(
            reason=(
                'Token limit exceeded:'
                f' completion_tokens={self._completion_tokens}'
                f' >= max_completion_tokens={self._max_completion_tokens}'
            )
        )

    return None

  async def reset(self) -> None:
    self._terminated = False
    self._total_tokens = 0
    self._prompt_tokens = 0
    self._completion_tokens = 0

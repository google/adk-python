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

"""Terminates when a specific text string is found in event content."""

from __future__ import annotations

from typing import Optional
from typing import Sequence

from ..events.event import Event
from .termination_condition import TerminationCondition
from .termination_condition import TerminationResult


def _stringify_event_content(event: Event) -> str:
  """Extracts a text representation from an event's content."""
  if not event.content or not event.content.parts:
    return ''
  texts = []
  for part in event.content.parts:
    if part.text:
      texts.append(part.text)
  return ' '.join(texts)


class TextMentionTermination(TerminationCondition):
  """Terminates the conversation when a specific text is found in event content.

  Example::

      # Stop when any agent says "TERMINATE"
      condition = TextMentionTermination('TERMINATE')

      # Stop only when the "critic" agent says "APPROVE"
      condition = TextMentionTermination('APPROVE', sources=['critic'])
  """

  def __init__(
      self,
      text: str,
      sources: Optional[Sequence[str]] = None,
  ) -> None:
    self._text = text
    self._sources = list(sources) if sources else None
    self._terminated = False

  @property
  def terminated(self) -> bool:
    return self._terminated

  async def check(
      self, events: Sequence[Event]
  ) -> Optional[TerminationResult]:
    if self._terminated:
      return None

    for event in events:
      if (
          self._sources
          and (event.author or '') not in self._sources
      ):
        continue

      if self._text in _stringify_event_content(event):
        self._terminated = True
        return TerminationResult(
            reason=f"Text '{self._text}' mentioned"
        )
    return None

  async def reset(self) -> None:
    self._terminated = False

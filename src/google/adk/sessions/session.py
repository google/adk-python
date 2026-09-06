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

"""Session model with event filtering capabilities.

This module contains the Session class which represents a series of interactions
between a user and agents, including methods for retrieving and filtering events.
"""

from __future__ import annotations

from typing import Any

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..events.event import Event


class Session(BaseModel):
  """Represents a series of interactions between a user and agents."""

  model_config = ConfigDict(
      extra="forbid",
      arbitrary_types_allowed=True,
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  id: str = Field(
      description="Unique identifier of the session.",
      examples=["session-abc123"],
  )
  app_name: str = Field(
      description="Application name that owns the session.",
      examples=["hello_world"],
  )
  user_id: str = Field(
      description="User ID that owns the session.",
      examples=["user-123"],
  )
  state: dict[str, Any] = Field(
      default_factory=dict,
      description="Current persisted session state.",
      examples=[{"locale": "en-US"}],
  )
  events: list[Event] = Field(
      default_factory=list,
      description=(
          "Ordered event history for the session, including user, model, and"
          " tool events (e.g. user input, model response, function"
          " call/response)."
      ),
  )
  last_update_time: float = Field(
      default=0.0,
      description=(
          "Unix timestamp in seconds for the most recent session update."
      ),
      examples=[1_742_000_000.0],
  )

  def get_events(self) -> list[Event]:
    """Returns all events in the session.

    This method provides a consistent API for accessing events alongside
    the filter_events() method.

    Returns:
        A list containing all events in the session.

    Example:
        >>> for event in session.get_events():
        ...     print(event.author, event.content)
    """
    return self.events

  def filter_events(self, *, exclude_rewound: bool = True) -> list[Event]:
    """Returns filtered events from the session.

    This method provides convenient filtering of session events, with the
    primary use case being exclusion of events that have been invalidated
    by rewind operations.

    Args:
        exclude_rewound: If True (default), excludes events that have been
            invalidated by a rewind operation. When a session is rewound,
            all events from the rewind target invocation onwards are
            considered "rewound" and will be excluded.

    Returns:
        A filtered list of events based on the specified criteria.

    Example:
        >>> # Get only active events (excluding rewound ones)
        >>> for event in session.filter_events():
        ...     process_event(event)

        >>> # Get all events including rewound ones
        >>> for event in session.filter_events(exclude_rewound=False):
        ...     process_all_events(event)
    """
    if not exclude_rewound:
      return self.events
    return self._filter_rewound_events()

  def _filter_rewound_events(self) -> list[Event]:
    """Filter out events that have been invalidated by a rewind operation.

    This method implements the rewind filtering logic: it iterates backward
    through the events, and when a rewind event is found (identified by
    having a non-None `actions.rewind_before_invocation_id`), it skips all
    events from the rewind target invocation up to and including the rewind
    event itself.

    The algorithm works as follows:
    1. Iterate through events from the end to the beginning
    2. When a rewind event is encountered, find the first event with the
       target invocation_id and skip all events from that point to the
       rewind event
    3. Events not affected by any rewind are included in the result
    4. The final list is reversed to maintain chronological order

    Returns:
        A list of events with rewound events filtered out.
    """
    if not self.events:
      return []

    filtered: list[Event] = []
    i = len(self.events) - 1

    while i >= 0:
      event = self.events[i]

      # Check if this is a rewind event
      if event.actions and event.actions.rewind_before_invocation_id:
        rewind_invocation_id = event.actions.rewind_before_invocation_id

        # Find the first event with the target invocation_id and skip to it
        for j in range(0, i):
          if self.events[j].invocation_id == rewind_invocation_id:
            # Skip all events from j to i (inclusive of the rewind event)
            i = j
            break
      else:
        # Not a rewind event, include it
        filtered.append(event)

      i -= 1

    # Reverse to restore chronological order
    filtered.reverse()
    return filtered

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
from __future__ import annotations

import abc
from typing import Optional

from google.genai.types import Content

from ..events.event import Event
from ..utils.feature_decorator import experimental


@experimental
class BaseEventsSummarizer(abc.ABC):
  """Base interface for compacting events."""

  @abc.abstractmethod
  async def maybe_summarize_events(
      self, *, events: list[Event]
  ) -> Optional[Event]:
    """Compact a list of events into a single event.

    If compaction failed, return None. Otherwise, compact into a content and
    return it.

    This method will summarize the events and return a new summray event
    indicating the range of events it summarized.

    Args:
      events: Events to compact.

    Returns:
      The new compacted event, or None if no compaction happended.
    """
    raise NotImplementedError()

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

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from ..utils.feature_decorator import experimental
from .base_events_summarizer import BaseEventsSummarizer


@experimental
class ResumabilityConfig(BaseModel):
  """The config of the resumability for an application.

  The "resumability" in ADK refers to the ability to:
  1. pause an invocation upon a long-running function call.
  2. resume an invocation from the last event, if it's paused or failed midway
  through.

  Note: ADK resumes the invocation in a best-effort manner:
  1. Tool call to resume needs to be idempotent because we only guarantee
  an at-least-once behavior once resumed.
  2. Any temporary / in-memory state will be lost upon resumption.
  """

  is_resumable: bool = False
  """Whether the app supports agent resumption.
  If enabled, the feature will be enabled for all agents in the app.
  """


@experimental
class EventsCompactionConfig(BaseModel):
  """The config of event compaction for an application."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )

  summarizer: Optional[BaseEventsSummarizer] = None
  """The event summarizer to use for compaction."""

  compaction_interval: int
  """The number of *new* user-initiated invocations that, once
  fully represented in the session's events, will trigger a compaction."""

  overlap_size: int
  """The number of preceding invocations to include from the
  end of the last compacted range. This creates an overlap between consecutive
  compacted summaries, maintaining context."""

  token_threshold: Optional[int] = Field(
      default=None,
      gt=0,
  )
  """Post-invocation token threshold trigger.

  If set, ADK will attempt a post-invocation compaction when the most recently
  observed prompt token count meets or exceeds this threshold.
  """

  event_retention_size: Optional[int] = Field(default=None, ge=0)
  """Post-invocation raw event retention size.

  If token-based post-invocation compaction is triggered, this keeps the last N
  raw events un-compacted.
  """

  @model_validator(mode="after")
  def _validate_token_params(self) -> EventsCompactionConfig:
    token_threshold_set = self.token_threshold is not None
    retention_size_set = self.event_retention_size is not None
    if token_threshold_set != retention_size_set:
      raise ValueError(
          "token_threshold and event_retention_size must be set together."
      )
    return self

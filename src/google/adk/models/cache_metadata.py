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

import time
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CacheMetadata(BaseModel):
  """Metadata for context cache associated with LLM responses.

  This class stores cache identification, usage tracking, and lifecycle
  information for a particular cache instance. It can be in two states:

  1. Active cache state: cache_name is set, all fields populated
  2. Fingerprint-only state: cache_name is None, only fingerprint and
     contents_count are set for prefix matching

  Token counts (cached and total) are available in the LlmResponse.usage_metadata
  and should be accessed from there to avoid duplication.

  Attributes:
      cache_name: The full resource name of the cached content (e.g.,
          'projects/123/locations/us-central1/cachedContents/456').
          None when no active cache exists (fingerprint-only state).
      expire_time: Unix timestamp when the cache expires. None when no
          active cache exists.
      fingerprint: Hash of cacheable contents (instruction + tools + contents).
          Always present for prefix matching.
      invocations_used: Number of invocations this cache has been used for.
          None when no active cache exists.
      contents_count: Number of contents. When active cache exists, this is
          the count of cached contents. When no active cache exists, this is
          the total count of contents in the request.
      created_at: Unix timestamp when the cache was created. None when
          no active cache exists.
  """

  model_config = ConfigDict(
      extra="forbid",
      frozen=True,  # Cache metadata should be immutable
  )

  cache_name: Optional[str] = Field(
      default=None,
      description=(
          "Full resource name of the cached content (None if no active cache)"
      ),
  )

  expire_time: Optional[float] = Field(
      default=None,
      description="Unix timestamp when cache expires (None if no active cache)",
  )

  fingerprint: str = Field(
      description="Hash of cacheable contents used to detect changes"
  )

  invocations_used: Optional[int] = Field(
      default=None,
      ge=0,
      description=(
          "Number of invocations this cache has been used for (None if no"
          " active cache)"
      ),
  )

  contents_count: int = Field(
      ge=0,
      description=(
          "Number of contents (cached contents when active cache exists, "
          "total contents in request when no active cache)"
      ),
  )

  created_at: Optional[float] = Field(
      default=None,
      description=(
          "Unix timestamp when cache was created (None if no active cache)"
      ),
  )

  @property
  def expire_soon(self) -> bool:
    """Check if the cache will expire soon (with 2-minute buffer)."""
    if self.expire_time is None:
      return False
    buffer_seconds = 120  # 2 minutes buffer for processing time
    return time.time() > (self.expire_time - buffer_seconds)

  def __str__(self) -> str:
    """String representation for logging and debugging."""
    if self.cache_name is None:
      return (
          f"Fingerprint-only: {self.contents_count} contents, "
          f"fingerprint={self.fingerprint[:8]}..."
      )
    cache_id = self.cache_name.split("/")[-1]
    time_until_expiry_minutes = (self.expire_time - time.time()) / 60
    return (
        f"Cache {cache_id}: used {self.invocations_used} invocations, "
        f"cached {self.contents_count} contents, "
        f"expires in {time_until_expiry_minutes:.1f}min"
    )

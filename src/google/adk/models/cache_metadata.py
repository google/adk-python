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
  information for a particular cache instance.

  Token counts (cached and total) are available in the LlmResponse.usage_metadata
  and should be accessed from there to avoid duplication.

  Attributes:
      cache_name: The full resource name of the cached content (e.g.,
          'projects/123/locations/us-central1/cachedContents/456')
      expire_time: Unix timestamp when the cache expires
      fingerprint: Hash of agent configuration (instruction + tools + model)
      invocations_used: Number of invocations this cache has been used for
      cached_contents_count: Number of contents stored in this cache
      created_at: Unix timestamp when the cache was created
  """

  model_config = ConfigDict(
      extra="forbid",
      frozen=True,  # Cache metadata should be immutable
  )

  cache_name: str = Field(
      description="Full resource name of the cached content"
  )

  expire_time: float = Field(description="Unix timestamp when cache expires")

  fingerprint: str = Field(
      description="Hash of agent configuration used to detect changes"
  )

  invocations_used: int = Field(
      ge=0,
      description="Number of invocations this cache has been used for",
  )

  cached_contents_count: int = Field(
      ge=0,
      description="Number of contents stored in this cache",
  )

  created_at: Optional[float] = Field(
      default=None,
      description=(
          "Unix timestamp when cache was created (None if reused existing)"
      ),
  )

  @property
  def expire_soon(self) -> bool:
    """Check if the cache will expire soon (with 2-minute buffer)."""
    buffer_seconds = 120  # 2 minutes buffer for processing time
    return time.time() > (self.expire_time - buffer_seconds)

  def __str__(self) -> str:
    """String representation for logging and debugging."""
    cache_id = self.cache_name.split("/")[-1]
    time_until_expiry_minutes = (self.expire_time - time.time()) / 60
    return (
        f"Cache {cache_id}: used {self.invocations_used} invocations, "
        f"cached {self.cached_contents_count} contents, "
        f"expires in {time_until_expiry_minutes:.1f}min"
    )

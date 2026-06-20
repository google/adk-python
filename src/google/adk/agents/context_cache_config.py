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

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..features import experimental
from ..features import FeatureName


@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
  """Configuration for context caching across all agents in an app.

  This configuration enables and controls context caching behavior for
  all LLM agents in an app. When this config is present on an app, context
  caching is enabled for all agents. When absent (None), context caching
  is disabled.

  Context caching can significantly reduce costs and improve response times
  by reusing previously processed context across multiple requests.

  Caching begins on the second turn of a session at the earliest and requires
  the prior request to reach Gemini's hard 4096-token minimum, so short or
  single-turn sessions are never cached.

  Attributes:
      cache_intervals: Maximum number of invocations to reuse the same cache before refreshing it
      ttl_seconds: Time-to-live for cache in seconds
      min_tokens: Minimum prior-request tokens required to enable caching
  """

  model_config = ConfigDict(
      extra="forbid",
  )

  cache_intervals: int = Field(
      default=10,
      ge=1,
      le=100,
      description=(
          "Maximum number of invocations to reuse the same cache before"
          " refreshing it"
      ),
  )

  ttl_seconds: int = Field(
      default=1800,  # 30 minutes
      gt=0,
      description="Time-to-live for cache in seconds",
  )

  min_tokens: int = Field(
      default=0,
      ge=0,
      description=(
          "Minimum prior-request tokens required to enable caching. This gates"
          " on the previous request's actual prompt token count, not an"
          " estimate of the current request. Gemini enforces a hard 4096-token"
          " minimum that always applies, so values below 4096 have no"
          " additional effect. No cache is created on the first request of a"
          " session; caching begins on the second turn once a previous token"
          " count is known. Set higher to avoid caching small requests where"
          " storage overhead may exceed benefits."
      ),
  )

  create_http_options: types.HttpOptions | None = Field(
      default=None,
      description=(
          "Optional HTTP options to pass to the GenAI client. Set this to add a"
          " timeout on CachedContent.create() calls (e.g."
          " types.HttpOptions(timeout=10000) for a 10-second timeout in"
          " milliseconds). When the cache creation call exceeds the timeout,"
          " it fails and the request proceeds without caching. None uses the"
          " client's default HTTP options."
      ),
  )

  @property
  def ttl_string(self) -> str:
    """Get TTL as string format for cache creation."""
    return f"{self.ttl_seconds}s"

  def __str__(self) -> str:
    """String representation for logging."""
    return (
        f"ContextCacheConfig(cache_intervals={self.cache_intervals}, "
        f"ttl={self.ttl_seconds}s, min_tokens={self.min_tokens}, "
        f"create_http_options={self.create_http_options})"
    )

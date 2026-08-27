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

"""Shared reading of ContextCacheConfig for models that cache a marked prefix.

Gemini caches by creating a server-side resource, which
``GeminiContextCacheManager`` owns. Claude instead caches whatever prefix the
request marks, and a model reached through LiteLLM inherits whichever of the
two its provider implements. The parts of the configuration that mean the same
thing for every prefix-marking model live here, so those callers cannot drift
apart on what one configuration means.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.context_cache_config import ContextCacheConfig
  from .llm_request import LlmRequest

logger = logging.getLogger("google_adk." + __name__)

# The longest prefix cache a prefix-marking model offers is an hour, and it
# costs more to write than the short-lived default. Only a configured lifetime
# of at least an hour is worth that price.
_ONE_HOUR_TTL_SECONDS = 3600


def resolve_cache_config(
    llm_request: LlmRequest,
) -> ContextCacheConfig | None:
  """Returns the cache config governing this request, or None to not cache.

  Args:
    llm_request: Request whose cache configuration is being resolved.

  Returns:
    The cache config to honor, or None when the request should not be cached.
  """
  cache_config = llm_request.cache_config
  if cache_config is None:
    return None

  # ``min_tokens`` gates on the previous turn's measured prompt size, the same
  # signal the Gemini path uses. That size is unknown on the first turn, where
  # marking a prefix costs nothing beyond writing the cache.
  previous_prompt_tokens = llm_request.cacheable_contents_token_count
  if (
      previous_prompt_tokens is not None
      and previous_prompt_tokens < cache_config.min_tokens
  ):
    logger.debug(
        "Skipping cache breakpoints: the previous prompt of %d tokens is below"
        " the configured minimum of %d.",
        previous_prompt_tokens,
        cache_config.min_tokens,
    )
    return None

  return cache_config


def use_one_hour_ttl(cache_config: ContextCacheConfig) -> bool:
  """Reports whether to ask for the hour-long cache instead of the default.

  An hour is the longest a prefix cache is kept, so a configured lifetime
  beyond that gets an hour rather than what it asked for.
  """
  return cache_config.ttl_seconds >= _ONE_HOUR_TTL_SECONDS

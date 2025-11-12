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

"""Manages context cache lifecycle for Gemini models."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from google.genai import Client
from google.genai import types

from ..utils.feature_decorator import experimental
from .cache_metadata import CacheMetadata
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)


@experimental
class GeminiContextCacheManager:
  """Manages context cache lifecycle for Gemini models.

  This manager handles cache creation, validation, cleanup, and metadata
  population for Gemini context caching. It uses content hashing to determine
  cache compatibility and implements efficient caching strategies.
  """

  def __init__(self, genai_client: Client):
    """Initialize cache manager with shared client.

    Args:
        genai_client: The GenAI client to use for cache operations.
    """
    self.genai_client = genai_client

  async def handle_context_caching(
      self, llm_request: LlmRequest
  ) -> Optional[CacheMetadata]:
    """Handle context caching for Gemini models.

    Validates existing cache or creates a new one if needed. Applies
    the cache to the request by setting cached_content and removing cached
    contents from the request.

    Args:
        llm_request: Request that may contain cache config and metadata.
                    Modified in-place to use the cache.

    Returns:
        Cache metadata to be included in response, or None if caching failed
    """
    # Check if we have existing cache metadata and if it's valid
    if llm_request.cache_metadata:
      logger.debug(
          "Found existing cache metadata: %s",
          llm_request.cache_metadata,
      )
      if await self._is_cache_valid(llm_request):
        # Valid cache found - use it
        logger.debug(
            "Cache is valid, reusing cache: %s",
            llm_request.cache_metadata.cache_name,
        )
        cache_name = llm_request.cache_metadata.cache_name
        cache_contents_count = llm_request.cache_metadata.contents_count
        self._apply_cache_to_request(
            llm_request, cache_name, cache_contents_count
        )
        return llm_request.cache_metadata.model_copy()
      else:
        # Invalid cache - clean it up and check if we should create new one
        old_cache_metadata = llm_request.cache_metadata

        # Only cleanup if there's an active cache
        if old_cache_metadata.cache_name is not None:
          logger.debug(
              "Cache is invalid, cleaning up: %s",
              old_cache_metadata.cache_name,
          )
          await self.cleanup_cache(old_cache_metadata.cache_name)

        # Calculate current fingerprint using contents count from old metadata
        cache_contents_count = old_cache_metadata.contents_count
        current_fingerprint = self._generate_cache_fingerprint(
            llm_request, cache_contents_count
        )

        # If fingerprints match, create new cache (expired but same content)
        if current_fingerprint == old_cache_metadata.fingerprint:
          logger.debug(
              "Fingerprints match after invalidation, creating new cache"
          )
          cache_metadata = await self._create_new_cache_with_contents(
              llm_request, cache_contents_count
          )
          if cache_metadata:
            self._apply_cache_to_request(
                llm_request, cache_metadata.cache_name, cache_contents_count
            )
            return cache_metadata

        # Fingerprints don't match - recalculate with total contents
        logger.debug(
            "Fingerprints don't match, returning fingerprint-only metadata"
        )
        total_contents_count = len(llm_request.contents)
        fingerprint_for_all = self._generate_cache_fingerprint(
            llm_request, total_contents_count
        )
        return CacheMetadata(
            fingerprint=fingerprint_for_all,
            contents_count=total_contents_count,
        )

    # No existing cache metadata - return fingerprint-only metadata
    # We don't create cache without previous fingerprint to match
    logger.debug(
        "No existing cache metadata, creating fingerprint-only metadata"
    )
    total_contents_count = len(llm_request.contents)
    fingerprint = self._generate_cache_fingerprint(
        llm_request, total_contents_count
    )
    return CacheMetadata(
        fingerprint=fingerprint,
        contents_count=total_contents_count,
    )

  def _find_count_of_contents_to_cache(
      self, contents: list[types.Content]
  ) -> int:
    """Find the number of contents to cache based on user content strategy.

    Strategy: Find the last continuous batch of user contents and cache
    all contents before them.

    Args:
        contents: List of contents from the LLM request

    Returns:
        Number of contents to cache (can be 0 if all contents are user contents)
    """
    if not contents:
      return 0

    # Find the last continuous batch of user contents
    last_user_batch_start = len(contents)

    # Scan backwards to find the start of the last user content batch
    for i in range(len(contents) - 1, -1, -1):
      if contents[i].role == "user":
        last_user_batch_start = i
      else:
        # Found non-user content, stop the batch
        break

    # Cache all contents before the last user batch
    # This ensures we always have some user content to send to the API
    return last_user_batch_start

  async def _is_cache_valid(self, llm_request: LlmRequest) -> bool:
    """Check if the cache from request metadata is still valid.

    Validates that it's an active cache (not fingerprint-only), checks expiry,
    cache intervals, and fingerprint compatibility.

    Args:
        llm_request: Request containing cache metadata to validate

    Returns:
        True if cache is valid, False otherwise
    """
    cache_metadata = llm_request.cache_metadata
    if not cache_metadata:
      return False

    # Fingerprint-only metadata is not a valid active cache
    if cache_metadata.cache_name is None:
      return False

    # Check if cache has expired
    if time.time() >= cache_metadata.expire_time:
      logger.info("Cache expired: %s", cache_metadata.cache_name)
      return False

    # Check if cache has been used for too many invocations
    if (
        cache_metadata.invocations_used
        > llm_request.cache_config.cache_intervals
    ):
      logger.info(
          "Cache exceeded cache intervals: %s (%d > %d intervals)",
          cache_metadata.cache_name,
          cache_metadata.invocations_used,
          llm_request.cache_config.cache_intervals,
      )
      return False

    # Check if fingerprint matches using cached contents count
    current_fingerprint = self._generate_cache_fingerprint(
        llm_request, cache_metadata.contents_count
    )
    if current_fingerprint != cache_metadata.fingerprint:
      logger.debug("Cache content fingerprint mismatch")
      return False

    return True

  def _generate_cache_fingerprint(
      self, llm_request: LlmRequest, cache_contents_count: int
  ) -> str:
    """Generate a fingerprint for cache validation.

    Includes system instruction, tools, tool_config, and first N contents.

    Args:
        llm_request: Request to generate fingerprint for
        cache_contents_count: Number of contents to include in fingerprint

    Returns:
        16-character hexadecimal fingerprint representing the cached state
    """
    # Create fingerprint from system instruction, tools, tool_config, and first N contents
    fingerprint_data = {}

    if llm_request.config and llm_request.config.system_instruction:
      fingerprint_data["system_instruction"] = (
          llm_request.config.system_instruction
      )

    if llm_request.config and llm_request.config.tools:
      # Simplified: just dump types.Tool instances to JSON
      tools_data = []
      for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool):
          tools_data.append(tool.model_dump())
      fingerprint_data["tools"] = tools_data

    if llm_request.config and llm_request.config.tool_config:
      fingerprint_data["tool_config"] = (
          llm_request.config.tool_config.model_dump()
      )

    # Include first N contents in fingerprint
    if cache_contents_count > 0 and llm_request.contents:
      contents_data = []
      for i in range(min(cache_contents_count, len(llm_request.contents))):
        content = llm_request.contents[i]
        contents_data.append(content.model_dump())
      fingerprint_data["cached_contents"] = contents_data

    # Generate hash using str() instead of json.dumps() to handle bytes
    fingerprint_str = str(fingerprint_data)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

  async def _create_new_cache_with_contents(
      self, llm_request: LlmRequest, cache_contents_count: int
  ) -> Optional[CacheMetadata]:
    """Create a new cache with specified number of contents.

    Args:
        llm_request: Request to create cache for
        cache_contents_count: Number of contents to include in cache

    Returns:
        Cache metadata if successful, None otherwise
    """
    # Check if we have token count from previous response for cache size validation
    if llm_request.cacheable_contents_token_count is None:
      logger.info(
          "No previous token count available, skipping cache creation for"
          " initial request"
      )
      return None

    if (
        llm_request.cacheable_contents_token_count
        < llm_request.cache_config.min_tokens
    ):
      logger.info(
          "Previous request too small for caching (%d < %d tokens)",
          llm_request.cacheable_contents_token_count,
          llm_request.cache_config.min_tokens,
      )
      return None

    try:
      # Create cache using Gemini API directly
      return await self._create_gemini_cache(llm_request, cache_contents_count)
    except Exception as e:
      logger.warning("Failed to create cache: %s", e)
      return None

  def _estimate_request_tokens(self, llm_request: LlmRequest) -> int:
    """Estimate token count for the request.

    This is a rough estimation based on content text length.

    Args:
        llm_request: Request to estimate tokens for

    Returns:
        Estimated token count
    """
    total_chars = 0

    # System instruction
    if llm_request.config and llm_request.config.system_instruction:
      total_chars += len(llm_request.config.system_instruction)

    # Tools
    if llm_request.config and llm_request.config.tools:
      for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool):
          tool_str = json.dumps(tool.model_dump())
          total_chars += len(tool_str)

    # Contents
    for content in llm_request.contents:
      for part in content.parts:
        if part.text:
          total_chars += len(part.text)

    # Rough estimate: 4 characters per token
    return total_chars // 4

  async def _create_gemini_cache(
      self, llm_request: LlmRequest, cache_contents_count: int
  ) -> CacheMetadata:
    """Create cache using Gemini API.

    Args:
        llm_request: Request to create cache for
        cache_contents_count: Number of contents to cache

    Returns:
        Cache metadata with precise creation timestamp
    """
    from ..telemetry.tracing import tracer

    with tracer.start_as_current_span("create_cache") as span:
      # Prepare cache contents (first N contents + system instruction + tools)
      cache_contents = llm_request.contents[:cache_contents_count]

      cache_config = types.CreateCachedContentConfig(
          contents=cache_contents,
          ttl=llm_request.cache_config.ttl_string,
          display_name=(
              f"adk-cache-{int(time.time())}-{cache_contents_count}contents"
          ),
      )

      # Add system instruction if present
      if llm_request.config and llm_request.config.system_instruction:
        cache_config.system_instruction = llm_request.config.system_instruction
        logger.debug(
            "Added system instruction to cache config (length=%d)",
            len(llm_request.config.system_instruction),
        )

      # Add tools if present
      if llm_request.config and llm_request.config.tools:
        cache_config.tools = llm_request.config.tools

      # Add tool config if present
      if llm_request.config and llm_request.config.tool_config:
        cache_config.tool_config = llm_request.config.tool_config

      span.set_attribute("cache_contents_count", cache_contents_count)
      span.set_attribute("model", llm_request.model)
      span.set_attribute("ttl_seconds", llm_request.cache_config.ttl_seconds)

      logger.debug(
          "Creating cache with model %s and config: %s",
          llm_request.model,
          cache_config,
      )
      cached_content = await self.genai_client.aio.caches.create(
          model=llm_request.model,
          config=cache_config,
      )
      # Set precise creation timestamp right after cache creation
      created_at = time.time()
      logger.info("Cache created successfully: %s", cached_content.name)

      span.set_attribute("cache_name", cached_content.name)

      # Return complete cache metadata with precise timing
      return CacheMetadata(
          cache_name=cached_content.name,
          expire_time=created_at + llm_request.cache_config.ttl_seconds,
          fingerprint=self._generate_cache_fingerprint(
              llm_request, cache_contents_count
          ),
          invocations_used=1,
          contents_count=cache_contents_count,
          created_at=created_at,
      )

  async def cleanup_cache(self, cache_name: str) -> None:
    """Clean up cache by deleting it.

    Args:
        cache_name: Name of cache to delete
    """
    logger.debug("Attempting to delete cache: %s", cache_name)
    try:
      await self.genai_client.aio.caches.delete(name=cache_name)
      logger.info("Cache cleaned up: %s", cache_name)
    except Exception as e:
      logger.warning("Failed to cleanup cache %s: %s", cache_name, e)

  def _apply_cache_to_request(
      self,
      llm_request: LlmRequest,
      cache_name: str,
      cache_contents_count: int,
  ) -> None:
    """Apply cache to the request by modifying it to use cached content.

    Args:
        llm_request: Request to modify
        cache_name: Name of cache to use
        cache_contents_count: Number of contents that are cached
    """
    # Remove system instruction, tools, and tool config from request config since they're in cache
    if llm_request.config:
      llm_request.config.system_instruction = None
      llm_request.config.tools = None
      llm_request.config.tool_config = None

    # Set cached content reference
    llm_request.config.cached_content = cache_name

    # Remove cached contents from the request (keep only uncached contents)
    llm_request.contents = llm_request.contents[cache_contents_count:]

  def populate_cache_metadata_in_response(
      self, llm_response: LlmResponse, cache_metadata: CacheMetadata
  ) -> None:
    """Populate cache metadata in LLM response.

    Args:
        llm_response: Response to populate metadata in
        cache_metadata: Cache metadata to copy into response
    """
    # Create a copy of cache metadata for the response
    llm_response.cache_metadata = cache_metadata.model_copy()

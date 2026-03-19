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

"""Tests for GeminiContextCacheManager."""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.gemini_context_cache_manager import GeminiContextCacheManager
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import Client
from google.genai import types
import pytest


class TestGeminiContextCacheManager:
  """Test suite for GeminiContextCacheManager."""

  def setup_method(self):
    """Set up test fixtures."""
    mock_client = AsyncMock(spec=Client)
    self.manager = GeminiContextCacheManager(mock_client)
    self.cache_config = ContextCacheConfig(
        cache_intervals=10,
        ttl_seconds=1800,
        min_tokens=0,  # Allow caching for tests
    )

  def create_llm_request(self, cache_metadata=None, contents_count=3):
    """Helper to create test LlmRequest."""
    contents = []
    for i in range(contents_count):
      contents.append(
          types.Content(
              role="user", parts=[types.Part(text=f"Test message {i}")]
          )
      )

    # Create tools for testing fingerprinting
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="test_tool",
                    description="A test tool",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "param": types.Schema(type=types.Type.STRING)
                        },
                    ),
                )
            ]
        )
    ]

    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
    )

    return LlmRequest(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction="Test instruction",
            tools=tools,
            tool_config=tool_config,
        ),
        cache_config=self.cache_config,
        cache_metadata=cache_metadata,
    )

  def create_cache_metadata(
      self, invocations_used=0, expired=False, contents_count=3
  ):
    """Helper to create test CacheMetadata."""
    current_time = time.time()
    expire_time = current_time - 300 if expired else current_time + 1800

    return CacheMetadata(
        cache_name="projects/test/locations/us-central1/cachedContents/test123",
        expire_time=expire_time,
        fingerprint="test_fingerprint",
        invocations_used=invocations_used,
        contents_count=contents_count,
        created_at=current_time - 600,
    )

  def test_init(self):
    """Test manager initialization."""
    mock_client = MagicMock(spec=Client)
    manager = GeminiContextCacheManager(mock_client)
    assert manager is not None
    assert manager.genai_client == mock_client

  async def test_handle_context_caching_no_existing_cache(self):
    """Test handling context caching with no existing cache returns fingerprint-only metadata."""
    llm_request = self.create_llm_request(contents_count=5)

    with patch.object(
        self.manager, "_generate_cache_fingerprint", return_value="test_fp"
    ):
      result = await self.manager.handle_context_caching(llm_request)

    assert result is not None
    # Should return fingerprint-only metadata (no active cache)
    assert result.cache_name is None
    assert result.expire_time is None
    assert result.invocations_used is None
    assert result.created_at is None
    assert result.fingerprint == "test_fp"
    assert result.contents_count == 5  # Total contents count

    # No cache should be created
    self.manager.genai_client.aio.caches.create.assert_not_called()

  async def test_handle_context_caching_valid_existing_cache(self):
    """Test handling context caching with valid existing cache."""

    # Create request with existing valid cache
    existing_cache = self.create_cache_metadata(invocations_used=5)
    llm_request = self.create_llm_request(cache_metadata=existing_cache)

    with patch.object(self.manager, "_is_cache_valid", return_value=True):
      result = await self.manager.handle_context_caching(llm_request)

    assert result is not None
    # Verify that existing cache metadata is preserved (copied)
    assert result.cache_name == existing_cache.cache_name
    assert (
        result.invocations_used == existing_cache.invocations_used
    )  # Should preserve original invocations_used
    assert (
        result.expire_time == existing_cache.expire_time
    )  # Should preserve original expire_time
    assert (
        result.fingerprint == existing_cache.fingerprint
    )  # Should preserve original fingerprint
    assert (
        result.created_at == existing_cache.created_at
    )  # Should preserve original created_at

    # Verify it's a copy, not the same object
    assert result is not existing_cache

    # Should not create new cache
    self.manager.genai_client.aio.caches.create.assert_not_called()

  async def test_handle_context_caching_invalid_cache_fingerprint_match(self):
    """Test invalid cache with matching fingerprint creates new cache."""
    # Setup mocks
    mock_cached_content = AsyncMock()
    mock_cached_content.name = (
        "projects/test/locations/us-central1/cachedContents/new456"
    )
    self.manager.genai_client.aio.caches.create = AsyncMock(
        return_value=mock_cached_content
    )

    # Create request with invalid existing cache
    existing_cache = self.create_cache_metadata(
        invocations_used=15
    )  # Exceeds cache_intervals
    llm_request = self.create_llm_request(cache_metadata=existing_cache)
    llm_request.cacheable_contents_token_count = (
        2048  # Add token count for cache creation
    )

    with (
        patch.object(self.manager, "_is_cache_valid", return_value=False),
        patch.object(self.manager, "cleanup_cache") as mock_cleanup,
        patch.object(
            self.manager,
            "_generate_cache_fingerprint",
            return_value="test_fingerprint",  # Match old fingerprint
        ),
    ):

      result = await self.manager.handle_context_caching(llm_request)

    assert result is not None
    # Should create new cache when fingerprints match
    assert (
        result.cache_name
        == "projects/test/locations/us-central1/cachedContents/new456"
    )
    mock_cleanup.assert_called_once_with(existing_cache.cache_name)
    self.manager.genai_client.aio.caches.create.assert_called_once()

  async def test_handle_context_caching_invalid_cache_fingerprint_mismatch(
      self,
  ):
    """Test invalid cache with mismatched fingerprint returns fingerprint-only metadata."""
    # Create request with invalid existing cache
    existing_cache = self.create_cache_metadata(
        invocations_used=15, contents_count=3
    )  # Exceeds cache_intervals
    llm_request = self.create_llm_request(
        cache_metadata=existing_cache, contents_count=5
    )

    with (
        patch.object(self.manager, "_is_cache_valid", return_value=False),
        patch.object(self.manager, "cleanup_cache") as mock_cleanup,
        patch.object(
            self.manager,
            "_generate_cache_fingerprint",
            side_effect=["old_fp", "new_fp"],  # Different fingerprints
        ),
    ):

      result = await self.manager.handle_context_caching(llm_request)

    assert result is not None
    # Should return fingerprint-only metadata
    assert result.cache_name is None
    assert result.expire_time is None
    assert result.invocations_used is None
    assert result.created_at is None
    assert result.fingerprint == "new_fp"
    assert result.contents_count == 5  # Total contents count
    mock_cleanup.assert_called_once_with(existing_cache.cache_name)
    self.manager.genai_client.aio.caches.create.assert_not_called()

  async def test_is_cache_valid_fingerprint_mismatch(self):
    """Test cache validation with fingerprint mismatch."""
    cache_metadata = self.create_cache_metadata()
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)

    with patch.object(
        self.manager,
        "_generate_cache_fingerprint",
        return_value="different_fingerprint",
    ):
      result = await self.manager._is_cache_valid(llm_request)

    assert result is False

  async def test_is_cache_valid_expired_cache(self):
    """Test cache validation with expired cache."""
    cache_metadata = self.create_cache_metadata(expired=True)
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)

    with patch.object(
        self.manager,
        "_generate_cache_fingerprint",
        return_value="test_fingerprint",
    ):
      result = await self.manager._is_cache_valid(llm_request)

    assert result is False

  async def test_is_cache_valid_fingerprint_only_metadata(self):
    """Test cache validation with fingerprint-only metadata (no active cache)."""
    # Create fingerprint-only metadata (cache_name is None)
    cache_metadata = CacheMetadata(
        fingerprint="test_fingerprint",
        contents_count=5,
    )
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)

    result = await self.manager._is_cache_valid(llm_request)

    assert (
        result is False
    )  # Fingerprint-only metadata is not a valid active cache

  async def test_is_cache_valid_cache_intervals_exceeded(self):
    """Test cache validation with max invocations exceeded."""
    cache_metadata = self.create_cache_metadata(
        invocations_used=15
    )  # Exceeds cache_intervals=10
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)

    with patch.object(
        self.manager,
        "_generate_cache_fingerprint",
        return_value="test_fingerprint",
    ):
      result = await self.manager._is_cache_valid(llm_request)

    assert result is False

  async def test_is_cache_valid_all_checks_pass(self):
    """Test cache validation when all checks pass."""
    cache_metadata = self.create_cache_metadata(
        invocations_used=5
    )  # Within cache_intervals=10
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)

    with patch.object(
        self.manager,
        "_generate_cache_fingerprint",
        return_value="test_fingerprint",
    ):
      result = await self.manager._is_cache_valid(llm_request)

    assert result is True

  async def test_cleanup_cache(self):
    """Test cache cleanup functionality."""
    cache_name = "projects/test/locations/us-central1/cachedContents/test123"

    await self.manager.cleanup_cache(cache_name)

    self.manager.genai_client.aio.caches.delete.assert_called_once_with(
        name=cache_name
    )

  def test_generate_cache_fingerprint(self):
    """Test cache fingerprint generation includes tools and tool_config."""
    llm_request = self.create_llm_request()
    cache_contents_count = 2  # Cache all but last content

    fingerprint1 = self.manager._generate_cache_fingerprint(
        llm_request, cache_contents_count
    )
    fingerprint2 = self.manager._generate_cache_fingerprint(
        llm_request, cache_contents_count
    )

    # Same request should generate same fingerprint
    assert fingerprint1 == fingerprint2
    assert isinstance(fingerprint1, str)
    assert len(fingerprint1) > 0

    # Test that tool_config and tools are included in fingerprint
    # Create request without tools/tool_config
    llm_request_no_tools = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Test")])],
        config=types.GenerateContentConfig(
            system_instruction="Test instruction"
        ),
        cache_config=self.cache_config,
    )

    fingerprint_no_tools = self.manager._generate_cache_fingerprint(
        llm_request_no_tools, cache_contents_count
    )

    # Should be different from request with tools
    assert fingerprint1 != fingerprint_no_tools

  def test_generate_cache_fingerprint_different_requests(self):
    """Test that different requests generate different fingerprints."""
    llm_request1 = self.create_llm_request()

    llm_request2 = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user", parts=[types.Part(text="Different message")]
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction="Different instruction"
        ),
        cache_config=self.cache_config,
    )

    cache_contents_count = 2
    fingerprint1 = self.manager._generate_cache_fingerprint(
        llm_request1, cache_contents_count
    )
    fingerprint2 = self.manager._generate_cache_fingerprint(
        llm_request2, cache_contents_count
    )

    assert fingerprint1 != fingerprint2

  def test_generate_cache_fingerprint_tool_config_variations(self):
    """Test that different tool configs generate different fingerprints."""
    # Request with AUTO mode
    llm_request_auto = self.create_llm_request()

    # Request with NONE mode
    tool_config_none = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="NONE")
    )

    llm_request_none = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Test")])],
        config=types.GenerateContentConfig(
            system_instruction="Test instruction",
            tools=llm_request_auto.config.tools,
            tool_config=tool_config_none,
        ),
        cache_config=self.cache_config,
    )

    cache_contents_count = 2
    fingerprint_auto = self.manager._generate_cache_fingerprint(
        llm_request_auto, cache_contents_count
    )
    fingerprint_none = self.manager._generate_cache_fingerprint(
        llm_request_none, cache_contents_count
    )

    assert fingerprint_auto != fingerprint_none

  async def test_populate_cache_metadata_in_response_no_invocations_increment(
      self,
  ):
    """Test that populate_cache_metadata_in_response doesn't increment invocations_used."""
    # Create mock response with usage metadata
    usage_metadata = MagicMock()
    usage_metadata.cached_content_token_count = 800
    usage_metadata.prompt_token_count = 1000

    llm_response = MagicMock(spec=LlmResponse)
    llm_response.usage_metadata = usage_metadata

    cache_metadata = self.create_cache_metadata(invocations_used=3)

    self.manager.populate_cache_metadata_in_response(
        llm_response, cache_metadata
    )

    # Verify response metadata preserves the original invocations_used (no increment)
    updated_metadata = llm_response.cache_metadata
    assert (
        updated_metadata.invocations_used == 3
    )  # Should preserve original value
    assert updated_metadata.cache_name == cache_metadata.cache_name
    assert updated_metadata.fingerprint == cache_metadata.fingerprint
    assert updated_metadata.expire_time == cache_metadata.expire_time
    assert updated_metadata.created_at == cache_metadata.created_at

  async def test_populate_cache_metadata_no_usage_metadata(self):
    """Test populating cache metadata when no usage metadata."""
    llm_response = MagicMock(spec=LlmResponse)
    llm_response.usage_metadata = None

    cache_metadata = self.create_cache_metadata(invocations_used=3)

    self.manager.populate_cache_metadata_in_response(
        llm_response, cache_metadata
    )

    # Should still create metadata even without usage info
    updated_metadata = llm_response.cache_metadata
    assert (
        updated_metadata.invocations_used == 3
    )  # Should preserve original value
    assert updated_metadata.cache_name == cache_metadata.cache_name

  async def test_create_new_cache_with_proper_ttl(self):
    """Test that new cache is created with proper TTL."""
    mock_cached_content = AsyncMock()
    mock_cached_content.name = (
        "projects/test/locations/us-central1/cachedContents/test123"
    )
    self.manager.genai_client.aio.caches.create = AsyncMock(
        return_value=mock_cached_content
    )

    llm_request = self.create_llm_request()

    cache_contents_count = max(0, len(llm_request.contents) - 1)

    with patch.object(
        self.manager, "_generate_cache_fingerprint", return_value="test_fp"
    ):
      await self.manager._create_gemini_cache(llm_request, cache_contents_count)

    # Verify cache creation call includes TTL
    create_call = self.manager.genai_client.aio.caches.create.call_args
    assert create_call is not None
    cache_config = create_call[1]["config"]
    assert cache_config.ttl == "1800s"  # From cache_config

  def test_all_but_last_content_caching(self):
    """Test that cache content counting works correctly."""
    # Test with multiple contents
    llm_request_multi = self.create_llm_request(contents_count=5)

    # Test cache contents count calculation
    cache_contents_count = max(0, len(llm_request_multi.contents) - 1)

    assert cache_contents_count == 4  # 5 contents, so cache 4 contents

    # Test with single content
    llm_request_single = self.create_llm_request(contents_count=1)
    single_cache_contents_count = max(0, len(llm_request_single.contents) - 1)

    assert single_cache_contents_count == 0  # Single content, cache 0 contents

  def test_edge_cases(self):
    """Test various edge cases."""
    # Test with None cache_config
    llm_request_no_config = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Test")])],
        config=types.GenerateContentConfig(system_instruction="Test"),
        cache_config=None,
    )

    # Should handle gracefully
    cache_contents_count = 2
    fingerprint = self.manager._generate_cache_fingerprint(
        llm_request_no_config, cache_contents_count
    )
    assert isinstance(fingerprint, str)

    # Test with empty contents
    llm_request_empty = LlmRequest(
        model="gemini-2.0-flash",
        contents=[],
        config=types.GenerateContentConfig(system_instruction="Test"),
        cache_config=self.cache_config,
    )

    empty_cache_contents_count = 0
    fingerprint = self.manager._generate_cache_fingerprint(
        llm_request_empty, empty_cache_contents_count
    )
    assert isinstance(fingerprint, str)

  def test_parameter_types_enforcement(self):
    """Test that method calls with correct parameter types work properly."""
    # Create proper objects
    usage_metadata = MagicMock()
    usage_metadata.cached_content_token_count = 500
    usage_metadata.prompt_token_count = 1000

    llm_response = MagicMock(spec=LlmResponse)
    llm_response.usage_metadata = usage_metadata

    cache_metadata = self.create_cache_metadata(invocations_used=3)

    # This should work fine (correct types and order)
    self.manager.populate_cache_metadata_in_response(
        llm_response, cache_metadata
    )
    updated_metadata = llm_response.cache_metadata
    assert updated_metadata.invocations_used == 3  # No increment in this method

    # Document expected types for integration tests
    assert isinstance(cache_metadata, CacheMetadata)
    assert hasattr(
        llm_response, "usage_metadata"
    )  # LlmResponse should have this
    assert not hasattr(
        cache_metadata, "usage_metadata"
    )  # CacheMetadata should NOT have this

  def create_llm_request_with_token_count(
      self, token_count=None, cache_metadata=None
  ):
    """Helper to create LlmRequest with cacheable_contents_token_count."""
    llm_request = self.create_llm_request(cache_metadata=cache_metadata)
    llm_request.cacheable_contents_token_count = token_count
    return llm_request

  async def test_cache_creation_with_sufficient_token_count(self):
    """Test that fingerprint-only metadata is returned even with sufficient tokens."""
    # With new prefix matching logic, no cache is created without existing metadata
    # Create request with sufficient token count
    llm_request = self.create_llm_request_with_token_count(token_count=2048)

    with patch.object(
        self.manager, "_generate_cache_fingerprint", return_value="test_fp"
    ):
      result = await self.manager.handle_context_caching(llm_request)

    # Should return fingerprint-only metadata (no cache creation)
    assert result is not None
    assert result.cache_name is None  # Fingerprint-only state
    assert result.fingerprint == "test_fp"
    assert result.contents_count == 3
    self.manager.genai_client.aio.caches.create.assert_not_called()

  async def test_cache_creation_with_insufficient_token_count(self):
    """Test that fingerprint-only metadata is returned even with insufficient tokens."""
    # Set higher minimum token requirement
    self.manager.cache_config = ContextCacheConfig(
        cache_intervals=10,
        ttl_seconds=1800,
        min_tokens=2048,
    )

    # Create request with insufficient token count
    llm_request = self.create_llm_request_with_token_count(token_count=1024)
    llm_request.cache_config = self.manager.cache_config

    with patch.object(
        self.manager, "_generate_cache_fingerprint", return_value="test_fp"
    ):
      result = await self.manager.handle_context_caching(llm_request)

    # Should return fingerprint-only metadata
    assert result is not None
    assert result.cache_name is None
    assert result.fingerprint == "test_fp"
    self.manager.genai_client.aio.caches.create.assert_not_called()

  async def test_cache_creation_without_token_count(self):
    """Test that fingerprint-only metadata is returned even without token count."""
    # Create request without token count (initial request)
    llm_request = self.create_llm_request_with_token_count(token_count=None)

    with patch.object(
        self.manager, "_generate_cache_fingerprint", return_value="test_fp"
    ):
      result = await self.manager.handle_context_caching(llm_request)

    # Should return fingerprint-only metadata
    assert result is not None
    assert result.cache_name is None
    assert result.fingerprint == "test_fp"
    self.manager.genai_client.aio.caches.create.assert_not_called()

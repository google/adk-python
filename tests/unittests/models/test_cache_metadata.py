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

"""Tests for CacheMetadata."""

import time

from google.adk.models.cache_metadata import CacheMetadata
from pydantic import ValidationError
import pytest


class TestCacheMetadata:
  """Test suite for CacheMetadata."""

  def test_required_fields(self):
    """Test that all required fields must be provided."""
    # Valid creation with all required fields
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=5,
        contents_count=3,
    )

    assert (
        metadata.cache_name
        == "projects/123/locations/us-central1/cachedContents/456"
    )
    assert metadata.expire_time > time.time()
    assert metadata.fingerprint == "abc123"
    assert metadata.invocations_used == 5
    assert metadata.contents_count == 3
    assert metadata.created_at is None  # Optional field

  def test_optional_created_at(self):
    """Test that created_at is optional."""
    current_time = time.time()

    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=3,
        contents_count=2,
        created_at=current_time,
    )

    assert metadata.created_at == current_time

  def test_invocations_used_validation(self):
    """Test invocations_used validation constraints."""
    # Valid: zero or positive
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=0,
        contents_count=1,
    )
    assert metadata.invocations_used == 0

    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=10,
        contents_count=1,
    )
    assert metadata.invocations_used == 10

    # Invalid: negative
    with pytest.raises(ValidationError) as exc_info:
      CacheMetadata(
          cache_name="projects/123/locations/us-central1/cachedContents/456",
          expire_time=time.time() + 1800,
          fingerprint="abc123",
          invocations_used=-1,
          contents_count=1,
      )
    assert "greater than or equal to 0" in str(exc_info.value)

  def test_contents_count_validation(self):
    """Test contents_count validation constraints."""
    # Valid: zero or positive
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=1,
        contents_count=0,
    )
    assert metadata.contents_count == 0

    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=1,
        contents_count=10,
    )
    assert metadata.contents_count == 10

    # Invalid: negative
    with pytest.raises(ValidationError) as exc_info:
      CacheMetadata(
          cache_name="projects/123/locations/us-central1/cachedContents/456",
          expire_time=time.time() + 1800,
          fingerprint="abc123",
          invocations_used=1,
          contents_count=-1,
      )
    assert "greater than or equal to 0" in str(exc_info.value)

  def test_expire_soon_property(self):
    """Test expire_soon property."""
    # Cache that expires in 10 minutes (should not expire soon)
    future_time = time.time() + 600  # 10 minutes
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=future_time,
        fingerprint="abc123",
        invocations_used=1,
        contents_count=1,
    )
    assert not metadata.expire_soon

    # Cache that expires in 1 minute (should expire soon)
    soon_time = time.time() + 60  # 1 minute
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=soon_time,
        fingerprint="abc123",
        invocations_used=1,
        contents_count=1,
    )
    assert metadata.expire_soon

  def test_str_representation(self):
    """Test string representation."""
    current_time = time.time()
    expire_time = current_time + 1800  # 30 minutes

    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/test456",
        expire_time=expire_time,
        fingerprint="abc123",
        invocations_used=7,
        contents_count=4,
    )

    str_repr = str(metadata)
    assert "test456" in str_repr  # Cache ID
    assert "used 7 invocations" in str_repr
    assert "cached 4 contents" in str_repr
    assert "expires in" in str_repr

  def test_immutability(self):
    """Test that CacheMetadata is immutable (frozen)."""
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=5,
        contents_count=3,
    )

    # Should not be able to modify fields
    with pytest.raises(ValidationError):
      metadata.invocations_used = 10

  def test_model_config(self):
    """Test that model config is set correctly."""
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=5,
        contents_count=3,
    )

    assert metadata.model_config["extra"] == "forbid"
    assert metadata.model_config["frozen"] == True

  def test_field_descriptions(self):
    """Test that fields have proper descriptions."""
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=5,
        contents_count=3,
    )
    schema = metadata.model_json_schema()

    assert "invocations_used" in schema["properties"]
    assert (
        "Number of invocations"
        in schema["properties"]["invocations_used"]["description"]
    )

    assert "contents_count" in schema["properties"]
    assert (
        "Number of contents"
        in schema["properties"]["contents_count"]["description"]
    )

  def test_realistic_cache_scenarios(self):
    """Test realistic cache scenarios."""
    current_time = time.time()

    # Fresh cache
    fresh_cache = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/fresh123",
        expire_time=current_time + 1800,
        fingerprint="fresh_fingerprint",
        invocations_used=1,
        contents_count=5,
        created_at=current_time,
    )
    assert fresh_cache.invocations_used == 1
    assert not fresh_cache.expire_soon

    # Well-used cache
    used_cache = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/used456",
        expire_time=current_time + 600,
        fingerprint="used_fingerprint",
        invocations_used=8,
        contents_count=3,
        created_at=current_time - 1200,
    )
    assert used_cache.invocations_used == 8

    # Expiring cache
    expiring_cache = CacheMetadata(
        cache_name=(
            "projects/123/locations/us-central1/cachedContents/expiring789"
        ),
        expire_time=current_time + 60,  # 1 minute
        fingerprint="expiring_fingerprint",
        invocations_used=15,
        contents_count=10,
    )
    assert expiring_cache.expire_soon

  def test_cache_name_extraction(self):
    """Test cache name ID extraction in string representation."""
    metadata = CacheMetadata(
        cache_name=(
            "projects/123/locations/us-central1/cachedContents/extracted_id"
        ),
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=1,
        contents_count=2,
    )

    str_repr = str(metadata)
    assert "extracted_id" in str_repr

  def test_no_performance_metrics(self):
    """Test that performance metrics are not in CacheMetadata."""
    metadata = CacheMetadata(
        cache_name="projects/123/locations/us-central1/cachedContents/456",
        expire_time=time.time() + 1800,
        fingerprint="abc123",
        invocations_used=5,
        contents_count=3,
    )

    # Verify that token counts are NOT in CacheMetadata
    # (they should be in LlmResponse.usage_metadata)
    assert not hasattr(metadata, "cached_tokens")
    assert not hasattr(metadata, "total_tokens")
    assert not hasattr(metadata, "prompt_tokens")

  def test_missing_required_fields(self):
    """Test validation when truly required fields are missing."""
    # Only fingerprint and contents_count are required now
    # Other fields are optional (for fingerprint-only state)
    required_fields = [
        "fingerprint",
        "contents_count",
    ]

    base_args = {
        "fingerprint": "abc123",
        "contents_count": 2,
    }

    for field in required_fields:
      args = base_args.copy()
      del args[field]

      with pytest.raises(ValidationError):
        CacheMetadata(**args)

    # Test that optional fields can be omitted (fingerprint-only state)
    metadata = CacheMetadata(
        fingerprint="abc123",
        contents_count=5,
    )
    assert metadata.cache_name is None
    assert metadata.expire_time is None
    assert metadata.invocations_used is None
    assert metadata.created_at is None

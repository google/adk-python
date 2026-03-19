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

"""Tests for ContextCacheConfig."""

from google.adk.agents.context_cache_config import ContextCacheConfig
from pydantic import ValidationError
import pytest


class TestContextCacheConfig:
  """Test suite for ContextCacheConfig."""

  def test_default_values(self):
    """Test that default values are set correctly."""
    config = ContextCacheConfig()

    assert config.cache_intervals == 10
    assert config.ttl_seconds == 1800  # 30 minutes
    assert config.min_tokens == 0

  def test_custom_values(self):
    """Test creating config with custom values."""
    config = ContextCacheConfig(
        cache_intervals=15, ttl_seconds=3600, min_tokens=1024
    )

    assert config.cache_intervals == 15
    assert config.ttl_seconds == 3600
    assert config.min_tokens == 1024

  def test_cache_intervals_validation(self):
    """Test cache_intervals validation constraints."""
    # Valid range
    config = ContextCacheConfig(cache_intervals=1)
    assert config.cache_intervals == 1

    config = ContextCacheConfig(cache_intervals=100)
    assert config.cache_intervals == 100

    # Invalid: too low
    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(cache_intervals=0)
    assert "greater than or equal to 1" in str(exc_info.value)

    # Invalid: too high
    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(cache_intervals=101)
    assert "less than or equal to 100" in str(exc_info.value)

  def test_ttl_seconds_validation(self):
    """Test ttl_seconds validation constraints."""
    # Valid range
    config = ContextCacheConfig(ttl_seconds=1)
    assert config.ttl_seconds == 1

    config = ContextCacheConfig(ttl_seconds=86400)  # 24 hours
    assert config.ttl_seconds == 86400

    # Invalid: zero or negative
    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(ttl_seconds=0)
    assert "greater than 0" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(ttl_seconds=-1)
    assert "greater than 0" in str(exc_info.value)

  def test_min_tokens_validation(self):
    """Test min_tokens validation constraints."""
    # Valid values
    config = ContextCacheConfig(min_tokens=0)
    assert config.min_tokens == 0

    config = ContextCacheConfig(min_tokens=1024)
    assert config.min_tokens == 1024

    # Invalid: negative
    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(min_tokens=-1)
    assert "greater than or equal to 0" in str(exc_info.value)

  def test_ttl_string_property(self):
    """Test ttl_string property returns correct format."""
    config = ContextCacheConfig(ttl_seconds=1800)
    assert config.ttl_string == "1800s"

    config = ContextCacheConfig(ttl_seconds=3600)
    assert config.ttl_string == "3600s"

  def test_str_representation(self):
    """Test string representation for logging."""
    config = ContextCacheConfig(
        cache_intervals=15, ttl_seconds=3600, min_tokens=1024
    )

    expected = (
        "ContextCacheConfig(cache_intervals=15, ttl=3600s, min_tokens=1024)"
    )
    assert str(config) == expected

  def test_str_representation_defaults(self):
    """Test string representation with default values."""
    config = ContextCacheConfig()

    expected = "ContextCacheConfig(cache_intervals=10, ttl=1800s, min_tokens=0)"
    assert str(config) == expected

  def test_pydantic_model_validation(self):
    """Test that Pydantic model validation works correctly."""
    # Test extra fields are forbidden
    with pytest.raises(ValidationError) as exc_info:
      ContextCacheConfig(cache_intervals=10, extra_field="not_allowed")
    assert "extra" in str(exc_info.value).lower()

  def test_field_descriptions(self):
    """Test that fields have proper descriptions."""
    config = ContextCacheConfig()
    schema = config.model_json_schema()

    assert "cache_intervals" in schema["properties"]
    assert (
        "Maximum number of invocations"
        in schema["properties"]["cache_intervals"]["description"]
    )

    assert "ttl_seconds" in schema["properties"]
    assert (
        "Time-to-live for cache"
        in schema["properties"]["ttl_seconds"]["description"]
    )

    assert "min_tokens" in schema["properties"]
    assert (
        "Minimum estimated request tokens"
        in schema["properties"]["min_tokens"]["description"]
    )

  def test_immutability_config(self):
    """Test that the model config is set correctly."""
    config = ContextCacheConfig()
    assert config.model_config["extra"] == "forbid"

  def test_realistic_scenarios(self):
    """Test realistic configuration scenarios."""
    # Quick caching for development
    dev_config = ContextCacheConfig(
        cache_intervals=5, ttl_seconds=600, min_tokens=0  # 10 minutes
    )
    assert dev_config.cache_intervals == 5
    assert dev_config.ttl_seconds == 600

    # Production caching
    prod_config = ContextCacheConfig(
        cache_intervals=20, ttl_seconds=7200, min_tokens=2048  # 2 hours
    )
    assert prod_config.cache_intervals == 20
    assert prod_config.ttl_seconds == 7200
    assert prod_config.min_tokens == 2048

    # Conservative caching
    conservative_config = ContextCacheConfig(
        cache_intervals=3, ttl_seconds=300, min_tokens=4096  # 5 minutes
    )
    assert conservative_config.cache_intervals == 3
    assert conservative_config.ttl_seconds == 300
    assert conservative_config.min_tokens == 4096

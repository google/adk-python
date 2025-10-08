"""Tests for cache_manager.py - Intelligent query caching system."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from agents._tools.cache_manager import SimpleCache, cached, cache_stats, clear_cache


class TestSimpleCache:
    """Test cases for SimpleCache class."""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache = SimpleCache()
        assert cache._max_size == 100
        assert cache._default_ttl == 600
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = SimpleCache()
        cache.set("test_key", "test_value")

        result = cache.get("test_key")
        assert result == "test_value"

    def test_cache_get_nonexistent(self):
        """Test getting a non-existent key returns None."""
        cache = SimpleCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = SimpleCache()
        cache.set("test_key", "test_value", ttl=1)  # 1 second TTL

        # Should exist immediately
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be None after expiration
        assert cache.get("test_key") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SimpleCache(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # All should exist
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Add fourth item (should evict key1)
        cache.set("key4", "value4")

        # key1 should be gone (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_lru_updates_on_access(self):
        """Test accessing a key updates its LRU position."""
        cache = SimpleCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 (makes it most recently used)
        cache.get("key1")

        # Add key4 (should evict key2, not key1)
        cache.set("key4", "value4")

        # key1 should still exist, key2 should be gone
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_delete(self):
        """Test deleting cache entries."""
        cache = SimpleCache()
        cache.set("test_key", "test_value")

        assert cache.delete("test_key") is True
        assert cache.get("test_key") is None
        assert cache.delete("test_key") is False  # Already deleted

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = SimpleCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.clear()

        assert len(cache._cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = SimpleCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Cause hits and misses
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("key3")  # miss
        cache.get("key2")  # hit

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["total_requests"] == 4
        assert "75.0%" in stats["hit_rate"]

    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            # Create cache with persistence
            cache1 = SimpleCache(persist_path=str(cache_file))
            cache1.set("key1", "value1", ttl=3600)  # Long TTL
            cache1.set("key2", "value2", ttl=3600)

            # Create new cache instance (should load from disk)
            cache2 = SimpleCache(persist_path=str(cache_file))

            assert cache2.get("key1") == "value1"
            assert cache2.get("key2") == "value2"

    def test_cache_persistence_ignores_expired(self):
        """Test persistence doesn't save expired entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"

            cache1 = SimpleCache(persist_path=str(cache_file))
            cache1.set("key1", "value1", ttl=1)  # Short TTL

            time.sleep(1.1)  # Wait for expiration

            # Trigger save by setting new value
            cache1.set("key2", "value2", ttl=3600)

            # Create new cache (should only load key2)
            cache2 = SimpleCache(persist_path=str(cache_file))

            assert cache2.get("key1") is None
            assert cache2.get("key2") == "value2"


class TestCachedDecorator:
    """Test cases for @cached decorator."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_cached_decorator_basic(self):
        """Test basic @cached decorator functionality."""
        call_count = {"count": 0}

        @cached(ttl=300)
        def expensive_function(x):
            call_count["count"] += 1
            return x * 2

        # First call - should execute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count["count"] == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count["count"] == 1  # Not incremented

    def test_cached_decorator_different_args(self):
        """Test @cached with different arguments."""
        call_count = {"count": 0}

        @cached(ttl=300)
        def expensive_function(x):
            call_count["count"] += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count["count"] == 2  # Two different calls

    def test_cached_decorator_kwargs(self):
        """Test @cached with keyword arguments."""
        call_count = {"count": 0}

        @cached(ttl=300)
        def expensive_function(x, multiplier=2):
            call_count["count"] += 1
            return x * multiplier

        result1 = expensive_function(5, multiplier=3)
        result2 = expensive_function(5, multiplier=3)
        result3 = expensive_function(5, multiplier=4)

        assert result1 == 15
        assert result2 == 15
        assert result3 == 20
        assert call_count["count"] == 2  # Two different calls

    def test_cached_decorator_ttl_expiration(self):
        """Test @cached decorator respects TTL."""
        call_count = {"count": 0}

        @cached(ttl=1)
        def expensive_function(x):
            call_count["count"] += 1
            return x * 2

        result1 = expensive_function(5)
        assert call_count["count"] == 1

        time.sleep(1.1)

        result2 = expensive_function(5)
        assert call_count["count"] == 2  # Cache expired, executed again

    def test_cached_decorator_key_prefix(self):
        """Test @cached with key prefix."""
        call_count = {"count": 0}

        @cached(ttl=300, key_prefix="security")
        def expensive_function(x):
            call_count["count"] += 1
            return x * 2

        result = expensive_function(5)
        assert result == 10
        assert call_count["count"] == 1


def test_cache_stats_function():
    """Test global cache_stats() function."""
    clear_cache()  # Start fresh

    @cached(ttl=300)
    def test_func(x):
        return x * 2

    # Generate some hits and misses
    test_func(5)
    test_func(5)  # hit
    test_func(10)

    stats = cache_stats()

    assert "size" in stats
    assert "hit_rate" in stats
    assert stats["total_requests"] >= 2


def test_clear_cache_function():
    """Test global clear_cache() function."""
    @cached(ttl=300)
    def test_func(x):
        return x * 2

    test_func(5)
    clear_cache()

    stats = cache_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

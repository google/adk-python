"""Tests for performance_tools.py - Performance monitoring tools."""

import pytest

from agents._tools.cache_manager import cached, clear_cache
from agents._tools.performance_tools import (
    clear_query_cache,
    get_cache_statistics,
)


class TestPerformanceTools:
    """Test cases for performance monitoring tools."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_get_cache_statistics_empty(self):
        """Test cache statistics with empty cache."""
        result = get_cache_statistics()

        assert isinstance(result, str)
        assert "Cache Performance Statistics" in result
        assert "0 / " in result  # Empty cache
        assert "Hit Rate" in result

    def test_get_cache_statistics_with_data(self):
        """Test cache statistics with cache data."""
        # Create some cached data
        @cached(ttl=300)
        def test_func(x):
            return x * 2

        test_func(5)
        test_func(5)  # Cache hit
        test_func(10)

        result = get_cache_statistics()

        assert "Cache Performance Statistics" in result
        assert "Cache Size" in result
        assert "Hit Rate" in result
        assert "Cache Hits" in result
        assert "Cache Misses" in result
        assert "Total Requests" in result

    def test_clear_query_cache(self):
        """Test clear_query_cache function."""
        # Create some cached data
        @cached(ttl=300)
        def test_func(x):
            return x * 2

        test_func(5)
        test_func(10)

        # Verify cache has data
        stats_before = get_cache_statistics()
        assert "2 / " in stats_before  # 2 entries

        # Clear cache
        result = clear_query_cache()

        assert isinstance(result, str)
        assert "cleared successfully" in result

        # Verify cache is empty
        stats_after = get_cache_statistics()
        assert "0 / " in stats_after  # 0 entries

    def test_clear_query_cache_idempotent(self):
        """Test clearing cache multiple times doesn't cause errors."""
        clear_query_cache()
        result = clear_query_cache()

        assert "cleared successfully" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

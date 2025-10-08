"""Performance and monitoring tools for the security platform."""

from .cache_manager import cache_stats, clear_cache


def get_cache_statistics() -> str:
    """Get performance cache statistics.

    Returns cache hit rate, size, and other performance metrics.
    Useful for monitoring query performance improvements.

    Returns:
        Formatted string with cache statistics
    """
    stats = cache_stats()

    summary_lines = [
        "📊 Cache Performance Statistics:",
        f"   Cache Size: {stats['size']} / {stats['max_size']} entries",
        f"   Hit Rate: {stats['hit_rate']}",
        f"   Cache Hits: {stats['hits']:,}",
        f"   Cache Misses: {stats['misses']:,}",
        f"   Total Requests: {stats['total_requests']:,}",
        "",
        "💡 Higher hit rates mean faster query responses!",
    ]

    return "\n".join(summary_lines)


def clear_query_cache() -> str:
    """Clear all cached query results.

    Use this if you need fresh data immediately or after making
    changes to the security findings table.

    Returns:
        Confirmation message
    """
    clear_cache()
    return "✅ Query cache cleared successfully. Next queries will fetch fresh data."

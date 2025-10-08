"""
Simple cache manager for query results with TTL and LRU eviction.

Provides in-memory caching with optional file-based persistence.
No external dependencies (Redis) required.
"""

import json
import logging
import os
import time
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SimpleCache:
    """Thread-safe in-memory cache with TTL and LRU eviction."""

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: int = 600,  # 10 minutes
        persist_path: Optional[str] = None,
    ):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached items (LRU eviction)
            default_ttl: Default time-to-live in seconds
            persist_path: Optional path for file-based persistence
        """
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._persist_path = persist_path
        self._hits = 0
        self._misses = 0

        # Load from persistence if available
        if self._persist_path and Path(self._persist_path).exists():
            self._load_from_disk()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        expires_at = entry["expires_at"]

        # Check if expired
        if time.time() > expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (LRU - most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self._default_ttl

        expires_at = time.time() + ttl

        # Add/update entry
        self._cache[key] = {"value": value, "expires_at": expires_at}
        self._cache.move_to_end(key)

        # Evict oldest if over max size (LRU)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        # Persist if enabled
        if self._persist_path:
            self._save_to_disk()

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            if self._persist_path:
                self._save_to_disk()
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        if self._persist_path:
            self._save_to_disk()

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (
            (self._hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "total_requests": total_requests,
        }

    def _save_to_disk(self) -> None:
        """Persist cache to disk (non-blocking)."""
        try:
            # Only save non-expired entries
            current_time = time.time()
            valid_entries = {
                k: v
                for k, v in self._cache.items()
                if v["expires_at"] > current_time
            }

            cache_data = {
                "entries": valid_entries,
                "stats": {"hits": self._hits, "misses": self._misses},
            }

            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "w") as f:
                json.dump(cache_data, f)

        except Exception as e:
            logger.warning(f"Failed to persist cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self._persist_path, "r") as f:
                cache_data = json.load(f)

            # Load non-expired entries
            current_time = time.time()
            for key, entry in cache_data.get("entries", {}).items():
                if entry["expires_at"] > current_time:
                    self._cache[key] = entry

            # Load stats
            stats = cache_data.get("stats", {})
            self._hits = stats.get("hits", 0)
            self._misses = stats.get("misses", 0)

            logger.info(
                f"Loaded {len(self._cache)} cached entries from disk"
            )

        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")


# Global cache instance
_global_cache: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    """Get or create global cache instance.

    Returns:
        Global SimpleCache instance
    """
    global _global_cache

    if _global_cache is None:
        # Create cache directory in project root
        cache_dir = Path(__file__).parent.parent.parent / ".cache"
        cache_file = cache_dir / "query_cache.json"

        _global_cache = SimpleCache(
            max_size=100,
            default_ttl=600,  # 10 minutes
            persist_path=str(cache_file),
        )
        logger.info("Initialized global cache with 10-minute TTL")

    return _global_cache


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (uses default if None)
        key_prefix: Prefix for cache key

    Example:
        @cached(ttl=300, key_prefix="security")
        def get_security_data():
            return expensive_query()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Build cache key from function name and arguments
            cache_key_parts = [key_prefix, func.__name__]

            # Add string representation of args
            if args:
                cache_key_parts.append(str(args))
            if kwargs:
                # Sort kwargs for consistent key
                sorted_kwargs = sorted(kwargs.items())
                cache_key_parts.append(str(sorted_kwargs))

            cache_key = ":".join(cache_key_parts)

            # Try to get from cache
            cache = get_cache()
            cached_value = cache.get(cache_key)

            if cached_value is not None:
                logger.debug(f"Cache HIT for {func.__name__}")
                return cached_value

            # Cache miss - execute function
            logger.debug(f"Cache MISS for {func.__name__}")
            result = func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def cache_stats() -> dict:
    """Get cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    return get_cache().stats()


def clear_cache() -> None:
    """Clear all cached data."""
    get_cache().clear()
    logger.info("Cache cleared")

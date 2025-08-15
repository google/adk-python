"""
Universal Caching Service for GCP API Calls
Provides TTL-based caching, deduplication, and performance optimization
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable
from functools import wraps
import pickle
from .asset_cache_manager import AssetCacheManager, get_asset_cache_manager

logger = logging.getLogger(__name__)

class CacheService:
    """
    Enhanced cache service with both in-memory and persistent JSON snapshot support
    Integrates with AssetCacheManager for file-based persistence
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache service
        
        Args:
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._asset_cache_manager: Optional[AssetCacheManager] = None
        
    def _generate_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate cache key from namespace and arguments"""
        key_data = {
            "namespace": namespace,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Returns None if key doesn't exist or is expired
        """
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry["expires_at"]:
                self.hit_count += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry["value"]
            else:
                # Expired entry, remove it
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key}")
        
        self.miss_count += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if not specified)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Deleted cache key: {key}")
            return True
        return False
    
    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            namespace: If specified, only clear keys in this namespace
            
        Returns:
            Number of entries cleared
        """
        if namespace:
            keys_to_delete = [
                key for key in self.cache.keys()
                if key.startswith(namespace)
            ]
            for key in keys_to_delete:
                del self.cache[key]
            count = len(keys_to_delete)
        else:
            count = len(self.cache)
            self.cache.clear()
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    async def get_asset_cache_manager(self) -> AssetCacheManager:
        """Get or initialize asset cache manager."""
        if self._asset_cache_manager is None:
            self._asset_cache_manager = await get_asset_cache_manager()
        return self._asset_cache_manager

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics including persistent cache."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            "memory_cache": {
                "entries": len(self.cache),
                "hits": self.hit_count,
                "misses": self.miss_count,
                "hit_rate": f"{hit_rate:.2f}%",
                "total_requests": total_requests
            }
        }
        
        # Add persistent cache stats if available
        try:
            asset_cache = await self.get_asset_cache_manager()
            persistent_stats = await asset_cache.get_cache_stats()
            stats["persistent_cache"] = persistent_stats
        except Exception as e:
            logger.warning(f"Could not get persistent cache stats: {e}")
            stats["persistent_cache"] = {"error": str(e)}
        
        return stats
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None,
        deduplicate: bool = True
    ) -> Any:
        """
        Get value from cache or compute and cache it
        
        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl: Time to live in seconds
            deduplicate: If True, deduplicate concurrent requests
            
        Returns:
            Cached or computed value
        """
        # Check cache first
        value = await self.get(key)
        if value is not None:
            return value
        
        # Handle deduplication of concurrent requests
        if deduplicate and key in self.pending_requests:
            logger.debug(f"Deduplicating request for key: {key}")
            return await self.pending_requests[key]
        
        # Create future for deduplication
        if deduplicate:
            future = asyncio.create_future()
            self.pending_requests[key] = future
        
        try:
            # Compute value
            logger.debug(f"Computing value for key: {key}")
            value = await factory()
            
            # Cache the result
            await self.set(key, value, ttl)
            
            # Resolve future for waiting requests
            if deduplicate:
                future.set_result(value)
            
            return value
            
        except Exception as e:
            # Propagate error to waiting requests
            if deduplicate and key in self.pending_requests:
                future.set_exception(e)
            raise
            
        finally:
            # Clean up pending request
            if deduplicate and key in self.pending_requests:
                del self.pending_requests[key]

    async def set_asset_cache(
        self,
        project_id: str,
        query_type: str,
        data: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Store asset data in persistent JSON snapshot cache.
        
        Args:
            project_id: GCP project ID
            query_type: Type of query (e.g., 'compute_instances', 'storage_buckets')
            data: Data to cache
            ttl: Time to live in seconds
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cache key for the stored data
        """
        asset_cache = await self.get_asset_cache_manager()
        return await asset_cache.set(project_id, query_type, data, ttl, **kwargs)

    async def get_asset_cache(
        self,
        project_id: str,
        query_type: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve asset data from persistent JSON snapshot cache.
        
        Args:
            project_id: GCP project ID
            query_type: Type of query
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        asset_cache = await self.get_asset_cache_manager()
        return await asset_cache.get(project_id, query_type, **kwargs)

    async def invalidate_asset_cache(
        self,
        project_id: str,
        query_type: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Invalidate asset cache entries.
        
        Args:
            project_id: GCP project ID
            query_type: Specific query type to invalidate (all if None)
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Number of entries invalidated
        """
        asset_cache = await self.get_asset_cache_manager()
        return await asset_cache.invalidate(project_id, query_type, **kwargs)

    async def warm_asset_cache(
        self,
        project_id: str,
        cache_warmers: list,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Warm asset cache with frequently accessed data.
        
        Args:
            project_id: GCP project ID
            cache_warmers: List of async functions that generate cache data
            parallel: Whether to run warmers in parallel
            
        Returns:
            Results of cache warming operations
        """
        asset_cache = await self.get_asset_cache_manager()
        return await asset_cache.warm_cache(project_id, cache_warmers, parallel)

class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    def cached(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """
        Decorator to cache function results
        
        Args:
            namespace: Cache namespace
            ttl: Time to live in seconds
            key_prefix: Optional prefix for cache key
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [namespace]
                if key_prefix:
                    key_parts.append(key_prefix)
                    
                cache_key = self.cache_service._generate_key(
                    "_".join(key_parts),
                    *args,
                    **kwargs
                )
                
                # Use get_or_set for automatic caching
                return await self.cache_service.get_or_set(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl=ttl
                )
            
            return wrapper
        return decorator

# Global cache instance
cache_service = CacheService(default_ttl=300)  # 5 minutes default
cache_decorator = CacheDecorator(cache_service)

# Export decorator for easy use
cached = cache_decorator.cached
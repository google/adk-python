"""
Intelligent caching strategy for Google Cloud Recommender API data.

Implements multi-tier caching with TTL, invalidation policies, and session-aware caching.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
import redis
from pathlib import Path

from ..models.recommender_models import (
    RecommendationInsight,
    RecommenderType,
    Priority,
    RecommendationState
)

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache level priorities."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

class CachePolicy(Enum):
    """Cache invalidation policies."""
    TTL_ONLY = "ttl_only"
    SMART_INVALIDATION = "smart_invalidation"
    SESSION_AWARE = "session_aware"
    DEPENDENCY_BASED = "dependency_based"

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 1800  # 30 minutes default
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    invalidation_triggers: Set[str] = field(default_factory=set)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> int:
        """Get age of cache entry in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())
    
    def access(self):
        """Mark cache entry as accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class IntelligentRecommendationCache:
    """Multi-tier intelligent caching system for recommendations."""
    
    def __init__(
        self,
        memory_size_limit: int = 1000,
        redis_url: Optional[str] = None,
        disk_cache_path: Optional[str] = None,
        default_ttl: int = 1800,
        enable_smart_invalidation: bool = True
    ):
        """Initialize the caching system.
        
        Args:
            memory_size_limit: Maximum number of entries in memory cache
            redis_url: Redis connection URL for distributed caching
            disk_cache_path: Path for disk-based persistent cache
            default_ttl: Default TTL in seconds
            enable_smart_invalidation: Enable intelligent cache invalidation
        """
        self.memory_size_limit = memory_size_limit
        self.default_ttl = default_ttl
        self.enable_smart_invalidation = enable_smart_invalidation
        
        # Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Redis cache
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        # Disk cache
        self.disk_cache_path = Path(disk_cache_path) if disk_cache_path else None
        if self.disk_cache_path:
            self.disk_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.stats = CacheStats()
        self.invalidation_rules = self._initialize_invalidation_rules()
        
        # Background tasks
        self._cleanup_task = None
        self._started = False
    
    async def start(self):
        """Start the cache system and background tasks."""
        if not self._started:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
            self._started = True
            logger.info("Recommendation cache system started")
    
    async def stop(self):
        """Stop the cache system and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self._started = False
        logger.info("Recommendation cache system stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache with multi-tier lookup.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if found, None otherwise
        """
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired:
                entry.access()
                self.stats.hits += 1
                logger.debug(f"Cache hit (memory): {key}")
                return entry.data
            else:
                # Remove expired entry
                del self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                redis_data = await self._get_from_redis(key)
                if redis_data:
                    # Promote to memory cache
                    await self._promote_to_memory(key, redis_data)
                    self.stats.hits += 1
                    logger.debug(f"Cache hit (redis): {key}")
                    return redis_data
            except Exception as e:
                logger.warning(f"Redis cache error for key {key}: {e}")
        
        # Try disk cache
        if self.disk_cache_path:
            try:
                disk_data = await self._get_from_disk(key)
                if disk_data:
                    # Promote to higher tiers
                    await self._promote_to_memory(key, disk_data)
                    if self.redis_client:
                        await self._store_in_redis(key, disk_data)
                    self.stats.hits += 1
                    logger.debug(f"Cache hit (disk): {key}")
                    return disk_data
            except Exception as e:
                logger.warning(f"Disk cache error for key {key}: {e}")
        
        self.stats.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None
    ):
        """Store data in cache with intelligent placement.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            tags: Tags for categorization
            dependencies: Cache dependencies for invalidation
        """
        ttl = ttl or self.default_ttl
        tags = tags or set()
        dependencies = dependencies or set()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl,
            tags=tags,
            dependencies=dependencies,
            invalidation_triggers=self._get_invalidation_triggers(key, tags)
        )
        
        # Store in memory cache
        await self._store_in_memory(entry)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self._store_in_redis(key, data, ttl)
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")
        
        # Store in disk cache for persistence
        if self.disk_cache_path:
            try:
                await self._store_in_disk(key, data, ttl)
            except Exception as e:
                logger.warning(f"Failed to store in disk cache: {e}")
        
        logger.debug(f"Cached data: {key}")
    
    async def invalidate(self, key: str):
        """Invalidate a specific cache entry."""
        # Remove from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis key {key}: {e}")
        
        # Remove from disk
        if self.disk_cache_path:
            disk_file = self.disk_cache_path / f"{self._hash_key(key)}.cache"
            if disk_file.exists():
                disk_file.unlink()
        
        logger.debug(f"Invalidated cache key: {key}")
    
    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate cache entries by tags."""
        keys_to_invalidate = []
        
        for key, entry in self.memory_cache.items():
            if tags.intersection(entry.tags):
                keys_to_invalidate.append(key)
        
        for key in keys_to_invalidate:
            await self.invalidate(key)
        
        logger.debug(f"Invalidated {len(keys_to_invalidate)} entries by tags: {tags}")
    
    async def invalidate_by_dependencies(self, dependency: str):
        """Invalidate cache entries that depend on a specific resource."""
        keys_to_invalidate = []
        
        for key, entry in self.memory_cache.items():
            if dependency in entry.dependencies:
                keys_to_invalidate.append(key)
        
        for key in keys_to_invalidate:
            await self.invalidate(key)
        
        logger.debug(f"Invalidated {len(keys_to_invalidate)} entries by dependency: {dependency}")
    
    def build_cache_key(
        self,
        operation: str,
        project_id: str,
        recommender_type: Optional[RecommenderType] = None,
        location: str = "global",
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Build a standardized cache key.
        
        Args:
            operation: Operation type (e.g., 'list_recommendations', 'get_insights')
            project_id: GCP project ID
            recommender_type: Specific recommender type
            location: GCP location/region
            filters: Query filters
            **kwargs: Additional parameters
            
        Returns:
            Standardized cache key
        """
        key_parts = [
            operation,
            project_id,
            location
        ]
        
        if recommender_type:
            key_parts.append(recommender_type.value)
        
        if filters:
            # Sort filters for consistent keys
            sorted_filters = sorted(filters.items())
            filter_str = json.dumps(sorted_filters, sort_keys=True)
            key_parts.append(hashlib.md5(filter_str.encode()).hexdigest()[:8])
        
        # Add additional parameters
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    def get_cache_tags(
        self,
        project_id: str,
        recommender_type: Optional[RecommenderType] = None,
        operation: Optional[str] = None
    ) -> Set[str]:
        """Generate cache tags for categorization.
        
        Args:
            project_id: GCP project ID
            recommender_type: Recommender type
            operation: Operation type
            
        Returns:
            Set of cache tags
        """
        tags = {f"project:{project_id}"}
        
        if recommender_type:
            tags.add(f"type:{recommender_type.value}")
            # Add category tag
            if "iam" in recommender_type.value.lower():
                tags.add("category:security")
            elif "firewall" in recommender_type.value.lower():
                tags.add("category:security")
            elif "machine" in recommender_type.value.lower():
                tags.add("category:cost")
            elif "commitment" in recommender_type.value.lower():
                tags.add("category:cost")
        
        if operation:
            tags.add(f"operation:{operation}")
        
        return tags
    
    async def warm_up_cache(
        self,
        projects: List[str],
        recommender_types: List[RecommenderType]
    ):
        """Pre-populate cache with commonly requested data.
        
        Args:
            projects: List of project IDs to warm up
            recommender_types: List of recommender types to pre-fetch
        """
        logger.info(f"Warming up cache for {len(projects)} projects and {len(recommender_types)} types")
        
        tasks = []
        for project_id in projects:
            for recommender_type in recommender_types:
                # This would call the actual recommender service
                # For now, we'll create placeholder tasks
                task = self._warm_up_project_type(project_id, recommender_type)
                tasks.append(task)
        
        # Execute warm-up tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cache warm-up completed")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_count = len(self.memory_cache)
        memory_size = sum(
            len(pickle.dumps(entry.data)) for entry in self.memory_cache.values()
        )
        
        redis_count = 0
        if self.redis_client:
            try:
                redis_count = await self.redis_client.dbsize()
            except Exception:
                redis_count = -1  # Error indicator
        
        disk_count = 0
        disk_size = 0
        if self.disk_cache_path and self.disk_cache_path.exists():
            cache_files = list(self.disk_cache_path.glob("*.cache"))
            disk_count = len(cache_files)
            disk_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "statistics": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "evictions": self.stats.evictions
            },
            "memory_cache": {
                "entries": memory_count,
                "size_bytes": memory_size,
                "limit": self.memory_size_limit
            },
            "redis_cache": {
                "entries": redis_count,
                "available": self.redis_client is not None
            },
            "disk_cache": {
                "entries": disk_count,
                "size_bytes": disk_size,
                "path": str(self.disk_cache_path) if self.disk_cache_path else None
            }
        }
    
    # Private methods
    
    async def _store_in_memory(self, entry: CacheEntry):
        """Store entry in memory cache with LRU eviction."""
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.memory_size_limit:
            await self._evict_lru_entries()
        
        self.memory_cache[entry.key] = entry
    
    async def _evict_lru_entries(self):
        """Evict least recently used entries."""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 10% or at least 1 entry
        num_to_evict = max(1, len(sorted_entries) // 10)
        
        for i in range(num_to_evict):
            key, _ = sorted_entries[i]
            del self.memory_cache[key]
            self.stats.evictions += 1
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def _store_in_redis(self, key: str, data: Any, ttl: int = None):
        """Store data in Redis cache."""
        if not self.redis_client:
            return
        
        try:
            serialized_data = json.dumps(data, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_data)
            else:
                await self.redis_client.set(key, serialized_data)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get data from disk cache."""
        if not self.disk_cache_path:
            return None
        
        cache_file = self.disk_cache_path / f"{self._hash_key(key)}.cache"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if expired
            if datetime.now() > cache_data['expires_at']:
                cache_file.unlink()
                return None
            
            return cache_data['data']
        except Exception as e:
            logger.warning(f"Disk cache read error: {e}")
            return None
    
    async def _store_in_disk(self, key: str, data: Any, ttl: int):
        """Store data in disk cache."""
        if not self.disk_cache_path:
            return
        
        cache_file = self.disk_cache_path / f"{self._hash_key(key)}.cache"
        cache_data = {
            'data': data,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Disk cache write error: {e}")
    
    async def _promote_to_memory(self, key: str, data: Any):
        """Promote cache entry to memory tier."""
        entry = CacheEntry(
            key=key,
            data=data,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=self.default_ttl
        )
        await self._store_in_memory(entry)
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _initialize_invalidation_rules(self) -> Dict[str, Set[str]]:
        """Initialize intelligent invalidation rules."""
        return {
            "project_updated": {"project:*"},
            "recommendation_applied": {"type:*", "operation:list_recommendations"},
            "policy_changed": {"type:google.iam.policy.Recommender", "category:security"},
            "firewall_changed": {"type:google.compute.firewall.Recommender", "category:security"},
            "instance_modified": {"type:google.compute.instance.MachineTypeRecommender"}
        }
    
    def _get_invalidation_triggers(self, key: str, tags: Set[str]) -> Set[str]:
        """Get invalidation triggers for a cache entry."""
        triggers = set()
        
        for tag in tags:
            if "project:" in tag:
                triggers.add("project_updated")
            elif "type:" in tag and "iam" in tag:
                triggers.add("policy_changed")
            elif "type:" in tag and "firewall" in tag:
                triggers.add("firewall_changed")
            elif "type:" in tag and "instance" in tag:
                triggers.add("instance_modified")
        
        return triggers
    
    async def _warm_up_project_type(self, project_id: str, recommender_type: RecommenderType):
        """Warm up cache for a specific project and recommender type."""
        # This would integrate with the actual recommender service
        # For now, create a placeholder cache entry
        cache_key = self.build_cache_key(
            "list_recommendations",
            project_id,
            recommender_type
        )
        
        # Simulate data fetching and caching
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Cache placeholder data
        placeholder_data = {
            "project_id": project_id,
            "recommender_type": recommender_type.value,
            "recommendations": [],
            "cached_at": datetime.now().isoformat()
        }
        
        tags = self.get_cache_tags(project_id, recommender_type, "list_recommendations")
        await self.set(cache_key, placeholder_data, tags=tags)
    
    async def _cleanup_expired_entries(self):
        """Background task to cleanup expired cache entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup memory cache
                expired_keys = [
                    key for key, entry in self.memory_cache.items()
                    if entry.is_expired
                ]
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")
                
                # Cleanup disk cache
                if self.disk_cache_path and self.disk_cache_path.exists():
                    await self._cleanup_expired_disk_entries()
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    async def _cleanup_expired_disk_entries(self):
        """Cleanup expired disk cache entries."""
        try:
            cache_files = list(self.disk_cache_path.glob("*.cache"))
            expired_count = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if datetime.now() > cache_data['expires_at']:
                        cache_file.unlink()
                        expired_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking disk cache file {cache_file}: {e}")
                    # Remove corrupted files
                    cache_file.unlink()
                    expired_count += 1
            
            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired disk cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up disk cache: {e}")

# Global cache instance
recommendation_cache = IntelligentRecommendationCache(
    memory_size_limit=1000,
    default_ttl=1800,  # 30 minutes
    enable_smart_invalidation=True
)
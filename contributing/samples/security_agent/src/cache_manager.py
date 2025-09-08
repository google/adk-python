"""
Advanced Multi-Tier Caching System for GCP Security Agent
Implements memory -> Redis -> SQLite caching hierarchy with compression and intelligent invalidation
"""

import asyncio
import json
import sqlite3
import zlib
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import wraps
import threading
from collections import OrderedDict
import weakref

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for multi-tier caching system"""
    # Memory cache settings
    memory_max_size: int = 1000  # Max items in memory
    memory_ttl: int = 300  # 5 minutes
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 3600  # 1 hour
    redis_max_connections: int = 20
    
    # SQLite cache settings
    sqlite_path: str = ".cache/cache.db"
    sqlite_ttl: int = 86400  # 24 hours
    sqlite_max_size: int = 1000000  # 1M items
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress items > 1KB
    compression_level: int = 6
    
    # Performance settings
    async_write: bool = True
    batch_size: int = 100
    cleanup_interval: int = 3600  # 1 hour

class CacheStats:
    """Thread-safe cache statistics"""
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
    
    def reset(self):
        with self._lock:
            self.hits = {"memory": 0, "redis": 0, "sqlite": 0}
            self.misses = {"memory": 0, "redis": 0, "sqlite": 0}
            self.writes = {"memory": 0, "redis": 0, "sqlite": 0}
            self.evictions = {"memory": 0, "redis": 0, "sqlite": 0}
            self.errors = {"memory": 0, "redis": 0, "sqlite": 0}
    
    def record_hit(self, tier: str):
        with self._lock:
            self.hits[tier] += 1
    
    def record_miss(self, tier: str):
        with self._lock:
            self.misses[tier] += 1
    
    def record_write(self, tier: str):
        with self._lock:
            self.writes[tier] += 1
    
    def record_eviction(self, tier: str):
        with self._lock:
            self.evictions[tier] += 1
    
    def record_error(self, tier: str):
        with self._lock:
            self.errors[tier] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                "hits": self.hits.copy(),
                "misses": self.misses.copy(),
                "writes": self.writes.copy(),
                "evictions": self.evictions.copy(),
                "errors": self.errors.copy()
            }

class MemoryCache:
    """High-performance in-memory LRU cache"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            
            # Check TTL
            if time.time() - item['timestamp'] > self.ttl:
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return item['data']
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            # Remove oldest items if at capacity
            while len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            
            self._cache[key] = {
                'data': value,
                'timestamp': time.time()
            }
            self._cache.move_to_end(key)
    
    async def delete(self, key: str):
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self):
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        async with self._lock:
            return len(self._cache)
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        async with self._lock:
            for key, item in self._cache.items():
                if current_time - item['timestamp'] > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        return len(expired_keys)

class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._pool = None
        self._redis = None
    
    async def initialize(self):
        """Initialize Redis connection pool"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis cache layer")
            return
        
        try:
            self._pool = redis.ConnectionPool(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                max_connections=self.config.redis_max_connections,
                decode_responses=False
            )
            self._redis = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self._redis = None
    
    async def get(self, key: str) -> Optional[Any]:
        if not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        if not self._redis:
            return
        
        try:
            ttl = ttl or self.config.redis_ttl
            data = pickle.dumps(value)
            await self._redis.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        if not self._redis:
            return
        
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def clear(self):
        if not self._redis:
            return
        
        try:
            await self._redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    async def close(self):
        if self._redis:
            await self._redis.close()

class SQLiteCache:
    """Persistent SQLite cache for long-term storage"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
    
    async def initialize(self):
        """Initialize SQLite cache database"""
        try:
            def setup_db():
                import os
                os.makedirs(os.path.dirname(self.config.sqlite_path), exist_ok=True)
                
                conn = sqlite3.connect(self.config.sqlite_path)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        timestamp REAL,
                        ttl INTEGER,
                        compressed INTEGER DEFAULT 0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache(ttl)")
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(self._executor, setup_db)
            self._initialized = True
            logger.info("SQLite cache initialized successfully")
            
        except Exception as e:
            logger.error(f"SQLite cache initialization failed: {e}")
    
    def _compress_data(self, data: bytes) -> tuple[bytes, bool]:
        """Compress data if above threshold"""
        if not self.config.enable_compression or len(data) < self.config.compression_threshold:
            return data, False
        
        compressed = zlib.compress(data, self.config.compression_level)
        return compressed, True
    
    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if compressed"""
        if is_compressed:
            return zlib.decompress(data)
        return data
    
    async def get(self, key: str) -> Optional[Any]:
        if not self._initialized:
            return None
        
        def get_from_db():
            conn = sqlite3.connect(self.config.sqlite_path)
            cursor = conn.execute(
                "SELECT value, timestamp, ttl, compressed FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            value_blob, timestamp, ttl, compressed = row
            
            # Check TTL
            if time.time() - timestamp > ttl:
                return "EXPIRED"
            
            # Decompress and deserialize
            decompressed = self._decompress_data(value_blob, compressed)
            return pickle.loads(decompressed)
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(self._executor, get_from_db)
            
            if result == "EXPIRED":
                await self.delete(key)
                return None
            
            return result
        except Exception as e:
            logger.error(f"SQLite get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        if not self._initialized:
            return
        
        def set_in_db():
            ttl_val = ttl or self.config.sqlite_ttl
            data = pickle.dumps(value)
            compressed_data, is_compressed = self._compress_data(data)
            
            conn = sqlite3.connect(self.config.sqlite_path)
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp, ttl, compressed) VALUES (?, ?, ?, ?, ?)",
                (key, compressed_data, time.time(), ttl_val, is_compressed)
            )
            conn.commit()
            conn.close()
        
        try:
            if self.config.async_write:
                asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(self._executor, set_in_db)
                )
            else:
                await asyncio.get_event_loop().run_in_executor(self._executor, set_in_db)
        except Exception as e:
            logger.error(f"SQLite set error: {e}")
    
    async def delete(self, key: str):
        if not self._initialized:
            return
        
        def delete_from_db():
            conn = sqlite3.connect(self.config.sqlite_path)
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            conn.close()
        
        try:
            await asyncio.get_event_loop().run_in_executor(self._executor, delete_from_db)
        except Exception as e:
            logger.error(f"SQLite delete error: {e}")
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        if not self._initialized:
            return 0
        
        def cleanup_db():
            current_time = time.time()
            conn = sqlite3.connect(self.config.sqlite_path)
            cursor = conn.execute(
                "DELETE FROM cache WHERE timestamp + ttl < ?",
                (current_time,)
            )
            deleted = cursor.rowcount
            
            # Vacuum if many items deleted
            if deleted > 1000:
                conn.execute("VACUUM")
            
            conn.commit()
            conn.close()
            return deleted
        
        try:
            return await asyncio.get_event_loop().run_in_executor(self._executor, cleanup_db)
        except Exception as e:
            logger.error(f"SQLite cleanup error: {e}")
            return 0

class MultiTierCache:
    """Multi-tier caching system with intelligent fallback"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # Initialize cache tiers
        self.memory = MemoryCache(self.config.memory_max_size, self.config.memory_ttl)
        self.redis = RedisCache(self.config)
        self.sqlite = SQLiteCache(self.config)
        
        # Warming strategies
        self._warming_tasks = {}
        self._cleanup_task = None
    
    async def initialize(self):
        """Initialize all cache tiers"""
        await self.redis.initialize()
        await self.sqlite.initialize()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Multi-tier cache system initialized")
    
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate namespaced cache key"""
        return f"{namespace}:{hashlib.md5(key.encode()).hexdigest()}"
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache tiers (memory -> redis -> sqlite)"""
        cache_key = self._generate_key(key, namespace)
        
        # Try memory cache first
        try:
            value = await self.memory.get(cache_key)
            if value is not None:
                self.stats.record_hit("memory")
                return value
        except Exception as e:
            logger.error(f"Memory cache error: {e}")
            self.stats.record_error("memory")
        
        self.stats.record_miss("memory")
        
        # Try Redis cache
        try:
            value = await self.redis.get(cache_key)
            if value is not None:
                self.stats.record_hit("redis")
                # Populate memory cache
                await self.memory.set(cache_key, value)
                return value
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            self.stats.record_error("redis")
        
        self.stats.record_miss("redis")
        
        # Try SQLite cache
        try:
            value = await self.sqlite.get(cache_key)
            if value is not None:
                self.stats.record_hit("sqlite")
                # Populate upper tiers
                await self.memory.set(cache_key, value)
                await self.redis.set(cache_key, value)
                return value
        except Exception as e:
            logger.error(f"SQLite cache error: {e}")
            self.stats.record_error("sqlite")
        
        self.stats.record_miss("sqlite")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None, namespace: str = "default"):
        """Set value in all cache tiers"""
        cache_key = self._generate_key(key, namespace)
        
        # Set in all tiers
        tasks = []
        
        # Memory
        try:
            tasks.append(self.memory.set(cache_key, value))
            self.stats.record_write("memory")
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            self.stats.record_error("memory")
        
        # Redis
        try:
            tasks.append(self.redis.set(cache_key, value, ttl))
            self.stats.record_write("redis")
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            self.stats.record_error("redis")
        
        # SQLite
        try:
            tasks.append(self.sqlite.set(cache_key, value, ttl))
            self.stats.record_write("sqlite")
        except Exception as e:
            logger.error(f"SQLite cache set error: {e}")
            self.stats.record_error("sqlite")
        
        # Execute all sets
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def delete(self, key: str, namespace: str = "default"):
        """Delete value from all cache tiers"""
        cache_key = self._generate_key(key, namespace)
        
        tasks = [
            self.memory.delete(cache_key),
            self.redis.delete(cache_key),
            self.sqlite.delete(cache_key)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def clear(self, namespace: str = None):
        """Clear cache (optionally by namespace)"""
        if namespace:
            # TODO: Implement namespace-specific clearing
            logger.warning("Namespace-specific clearing not implemented")
            return
        
        tasks = [
            self.memory.clear(),
            self.redis.clear()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def warm_cache(self, key: str, fetch_func: Callable, namespace: str = "default", ttl: int = None):
        """Warm cache with data from function"""
        cache_key = self._generate_key(key, namespace)
        
        # Check if already warming
        if cache_key in self._warming_tasks:
            return await self._warming_tasks[cache_key]
        
        async def warm():
            try:
                data = await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
                await self.set(key, data, ttl, namespace)
                return data
            finally:
                self._warming_tasks.pop(cache_key, None)
        
        task = asyncio.create_task(warm())
        self._warming_tasks[cache_key] = task
        return await task
    
    async def get_or_set(self, key: str, fetch_func: Callable, ttl: int = None, namespace: str = "default") -> Any:
        """Get from cache or fetch and set"""
        value = await self.get(key, namespace)
        if value is not None:
            return value
        
        return await self.warm_cache(key, fetch_func, namespace, ttl)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = self.stats.get_stats()
        
        # Add size information
        base_stats["sizes"] = {
            "memory": await self.memory.size(),
            "redis": "N/A",  # Redis size would require additional calls
            "sqlite": "N/A"  # SQLite size would require additional calls
        }
        
        # Calculate hit rates
        base_stats["hit_rates"] = {}
        for tier in ["memory", "redis", "sqlite"]:
            total = base_stats["hits"][tier] + base_stats["misses"][tier]
            if total > 0:
                base_stats["hit_rates"][tier] = base_stats["hits"][tier] / total
            else:
                base_stats["hit_rates"][tier] = 0.0
        
        return base_stats
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Cleanup expired entries
                memory_cleaned = await self.memory.cleanup_expired()
                sqlite_cleaned = await self.sqlite.cleanup_expired()
                
                if memory_cleaned + sqlite_cleaned > 0:
                    logger.info(f"Cache cleanup: {memory_cleaned} memory, {sqlite_cleaned} SQLite entries removed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def close(self):
        """Close all cache connections"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        await self.redis.close()
        logger.info("Multi-tier cache system closed")

# Caching decorators
def cached(ttl: int = 300, namespace: str = "default", key_func: Callable = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = ":".join(key_parts)
            
            # Get or set cache
            return await cache.get_or_set(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl,
                namespace
            )
        
        return wrapper
    return decorator

def invalidate_cache(key_pattern: str = None, namespace: str = "default"):
    """Decorator to invalidate cache after function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            if key_pattern:
                # Invalidate specific pattern
                await cache.delete(key_pattern, namespace)
            else:
                # Clear entire namespace
                await cache.clear(namespace)
            
            return result
        
        return wrapper
    return decorator

# Global cache instance
cache = MultiTierCache()
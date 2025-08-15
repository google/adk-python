"""
Asset Cache Manager for GCP Asset Inventory Data
Provides JSON snapshot persistence, TTL management, and cache warming strategies
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import hashlib
import uuid
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

@dataclass
class CacheMetadata:
    """Metadata for cache entries."""
    created_at: datetime
    expires_at: datetime
    version: str
    ttl_seconds: int
    project_id: str
    cache_key: str
    data_hash: str
    file_size: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(), 
            "version": self.version,
            "ttl_seconds": self.ttl_seconds,
            "project_id": self.project_id,
            "cache_key": self.cache_key,
            "data_hash": self.data_hash,
            "file_size": self.file_size,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        """Create from dictionary."""
        return cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            version=data["version"],
            ttl_seconds=data["ttl_seconds"],
            project_id=data["project_id"],
            cache_key=data["cache_key"],
            data_hash=data["data_hash"],
            file_size=data["file_size"],
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.expires_at.replace(tzinfo=timezone.utc)

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

class AssetCacheManager:
    """
    Comprehensive JSON snapshot persistence system for GCP asset data.
    
    Features:
    - Atomic file operations for data integrity
    - TTL-based expiration with configurable timeouts
    - Project-based file organization
    - Cache warming strategies
    - Performance monitoring and metrics
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour
        max_cache_size: int = 1000,  # Max number of cache files
        cleanup_interval: int = 300,  # 5 minutes
        enable_compression: bool = False
    ):
        """
        Initialize the asset cache manager.
        
        Args:
            cache_dir: Directory for cache files (defaults to backend/cache/assets)
            default_ttl: Default TTL in seconds
            max_cache_size: Maximum number of cache files to maintain
            cleanup_interval: Cleanup interval in seconds
            enable_compression: Enable gzip compression for cache files
        """
        self.cache_dir = Path(cache_dir or "cache/assets").resolve()
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.cleanup_interval = cleanup_interval
        self.enable_compression = enable_compression
        
        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.write_count = 0
        self.cleanup_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache")
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Cache version for compatibility
        self.cache_version = "1.0.0"
        
        # Ensure directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AssetCacheManager initialized with cache_dir={self.cache_dir}")

    async def start(self) -> None:
        """Start the cache manager and background tasks."""
        if self._running:
            return
            
        self._running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info("AssetCacheManager started")

    async def stop(self) -> None:
        """Stop the cache manager and cleanup resources."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("AssetCacheManager stopped")

    def _generate_cache_key(self, project_id: str, query_type: str, **kwargs) -> str:
        """Generate a unique cache key for the given parameters."""
        key_data = {
            "project_id": project_id,
            "query_type": query_type,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_file_path(self, project_id: str, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        project_dir = self.cache_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        extension = ".json.gz" if self.enable_compression else ".json"
        return project_dir / f"{cache_key}{extension}"

    def _get_metadata_file_path(self, project_id: str, cache_key: str) -> Path:
        """Get the metadata file path for a cache entry."""
        project_dir = self.cache_dir / project_id
        project_dir.mkdir(exist_ok=True)
        return project_dir / f"{cache_key}.meta"

    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of data for integrity checking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def _write_file_atomic(self, file_path: Path, data: bytes) -> None:
        """Write file atomically using temporary file and rename."""
        temp_path = file_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        
        try:
            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(data)
            
            # Atomic rename
            await aiofiles.os.rename(str(temp_path), str(file_path))
            
        except Exception:
            # Cleanup temp file on error
            try:
                await aiofiles.os.remove(str(temp_path))
            except Exception:
                pass
            raise

    async def _read_metadata(self, project_id: str, cache_key: str) -> Optional[CacheMetadata]:
        """Read metadata for a cache entry."""
        metadata_path = self._get_metadata_file_path(project_id, cache_key)
        
        if not metadata_path.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_path, "r") as f:
                metadata_dict = json.loads(await f.read())
            return CacheMetadata.from_dict(metadata_dict)
        except Exception as e:
            logger.warning(f"Failed to read metadata {metadata_path}: {e}")
            return None

    async def _write_metadata(self, metadata: CacheMetadata) -> None:
        """Write metadata for a cache entry."""
        metadata_path = self._get_metadata_file_path(metadata.project_id, metadata.cache_key)
        
        try:
            metadata_json = json.dumps(metadata.to_dict(), indent=2)
            await self._write_file_atomic(metadata_path, metadata_json.encode())
        except Exception as e:
            logger.error(f"Failed to write metadata {metadata_path}: {e}")
            raise

    async def set(
        self,
        project_id: str,
        query_type: str,
        data: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Store data in cache with JSON snapshot persistence.
        
        Args:
            project_id: GCP project ID
            query_type: Type of query (e.g., 'compute_instances', 'storage_buckets')
            data: Data to cache
            ttl: Time to live in seconds
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cache key for the stored data
        """
        ttl = ttl or self.default_ttl
        cache_key = self._generate_cache_key(project_id, query_type, **kwargs)
        
        with self._lock:
            try:
                # Prepare data and metadata
                now = datetime.now(timezone.utc)
                expires_at = now + timedelta(seconds=ttl)
                data_hash = self._calculate_data_hash(data)
                
                # Serialize data
                if self.enable_compression:
                    import gzip
                    data_json = json.dumps(data, indent=2, default=str)
                    data_bytes = gzip.compress(data_json.encode())
                else:
                    data_json = json.dumps(data, indent=2, default=str)
                    data_bytes = data_json.encode()
                
                # Write data file
                cache_file_path = self._get_cache_file_path(project_id, cache_key)
                await self._write_file_atomic(cache_file_path, data_bytes)
                
                # Create and write metadata
                metadata = CacheMetadata(
                    created_at=now,
                    expires_at=expires_at,
                    version=self.cache_version,
                    ttl_seconds=ttl,
                    project_id=project_id,
                    cache_key=cache_key,
                    data_hash=data_hash,
                    file_size=len(data_bytes)
                )
                
                await self._write_metadata(metadata)
                
                self.write_count += 1
                logger.debug(f"Cached data for {project_id}/{query_type} (key: {cache_key[:8]}...)")
                
                return cache_key
                
            except Exception as e:
                logger.error(f"Failed to cache data for {project_id}/{query_type}: {e}")
                raise

    async def get(
        self,
        project_id: str,
        query_type: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from cache.
        
        Args:
            project_id: GCP project ID
            query_type: Type of query
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(project_id, query_type, **kwargs)
        
        with self._lock:
            try:
                # Read metadata first
                metadata = await self._read_metadata(project_id, cache_key)
                if not metadata:
                    self.miss_count += 1
                    return None
                
                # Check expiration
                if metadata.is_expired():
                    logger.debug(f"Cache expired for {project_id}/{query_type}")
                    await self._delete_cache_entry(project_id, cache_key)
                    self.miss_count += 1
                    return None
                
                # Read data file
                cache_file_path = self._get_cache_file_path(project_id, cache_key)
                if not cache_file_path.exists():
                    logger.warning(f"Cache file missing: {cache_file_path}")
                    self.miss_count += 1
                    return None
                
                async with aiofiles.open(cache_file_path, "rb") as f:
                    data_bytes = await f.read()
                
                # Decompress if needed
                if self.enable_compression:
                    import gzip
                    data_json = gzip.decompress(data_bytes).decode()
                else:
                    data_json = data_bytes.decode()
                
                data = json.loads(data_json)
                
                # Verify data integrity
                current_hash = self._calculate_data_hash(data)
                if current_hash != metadata.data_hash:
                    logger.warning(f"Data integrity check failed for cache {cache_key}")
                    await self._delete_cache_entry(project_id, cache_key)
                    self.miss_count += 1
                    return None
                
                # Update access statistics
                metadata.update_access()
                await self._write_metadata(metadata)
                
                self.hit_count += 1
                logger.debug(f"Cache hit for {project_id}/{query_type}")
                
                # Add cache metadata to response
                return {
                    "data": data,
                    "cache_metadata": {
                        "cached_at": metadata.created_at.isoformat(),
                        "expires_at": metadata.expires_at.isoformat(),
                        "cache_key": cache_key,
                        "access_count": metadata.access_count
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to read cache for {project_id}/{query_type}: {e}")
                self.miss_count += 1
                return None

    async def invalidate(
        self,
        project_id: str,
        query_type: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            project_id: GCP project ID
            query_type: Specific query type to invalidate (all if None)
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            count = 0
            
            if query_type:
                # Invalidate specific entry
                cache_key = self._generate_cache_key(project_id, query_type, **kwargs)
                if await self._delete_cache_entry(project_id, cache_key):
                    count = 1
            else:
                # Invalidate all entries for project
                project_dir = self.cache_dir / project_id
                if project_dir.exists():
                    for file_path in project_dir.glob("*.json*"):
                        cache_key = file_path.stem.replace(".json", "")
                        if await self._delete_cache_entry(project_id, cache_key):
                            count += 1
            
            logger.info(f"Invalidated {count} cache entries for {project_id}/{query_type or 'all'}")
            return count

    async def _delete_cache_entry(self, project_id: str, cache_key: str) -> bool:
        """Delete a cache entry and its metadata."""
        try:
            cache_file = self._get_cache_file_path(project_id, cache_key)
            metadata_file = self._get_metadata_file_path(project_id, cache_key)
            
            deleted = False
            if cache_file.exists():
                await aiofiles.os.remove(str(cache_file))
                deleted = True
            
            if metadata_file.exists():
                await aiofiles.os.remove(str(metadata_file))
                deleted = True
            
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete cache entry {cache_key}: {e}")
            return False

    async def warm_cache(
        self,
        project_id: str,
        cache_warmers: List[Callable],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Warm cache with frequently accessed data.
        
        Args:
            project_id: GCP project ID
            cache_warmers: List of async functions that generate cache data
            parallel: Whether to run warmers in parallel
            
        Returns:
            Results of cache warming operations
        """
        results = {
            "warmed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            if parallel:
                tasks = [warmer(project_id) for warmer in cache_warmers]
                warmer_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                warmer_results = []
                for warmer in cache_warmers:
                    try:
                        result = await warmer(project_id)
                        warmer_results.append(result)
                    except Exception as e:
                        warmer_results.append(e)
            
            for result in warmer_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                    results["errors"].append(str(result))
                else:
                    results["warmed"] += 1
            
            logger.info(f"Cache warming completed for {project_id}: {results}")
            
        except Exception as e:
            logger.error(f"Cache warming failed for {project_id}: {e}")
            results["errors"].append(str(e))
        
        return results

    async def get_cache_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics and status.
        
        Args:
            project_id: Specific project to get stats for (all if None)
            
        Returns:
            Cache statistics
        """
        stats = {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "write_count": self.write_count,
            "cleanup_count": self.cleanup_count,
            "hit_rate": 0.0,
            "total_entries": 0,
            "total_size_bytes": 0,
            "projects": []
        }
        
        total_requests = self.hit_count + self.miss_count
        if total_requests > 0:
            stats["hit_rate"] = (self.hit_count / total_requests) * 100
        
        try:
            if project_id:
                project_dirs = [self.cache_dir / project_id]
            else:
                project_dirs = [d for d in self.cache_dir.iterdir() if d.is_dir()]
            
            for project_dir in project_dirs:
                if not project_dir.exists():
                    continue
                
                project_stats = {
                    "project_id": project_dir.name,
                    "entries": 0,
                    "size_bytes": 0,
                    "expired_entries": 0
                }
                
                for cache_file in project_dir.glob("*.json*"):
                    if cache_file.suffix in [".json", ".gz"]:
                        project_stats["entries"] += 1
                        project_stats["size_bytes"] += cache_file.stat().st_size
                        
                        # Check if expired
                        cache_key = cache_file.stem.replace(".json", "")
                        metadata = await self._read_metadata(project_dir.name, cache_key)
                        if metadata and metadata.is_expired():
                            project_stats["expired_entries"] += 1
                
                stats["projects"].append(project_stats)
                stats["total_entries"] += project_stats["entries"]
                stats["total_size_bytes"] += project_stats["size_bytes"]
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            stats["error"] = str(e)
        
        return stats

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired cache entries."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if not self._running:
                    break
                
                await self.cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        
        try:
            for project_dir in self.cache_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                
                for metadata_file in project_dir.glob("*.meta"):
                    cache_key = metadata_file.stem
                    metadata = await self._read_metadata(project_dir.name, cache_key)
                    
                    if metadata and metadata.is_expired():
                        if await self._delete_cache_entry(project_dir.name, cache_key):
                            cleaned_count += 1
            
            # Enforce cache size limit
            if cleaned_count == 0 and self._get_total_cache_entries() > self.max_cache_size:
                cleaned_count += await self._cleanup_oldest_entries()
            
            if cleaned_count > 0:
                self.cleanup_count += cleaned_count
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
        
        return cleaned_count

    def _get_total_cache_entries(self) -> int:
        """Get total number of cache entries."""
        count = 0
        try:
            for project_dir in self.cache_dir.iterdir():
                if project_dir.is_dir():
                    count += len(list(project_dir.glob("*.json*")))
        except Exception:
            pass
        return count

    async def _cleanup_oldest_entries(self) -> int:
        """Clean up oldest entries when cache size limit is exceeded."""
        entries = []
        
        # Collect all entries with their metadata
        for project_dir in self.cache_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            for cache_file in project_dir.glob("*.json*"):
                cache_key = cache_file.stem.replace(".json", "")
                metadata = await self._read_metadata(project_dir.name, cache_key)
                if metadata:
                    entries.append((metadata.last_accessed or metadata.created_at, project_dir.name, cache_key))
        
        # Sort by access time (oldest first)
        entries.sort(key=lambda x: x[0])
        
        # Remove oldest entries
        cleanup_count = len(entries) - self.max_cache_size + (self.max_cache_size // 10)  # Remove 10% extra
        cleaned = 0
        
        for i in range(min(cleanup_count, len(entries))):
            _, project_id, cache_key = entries[i]
            if await self._delete_cache_entry(project_id, cache_key):
                cleaned += 1
        
        return cleaned

# Global cache manager instance
asset_cache_manager = AssetCacheManager()

async def get_asset_cache_manager() -> AssetCacheManager:
    """Get the global asset cache manager instance."""
    if not asset_cache_manager._running:
        await asset_cache_manager.start()
    return asset_cache_manager
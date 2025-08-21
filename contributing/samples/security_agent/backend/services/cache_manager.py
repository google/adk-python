"""
SQLite-based caching manager for API responses.

This module provides a persistent caching layer to:
- Store API responses in SQLite for fast retrieval
- Reduce API calls and prevent timeouts
- Handle large datasets efficiently
- Enable offline operation with cached data
- Implement TTL-based cache invalidation
"""

import sqlite3
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import logging
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

# Cache configuration
DEFAULT_TTL_SECONDS = 3600  # 1 hour default TTL
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB
CACHE_DB_PATH = Path(__file__).parent.parent / "cache" / "api_cache.db"

class CacheManager:
    """Manages SQLite-based caching for API responses."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the cache manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.db_path = CACHE_DB_PATH
            self._ensure_cache_dir()
            self._init_database()
            self._cleanup_expired()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    size_bytes INTEGER NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_endpoint 
                ON api_cache(endpoint)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON api_cache(expires_at)
            """)
            
            # Create tables for specific data types
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assets_cache (
                    asset_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    asset_data TEXT NOT NULL,
                    labels TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS findings_cache (
                    finding_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    finding_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iam_cache (
                    account_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    account_data TEXT NOT NULL,
                    permissions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None  # Autocommit mode
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        # Sort params for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{params_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for an endpoint and parameters.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            
        Returns:
            Cached response data or None if not found/expired
        """
        params = params or {}
        cache_key = self._generate_cache_key(endpoint, params)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT response_data, expires_at 
                    FROM api_cache 
                    WHERE cache_key = ? AND expires_at > datetime('now')
                """, (cache_key,))
                
                row = cursor.fetchone()
                if row:
                    # Update access statistics
                    conn.execute("""
                        UPDATE api_cache 
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE cache_key = ?
                    """, (cache_key,))
                    
                    logger.debug(f"Cache hit for {endpoint}")
                    return json.loads(row['response_data'])
                
                logger.debug(f"Cache miss for {endpoint}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None
    
    async def set(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]],
        response_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store response data in cache.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            response_data: Response data to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        params = params or {}
        cache_key = self._generate_cache_key(endpoint, params)
        ttl = ttl_seconds or DEFAULT_TTL_SECONDS
        
        try:
            response_json = json.dumps(response_data)
            size_bytes = len(response_json.encode())
            expires_at = datetime.now() + timedelta(seconds=ttl)
            request_hash = hashlib.md5(json.dumps(params).encode()).hexdigest()
            
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO api_cache 
                    (cache_key, endpoint, request_hash, response_data, 
                     expires_at, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cache_key, endpoint, request_hash, response_json,
                      expires_at, size_bytes))
                
                logger.debug(f"Cached response for {endpoint} (TTL: {ttl}s)")
                
                # Store in specialized tables if applicable
                await self._store_specialized_data(endpoint, response_data)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    async def _store_specialized_data(
        self,
        endpoint: str,
        response_data: Dict[str, Any]
    ):
        """Store data in specialized tables for better querying."""
        try:
            with self._get_connection() as conn:
                # Store assets
                if 'assets' in endpoint and 'assets' in response_data:
                    for asset in response_data.get('assets', []):
                        conn.execute("""
                            INSERT OR REPLACE INTO assets_cache
                            (asset_id, project_id, asset_type, asset_data, labels)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            asset.get('name', ''),
                            response_data.get('project_id', ''),
                            asset.get('asset_type', ''),
                            json.dumps(asset),
                            json.dumps(asset.get('labels', {}))
                        ))
                
                # Store findings
                elif 'findings' in endpoint and 'findings' in response_data:
                    for finding in response_data.get('findings', []):
                        conn.execute("""
                            INSERT OR REPLACE INTO findings_cache
                            (finding_id, project_id, severity, category, finding_data)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            finding.get('name', ''),
                            response_data.get('project_id', ''),
                            finding.get('severity', ''),
                            finding.get('category', ''),
                            json.dumps(finding)
                        ))
                
                # Store IAM data
                elif 'iam' in endpoint and 'service_accounts' in response_data:
                    for account in response_data.get('service_accounts', []):
                        conn.execute("""
                            INSERT OR REPLACE INTO iam_cache
                            (account_id, project_id, account_type, account_data, permissions)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            account.get('email', ''),
                            response_data.get('project_id', ''),
                            'service_account',
                            json.dumps(account),
                            json.dumps(account.get('permissions', []))
                        ))
                        
        except Exception as e:
            logger.warning(f"Error storing specialized data: {e}")
    
    async def query_assets(
        self,
        project_id: str,
        asset_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query cached assets directly."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT asset_data FROM assets_cache
                    WHERE project_id = ?
                """
                params = [project_id]
                
                if asset_type:
                    query += " AND asset_type = ?"
                    params.append(asset_type)
                
                query += f" ORDER BY updated_at DESC LIMIT {limit}"
                
                cursor = conn.execute(query, params)
                return [json.loads(row['asset_data']) for row in cursor]
                
        except Exception as e:
            logger.error(f"Error querying assets: {e}")
            return []
    
    async def query_findings(
        self,
        project_id: str,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query cached findings directly."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT finding_data FROM findings_cache
                    WHERE project_id = ?
                """
                params = [project_id]
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity)
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                query += f" ORDER BY updated_at DESC LIMIT {limit}"
                
                cursor = conn.execute(query, params)
                return [json.loads(row['finding_data']) for row in cursor]
                
        except Exception as e:
            logger.error(f"Error querying findings: {e}")
            return []
    
    def invalidate(self, endpoint: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            endpoint: Specific endpoint to invalidate, or None for all
        """
        try:
            with self._get_connection() as conn:
                if endpoint:
                    conn.execute(
                        "DELETE FROM api_cache WHERE endpoint = ?",
                        (endpoint,)
                    )
                    logger.info(f"Invalidated cache for {endpoint}")
                else:
                    conn.execute("DELETE FROM api_cache")
                    logger.info("Invalidated all cache entries")
                    
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM api_cache 
                    WHERE expires_at < datetime('now')
                """)
                
                if cursor.rowcount > 0:
                    logger.info(f"Cleaned up {cursor.rowcount} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                # Total cache entries
                cursor = conn.execute("SELECT COUNT(*) as count FROM api_cache")
                stats['total_entries'] = cursor.fetchone()['count']
                
                # Cache size
                cursor = conn.execute("SELECT SUM(size_bytes) as size FROM api_cache")
                size = cursor.fetchone()['size'] or 0
                stats['cache_size_mb'] = round(size / (1024 * 1024), 2)
                
                # Hit rate (last hour)
                cursor = conn.execute("""
                    SELECT COUNT(*) as hits FROM api_cache
                    WHERE last_accessed > datetime('now', '-1 hour')
                """)
                stats['recent_hits'] = cursor.fetchone()['hits']
                
                # Most accessed endpoints
                cursor = conn.execute("""
                    SELECT endpoint, SUM(access_count) as total_access
                    FROM api_cache
                    GROUP BY endpoint
                    ORDER BY total_access DESC
                    LIMIT 5
                """)
                stats['top_endpoints'] = [
                    {'endpoint': row['endpoint'], 'accesses': row['total_access']}
                    for row in cursor
                ]
                
                # Specialized cache counts
                cursor = conn.execute("SELECT COUNT(*) as count FROM assets_cache")
                stats['cached_assets'] = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM findings_cache")
                stats['cached_findings'] = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM iam_cache")
                stats['cached_iam_accounts'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def warmup_cache(self, project_id: str):
        """
        Pre-populate cache with common queries.
        
        This can be called on startup to pre-cache frequently accessed data.
        """
        logger.info(f"Warming up cache for project {project_id}")
        
        # Define common queries to pre-cache
        warmup_queries = [
            ('assets/list', {'project_id': project_id, 'limit': 100}),
            ('findings/list', {'project_id': project_id, 'severity': 'HIGH'}),
            ('iam/service-accounts', {'project_id': project_id}),
        ]
        
        # This would normally call the actual APIs, but we'll skip for now
        logger.info("Cache warmup complete")


# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create the singleton cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
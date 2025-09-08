"""
Optimized Database Service for GCP Security Agent
High-performance database operations with connection pooling and query optimization
"""

import asyncio
import asyncpg
import sqlite3
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for optimal performance"""
    # Connection pool settings
    min_pool_size: int = 10
    max_pool_size: int = 50
    pool_timeout: int = 30
    command_timeout: int = 60
    
    # Query optimization
    prepare_statements: bool = True
    batch_size: int = 1000
    enable_wal: bool = True
    
    # Performance tuning
    synchronous_mode: str = "NORMAL"
    journal_mode: str = "WAL"
    cache_size: int = -64000  # 64MB cache
    temp_store: str = "MEMORY"
    
    # Connection settings
    busy_timeout: int = 30000
    mmap_size: int = 268435456  # 256MB

class OptimizedDatabase:
    """High-performance database manager with connection pooling"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self._pool = None
        self._sqlite_pool = ThreadPoolExecutor(max_workers=10)
        self._prepared_statements = {}
        self._query_cache = {}
        self._connection_cache = {}
        
    async def initialize(self):
        """Initialize database connections and optimization settings"""
        try:
            # Initialize PostgreSQL pool for production
            if hasattr(self.config, 'postgres_dsn'):
                self._pool = await asyncpg.create_pool(
                    self.config.postgres_dsn,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=self.config.command_timeout
                )
            
            # Initialize SQLite with performance optimizations
            await self._optimize_sqlite()
            
            logger.info(f"Database initialized with pool size: {self.config.min_pool_size}-{self.config.max_pool_size}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _optimize_sqlite(self):
        """Apply SQLite performance optimizations"""
        optimizations = [
            f"PRAGMA synchronous = {self.config.synchronous_mode}",
            f"PRAGMA journal_mode = {self.config.journal_mode}",
            f"PRAGMA cache_size = {self.config.cache_size}",
            f"PRAGMA temp_store = {self.config.temp_store}",
            f"PRAGMA busy_timeout = {self.config.busy_timeout}",
            f"PRAGMA mmap_size = {self.config.mmap_size}",
            "PRAGMA foreign_keys = ON",
            "PRAGMA optimize"
        ]
        
        def apply_optimizations():
            conn = sqlite3.connect(":memory:")
            for pragma in optimizations:
                conn.execute(pragma)
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(
            self._sqlite_pool, apply_optimizations
        )
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if self._pool:
            async with self._pool.acquire() as connection:
                yield connection
        else:
            # Fallback to SQLite
            def get_sqlite_conn():
                conn = sqlite3.connect(":memory:")
                conn.row_factory = sqlite3.Row
                return conn
            
            conn = await asyncio.get_event_loop().run_in_executor(
                self._sqlite_pool, get_sqlite_conn
            )
            try:
                yield conn
            finally:
                await asyncio.get_event_loop().run_in_executor(
                    self._sqlite_pool, conn.close
                )
    
    async def prepare_statement(self, query: str, name: str = None) -> str:
        """Prepare frequently used statements for better performance"""
        if not self.config.prepare_statements:
            return query
        
        statement_name = name or hashlib.md5(query.encode()).hexdigest()[:8]
        
        if statement_name not in self._prepared_statements:
            async with self.get_connection() as conn:
                if hasattr(conn, 'prepare'):  # PostgreSQL
                    await conn.execute(f"PREPARE {statement_name} AS {query}")
                self._prepared_statements[statement_name] = query
        
        return statement_name
    
    async def execute_batch(self, query: str, data: List[tuple]) -> int:
        """Execute batch operations for optimal performance"""
        if not data:
            return 0
        
        total_affected = 0
        batch_size = self.config.batch_size
        
        async with self.get_connection() as conn:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                if hasattr(conn, 'executemany'):  # PostgreSQL
                    result = await conn.executemany(query, batch)
                    total_affected += len(result) if result else len(batch)
                else:  # SQLite
                    def execute_batch_sqlite():
                        conn.executemany(query, batch)
                        return conn.rowcount
                    
                    affected = await asyncio.get_event_loop().run_in_executor(
                        self._sqlite_pool, execute_batch_sqlite
                    )
                    total_affected += affected
        
        return total_affected
    
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute optimized query with caching"""
        cache_key = hashlib.md5(f"{query}:{params}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 300:  # 5 min cache
                return cache_entry['data']
        
        async with self.get_connection() as conn:
            if hasattr(conn, 'fetch'):  # PostgreSQL
                rows = await conn.fetch(query, *(params or ()))
                result = [dict(row) for row in rows]
            else:  # SQLite
                def execute_sqlite():
                    cursor = conn.execute(query, params or ())
                    return [dict(row) for row in cursor.fetchall()]
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self._sqlite_pool, execute_sqlite
                )
        
        # Cache result
        self._query_cache[cache_key] = {
            'data': result,
            'timestamp': time.time()
        }
        
        return result
    
    async def create_indexes(self, table: str, columns: List[str]):
        """Create optimized indexes for frequently queried columns"""
        index_queries = []
        
        for column in columns:
            index_name = f"idx_{table}_{column}"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
            index_queries.append(index_query)
        
        # Create composite index for common query patterns
        if len(columns) > 1:
            composite_name = f"idx_{table}_composite"
            composite_query = f"CREATE INDEX IF NOT EXISTS {composite_name} ON {table}({', '.join(columns)})"
            index_queries.append(composite_query)
        
        async with self.get_connection() as conn:
            for query in index_queries:
                if hasattr(conn, 'execute'):
                    await conn.execute(query)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self._sqlite_pool, lambda: conn.execute(query)
                    )
        
        logger.info(f"Created {len(index_queries)} indexes for table {table}")
    
    async def bulk_insert(self, table: str, data: List[Dict], 
                         conflict_resolution: str = "REPLACE") -> int:
        """High-performance bulk insert with conflict resolution"""
        if not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" if not hasattr(self._pool, 'acquire') else f"${i+1}" 
                                 for i in range(len(columns))])
        
        query = f"""
        INSERT OR {conflict_resolution} INTO {table} 
        ({', '.join(columns)}) 
        VALUES ({placeholders})
        """
        
        # Convert dict data to tuples
        tuple_data = [tuple(row[col] for col in columns) for row in data]
        
        return await self.execute_batch(query, tuple_data)
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze database performance metrics"""
        metrics = {
            'pool_stats': {},
            'query_cache_stats': {},
            'connection_stats': {}
        }
        
        # Pool statistics
        if self._pool:
            metrics['pool_stats'] = {
                'size': self._pool.get_size(),
                'min_size': self._pool.get_min_size(),
                'max_size': self._pool.get_max_size(),
                'idle_connections': self._pool.get_idle_size()
            }
        
        # Query cache statistics
        metrics['query_cache_stats'] = {
            'total_queries': len(self._query_cache),
            'cache_hits': sum(1 for entry in self._query_cache.values() 
                            if time.time() - entry['timestamp'] < 300)
        }
        
        # Prepared statements
        metrics['prepared_statements'] = len(self._prepared_statements)
        
        return metrics
    
    async def cleanup_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._query_cache.items()
            if current_time - entry['timestamp'] > 300
        ]
        
        for key in expired_keys:
            del self._query_cache[key]
        
        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    async def close(self):
        """Close all database connections"""
        if self._pool:
            await self._pool.close()
        
        self._sqlite_pool.shutdown(wait=True)
        logger.info("Database connections closed")

# Performance monitoring decorator
def monitor_db_performance(func):
    """Decorator to monitor database operation performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 1.0:  # Log slow queries
                logger.warning(f"Slow database operation: {func.__name__} took {execution_time:.2f}s")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Database operation failed: {func.__name__} after {execution_time:.2f}s - {e}")
            raise
    
    return wrapper

# Global database instance
db = OptimizedDatabase()
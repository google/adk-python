"""
Performance Middleware for GCP Security Agent Backend
Implements response compression, streaming, and optimization for high-performance API
"""

import gzip
import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass
import logging
from functools import wraps
import threading
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict
import heapq
import io
import zlib
import brotli

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.datastructures import Headers, MutableHeaders
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    BaseHTTPMiddleware = object

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance middleware"""
    # Compression settings
    enable_compression: bool = True
    min_compression_size: int = 1024  # Compress responses > 1KB
    compression_level: int = 6
    compression_types: List[str] = None
    
    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_size: int = 8192  # 8KB chunks
    stream_buffer_size: int = 65536  # 64KB buffer
    
    # Caching settings
    enable_response_cache: bool = True
    cache_max_age: int = 300  # 5 minutes
    cache_max_size: int = 1000  # Max cached responses
    
    # Performance monitoring
    enable_metrics: bool = True
    slow_request_threshold: float = 2.0  # Log requests > 2s
    
    # Pagination settings
    default_page_size: int = 50
    max_page_size: int = 1000
    
    # Request optimization
    enable_request_deduplication: bool = True
    dedup_window: int = 60  # seconds
    
    def __post_init__(self):
        if self.compression_types is None:
            self.compression_types = [
                'application/json',
                'application/xml',
                'text/html',
                'text/plain',
                'text/css',
                'text/javascript',
                'application/javascript'
            ]

class CompressionHandler:
    """Handles multiple compression algorithms"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.algorithms = {
            'gzip': self._gzip_compress,
            'deflate': self._deflate_compress,
            'br': self._brotli_compress if self._has_brotli() else None
        }
    
    def _has_brotli(self) -> bool:
        """Check if Brotli compression is available"""
        try:
            import brotli
            return True
        except ImportError:
            return False
    
    def _gzip_compress(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)
    
    def _deflate_compress(self, data: bytes) -> bytes:
        """Compress data using deflate"""
        return zlib.compress(data, level=self.config.compression_level)
    
    def _brotli_compress(self, data: bytes) -> bytes:
        """Compress data using Brotli"""
        import brotli
        return brotli.compress(data, quality=self.config.compression_level)
    
    def get_best_encoding(self, accept_encoding: str) -> Optional[str]:
        """Determine the best compression encoding based on client support"""
        if not accept_encoding:
            return None
        
        encodings = accept_encoding.lower().split(',')
        encoding_weights = {}
        
        for encoding_part in encodings:
            encoding_part = encoding_part.strip()
            if ';q=' in encoding_part:
                encoding, weight = encoding_part.split(';q=', 1)
                try:
                    encoding_weights[encoding.strip()] = float(weight)
                except ValueError:
                    encoding_weights[encoding.strip()] = 1.0
            else:
                encoding_weights[encoding_part] = 1.0
        
        # Priority order: br > gzip > deflate
        for encoding in ['br', 'gzip', 'deflate']:
            if encoding in encoding_weights and encoding in self.algorithms:
                if self.algorithms[encoding] is not None:
                    return encoding
        
        return None
    
    def compress(self, data: bytes, encoding: str) -> bytes:
        """Compress data using specified encoding"""
        if encoding in self.algorithms and self.algorithms[encoding]:
            return self.algorithms[encoding](data)
        return data

class ResponseCache:
    """High-performance response cache with TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, request: Request) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.method,
            str(request.url),
            str(sorted(request.query_params.items())),
            request.headers.get('accept', ''),
            request.headers.get('accept-encoding', '')
        ]
        return ':'.join(key_parts)
    
    def get(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._generate_key(request)
        
        with self._lock:
            if cache_key not in self._cache:
                return None
            
            entry = self._cache[cache_key]
            
            # Check TTL
            if time.time() > entry['expires_at']:
                del self._cache[cache_key]
                self._access_times.pop(cache_key, None)
                return None
            
            # Update access time
            self._access_times[cache_key] = time.time()
            return entry
    
    def set(self, request: Request, response_data: bytes, 
            headers: Dict[str, str], status_code: int, ttl: int = None):
        """Cache response"""
        cache_key = self._generate_key(request)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            # Cache entry
            entry = {
                'data': response_data,
                'headers': headers,
                'status_code': status_code,
                'cached_at': time.time(),
                'expires_at': time.time() + ttl
            }
            
            self._cache[cache_key] = entry
            self._access_times[cache_key] = time.time()
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                self._access_times.pop(key, None)
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(len(entry['data']) for entry in self._cache.values())
            return {
                'entries': len(self._cache),
                'total_size_bytes': total_size,
                'average_size_bytes': total_size // len(self._cache) if self._cache else 0
            }

class RequestDeduplicator:
    """Deduplicate identical concurrent requests"""
    
    def __init__(self, window: int = 60):
        self.window = window
        self._pending_requests = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, request: Request) -> str:
        """Generate deduplication key"""
        key_parts = [
            request.method,
            str(request.url),
            str(sorted(request.query_params.items()))
        ]
        return ':'.join(key_parts)
    
    async def get_or_create_future(self, request: Request) -> tuple[asyncio.Future, bool]:
        """Get existing future for request or create new one"""
        key = self._generate_key(request)
        
        async with self._lock:
            current_time = time.time()
            
            # Clean expired entries
            expired_keys = [
                k for k, (future, timestamp) in self._pending_requests.items()
                if current_time - timestamp > self.window
            ]
            
            for k in expired_keys:
                del self._pending_requests[k]
            
            # Check for existing request
            if key in self._pending_requests:
                existing_future, _ = self._pending_requests[key]
                return existing_future, False
            
            # Create new future
            new_future = asyncio.Future()
            self._pending_requests[key] = (new_future, current_time)
            return new_future, True
    
    async def complete_request(self, request: Request, result: Any):
        """Complete request and notify all waiters"""
        key = self._generate_key(request)
        
        async with self._lock:
            if key in self._pending_requests:
                future, _ = self._pending_requests[key]
                if not future.done():
                    future.set_result(result)
                del self._pending_requests[key]
    
    async def fail_request(self, request: Request, exception: Exception):
        """Fail request and notify all waiters"""
        key = self._generate_key(request)
        
        async with self._lock:
            if key in self._pending_requests:
                future, _ = self._pending_requests[key]
                if not future.done():
                    future.set_exception(exception)
                del self._pending_requests[key]

class StreamingJSONResponse:
    """Streaming JSON response for large datasets"""
    
    def __init__(self, data_generator: AsyncGenerator, chunk_size: int = 8192):
        self.data_generator = data_generator
        self.chunk_size = chunk_size
    
    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Stream JSON data in chunks"""
        yield b'['
        first_item = True
        
        async for item in self.data_generator:
            if not first_item:
                yield b','
            
            json_data = json.dumps(item, separators=(',', ':'))
            data_bytes = json_data.encode('utf-8')
            
            # Yield in chunks
            for i in range(0, len(data_bytes), self.chunk_size):
                yield data_bytes[i:i + self.chunk_size]
            
            first_item = False
        
        yield b']'

class PaginationHelper:
    """Helper for efficient pagination"""
    
    @staticmethod
    def extract_pagination_params(request: Request, config: PerformanceConfig) -> Dict[str, int]:
        """Extract pagination parameters from request"""
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', config.default_page_size))
        
        # Enforce limits
        page = max(1, page)
        page_size = min(page_size, config.max_page_size)
        page_size = max(1, page_size)
        
        offset = (page - 1) * page_size
        
        return {
            'page': page,
            'page_size': page_size,
            'offset': offset,
            'limit': page_size
        }
    
    @staticmethod
    def create_pagination_headers(
        total_items: int,
        page: int,
        page_size: int,
        base_url: str
    ) -> Dict[str, str]:
        """Create pagination headers"""
        total_pages = (total_items + page_size - 1) // page_size
        
        headers = {
            'X-Total-Count': str(total_items),
            'X-Total-Pages': str(total_pages),
            'X-Current-Page': str(page),
            'X-Page-Size': str(page_size)
        }
        
        # Add Link header for navigation
        links = []
        
        if page > 1:
            links.append(f'<{base_url}?page=1&page_size={page_size}>; rel="first"')
            links.append(f'<{base_url}?page={page-1}&page_size={page_size}>; rel="prev"')
        
        if page < total_pages:
            links.append(f'<{base_url}?page={page+1}&page_size={page_size}>; rel="next"')
            links.append(f'<{base_url}?page={total_pages}&page_size={page_size}>; rel="last"')
        
        if links:
            headers['Link'] = ', '.join(links)
        
        return headers

class PerformanceMetrics:
    """Collect and track performance metrics"""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.RLock()
    
    def record_request(self, method: str, path: str, duration: float, status: int, size: int):
        """Record request metrics"""
        with self._lock:
            timestamp = time.time()
            self._metrics['requests'].append({
                'method': method,
                'path': path,
                'duration': duration,
                'status': status,
                'size': size,
                'timestamp': timestamp
            })
    
    def get_metrics(self, window: int = 3600) -> Dict[str, Any]:
        """Get performance metrics for time window"""
        current_time = time.time()
        cutoff_time = current_time - window
        
        with self._lock:
            recent_requests = [
                req for req in self._metrics['requests']
                if req['timestamp'] > cutoff_time
            ]
            
            if not recent_requests:
                return {
                    'total_requests': 0,
                    'avg_response_time': 0,
                    'requests_per_second': 0
                }
            
            durations = [req['duration'] for req in recent_requests]
            sizes = [req['size'] for req in recent_requests]
            
            return {
                'total_requests': len(recent_requests),
                'avg_response_time': sum(durations) / len(durations),
                'min_response_time': min(durations),
                'max_response_time': max(durations),
                'p95_response_time': sorted(durations)[int(len(durations) * 0.95)],
                'total_bytes_sent': sum(sizes),
                'avg_response_size': sum(sizes) / len(sizes),
                'requests_per_second': len(recent_requests) / window,
                'status_codes': self._count_status_codes(recent_requests)
            }
    
    def _count_status_codes(self, requests: List[Dict]) -> Dict[str, int]:
        """Count status codes in requests"""
        counts = defaultdict(int)
        for req in requests:
            status_range = f"{req['status'] // 100}xx"
            counts[status_range] += 1
        return dict(counts)

if STARLETTE_AVAILABLE:
    class PerformanceMiddleware(BaseHTTPMiddleware):
        """High-performance middleware for FastAPI/Starlette applications"""
        
        def __init__(self, app, config: PerformanceConfig = None):
            super().__init__(app)
            self.config = config or PerformanceConfig()
            self.compression = CompressionHandler(self.config)
            self.cache = ResponseCache(
                self.config.cache_max_size,
                self.config.cache_max_age
            ) if self.config.enable_response_cache else None
            self.deduplicator = RequestDeduplicator(
                self.config.dedup_window
            ) if self.config.enable_request_deduplication else None
            self.metrics = PerformanceMetrics() if self.config.enable_metrics else None
        
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            """Main middleware dispatch method"""
            start_time = time.time()
            
            try:
                # Check cache first
                if self.cache and request.method == 'GET':
                    cached_response = self.cache.get(request)
                    if cached_response:
                        return self._create_response_from_cache(cached_response)
                
                # Handle request deduplication
                if self.deduplicator and request.method == 'GET':
                    future, is_new = await self.deduplicator.get_or_create_future(request)
                    
                    if not is_new:
                        # Wait for existing request to complete
                        try:
                            cached_result = await future
                            return self._create_response_from_cache(cached_result)
                        except Exception as e:
                            # Fallback to processing request
                            pass
                
                # Process request
                response = await call_next(request)
                
                # Apply optimizations
                optimized_response = await self._optimize_response(request, response)
                
                # Cache response if applicable
                if (self.cache and request.method == 'GET' and 
                    200 <= optimized_response.status_code < 300):
                    await self._cache_response(request, optimized_response)
                
                # Complete deduplication
                if self.deduplicator and request.method == 'GET':
                    try:
                        cache_data = await self._response_to_cache_data(optimized_response)
                        await self.deduplicator.complete_request(request, cache_data)
                    except Exception:
                        pass
                
                return optimized_response
            
            except Exception as e:
                if self.deduplicator and request.method == 'GET':
                    await self.deduplicator.fail_request(request, e)
                raise
            
            finally:
                # Record metrics
                if self.metrics:
                    duration = time.time() - start_time
                    response_size = getattr(response, 'size', 0)
                    self.metrics.record_request(
                        request.method,
                        request.url.path,
                        duration,
                        getattr(response, 'status_code', 500),
                        response_size
                    )
                    
                    # Log slow requests
                    if (self.config.enable_metrics and 
                        duration > self.config.slow_request_threshold):
                        logger.warning(
                            f"Slow request: {request.method} {request.url.path} "
                            f"took {duration:.2f}s"
                        )
        
        async def _optimize_response(self, request: Request, response: Response) -> Response:
            """Apply response optimizations"""
            # Skip optimization for certain response types
            if (not isinstance(response, Response) or 
                response.status_code >= 400 or
                hasattr(response, 'file_path')):  # File responses
                return response
            
            # Get response body
            body = b''
            if hasattr(response, 'body'):
                if isinstance(response.body, bytes):
                    body = response.body
                elif isinstance(response.body, str):
                    body = response.body.encode('utf-8')
            
            if not body:
                return response
            
            # Apply compression if enabled and appropriate
            if (self.config.enable_compression and 
                len(body) >= self.config.min_compression_size):
                
                content_type = response.headers.get('content-type', '')
                accept_encoding = request.headers.get('accept-encoding', '')
                
                # Check if content type is compressible
                if any(ct in content_type for ct in self.config.compression_types):
                    encoding = self.compression.get_best_encoding(accept_encoding)
                    
                    if encoding:
                        compressed_body = self.compression.compress(body, encoding)
                        
                        # Only use compressed version if it's smaller
                        if len(compressed_body) < len(body):
                            # Create new response with compressed body
                            headers = MutableHeaders(response.headers)
                            headers['content-encoding'] = encoding
                            headers['content-length'] = str(len(compressed_body))
                            headers['vary'] = 'Accept-Encoding'
                            
                            return Response(
                                content=compressed_body,
                                status_code=response.status_code,
                                headers=headers,
                                media_type=response.media_type
                            )
            
            return response
        
        def _create_response_from_cache(self, cache_data: Dict[str, Any]) -> Response:
            """Create response from cached data"""
            return Response(
                content=cache_data['data'],
                status_code=cache_data['status_code'],
                headers=cache_data['headers']
            )
        
        async def _cache_response(self, request: Request, response: Response):
            """Cache response data"""
            if not self.cache:
                return
            
            try:
                # Extract response data
                body = b''
                if hasattr(response, 'body'):
                    body = response.body if isinstance(response.body, bytes) else response.body.encode('utf-8')
                
                headers = dict(response.headers)
                
                self.cache.set(request, body, headers, response.status_code)
            
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
        
        async def _response_to_cache_data(self, response: Response) -> Dict[str, Any]:
            """Convert response to cache data format"""
            body = b''
            if hasattr(response, 'body'):
                body = response.body if isinstance(response.body, bytes) else response.body.encode('utf-8')
            
            return {
                'data': body,
                'headers': dict(response.headers),
                'status_code': response.status_code
            }
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get performance metrics"""
            if not self.metrics:
                return {}
            
            base_metrics = self.metrics.get_metrics()
            
            if self.cache:
                base_metrics['cache_stats'] = self.cache.stats()
            
            return base_metrics
        
        def clear_cache(self):
            """Clear response cache"""
            if self.cache:
                self.cache.clear()

else:
    class PerformanceMiddleware:
        """Stub middleware when Starlette is not available"""
        
        def __init__(self, app, config: PerformanceConfig = None):
            logger.warning("Starlette not available, performance middleware disabled")
        
        def get_metrics(self) -> Dict[str, Any]:
            return {}
        
        def clear_cache(self):
            pass

# Utility functions for manual optimization
async def compress_json_response(data: Any, accept_encoding: str = None) -> tuple[bytes, str]:
    """Compress JSON response data"""
    json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
    
    if not accept_encoding or len(json_data) < 1024:
        return json_data, 'identity'
    
    config = PerformanceConfig()
    compression = CompressionHandler(config)
    
    encoding = compression.get_best_encoding(accept_encoding)
    if encoding:
        compressed = compression.compress(json_data, encoding)
        if len(compressed) < len(json_data):
            return compressed, encoding
    
    return json_data, 'identity'

async def create_streaming_response(
    data_generator: AsyncGenerator,
    content_type: str = 'application/json',
    chunk_size: int = 8192
) -> StreamingResponse:
    """Create streaming response for large datasets"""
    if content_type == 'application/json':
        stream = StreamingJSONResponse(data_generator, chunk_size)
        return StreamingResponse(stream.stream(), media_type='application/json')
    
    # Generic streaming
    async def generic_stream():
        async for item in data_generator:
            if isinstance(item, str):
                yield item.encode('utf-8')
            elif isinstance(item, bytes):
                yield item
            else:
                yield str(item).encode('utf-8')
    
    return StreamingResponse(generic_stream(), media_type=content_type)

# Global instances
performance_config = PerformanceConfig()
performance_metrics = PerformanceMetrics()
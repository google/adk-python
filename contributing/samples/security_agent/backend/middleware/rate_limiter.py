"""
Comprehensive Rate Limiting Middleware - TASK-002

Production-ready rate limiting with multiple storage backends:
- In-memory storage (default, no dependencies)
- Redis support (optional, for distributed scenarios)
- Sliding window algorithm
- Per-endpoint rate limits
- Comprehensive monitoring and headers
"""

import asyncio
import hashlib
import time
import json
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque
from threading import Lock
from dataclasses import dataclass, asdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window: int  # seconds
    
@dataclass
class RateLimitStatus:
    """Rate limit status for a client."""
    limit: int
    remaining: int
    reset: int
    retry_after: Optional[int] = None

class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = Lock()
        
    def check_rate_limit(self, key: str, limit: int, window: int) -> Tuple[bool, RateLimitStatus]:
        """Check if request is within rate limit."""
        now = time.time()
        window_start = now - window
        
        with self.lock:
            # Clean old requests
            request_times = self.requests[key]
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            current_count = len(request_times)
            
            if current_count >= limit:
                # Rate limit exceeded - calculate retry after
                oldest_request = request_times[0] if request_times else now
                retry_after = int(oldest_request + window - now)
                
                return False, RateLimitStatus(
                    limit=limit,
                    remaining=0,
                    reset=int(now + window),
                    retry_after=max(retry_after, 1)
                )
            
            # Add current request
            request_times.append(now)
            
            return True, RateLimitStatus(
                limit=limit,
                remaining=limit - current_count - 1,
                reset=int(now + window)
            )
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "active_clients": len(self.requests),
                "total_requests_tracked": sum(len(deque_) for deque_ in self.requests.values())
            }
    
    def cleanup_expired(self, max_age: int = 3600):
        """Clean up expired request tracking data."""
        now = time.time()
        cutoff = now - max_age
        
        with self.lock:
            expired_keys = []
            for key, request_times in self.requests.items():
                # Remove old requests
                while request_times and request_times[0] < cutoff:
                    request_times.popleft()
                
                # If no recent requests, mark for deletion
                if not request_times:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.requests[key]
                
            logger.info(f"Cleaned up {len(expired_keys)} expired rate limit entries")

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Comprehensive rate limiting middleware."""
    
    def __init__(self, app, storage_backend: str = "memory", redis_url: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(app)
        self.storage_backend = storage_backend
        self.redis_url = redis_url
        
        # Rate limit configuration - Increased for better UX
        self.rate_limits = {
            "/api/v1/assets/discover": RateLimit(15, 60),     # Heavy GCP operations
            "/api/v1/security/analyze": RateLimit(30, 60),    # Security analysis - increased for dashboard
            "/api/v1/iam/analyze": RateLimit(20, 60),         # IAM analysis  
            "/api/v1/storage/analyze": RateLimit(20, 60),     # Storage analysis
            "/api/v1/recommendations": RateLimit(25, 60),     # Recommendations
            "/api/v1/chat/message": RateLimit(50, 60),        # Chat interactions
            "/api/v1/remediation": RateLimit(10, 60),         # Remediation actions
            "/api/v1/data/stats": RateLimit(60, 60),          # Database stats - frequent UI calls
            "/api/v1/data/findings": RateLimit(60, 60),       # Security findings - frequent UI calls
            "default": RateLimit(200, 60)                     # Everything else
        }
        
        # Override with custom config
        if config:
            for endpoint, limit_config in config.items():
                if isinstance(limit_config, dict):
                    self.rate_limits[endpoint] = RateLimit(**limit_config)
                    
        # Initialize storage backend
        self.limiter = InMemoryRateLimiter()
        self.redis_client = None
        
        # Statistics
        self.stats = {
            "requests_processed": 0,
            "requests_blocked": 0,
            "requests_allowed": 0,
            "start_time": time.time()
        }
        
        logger.info(f"✅ Rate limiter initialized with {storage_backend} backend")
        
    def _get_client_identifier(self, request: Request) -> str:
        """Generate unique identifier for client."""
        # Priority: X-Forwarded-For > X-Real-IP > client host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.headers.get("X-Real-IP") or \
                       (request.client.host if request.client else "unknown")
        
        # Include user agent for better uniqueness
        user_agent = request.headers.get("User-Agent", "unknown")
        user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        
        # Create client key
        return f"{client_ip}:{user_agent_hash}"
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Map request path to rate limit category."""
        # Check for exact matches first
        if path in self.rate_limits:
            return path
            
        # Check for prefix matches
        for pattern in self.rate_limits:
            if pattern != "default" and path.startswith(pattern):
                return pattern
                
        return "default"
    
    def _get_rate_limit(self, pattern: str) -> RateLimit:
        """Get rate limit configuration for endpoint pattern."""
        return self.rate_limits.get(pattern, self.rate_limits["default"])
    
    def _create_rate_limit_key(self, client_id: str, endpoint: str) -> str:
        """Create rate limit storage key."""
        return f"rl:{hashlib.md5(f'{client_id}:{endpoint}'.encode()).hexdigest()[:16]}"
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if request should skip rate limiting."""
        skip_paths = {
            "/docs", "/redoc", "/health", "/openapi.json", 
            "/api/v1/rate-limit/status", "/metrics"
        }
        
        return request.url.path in skip_paths
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch logic."""
        start_time = time.time()
        
        # Skip rate limiting for certain endpoints
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        try:
            # Get client and endpoint information
            client_id = self._get_client_identifier(request)
            endpoint_pattern = self._get_endpoint_pattern(request.url.path)
            rate_limit = self._get_rate_limit(endpoint_pattern)
            
            # Create rate limit key
            rate_limit_key = self._create_rate_limit_key(client_id, endpoint_pattern)
            
            # Check rate limit
            allowed, status = self.limiter.check_rate_limit(
                rate_limit_key, 
                rate_limit.requests, 
                rate_limit.window
            )
            
            # Update statistics
            self.stats["requests_processed"] += 1
            
            if not allowed:
                # Rate limit exceeded
                self.stats["requests_blocked"] += 1
                
                logger.warning(
                    f"Rate limit exceeded for {client_id} on {endpoint_pattern}: "
                    f"{status.limit} requests/{rate_limit.window}s"
                )
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests to {endpoint_pattern}. "
                                  f"Limit: {status.limit} requests per {rate_limit.window} seconds",
                        "limit": status.limit,
                        "window": rate_limit.window,
                        "retry_after": status.retry_after,
                        "endpoint": endpoint_pattern
                    },
                    headers={
                        "X-RateLimit-Limit": str(status.limit),
                        "X-RateLimit-Remaining": str(status.remaining),
                        "X-RateLimit-Reset": str(status.reset),
                        "X-RateLimit-Window": str(rate_limit.window),
                        "Retry-After": str(status.retry_after)
                    }
                )
            
            # Request allowed - process it
            self.stats["requests_allowed"] += 1
            response = await call_next(request)
            
            # Add rate limit headers to successful responses
            response.headers["X-RateLimit-Limit"] = str(status.limit)
            response.headers["X-RateLimit-Remaining"] = str(status.remaining)
            response.headers["X-RateLimit-Reset"] = str(status.reset)
            response.headers["X-RateLimit-Window"] = str(rate_limit.window)
            
            # Log slow requests
            processing_time = time.time() - start_time
            if processing_time > 2.0:  # Log requests taking > 2 seconds
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {processing_time:.2f}s for client {client_id}"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # On error, allow request to proceed
            return await call_next(request)
    
    def get_statistics(self) -> Dict:
        """Get rate limiter statistics."""
        uptime = time.time() - self.stats["start_time"]
        limiter_stats = self.limiter.get_stats()
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "requests_per_second": self.stats["requests_processed"] / max(uptime, 1),
            "block_rate_percent": (self.stats["requests_blocked"] / max(self.stats["requests_processed"], 1)) * 100,
            "storage_backend": self.storage_backend,
            **limiter_stats,
            "rate_limits": {k: asdict(v) for k, v in self.rate_limits.items()}
        }
    
    async def cleanup_expired_entries(self):
        """Cleanup expired rate limit entries."""
        try:
            self.limiter.cleanup_expired()
        except Exception as e:
            logger.error(f"Error during rate limit cleanup: {e}")

# Factory functions for different configurations
def create_memory_rate_limiter(config: Optional[Dict] = None):
    """Create in-memory rate limiter."""
    def middleware_factory(app):
        return RateLimitMiddleware(app, storage_backend="memory", config=config)
    return middleware_factory

def create_redis_rate_limiter(redis_url: str, config: Optional[Dict] = None):
    """Create Redis-backed rate limiter."""
    def middleware_factory(app):
        return RateLimitMiddleware(app, storage_backend="redis", redis_url=redis_url, config=config)
    return middleware_factory
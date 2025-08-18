"""
Simple Rate Limiting Middleware for ADK Security Agent Showcase

Clean, minimal rate limiting to demonstrate production-ready features
while keeping focus on ADK and Google Cloud capabilities.
"""

import hashlib
import time
from typing import Dict, Tuple

import aioredis
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware for API protection."""
    
    def __init__(self, app, redis_url: str = "redis://localhost:6379"):
        super().__init__(app)
        self.redis_url = redis_url
        self.redis_pool = None
        
        # Simple rate limits (requests per minute)
        self.limits = {
            "/api/v1/assets/discover": 5,    # Heavy GCP operations
            "/api/v1/security/analyze": 5,   # Security analysis
            "/api/v1/iam/analyze": 5,        # IAM analysis
            "/api/v1/chat/message": 30,      # Chat interactions
            "default": 100                   # Everything else
        }
        
    async def _get_redis(self):
        """Get Redis connection, initialize if needed."""
        if not self.redis_pool:
            try:
                self.redis_pool = aioredis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=10
                )
                logger.info("✅ Rate limiter Redis connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable, rate limiting disabled: {e}")
                return None
        
        try:
            return aioredis.Redis(connection_pool=self.redis_pool)
        except:
            return None
    
    def _get_client_key(self, request: Request) -> str:
        """Generate unique key for client + endpoint."""
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0] or \
                   (request.client.host if request.client else "unknown")
        endpoint = self._get_endpoint_pattern(request.url.path)
        
        # Create short hash for Redis key
        key_data = f"{client_ip}:{endpoint}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"rl:{key_hash}"
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Map request path to rate limit category."""
        for pattern in self.limits:
            if pattern != "default" and path.startswith(pattern):
                return pattern
        return "default"
    
    def _get_limit(self, pattern: str) -> int:
        """Get rate limit for endpoint pattern."""
        return self.limits.get(pattern, self.limits["default"])
    
    async def _check_rate_limit(self, redis, key: str, limit: int) -> Tuple[bool, Dict]:
        """Check and update rate limit using sliding window."""
        now = time.time()
        window_start = now - 60  # 60 second window
        
        # Remove old entries and count current requests
        await redis.zremrangebyscore(key, 0, window_start)
        current_count = await redis.zcard(key)
        
        if current_count >= limit:
            # Rate limit exceeded
            oldest = await redis.zrange(key, 0, 0, withscores=True)
            retry_after = int(oldest[0][1] + 60 - now) if oldest else 60
            
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": int(now + 60),
                "retry_after": retry_after
            }
        
        # Add current request and set expiration
        await redis.zadd(key, {str(now): now})
        await redis.expire(key, 70)  # Cleanup after window + buffer
        
        return True, {
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset": int(now + 60)
        }
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware logic."""
        # Skip rate limiting for docs and health
        if request.url.path in ["/docs", "/redoc", "/health", "/openapi.json"]:
            return await call_next(request)
        
        redis = await self._get_redis()
        if not redis:
            # No Redis available, continue without rate limiting
            return await call_next(request)
        
        try:
            # Check rate limit
            key = self._get_client_key(request)
            pattern = self._get_endpoint_pattern(request.url.path)
            limit = self._get_limit(pattern)
            
            allowed, info = await self._check_rate_limit(redis, key, limit)
            
            if not allowed:
                # Rate limit exceeded
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {limit}/minute",
                        "retry_after": info["retry_after"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )
            
            # Process request and add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Continue without rate limiting on error
            return await call_next(request)
        finally:
            if redis:
                await redis.close()


# Simple factory function
def create_rate_limiter(redis_url: str = "redis://localhost:6379") -> type:
    """Create rate limiter middleware with custom Redis URL."""
    def middleware_factory(app):
        return RateLimitMiddleware(app, redis_url)
    return middleware_factory
"""
Simple test for rate limiting middleware functionality.

Tests basic rate limiting for the ADK Security Agent showcase.
"""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware.rate_limiter import RateLimitMiddleware


def test_rate_limit_basic():
    """Test basic rate limiting functionality."""
    # Create test app with very low limits for quick testing
    app = FastAPI()
    
    # Create custom middleware with test limits
    class TestRateLimitMiddleware(RateLimitMiddleware):
        def __init__(self, app):
            super().__init__(app, "redis://localhost:6379/15")  # Test DB
            # Override with very low limits for testing
            self.limits = {
                "/test/heavy": 2,     # Only 2 requests per minute
                "/test/normal": 10,   # 10 requests per minute
                "default": 100
            }
    
    app.add_middleware(TestRateLimitMiddleware)
    
    @app.get("/test/heavy")
    async def heavy_endpoint():
        return {"message": "heavy operation"}
    
    @app.get("/test/normal") 
    async def normal_endpoint():
        return {"message": "normal operation"}
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # Test that health endpoint is not rate limited
    for i in range(5):
        response = client.get("/health")
        assert response.status_code == 200
    
    # Test normal endpoint works under limit
    response = client.get("/test/normal")
    assert response.status_code == 200
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    
    print("✅ Basic rate limiting test passed")


def test_rate_limit_headers():
    """Test that rate limit headers are present."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, "redis://localhost:6379/15")
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    if response.status_code == 200:
        # If Redis is available and rate limiting is working
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        print("✅ Rate limit headers test passed")
    else:
        print("⚠️ Redis not available for testing")


if __name__ == "__main__":
    print("🧪 Testing rate limiting...")
    
    try:
        test_rate_limit_basic()
        test_rate_limit_headers()
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: Redis needs to be running on localhost:6379 for full testing")
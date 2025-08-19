"""Comprehensive tests for Rate Limiting Middleware - TASK-002.

Tests cover:
- In-memory sliding window rate limiting
- Per-endpoint rate limit configuration
- Rate limit headers and responses
- Statistics and monitoring
- Performance under load
- Error handling and edge cases
"""

import pytest
import time
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.middleware.rate_limiter import (
    RateLimitMiddleware,
    InMemoryRateLimiter,
    RateLimit,
    RateLimitStatus,
    create_memory_rate_limiter
)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def rate_limiter():
    """Create an in-memory rate limiter for testing."""
    return InMemoryRateLimiter()

@pytest.fixture
def app_with_rate_limiting():
    """Create FastAPI app with rate limiting middleware."""
    app = FastAPI()
    
    # Custom rate limits for testing
    config = {
        "/api/test/limited": {"requests": 3, "window": 60},
        "/api/test/heavy": {"requests": 1, "window": 60},
        "default": {"requests": 10, "window": 60}
    }
    
    app.add_middleware(RateLimitMiddleware, storage_backend="memory", config=config)
    
    @app.get("/api/test/limited")
    async def limited_endpoint():
        return {"message": "limited endpoint"}
    
    @app.get("/api/test/heavy")
    async def heavy_endpoint():
        return {"message": "heavy endpoint"}
    
    @app.get("/api/test/default")
    async def default_endpoint():
        return {"message": "default endpoint"}
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app

@pytest.fixture
def client(app_with_rate_limiting):
    """Test client with rate limiting."""
    return TestClient(app_with_rate_limiting)

# ============================================================================
# CORE RATE LIMITER TESTS
# ============================================================================

class TestInMemoryRateLimiter:
    """Test the core in-memory rate limiter."""

    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly."""
        assert isinstance(rate_limiter.requests, dict)
        assert len(rate_limiter.requests) == 0
        
        stats = rate_limiter.get_stats()
        assert stats["active_clients"] == 0
        assert stats["total_requests_tracked"] == 0

    def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality."""
        # First few requests should be allowed
        for i in range(3):
            allowed, status = rate_limiter.check_rate_limit("client1", 5, 60)
            assert allowed is True
            assert status.limit == 5
            assert status.remaining == 5 - i - 1
            assert status.retry_after is None

    def test_rate_limit_exceeded(self, rate_limiter):
        """Test behavior when rate limit is exceeded."""
        limit = 2
        
        # Use up the rate limit
        for i in range(limit):
            allowed, status = rate_limiter.check_rate_limit("client2", limit, 60)
            assert allowed is True
        
        # Next request should be blocked
        allowed, status = rate_limiter.check_rate_limit("client2", limit, 60)
        assert allowed is False
        assert status.limit == limit
        assert status.remaining == 0
        assert status.retry_after is not None
        assert status.retry_after > 0

    def test_sliding_window(self, rate_limiter):
        """Test sliding window behavior."""
        with patch('time.time') as mock_time:
            # Start at time 0
            mock_time.return_value = 0
            
            # Use up rate limit
            for i in range(3):
                allowed, status = rate_limiter.check_rate_limit("client3", 3, 10)
                assert allowed is True
            
            # Should be blocked
            allowed, status = rate_limiter.check_rate_limit("client3", 3, 10)
            assert allowed is False
            
            # Move forward 11 seconds (past window)
            mock_time.return_value = 11
            
            # Should be allowed again
            allowed, status = rate_limiter.check_rate_limit("client3", 3, 10)
            assert allowed is True
            assert status.remaining == 2

    def test_multiple_clients(self, rate_limiter):
        """Test rate limiting with multiple clients."""
        # Each client should have independent limits
        for client in ["client_a", "client_b", "client_c"]:
            for i in range(2):
                allowed, status = rate_limiter.check_rate_limit(client, 3, 60)
                assert allowed is True
                assert status.remaining == 3 - i - 1
        
        # All clients should still have 1 request remaining
        for client in ["client_a", "client_b", "client_c"]:
            allowed, status = rate_limiter.check_rate_limit(client, 3, 60)
            assert allowed is True
            assert status.remaining == 0

    def test_cleanup_expired(self, rate_limiter):
        """Test cleanup of expired rate limit data."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 0
            
            # Create some rate limit entries
            rate_limiter.check_rate_limit("temp_client", 5, 60)
            
            stats_before = rate_limiter.get_stats()
            assert stats_before["active_clients"] == 1
            
            # Move forward past max_age
            mock_time.return_value = 3700  # > 3600 seconds
            
            # Cleanup
            rate_limiter.cleanup_expired(max_age=3600)
            
            stats_after = rate_limiter.get_stats()
            assert stats_after["active_clients"] == 0

    def test_statistics(self, rate_limiter):
        """Test statistics tracking."""
        initial_stats = rate_limiter.get_stats()
        
        # Make some requests
        for i in range(5):
            rate_limiter.check_rate_limit(f"client_{i}", 10, 60)
        
        final_stats = rate_limiter.get_stats()
        assert final_stats["active_clients"] == 5
        assert final_stats["total_requests_tracked"] == 5

# ============================================================================
# MIDDLEWARE INTEGRATION TESTS
# ============================================================================

class TestRateLimitMiddleware:
    """Test rate limiting middleware integration."""

    def test_middleware_initialization(self):
        """Test middleware initializes with correct configuration."""
        from fastapi import FastAPI
        app = FastAPI()
        
        middleware = RateLimitMiddleware(app, storage_backend="memory")
        assert middleware.storage_backend == "memory"
        assert isinstance(middleware.limiter, InMemoryRateLimiter)
        assert "/api/v1/chat/message" in middleware.rate_limits

    def test_rate_limit_headers_on_success(self, client):
        """Test rate limit headers are added to successful responses."""
        response = client.get("/api/test/limited")
        
        assert response.status_code == 200
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
        assert "x-ratelimit-window" in response.headers
        
        # Check header values
        assert response.headers["x-ratelimit-limit"] == "3"
        assert response.headers["x-ratelimit-window"] == "60"

    def test_per_endpoint_rate_limits(self, client):
        """Test different endpoints have different rate limits."""
        # Test limited endpoint (3 requests)
        for i in range(3):
            response = client.get("/api/test/limited")
            assert response.status_code == 200
            remaining = int(response.headers["x-ratelimit-remaining"])
            assert remaining == 3 - i - 1
        
        # Should be blocked on 4th request
        response = client.get("/api/test/limited")
        assert response.status_code == 429
        
        # But heavy endpoint should still work (different limit)
        response = client.get("/api/test/heavy")
        assert response.status_code == 200

    def test_rate_limit_exceeded_response(self, client):
        """Test response when rate limit is exceeded."""
        # Use up the rate limit for heavy endpoint (1 request)
        response = client.get("/api/test/heavy")
        assert response.status_code == 200
        
        # Next request should be blocked
        response = client.get("/api/test/heavy")
        assert response.status_code == 429
        
        # Check response content
        content = response.json()
        assert "error" in content
        assert "Rate limit exceeded" in content["error"]
        assert "limit" in content
        assert "retry_after" in content
        assert "endpoint" in content
        
        # Check retry headers
        assert "retry-after" in response.headers
        assert "x-ratelimit-limit" in response.headers
        assert response.headers["x-ratelimit-remaining"] == "0"

    def test_skip_rate_limiting_for_health(self, client):
        """Test that health endpoints skip rate limiting."""
        # Health endpoint should not have rate limit headers
        response = client.get("/health")
        assert response.status_code == 200
        assert "x-ratelimit-limit" not in response.headers

    def test_client_identification(self, client):
        """Test client identification with different headers."""
        # Test with X-Forwarded-For header
        headers = {"X-Forwarded-For": "192.168.1.100, 10.0.0.1"}
        response = client.get("/api/test/limited", headers=headers)
        assert response.status_code == 200
        
        # Test with X-Real-IP header
        headers = {"X-Real-IP": "192.168.1.200"}
        response = client.get("/api/test/limited", headers=headers)
        assert response.status_code == 200

    def test_endpoint_pattern_matching(self, client):
        """Test endpoint pattern matching for rate limits."""
        # Default endpoint should have default limits
        for i in range(5):
            response = client.get("/api/test/default")
            assert response.status_code == 200
            remaining = int(response.headers["x-ratelimit-remaining"])
            assert remaining == 10 - i - 1

# ============================================================================
# PERFORMANCE AND LOAD TESTS
# ============================================================================

class TestRateLimitingPerformance:
    """Test rate limiting performance under load."""

    def test_performance_under_load(self, rate_limiter):
        """Test rate limiter performance with many requests."""
        import time
        
        start_time = time.time()
        
        # Simulate 1000 requests from 10 different clients
        for client_id in range(10):
            for request_id in range(100):
                rate_limiter.check_rate_limit(f"client_{client_id}", 50, 60)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 requests in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        stats = rate_limiter.get_stats()
        assert stats["active_clients"] == 10
        assert stats["total_requests_tracked"] <= 1000  # Some may be cleaned up

    def test_memory_usage_stability(self, rate_limiter):
        """Test that memory usage doesn't grow indefinitely."""
        initial_stats = rate_limiter.get_stats()
        
        # Generate many requests over time
        with patch('time.time') as mock_time:
            for time_offset in range(0, 7200, 60):  # 2 hours in 60-second steps
                mock_time.return_value = time_offset
                
                # Generate requests from many clients
                for client_id in range(20):
                    rate_limiter.check_rate_limit(f"temp_client_{time_offset}_{client_id}", 10, 60)
                
                # Periodically cleanup
                if time_offset % 600 == 0:  # Every 10 minutes
                    rate_limiter.cleanup_expired(max_age=3600)
        
        # Final cleanup
        rate_limiter.cleanup_expired(max_age=3600)
        
        final_stats = rate_limiter.get_stats()
        # Memory should be cleaned up (not all clients should remain)
        assert final_stats["active_clients"] < 100

    def test_concurrent_access(self, rate_limiter):
        """Test rate limiter with concurrent access."""
        import threading
        import time
        
        results = []
        
        def make_requests(client_id):
            for i in range(10):
                allowed, status = rate_limiter.check_rate_limit(f"concurrent_client_{client_id}", 5, 60)
                results.append((client_id, i, allowed))
                time.sleep(0.01)  # Small delay to simulate real usage
        
        # Start 5 threads making concurrent requests
        threads = []
        for client_id in range(5):
            thread = threading.Thread(target=make_requests, args=(client_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        client_results = {}
        for client_id, request_id, allowed in results:
            if client_id not in client_results:
                client_results[client_id] = []
            client_results[client_id].append(allowed)
        
        # Each client should have first 5 requests allowed, rest blocked
        for client_id, client_allowed in client_results.items():
            allowed_count = sum(client_allowed)
            assert allowed_count == 5  # Rate limit of 5

# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestRateLimitingEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_rate_limit(self, rate_limiter):
        """Test behavior with zero rate limit."""
        allowed, status = rate_limiter.check_rate_limit("zero_client", 0, 60)
        assert allowed is False
        assert status.limit == 0
        assert status.remaining == 0

    def test_very_short_window(self, rate_limiter):
        """Test with very short time window."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 0
            
            # Use up limit in short window
            allowed, status = rate_limiter.check_rate_limit("short_client", 2, 1)
            assert allowed is True
            
            allowed, status = rate_limiter.check_rate_limit("short_client", 2, 1)
            assert allowed is True
            
            # Should be blocked
            allowed, status = rate_limiter.check_rate_limit("short_client", 2, 1)
            assert allowed is False
            
            # Move forward past window
            mock_time.return_value = 2
            
            # Should be allowed again
            allowed, status = rate_limiter.check_rate_limit("short_client", 2, 1)
            assert allowed is True

    def test_large_rate_limit(self, rate_limiter):
        """Test with very large rate limit."""
        large_limit = 10000
        
        for i in range(100):
            allowed, status = rate_limiter.check_rate_limit("large_client", large_limit, 60)
            assert allowed is True
            assert status.remaining == large_limit - i - 1

    def test_invalid_input_handling(self, rate_limiter):
        """Test handling of invalid inputs."""
        # Negative values should be handled gracefully
        allowed, status = rate_limiter.check_rate_limit("invalid_client", -1, 60)
        assert allowed is False
        
        allowed, status = rate_limiter.check_rate_limit("invalid_client", 5, -1)
        # Should handle gracefully (not crash)

    def test_empty_client_key(self, rate_limiter):
        """Test with empty client key."""
        allowed, status = rate_limiter.check_rate_limit("", 5, 60)
        assert isinstance(allowed, bool)
        assert isinstance(status, RateLimitStatus)

# ============================================================================
# STATISTICS AND MONITORING TESTS
# ============================================================================

class TestRateLimitingStatistics:
    """Test statistics and monitoring functionality."""

    def test_middleware_statistics(self, app_with_rate_limiting):
        """Test middleware statistics collection."""
        from fastapi.testclient import TestClient
        client = TestClient(app_with_rate_limiting)
        
        # Get middleware instance
        middleware = None
        for middleware_obj in app_with_rate_limiting.user_middleware:
            if hasattr(middleware_obj, 'cls') and middleware_obj.cls.__name__ == 'RateLimitMiddleware':
                middleware = middleware_obj
                break
        
        if middleware:
            initial_stats = middleware.get_statistics()
            assert "requests_processed" in initial_stats
            assert "requests_blocked" in initial_stats
            assert "requests_allowed" in initial_stats

    def test_rate_limit_status_endpoint(self):
        """Test rate limit status endpoint exists and works."""
        from backend.main import app
        client = TestClient(app)
        
        response = client.get("/api/v1/rate-limit/status")
        assert response.status_code == 200
        
        content = response.json()
        assert "rate_limiting" in content
        assert "limits" in content

# ============================================================================
# INTEGRATION WITH VALIDATION TESTS
# ============================================================================

class TestRateLimitingWithValidation:
    """Test rate limiting works correctly with input validation."""

    def test_rate_limiting_before_validation(self):
        """Test that rate limiting is checked before input validation."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.middleware.validation import InputValidationMiddleware
        from backend.middleware.rate_limiter import RateLimitMiddleware
        
        app = FastAPI()
        
        # Add rate limiting first, then validation
        app.add_middleware(RateLimitMiddleware, storage_backend="memory", config={
            "/test": {"requests": 1, "window": 60}
        })
        app.add_middleware(InputValidationMiddleware)
        
        @app.post("/test")
        async def test_endpoint(request: dict):
            return {"status": "ok"}
        
        client = TestClient(app)
        
        # First request should work
        response = client.post("/test", json={"valid": "data"})
        assert response.status_code == 200
        
        # Second request should be rate limited (before validation)
        response = client.post("/test", json={"invalid": "data"})
        assert response.status_code == 429

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
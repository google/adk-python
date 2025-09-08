"""
Performance Benchmark Tests
==========================

Tests performance aspects of the application:
- Response time benchmarks
- Memory usage monitoring
- Concurrent request handling
- Database query performance
- Large data processing
- Background task performance
"""

import pytest
import asyncio
import time
import threading
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add backend to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.main import app


class TestResponseTimePerformance:
    """Test response time performance for various endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_endpoint_response_time(self):
        """Test health endpoint responds quickly."""
        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Health check should be very fast (under 1 second)
        assert response_time < 1.0, f"Health check too slow: {response_time:.3f}s"
    
    def test_metrics_endpoint_response_time(self):
        """Test metrics endpoint response time."""
        start_time = time.time()
        response = self.client.get("/metrics")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Metrics should be fast (under 2 seconds)
        assert response_time < 2.0, f"Metrics too slow: {response_time:.3f}s"
    
    def test_chat_endpoint_response_time(self):
        """Test chat endpoint response time."""
        request_data = {
            "query": "Quick performance test",
            "session_id": "perf-test",
            "user_id": "perf-user"
        }
        
        start_time = time.time()
        response = self.client.post("/api/v1/chat/message", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Chat should respond within reasonable time (under 10 seconds)
        assert response_time < 10.0, f"Chat response too slow: {response_time:.3f}s"
    
    def test_status_endpoint_response_time(self):
        """Test status endpoint response time."""
        start_time = time.time()
        response = self.client.get("/status")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Status should be reasonably fast (under 3 seconds due to system metrics)
        assert response_time < 3.0, f"Status too slow: {response_time:.3f}s"


class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    def setup_method(self):
        """Set up test client and get initial memory."""
        self.client = TestClient(app)
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively with requests."""
        # Make multiple requests and monitor memory
        memory_measurements = [self.initial_memory]
        
        for i in range(50):  # 50 requests
            response = self.client.get("/health")
            assert response.status_code == 200
            
            if i % 10 == 0:  # Measure every 10 requests
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements.append(final_memory)
        
        # Memory shouldn't grow too much (allowing for some reasonable growth)
        memory_growth = final_memory - self.initial_memory
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.2f}MB"
        
        # Check for memory leaks (no continuous growth)
        if len(memory_measurements) >= 3:
            # Shouldn't have continuous upward trend
            last_three = memory_measurements[-3:]
            assert not (last_three[0] < last_three[1] < last_three[2]), "Possible memory leak detected"
    
    def test_chat_memory_efficiency(self):
        """Test chat endpoint memory efficiency."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Make several chat requests
        for i in range(20):
            request_data = {
                "query": f"Memory test query {i}",
                "session_id": f"memory-test-{i}",
                "user_id": "memory-test-user"
            }
            
            response = self.client.post("/api/v1/chat/message", json=request_data)
            assert response.status_code == 200
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Should not use excessive memory for chat
        assert memory_growth < 50, f"Chat memory usage too high: {memory_growth:.2f}MB"


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_concurrent_health_checks(self):
        """Test concurrent health check performance."""
        num_requests = 20
        
        def make_health_request():
            start_time = time.time()
            response = self.client.get("/health")
            end_time = time.time()
            return response.status_code == 200, end_time - start_time
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        success_count = sum(1 for success, _ in results if success)
        assert success_count == num_requests, f"Only {success_count}/{num_requests} requests succeeded"
        
        # Average response time should be reasonable
        response_times = [response_time for _, response_time in results]
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 2.0, f"Average response time too slow: {avg_response_time:.3f}s"
        
        # Total time should be less than sequential time (showing concurrency works)
        sequential_estimate = avg_response_time * num_requests
        assert total_time < sequential_estimate * 0.8, f"No concurrency benefit: {total_time:.3f}s vs {sequential_estimate:.3f}s"
    
    def test_concurrent_chat_requests(self):
        """Test concurrent chat request performance."""
        num_requests = 10  # Fewer for chat as it's more expensive
        
        def make_chat_request(request_id):
            request_data = {
                "query": f"Concurrent test {request_id}",
                "session_id": f"concurrent-{request_id}",
                "user_id": f"user-{request_id}"
            }
            
            start_time = time.time()
            response = self.client.post("/api/v1/chat/message", json=request_data)
            end_time = time.time()
            
            return response.status_code == 200, end_time - start_time
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_chat_request, i) for i in range(num_requests)]
            results = [future.result() for future in futures]
        
        # Most requests should succeed (some might be rate-limited)
        success_count = sum(1 for success, _ in results if success)
        assert success_count >= num_requests * 0.7, f"Too many failures: {success_count}/{num_requests}"
        
        # Successful responses should be reasonably fast
        successful_times = [response_time for success, response_time in results if success]
        if successful_times:
            avg_response_time = sum(successful_times) / len(successful_times)
            assert avg_response_time < 15.0, f"Average chat response time too slow: {avg_response_time:.3f}s"


class TestDatabasePerformance:
    """Test database operation performance."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_status_database_query_performance(self):
        """Test database queries in status endpoint are fast."""
        # Status endpoint includes database information
        start_time = time.time()
        response = self.client.get("/status")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should be reasonably fast even with database queries
        assert response_time < 5.0, f"Status with DB query too slow: {response_time:.3f}s"
        
        # Check that database info is included
        data = response.json()
        assert "database" in data
    
    def test_repeated_database_access_performance(self):
        """Test repeated database access doesn't degrade performance."""
        response_times = []
        
        # Make multiple requests that involve database access
        for i in range(10):
            start_time = time.time()
            response = self.client.get("/status")
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Response times should be consistent (no significant degradation)
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # Max time shouldn't be much worse than average (allowing for some variance)
        assert max_time < avg_time * 3, f"Inconsistent performance: avg={avg_time:.3f}s, max={max_time:.3f}s"


class TestLargeDataPerformance:
    """Test performance with large data inputs."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_large_query_performance(self):
        """Test performance with large query inputs."""
        # Test with increasingly large queries
        query_sizes = [100, 1000, 5000]  # Characters
        
        for size in query_sizes:
            large_query = "A" * size
            request_data = {
                "query": large_query,
                "session_id": f"large-query-{size}",
                "user_id": "large-query-user"
            }
            
            start_time = time.time()
            response = self.client.post("/api/v1/chat/message", json=request_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should handle large queries (might be slower but shouldn't fail)
            assert response.status_code in [200, 413, 422]  # 200=success, 413=too large, 422=validation error
            
            if response.status_code == 200:
                # If accepted, should respond within reasonable time
                assert response_time < 30.0, f"Large query ({size} chars) too slow: {response_time:.3f}s"
    
    def test_multiple_session_performance(self):
        """Test performance with multiple concurrent sessions."""
        num_sessions = 20
        
        def create_session_request(session_id):
            request_data = {
                "query": f"Performance test for session {session_id}",
                "session_id": f"perf-session-{session_id}",
                "user_id": f"perf-user-{session_id}"
            }
            
            start_time = time.time()
            response = self.client.post("/api/v1/chat/message", json=request_data)
            end_time = time.time()
            
            return response.status_code == 200, end_time - start_time
        
        # Create requests for multiple sessions
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(create_session_request, i) for i in range(num_sessions)]
            results = [future.result() for future in futures]
        
        # Most should succeed
        success_count = sum(1 for success, _ in results if success)
        assert success_count >= num_sessions * 0.8, f"Too many session failures: {success_count}/{num_sessions}"
        
        # Response times should be reasonable
        successful_times = [time for success, time in results if success]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            assert avg_time < 20.0, f"Multi-session performance too slow: {avg_time:.3f}s"


class TestBackgroundTaskPerformance:
    """Test background task performance."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
    
    def test_background_task_does_not_block_requests(self):
        """Test background tasks don't block request handling."""
        # Make requests while background tasks might be running
        start_time = time.time()
        
        responses = []
        for i in range(10):
            response = self.client.get("/health")
            responses.append(response.status_code)
            time.sleep(0.1)  # Small delay between requests
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        assert all(status == 200 for status in responses)
        
        # Total time should be close to expected (10 * 0.1s + processing time)
        # Should not be significantly delayed by background tasks
        assert total_time < 5.0, f"Requests blocked by background tasks: {total_time:.3f}s"
    
    def test_application_startup_performance(self):
        """Test application startup is not excessively slow."""
        # This test verifies the app can respond quickly after startup
        # (app is already started by test framework, so we test responsiveness)
        
        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should be responsive immediately (not waiting for background tasks to complete)
        assert response_time < 2.0, f"App not responsive after startup: {response_time:.3f}s"


class TestResourceUtilization:
    """Test overall resource utilization."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_cpu_utilization_under_load(self):
        """Test CPU utilization remains reasonable under load."""
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Generate some load
        for i in range(30):
            response = self.client.get("/health")
            assert response.status_code == 200
            
            if i % 10 == 0:
                time.sleep(0.1)  # Brief pause to allow CPU measurement
        
        final_cpu = psutil.cpu_percent(interval=1)
        
        # CPU usage shouldn't be excessive (allowing for test environment variations)
        # This is more of a smoke test than strict validation
        assert final_cpu < 95, f"High CPU usage: {final_cpu}%"
    
    def test_file_descriptor_usage(self):
        """Test file descriptor usage doesn't leak."""
        initial_fds = len(psutil.Process().open_files())
        
        # Make requests that might open files/connections
        for i in range(20):
            response = self.client.get("/status")  # Status checks database
            assert response.status_code == 200
        
        final_fds = len(psutil.Process().open_files())
        fd_growth = final_fds - initial_fds
        
        # Should not leak file descriptors excessively
        assert fd_growth < 10, f"Possible FD leak: {fd_growth} new descriptors"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
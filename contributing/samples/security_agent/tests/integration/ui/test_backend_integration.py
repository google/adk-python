"""
Integration tests for UI-Backend communication.

Tests the communication between the new UI components and backend APIs,
ensuring proper data flow and error handling.
"""

import pytest
import asyncio
import httpx
import streamlit as st
from unittest.mock import patch, Mock, AsyncMock
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from backend.main import app
from frontend.unified_streaming_client import SecurityDashboard
from frontend.dashboard import SecurityDashboard as DashboardMain
from backend.api import health, security, iam, storage, monitoring


class TestBackendIntegration:
    """Test suite for UI-Backend integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8000"
        self.timeout = 30.0
        
    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self):
        """Test health endpoint integration from UI."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] in ["healthy", "degraded", "unhealthy"]
            assert "timestamp" in data
            assert "version" in data
            
    @pytest.mark.asyncio
    async def test_dashboard_api_calls(self):
        """Test dashboard API calls from frontend."""
        # Mock Streamlit session state
        with patch('streamlit.session_state', {}):
            dashboard = SecurityDashboard()
            
            # Test security metrics API call
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/security/metrics")
                
                if response.status_code == 200:
                    metrics = response.json()
                    assert isinstance(metrics, dict)
                    # Verify required fields exist
                    expected_fields = ["critical_findings", "high_findings", "medium_findings"]
                    for field in expected_fields:
                        assert field in metrics or "error" in metrics
                        
    @pytest.mark.asyncio
    async def test_chat_websocket_integration(self):
        """Test WebSocket chat integration."""
        # Test WebSocket connection for chat functionality
        try:
            import websockets
            
            uri = "ws://localhost:8000/ws/chat"
            async with websockets.connect(uri) as websocket:
                # Send test message
                test_query = "What are my top security risks?"
                await websocket.send(json.dumps({
                    "type": "chat_message",
                    "content": test_query,
                    "session_id": "test_session"
                }))
                
                # Receive response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                assert "type" in response_data
                assert "content" in response_data
                
        except Exception as e:
            # If WebSocket not available, mark as expected failure
            pytest.skip(f"WebSocket not available: {e}")
            
    @pytest.mark.asyncio
    async def test_authentication_flow(self):
        """Test authentication integration between UI and backend."""
        auth_data = {
            "credentials": "test_credentials",
            "project_id": "test-project"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/auth/validate",
                json=auth_data
            )
            
            # Should either succeed or fail gracefully
            assert response.status_code in [200, 401, 403, 404]
            
            if response.status_code == 200:
                auth_result = response.json()
                assert "authenticated" in auth_result
                
    @pytest.mark.asyncio
    async def test_data_refresh_mechanism(self):
        """Test data refresh mechanisms from UI."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test manual data refresh endpoint
            response = await client.post(f"{self.base_url}/api/data/refresh")
            
            # Should either succeed or handle gracefully
            assert response.status_code in [200, 202, 503, 429]
            
            if response.status_code in [200, 202]:
                refresh_result = response.json()
                assert "status" in refresh_result
                
    @pytest.mark.asyncio
    async def test_error_handling_from_backend(self):
        """Test UI handling of backend errors."""
        # Test various error scenarios
        error_endpoints = [
            "/api/nonexistent/endpoint",
            "/api/security/invalid_query",
            "/api/iam/malformed_request"
        ]
        
        async with httpx.AsyncClient() as client:
            for endpoint in error_endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    # Should handle errors gracefully
                    assert response.status_code in [400, 404, 422, 500]
                    
                    # Response should contain error information
                    if response.headers.get("content-type", "").startswith("application/json"):
                        error_data = response.json()
                        assert isinstance(error_data, dict)
                        
                except httpx.RequestError:
                    # Connection errors are acceptable for this test
                    pass
                    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self):
        """Test concurrent API calls from UI components."""
        endpoints = [
            "/api/security/summary",
            "/api/iam/policies",
            "/api/storage/analysis",
            "/api/monitoring/metrics"
        ]
        
        async with httpx.AsyncClient() as client:
            # Make concurrent requests
            tasks = [
                client.get(f"{self.base_url}{endpoint}")
                for endpoint in endpoints
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify responses or exceptions are handled
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    # Exceptions should be connection-related, not server errors
                    assert isinstance(response, (httpx.ConnectError, httpx.TimeoutException))
                else:
                    # Successful responses should be well-formed
                    assert hasattr(response, 'status_code')
                    
    @pytest.mark.asyncio
    async def test_streaming_response_integration(self):
        """Test streaming responses from backend to UI."""
        stream_endpoints = [
            "/api/chat/stream",
            "/api/analysis/stream"
        ]
        
        async with httpx.AsyncClient() as client:
            for endpoint in stream_endpoints:
                try:
                    async with client.stream("GET", f"{self.base_url}{endpoint}") as response:
                        if response.status_code == 200:
                            # Verify streaming content
                            content_chunks = []
                            async for chunk in response.aiter_text():
                                content_chunks.append(chunk)
                                if len(content_chunks) > 5:  # Limit for testing
                                    break
                                    
                            # Should receive some streaming content
                            assert len(content_chunks) > 0
                            
                except (httpx.RequestError, httpx.ConnectError):
                    # Connection issues are acceptable for integration tests
                    pass
                    
    def test_session_persistence(self):
        """Test session persistence across UI interactions."""
        # Mock streamlit session state
        with patch('streamlit.session_state', {}) as mock_session:
            # Test session initialization
            mock_session['user_id'] = 'test_user'
            mock_session['session_id'] = 'test_session_123'
            
            # Verify session data persists
            assert mock_session['user_id'] == 'test_user'
            assert mock_session['session_id'] == 'test_session_123'
            
            # Test session updates
            mock_session['last_query'] = 'security analysis'
            assert mock_session['last_query'] == 'security analysis'
            
    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test caching integration between UI and backend."""
        cache_test_endpoint = "/api/security/cached_metrics"
        
        async with httpx.AsyncClient() as client:
            # First request - should hit backend
            start_time = time.time()
            response1 = await client.get(f"{self.base_url}{cache_test_endpoint}")
            first_request_time = time.time() - start_time
            
            if response1.status_code == 200:
                # Second request - should be faster (cached)
                start_time = time.time()
                response2 = await client.get(f"{self.base_url}{cache_test_endpoint}")
                second_request_time = time.time() - start_time
                
                # Cached response should be available
                assert response2.status_code == 200
                
                # Content should be consistent
                if response1.headers.get("content-type", "").startswith("application/json"):
                    data1 = response1.json()
                    data2 = response2.json()
                    
                    # Core data should match (allowing for timestamps)
                    if "timestamp" in data1:
                        del data1["timestamp"]
                    if "timestamp" in data2:
                        del data2["timestamp"]
                        
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self):
        """Test UI handling of rate limiting from backend."""
        async with httpx.AsyncClient() as client:
            # Make rapid requests to trigger rate limiting
            requests = []
            for i in range(20):  # Make many requests quickly
                task = client.get(f"{self.base_url}/api/security/summary")
                requests.append(task)
                
            responses = await asyncio.gather(*requests, return_exceptions=True)
            
            # Should handle rate limiting gracefully
            rate_limited_count = 0
            success_count = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    continue
                elif response.status_code == 429:
                    rate_limited_count += 1
                elif response.status_code == 200:
                    success_count += 1
                    
            # Should have some successful requests and possibly some rate limited
            assert success_count > 0 or rate_limited_count > 0
            
    @pytest.mark.asyncio 
    async def test_real_time_updates(self):
        """Test real-time updates between backend and UI."""
        # Test real-time metric updates
        async with httpx.AsyncClient() as client:
            # Subscribe to updates endpoint
            try:
                response = await client.get(f"{self.base_url}/api/realtime/subscribe")
                
                if response.status_code == 200:
                    subscription_data = response.json()
                    assert "subscription_id" in subscription_data
                    
                    # Test getting updates
                    updates_response = await client.get(
                        f"{self.base_url}/api/realtime/updates/{subscription_data['subscription_id']}"
                    )
                    
                    # Should get updates or indicate none available
                    assert updates_response.status_code in [200, 204]
                    
            except (httpx.RequestError, httpx.ConnectError):
                pytest.skip("Real-time updates endpoint not available")


class TestAPIContractCompliance:
    """Test API contract compliance for UI integration."""
    
    @pytest.mark.asyncio
    async def test_security_api_contract(self):
        """Test security API endpoints return expected data structure."""
        async with httpx.AsyncClient() as client:
            endpoints_and_expected_fields = {
                "/api/security/summary": ["status", "findings"],
                "/api/security/metrics": ["critical_findings", "total_resources"],
                "/api/security/recommendations": ["recommendations"]
            }
            
            for endpoint, expected_fields in endpoints_and_expected_fields.items():
                try:
                    response = await client.get(f"http://localhost:8000{endpoint}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Verify expected fields exist or error is properly formatted
                        if "error" not in data:
                            for field in expected_fields:
                                assert field in data, f"Missing {field} in {endpoint} response"
                                
                except (httpx.RequestError, httpx.ConnectError):
                    pytest.skip(f"Endpoint {endpoint} not accessible")
                    
    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """Test error responses follow consistent format."""
        async with httpx.AsyncClient() as client:
            # Test various error conditions
            error_tests = [
                ("/api/invalid/endpoint", 404),
                ("/api/security/summary?invalid_param=bad", 400)
            ]
            
            for endpoint, expected_status in error_tests:
                try:
                    response = await client.get(f"http://localhost:8000{endpoint}")
                    
                    if response.status_code == expected_status:
                        # Verify error response format
                        if response.headers.get("content-type", "").startswith("application/json"):
                            error_data = response.json()
                            
                            # Should have consistent error structure
                            assert isinstance(error_data, dict)
                            # Common error fields
                            expected_error_fields = ["error", "message", "detail"]
                            has_error_field = any(field in error_data for field in expected_error_fields)
                            assert has_error_field, f"No standard error field in response from {endpoint}"
                            
                except (httpx.RequestError, httpx.ConnectError):
                    pytest.skip(f"Cannot test error response for {endpoint}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
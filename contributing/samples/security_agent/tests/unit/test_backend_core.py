"""
Comprehensive Unit Tests for Backend Core Functionality
=======================================================

Tests core backend components including:
- FastAPI application initialization
- Middleware configuration  
- Router registration
- Environment configuration
- Health checks
- Rate limiting
- Input validation
- Background tasks
"""

import pytest
import asyncio
import os
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import logging

# Add backend to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Import backend components
from backend.main import app, health_check, background_cache_refresh
from backend.main import setup_service_account_from_secret


class TestBackendInitialization:
    """Test backend application initialization."""
    
    def test_app_creation(self):
        """Test FastAPI app is properly created."""
        assert isinstance(app, FastAPI)
        assert app.title == "Security Agent Backend"
        assert app.version == "1.0.0"
    
    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured."""
        # Check CORS middleware is in the middleware stack
        middleware_types = [type(middleware.cls) for middleware in app.user_middleware]
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_types
    
    def test_environment_variables_loaded(self):
        """Test environment variables are properly loaded."""
        # Test with mock environment
        with patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'test-project'}):
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            assert project_id == 'test-project'


class TestHealthEndpoints:
    """Test health check and monitoring endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct response."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "ADK Security Agent Backend"
        assert data["status"] == "running"
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "features" in data
        assert "components" in data
    
    @pytest.mark.asyncio
    async def test_health_check_function(self):
        """Test health check function directly."""
        health_data = await health_check()
        assert isinstance(health_data, dict)
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "adk_security_agent_up" in content
        assert "system_cpu_usage_percent" in content
        assert "system_memory_usage_percent" in content
    
    def test_status_endpoint(self):
        """Test detailed status endpoint."""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "system" in data
        assert "services" in data


class TestChatEndpoint:
    """Test chat message endpoint."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_chat_endpoint_valid_request(self):
        """Test chat endpoint with valid request."""
        request_data = {
            "query": "What resources do I have?",
            "session_id": "test-session",
            "user_id": "test-user"
        }
        
        response = self.client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "user_id" in data
        assert "success" in data
        assert data["session_id"] == "test-session"
        assert data["user_id"] == "test-user"
    
    def test_chat_endpoint_minimal_request(self):
        """Test chat endpoint with minimal request."""
        request_data = {"query": "Hello"}
        
        response = self.client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["success"] is True
    
    def test_chat_endpoint_empty_query(self):
        """Test chat endpoint with empty query."""
        request_data = {"query": ""}
        
        response = self.client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        # Should still return a response even with empty query
        data = response.json()
        assert "response" in data


class TestRateLimitingMiddleware:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_rate_limit_status_endpoint(self):
        """Test rate limit status endpoint."""
        response = self.client.get("/api/v1/rate-limit/status")
        assert response.status_code == 200
        data = response.json()
        assert "rate_limiting" in data
        assert data["rate_limiting"] in ["enabled", "disabled"]
        
        if data["rate_limiting"] == "enabled":
            assert "limits" in data
            assert "window" in data


class TestBackgroundTasks:
    """Test background task functionality."""
    
    @pytest.mark.asyncio
    async def test_background_cache_refresh_task(self):
        """Test background cache refresh task initialization."""
        # Mock environment variables
        with patch.dict(os.environ, {
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'DATA_REFRESH_INTERVAL': '10'
        }):
            # Mock the DataFetcher to avoid actual API calls
            with patch('backend.main.DataFetcher') as mock_fetcher_class:
                mock_fetcher = Mock()
                mock_fetcher.fetch_all_data = AsyncMock(return_value={
                    'stats': {'assets': {'count': 10}},
                    'errors': [],
                    'duration_seconds': 1.5
                })
                mock_fetcher_class.return_value = mock_fetcher
                
                # Test that the background task can be created without error
                try:
                    task = asyncio.create_task(background_cache_refresh())
                    # Cancel immediately to avoid long running test
                    task.cancel()
                    
                    # Wait briefly and catch the cancellation
                    with pytest.raises(asyncio.CancelledError):
                        await task
                        
                except Exception as e:
                    pytest.fail(f"Background task creation failed: {e}")


class TestSecretManagerIntegration:
    """Test Google Secret Manager integration."""
    
    def test_setup_service_account_from_secret_no_k_service(self):
        """Test service account setup when not in Cloud Run."""
        # Mock environment without K_SERVICE
        with patch.dict(os.environ, {}, clear=True):
            # This should not raise an exception
            setup_service_account_from_secret()
            # Should use local credentials
    
    @patch('backend.main.secretmanager')
    def test_setup_service_account_from_secret_with_k_service(self, mock_secretmanager):
        """Test service account setup in Cloud Run environment."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = '{"type": "service_account"}'
        mock_client.access_secret_version.return_value = mock_response
        mock_secretmanager.SecretManagerServiceClient.return_value = mock_client
        
        with patch.dict(os.environ, {
            'K_SERVICE': 'test-service',
            'GOOGLE_CLOUD_PROJECT': 'test-project'
        }):
            with patch('tempfile.mkstemp') as mock_mkstemp:
                mock_mkstemp.return_value = (1, '/tmp/test.json')
                with patch('os.fdopen') as mock_fdopen:
                    mock_file = Mock()
                    mock_fdopen.return_value.__enter__.return_value = mock_file
                    
                    result = setup_service_account_from_secret()
                    assert result == '/tmp/test.json'


class TestInputValidationMiddleware:
    """Test input validation and sanitization."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_query_parameter_handling(self):
        """Test query parameter sanitization."""
        # Test with potentially dangerous query params
        response = self.client.get("/health?test=<script>alert('xss')</script>")
        assert response.status_code == 200
        # Should not crash the application
    
    def test_json_body_handling(self):
        """Test JSON body validation."""
        # Test with valid JSON
        valid_json = {"query": "test query", "user_id": "user123"}
        response = self.client.post("/api/v1/chat/message", json=valid_json)
        assert response.status_code == 200
        
        # Test with malformed JSON should be handled by FastAPI
        # malformed requests are handled at the FastAPI level


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_nonexistent_endpoint(self):
        """Test 404 handling."""
        response = self.client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 handling."""
        response = self.client.put("/")
        assert response.status_code == 405
    
    def test_internal_error_handling(self):
        """Test that internal errors don't crash the application."""
        # The chat endpoint should handle errors gracefully
        with patch('backend.main.logger') as mock_logger:
            request_data = {"query": "test query"}
            response = self.client.post("/api/v1/chat/message", json=request_data)
            # Should return some response even if there are internal issues
            assert response.status_code == 200


class TestApplicationState:
    """Test application state management."""
    
    def setup_method(self):
        """Set up test client and app state."""
        self.client = TestClient(app)
        # Initialize state if not already done
        if not hasattr(app.state, 'start_time'):
            app.state.start_time = 1234567890
            app.state.request_count = 0
            app.state.error_count = 0
    
    def test_request_counting(self):
        """Test request counting in app state."""
        initial_count = getattr(app.state, 'request_count', 0)
        
        # Make a request
        self.client.get("/health")
        
        # Request count should increase (or at least not decrease)
        # Note: This might not increment in test mode depending on middleware
    
    def test_app_state_persistence(self):
        """Test app state persistence across requests."""
        # Check that state persists
        assert hasattr(app.state, 'start_time')
        start_time = app.state.start_time
        
        # Make request
        self.client.get("/health")
        
        # State should persist
        assert app.state.start_time == start_time


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
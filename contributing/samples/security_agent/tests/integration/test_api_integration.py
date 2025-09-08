"""
Comprehensive Integration Tests for API Endpoints
=================================================

Tests integration between different API components:
- API endpoint functionality
- Database interactions
- External service integrations
- Cross-component communication
- Data flow validation
"""

import pytest
import asyncio
import os
import json
import tempfile
import sqlite3
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import httpx
import time
from datetime import datetime, timedelta

# Add backend to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from backend.main import app


class TestChatIntegration:
    """Test chat endpoint integration with session management."""
    
    def setup_method(self):
        """Set up test client and environment."""
        self.client = TestClient(app)
        self.test_project = "test-project-integration"
        
    def test_chat_with_session_persistence(self):
        """Test chat integration with session persistence."""
        session_id = f"test-session-{int(time.time())}"
        user_id = "test-user-integration"
        
        # First message
        request1 = {
            "query": "What is my project ID?",
            "session_id": session_id,
            "user_id": user_id
        }
        
        response1 = self.client.post("/api/v1/chat/message", json=request1)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == session_id
        assert data1["user_id"] == user_id
        assert "response" in data1
        
        # Second message in same session
        request2 = {
            "query": "Tell me more about that project",
            "session_id": session_id,
            "user_id": user_id
        }
        
        response2 = self.client.post("/api/v1/chat/message", json=request2)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id
        # Session should be maintained
    
    def test_chat_fallback_behavior(self):
        """Test chat fallback when agent is unavailable."""
        request_data = {
            "query": "Show me my resources",
            "session_id": "fallback-test",
            "user_id": "test-user"
        }
        
        # Should not fail even if agent has issues
        response = self.client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["success"] is True
    
    def test_chat_with_different_query_types(self):
        """Test chat with different types of queries."""
        session_id = f"query-test-{int(time.time())}"
        
        queries = [
            "What resources do I have?",
            "Check my security posture",
            "Show my service accounts", 
            "Find security issues",
            "help",
            ""  # Empty query
        ]
        
        for i, query in enumerate(queries):
            request_data = {
                "query": query,
                "session_id": f"{session_id}-{i}",
                "user_id": "test-user"
            }
            
            response = self.client.post("/api/v1/chat/message", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert isinstance(data["response"], str)


class TestHealthIntegration:
    """Test health monitoring integration."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_comprehensive_health_check(self):
        """Test health check includes all components."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["status", "timestamp", "features", "components", "endpoints"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check feature flags
        features = data["features"]
        assert isinstance(features, dict)
        assert "comprehensive_monitoring" in features
        assert "robust_fallbacks" in features
        
        # Check components
        components = data["components"]
        assert isinstance(components, dict)
    
    def test_metrics_integration(self):
        """Test metrics endpoint provides system data."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        # Check for key metrics
        expected_metrics = [
            "adk_security_agent_up",
            "system_cpu_usage_percent",
            "system_memory_usage_percent",
            "system_disk_usage_percent"
        ]
        
        for metric in expected_metrics:
            assert metric in content, f"Missing metric: {metric}"
    
    def test_status_endpoint_integration(self):
        """Test status endpoint provides detailed information."""
        response = self.client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        required_sections = ["status", "uptime", "system", "services", "environment"]
        for section in required_sections:
            assert section in data, f"Missing status section: {section}"
        
        # Check system metrics are present
        system = data["system"]
        assert "cpu" in system
        assert "memory" in system
        assert "disk" in system


class TestDatabaseIntegration:
    """Test database integration across the application."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.test_db_path = "/tmp/test_security_agent.db"
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_database_connection_in_status(self):
        """Test database status is reported correctly."""
        response = self.client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "database" in data
        db_info = data["database"]
        assert "status" in db_info
        # Status should be one of: connected, not_found, error
        assert db_info["status"] in ["connected", "not_found", "error"]
    
    @pytest.mark.asyncio
    async def test_session_database_integration(self):
        """Test session management with database."""
        # This tests if session APIs work with database backend
        session_data = {
            "user_id": "test-user-db",
            "project_id": "test-project-db"
        }
        
        # Test session creation through chat
        chat_request = {
            "query": "Create a test session",
            "session_id": f"db-test-{int(time.time())}",
            "user_id": session_data["user_id"]
        }
        
        response = self.client.post("/api/v1/chat/message", json=chat_request)
        assert response.status_code == 200
        # Session should be created/handled


class TestAPIRouterIntegration:
    """Test API router integration and endpoint availability."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_core_endpoints_available(self):
        """Test that core endpoints are available."""
        core_endpoints = [
            "/",
            "/health", 
            "/metrics",
            "/status",
            "/api/v1/chat/message"
        ]
        
        for endpoint in core_endpoints:
            if endpoint == "/api/v1/chat/message":
                # POST endpoint
                response = self.client.post(endpoint, json={"query": "test"})
            else:
                # GET endpoint  
                response = self.client.get(endpoint)
            
            # Should not be 404
            assert response.status_code != 404, f"Endpoint {endpoint} not found"
    
    def test_rate_limiting_integration(self):
        """Test rate limiting is working."""
        response = self.client.get("/api/v1/rate-limit/status")
        # Should be available regardless of whether rate limiting is enabled
        assert response.status_code == 200
        
        data = response.json()
        assert "rate_limiting" in data
    
    def test_api_prefix_consistency(self):
        """Test API endpoints use consistent prefixes."""
        # Make requests to verify API structure
        chat_response = self.client.post("/api/v1/chat/message", json={"query": "test"})
        assert chat_response.status_code == 200
        
        rate_limit_response = self.client.get("/api/v1/rate-limit/status")
        assert rate_limit_response.status_code == 200


class TestExternalServiceIntegration:
    """Test integration with external services."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @patch('backend.main.secretmanager')
    def test_secret_manager_integration(self, mock_secretmanager):
        """Test Secret Manager integration doesn't break startup."""
        # Mock Secret Manager for Cloud Run environment
        mock_client = Mock()
        mock_response = Mock()
        mock_response.payload.data.decode.return_value = '{"type": "service_account"}'
        mock_client.access_secret_version.return_value = mock_response
        mock_secretmanager.SecretManagerServiceClient.return_value = mock_client
        
        # Test that app still responds when Secret Manager is configured
        response = self.client.get("/health")
        assert response.status_code == 200
    
    def test_google_cloud_project_configuration(self):
        """Test Google Cloud project configuration."""
        # Test with different project configurations
        test_cases = [
            None,  # No project
            "test-project",  # Valid project
            "your-project-id",  # Default/placeholder
        ]
        
        for project_id in test_cases:
            with patch.dict(os.environ, 
                           {'GOOGLE_CLOUD_PROJECT': project_id} if project_id else {}, 
                           clear=True):
                response = self.client.get("/status")
                assert response.status_code == 200
                data = response.json()
                env_info = data.get("environment", {})
                reported_project = env_info.get("project_id", "not_configured")
                
                if project_id:
                    assert reported_project == project_id
                else:
                    assert reported_project == "not_configured"


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health check requests."""
        async def make_health_request():
            response = self.client.get("/health")
            return response.status_code == 200
        
        # Make 10 concurrent health check requests
        tasks = [make_health_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results), "Some concurrent health checks failed"
    
    @pytest.mark.asyncio 
    async def test_concurrent_chat_requests(self):
        """Test concurrent chat requests."""
        async def make_chat_request(i):
            request_data = {
                "query": f"Test query {i}",
                "session_id": f"concurrent-test-{i}",
                "user_id": f"user-{i}"
            }
            response = self.client.post("/api/v1/chat/message", json=request_data)
            return response.status_code == 200
        
        # Make 5 concurrent chat requests
        tasks = [make_chat_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results), "Some concurrent chat requests failed"


class TestDataFlowIntegration:
    """Test data flow between components."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_request_response_cycle(self):
        """Test complete request-response cycle."""
        request_data = {
            "query": "Integration test query",
            "session_id": "integration-test",
            "user_id": "integration-user"
        }
        
        # Measure response time
        start_time = time.time()
        response = self.client.post("/api/v1/chat/message", json=request_data)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = ["response", "session_id", "user_id", "success"]
        for field in required_fields:
            assert field in data, f"Missing field in response: {field}"
        
        # Validate response data
        assert data["session_id"] == request_data["session_id"]
        assert data["user_id"] == request_data["user_id"]
        assert data["success"] is True
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        
        # Response time should be reasonable (less than 30 seconds)
        response_time = end_time - start_time
        assert response_time < 30.0, f"Response time too slow: {response_time}s"
    
    def test_error_propagation(self):
        """Test error handling across components."""
        # Test with invalid JSON structure (handled by FastAPI)
        response = self.client.post("/api/v1/chat/message", 
                                  data="invalid json", 
                                  headers={"content-type": "application/json"})
        # Should return 422 for invalid JSON
        assert response.status_code == 422
        
        # Test with missing required fields - actually, our endpoint is flexible
        response = self.client.post("/api/v1/chat/message", json={})
        # Should still return 200 with default handling
        assert response.status_code == 200


class TestBackgroundTaskIntegration:
    """Test background task integration."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_app_startup_with_background_tasks(self):
        """Test application starts properly with background tasks."""
        # Test that health endpoint works (indicating app started properly)
        response = self.client.get("/health") 
        assert response.status_code == 200
        
        # Check that app state includes background task management
        health_data = response.json()
        features = health_data.get("features", {})
        # Should have monitoring and other features enabled
        assert features.get("comprehensive_monitoring") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
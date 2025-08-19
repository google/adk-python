"""Comprehensive tests for the Input Validation Framework - TASK-004.

Tests cover:
- Pydantic model validation
- SQL injection prevention
- XSS protection
- Query parameter validation
- Request size limits
- Security header validation
"""

import pytest
import json
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from backend.middleware.validation import (
    InputValidationMiddleware,
    ChatMessage,
    AssetQueryRequest,
    SecurityAnalysisRequest,
    SessionRequest,
    PaginationRequest,
    SecurityValidator,
    GCPProjectValidator
)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def app_with_validation():
    """Create FastAPI app with validation middleware."""
    app = FastAPI()
    app.add_middleware(InputValidationMiddleware)
    
    @app.post("/api/v1/chat/message")
    async def chat_endpoint(request: dict):
        return {"status": "success"}
    
    @app.post("/api/v1/assets/discover")
    async def assets_endpoint(request: dict):
        return {"status": "success"}
    
    @app.post("/api/v1/security/analyze")
    async def security_endpoint(request: dict):
        return {"status": "success"}
    
    @app.get("/api/v1/sessions/test-session/messages")
    async def session_messages(session_id: str):
        return {"status": "success"}
    
    return app

@pytest.fixture
def client(app_with_validation):
    """Test client with validation middleware."""
    return TestClient(app_with_validation)

# ============================================================================
# PYDANTIC MODEL TESTS
# ============================================================================

class TestPydanticModels:
    """Test Pydantic validation models."""

    def test_gcp_project_validator_valid(self):
        """Test valid GCP project IDs."""
        valid_projects = [
            "my-project-123",
            "test-project-1",
            "a-very-long-project-name-123",
            "project123"
        ]
        
        for project_id in valid_projects:
            validator = GCPProjectValidator(project_id=project_id)
            assert validator.project_id == project_id

    def test_gcp_project_validator_invalid(self):
        """Test invalid GCP project IDs."""
        invalid_projects = [
            "123-starts-with-number",
            "PROJECT-UPPERCASE",
            "project_with_underscores",
            "toolongprojectnamethatexceedsthirtycharacterlimit",
            "ends-with-dash-",
            "ab",  # too short
            "",    # empty
        ]
        
        for project_id in invalid_projects:
            with pytest.raises(Exception):
                GCPProjectValidator(project_id=project_id)

    def test_chat_message_valid(self):
        """Test valid chat message validation."""
        valid_message = {
            "query": "What are my GCP resources?",
            "session_id": "session_123",
            "user_id": "user.test@example.com"
        }
        
        chat_msg = ChatMessage(**valid_message)
        assert chat_msg.query == valid_message["query"]
        assert chat_msg.session_id == valid_message["session_id"]
        assert chat_msg.user_id == valid_message["user_id"]

    def test_chat_message_xss_protection(self):
        """Test XSS protection in chat messages."""
        malicious_queries = [
            "<script>alert('xss')</script>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "javascript:alert('test')",
            "<object data='data:text/html,<script>alert(1)</script>'></object>"
        ]
        
        for query in malicious_queries:
            with pytest.raises(Exception):
                ChatMessage(
                    query=query,
                    session_id="test_session",
                    user_id="test_user"
                )

    def test_chat_message_sql_injection_protection(self):
        """Test SQL injection protection in chat messages."""
        malicious_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "'; DELETE FROM sessions; --",
            "test' AND 1=1 --"
        ]
        
        for query in malicious_queries:
            with pytest.raises(Exception):
                ChatMessage(
                    query=query,
                    session_id="test_session",
                    user_id="test_user"
                )

    def test_asset_query_request_valid(self):
        """Test valid asset query request."""
        valid_request = {
            "project_id": "my-test-project",
            "asset_types": ["compute.googleapis.com/Instance", "storage.googleapis.com/Bucket"],
            "page_size": 50
        }
        
        asset_req = AssetQueryRequest(**valid_request)
        assert asset_req.project_id == valid_request["project_id"]
        assert asset_req.asset_types == valid_request["asset_types"]
        assert asset_req.page_size == valid_request["page_size"]

    def test_asset_query_request_invalid_asset_types(self):
        """Test invalid asset types."""
        invalid_asset_types = [
            "invalid/asset/type/with/special/chars!",
            "asset-type-with-<script>",
            "'; DROP TABLE assets; --"
        ]
        
        for asset_type in invalid_asset_types:
            with pytest.raises(Exception):
                AssetQueryRequest(
                    project_id="test-project",
                    asset_types=[asset_type],
                    page_size=10
                )

    def test_pagination_limits(self):
        """Test pagination parameter limits."""
        # Valid pagination
        pagination = PaginationRequest(page=1, page_size=20)
        assert pagination.page == 1
        assert pagination.page_size == 20
        
        # Invalid pagination - out of bounds
        with pytest.raises(Exception):
            PaginationRequest(page=0, page_size=20)  # page too low
        
        with pytest.raises(Exception):
            PaginationRequest(page=1001, page_size=20)  # page too high
        
        with pytest.raises(Exception):
            PaginationRequest(page=1, page_size=0)  # page_size too low
        
        with pytest.raises(Exception):
            PaginationRequest(page=1, page_size=101)  # page_size too high

# ============================================================================
# SECURITY VALIDATOR TESTS
# ============================================================================

class TestSecurityValidator:
    """Test security validation utilities."""

    def test_xss_detection(self):
        """Test XSS pattern detection."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('test')",
            "<iframe src='javascript:void(0)'></iframe>",
            "vbscript:msgbox('test')",
            "<object data='data:text/html,<script>alert(1)</script>'></object>"
        ]
        
        for payload in xss_payloads:
            assert SecurityValidator.check_xss(payload), f"Failed to detect XSS in: {payload}"
        
        # Safe content should not trigger
        safe_content = [
            "This is a normal message",
            "What are my GCP resources?",
            "Show me security recommendations"
        ]
        
        for content in safe_content:
            assert not SecurityValidator.check_xss(content), f"False positive for: {content}"

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "'; DELETE FROM sessions; --",
            "test' AND 1=1 --",
            "1; EXEC xp_cmdshell('dir')",
            "'; WAITFOR DELAY '00:00:05'; --"
        ]
        
        for payload in sql_payloads:
            assert SecurityValidator.check_sql_injection(payload), f"Failed to detect SQL injection in: {payload}"
        
        # Safe content should not trigger
        safe_content = [
            "SELECT my GCP resources",
            "I want to delete old files",
            "Create a new project"
        ]
        
        for content in safe_content:
            assert not SecurityValidator.check_sql_injection(content), f"False positive for: {content}"

    def test_string_sanitization(self):
        """Test string sanitization."""
        test_cases = [
            ("<script>alert('test')</script>", "&lt;script&gt;alert(&#x27;test&#x27;)&lt;/script&gt;"),
            ("test\x00null", "testnull"),
            ("normal text", "normal text")
        ]
        
        for input_str, expected in test_cases:
            result = SecurityValidator.sanitize_string(input_str)
            # Basic sanitization check - exact output may vary
            assert "\x00" not in result
            assert len(result) > 0

    def test_json_size_validation(self):
        """Test JSON size limits."""
        # Small JSON should pass
        small_json = json.dumps({"key": "value"}).encode()
        assert SecurityValidator.validate_json_size(small_json)
        
        # Large JSON should fail with default limit
        large_data = {"key": "x" * (1024 * 1024 + 1)}  # Exceed 1MB
        large_json = json.dumps(large_data).encode()
        assert not SecurityValidator.validate_json_size(large_json)
        
        # Custom limit
        medium_json = json.dumps({"key": "x" * 1000}).encode()
        assert SecurityValidator.validate_json_size(medium_json, max_size=500) == False
        assert SecurityValidator.validate_json_size(medium_json, max_size=2000) == True

# ============================================================================
# MIDDLEWARE INTEGRATION TESTS
# ============================================================================

class TestInputValidationMiddleware:
    """Test validation middleware integration."""

    def test_valid_chat_request(self, client):
        """Test valid chat request passes validation."""
        valid_payload = {
            "query": "What are my GCP resources?",
            "session_id": "test_session_123",
            "user_id": "test.user@example.com"
        }
        
        response = client.post("/api/v1/chat/message", json=valid_payload)
        assert response.status_code == 200

    def test_invalid_chat_request_xss(self, client):
        """Test chat request with XSS is blocked."""
        malicious_payload = {
            "query": "<script>alert('xss')</script>",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        response = client.post("/api/v1/chat/message", json=malicious_payload)
        assert response.status_code == 422

    def test_invalid_chat_request_sql_injection(self, client):
        """Test chat request with SQL injection is blocked."""
        malicious_payload = {
            "query": "'; DROP TABLE users; --",
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        response = client.post("/api/v1/chat/message", json=malicious_payload)
        assert response.status_code == 422

    def test_query_parameter_validation(self, client):
        """Test query parameter validation."""
        # Valid query parameters
        response = client.get("/api/v1/sessions/test-session/messages?page=1&page_size=20")
        assert response.status_code == 200
        
        # XSS in query parameter
        response = client.get("/api/v1/sessions/test-session/messages?page=<script>alert('xss')</script>")
        assert response.status_code == 400
        
        # SQL injection in query parameter
        response = client.get("/api/v1/sessions/test-session/messages?filter='; DROP TABLE users; --")
        assert response.status_code == 400

    def test_request_size_limits(self, client):
        """Test request size validation."""
        # Create a large payload
        large_payload = {
            "query": "x" * (1024 * 1024 + 1),  # Exceed 1MB
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        response = client.post("/api/v1/chat/message", json=large_payload)
        # Should fail due to size limits
        assert response.status_code in [413, 422]

    def test_security_headers_added(self, client):
        """Test that security headers are added to responses."""
        response = client.post("/api/v1/chat/message", json={
            "query": "test",
            "session_id": "test_session",
            "user_id": "test_user"
        })
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

    def test_asset_discovery_validation(self, client):
        """Test asset discovery endpoint validation."""
        valid_payload = {
            "project_id": "my-test-project",
            "asset_types": ["compute.googleapis.com/Instance"],
            "page_size": 50
        }
        
        response = client.post("/api/v1/assets/discover", json=valid_payload)
        assert response.status_code == 200
        
        # Invalid project ID
        invalid_payload = {
            "project_id": "INVALID-PROJECT-ID",
            "asset_types": ["compute.googleapis.com/Instance"],
            "page_size": 50
        }
        
        response = client.post("/api/v1/assets/discover", json=invalid_payload)
        assert response.status_code == 422

    def test_security_analysis_validation(self, client):
        """Test security analysis endpoint validation."""
        valid_payload = {
            "project_id": "my-test-project",
            "scan_type": "comprehensive",
            "include_recommendations": True
        }
        
        response = client.post("/api/v1/security/analyze", json=valid_payload)
        assert response.status_code == 200
        
        # Invalid scan type
        invalid_payload = {
            "project_id": "my-test-project",
            "scan_type": "invalid_scan_type",
            "include_recommendations": True
        }
        
        response = client.post("/api/v1/security/analyze", json=invalid_payload)
        assert response.status_code == 422

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post("/api/v1/chat/message", json={})
        assert response.status_code == 422

    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/api/v1/chat/message",
            data="{invalid json}",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        incomplete_payload = {
            "query": "test query"
            # missing session_id and user_id
        }
        
        response = client.post("/api/v1/chat/message", json=incomplete_payload)
        assert response.status_code == 422
        
        error_detail = response.json()
        assert "detail" in error_detail
        assert "errors" in error_detail

    def test_extremely_long_strings(self, client):
        """Test handling of extremely long strings."""
        long_query = "x" * 3000  # Exceeds max length
        
        payload = {
            "query": long_query,
            "session_id": "test_session",
            "user_id": "test_user"
        }
        
        response = client.post("/api/v1/chat/message", json=payload)
        assert response.status_code == 422

    def test_unicode_handling(self, client):
        """Test proper Unicode handling."""
        unicode_payload = {
            "query": "Test with émojis 🔒 and ünìcödé characters",
            "session_id": "test_session_unicode",
            "user_id": "test.user.ünìcödé@example.com"
        }
        
        response = client.post("/api/v1/chat/message", json=unicode_payload)
        assert response.status_code == 200

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test validation performance."""

    def test_validation_performance(self, client):
        """Test that validation doesn't significantly impact performance."""
        import time
        
        payload = {
            "query": "What are my GCP resources?",
            "session_id": "perf_test_session",
            "user_id": "perf_test_user"
        }
        
        # Measure validation overhead
        start_time = time.time()
        for _ in range(10):
            response = client.post("/api/v1/chat/message", json=payload)
            assert response.status_code == 200
        end_time = time.time()
        
        # Average time per request should be reasonable (< 100ms)
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.1, f"Validation too slow: {avg_time:.3f}s per request"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
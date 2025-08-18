"""
Comprehensive test suite for API Keys management endpoints.
Tests key lifecycle, security analysis, and restrictions management.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Import the keys module and related components
from backend.api import keys
from backend.api.keys import (
    get_api_keys_client, router,
    ApiKeyListRequest, ApiKeyCreateRequest, ApiKeyUpdateRequest, ApiKeyRestrictionsRequest
)
from backend.main import app

client = TestClient(app)

# Test fixtures
@pytest.fixture
def mock_api_keys_client():
    """Mock Google Cloud API Keys client."""
    with patch('backend.api.keys.api_keys_v2.ApiKeysClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_api_key():
    """Mock API key object."""
    key = Mock()
    key.name = "projects/test-project/locations/global/keys/test-key-1"
    key.uid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    key.display_name = "Test API Key"
    key.key_string = "AIzaSyC-test-key-string"
    key.create_time = Mock()
    key.create_time.isoformat.return_value = "2024-01-15T10:30:00Z"
    key.update_time = Mock()
    key.update_time.isoformat.return_value = "2024-01-15T11:00:00Z"
    key.delete_time = None
    key.etag = "abc123"
    
    # Mock restrictions
    key.restrictions = Mock()
    key.restrictions.browser_key_restrictions = Mock()
    key.restrictions.browser_key_restrictions.allowed_referrers = ["https://example.com/*"]
    key.restrictions.server_key_restrictions = None
    key.restrictions.android_key_restrictions = None
    key.restrictions.ios_key_restrictions = None
    key.restrictions.api_targets = []
    
    return key

@pytest.fixture
def sample_api_key_data():
    """Sample API key data for testing."""
    return {
        "name": "projects/test-project/locations/global/keys/sample-key-1",
        "uid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "display_name": "Sample API Key 1",
        "key_string": "REDACTED",
        "create_time": "2024-01-15T10:30:00Z",
        "update_time": "2024-01-15T10:30:00Z",
        "restrictions": {
            "browser_key_restrictions": {
                "allowed_referrers": ["https://example.com/*"]
            }
        },
        "etag": "abc123"
    }

@pytest.fixture
def mock_operation():
    """Mock long-running operation."""
    operation = Mock()
    operation.result.return_value = Mock()
    return operation

class TestAPIKeysClientSetup:
    """Test class for API Keys client setup."""

    def test_get_api_keys_client_success(self, mock_api_keys_client):
        """Test successful API Keys client creation."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        client = get_api_keys_client()
        
        assert client is not None
        mock_api_keys_client.assert_called_once()

    def test_get_api_keys_client_unavailable(self):
        """Test API Keys client when library is unavailable."""
        with patch('backend.api.keys.API_KEYS_CLIENT_AVAILABLE', False):
            client = get_api_keys_client()
            assert client is None

    def test_get_api_keys_client_exception(self, mock_api_keys_client):
        """Test API Keys client creation exception."""
        mock_api_keys_client.side_effect = Exception("Client creation failed")
        
        client = get_api_keys_client()
        assert client is None

class TestListAPIKeysEndpoint:
    """Test class for list API keys endpoint."""

    def test_list_api_keys_success_with_real_data(self, mock_api_keys_client, mock_api_key):
        """Test successful API keys listing with real data."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.list_keys.return_value = [mock_api_key]
        
        response = client.post("/api/v1/keys/list", json={
            "project_id": "test-project",
            "page_size": 50,
            "show_deleted": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"] == "api_keys_v2"
        assert data["project_id"] == "test-project"
        assert len(data["keys"]) == 1
        assert data["keys"][0]["name"] == mock_api_key.name
        assert data["keys"][0]["key_string"] == "REDACTED"  # Should be redacted

    def test_list_api_keys_client_unavailable(self):
        """Test list API keys when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.post("/api/v1/keys/list", json={})
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["source"] == "sample_data"
            assert "Install google-cloud-api-keys" in data["message"]
            assert len(data["keys"]) == 2  # Sample data

    def test_list_api_keys_permission_denied(self, mock_api_keys_client):
        """Test list API keys with permission denied error."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.list_keys.side_effect = gcp_exceptions.PermissionDenied("Permission denied")
        
        response = client.post("/api/v1/keys/list", json={"project_id": "test-project"})
        
        assert response.status_code == 403
        assert "Permission denied" in response.json()["detail"]

    def test_list_api_keys_general_exception(self, mock_api_keys_client):
        """Test list API keys with general exception."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.list_keys.side_effect = Exception("API Error")
        
        response = client.post("/api/v1/keys/list", json={"project_id": "test-project"})
        
        assert response.status_code == 500

    def test_list_api_keys_with_complex_restrictions(self, mock_api_keys_client):
        """Test listing API keys with complex restrictions."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Create mock key with multiple restriction types
        complex_key = Mock()
        complex_key.name = "projects/test-project/locations/global/keys/complex-key"
        complex_key.uid = "complex-uid"
        complex_key.display_name = "Complex Key"
        complex_key.key_string = "AIzaComplex"
        complex_key.create_time = Mock()
        complex_key.create_time.isoformat.return_value = "2024-01-15T10:30:00Z"
        complex_key.update_time = None
        complex_key.delete_time = None
        complex_key.etag = "complex123"
        
        # Mock complex restrictions
        complex_key.restrictions = Mock()
        
        # Browser restrictions
        complex_key.restrictions.browser_key_restrictions = Mock()
        complex_key.restrictions.browser_key_restrictions.allowed_referrers = ["https://example.com/*"]
        
        # Server restrictions
        complex_key.restrictions.server_key_restrictions = Mock()
        complex_key.restrictions.server_key_restrictions.allowed_ips = ["192.168.1.0/24"]
        
        # Android restrictions
        complex_key.restrictions.android_key_restrictions = Mock()
        mock_app = Mock()
        mock_app.sha1_fingerprint = "ABC123"
        mock_app.package_name = "com.example.app"
        complex_key.restrictions.android_key_restrictions.allowed_applications = [mock_app]
        
        # iOS restrictions
        complex_key.restrictions.ios_key_restrictions = Mock()
        complex_key.restrictions.ios_key_restrictions.allowed_bundle_ids = ["com.example.ios"]
        
        # API targets
        mock_target = Mock()
        mock_target.service = "translate.googleapis.com"
        mock_target.methods = ["translate"]
        complex_key.restrictions.api_targets = [mock_target]
        
        mock_client_instance.list_keys.return_value = [complex_key]
        
        response = client.post("/api/v1/keys/list", json={"project_id": "test-project"})
        
        assert response.status_code == 200
        data = response.json()
        key_data = data["keys"][0]
        
        # Verify all restriction types are captured
        restrictions = key_data["restrictions"]
        assert "browser_key_restrictions" in restrictions
        assert "server_key_restrictions" in restrictions
        assert "android_key_restrictions" in restrictions
        assert "ios_key_restrictions" in restrictions
        assert "api_targets" in restrictions
        
        assert restrictions["android_key_restrictions"]["allowed_applications"][0]["package_name"] == "com.example.app"

class TestCreateAPIKeyEndpoint:
    """Test class for create API key endpoint."""

    def test_create_api_key_success(self, mock_api_keys_client, mock_operation, mock_api_key):
        """Test successful API key creation."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Mock the operation result
        mock_operation.result.return_value = mock_api_key
        mock_client_instance.create_key.return_value = mock_operation
        
        response = client.post("/api/v1/keys/create", json={
            "project_id": "test-project",
            "display_name": "Test API Key",
            "restrictions": {
                "browser_key_restrictions": {
                    "allowed_referrers": ["https://example.com/*"]
                }
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "key" in data
        assert data["key"]["display_name"] == "Test API Key"
        assert "key_string" in data["key"]  # Should include actual key on creation
        assert "Save the key_string" in data["message"]

    def test_create_api_key_client_unavailable(self):
        """Test create API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.post("/api/v1/keys/create", json={
                "display_name": "Test Key"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "not available" in data["message"]

    def test_create_api_key_already_exists(self, mock_api_keys_client):
        """Test create API key when key already exists."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.create_key.side_effect = gcp_exceptions.AlreadyExists("Key already exists")
        
        response = client.post("/api/v1/keys/create", json={
            "display_name": "Duplicate Key"
        })
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_api_key_with_server_restrictions(self, mock_api_keys_client, mock_operation, mock_api_key):
        """Test API key creation with server restrictions."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_operation.result.return_value = mock_api_key
        mock_client_instance.create_key.return_value = mock_operation
        
        response = client.post("/api/v1/keys/create", json={
            "display_name": "Server Key",
            "restrictions": {
                "server_key_restrictions": {
                    "allowed_ips": ["192.168.1.0/24", "10.0.0.0/8"]
                },
                "api_targets": [
                    {"service": "storage.googleapis.com", "methods": ["get", "list"]}
                ]
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

class TestUpdateAPIKeyEndpoint:
    """Test class for update API key endpoint."""

    def test_update_api_key_success(self, mock_api_keys_client, mock_operation, mock_api_key):
        """Test successful API key update."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Mock get existing key
        mock_client_instance.get_key.return_value = mock_api_key
        
        # Mock update operation
        mock_operation.result.return_value = mock_api_key
        mock_client_instance.update_key.return_value = mock_operation
        
        response = client.patch("/api/v1/keys/update", json={
            "key_id": "test-key-1",
            "display_name": "Updated API Key",
            "restrictions": {
                "browser_key_restrictions": {
                    "allowed_referrers": ["https://newdomain.com/*"]
                }
            }
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "updated successfully" in data["message"]

    def test_update_api_key_client_unavailable(self):
        """Test update API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.patch("/api/v1/keys/update", json={
                "key_id": "test-key-1",
                "display_name": "Updated Key"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

class TestDeleteAPIKeyEndpoint:
    """Test class for delete API key endpoint."""

    def test_delete_api_key_success(self, mock_api_keys_client, mock_operation):
        """Test successful API key deletion."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.delete_key.return_value = mock_operation
        
        response = client.delete("/api/v1/keys/delete/test-key-1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]
        assert "deleted_at" in data

    def test_delete_api_key_not_found(self, mock_api_keys_client):
        """Test delete API key when key not found."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.delete_key.side_effect = gcp_exceptions.NotFound("Key not found")
        
        response = client.delete("/api/v1/keys/delete/nonexistent-key")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_api_key_client_unavailable(self):
        """Test delete API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.delete("/api/v1/keys/delete/test-key-1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

class TestUndeleteAPIKeyEndpoint:
    """Test class for undelete API key endpoint."""

    def test_undelete_api_key_success(self, mock_api_keys_client, mock_operation, mock_api_key):
        """Test successful API key undeletion."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_operation.result.return_value = mock_api_key
        mock_client_instance.undelete_key.return_value = mock_operation
        
        response = client.post("/api/v1/keys/undelete/test-key-1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "restored successfully" in data["message"]

    def test_undelete_api_key_client_unavailable(self):
        """Test undelete API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.post("/api/v1/keys/undelete/test-key-1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

class TestGetAPIKeyEndpoint:
    """Test class for get API key endpoint."""

    def test_get_api_key_success(self, mock_api_keys_client, mock_api_key):
        """Test successful API key retrieval."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.get_key.return_value = mock_api_key
        
        response = client.get("/api/v1/keys/get/test-key-1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"] == "api_keys_v2"
        assert "key" in data
        assert data["key"]["name"] == mock_api_key.name

    def test_get_api_key_not_found(self, mock_api_keys_client):
        """Test get API key when key not found."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.get_key.side_effect = gcp_exceptions.NotFound("Key not found")
        
        response = client.get("/api/v1/keys/get/nonexistent-key")
        
        assert response.status_code == 404

    def test_get_api_key_client_unavailable(self):
        """Test get API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.get("/api/v1/keys/get/test-key-1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

class TestLookupAPIKeyEndpoint:
    """Test class for lookup API key endpoint."""

    def test_lookup_api_key_success(self, mock_api_keys_client):
        """Test successful API key lookup."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        mock_response = Mock()
        mock_response.parent = "projects/test-project/locations/global"
        mock_response.name = "projects/test-project/locations/global/keys/test-key-1"
        mock_client_instance.lookup_key.return_value = mock_response
        
        response = client.get("/api/v1/keys/lookup?key_string=AIzaSyTest123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "parent" in data
        assert "name" in data

    def test_lookup_api_key_not_found(self, mock_api_keys_client):
        """Test lookup API key when key not found."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.lookup_key.side_effect = gcp_exceptions.NotFound("Key not found")
        
        response = client.get("/api/v1/keys/lookup?key_string=InvalidKey")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"]

    def test_lookup_api_key_client_unavailable(self):
        """Test lookup API key when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.get("/api/v1/keys/lookup?key_string=AIzaSyTest123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False

class TestAnalyzeAPIKeysSecurityEndpoint:
    """Test class for analyze API keys security endpoint."""

    def test_analyze_api_keys_security_with_real_data(self, mock_api_keys_client):
        """Test security analysis with real API data."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Mock keys with different security configurations
        unrestricted_key = Mock()
        unrestricted_key.display_name = "Unrestricted Key"
        unrestricted_key.restrictions = None
        
        restricted_key = Mock()
        restricted_key.display_name = "Restricted Key"
        restricted_key.restrictions = Mock()
        restricted_key.restrictions.browser_key_restrictions = Mock()
        restricted_key.restrictions.server_key_restrictions = None
        restricted_key.restrictions.api_targets = []
        
        secure_key = Mock()
        secure_key.display_name = "Secure Key"
        secure_key.restrictions = Mock()
        secure_key.restrictions.browser_key_restrictions = None
        secure_key.restrictions.server_key_restrictions = Mock()
        secure_key.restrictions.api_targets = [Mock()]
        
        mock_client_instance.list_keys.return_value = [unrestricted_key, restricted_key, secure_key]
        
        response = client.get("/api/v1/keys/analyze?project_id=test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["source"] == "live_analysis"
        
        analysis = data["analysis"]
        assert analysis["statistics"]["total_keys"] == 3
        assert analysis["statistics"]["unrestricted_keys"] == 1
        assert analysis["statistics"]["keys_with_browser_restrictions"] == 1
        assert analysis["statistics"]["keys_with_server_restrictions"] == 1
        assert analysis["statistics"]["keys_with_api_restrictions"] == 1
        
        # Should have risks for unrestricted key
        assert len(analysis["risks"]) >= 1
        assert any(risk["level"] == "HIGH" for risk in analysis["risks"])

    def test_analyze_api_keys_security_client_unavailable(self):
        """Test security analysis when client is unavailable."""
        with patch('backend.api.keys.get_api_keys_client', return_value=None):
            response = client.get("/api/v1/keys/analyze")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["source"] == "sample_analysis"
            
            analysis = data["analysis"]
            assert "risks" in analysis
            assert "recommendations" in analysis
            assert "statistics" in analysis

    def test_analyze_api_keys_security_exception(self, mock_api_keys_client):
        """Test security analysis with exception."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.list_keys.side_effect = Exception("API Error")
        
        response = client.get("/api/v1/keys/analyze")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

class TestHealthCheckEndpoint:
    """Test class for health check endpoint."""

    def test_health_check_success(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/keys/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api_keys"
        assert "client_available" in data
        assert "project_id" in data
        assert "timestamp" in data

class TestPydanticModels:
    """Test class for Pydantic model validation."""

    def test_api_key_list_request_validation(self):
        """Test ApiKeyListRequest validation."""
        request = ApiKeyListRequest(
            project_id="test-project",
            page_size=50,
            show_deleted=True
        )
        
        assert request.project_id == "test-project"
        assert request.page_size == 50
        assert request.show_deleted is True

    def test_api_key_create_request_validation(self):
        """Test ApiKeyCreateRequest validation."""
        request = ApiKeyCreateRequest(
            project_id="test-project",
            display_name="Test Key",
            restrictions={
                "browser_key_restrictions": {
                    "allowed_referrers": ["https://example.com/*"]
                }
            }
        )
        
        assert request.project_id == "test-project"
        assert request.display_name == "Test Key"
        assert "browser_key_restrictions" in request.restrictions

    def test_api_key_update_request_validation(self):
        """Test ApiKeyUpdateRequest validation."""
        request = ApiKeyUpdateRequest(
            key_id="test-key-1",
            display_name="Updated Key",
            restrictions={
                "api_targets": [
                    {"service": "translate.googleapis.com"}
                ]
            }
        )
        
        assert request.key_id == "test-key-1"
        assert request.display_name == "Updated Key"
        assert "api_targets" in request.restrictions

    def test_api_key_restrictions_request_validation(self):
        """Test ApiKeyRestrictionsRequest validation."""
        request = ApiKeyRestrictionsRequest(
            browser_key_restrictions={"allowed_referrers": ["https://example.com/*"]},
            server_key_restrictions={"allowed_ips": ["192.168.1.0/24"]},
            android_key_restrictions={"allowed_applications": ["com.example.app"]},
            ios_key_restrictions={"allowed_bundle_ids": ["com.example.ios"]},
            api_targets=[{"service": "storage.googleapis.com", "methods": ["get"]}]
        )
        
        assert request.browser_key_restrictions is not None
        assert request.server_key_restrictions is not None
        assert request.android_key_restrictions is not None
        assert request.ios_key_restrictions is not None
        assert len(request.api_targets) == 1

class TestErrorHandling:
    """Test class for error handling scenarios."""

    def test_invalid_request_data(self):
        """Test handling of invalid request data."""
        response = client.post("/api/v1/keys/create", json={
            # Missing required display_name
        })
        
        assert response.status_code == 422  # Validation error

    def test_api_timeout_handling(self, mock_api_keys_client):
        """Test handling of API timeout errors."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.list_keys.side_effect = gcp_exceptions.DeadlineExceeded("Timeout")
        
        response = client.post("/api/v1/keys/list", json={"project_id": "test-project"})
        
        assert response.status_code == 500

    def test_quota_exceeded_handling(self, mock_api_keys_client):
        """Test handling of quota exceeded errors."""
        from google.api_core import exceptions as gcp_exceptions
        
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        mock_client_instance.create_key.side_effect = gcp_exceptions.ResourceExhausted("Quota exceeded")
        
        response = client.post("/api/v1/keys/create", json={"display_name": "Test Key"})
        
        assert response.status_code == 500

class TestIntegrationScenarios:
    """Test class for integration scenarios."""

    def test_full_api_key_lifecycle(self, mock_api_keys_client, mock_operation, mock_api_key):
        """Test complete API key lifecycle."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Test creation
        mock_operation.result.return_value = mock_api_key
        mock_client_instance.create_key.return_value = mock_operation
        
        create_response = client.post("/api/v1/keys/create", json={
            "display_name": "Lifecycle Test Key",
            "restrictions": {
                "browser_key_restrictions": {
                    "allowed_referrers": ["https://example.com/*"]
                }
            }
        })
        assert create_response.status_code == 200
        
        # Test retrieval
        mock_client_instance.get_key.return_value = mock_api_key
        
        get_response = client.get("/api/v1/keys/get/test-key-1")
        assert get_response.status_code == 200
        
        # Test update
        mock_client_instance.update_key.return_value = mock_operation
        
        update_response = client.patch("/api/v1/keys/update", json={
            "key_id": "test-key-1",
            "display_name": "Updated Lifecycle Key"
        })
        assert update_response.status_code == 200
        
        # Test deletion
        mock_client_instance.delete_key.return_value = mock_operation
        
        delete_response = client.delete("/api/v1/keys/delete/test-key-1")
        assert delete_response.status_code == 200

    def test_security_analysis_comprehensive(self, mock_api_keys_client):
        """Test comprehensive security analysis scenario."""
        mock_client_instance = Mock()
        mock_api_keys_client.return_value = mock_client_instance
        
        # Create multiple keys with various security issues
        keys = []
        
        # Unrestricted key (high risk)
        unrestricted = Mock()
        unrestricted.display_name = "Unrestricted Production Key"
        unrestricted.restrictions = None
        keys.append(unrestricted)
        
        # Key with weak restrictions (medium risk)
        weak = Mock()
        weak.display_name = "Weak Development Key"
        weak.restrictions = Mock()
        weak.restrictions.browser_key_restrictions = Mock()
        weak.restrictions.server_key_restrictions = None
        weak.restrictions.api_targets = []  # No API restrictions
        keys.append(weak)
        
        # Properly secured key (low risk)
        secure = Mock()
        secure.display_name = "Secure API Key"
        secure.restrictions = Mock()
        secure.restrictions.browser_key_restrictions = None
        secure.restrictions.server_key_restrictions = Mock()
        secure.restrictions.api_targets = [Mock()]  # Has API restrictions
        keys.append(secure)
        
        mock_client_instance.list_keys.return_value = keys
        
        response = client.get("/api/v1/keys/analyze")
        
        assert response.status_code == 200
        data = response.json()
        analysis = data["analysis"]
        
        # Should identify security risks
        assert analysis["statistics"]["unrestricted_keys"] == 1
        assert len(analysis["risks"]) >= 1
        
        # Should provide recommendations
        assert len(analysis["recommendations"]) > 0
        assert any("restrictions" in rec for rec in analysis["recommendations"])

if __name__ == "__main__":
    pytest.main([__file__])
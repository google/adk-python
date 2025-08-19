"""
Comprehensive test suite for IAM API endpoints - TASK-003.

Tests IAM analysis, overprivileged account detection, service account analysis,
role management, and security assessments with comprehensive mocking.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json

# Import the IAM module and related components
from backend.api.iam import router
from backend.main import app

client = TestClient(app)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_iam_client():
    """Mock Google Cloud IAM client."""
    with patch('backend.api.iam.iam.IAMCredentialsServiceClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_service_account():
    """Mock service account object."""
    sa = Mock()
    sa.name = "projects/test-project/serviceAccounts/test-sa@test-project.iam.gserviceaccount.com"
    sa.project_id = "test-project"
    sa.unique_id = "123456789012345678901"
    sa.email = "test-sa@test-project.iam.gserviceaccount.com"
    sa.display_name = "Test Service Account"
    sa.description = "Test service account for unit tests"
    sa.oauth2_client_id = "123456789012345678901"
    sa.disabled = False
    sa.etag = "abc123"
    return sa

@pytest.fixture
def mock_service_account_key():
    """Mock service account key object."""
    key = Mock()
    key.name = "projects/test-project/serviceAccounts/test-sa@test-project.iam.gserviceaccount.com/keys/1"
    key.private_key_type = "TYPE_GOOGLE_CREDENTIALS_FILE"
    key.key_algorithm = "KEY_ALG_RSA_2048"
    key.private_key_data = b"fake-key-data"
    key.public_key_data = "fake-public-key"
    key.valid_after_time = datetime.now() - timedelta(days=30)
    key.valid_before_time = datetime.now() + timedelta(days=365)
    key.key_origin = "GOOGLE_PROVIDED"
    key.key_type = "USER_MANAGED"
    return key

@pytest.fixture
def mock_iam_policy():
    """Mock IAM policy object."""
    policy = Mock()
    policy.version = 3
    policy.etag = "policy-etag-123"
    
    # Mock policy bindings
    binding1 = Mock()
    binding1.role = "roles/storage.admin"
    binding1.members = ["serviceAccount:test-sa@test-project.iam.gserviceaccount.com"]
    binding1.condition = None
    
    binding2 = Mock()
    binding2.role = "roles/iam.serviceAccountUser"
    binding2.members = ["user:overprivileged@example.com", "allUsers"]
    binding2.condition = None
    
    policy.bindings = [binding1, binding2]
    return policy

@pytest.fixture
def mock_credentials():
    """Mock GCP credentials."""
    with patch('backend.api.iam._get_credentials') as mock_creds:
        mock_creds.return_value = Mock()
        yield mock_creds

# ============================================================================
# IAM ANALYSIS ENDPOINT TESTS
# ============================================================================

class TestIAMAnalysisEndpoints:
    """Test IAM analysis endpoints."""

    def test_analyze_iam_success(self, mock_credentials, mock_iam_client, mock_service_account):
        """Test successful IAM analysis."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_iam_client.return_value = mock_client_instance
        
        # Mock service account listing
        with patch('backend.api.iam.service_account.ServiceAccountServiceClient') as mock_sa_client:
            mock_sa_instance = Mock()
            mock_sa_client.return_value = mock_sa_instance
            mock_sa_instance.list_service_accounts.return_value = [mock_service_account]
            
            # Mock key listing
            mock_sa_instance.list_service_account_keys.return_value = []
            
            response = client.get("/api/v1/iam/analyze/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["project_id"] == "test-project"
        assert "service_accounts" in data
        assert "security_findings" in data
        assert "summary" in data

    def test_analyze_iam_with_filters(self, mock_credentials, mock_iam_client):
        """Test IAM analysis with filters."""
        response = client.get("/api/v1/iam/analyze/test-project?include_keys=true&check_overprivileged=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detailed_analysis" in data

    def test_analyze_iam_no_credentials(self, mock_credentials):
        """Test IAM analysis without credentials."""
        mock_credentials.return_value = None
        
        response = client.get("/api/v1/iam/analyze/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data_source"] == "api_failed"  # Falls back to mock data

    def test_get_service_account_details(self):
        """Test getting details for specific service account."""
        sa_email = "test-sa@test-project.iam.gserviceaccount.com"
        response = client.get(f"/api/v1/iam/service-accounts/test-project/{sa_email}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "service_account" in data
        assert "keys" in data
        assert "risk_assessment" in data

    def test_get_service_account_not_found(self):
        """Test getting details for non-existent service account."""
        response = client.get("/api/v1/iam/service-accounts/test-project/nonexistent@test-project.iam.gserviceaccount.com")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

# ============================================================================
# SERVICE ACCOUNT MANAGEMENT TESTS
# ============================================================================

class TestServiceAccountManagement:
    """Test service account management operations."""

    def test_list_service_accounts(self, mock_credentials):
        """Test listing service accounts."""
        response = client.get("/api/v1/iam/service-accounts/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "service_accounts" in data
        assert "total_count" in data

    def test_create_service_account_request(self):
        """Test service account creation request."""
        sa_data = {
            "account_id": "new-test-sa",
            "display_name": "New Test Service Account",
            "description": "A new service account for testing"
        }
        
        response = client.post("/api/v1/iam/service-accounts/test-project", json=sa_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preview_commands" in data  # Returns commands since this is read-only

    def test_delete_service_account_request(self):
        """Test service account deletion request."""
        response = client.delete("/api/v1/iam/service-accounts/test-project/test-sa@test-project.iam.gserviceaccount.com")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preview_commands" in data

# ============================================================================
# SECURITY ANALYSIS TESTS
# ============================================================================

class TestIAMSecurityAnalysis:
    """Test IAM security analysis functionality."""

    def test_overprivileged_account_detection(self, mock_credentials, mock_iam_policy):
        """Test detection of overprivileged accounts."""
        with patch('backend.api.iam._check_overprivileged_accounts') as mock_check:
            # Mock overprivileged detection
            mock_check.return_value = [
                {
                    "account": "overprivileged@example.com",
                    "risk_score": 85,
                    "issues": ["Has admin roles", "External domain"],
                    "recommendations": ["Remove unnecessary roles", "Add domain restrictions"]
                }
            ]
            
            response = client.get("/api/v1/iam/analyze/test-project?check_overprivileged=true")
            
            assert response.status_code == 200
            data = response.json()
            
            # Check for overprivileged analysis
            assert "security_findings" in data
            findings = data["security_findings"]
            
            # Should have high-risk findings
            assert any(finding["risk_score"] > 80 for finding in findings.get("critical", []))

    def test_stale_key_detection(self, mock_credentials, mock_service_account_key):
        """Test detection of stale service account keys."""
        # Create old key (91 days old)
        old_key = mock_service_account_key
        old_key.valid_after_time = datetime.now() - timedelta(days=91)
        
        with patch('backend.api.iam._get_service_account_keys') as mock_keys:
            mock_keys.return_value = [old_key]
            
            response = client.get("/api/v1/iam/analyze/test-project?include_keys=true")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should detect stale keys
            findings = data["security_findings"]
            assert any("stale" in str(finding).lower() or "old" in str(finding).lower() 
                     for finding_list in findings.values() 
                     for finding in finding_list)

    def test_external_user_detection(self, mock_iam_policy):
        """Test detection of external users."""
        # Add external user to policy
        external_binding = Mock()
        external_binding.role = "roles/owner"
        external_binding.members = ["user:external@external-domain.com"]
        external_binding.condition = None
        
        mock_iam_policy.bindings.append(external_binding)
        
        with patch('backend.api.iam._analyze_iam_bindings') as mock_analyze:
            mock_analyze.return_value = {
                "external_users": ["external@external-domain.com"],
                "wildcard_bindings": [],
                "admin_roles": []
            }
            
            response = client.get("/api/v1/iam/analyze/test-project")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should detect external users
            findings = data["security_findings"]
            assert any("external" in str(finding).lower() 
                     for finding_list in findings.values() 
                     for finding in finding_list)

    def test_wildcard_binding_detection(self, mock_iam_policy):
        """Test detection of wildcard bindings (allUsers, allAuthenticatedUsers)."""
        with patch('backend.api.iam._analyze_iam_bindings') as mock_analyze:
            mock_analyze.return_value = {
                "external_users": [],
                "wildcard_bindings": [
                    {
                        "role": "roles/iam.serviceAccountUser",
                        "members": ["allUsers"],
                        "risk_level": "CRITICAL"
                    }
                ],
                "admin_roles": []
            }
            
            response = client.get("/api/v1/iam/analyze/test-project")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should detect wildcard bindings
            findings = data["security_findings"]
            critical_findings = findings.get("critical", [])
            assert any("allUsers" in str(finding) or "wildcard" in str(finding).lower() 
                     for finding in critical_findings)

# ============================================================================
# KEY MANAGEMENT TESTS
# ============================================================================

class TestServiceAccountKeyManagement:
    """Test service account key management."""

    def test_list_service_account_keys(self, mock_credentials, mock_service_account_key):
        """Test listing service account keys."""
        with patch('backend.api.iam._get_service_account_keys') as mock_keys:
            mock_keys.return_value = [mock_service_account_key]
            
            sa_email = "test-sa@test-project.iam.gserviceaccount.com"
            response = client.get(f"/api/v1/iam/service-accounts/test-project/{sa_email}/keys")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "keys" in data
            assert len(data["keys"]) > 0

    def test_create_service_account_key_request(self):
        """Test service account key creation request."""
        sa_email = "test-sa@test-project.iam.gserviceaccount.com"
        key_data = {
            "key_algorithm": "KEY_ALG_RSA_2048",
            "private_key_type": "TYPE_GOOGLE_CREDENTIALS_FILE"
        }
        
        response = client.post(f"/api/v1/iam/service-accounts/test-project/{sa_email}/keys", json=key_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preview_commands" in data

    def test_delete_service_account_key_request(self):
        """Test service account key deletion request."""
        sa_email = "test-sa@test-project.iam.gserviceaccount.com"
        key_id = "key-12345"
        
        response = client.delete(f"/api/v1/iam/service-accounts/test-project/{sa_email}/keys/{key_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "preview_commands" in data

# ============================================================================
# ROLE ANALYSIS TESTS
# ============================================================================

class TestRoleAnalysis:
    """Test role analysis functionality."""

    def test_analyze_roles(self):
        """Test role analysis endpoint."""
        response = client.get("/api/v1/iam/roles/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "roles" in data
        assert "predefined_roles" in data
        assert "custom_roles" in data

    def test_get_role_details(self):
        """Test getting details for specific role."""
        response = client.get("/api/v1/iam/roles/test-project/roles/storage.admin")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "role" in data
        assert "permissions" in data
        assert "usage_analysis" in data

    def test_analyze_custom_roles(self):
        """Test custom role analysis."""
        response = client.get("/api/v1/iam/roles/test-project?type=custom")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "custom_roles" in data

# ============================================================================
# PERMISSIONS ANALYSIS TESTS
# ============================================================================

class TestPermissionsAnalysis:
    """Test permissions analysis functionality."""

    def test_analyze_permissions(self):
        """Test permissions analysis endpoint."""
        response = client.get("/api/v1/iam/permissions/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "permissions" in data
        assert "analysis" in data

    def test_check_user_permissions(self):
        """Test checking permissions for specific user."""
        user_email = "test@example.com"
        response = client.get(f"/api/v1/iam/permissions/test-project/users/{user_email}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "user" in data
        assert "effective_permissions" in data
        assert "role_bindings" in data

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestIAMErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_project_id(self):
        """Test handling of invalid project ID."""
        response = client.get("/api/v1/iam/analyze/invalid-project-id!")
        
        # Should handle gracefully or return appropriate error
        assert response.status_code in [400, 200]  # May return mock data

    def test_service_account_not_found(self):
        """Test handling of non-existent service account."""
        response = client.get("/api/v1/iam/service-accounts/test-project/nonexistent@test-project.iam.gserviceaccount.com")
        
        assert response.status_code == 404

    def test_api_permission_denied(self, mock_credentials):
        """Test handling of permission denied errors."""
        with patch('backend.api.iam.service_account.ServiceAccountServiceClient') as mock_client:
            mock_client.side_effect = Exception("Permission denied")
            
            response = client.get("/api/v1/iam/analyze/test-project")
            
            # Should handle gracefully
            assert response.status_code == 200
            data = response.json()
            # Should fallback to mock data
            assert data["data_source"] == "api_failed"

    def test_malformed_service_account_email(self):
        """Test handling of malformed service account email."""
        response = client.get("/api/v1/iam/service-accounts/test-project/invalid-email")
        
        assert response.status_code in [400, 404]

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIAMIntegration:
    """Test IAM integration scenarios."""

    def test_full_iam_security_assessment(self, mock_credentials):
        """Test complete IAM security assessment workflow."""
        response = client.get("/api/v1/iam/analyze/test-project?detailed=true&include_keys=true&check_overprivileged=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Should have comprehensive analysis
        assert "service_accounts" in data
        assert "security_findings" in data
        assert "summary" in data
        assert "recommendations" in data
        
        # Check security findings structure
        findings = data["security_findings"]
        assert "critical" in findings
        assert "high" in findings
        assert "medium" in findings
        assert "low" in findings

    def test_iam_compliance_check(self):
        """Test IAM compliance checking."""
        response = client.post("/api/v1/iam/compliance/test-project", json={
            "frameworks": ["SOC2", "HIPAA", "PCI-DSS"],
            "include_remediation": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "compliance_results" in data
        assert "gaps" in data
        assert "remediation_plan" in data

    def test_iam_recommendations_generation(self):
        """Test IAM recommendations generation."""
        response = client.get("/api/v1/iam/recommendations/test-project")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recommendations" in data
        assert "priority_actions" in data
        assert "automation_scripts" in data

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestIAMPerformance:
    """Test IAM API performance."""

    def test_large_project_analysis_performance(self, mock_credentials):
        """Test performance with large project (many service accounts)."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/iam/analyze/large-project")
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max
        assert response.status_code == 200

    def test_concurrent_requests_handling(self, mock_credentials):
        """Test handling of concurrent IAM analysis requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/iam/analyze/test-project")
            results.append(response.status_code)
        
        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
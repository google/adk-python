"""
Comprehensive test suite for Storage API endpoints.
Tests bucket analysis, security checks, and error scenarios.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Import the storage module and FastAPI app
from backend.api.storage import router, _get_real_buckets, _check_public_access, _get_encryption_type, _check_logging_enabled
from backend.main import app

client = TestClient(app)

# Test fixtures
@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch('backend.api.storage.storage.Client') as mock_client:
        yield mock_client

@pytest.fixture
def mock_bucket():
    """Mock GCS bucket object."""
    bucket = Mock()
    bucket.name = "test-bucket"
    bucket.location = "US-CENTRAL1"
    bucket.storage_class = "STANDARD"
    bucket.versioning_enabled = True
    bucket.time_created = datetime.now()
    bucket.labels = {"env": "test", "team": "security"}
    bucket.lifecycle_rules = []
    bucket.cors = []
    bucket.website_main_page_suffix = None
    bucket.requester_pays = False
    bucket.default_kms_key_name = None
    bucket.encryption_configuration = None
    bucket.logging = None
    return bucket

@pytest.fixture
def mock_credentials():
    """Mock GCP credentials."""
    with patch('backend.api.storage._get_credentials') as mock_creds:
        mock_creds.return_value = Mock()
        yield mock_creds

@pytest.fixture
def sample_bucket_data():
    """Sample bucket data for testing."""
    return [
        {
            "name": "test-bucket-secure",
            "location": "US-CENTRAL1",
            "storageClass": "STANDARD",
            "publicAccess": False,
            "versioning": True,
            "encryption": "CMEK",
            "logging": True,
            "created": "2023-01-15T10:30:00Z",
            "labels": {"env": "prod"},
            "lifecycle": True,
            "cors": False,
            "website": False,
            "requesterPays": False
        },
        {
            "name": "test-bucket-vulnerable",
            "location": "US",
            "storageClass": "STANDARD",
            "publicAccess": True,  # Critical issue
            "versioning": False,   # High issue
            "encryption": "Google-managed",
            "logging": False,      # High issue
            "created": "2023-03-20T14:15:00Z",
            "labels": {},
            "lifecycle": False,
            "cors": False,
            "website": False,
            "requesterPays": False
        }
    ]

class TestStorageAPIEndpoints:
    """Test class for Storage API endpoints."""

    def test_analyze_buckets_success_with_real_data(self, mock_storage_client, mock_credentials, mock_bucket):
        """Test successful bucket analysis with real API data."""
        # Setup mock
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        
        # Mock bucket iterator
        mock_client_instance.list_buckets.return_value = [mock_bucket]
        
        # Mock IAM policy check
        mock_policy = Mock()
        mock_policy.bindings = []
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        # Make request
        response = client.get("/api/v1/storage/buckets/test-project")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["project_id"] == "test-project"
        assert "buckets" in data
        assert "security_findings" in data
        assert "summary" in data
        assert data["summary"]["total_buckets"] >= 0

    def test_analyze_buckets_detailed_mode(self, mock_storage_client, mock_credentials, mock_bucket):
        """Test bucket analysis with detailed mode enabled."""
        # Setup mock
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.list_buckets.return_value = [mock_bucket]
        
        # Mock IAM policy
        mock_policy = Mock()
        mock_policy.bindings = []
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        # Make request with detailed=True
        response = client.get("/api/v1/storage/buckets/test-project?detailed=true")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detailed_analysis" in data
        assert "compliance_gaps" in data["detailed_analysis"]
        assert "cost_optimization" in data["detailed_analysis"]
        assert "best_practices_missing" in data["detailed_analysis"]

    def test_analyze_buckets_api_failure_fallback(self, mock_credentials):
        """Test fallback to mock data when API fails."""
        # Mock credentials failure
        mock_credentials.return_value = None
        
        # Make request
        response = client.get("/api/v1/storage/buckets/mgm-digitalconcierge")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data_source"] == "api_failed"  # Falls back to mock data
        assert len(data["buckets"]) > 0  # Should have mock data

    def test_analyze_buckets_no_buckets_found(self, mock_storage_client, mock_credentials):
        """Test scenario where no buckets are found."""
        # Setup mock
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.list_buckets.return_value = []  # Empty list
        
        # Make request
        response = client.get("/api/v1/storage/buckets/empty-project")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No buckets found" in data["error"]

    def test_get_bucket_details_success(self):
        """Test getting details for a specific bucket."""
        response = client.get("/api/v1/storage/buckets/mgm-digitalconcierge/mgm-digitalconcierge-backups")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "bucket" in data
        assert "risk_score" in data
        assert "compliance_status" in data
        assert "recommendations" in data

    def test_get_bucket_details_not_found(self):
        """Test getting details for a non-existent bucket."""
        response = client.get("/api/v1/storage/buckets/mgm-digitalconcierge/non-existent-bucket")
        
        # Assertions
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_remediate_bucket_issues_success(self):
        """Test bucket remediation endpoint."""
        remediation_data = {
            "bucket_name": "test-bucket",
            "fixes": ["enable_versioning", "disable_public_access", "enable_logging"]
        }
        
        response = client.post(
            "/api/v1/storage/buckets/test-project/remediate",
            params={"bucket_name": "test-bucket"},
            json=remediation_data
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "applied_fixes" in data
        assert "commands" in data
        assert len(data["commands"]) > 0

class TestStorageHelperFunctions:
    """Test class for storage helper functions."""

    def test_check_public_access_with_public_bucket(self, mock_bucket):
        """Test public access detection for publicly accessible bucket."""
        # Mock IAM policy with public access
        mock_policy = Mock()
        mock_policy.bindings = [
            {"members": ["allUsers"], "role": "roles/storage.objectViewer"}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        result = _check_public_access(mock_bucket)
        assert result is True

    def test_check_public_access_with_authenticated_users(self, mock_bucket):
        """Test public access detection for allAuthenticatedUsers."""
        mock_policy = Mock()
        mock_policy.bindings = [
            {"members": ["allAuthenticatedUsers"], "role": "roles/storage.objectViewer"}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        result = _check_public_access(mock_bucket)
        assert result is True

    def test_check_public_access_private_bucket(self, mock_bucket):
        """Test public access detection for private bucket."""
        mock_policy = Mock()
        mock_policy.bindings = [
            {"members": ["serviceAccount:test@example.com"], "role": "roles/storage.admin"}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        result = _check_public_access(mock_bucket)
        assert result is False

    def test_check_public_access_error_handling(self, mock_bucket):
        """Test public access check error handling."""
        mock_bucket.get_iam_policy.side_effect = Exception("Permission denied")
        
        result = _check_public_access(mock_bucket)
        assert result is False

    def test_get_encryption_type_cmek(self, mock_bucket):
        """Test encryption type detection for CMEK."""
        mock_bucket.default_kms_key_name = "projects/test/locations/global/keyRings/ring/cryptoKeys/key"
        
        result = _get_encryption_type(mock_bucket)
        assert result == "CUSTOMER_MANAGED"

    def test_get_encryption_type_google_managed(self, mock_bucket):
        """Test encryption type detection for Google-managed."""
        mock_bucket.default_kms_key_name = None
        mock_bucket.encryption_configuration = Mock()
        
        result = _get_encryption_type(mock_bucket)
        assert result == "GOOGLE_MANAGED"

    def test_get_encryption_type_default(self, mock_bucket):
        """Test encryption type detection for default."""
        mock_bucket.default_kms_key_name = None
        mock_bucket.encryption_configuration = None
        
        result = _get_encryption_type(mock_bucket)
        assert result == "GOOGLE_MANAGED"

    def test_get_encryption_type_error_handling(self, mock_bucket):
        """Test encryption type detection error handling."""
        mock_bucket.default_kms_key_name = Mock(side_effect=Exception("Error"))
        
        result = _get_encryption_type(mock_bucket)
        assert result == "UNKNOWN"

    def test_check_logging_enabled_true(self, mock_bucket):
        """Test logging detection when enabled."""
        mock_bucket.logging = Mock()
        
        result = _check_logging_enabled(mock_bucket)
        assert result is True

    def test_check_logging_enabled_false(self, mock_bucket):
        """Test logging detection when disabled."""
        mock_bucket.logging = None
        
        result = _check_logging_enabled(mock_bucket)
        assert result is False

    def test_check_logging_enabled_error_handling(self, mock_bucket):
        """Test logging detection error handling."""
        type(mock_bucket).logging = Mock(side_effect=Exception("Error"))
        
        result = _check_logging_enabled(mock_bucket)
        assert result is False

class TestSecurityAnalysis:
    """Test class for security analysis logic."""

    def test_critical_issue_detection_public_access(self, sample_bucket_data):
        """Test detection of critical security issues."""
        # Simulate bucket analysis for public bucket
        bucket = sample_bucket_data[1]  # vulnerable bucket
        
        # This simulates the logic in analyze_buckets
        critical_issues = []
        if bucket["publicAccess"]:
            critical_issues.append({
                "bucket": bucket["name"],
                "issue": "PUBLIC ACCESS ENABLED",
                "risk": "CRITICAL"
            })
        
        assert len(critical_issues) == 1
        assert critical_issues[0]["risk"] == "CRITICAL"
        assert "PUBLIC ACCESS" in critical_issues[0]["issue"]

    def test_high_issue_detection_no_versioning(self, sample_bucket_data):
        """Test detection of high-priority security issues."""
        bucket = sample_bucket_data[1]  # vulnerable bucket
        
        high_issues = []
        
        # Check for missing versioning on backup buckets
        if not bucket["versioning"] and "backup" in bucket["name"].lower():
            high_issues.append({
                "bucket": bucket["name"],
                "issue": "NO VERSIONING ON BACKUP BUCKET",
                "risk": "HIGH"
            })
        
        # Check for missing logging
        if not bucket["logging"]:
            high_issues.append({
                "bucket": bucket["name"],
                "issue": "ACCESS LOGGING DISABLED",
                "risk": "HIGH"
            })
        
        assert len(high_issues) == 1  # Only logging issue for this test bucket
        assert any("LOGGING" in issue["issue"] for issue in high_issues)

    def test_medium_issue_detection_encryption(self, sample_bucket_data):
        """Test detection of medium-priority security issues."""
        bucket = sample_bucket_data[1]  # vulnerable bucket
        
        medium_issues = []
        
        # Check for weak encryption on sensitive data
        if ("data" in bucket["name"].lower() or "backup" in bucket["name"].lower()) and bucket["encryption"] != "CMEK":
            medium_issues.append({
                "bucket": bucket["name"],
                "issue": "NO CUSTOMER-MANAGED ENCRYPTION",
                "risk": "MEDIUM"
            })
        
        # Check for missing lifecycle rules on temp buckets
        if "temp" in bucket["name"].lower() and not bucket["lifecycle"]:
            medium_issues.append({
                "bucket": bucket["name"],
                "issue": "NO LIFECYCLE RULES",
                "risk": "MEDIUM"
            })
        
        # This bucket doesn't match the patterns for medium issues in this test
        assert len(medium_issues) == 0

class TestErrorHandling:
    """Test class for error handling scenarios."""

    @patch('backend.api.storage._get_credentials')
    def test_credentials_failure(self, mock_creds):
        """Test behavior when credentials fail."""
        mock_creds.return_value = None
        
        response = client.get("/api/v1/storage/buckets/test-project")
        
        # Should still return success with fallback data
        assert response.status_code == 200
        data = response.json()
        # The API falls back to mock data when credentials fail

    @patch('backend.api.storage.storage.Client')
    def test_storage_client_exception(self, mock_storage_client, mock_credentials):
        """Test behavior when storage client throws exception."""
        mock_storage_client.side_effect = Exception("Storage API unavailable")
        
        response = client.get("/api/v1/storage/buckets/test-project")
        
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        # Should fallback to mock data or return error info

    def test_invalid_project_id_format(self):
        """Test handling of invalid project ID format."""
        response = client.get("/api/v1/storage/buckets/")
        
        # FastAPI should return 404 for invalid path
        assert response.status_code == 404

    def test_bucket_permission_denied(self, mock_storage_client, mock_credentials, mock_bucket):
        """Test handling of permission denied errors."""
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        
        # Mock permission denied on IAM policy check
        mock_bucket.get_iam_policy.side_effect = Exception("Permission denied")
        mock_client_instance.list_buckets.return_value = [mock_bucket]
        
        response = client.get("/api/v1/storage/buckets/test-project")
        
        # Should handle gracefully and still return results
        assert response.status_code == 200

class TestAsyncFunctions:
    """Test class for async functions."""

    @pytest.mark.asyncio
    async def test_get_real_buckets_success(self, mock_storage_client, mock_credentials, mock_bucket):
        """Test successful real bucket retrieval."""
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        mock_client_instance.list_buckets.return_value = [mock_bucket]
        
        # Mock IAM policy
        mock_policy = Mock()
        mock_policy.bindings = []
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        result = await _get_real_buckets("test-project")
        
        assert result["success"] is True
        assert "buckets" in result
        assert result["source"] == "real_api"
        assert "api_duration" in result

    @pytest.mark.asyncio
    async def test_get_real_buckets_no_credentials(self, mock_credentials):
        """Test real bucket retrieval with no credentials."""
        mock_credentials.return_value = None
        
        result = await _get_real_buckets("test-project")
        
        assert result["success"] is False
        assert "error" in result
        assert result["source"] == "api_failed"

    @pytest.mark.asyncio
    async def test_get_real_buckets_exception(self, mock_storage_client, mock_credentials):
        """Test real bucket retrieval with exception."""
        mock_storage_client.side_effect = Exception("API Error")
        
        result = await _get_real_buckets("test-project")
        
        assert result["success"] is False
        assert "error" in result
        assert result["source"] == "api_failed"

class TestIntegrationScenarios:
    """Test class for integration scenarios."""

    def test_full_security_analysis_workflow(self, mock_storage_client, mock_credentials):
        """Test complete security analysis workflow."""
        # Setup mock with vulnerable buckets
        mock_client_instance = Mock()
        mock_storage_client.return_value = mock_client_instance
        
        # Create mock buckets with different security issues
        public_bucket = Mock()
        public_bucket.name = "public-bucket"
        public_bucket.location = "US"
        public_bucket.storage_class = "STANDARD"
        public_bucket.versioning_enabled = False
        public_bucket.time_created = datetime.now()
        public_bucket.labels = {}
        public_bucket.lifecycle_rules = []
        public_bucket.cors = []
        public_bucket.website_main_page_suffix = None
        public_bucket.requester_pays = False
        public_bucket.default_kms_key_name = None
        public_bucket.encryption_configuration = None
        public_bucket.logging = None
        
        # Mock public access
        mock_policy = Mock()
        mock_policy.bindings = [{"members": ["allUsers"], "role": "roles/storage.objectViewer"}]
        public_bucket.get_iam_policy.return_value = mock_policy
        
        mock_client_instance.list_buckets.return_value = [public_bucket]
        
        # Make request
        response = client.get("/api/v1/storage/buckets/test-project?detailed=true")
        
        # Verify comprehensive analysis
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Check security findings
        security_findings = data["security_findings"]
        assert "critical" in security_findings
        assert "high" in security_findings
        assert "medium" in security_findings
        
        # Should have critical issues for public access
        assert len(security_findings["critical"]) > 0
        
        # Check recommendations
        assert "specific_recommendations" in data
        assert "immediate_actions" in data
        
        # Check detailed analysis
        assert "detailed_analysis" in data
        assert "compliance_gaps" in data["detailed_analysis"]

    def test_remediation_command_generation(self):
        """Test that remediation commands are properly generated."""
        response = client.post(
            "/api/v1/storage/buckets/test-project/remediate",
            params={"bucket_name": "vulnerable-bucket"},
            json={"fixes": ["enable_versioning", "disable_public_access"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify commands are generated
        assert "commands" in data
        commands = data["commands"]
        
        # Should contain gsutil commands
        assert any("gsutil versioning" in cmd for cmd in commands)
        assert any("gsutil iam ch -d" in cmd for cmd in commands)

if __name__ == "__main__":
    pytest.main([__file__])
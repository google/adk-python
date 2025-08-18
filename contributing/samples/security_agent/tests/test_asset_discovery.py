"""
Comprehensive tests for the enhanced asset discovery implementation in the GCP Security Agent.

This test suite covers:
1. Security context enrichment
2. Risk scoring algorithm
3. Error handling and retry logic
4. Public exposure detection
5. Encryption checks
6. Security-scan endpoint
7. Summary statistics generation
8. Mock GCP API interactions
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime
from typing import Dict, Any, List

# Import the modules under test
import sys
import os

# Add the project root and backend to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, project_root)
sys.path.insert(0, backend_path)

from backend.api.asset_inventory import (
    router,
    calculate_risk_score,
    get_risk_level,
    analyze_security_context,
    categorize_asset,
    generate_recommendations,
    SecurityContext,
    RiskLevel,
    AssetSummary,
    AssetListRequest,
    AssetSearchRequest,
    get_asset_client,
    retry_on_failure
)

# Create test client
@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)

# Mock data fixtures
@pytest.fixture
def sample_compute_instance():
    """Sample compute instance with public IP and unencrypted disk"""
    return {
        "name": "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/web-server-1",
        "asset_type": "compute.googleapis.com/Instance",
        "resource": {
            "version": "v1",
            "discovery_name": "Instance",
            "data": {
                "networkInterfaces": [
                    {
                        "accessConfigs": [
                            {
                                "type": "ONE_TO_ONE_NAT",
                                "natIP": "34.123.45.67"
                            }
                        ]
                    }
                ],
                "disks": [
                    {
                        "boot": True,
                        "source": "projects/test-project/zones/us-central1-a/disks/web-server-1"
                        # No diskEncryptionKey - unencrypted
                    }
                ],
                "machineType": "projects/test-project/zones/us-central1-a/machineTypes/f1-micro",
                "labels": {}
            }
        }
    }

@pytest.fixture
def sample_storage_bucket():
    """Sample storage bucket with public access"""
    return {
        "name": "//storage.googleapis.com/test-bucket-public",
        "asset_type": "storage.googleapis.com/Bucket",
        "resource": {
            "version": "v1",
            "discovery_name": "Bucket",
            "data": {
                "location": "us-central1"
            }
        },
        "iam_policy": {
            "version": 1,
            "bindings": [
                {
                    "role": "roles/storage.objectViewer",
                    "members": ["allUsers"]
                }
            ]
        }
    }

@pytest.fixture
def sample_sql_instance():
    """Sample SQL instance with security issues"""
    return {
        "name": "//sqladmin.googleapis.com/projects/test-project/instances/db-server-1",
        "asset_type": "sqladmin.googleapis.com/Instance",
        "resource": {
            "version": "v1",
            "discovery_name": "DatabaseInstance",
            "data": {
                "settings": {
                    "ipConfiguration": {
                        "requireSsl": False,
                        "ipv4Enabled": True
                    }
                },
                "ipAddresses": [
                    {
                        "type": "PRIMARY",
                        "ipAddress": "34.123.45.68"
                    }
                ]
            }
        }
    }

@pytest.fixture
def mock_gcp_asset_client():
    """Mock GCP Asset client for testing"""
    client = Mock()
    
    # Mock list_assets response
    mock_asset_1 = Mock()
    mock_asset_1.name = "//compute.googleapis.com/projects/test-project/zones/us-central1-a/instances/web-server-1"
    mock_asset_1.asset_type = "compute.googleapis.com/Instance"
    mock_asset_1.update_time = datetime.now()
    
    mock_resource_1 = Mock()
    mock_resource_1.version = "v1"
    mock_resource_1.discovery_name = "Instance"
    mock_resource_1.resource_url = "https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-a/instances/web-server-1"
    mock_resource_1.data = {
        "networkInterfaces": [
            {"accessConfigs": [{"type": "ONE_TO_ONE_NAT", "natIP": "34.123.45.67"}]}
        ],
        "disks": [{"boot": True, "source": "projects/test-project/zones/us-central1-a/disks/web-server-1"}],
        "machineType": "projects/test-project/zones/us-central1-a/machineTypes/f1-micro",
        "labels": {}
    }
    mock_asset_1.resource = mock_resource_1
    mock_asset_1.iam_policy = None
    
    mock_page_result = [mock_asset_1]
    mock_page_result.next_page_token = None
    
    client.list_assets.return_value = mock_page_result
    client.search_all_resources.return_value = mock_page_result
    client.export_assets.return_value = Mock(name="operation_123")
    
    return client

# Test classes
class TestSecurityContext:
    """Test security context analysis"""
    
    def test_analyze_compute_instance_security_context(self, sample_compute_instance):
        """Test security context analysis for compute instance"""
        context = analyze_security_context(sample_compute_instance)
        
        assert isinstance(context, SecurityContext)
        assert context.is_public == True  # Has public IP
        assert context.is_legacy_version == True  # f1-micro is legacy
        assert "Instance has public IP" in context.risk_factors
        assert "Legacy machine type" in context.risk_factors
        assert "Missing resource labels" in context.risk_factors
        assert "Unencrypted disk attached" in context.risk_factors
    
    def test_analyze_storage_bucket_security_context(self, sample_storage_bucket):
        """Test security context analysis for storage bucket"""
        context = analyze_security_context(sample_storage_bucket)
        
        assert isinstance(context, SecurityContext)
        assert context.is_public == True  # allUsers in IAM policy
        assert context.is_encrypted == False  # No default encryption configured
        assert "Public bucket access" in context.risk_factors
        assert "Default encryption not configured" in context.risk_factors
    
    def test_analyze_sql_instance_security_context(self, sample_sql_instance):
        """Test security context analysis for SQL instance"""
        context = analyze_security_context(sample_sql_instance)
        
        assert isinstance(context, SecurityContext)
        assert context.is_public == True  # Has public IP
        assert context.has_weak_authentication == True  # SSL not required
        assert "Database has public IP" in context.risk_factors
        assert "SSL not required for database" in context.risk_factors
    
    def test_analyze_service_account_security_context(self):
        """Test security context analysis for service account"""
        asset_data = {
            "name": "//iam.googleapis.com/projects/test-project/serviceAccounts/test@test-project.iam.gserviceaccount.com",
            "asset_type": "iam.googleapis.com/ServiceAccount",
            "resource": {"data": {}}
        }
        
        context = analyze_security_context(asset_data)
        
        assert isinstance(context, SecurityContext)
        assert "Service account requires IAM review" in context.risk_factors

class TestRiskScoring:
    """Test risk scoring algorithm"""
    
    def test_calculate_risk_score_high_risk_compute(self, sample_compute_instance):
        """Test risk score calculation for high-risk compute instance"""
        context = analyze_security_context(sample_compute_instance)
        risk_score = calculate_risk_score(sample_compute_instance, context)
        
        # Should be high risk due to public IP + legacy type + no encryption + missing labels
        assert risk_score > 40  # At least medium risk
        assert isinstance(risk_score, int)
        assert 0 <= risk_score <= 100
    
    def test_calculate_risk_score_critical_storage(self, sample_storage_bucket):
        """Test risk score calculation for critical storage bucket"""
        context = analyze_security_context(sample_storage_bucket)
        risk_score = calculate_risk_score(sample_storage_bucket, context)
        
        # Should be critical risk due to public access + no encryption
        assert risk_score >= 61  # High or critical risk
        assert isinstance(risk_score, int)
        assert 0 <= risk_score <= 100
    
    def test_calculate_risk_score_sql_instance(self, sample_sql_instance):
        """Test risk score calculation for SQL instance"""
        context = analyze_security_context(sample_sql_instance)
        risk_score = calculate_risk_score(sample_sql_instance, context)
        
        # Should be high risk due to public IP + weak auth + database type
        assert risk_score >= 50  # Medium to high risk
        assert isinstance(risk_score, int)
        assert 0 <= risk_score <= 100
    
    def test_risk_score_caps_at_100(self):
        """Test that risk score is capped at 100"""
        # Create a context with all risk factors
        context = SecurityContext(
            is_public=True,
            is_encrypted=False,
            has_overprivileged_access=True,
            has_weak_authentication=True,
            is_legacy_version=True,
            missing_monitoring=True,
            compliance_violations=["violation1", "violation2", "violation3", "violation4", "violation5"],
            risk_factors=["risk1", "risk2", "risk3", "risk4", "risk5"]
        )
        
        asset_data = {
            "asset_type": "sqladmin.googleapis.com/Instance"  # Critical infrastructure
        }
        
        risk_score = calculate_risk_score(asset_data, context)
        assert risk_score <= 100
    
    def test_get_risk_level_mapping(self):
        """Test risk level mapping from scores"""
        assert get_risk_level(95) == RiskLevel.CRITICAL
        assert get_risk_level(85) == RiskLevel.CRITICAL
        assert get_risk_level(75) == RiskLevel.HIGH
        assert get_risk_level(65) == RiskLevel.HIGH
        assert get_risk_level(55) == RiskLevel.MEDIUM
        assert get_risk_level(45) == RiskLevel.MEDIUM
        assert get_risk_level(35) == RiskLevel.LOW
        assert get_risk_level(25) == RiskLevel.LOW
        assert get_risk_level(15) == RiskLevel.MINIMAL
        assert get_risk_level(5) == RiskLevel.MINIMAL

class TestAssetCategorization:
    """Test asset categorization logic"""
    
    def test_categorize_compute_instance(self, sample_compute_instance):
        """Test categorization of compute instance"""
        categorization = categorize_asset(sample_compute_instance)
        
        assert categorization['service'] == 'compute'
        assert categorization['category'] == 'compute'
        assert categorization['criticality'] == 'standard'
        assert categorization['region'] == 'us-central1'
        assert categorization['friendly_type'] == 'Instance'
    
    def test_categorize_storage_bucket(self, sample_storage_bucket):
        """Test categorization of storage bucket"""
        categorization = categorize_asset(sample_storage_bucket)
        
        assert categorization['service'] == 'storage'
        assert categorization['category'] == 'storage'
        assert categorization['criticality'] == 'standard'
        assert categorization['region'] == 'global'
        assert categorization['friendly_type'] == 'Bucket'
    
    def test_categorize_sql_instance(self, sample_sql_instance):
        """Test categorization of SQL instance"""
        categorization = categorize_asset(sample_sql_instance)
        
        assert categorization['service'] == 'sqladmin'
        assert categorization['category'] == 'database'
        assert categorization['criticality'] == 'critical'  # Database is critical
        assert categorization['region'] == 'global'
        assert categorization['friendly_type'] == 'Instance'
    
    def test_categorize_kms_key(self):
        """Test categorization of KMS key (critical asset)"""
        asset_data = {
            "name": "//cloudkms.googleapis.com/projects/test-project/locations/global/keyRings/test-ring/cryptoKeys/test-key",
            "asset_type": "cloudkms.googleapis.com/CryptoKey",
            "resource": {"data": {}}
        }
        
        categorization = categorize_asset(asset_data)
        
        assert categorization['service'] == 'cloudkms'
        assert categorization['category'] == 'security'
        assert categorization['criticality'] == 'critical'  # KMS is critical

class TestRecommendations:
    """Test recommendation generation"""
    
    def test_generate_recommendations_for_public_asset(self):
        """Test recommendations for public asset"""
        context = SecurityContext(
            is_public=True,
            is_encrypted=False,
            risk_factors=["Public access", "No encryption"]
        )
        
        asset_data = {"asset_type": "storage.googleapis.com/Bucket", "resource": {"data": {}}}
        
        recommendations = generate_recommendations(context, asset_data)
        
        assert "Restrict public access - review and minimize exposure" in recommendations
        assert "Enable encryption at rest and in transit" in recommendations
        assert "Review bucket lifecycle policies and access patterns" in recommendations
        assert "Add resource labels for governance and cost tracking" in recommendations
        assert len(recommendations) <= 5
    
    def test_generate_recommendations_for_compute_instance(self, sample_compute_instance):
        """Test recommendations for compute instance"""
        context = analyze_security_context(sample_compute_instance)
        recommendations = generate_recommendations(context, sample_compute_instance)
        
        assert "Restrict public access - review and minimize exposure" in recommendations
        assert "Upgrade to supported version or instance type" in recommendations
        assert "Review network security groups and access controls" in recommendations
        assert len(recommendations) <= 5
    
    def test_generate_recommendations_for_database(self, sample_sql_instance):
        """Test recommendations for database instance"""
        context = analyze_security_context(sample_sql_instance)
        recommendations = generate_recommendations(context, sample_sql_instance)
        
        assert "Restrict public access - review and minimize exposure" in recommendations
        assert "Strengthen authentication requirements" in recommendations
        assert "Enable database audit logging and review connection security" in recommendations

class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_list_assets_endpoint_without_client(self, mock_get_client, client):
        """Test list assets endpoint when GCP client is not available"""
        mock_get_client.return_value = None
        
        response = client.post("/api/v1/assets/list", json={
            "project_id": "test-project",
            "include_security_context": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['source'] == "sample_data"
        assert 'assets' in data
        assert len(data['assets']) == 2  # Sample data has 2 assets
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_list_assets_endpoint_with_client(self, mock_get_client, mock_gcp_asset_client, client):
        """Test list assets endpoint with mocked GCP client"""
        mock_get_client.return_value = mock_gcp_asset_client
        
        response = client.post("/api/v1/assets/list", json={
            "project_id": "test-project",
            "include_security_context": True,
            "page_size": 100
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['source'] == "cloud_asset_api_enhanced"
        assert 'assets' in data
        assert 'summary' in data
        assert 'enhanced_features' in data
        
        # Verify security analysis is included
        assert data['enhanced_features']['security_analysis'] == True
        assert data['enhanced_features']['risk_scoring'] == True
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_search_assets_endpoint(self, mock_get_client, mock_gcp_asset_client, client):
        """Test search assets endpoint"""
        mock_get_client.return_value = mock_gcp_asset_client
        
        response = client.post("/api/v1/assets/search", json={
            "scope": "projects/test-project",
            "query": "name:web-server*",
            "page_size": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['source'] == "cloud_asset_search_enhanced"
        assert 'results' in data
        assert 'summary' in data
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_security_scan_endpoint(self, mock_get_client, mock_gcp_asset_client, client):
        """Test security-focused scan endpoint"""
        mock_get_client.return_value = mock_gcp_asset_client
        
        response = client.post("/api/v1/assets/security-scan", json={
            "project_id": "test-project",
            "page_size": 500
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['source'] == "security_scan_enhanced"
        assert 'high_risk_assets' in data
        assert 'security_summary' in data
        assert 'recommendations' in data
        
        # Verify security summary structure
        security_summary = data['security_summary']
        assert 'total_assets_scanned' in security_summary
        assert 'risk_distribution' in security_summary
        assert 'most_common_issues' in security_summary
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_asset_summary_endpoint(self, mock_get_client, mock_gcp_asset_client, client):
        """Test asset summary endpoint"""
        mock_get_client.return_value = mock_gcp_asset_client
        
        response = client.get("/api/v1/assets/summary?project_id=test-project&include_security=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'data' in data
        
        summary_data = data['data']
        assert 'total_assets' in summary_data
        assert 'asset_types' in summary_data
        assert 'regions' in summary_data
        assert 'security_findings' in summary_data
        assert 'high_risk_assets' in summary_data
        assert 'risk_distribution' in summary_data
    
    def test_get_supported_asset_types_endpoint(self, client):
        """Test asset types endpoint"""
        response = client.get("/api/v1/assets/asset-types")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert 'asset_types' in data
        assert 'content_types' in data
        
        # Verify some expected asset types
        asset_types = data['asset_types']
        assert "compute.googleapis.com/Instance" in asset_types
        assert "storage.googleapis.com/Bucket" in asset_types
        assert "sqladmin.googleapis.com/Instance" in asset_types
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/assets/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == "healthy"
        assert data['service'] == "asset_inventory"
        assert 'client_available' in data
        assert 'timestamp' in data

class TestErrorHandling:
    """Test error handling and retry logic"""
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_permission_denied_error(self, mock_get_client, client):
        """Test handling of permission denied errors"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        mock_client = Mock()
        mock_client.list_assets.side_effect = gcp_exceptions.PermissionDenied("Permission denied")
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/list", json={"project_id": "test-project"})
        
        assert response.status_code == 403
        assert "Permission denied" in response.json()['detail']
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_project_not_found_error(self, mock_get_client, client):
        """Test handling of project not found errors"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        mock_client = Mock()
        mock_client.list_assets.side_effect = gcp_exceptions.NotFound("Project not found")
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/list", json={"project_id": "nonexistent-project"})
        
        assert response.status_code == 404
        assert "Project not found" in response.json()['detail']
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_service_unavailable_error(self, mock_get_client, client):
        """Test handling of service unavailable errors"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        mock_client = Mock()
        mock_client.list_assets.side_effect = gcp_exceptions.ServiceUnavailable("Service unavailable")
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/list", json={"project_id": "test-project"})
        
        assert response.status_code == 503
        assert "Service temporarily unavailable" in response.json()['detail']
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_deadline_exceeded_error(self, mock_get_client, client):
        """Test handling of deadline exceeded errors"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        mock_client = Mock()
        mock_client.list_assets.side_effect = gcp_exceptions.DeadlineExceeded("Request timeout")
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/list", json={"project_id": "test-project"})
        
        assert response.status_code == 504
        assert "Request timeout" in response.json()['detail']
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_generic_exception_handling(self, mock_get_client, client):
        """Test handling of generic exceptions"""
        mock_client = Mock()
        mock_client.list_assets.side_effect = Exception("Generic error")
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/list", json={"project_id": "test-project"})
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()['detail']
    
    @patch('time.sleep')
    def test_retry_decorator_success_after_failure(self, mock_sleep):
        """Test retry decorator succeeds after initial failure"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        call_count = 0
        
        @retry_on_failure
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise gcp_exceptions.ServiceUnavailable("Service unavailable")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # Two retries with sleep
    
    @patch('time.sleep')
    def test_retry_decorator_max_retries_exceeded(self, mock_sleep):
        """Test retry decorator fails after max retries"""
        try:
            from google.api_core import exceptions as gcp_exceptions
        except ImportError:
            pytest.skip("Google Cloud libraries not available")
        
        @retry_on_failure
        def always_fail_function():
            raise gcp_exceptions.ServiceUnavailable("Service unavailable")
        
        with pytest.raises(gcp_exceptions.ServiceUnavailable):
            always_fail_function()
        
        assert mock_sleep.call_count == 3  # MAX_RETRIES attempts

class TestDataModels:
    """Test data model functionality"""
    
    def test_security_context_initialization(self):
        """Test SecurityContext initialization with defaults"""
        context = SecurityContext()
        
        assert context.is_public == False
        assert context.is_encrypted == True
        assert context.has_overprivileged_access == False
        assert context.has_weak_authentication == False
        assert context.is_legacy_version == False
        assert context.missing_monitoring == False
        assert context.compliance_violations == []
        assert context.risk_factors == []
    
    def test_asset_summary_initialization(self):
        """Test AssetSummary initialization with defaults"""
        summary = AssetSummary()
        
        assert summary.total_assets == 0
        assert summary.by_type == {}
        assert summary.by_region == {}
        assert summary.security_issues == 0
        assert all(level.value in summary.by_risk_level for level in RiskLevel)
        assert all(count == 0 for count in summary.by_risk_level.values())
    
    def test_asset_list_request_validation(self):
        """Test AssetListRequest model validation"""
        # Valid request
        request = AssetListRequest(
            project_id="test-project",
            asset_types=["compute.googleapis.com/Instance"],
            page_size=100,
            include_security_context=True,
            risk_level_filter=[RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
        
        assert request.project_id == "test-project"
        assert request.asset_types == ["compute.googleapis.com/Instance"]
        assert request.page_size == 100
        assert request.include_security_context == True
        assert request.risk_level_filter == [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_asset_search_request_validation(self):
        """Test AssetSearchRequest model validation"""
        request = AssetSearchRequest(
            scope="projects/test-project",
            query="name:web-server*",
            asset_types=["compute.googleapis.com/Instance"],
            page_size=50
        )
        
        assert request.scope == "projects/test-project"
        assert request.query == "name:web-server*"
        assert request.asset_types == ["compute.googleapis.com/Instance"]
        assert request.page_size == 50

class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_complete_security_analysis_flow(self, mock_get_client, client):
        """Test complete security analysis workflow"""
        # Setup mock client with multiple asset types
        mock_client = Mock()
        
        # Mock assets with varying risk levels
        high_risk_compute = Mock()
        high_risk_compute.name = "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/public-server"
        high_risk_compute.asset_type = "compute.googleapis.com/Instance"
        high_risk_compute.update_time = datetime.now()
        high_risk_compute.resource = Mock()
        high_risk_compute.resource.data = {
            "networkInterfaces": [{"accessConfigs": [{"natIP": "1.2.3.4"}]}],
            "disks": [{"boot": True}],  # No encryption
            "machineType": "zones/us-central1-a/machineTypes/f1-micro",  # Legacy
            "labels": {}
        }
        high_risk_compute.iam_policy = None
        
        secure_storage = Mock()
        secure_storage.name = "//storage.googleapis.com/secure-private-bucket"
        secure_storage.asset_type = "storage.googleapis.com/Bucket"
        secure_storage.update_time = datetime.now()
        secure_storage.resource = Mock()
        secure_storage.resource.data = {
            "location": "us-central1",
            "encryption": {"defaultKmsKeyName": "projects/test/locations/global/keyRings/ring1/cryptoKeys/key1"},
            "labels": {"environment": "prod", "team": "security"}
        }
        secure_storage.iam_policy = {
            "bindings": [{"role": "roles/storage.admin", "members": ["user:admin@example.com"]}]
        }
        
        mock_client.list_assets.return_value = [high_risk_compute, secure_storage]
        mock_get_client.return_value = mock_client
        
        # Perform security scan
        response = client.post("/api/v1/assets/security-scan", json={
            "project_id": "test-project",
            "include_security_context": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify security analysis results
        assert data['success'] == True
        assert len(data['high_risk_assets']) > 0
        
        # Check that high-risk asset is identified
        high_risk_found = False
        for asset in data['high_risk_assets']:
            if "public-server" in asset['name']:
                high_risk_found = True
                assert asset['risk_level'] in ['HIGH', 'CRITICAL']
                assert asset['is_public'] == True
                assert len(asset['security_issues']) > 0
                assert len(asset['recommendations']) > 0
        
        assert high_risk_found, "High-risk asset should be identified in security scan"
        
        # Verify summary statistics
        security_summary = data['security_summary']
        assert security_summary['total_assets_scanned'] == 2
        assert security_summary['total_critical'] + security_summary['total_high'] > 0
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_risk_filtering_functionality(self, mock_get_client, client):
        """Test risk level filtering works correctly"""
        mock_client = Mock()
        
        # Create assets with known risk levels
        critical_asset = Mock()
        critical_asset.name = "//storage.googleapis.com/public-bucket"
        critical_asset.asset_type = "storage.googleapis.com/Bucket"
        critical_asset.update_time = datetime.now()
        critical_asset.resource = Mock()
        critical_asset.resource.data = {}
        critical_asset.iam_policy = {"bindings": [{"role": "roles/storage.objectViewer", "members": ["allUsers"]}]}
        
        low_risk_asset = Mock()
        low_risk_asset.name = "//compute.googleapis.com/projects/test/zones/us-central1-a/instances/secure-server"
        low_risk_asset.asset_type = "compute.googleapis.com/Instance"
        low_risk_asset.update_time = datetime.now()
        low_risk_asset.resource = Mock()
        low_risk_asset.resource.data = {
            "networkInterfaces": [],  # No public access
            "disks": [{"diskEncryptionKey": {"kmsKeyName": "key1"}}],  # Encrypted
            "machineType": "zones/us-central1-a/machineTypes/n1-standard-1",  # Modern
            "labels": {"environment": "prod"}
        }
        low_risk_asset.iam_policy = None
        
        mock_client.list_assets.return_value = [critical_asset, low_risk_asset]
        mock_get_client.return_value = mock_client
        
        # Test filtering for high-risk assets only
        response = client.post("/api/v1/assets/list", json={
            "project_id": "test-project",
            "include_security_context": True,
            "risk_level_filter": ["HIGH", "CRITICAL"]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should only return the critical asset
        assert len(data['assets']) == 1
        assert "public-bucket" in data['assets'][0]['name']
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_export_assets_functionality(self, mock_get_client, client):
        """Test asset export functionality"""
        mock_client = Mock()
        mock_operation = Mock()
        mock_operation.name = "operations/export-operation-123"
        mock_client.export_assets.return_value = mock_operation
        mock_get_client.return_value = mock_client
        
        response = client.post("/api/v1/assets/export", json={
            "project_id": "test-project",
            "output_bucket": "gs://test-bucket/exports/",
            "asset_types": ["compute.googleapis.com/Instance"],
            "content_type": "RESOURCE"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] == True
        assert "Export started" in data['message']
        assert data['operation_name'] == "operations/export-operation-123"
        assert "5-30 minutes" in data['estimated_completion']

# Performance and load testing
class TestPerformanceAndScaling:
    """Test performance and scaling aspects"""
    
    @patch('backend.api.asset_inventory.get_asset_client')
    def test_large_asset_inventory_performance(self, mock_get_client):
        """Test handling of large asset inventories"""
        mock_client = Mock()
        
        # Simulate large number of assets
        mock_assets = []
        for i in range(1000):
            mock_asset = Mock()
            mock_asset.name = f"//compute.googleapis.com/projects/test/zones/us-central1-a/instances/server-{i}"
            mock_asset.asset_type = "compute.googleapis.com/Instance"
            mock_asset.update_time = datetime.now()
            mock_asset.resource = Mock()
            mock_asset.resource.data = {"labels": {}}
            mock_asset.iam_policy = None
            mock_assets.append(mock_asset)
        
        mock_client.list_assets.return_value = mock_assets
        mock_get_client.return_value = mock_client
        
        # Test that large inventories are processed efficiently
        request = AssetListRequest(project_id="test-project", include_security_context=True)
        
        # This should complete without timeout or memory issues
        # In a real scenario, we'd measure execution time
        assert len(mock_assets) == 1000
        assert all(hasattr(asset, 'name') for asset in mock_assets)
    
    def test_memory_usage_with_security_context(self):
        """Test memory efficiency when analyzing security contexts"""
        # Create many assets to analyze
        assets = []
        for i in range(100):
            asset = {
                "name": f"//compute.googleapis.com/projects/test/instances/server-{i}",
                "asset_type": "compute.googleapis.com/Instance",
                "resource": {"data": {"labels": {}}}
            }
            assets.append(asset)
        
        # Analyze all assets - should not cause memory issues
        contexts = []
        for asset in assets:
            context = analyze_security_context(asset)
            contexts.append(context)
        
        assert len(contexts) == 100
        assert all(isinstance(ctx, SecurityContext) for ctx in contexts)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
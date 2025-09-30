#!/usr/bin/env python3
"""
Shared test fixtures for Cloud Functions testing.
Provides mock GCP service clients and common test data.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock environment variables
os.environ['PROJECT_ID'] = 'test-project-123'
os.environ['BQ_DATASET_ID'] = 'test_security_insights'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/test/credentials.json'


# ==================== BigQuery Fixtures ====================

@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client for testing."""
    with patch('google.cloud.bigquery.Client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock dataset operations
        mock_dataset = MagicMock()
        mock_instance.dataset.return_value = mock_dataset
        mock_instance.get_dataset.return_value = mock_dataset
        
        # Mock table operations
        mock_table = MagicMock()
        mock_instance.create_table.return_value = mock_table
        
        # Mock insert operations
        mock_instance.insert_rows_json.return_value = []  # No errors
        
        yield mock_instance


@pytest.fixture
def mock_bigquery_with_errors():
    """Mock BigQuery client that returns errors."""
    with patch('google.cloud.bigquery.Client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock insert with errors
        mock_instance.insert_rows_json.return_value = [
            {"errors": [{"message": "Invalid row"}]}
        ]
        
        yield mock_instance


# ==================== IAM Fixtures ====================

@pytest.fixture
def mock_iam_client():
    """Mock IAM client for testing."""
    with patch('google.cloud.iam_admin_v1.IAMClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock custom roles
        mock_custom_role = MagicMock()
        mock_custom_role.name = "projects/test-project-123/roles/customRole1"
        mock_custom_role.title = "Custom Role 1"
        mock_custom_role.description = "Test custom role"
        mock_custom_role.included_permissions = [
            "storage.buckets.create",
            "storage.buckets.delete",
            "iam.roles.update"
        ]
        mock_custom_role.stage = MagicMock(name="GA")
        
        # Mock list roles
        mock_instance.list_roles.return_value = [mock_custom_role]
        
        # Mock standard roles
        mock_standard_role = MagicMock()
        mock_standard_role.name = "roles/storage.admin"
        mock_standard_role.title = "Storage Admin"
        mock_standard_role.description = "Full control of GCS resources"
        mock_standard_role.included_permissions = [
            "storage.buckets.create",
            "storage.buckets.delete",
            "storage.buckets.get",
            "storage.buckets.list"
        ]
        mock_standard_role.stage = MagicMock(name="GA")
        
        mock_instance.query_grantable_roles.return_value = [mock_standard_role]
        
        yield mock_instance


@pytest.fixture
def mock_crm_client():
    """Mock Cloud Resource Manager client for IAM bindings."""
    with patch('google.cloud.resourcemanager_v3.ProjectsClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock IAM policy
        mock_policy = MagicMock()
        mock_binding = MagicMock()
        mock_binding.role = "roles/owner"
        mock_binding.members = [
            "user:admin@example.com",
            "serviceAccount:test-sa@test-project-123.iam.gserviceaccount.com"
        ]
        mock_policy.bindings = [mock_binding]
        
        mock_instance.get_iam_policy.return_value = mock_policy
        
        yield mock_instance


@pytest.fixture  
def mock_iam_service_client():
    """Mock IAM Service Account client."""
    with patch('google.cloud.iam_admin_v1.IAMClient') as mock_iam:
        mock_instance = MagicMock()
        mock_iam.return_value = mock_instance
        
        # Mock service accounts
        mock_sa = MagicMock()
        mock_sa.name = "projects/test-project-123/serviceAccounts/test-sa@test-project-123.iam.gserviceaccount.com"
        mock_sa.email = "test-sa@test-project-123.iam.gserviceaccount.com"
        mock_sa.display_name = "Test Service Account"
        mock_sa.disabled = False
        
        mock_instance.list_service_accounts.return_value = [mock_sa]
        
        # Mock keys
        mock_key = MagicMock()
        mock_key.name = "projects/test-project-123/serviceAccounts/test-sa@test-project-123.iam.gserviceaccount.com/keys/key123"
        mock_key.key_type = "USER_MANAGED"
        mock_key.valid_after_time = datetime.utcnow() - timedelta(days=30)
        mock_key.valid_before_time = datetime.utcnow() + timedelta(days=30)
        
        mock_instance.list_service_account_keys.return_value = [mock_key]
        
        yield mock_instance


# ==================== Compute Fixtures ====================

@pytest.fixture
def mock_compute_client():
    """Mock Compute Engine client."""
    with patch('google.cloud.compute_v1.InstancesClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock compute instance
        mock_vm = MagicMock()
        mock_vm.name = "test-instance-1"
        mock_vm.status = "RUNNING"
        mock_vm.machine_type = "zones/us-central1-a/machineTypes/n1-standard-1"
        mock_vm.creation_timestamp = "2024-01-01T00:00:00Z"
        
        # Mock network interfaces
        mock_network = MagicMock()
        mock_network.network = "global/networks/default"
        mock_network.access_configs = [MagicMock(nat_i_p="34.123.45.67")]
        mock_vm.network_interfaces = [mock_network]
        
        # Mock disks
        mock_disk = MagicMock()
        mock_disk.source = "zones/us-central1-a/disks/test-disk"
        mock_disk.auto_delete = True
        mock_vm.disks = [mock_disk]
        
        # Mock service accounts
        mock_sa = MagicMock()
        mock_sa.email = "default@test-project-123.iam.gserviceaccount.com"
        mock_sa.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        mock_vm.service_accounts = [mock_sa]
        
        # Mock metadata
        mock_metadata = MagicMock()
        mock_metadata.items = []
        mock_vm.metadata = mock_metadata
        
        # Mock tags
        mock_tags = MagicMock()
        mock_tags.items = ["http-server", "https-server"]
        mock_vm.tags = mock_tags
        
        # Mock aggregated list response
        mock_response = MagicMock()
        mock_response.items = {
            "zones/us-central1-a": MagicMock(instances=[mock_vm])
        }
        
        mock_instance.aggregated_list.return_value = mock_response
        
        yield mock_instance


@pytest.fixture
def mock_firewall_client():
    """Mock Firewall client."""
    with patch('google.cloud.compute_v1.FirewallsClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock firewall rule
        mock_rule = MagicMock()
        mock_rule.name = "allow-http"
        mock_rule.direction = "INGRESS"
        mock_rule.priority = 1000
        mock_rule.source_ranges = ["0.0.0.0/0"]
        mock_rule.target_tags = ["http-server"]
        
        # Mock allowed ports
        mock_allowed = MagicMock()
        mock_allowed.i_p_protocol = "tcp"
        mock_allowed.ports = ["80"]
        mock_rule.allowed = [mock_allowed]
        
        mock_rule.network = "global/networks/default"
        mock_rule.creation_timestamp = "2024-01-01T00:00:00Z"
        
        mock_instance.list.return_value = [mock_rule]
        
        yield mock_instance


# ==================== Storage Fixtures ====================

@pytest.fixture
def mock_storage_client():
    """Mock Storage client."""
    with patch('google.cloud.storage.Client') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock bucket
        mock_bucket = MagicMock()
        mock_bucket.name = "test-bucket-123"
        mock_bucket.location = "US"
        mock_bucket.storage_class = "STANDARD"
        mock_bucket.time_created = datetime.utcnow() - timedelta(days=30)
        
        # Mock IAM policy
        mock_policy = MagicMock()
        mock_policy.bindings = [
            {"role": "roles/storage.objectViewer", "members": ["allUsers"]}
        ]
        mock_bucket.get_iam_policy.return_value = mock_policy
        
        # Mock lifecycle rules
        mock_bucket.lifecycle_rules = [
            {"action": {"type": "Delete"}, "condition": {"age": 30}}
        ]
        
        # Mock encryption
        mock_bucket.default_kms_key_name = None
        
        # Mock versioning
        mock_bucket.versioning_enabled = False
        
        # Mock logging
        mock_bucket.logging = None
        
        # Mock uniform bucket level access
        mock_bucket.iam_configuration = MagicMock(
            uniform_bucket_level_access_enabled=False
        )
        
        mock_instance.list_buckets.return_value = [mock_bucket]
        
        yield mock_instance


# ==================== Security Command Center Fixtures ====================

@pytest.fixture
def mock_scc_client():
    """Mock Security Command Center client."""
    with patch('google.cloud.securitycenter.SecurityCenterClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock security finding
        mock_finding = MagicMock()
        mock_finding.name = "organizations/123/sources/456/findings/finding-1"
        mock_finding.category = "PUBLIC_BUCKET"
        mock_finding.severity = "HIGH"
        mock_finding.state = "ACTIVE"
        mock_finding.resource_name = "//storage.googleapis.com/test-bucket-123"
        mock_finding.finding_class = "VULNERABILITY"
        mock_finding.event_time = datetime.utcnow()
        
        mock_instance.list_findings.return_value = MagicMock(
            finding_result_list=[MagicMock(finding=mock_finding)]
        )
        
        yield mock_instance


# ==================== Test Data Fixtures ====================

@pytest.fixture
def sample_iam_policy():
    """Sample IAM policy for testing."""
    return {
        "bindings": [
            {
                "role": "roles/owner",
                "members": [
                    "user:admin@example.com",
                    "user:owner@example.com"
                ]
            },
            {
                "role": "roles/editor",
                "members": [
                    "user:developer@example.com",
                    "serviceAccount:app-sa@test-project-123.iam.gserviceaccount.com"
                ]
            },
            {
                "role": "roles/viewer",
                "members": [
                    "user:viewer@external.com",
                    "group:viewers@example.com"
                ]
            }
        ]
    }


@pytest.fixture
def sample_custom_role():
    """Sample custom role for testing."""
    return {
        "name": "projects/test-project-123/roles/customDataAnalyst",
        "title": "Custom Data Analyst",
        "description": "Custom role for data analysis",
        "included_permissions": [
            "bigquery.datasets.create",
            "bigquery.datasets.get",
            "bigquery.tables.create",
            "bigquery.tables.get",
            "bigquery.tables.list",
            "bigquery.tables.delete",
            "storage.buckets.get",
            "storage.objects.get",
            "storage.objects.list"
        ],
        "stage": "GA"
    }


@pytest.fixture
def sample_compute_instance():
    """Sample compute instance for testing."""
    return {
        "name": "web-server-1",
        "zone": "us-central1-a",
        "machine_type": "n1-standard-2",
        "status": "RUNNING",
        "network_interfaces": [
            {
                "network": "default",
                "subnetwork": "default",
                "external_ip": "34.123.45.67",
                "internal_ip": "10.128.0.2"
            }
        ],
        "disks": [
            {
                "source": "boot-disk",
                "type": "PERSISTENT",
                "size_gb": 100,
                "auto_delete": True
            }
        ],
        "service_accounts": [
            {
                "email": "default@test-project-123.iam.gserviceaccount.com",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
            }
        ],
        "tags": ["http-server", "https-server"],
        "metadata": {
            "enable-oslogin": "TRUE",
            "startup-script": "#!/bin/bash\napt-get update"
        }
    }


@pytest.fixture
def sample_firewall_rule():
    """Sample firewall rule for testing."""
    return {
        "name": "allow-ssh",
        "direction": "INGRESS",
        "priority": 1000,
        "source_ranges": ["0.0.0.0/0"],
        "allowed": [
            {"IPProtocol": "tcp", "ports": ["22"]}
        ],
        "target_tags": ["ssh-server"],
        "network": "default",
        "description": "Allow SSH from anywhere"
    }


@pytest.fixture
def sample_storage_bucket():
    """Sample storage bucket for testing."""
    return {
        "name": "my-public-bucket",
        "location": "US",
        "storage_class": "STANDARD",
        "public_access": True,
        "versioning": False,
        "encryption": None,
        "lifecycle_rules": [],
        "iam_policy": {
            "bindings": [
                {
                    "role": "roles/storage.objectViewer",
                    "members": ["allUsers"]
                }
            ]
        },
        "uniform_bucket_level_access": False,
        "logging": None
    }


@pytest.fixture
def sample_security_finding():
    """Sample security finding for testing."""
    return {
        "name": "finding-123",
        "category": "PUBLIC_BUCKET",
        "severity": "HIGH",
        "state": "ACTIVE",
        "resource": "//storage.googleapis.com/my-public-bucket",
        "finding_class": "VULNERABILITY",
        "description": "Storage bucket is publicly accessible",
        "recommendation": "Remove allUsers from bucket IAM policy",
        "event_time": datetime.utcnow().isoformat()
    }


# ==================== HTTP Request/Response Fixtures ====================

@pytest.fixture
def mock_http_request():
    """Mock HTTP request object for Cloud Functions."""
    class MockRequest:
        def __init__(self, json_data=None, args=None, method="POST"):
            self.json = json_data or {}
            self.args = args or {}
            self.method = method
            self.headers = {"Content-Type": "application/json"}
        
        def get_json(self):
            return self.json
    
    return MockRequest


@pytest.fixture
def mock_functions_framework():
    """Mock functions_framework decorator."""
    def decorator(func):
        return func
    
    with patch('functions_framework.http', decorator):
        yield


# ==================== Environment Fixtures ====================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Automatically set up test environment variables."""
    monkeypatch.setenv("PROJECT_ID", "test-project-123")
    monkeypatch.setenv("BQ_DATASET_ID", "test_security_insights")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/test/creds.json")
    monkeypatch.setenv("TESTING", "true")


@pytest.fixture
def cleanup_bigquery():
    """Fixture to clean up BigQuery resources after tests."""
    yield
    # Cleanup code would go here if needed
    pass


# ==================== Performance Testing Fixtures ====================

@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    def generate(size=1000):
        return [
            {
                "id": i,
                "name": f"item-{i}",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"value": i * 100}
            }
            for i in range(size)
        ]
    return generate
"""
Core ADK Test Configuration and Fixtures

This module provides shared test configuration, fixtures, and utilities
for testing the ADK core framework functionality.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path

# Test configuration
TEST_PROJECT_ID = "adk-test-project"
TEST_REGION = "us-central1"
TEST_ZONE = "us-central1-a"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration dictionary."""
    return {
        "project_id": TEST_PROJECT_ID,
        "region": TEST_REGION,
        "zone": TEST_ZONE,
        "test_mode": True,
        "mock_apis": True,
        "cache_enabled": False,
    }


@pytest.fixture
def mock_gcp_credentials():
    """Mock Google Cloud credentials for testing."""
    mock_creds = Mock()
    mock_creds.expired = False
    mock_creds.valid = True
    mock_creds.token = "mock-token-12345"
    return mock_creds


@pytest.fixture
def mock_asset_client():
    """Mock Google Cloud Asset Inventory client."""
    client = MagicMock()
    client.search_all_resources.return_value = []
    client.list_assets.return_value = []
    return client


@pytest.fixture
def mock_compute_client():
    """Mock Google Cloud Compute client."""
    client = MagicMock()
    client.instances.return_value.list.return_value.execute.return_value = {
        "items": []
    }
    return client


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    client = MagicMock()
    client.list_buckets.return_value = []
    return client


@pytest.fixture
def sample_compute_instance():
    """Sample compute instance data for testing."""
    return {
        "name": "test-instance-1",
        "zone": f"projects/{TEST_PROJECT_ID}/zones/{TEST_ZONE}",
        "machineType": f"projects/{TEST_PROJECT_ID}/zones/{TEST_ZONE}/machineTypes/e2-medium",
        "status": "RUNNING",
        "networkInterfaces": [
            {
                "network": f"projects/{TEST_PROJECT_ID}/global/networks/default",
                "accessConfigs": [
                    {
                        "type": "ONE_TO_ONE_NAT",
                        "name": "External NAT",
                        "natIP": "1.2.3.4"
                    }
                ]
            }
        ],
        "disks": [
            {
                "boot": True,
                "deviceName": "persistent-disk-0",
                "source": f"projects/{TEST_PROJECT_ID}/zones/{TEST_ZONE}/disks/test-disk"
            }
        ],
        "serviceAccounts": [
            {
                "email": f"test-sa@{TEST_PROJECT_ID}.iam.gserviceaccount.com",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
            }
        ]
    }


@pytest.fixture
def sample_storage_bucket():
    """Sample storage bucket data for testing."""
    return {
        "name": "test-bucket-12345",
        "location": "US",
        "storageClass": "STANDARD",
        "versioning": {"enabled": False},
        "encryption": {
            "defaultKmsKeyName": None
        },
        "iamConfiguration": {
            "uniformBucketLevelAccess": {
                "enabled": False
            },
            "publicAccessPrevention": "inherited"
        },
        "lifecycle": {"rule": []},
        "cors": [],
        "website": {}
    }


@pytest.fixture
def sample_iam_policy():
    """Sample IAM policy data for testing."""
    return {
        "version": 1,
        "etag": "test-etag",
        "bindings": [
            {
                "role": "roles/owner",
                "members": [f"user:admin@{TEST_PROJECT_ID}.example.com"]
            },
            {
                "role": "roles/viewer", 
                "members": [
                    f"serviceAccount:test-sa@{TEST_PROJECT_ID}.iam.gserviceaccount.com"
                ]
            }
        ]
    }


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock environment variables for testing."""
    test_env = {
        "GOOGLE_CLOUD_PROJECT": TEST_PROJECT_ID,
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/test/credentials.json",
        "ADK_TEST_MODE": "true",
        "ADK_LOG_LEVEL": "DEBUG"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture
def mock_adk_agent():
    """Mock ADK agent for testing."""
    from unittest.mock import MagicMock
    
    agent = MagicMock()
    agent.name = "test-agent"
    agent.project_id = TEST_PROJECT_ID
    agent.capabilities = ["asset_discovery", "security_analysis"]
    agent.status = "active"
    
    # Mock async methods
    agent.initialize = MagicMock(return_value=None)
    agent.process_query = MagicMock(return_value={"response": "test response"})
    agent.cleanup = MagicMock(return_value=None)
    
    return agent


@pytest.fixture
def sample_security_finding():
    """Sample security finding for testing."""
    return {
        "finding_id": "test-finding-001",
        "resource_name": "test-bucket-12345",
        "resource_type": "storage.googleapis.com/Bucket",
        "category": "public_access",
        "severity": "HIGH",
        "state": "ACTIVE",
        "description": "Storage bucket allows public read access",
        "recommendation": "Remove public access and use IAM policies for access control",
        "remediation_steps": [
            "Remove allUsers and allAuthenticatedUsers from bucket IAM policy",
            "Add specific users or service accounts as needed",
            "Enable uniform bucket-level access",
            "Review and update bucket ACLs"
        ],
        "compliance_violations": ["CIS-3.3", "SOC2-CC6.1"],
        "risk_score": 8.5,
        "created_time": "2025-01-15T10:30:00Z"
    }


class MockGCPService:
    """Base class for mocking GCP services."""
    
    def __init__(self, project_id: str = TEST_PROJECT_ID):
        self.project_id = project_id
        self._client = MagicMock()
    
    @property
    def client(self):
        return self._client


@pytest.fixture
def mock_vertex_ai_client():
    """Mock Vertex AI client for testing."""
    client = MagicMock()
    client.generate_content.return_value.text = "Mock AI response for testing"
    client.predict.return_value.predictions = [{"content": "Mock prediction"}]
    return client


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "gcp: marks tests that require GCP access"
    )
    config.addinivalue_line(
        "markers", "security: marks tests related to security"
    )


# Test utilities
def assert_valid_gcp_resource_name(resource_name: str, resource_type: str):
    """Assert that a GCP resource name follows expected format."""
    if resource_type == "compute_instance":
        assert resource_name.startswith("projects/")
        assert "/zones/" in resource_name
        assert "/instances/" in resource_name
    elif resource_type == "storage_bucket":
        assert resource_name.startswith("projects/_/buckets/")
    elif resource_type == "iam_policy":
        assert resource_name.startswith("//cloudresourcemanager.googleapis.com/projects/")


def create_test_response(data: Any, status: str = "success") -> Dict[str, Any]:
    """Create standardized test response format."""
    return {
        "status": status,
        "data": data,
        "timestamp": "2025-01-15T10:30:00Z",
        "source": "test"
    }


async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true within timeout."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False
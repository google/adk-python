"""
Pytest configuration and fixtures for unified Cloud Functions tests
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ['PROJECT_ID'] = 'test-project'
os.environ['BQ_DATASET_ID'] = 'test_dataset'
os.environ['BQ_LOCATION'] = 'us-central1'
os.environ['ENABLE_SAMPLE_DATA'] = 'true'


@pytest.fixture
def mock_bq_client():
    """Mock BigQuery client"""
    with patch('google.cloud.bigquery.Client') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock dataset methods
        client_instance.dataset.return_value = MagicMock()
        client_instance.get_dataset.return_value = MagicMock()
        client_instance.create_dataset.return_value = MagicMock()

        # Mock table methods
        client_instance.get_table.return_value = MagicMock()
        client_instance.create_table.return_value = MagicMock()
        client_instance.insert_rows_json.return_value = []

        yield client_instance


@pytest.fixture
def mock_iam_client():
    """Mock IAM client"""
    with patch('google.cloud.iam_admin_v1.IAMClient') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock list_roles response
        mock_role = MagicMock()
        mock_role.name = "projects/test-project/roles/CustomRole"
        mock_role.title = "Custom Role"
        mock_role.description = "Test custom role"
        mock_role.stage = MagicMock(name="GA")
        mock_role.deleted = False
        mock_role.included_permissions = ["compute.instances.get", "compute.instances.list"]

        client_instance.list_roles.return_value = [mock_role]

        yield client_instance


@pytest.fixture
def mock_scc_client():
    """Mock Security Command Center client"""
    with patch('google.cloud.securitycenter_v2.SecurityCenterClient') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Mock list_findings response
        mock_finding_result = MagicMock()
        mock_finding = MagicMock()
        mock_finding.name = "organizations/test/sources/test/findings/finding-001"
        mock_finding.parent = "organizations/test/sources/test"
        mock_finding.resource_name = "//compute.googleapis.com/projects/test/instances/test-instance"
        mock_finding.state = MagicMock(name="ACTIVE")
        mock_finding.category = "PUBLIC_IP_ADDRESS"
        mock_finding.severity = MagicMock(name="HIGH")
        mock_finding.finding_class = MagicMock(name="VULNERABILITY")

        mock_finding_result.finding = mock_finding
        client_instance.list_findings.return_value = [mock_finding_result]

        yield client_instance


@pytest.fixture
def test_client():
    """Create test client for FastAPI app"""
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_fetch_request():
    """Sample fetch request data"""
    return {
        "fetcher": "security_findings",
        "async_mode": False,
        "force_refresh": False
    }
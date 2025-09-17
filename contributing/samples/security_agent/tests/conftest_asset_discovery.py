"""
Configuration for asset discovery tests.

This module provides common test fixtures and configuration for the asset discovery tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Mock Google Cloud libraries if not available
def mock_google_cloud_imports():
    """Mock Google Cloud imports if they're not available"""
    
    # Create mock modules
    mock_asset_v1 = Mock()
    mock_exceptions = Mock()
    
    # Mock the AssetServiceClient
    mock_client = Mock()
    mock_asset_v1.AssetServiceClient.return_value = mock_client
    
    # Mock common exception classes
    class MockException(Exception):
        pass
    
    mock_exceptions.PermissionDenied = type('PermissionDenied', (MockException,), {})
    mock_exceptions.NotFound = type('NotFound', (MockException,), {})
    mock_exceptions.ServiceUnavailable = type('ServiceUnavailable', (MockException,), {})
    mock_exceptions.DeadlineExceeded = type('DeadlineExceeded', (MockException,), {})
    mock_exceptions.InvalidArgument = type('InvalidArgument', (MockException,), {})
    
    # Mock ContentType enum
    class ContentType:
        RESOURCE = "RESOURCE"
        IAM_POLICY = "IAM_POLICY"
        ORG_POLICY = "ORG_POLICY"
        ACCESS_POLICY = "ACCESS_POLICY"
        OS_INVENTORY = "OS_INVENTORY"
    
    mock_asset_v1.ContentType = ContentType
    
    # Mock request classes
    mock_asset_v1.ListAssetsRequest = Mock()
    mock_asset_v1.SearchAllResourcesRequest = Mock()
    mock_asset_v1.ExportAssetsRequest = Mock()
    mock_asset_v1.OutputConfig = Mock()
    mock_asset_v1.GcsDestination = Mock()
    
    return mock_asset_v1, mock_exceptions

# Check if Google Cloud libraries are available
try:
    from google.cloud import asset_v1
    from google.api_core import exceptions as gcp_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    # Apply mocks if not available
    mock_asset_v1, mock_exceptions = mock_google_cloud_imports()
    sys.modules['google.cloud.asset_v1'] = mock_asset_v1
    sys.modules['google.api_core.exceptions'] = mock_exceptions

@pytest.fixture(scope="session")
def gcp_available():
    """Fixture that indicates whether GCP libraries are available"""
    return GCP_AVAILABLE

@pytest.fixture
def mock_environment():
    """Fixture to mock environment variables"""
    with patch.dict(os.environ, {
        'GOOGLE_CLOUD_PROJECT': 'test-project',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test-credentials.json'
    }):
        yield

@pytest.fixture
def disable_logging():
    """Disable logging during tests to reduce output noise"""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance-related"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Add markers based on test class names
        if "Integration" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.integration)
        elif "Performance" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.performance)
        elif "Security" in item.cls.__name__ if item.cls else "":
            item.add_marker(pytest.mark.security)
        else:
            item.add_marker(pytest.mark.unit)
"""
Pytest configuration file for security agent tests.
Provides common fixtures and configuration for all test modules.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ["TESTING"] = "true"
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["ADK_AGENT_MODEL"] = "gemini-1.5-flash"

# Mock streamlit for tests that import frontend modules
mock_st = MagicMock()
mock_st.session_state = {}
mock_st.sidebar = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
mock_st.container = MagicMock()
mock_st.metric = MagicMock()
mock_st.plotly_chart = MagicMock()
mock_st.title = MagicMock()
mock_st.write = MagicMock()
mock_st.markdown = MagicMock()
sys.modules['streamlit'] = mock_st

# Common test fixtures
@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    with patch('sqlite3.connect') as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        yield mock_conn

@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    with patch('backend.services.client.APIClient') as mock_client:
        instance = MagicMock()
        mock_client.return_value = instance
        instance.health_check.return_value = {"status": "healthy"}
        yield instance

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state."""
    return {
        'current_page': 'Dashboard',
        'chat_messages': [],
        'critical_findings_count': 0
    }

# Mock the unified_streaming_client module that doesn't exist
class MockSecurityDashboard:
    def __init__(self):
        self.client = MagicMock()

    def get_metrics(self):
        return {"test": "metrics"}

    def stream_message(self, message):
        return {"response": "test response"}

sys.modules['frontend.unified_streaming_client'] = MagicMock()
sys.modules['frontend.unified_streaming_client'].SecurityDashboard = MockSecurityDashboard

# Mock selenium for UI tests
mock_selenium = MagicMock()
sys.modules['selenium'] = mock_selenium
sys.modules['selenium.webdriver'] = MagicMock()
sys.modules['selenium.webdriver.chrome.options'] = MagicMock()
sys.modules['selenium.webdriver.common.by'] = MagicMock()
sys.modules['selenium.webdriver.support'] = MagicMock()
sys.modules['selenium.webdriver.support.ui'] = MagicMock()
sys.modules['selenium.webdriver.support.expected_conditions'] = MagicMock()
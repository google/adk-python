"""
Comprehensive integration test for multi-turn, cross-domain conversations.
Validates context preservation, tool routing, and data synthesis.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app
from backend.api.sessions import Session
from backend.services.agent_cache_wrapper import AgentCacheWrapper
import agents.adk_agent as agent

@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)

@pytest.fixture
def session_id():
    """Creates a consistent session ID for the test."""
    return "test-session-comprehensive"

    with patch('agent.cache_wrapper', autospec=True) as mock_cache_wrapper:
        # Arrange
        mock_cache_wrapper.analyze_storage_cached.return_value = {
            "security_findings": {
                "critical": [{"bucket": "my-critical-public-bucket", "issue": "PUBLIC ACCESS"}]
            }
        }
        
        # Act
        response1 = client.post(
            "/api/v1/chat/message",
            json={"query": "Which of my storage buckets are public?", "session_id": session_id}
        )
        
        # Assert
        assert response1.status_code == 200
        assert "my-critical-public-bucket" in response1.text
        mock_cache_wrapper.analyze_storage_cached.assert_called_once()

    # 2. Second Turn: Ask about IAM roles for the bucket identified in the first turn.
    #    This tests context preservation and the IAM tool.
    with patch('agent.cache_wrapper', autospec=True) as mock_cache_wrapper:
        # Arrange
        mock_cache_wrapper.analyze_iam_cached.return_value = {
            "iam_analysis": {
                "my-critical-public-bucket": {"roles": ["roles/storage.objectAdmin"]}
            }
        }
        
        # Act
        response2 = client.post(
            "/api/v1/chat/message",
            json={"query": "Who has admin access to that bucket?", "session_id": session_id}
        )
        
        # Assert
        assert response2.status_code == 200
        assert "roles/storage.objectAdmin" in response2.text
        # The agent should have passed the bucket name from context to the IAM tool
        mock_cache_wrapper.analyze_iam_cached.assert_called_once()
        # You could add more specific assertions on the call arguments here

    # 3. Third Turn: Ask for a remediation plan for the issue.
    #    This tests data synthesis, requiring context from both previous turns.
    with patch('agent.cache_wrapper', autospec=True) as mock_cache_wrapper:
        # Arrange
        mock_cache_wrapper.get_security_recommendations_cached.return_value = {
            "recommendations": [{
                "action": "Disable public access", 
                "command": "gcloud storage buckets update gs://my-critical-public-bucket --remove-public-access"
            }]
        }

        # Act
        response3 = client.post(
            "/api/v1/chat/message",
            json={"query": "How do I fix that public access issue?", "session_id": session_id}
        )
        
        # Assert
        assert response3.status_code == 200
        assert "gcloud storage buckets update" in response3.text
        mock_cache_wrapper.get_security_recommendations_cached.assert_called_once()

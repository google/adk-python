# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for backend blocking poll core logic (ADK-compliant, no ADK imports).

Tests the core approval polling logic with mocked HTTP calls without
requiring full ADK installation.
"""

from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
import time


# Configuration
APPROVAL_API_URL = "http://localhost:8003"
APPROVAL_POLL_INTERVAL = 2
APPROVAL_TIMEOUT = 10
MAX_PROPOSAL_LENGTH = 10000


def request_approval_blocking_testable(proposal: str, context: dict = None) -> str:
    """
    Testable version of request_approval_blocking (no ADK dependencies).
    This is the core logic that will be tested.
    """
    import requests

    # Input validation
    if not proposal or not proposal.strip():
        return "❌ Invalid proposal: cannot be empty"

    if len(proposal) > MAX_PROPOSAL_LENGTH:
        return f"❌ Invalid proposal: exceeds {MAX_PROPOSAL_LENGTH} character limit"

    try:
        # Create approval ticket
        response = requests.post(
            f"{APPROVAL_API_URL}/approvals",
            json={
                "proposal": proposal,
                "requester": context.get("requester", "test") if context else "test",
                "metadata": context or {}
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            return f"❌ Failed to create approval ticket: {result.get('message', 'Unknown error')}"

        ticket_id = result["ticket"]["ticket_id"]

        # Poll for decision
        elapsed_time = 0
        poll_count = 0

        while elapsed_time < APPROVAL_TIMEOUT:
            poll_count += 1

            try:
                status_response = requests.get(
                    f"{APPROVAL_API_URL}/approvals/{ticket_id}/status",
                    timeout=10
                )
                status_response.raise_for_status()
                status_data = status_response.json()
            except requests.exceptions.RequestException:
                time.sleep(APPROVAL_POLL_INTERVAL)
                elapsed_time += APPROVAL_POLL_INTERVAL
                continue

            ticket_status = status_data.get("status")

            if ticket_status == "approved":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "Approved")
                return f"✅ APPROVED by {reviewer}\nReason: {reason}"

            elif ticket_status == "rejected":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "No reason")
                return f"❌ REJECTED by {reviewer}\nReason: {reason}"

            elif ticket_status == "changes_requested":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "No details")
                return f"🔄 CHANGES REQUESTED by {reviewer}\nDetails: {reason}"

            elif ticket_status == "pending":
                time.sleep(APPROVAL_POLL_INTERVAL)
                elapsed_time += APPROVAL_POLL_INTERVAL
                continue

            else:
                return f"⚠️ Unknown status: {ticket_status}"

        return f"⏱️ Timeout: No decision after {APPROVAL_TIMEOUT}s"

    except requests.exceptions.ConnectionError as e:
        return f"❌ Cannot connect to API at {APPROVAL_API_URL}: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


class TestBlockingPollCoreLogic:
    """Test suite for core blocking poll logic (no ADK dependencies)."""

    @patch('requests.post')
    @patch('requests.get')
    @patch('time.sleep', return_value=None)
    def test_successful_approval_flow(self, mock_sleep, mock_get, mock_post):
        """Test successful approval workflow with polling."""
        # Mock ticket creation response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": True,
            "ticket": {
                "ticket_id": "APR-TEST123",
                "status": "pending"
            }
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock polling responses: pending, pending, approved
        mock_get_responses = [
            Mock(json=Mock(return_value={"status": "pending"})),
            Mock(json=Mock(return_value={"status": "pending"})),
            Mock(json=Mock(return_value={
                "status": "approved",
                "reviewer": "test_reviewer",
                "decision_reason": "Approved for testing"
            }))
        ]
        for resp in mock_get_responses:
            resp.raise_for_status = Mock()
        mock_get.side_effect = mock_get_responses

        # Execute
        result = request_approval_blocking_testable(
            proposal="Test deployment",
            context={"priority": "high"}
        )

        # Assertions
        assert "APPROVED" in result
        assert "test_reviewer" in result
        assert "Approved for testing" in result
        assert mock_post.call_count == 1
        assert mock_get.call_count == 3

    @patch('requests.post')
    @patch('requests.get')
    def test_rejection_flow(self, mock_get, mock_post):
        """Test rejection workflow."""
        # Mock ticket creation
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": True,
            "ticket": {
                "ticket_id": "APR-TEST456",
                "status": "pending"
            }
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock polling response: rejected
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "status": "rejected",
            "reviewer": "test_reviewer",
            "decision_reason": "Does not meet criteria"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Execute
        result = request_approval_blocking_testable(
            proposal="Test deployment",
            context={"priority": "low"}
        )

        # Assertions
        assert "REJECTED" in result
        assert "test_reviewer" in result
        assert "Does not meet criteria" in result

    def test_empty_proposal_validation(self):
        """Test input validation for empty proposals."""
        # Test empty string
        result = request_approval_blocking_testable("")
        assert "cannot be empty" in result

        # Test whitespace only
        result = request_approval_blocking_testable("   ")
        assert "cannot be empty" in result

    def test_proposal_length_validation(self):
        """Test input validation for proposal length."""
        # Create proposal exceeding MAX_PROPOSAL_LENGTH
        long_proposal = "A" * (MAX_PROPOSAL_LENGTH + 1)

        result = request_approval_blocking_testable(long_proposal)
        assert "exceeds" in result
        assert "character limit" in result

    @patch('requests.post')
    def test_ticket_creation_failure(self, mock_post):
        """Test handling of ticket creation failure."""
        # Mock failed ticket creation
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": False,
            "message": "API error: rate limit exceeded"
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        result = request_approval_blocking_testable(
            proposal="Test deployment"
        )

        assert "Failed to create approval ticket" in result
        assert "rate limit exceeded" in result

    @patch('requests.post')
    def test_connection_error(self, mock_post):
        """Test handling of connection errors."""
        import requests

        # Mock connection error
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Cannot connect to API"
        )

        result = request_approval_blocking_testable(
            proposal="Test deployment"
        )

        assert "Cannot connect" in result

    @patch('requests.post')
    @patch('requests.get')
    @patch('time.sleep', return_value=None)
    def test_timeout_scenario(self, mock_sleep, mock_get, mock_post):
        """Test timeout when approval not received in time."""
        # Mock ticket creation
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": True,
            "ticket": {
                "ticket_id": "APR-TIMEOUT",
                "status": "pending"
            }
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock polling response: always pending
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"status": "pending"}
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        result = request_approval_blocking_testable(
            proposal="Test deployment"
        )

        assert "Timeout" in result or "timeout" in result

    @patch('requests.post')
    @patch('requests.get')
    def test_changes_requested_flow(self, mock_get, mock_post):
        """Test changes_requested workflow."""
        # Mock ticket creation
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": True,
            "ticket": {
                "ticket_id": "APR-CHANGES",
                "status": "pending"
            }
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock polling response: changes_requested
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "status": "changes_requested",
            "reviewer": "test_reviewer",
            "decision_reason": "Need more details on rollback plan"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        result = request_approval_blocking_testable(
            proposal="Test deployment"
        )

        assert "CHANGES REQUESTED" in result
        assert "test_reviewer" in result
        assert "rollback plan" in result

    @patch('requests.post')
    @patch('requests.get')
    @patch('time.sleep', return_value=None)
    def test_transient_error_retry(self, mock_sleep, mock_get, mock_post):
        """Test that transient errors during polling are retried."""
        import requests

        # Mock ticket creation
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "success": True,
            "ticket": {
                "ticket_id": "APR-RETRY",
                "status": "pending"
            }
        }
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock polling responses: error, error, then approved
        mock_get.side_effect = [
            requests.exceptions.RequestException("Transient error"),
            requests.exceptions.RequestException("Another transient error"),
            Mock(json=Mock(return_value={
                "status": "approved",
                "reviewer": "test_reviewer",
                "decision_reason": "Approved after retries"
            }), raise_for_status=Mock())
        ]

        result = request_approval_blocking_testable(
            proposal="Test deployment"
        )

        assert "APPROVED" in result
        assert "Approved after retries" in result
        assert mock_get.call_count == 3  # 2 errors + 1 success


class TestConfigurationConstants:
    """Test suite for configuration constants."""

    def test_timeout_greater_than_poll_interval(self):
        """Verify APPROVAL_TIMEOUT is greater than APPROVAL_POLL_INTERVAL."""
        assert APPROVAL_TIMEOUT > APPROVAL_POLL_INTERVAL, \
            "APPROVAL_TIMEOUT must be greater than APPROVAL_POLL_INTERVAL"

    def test_poll_interval_reasonable(self):
        """Verify poll interval is reasonable (not too short)."""
        assert APPROVAL_POLL_INTERVAL >= 1, \
            "APPROVAL_POLL_INTERVAL should be at least 1 second"

    def test_max_proposal_length_reasonable(self):
        """Verify max proposal length is reasonable."""
        assert MAX_PROPOSAL_LENGTH == 10000, \
            "MAX_PROPOSAL_LENGTH should be 10,000 characters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

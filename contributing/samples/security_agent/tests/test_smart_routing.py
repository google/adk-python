# Copyright 2024 Google LLC
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

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the problematic modules before they're imported by other modules.
mock_google = MagicMock()
mock_google.__path__ = []
mock_google.__spec__ = MagicMock()
mock_google.adk = MagicMock()
mock_google.genai = MagicMock()
sys.modules["google"] = mock_google
sys.modules["google.adk"] = mock_google.adk
sys.modules["google.adk"] = mock_google.adk
sys.modules["google.genai"] = mock_google.genai
mock_tools = MagicMock()
mock_tools.__path__ = []
mock_tools.gcp = MagicMock()
sys.modules["tools"] = mock_tools
sys.modules["tools.gcp"] = mock_tools.gcp
sys.modules["vertexai"] = MagicMock()

# Mock RouterAgent for testing
class RouterAgent:
    def __init__(self, llm=None):
        self.llm = llm
from contributing.samples.security_agent.backend.api.agent_llm import process_with_llm_agent


class TestIntelligentRouting(unittest.TestCase):
    """Tests for the intelligent routing flow."""

    def setUp(self):
        """Set up the test environment."""
        self.mock_llm = MagicMock()
        self.router_agent = RouterAgent(llm=self.mock_llm)

    def test_route_to_storage_agent(self):
        """Test that a storage-related query routes to the StorageAgent."""
        query = "check my storage buckets"
        expected_agent = "StorageAgent"
        self.mock_llm.predict.return_value = f"Agent: {expected_agent}"
        actual_agent = self.router_agent.route(query)
        self.assertEqual(actual_agent, expected_agent)

    def test_route_to_iam_agent(self):
        """Test that an IAM-related query routes to the IAMAgent."""
        query = "who has access to my project?"
        expected_agent = "IAMAgent"
        self.mock_llm.predict.return_value = f"Agent: {expected_agent}"
        actual_agent = self.router_agent.route(query)
        self.assertEqual(actual_agent, expected_agent)



        query = "is my project secure?"

        self.mock_llm.predict.return_value = "Agent: Unknown"
        actual_agent = self.router_agent.route(query)
        self.assertEqual(actual_agent, expected_agent)

    @patch(
        "contributing.samples.security_agent.backend.api.agent_llm.get_agent"
    )
    def test_process_with_llm_agent_integration(self, mock_get_agent):
        """Test the end-to-end flow from API to specialist agent."""
        mock_router = MagicMock()
        mock_specialist = MagicMock()

        mock_get_agent.side_effect = [mock_router, mock_specialist]
        mock_router.route.return_value = "StorageAgent"
        mock_specialist.chat.return_value = "Here are your buckets."

        response = process_with_llm_agent("show me my buckets", "test_session")

        mock_router.route.assert_called_once_with("show me my buckets")
        mock_get_agent.assert_any_call("RouterAgent", "test_session")
        mock_get_agent.assert_any_call("StorageAgent", "test_session")
        mock_specialist.chat.assert_called_once_with("show me my buckets")
        self.assertEqual(response, "Here are your buckets.")


if __name__ == "__main__":
    unittest.main()
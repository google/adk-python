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

"""Test fixtures for utils tests."""

import sys
from unittest import mock

# Create mocks for modules before they're imported
mock_event = mock.MagicMock()
mock_event.Event = mock.MagicMock()

mock_adk_to_mcp_tool_type = mock.MagicMock()
mock_async_utils = mock.MagicMock()
mock_date_utils = mock.MagicMock()
mock_image_utils = mock.MagicMock()
mock_multipart_utils = mock.MagicMock()
mock_structured_output_utils = mock.MagicMock()
mock_truncate_utils = mock.MagicMock()
mock_uri_utils = mock.MagicMock()
mock_llm_response = mock.MagicMock()
mock_variant_utils = mock.MagicMock()
mock_variant_utils.GoogleLLMVariant = mock.MagicMock()

# Apply mocks at the module level before any imports happen
sys.modules["google.adk.events.event"] = mock_event
sys.modules["google.adk.utils.adk_to_mcp_tool_type"] = mock_adk_to_mcp_tool_type
sys.modules["google.adk.utils.async_utils"] = mock_async_utils
sys.modules["google.adk.utils.date_utils"] = mock_date_utils
sys.modules["google.adk.utils.image_utils"] = mock_image_utils
sys.modules["google.adk.utils.multipart_utils"] = mock_multipart_utils
sys.modules["google.adk.utils.structured_output_utils"] = mock_structured_output_utils
sys.modules["google.adk.utils.truncate_utils"] = mock_truncate_utils
sys.modules["google.adk.utils.uri_utils"] = mock_uri_utils
sys.modules["google.adk.models.llm_response"] = mock_llm_response
sys.modules["google.adk.utils.variant_utils"] = mock_variant_utils


# Function to reset mocks after tests
def pytest_unconfigure(config):
    """Reset any global state modified by the tests."""
    # Remove our mocked modules
    for module_name in [
        "google.adk.events.event",
        "google.adk.utils.adk_to_mcp_tool_type",
        "google.adk.utils.async_utils",
        "google.adk.utils.date_utils",
        "google.adk.utils.image_utils",
        "google.adk.utils.multipart_utils",
        "google.adk.utils.structured_output_utils",
        "google.adk.utils.truncate_utils",
        "google.adk.utils.uri_utils",
        "google.adk.models.llm_response",
        "google.adk.utils.variant_utils",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

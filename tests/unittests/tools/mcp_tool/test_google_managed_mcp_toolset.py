# Copyright 2026 Google LLC
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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
import pytest


@pytest.fixture
def mock_google_auth():
  """Fixture that patches google.auth.default and transport.requests.Request.

  Since google.auth is lazy-imported inside _create_adc_header_provider,
  we patch the actual google.auth module, not via the toolset's namespace.
  """
  mock_credentials = MagicMock()
  mock_credentials.token = "fake-token-123"

  with patch(
      "google.auth.default",
      return_value=(mock_credentials, "fake-project"),
  ) as mock_default, patch(
      "google.auth.transport.requests.Request",
      return_value=MagicMock(),
  ) as mock_request_cls:
    yield {
        "credentials": mock_credentials,
        "mock_default": mock_default,
        "mock_request_cls": mock_request_cls,
    }


class TestGoogleManagedMcpToolset:
  """Test suite for GoogleManagedMcpToolset class."""

  def test_init_valid_product(self, mock_google_auth):
    """Creating with a supported product name succeeds."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    assert toolset is not None

  def test_init_unknown_product_raises(self):
    """Creating with an unsupported product name raises ValueError."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    with pytest.raises(ValueError, match="Unknown Google MCP product"):
      GoogleManagedMcpToolset(product="unknown_product")

  def test_init_case_insensitive(self, mock_google_auth):
    """Product name matching is case-insensitive."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="BigQuery")
    assert toolset is not None

    toolset2 = GoogleManagedMcpToolset(product="BIGQUERY")
    assert toolset2 is not None

  def test_connection_params_url(self, mock_google_auth):
    """Internal connection params use the correct BigQuery MCP endpoint."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    assert isinstance(
        toolset._connection_params, StreamableHTTPConnectionParams
    )
    assert (
        toolset._connection_params.url
        == "https://bigquery.googleapis.com/mcp"
    )

  def test_header_provider_is_set(self, mock_google_auth):
    """A header_provider is automatically configured."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    assert toolset._header_provider is not None
    assert callable(toolset._header_provider)

  def test_header_provider_returns_bearer_token(self, mock_google_auth):
    """The header_provider returns an Authorization Bearer header."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    mock_context = Mock()

    headers = toolset._header_provider(mock_context)

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer fake-token-123"

  def test_header_provider_refreshes_credentials(self, mock_google_auth):
    """The header_provider calls credentials.refresh() each time."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    mock_context = Mock()
    mock_credentials = mock_google_auth["credentials"]

    # Call header_provider twice
    toolset._header_provider(mock_context)
    toolset._header_provider(mock_context)

    assert mock_credentials.refresh.call_count == 2

  def test_custom_scopes(self, mock_google_auth):
    """Custom scopes are passed to google.auth.default()."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    custom_scopes = [
        "https://www.googleapis.com/auth/bigquery.readonly"
    ]
    GoogleManagedMcpToolset(product="bigquery", scopes=custom_scopes)

    mock_google_auth["mock_default"].assert_called_with(
        scopes=custom_scopes
    )

  def test_default_scopes(self, mock_google_auth):
    """Default product scopes are used when none are specified."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        _GOOGLE_MCP_PRODUCTS,
        GoogleManagedMcpToolset,
    )

    GoogleManagedMcpToolset(product="bigquery")

    _, default_scopes = _GOOGLE_MCP_PRODUCTS["bigquery"]
    mock_google_auth["mock_default"].assert_called_with(
        scopes=default_scopes
    )

  def test_custom_project_id(self, mock_google_auth):
    """Custom project_id is included in the x-goog-user-project header."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(
        product="bigquery", project_id="my-custom-project"
    )
    mock_context = Mock()

    headers = toolset._header_provider(mock_context)

    assert headers.get("x-goog-user-project") == "my-custom-project"

  def test_default_project_from_adc(self, mock_google_auth):
    """When no project_id is given, ADC default project is used."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(product="bigquery")
    mock_context = Mock()

    headers = toolset._header_provider(mock_context)

    # The mock returns "fake-project" as the default project
    assert headers.get("x-goog-user-project") == "fake-project"

  def test_tool_filter_forwarded(self, mock_google_auth):
    """tool_filter parameter is forwarded to the parent McpToolset."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    tool_filter = ["list_datasets", "execute_sql"]
    toolset = GoogleManagedMcpToolset(
        product="bigquery", tool_filter=tool_filter
    )
    # Verify the parent stored the filter
    assert toolset._is_tool_selected is not None

  def test_kwargs_forwarded(self, mock_google_auth):
    """Extra kwargs like use_mcp_resources are forwarded to parent."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(
        product="bigquery", use_mcp_resources=True
    )
    assert toolset._use_mcp_resources is True

  def test_tool_name_prefix_forwarded(self, mock_google_auth):
    """tool_name_prefix is forwarded to parent McpToolset."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        GoogleManagedMcpToolset,
    )

    toolset = GoogleManagedMcpToolset(
        product="bigquery", tool_name_prefix="bq"
    )
    assert toolset.tool_name_prefix == "bq"


class TestCreateAdcHeaderProvider:
  """Tests for the _create_adc_header_provider helper function."""

  def test_returns_callable(self, mock_google_auth):
    """Returns a callable header_provider."""
    from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
        _create_adc_header_provider,
    )

    provider = _create_adc_header_provider(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    assert callable(provider)

  def test_no_project_header_when_none(self):
    """No x-goog-user-project when both project_id and default are None."""
    mock_credentials = MagicMock()
    mock_credentials.token = "test-token"

    with patch(
        "google.auth.default",
        return_value=(mock_credentials, None),
    ), patch(
        "google.auth.transport.requests.Request",
        return_value=MagicMock(),
    ):
      from google.adk.tools.mcp_tool.google_managed_mcp_toolset import (
          _create_adc_header_provider,
      )

      provider = _create_adc_header_provider(
          scopes=["https://www.googleapis.com/auth/bigquery"],
          project_id=None,
      )
      headers = provider(Mock())
      assert "x-goog-user-project" not in headers

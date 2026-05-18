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

"""Tests for JWT token propagation feature in MCP toolset."""

import logging
import sys
from unittest.mock import Mock

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from google.adk.agents.readonly_context import ReadonlyContext
  from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider
  from google.adk.tools.mcp_tool.mcp_toolset import McpToolsetConfig
  from mcp import StdioServerParameters
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    class DummyClass:
      pass

    create_session_state_header_provider = DummyClass
    McpToolsetConfig = DummyClass
    StdioServerParameters = DummyClass
    ReadonlyContext = DummyClass
  else:
    raise e


class TestCreateSessionStateHeaderProvider:
  """Test suite for create_session_state_header_provider function."""

  def test_extract_jwt_token_default_format(self):
    """Test extracting JWT token with default Authorization Bearer format."""
    # Create mock context with JWT token in state
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {
        "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }

    # Create header provider
    provider = create_session_state_header_provider(state_key="jwt_token")

    # Call provider
    headers = provider(mock_context)

    # Verify headers
    assert headers == {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }

  def test_extract_with_custom_header_name(self):
    """Test extracting token with custom header name."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"api_key": "secret-key-123"}

    provider = create_session_state_header_provider(
        state_key="api_key", header_name="X-API-Key", header_format="{value}"
    )

    headers = provider(mock_context)

    assert headers == {"X-API-Key": "secret-key-123"}

  def test_extract_with_custom_format(self):
    """Test extracting token with custom formatting."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"tenant_id": "tenant-123"}

    provider = create_session_state_header_provider(
        state_key="tenant_id",
        header_name="X-Tenant-ID",
        header_format="tenant:{value}",
    )

    headers = provider(mock_context)

    assert headers == {"X-Tenant-ID": "tenant:tenant-123"}

  def test_missing_state_key_returns_empty(self):
    """Test that missing state key returns empty dict when no default."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {}

    provider = create_session_state_header_provider(state_key="jwt_token")

    headers = provider(mock_context)

    assert headers == {}

  def test_missing_state_key_uses_default(self):
    """Test that missing state key uses default value if provided."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {}

    provider = create_session_state_header_provider(
        state_key="jwt_token", default_value="default-token"
    )

    headers = provider(mock_context)

    assert headers == {"Authorization": "Bearer default-token"}

  def test_none_value_in_state_returns_empty(self):
    """Test that None value in state returns empty dict."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": None}

    provider = create_session_state_header_provider(state_key="jwt_token")

    headers = provider(mock_context)

    assert headers == {}

  def test_empty_string_value_returns_empty(self):
    """Test that empty string value in state returns empty dict."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": ""}

    provider = create_session_state_header_provider(state_key="jwt_token")

    headers = provider(mock_context)

    assert headers == {}

  def test_strict_mode_with_primitive_types(self):
    """Test that strict mode works properly with primitive types."""
    mock_context = Mock(spec=ReadonlyContext)

    # Test with string
    mock_context.state = {"token": "my-token"}
    provider = create_session_state_header_provider(
        state_key="token", strict=True
    )
    headers = provider(mock_context)
    assert headers == {"Authorization": "Bearer my-token"}

    # Test with int
    mock_context.state = {"count": 42}
    provider = create_session_state_header_provider(
        state_key="count",
        header_name="X-Count",
        header_format="{value}",
        strict=True,
    )
    headers = provider(mock_context)
    assert headers == {"X-Count": "42"}

  def test_strict_mode_raises_on_non_primitive_types(self):
    """Test that strict mode raises ValueError for non-primitive types."""
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"complex_data": {"nested": "dict"}}

    provider = create_session_state_header_provider(
        state_key="complex_data", strict=True
    )

    with pytest.raises(ValueError) as exc_info:
      provider(mock_context)

    assert "complex_data" in str(exc_info.value)
    assert "dict" in str(exc_info.value)
    assert "may not serialize correctly" in str(exc_info.value)


class TestMcpToolsetConfigStateHeaderMapping:
  """Test suite for state_header_mapping configuration."""

  def test_config_with_single_state_mapping(self):
    """Test config with single state to header mapping."""
    config = McpToolsetConfig(
        stdio_server_params=StdioServerParameters(
            command="test_command", args=[]
        ),
        state_header_mapping={"jwt_token": "Authorization"},
        state_header_format={"Authorization": "Bearer {value}"},
    )

    assert config.state_header_mapping == {"jwt_token": "Authorization"}
    assert config.state_header_format == {"Authorization": "Bearer {value}"}

  def test_config_with_multiple_state_mappings(self):
    """Test config with multiple state to header mappings."""
    config = McpToolsetConfig(
        stdio_server_params=StdioServerParameters(
            command="test_command", args=[]
        ),
        state_header_mapping={
            "jwt_token": "Authorization",
            "tenant_id": "X-Tenant-ID",
            "api_key": "X-API-Key",
        },
        state_header_format={
            "Authorization": "Bearer {value}",
            "X-API-Key": "key:{value}",
        },
    )

    assert len(config.state_header_mapping) == 3
    assert config.state_header_mapping["jwt_token"] == "Authorization"
    assert config.state_header_mapping["tenant_id"] == "X-Tenant-ID"
    assert config.state_header_format["Authorization"] == "Bearer {value}"

  def test_config_without_state_mapping(self):
    """Test config without state mapping (backward compatibility)."""
    config = McpToolsetConfig(
        stdio_server_params=StdioServerParameters(
            command="test_command", args=[]
        )
    )

    assert config.state_header_mapping is None
    assert config.state_header_format is None


class TestHeaderSecurityValidation:
  """Test suite for header security validation features."""

  def test_header_name_validation_valid_names(self):
    """Test that valid header names are accepted."""
    from google.adk.tools.mcp_tool._internal import validate_header_name

    # Valid header names should not raise exceptions
    valid_names = [
        "Authorization",
        "X-API-Key",
        "Content-Type",
        "X-Custom-Header",
    ]

    for name in valid_names:
      validate_header_name(name)  # Should not raise

  def test_header_name_validation_invalid_names(self):
    """Test that invalid header names are rejected."""
    from google.adk.tools.mcp_tool._internal import validate_header_name

    # Invalid header names should raise ValueError
    invalid_names = [
        "",  # Empty string
        "Authorization\n",  # Newline
        "X-API:Key",  # Colon
        "X-API Key",  # Space
        "X-API\x01Key",  # Control character
    ]

    for name in invalid_names:
      with pytest.raises(ValueError) as exc_info:
        validate_header_name(name)
      assert (
          "invalid characters" in str(exc_info.value).lower()
          or "empty" in str(exc_info.value).lower()
      )

  def test_header_value_sanitization_safe_values(self):
    """Test that safe header values are unchanged."""
    from google.adk.tools.mcp_tool._internal import sanitize_header_value

    safe_values = [
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "api-key-123",
        "tenant-456",
        "Basic dXNlcjpwYXNz",  # Base64 auth
    ]

    for value in safe_values:
      result = sanitize_header_value(value)
      assert result == value

  def test_header_value_sanitization_dangerous_values(self):
    """Test that dangerous characters are removed from header values."""
    from google.adk.tools.mcp_tool._internal import sanitize_header_value

    dangerous_values = [
        ("Bearer token\x00injected", "Bearer tokeninjected"),
        ("api-key\x00malicious", "api-keymalicious"),
        ("value\x00more", "valuemore"),
        ("token\x00data", "tokendata"),
    ]

    for input_val, expected in dangerous_values:
      result = sanitize_header_value(input_val)
      assert result == expected

  def test_header_value_sanitization_non_string_values(self):
    """Test that non-string values are converted to string."""
    from google.adk.tools.mcp_tool._internal import sanitize_header_value

    result_int = sanitize_header_value(123)
    assert result_int == "123"

    result_bool = sanitize_header_value(True)
    assert result_bool == "True"

  def test_session_state_header_provider_with_invalid_header_name(self):
    """Test that invalid header names raise ValueError during provider creation."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    with pytest.raises(ValueError) as exc_info:
      create_session_state_header_provider(
          state_key="token",
          header_name="Authorization\n",  # Invalid header name
      )

    assert "invalid characters" in str(exc_info.value).lower()

  def test_session_state_header_provider_sanitizes_values(self):
    """Test that header provider sanitizes values from state."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"token": "Bearer\x00token\x01injected"}

    provider = create_session_state_header_provider(
        state_key="token", header_name="Authorization", header_format="{value}"
    )

    headers = provider(mock_context)

    # The provider should sanitize the dangerous characters
    assert headers == {"Authorization": "Bearertokeninjected"}


class TestMcpToolsetFromConfigWithStateMapping:
  """Test suite for McpToolset.from_config with state header mapping."""

  def test_from_config_creates_header_provider(self):
    """Test that from_config creates header provider from state mapping."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_configs import ToolArgsConfig

    config = ToolArgsConfig(
        stdio_server_params={"command": "test_command", "args": []},
        state_header_mapping={"jwt_token": "Authorization"},
        state_header_format={"Authorization": "Bearer {value}"},
    )

    toolset = McpToolset.from_config(config, "/fake/path")

    # Verify header_provider was created
    assert toolset._header_provider is not None

    # Test the created header provider
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": "test-token-123"}

    headers = toolset._header_provider(mock_context)

    assert headers == {"Authorization": "Bearer test-token-123"}

  def test_from_config_multiple_headers(self):
    """Test from_config with multiple header mappings."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_configs import ToolArgsConfig

    config = ToolArgsConfig(
        stdio_server_params={"command": "test_command", "args": []},
        state_header_mapping={
            "jwt_token": "Authorization",
            "tenant_id": "X-Tenant-ID",
        },
        state_header_format={"Authorization": "Bearer {value}"},
    )

    toolset = McpToolset.from_config(config, "/fake/path")

    # Test the created header provider
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": "token-123", "tenant_id": "tenant-456"}

    headers = toolset._header_provider(mock_context)

    assert headers["Authorization"] == "Bearer token-123"
    assert headers["X-Tenant-ID"] == "tenant-456"

  def test_from_config_omits_missing_state_keys(self):
    """Test that missing state keys are omitted from headers."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_configs import ToolArgsConfig

    config = ToolArgsConfig(
        stdio_server_params={"command": "test_command", "args": []},
        state_header_mapping={
            "jwt_token": "Authorization",
            "tenant_id": "X-Tenant-ID",
        },
    )

    toolset = McpToolset.from_config(config, "/fake/path")

    # Only include jwt_token in state
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": "token-123"}

    headers = toolset._header_provider(mock_context)

    # Only Authorization header should be present
    assert "Authorization" in headers
    assert "X-Tenant-ID" not in headers

  def test_from_config_no_state_mapping_no_provider(self):
    """Test that no header provider is created when no state mapping."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_configs import ToolArgsConfig

    config = ToolArgsConfig(
        stdio_server_params={"command": "test_command", "args": []}
    )

    toolset = McpToolset.from_config(config, "/fake/path")

    # No header provider should be created
    assert toolset._header_provider is None

  def test_from_config_with_strict_mode(self):
    """Test that from_config respects state_header_strict setting."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_configs import ToolArgsConfig

    config = ToolArgsConfig(
        stdio_server_params={"command": "test_command", "args": []},
        state_header_mapping={"data": "X-Data"},
        state_header_strict=True,  # Enable strict mode
    )

    toolset = McpToolset.from_config(config, "/fake/path")

    # Test with non-primitive type - should raise ValueError
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"data": {"nested": "object"}}

    with pytest.raises(ValueError) as exc_info:
      toolset._header_provider(mock_context)

    assert "data" in str(exc_info.value)
    assert "dict" in str(exc_info.value)


class TestRFC7230Compliance:
  """Test suite for RFC 7230 compliant header handling."""

  def test_header_name_validation_rfc_compliant(self):
    """Test that header name validation follows RFC 7230."""
    from google.adk.tools.mcp_tool._internal import validate_header_name

    # RFC 7230 compliant header names should be accepted
    valid_names = [
        "Authorization",
        "X-API-Key",
        "Content-Type",
        "X-Custom-Header-123",
        "Accept-Encoding",
        "User-Agent",
        "If-Modified-Since",
    ]

    for name in valid_names:
      validate_header_name(name)  # Should not raise

    # RFC 7230 invalid header names should be rejected
    invalid_names = [
        "",  # Empty
        "Authorization\n",  # Newline
        "X-API:Key",  # Colon
        "X API Key",  # Space
        "X-API\x01Key",  # Control character
        "X-API()Key",  # Parentheses
        "X-API@Key",  # At symbol
        "X-API,Key",  # Comma
        "X-API;Key",  # Semicolon
        'X-API"Key',  # Double quote
        "X-API\\Key",  # Backslash
        "X-API/Key",  # Forward slash
        "X-API[Key]",  # Brackets
        "X-API?Key",  # Question mark
        "X-API=Key",  # Equals
        "X-API{Key}",  # Braces
        "X-API\tKey",  # Tab
    ]

    for name in invalid_names:
      with pytest.raises(ValueError) as exc_info:
        validate_header_name(name)
      assert (
          "invalid characters" in str(exc_info.value).lower()
          or "empty" in str(exc_info.value).lower()
      )

  def test_header_value_sanitization_rfc_compliant(self):
    """Test that header value sanitization is RFC 7230 compliant."""
    from google.adk.tools.mcp_tool._internal import sanitize_header_value

    # Safe header values should remain unchanged
    safe_values = [
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "application/json",
        "text/html; charset=utf-8",
        "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        "Basic dXNlcjpwYXNz",  # Base64 auth
        "token-123456789",
        "api-key-secret",
    ]

    for value in safe_values:
      result = sanitize_header_value(value)
      assert result == value

    # Only truly dangerous characters should be removed
    dangerous_cases = [
        ("Bearer\x00token", "Bearertoken"),  # Null byte
        ("token\x01inject", "tokeninject"),  # SOH
        ("data\x02malicious", "datamalicious"),  # STX
        ("value\x03attack", "valueattack"),  # ETX
        ("header\x04break", "headerbreak"),  # EOT
        ("content\x05with\x06control", "contentwithcontrol"),  # ENQ, ACK
        ("test\x07bell\x08backspace", "testbellbackspace"),  # BEL, BS
        ("text\x0Bvertical\x0Ctab", "textverticaltab"),  # VT, FF
        ("shift\x0E\x0Fin", "shiftin"),  # SO, SI
        ("dle\x10control", "dlecontrol"),  # DLE
        ("data\x11\x12\x13\x14chars", "datachars"),  # DC1-DC4
        ("nack\x15syn\x16etb\x17", "nacksynetb"),  # NAK, SYN, ETB
        ("can\x18em\x19sub\x1Aesc", "canemsubesc"),  # CAN, EM, SUB, ESC
        ("fs\x1ags\x1brs\x1cus", "fsgsrsus"),  # FS, GS, RS, US
        ("tok\r\nX-Injected: evil", "tokX-Injected: evil"),  # CRLF
        ("tok\nInjected: bad", "tokInjected: bad"),  # LF
        ("tok\rAnother: header", "tokAnother: header"),  # CR
        ("tab\ttest", "tabtest"),  # TAB should be removed
        ("space\x20test", "space test"),  # Space should be preserved
        (
            "normal!@#$%^&*()test",
            "normal!@#$%^&*()test",
        ),  # Special chars preserved
    ]

    for input_val, expected in dangerous_cases:
      result = sanitize_header_value(input_val)
      assert result == expected

  def test_header_value_crlf_stripped_in_provider(self):
    """Test that CRLF in state values is stripped to prevent header injection."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    # CRLF in the token value should be stripped from the final header
    malicious_cases = [
        ("tok\r\nX-Injected: evil", "Bearer tokX-Injected: evil"),
        ("tok\nInjected: bad", "Bearer tokInjected: bad"),
        ("tok\rAnother: header", "Bearer tokAnother: header"),
        ("tok\r\n", "Bearer tok"),
        ("tok\n", "Bearer tok"),
        ("tok\r", "Bearer tok"),
    ]

    for malicious_value, expected_header in malicious_cases:
      mock_context = Mock(spec=ReadonlyContext)
      mock_context.state = {"jwt_token": malicious_value}
      provider = create_session_state_header_provider(state_key="jwt_token")
      headers = provider(mock_context)
      assert headers == {
          "Authorization": expected_header
      }, f"Failed for value {malicious_value!r}"

    # Clean values should be unchanged
    mock_context = Mock(spec=ReadonlyContext)
    mock_context.state = {"jwt_token": "clean-token-123"}
    provider = create_session_state_header_provider(state_key="jwt_token")
    headers = provider(mock_context)
    assert headers == {"Authorization": "Bearer clean-token-123"}

  def test_header_value_sanitization_in_provider(self):
    """Test that header value sanitization works through the provider path."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    mock_context = Mock(spec=ReadonlyContext)

    # Test that dangerous characters in values get sanitized
    mock_context.state = {"token": "clean-token"}
    provider = create_session_state_header_provider(
        state_key="token", header_name="Authorization", header_format="{value}"
    )
    headers = provider(mock_context)
    assert headers == {"Authorization": "clean-token"}

    # Test that numbers and booleans are handled correctly
    mock_context.state = {"count": 123}
    provider = create_session_state_header_provider(
        state_key="count", header_name="X-Count", header_format="{value}"
    )
    headers = provider(mock_context)
    assert headers == {"X-Count": "123"}

  def test_session_state_header_provider_rfc_compliant(self):
    """Test that session state header provider handles edge cases correctly."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    mock_context = Mock(spec=ReadonlyContext)

    # Test with dangerous characters that get sanitized
    mock_context.state = {"token": "Bearer\x00token\x01with\x02control"}
    provider = create_session_state_header_provider(
        state_key="token", header_name="Authorization", header_format="{value}"
    )

    headers = provider(mock_context)
    assert headers == {"Authorization": "Bearertokenwithcontrol"}

    # Test with None and empty values
    mock_context.state = {"token": None}
    provider = create_session_state_header_provider(state_key="token")
    headers = provider(mock_context)
    assert headers == {}

    mock_context.state = {"token": ""}
    provider = create_session_state_header_provider(state_key="token")
    headers = provider(mock_context)
    assert headers == {}

    # Test with default value
    mock_context.state = {}
    provider = create_session_state_header_provider(
        state_key="token", default_value="default-token"
    )
    headers = provider(mock_context)
    assert headers == {"Authorization": "Bearer default-token"}

  def test_header_format_crlf_injection_protection(self):
    """Test that header format strings with CRLF sequences are rejected."""
    from google.adk.tools.mcp_tool._internal import validate_header_format
    from google.adk.tools.mcp_tool.mcp_toolset import create_session_state_header_provider

    # Valid format strings should be accepted
    valid_formats = [
        "Bearer {value}",
        "Basic {value}",
        "key:{value}",
        "Token {value}",
        "{value}",
    ]
    for fmt in valid_formats:
      validate_header_format(fmt)  # Should not raise

    # Format strings with CRLF should be rejected
    invalid_formats = [
        "Bearer {value}\r\nX-Injected: evil",
        "Bearer {value}\nInjected-Header: bad",
        "Bearer {value}\rAnother: header",
        "Bearer {value}\r\n",
        "Bearer {value}\n",
        "Bearer {value}\r",
    ]
    for fmt in invalid_formats:
      with pytest.raises(ValueError, match="CRLF"):
        validate_header_format(fmt)

    # Test that create_session_state_header_provider validates format
    with pytest.raises(ValueError, match="CRLF"):
      create_session_state_header_provider(
          state_key="token",
          header_name="Authorization",
          header_format="Bearer {value}\r\nX-Injected: evil",
      )


class TestMcpToolsetConfigValidation:
  """Test suite for McpToolsetConfig state header validation."""

  def test_format_without_mapping_raises(self):
    """Test that state_header_format without mapping raises ValueError."""
    with pytest.raises(ValueError, match="state_header_format cannot be set"):
      McpToolsetConfig(
          stdio_server_params=StdioServerParameters(
              command="test_command", args=[]
          ),
          state_header_format={"Authorization": "Bearer {value}"},
      )

  def test_format_key_not_in_mapping_values_raises(self):
    """Test that format key not matching any mapping value raises."""
    with pytest.raises(ValueError, match="does not match"):
      McpToolsetConfig(
          stdio_server_params=StdioServerParameters(
              command="test_command", args=[]
          ),
          state_header_mapping={"jwt_token": "Authorization"},
          state_header_format={"X-Wrong-Header": "Bearer {value}"},
      )

  def test_invalid_header_name_in_mapping_raises(self):
    """Test that invalid header name in mapping value raises ValueError."""
    with pytest.raises(ValueError, match="invalid characters"):
      McpToolsetConfig(
          stdio_server_params=StdioServerParameters(
              command="test_command", args=[]
          ),
          state_header_mapping={"jwt_token": "Authorization\n"},
      )

  def test_crlf_in_format_value_raises(self):
    """Test that CRLF in format string raises ValueError."""
    with pytest.raises(ValueError, match="CRLF"):
      McpToolsetConfig(
          stdio_server_params=StdioServerParameters(
              command="test_command", args=[]
          ),
          state_header_mapping={"jwt_token": "Authorization"},
          state_header_format={
              "Authorization": "Bearer {value}\r\nX-Injected: evil"
          },
      )

  def test_valid_config_passes_validation(self):
    """Test that valid config passes all validation."""
    config = McpToolsetConfig(
        stdio_server_params=StdioServerParameters(
            command="test_command", args=[]
        ),
        state_header_mapping={
            "jwt_token": "Authorization",
            "tenant_id": "X-Tenant-ID",
        },
        state_header_format={"Authorization": "Bearer {value}"},
    )
    assert config.state_header_mapping is not None


class TestCombinedHeaderProviderDuplicateWarning:
  """Test suite for duplicate header warning in combined provider."""

  def test_warns_on_duplicate_headers(self, caplog):
    """Test that duplicate header names trigger a warning."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_combined_header_provider

    provider1 = lambda ctx: {"Authorization": "Bearer token1"}  # noqa: E731
    provider2 = lambda ctx: {"Authorization": "Bearer token2"}  # noqa: E731

    combined = create_combined_header_provider([provider1, provider2])

    with caplog.at_level(logging.WARNING, logger="google_adk"):
      headers = combined(Mock(spec=ReadonlyContext))

    assert headers["Authorization"] == "Bearer token2"
    assert "Duplicate header names" in caplog.text

  def test_no_warning_without_duplicates(self, caplog):
    """Test that no warning is logged when headers don't overlap."""
    from google.adk.tools.mcp_tool.mcp_toolset import create_combined_header_provider

    provider1 = lambda ctx: {"Authorization": "Bearer token1"}  # noqa: E731
    provider2 = lambda ctx: {"X-Tenant-ID": "tenant-123"}  # noqa: E731

    combined = create_combined_header_provider([provider1, provider2])

    with caplog.at_level(logging.WARNING, logger="google_adk"):
      headers = combined(Mock(spec=ReadonlyContext))

    assert "Duplicate header names" not in caplog.text
    assert headers == {
        "Authorization": "Bearer token1",
        "X-Tenant-ID": "tenant-123",
    }

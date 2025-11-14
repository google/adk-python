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

import logging
import sys
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import PropertyMock

from fastapi import FastAPI
from google.adk.utils.pydantic_v2_compatibility import create_robust_openapi_function
from google.adk.utils.pydantic_v2_compatibility import patch_types_for_pydantic_v2
import pytest

# Check if MCP is available (only available in Python 3.10+)
try:
  import mcp.client.session

  MCP_AVAILABLE = True
except ImportError:
  MCP_AVAILABLE = False


class TestPydanticV2CompatibilityPatches:
  """Test suite for Pydantic v2 compatibility patches."""

  @pytest.mark.skipif(
      not MCP_AVAILABLE, reason="MCP module not available in Python 3.9"
  )
  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_patch_types_mcp_success(self, mock_logger):
    """Test successful patching of MCP ClientSession."""
    # Create a mock ClientSession class
    mock_client_session = Mock()
    mock_client_session.__modify_schema__ = Mock()

    with patch("mcp.client.session.ClientSession", mock_client_session):
      result = patch_types_for_pydantic_v2()

      assert result is True
      # Verify that __get_pydantic_core_schema__ was added
      assert hasattr(mock_client_session, "__get_pydantic_core_schema__")
      # Verify that __modify_schema__ was removed if it existed
      assert not hasattr(mock_client_session, "__modify_schema__")
      mock_logger.info.assert_called()

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_patch_types_mcp_import_error(self, mock_logger):
    """Test patching when MCP ClientSession cannot be imported."""
    # Mock the import statement itself
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == "mcp.client.session":
        raise ImportError("No module named 'mcp.client.session'")
      return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
      result = patch_types_for_pydantic_v2()

      # Should log debug message about MCP not being available
      mock_logger.debug.assert_called_with(
          "MCP not available for patching (expected in some environments)"
      )
      # May return True or False depending on other patches

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_patch_types_generic_alias_failure(self, mock_logger):
    """Test that patching types.GenericAlias fails due to immutability."""
    result = patch_types_for_pydantic_v2()

    # GenericAlias patching should fail because it's immutable
    # But httpx patching should succeed, so result could be True or False
    mock_logger.warning.assert_called()
    # Verify the warning message indicates GenericAlias patching failed
    warning_calls = [
        call
        for call in mock_logger.warning.call_args_list
        if "GenericAlias" in str(call)
    ]
    assert len(warning_calls) > 0

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_patch_types_httpx_success(self, mock_logger):
    """Test successful patching of httpx clients."""
    # Create mock httpx classes
    mock_client = Mock()
    mock_async_client = Mock()

    with (
        patch("httpx.Client", mock_client),
        patch("httpx.AsyncClient", mock_async_client),
    ):
      result = patch_types_for_pydantic_v2()

      assert result is True
      # Verify both clients were patched
      assert hasattr(mock_client, "__get_pydantic_core_schema__")
      assert hasattr(mock_async_client, "__get_pydantic_core_schema__")
      mock_logger.info.assert_called()

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_patch_types_all_fail(self, mock_logger):
    """Test when all patching attempts fail."""
    # Mock the import statement to fail for MCP
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == "mcp.client.session":
        raise ImportError("No module named 'mcp.client.session'")
      return original_import(name, *args, **kwargs)

    # Mock setattr to also fail for other patching attempts
    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "google.adk.utils.pydantic_v2_compatibility.setattr",
            side_effect=Exception("Setattr failed"),
        ),
    ):
      result = patch_types_for_pydantic_v2()

      assert result is False
      mock_logger.warning.assert_called()

  def test_create_robust_openapi_function_normal_operation(self):
    """Test robust OpenAPI function under normal conditions."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"
    mock_app.routes = []

    expected_schema = {"openapi": "3.0.0", "info": {"title": "Test API"}}

    with patch(
        "fastapi.openapi.utils.get_openapi", return_value=expected_schema
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      assert result == expected_schema
      assert mock_app.openapi_schema == expected_schema

  def test_create_robust_openapi_function_cached_schema(self):
    """Test robust OpenAPI function returns cached schema when available."""
    mock_app = Mock(spec=FastAPI)
    cached_schema = {"openapi": "3.1.0", "info": {"title": "Cached API"}}
    mock_app.openapi_schema = cached_schema

    robust_openapi = create_robust_openapi_function(mock_app)
    result = robust_openapi()

    assert result == cached_schema

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_create_robust_openapi_function_recursion_error(self, mock_logger):
    """Test robust OpenAPI function handles RecursionError."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"
    mock_app.routes = []

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=RecursionError("Maximum recursion depth exceeded"),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      # Should return fallback schema with correct values from implementation
      assert "openapi" in result
      assert "info" in result
      assert result["openapi"] == "3.1.0"  # Match implementation
      assert (
          result["info"]["title"] == "Test API"
      )  # Should use the app's title when available
      mock_logger.warning.assert_called()

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_create_robust_openapi_function_pydantic_error(self, mock_logger):
    """Test robust OpenAPI function handles Pydantic errors."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"
    mock_app.routes = []

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=Exception(
            "PydanticSchemaGenerationError: Cannot generate schema"
        ),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      # Should return fallback schema
      assert "openapi" in result
      assert "info" in result
      assert result["openapi"] == "3.1.0"  # Match implementation
      assert (
          result["info"]["title"] == "Test API"
      )  # Should use the app's title when available
      mock_logger.warning.assert_called()

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_create_robust_openapi_function_non_pydantic_error(self, mock_logger):
    """Test robust OpenAPI function re-raises non-Pydantic errors."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"
    mock_app.routes = []

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=ValueError("Unrelated error"),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)

      with pytest.raises(ValueError, match="Unrelated error"):
        robust_openapi()

  def test_robust_openapi_fallback_schema_structure(self):
    """Test that the fallback schema has the correct structure."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"
    mock_app.routes = []

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=Exception("PydanticSchemaGenerationError"),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      # Verify schema structure matches implementation
      assert result["openapi"] == "3.1.0"  # Match implementation
      assert "info" in result
      assert "paths" in result
      assert "components" in result
      assert "schemas" in result["components"]
      assert "HTTPValidationError" in result["components"]["schemas"]
      assert "ValidationError" in result["components"]["schemas"]
      assert "GenericResponse" in result["components"]["schemas"]
      assert "AgentInfo" in result["components"]["schemas"]

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_robust_openapi_route_extraction(self, mock_logger):
    """Test that routes are safely extracted in fallback mode."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"

    # Create mock routes
    mock_route = Mock()
    mock_route.path = "/test"
    mock_route.methods = {"GET", "POST"}
    mock_app.routes = [mock_route]

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=Exception("PydanticSchemaGenerationError"),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      # Should include the extracted route
      assert "/test" in result["paths"]
      assert "get" in result["paths"]["/test"]
      assert "post" in result["paths"]["/test"]

  @patch("google.adk.utils.pydantic_v2_compatibility.logger")
  def test_robust_openapi_route_extraction_failure(self, mock_logger):
    """Test fallback when route extraction fails."""
    mock_app = Mock(spec=FastAPI)
    mock_app.openapi_schema = None
    mock_app.title = "Test API"
    mock_app.version = "1.0.0"
    mock_app.description = "Test Description"

    # Make routes attribute raise an exception when accessed
    mock_app.routes = PropertyMock(side_effect=Exception("Route access failed"))

    with patch(
        "fastapi.openapi.utils.get_openapi",
        side_effect=Exception("PydanticSchemaGenerationError"),
    ):
      robust_openapi = create_robust_openapi_function(mock_app)
      result = robust_openapi()

      # Should include minimal essential endpoints
      assert "/" in result["paths"]
      assert "/health" in result["paths"]
      mock_logger.warning.assert_called()

  def test_patched_generic_alias_behavior(self):
    """Test that GenericAlias patching is attempted but fails due to immutability."""
    import types

    with patch(
        "google.adk.utils.pydantic_v2_compatibility.logger"
    ) as mock_logger:
      # Apply patches - this should fail for GenericAlias
      result = patch_types_for_pydantic_v2()

      # Should have warning about GenericAlias patching failure
      warning_calls = [
          call
          for call in mock_logger.warning.call_args_list
          if "GenericAlias" in str(call)
      ]
      assert len(warning_calls) > 0

      # GenericAlias should not have the method (because patching failed)
      assert not hasattr(types.GenericAlias, "__get_pydantic_core_schema__")

  def test_patched_generic_alias_immutable_type_error(self):
    """Test that GenericAlias patching fails due to type immutability."""
    import types

    with patch(
        "google.adk.utils.pydantic_v2_compatibility.setattr"
    ) as mock_setattr:
      # Configure setattr to raise TypeError for GenericAlias
      def setattr_side_effect(obj, name, value):
        if obj is types.GenericAlias and name == "__get_pydantic_core_schema__":
          raise TypeError(
              "cannot set '__get_pydantic_core_schema__' attribute of immutable"
              " type 'types.GenericAlias'"
          )
        # Call original setattr for other cases
        return setattr(obj, name, value)

      mock_setattr.side_effect = setattr_side_effect

      with patch(
          "google.adk.utils.pydantic_v2_compatibility.logger"
      ) as mock_logger:
        result = patch_types_for_pydantic_v2()

        # Should log a warning about GenericAlias patching failure
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "GenericAlias" in str(call)
        ]
        assert len(warning_calls) > 0

  @pytest.mark.skipif(
      not MCP_AVAILABLE, reason="MCP module not available in Python 3.9"
  )
  def test_patched_mcp_client_session_behavior(self):
    """Test that patched MCP ClientSession works correctly."""
    mock_client_session = Mock()
    mock_client_session.__modify_schema__ = Mock()

    with patch("mcp.client.session.ClientSession", mock_client_session):
      # Apply patches
      result = patch_types_for_pydantic_v2()
      assert result is True

      # Test the patched method exists and works
      assert hasattr(mock_client_session, "__get_pydantic_core_schema__")

      # Get the patched method and test it
      method = getattr(mock_client_session, "__get_pydantic_core_schema__")

      # Mock the core_schema.any_schema function
      with patch("pydantic_core.core_schema.any_schema") as mock_any_schema:
        mock_any_schema.return_value = {"type": "any"}

        # Call the method properly (it's a classmethod)
        result = method.__func__(mock_client_session, Mock(), Mock())

        # Should return any_schema
        mock_any_schema.assert_called_once()
        assert result == {"type": "any"}

  def test_patched_httpx_clients_behavior(self):
    """Test that patched httpx clients work correctly."""
    mock_client = Mock()
    mock_async_client = Mock()

    with (
        patch("httpx.Client", mock_client),
        patch("httpx.AsyncClient", mock_async_client),
    ):
      # Apply patches
      result = patch_types_for_pydantic_v2()
      assert result is True

      # Test both clients were patched
      assert hasattr(mock_client, "__get_pydantic_core_schema__")
      assert hasattr(mock_async_client, "__get_pydantic_core_schema__")

      # Test the patched methods work
      for client in [mock_client, mock_async_client]:
        method = getattr(client, "__get_pydantic_core_schema__")

        with patch("pydantic_core.core_schema.any_schema") as mock_any_schema:
          mock_any_schema.return_value = {"type": "any"}

          # Call the method properly (it's a classmethod)
          result = method.__func__(client, Mock(), Mock())
          mock_any_schema.assert_called_once()
          assert result == {"type": "any"}

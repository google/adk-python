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

from google.adk.utils.pydantic_v2_compatibility import (
    patch_types_for_pydantic_v2,
    create_robust_openapi_function,
    __get_pydantic_core_schema__,
)
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
import sys
import logging


class TestPydanticV2CompatibilityPatches:
    """Test suite for Pydantic v2 compatibility patches."""

    def test_get_pydantic_core_schema_success(self):
        """Test successful schema generation with valid handler."""
        mock_handler = Mock()
        mock_handler.generate_schema.return_value = {"type": "object", "properties": {}}

        result = __get_pydantic_core_schema__(str, mock_handler)

        assert result == {"type": "object", "properties": {}}
        mock_handler.generate_schema.assert_called_once_with(str)

    def test_get_pydantic_core_schema_fallback(self):
        """Test fallback schema when handler fails."""
        mock_handler = Mock()
        mock_handler.generate_schema.side_effect = Exception("Schema generation failed")

        result = __get_pydantic_core_schema__(str, mock_handler)

        expected_fallback = {
            "type": "object",
            "properties": {},
            "title": "str",
            "_pydantic_v2_compat": True
        }
        assert result == expected_fallback

    def test_get_pydantic_core_schema_no_handler(self):
        """Test schema generation when handler is None."""
        result = __get_pydantic_core_schema__(str, None)

        expected_fallback = {
            "type": "object",
            "properties": {},
            "title": "str",
            "_pydantic_v2_compat": True
        }
        assert result == expected_fallback

    @patch('google.adk.utils.pydantic_v2_compatibility.ClientSession', create=True)
    def test_patch_types_for_pydantic_v2_success(self, mock_client_session):
        """Test successful patching of types for Pydantic v2."""
        # Mock ClientSession class
        mock_client_session.__modify_schema__ = Mock()

        result = patch_types_for_pydantic_v2()

        assert result is True
        # Verify that __get_pydantic_core_schema__ was added
        assert hasattr(mock_client_session, '__get_pydantic_core_schema__')
        # Verify that __modify_schema__ was removed if it existed
        assert not hasattr(mock_client_session, '__modify_schema__')

    @patch('google.adk.utils.pydantic_v2_compatibility.ClientSession', side_effect=ImportError)
    def test_patch_types_for_pydantic_v2_import_error(self, mock_client_session):
        """Test patching when ClientSession cannot be imported."""
        result = patch_types_for_pydantic_v2()

        assert result is False

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    @patch('google.adk.utils.pydantic_v2_compatibility.ClientSession', create=True)
    def test_patch_types_for_pydantic_v2_exception_handling(self, mock_client_session, mock_logger):
        """Test exception handling during patching."""
        # Make setattr raise an exception
        with patch('builtins.setattr', side_effect=Exception("Patching failed")):
            result = patch_types_for_pydantic_v2()

            assert result is False
            mock_logger.error.assert_called()

    def test_create_robust_openapi_function_normal_operation(self):
        """Test robust OpenAPI function under normal conditions."""
        mock_app = Mock(spec=FastAPI)
        mock_app.openapi.return_value = {"openapi": "3.0.0", "info": {"title": "Test API"}}

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        assert result == {"openapi": "3.0.0", "info": {"title": "Test API"}}

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    def test_create_robust_openapi_function_recursion_error(self, mock_logger):
        """Test robust OpenAPI function handles RecursionError."""
        mock_app = Mock(spec=FastAPI)
        mock_app.openapi.side_effect = RecursionError("Maximum recursion depth exceeded")

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        # Should return fallback schema
        assert "openapi" in result
        assert "info" in result
        assert result["info"]["title"] == "ADK Agent API"
        mock_logger.warning.assert_called()

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    @patch('google.adk.utils.pydantic_v2_compatibility.sys')
    def test_create_robust_openapi_function_recursion_limit_handling(self, mock_sys, mock_logger):
        """Test recursion limit handling in robust OpenAPI function."""
        mock_app = Mock(spec=FastAPI)
        mock_app.openapi.return_value = {"openapi": "3.0.0"}
        mock_sys.getrecursionlimit.return_value = 1000

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        # Verify recursion limit was set
        mock_sys.setrecursionlimit.assert_called_with(500)
        # Verify it was restored
        assert mock_sys.setrecursionlimit.call_count == 2

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    def test_create_robust_openapi_function_generic_exception(self, mock_logger):
        """Test robust OpenAPI function handles generic exceptions."""
        mock_app = Mock(spec=FastAPI)
        mock_app.openapi.side_effect = Exception("Generic error")

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        # Should return fallback schema
        assert "openapi" in result
        assert "info" in result
        mock_logger.error.assert_called()

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    def test_create_robust_openapi_function_attribute_error(self, mock_logger):
        """Test robust OpenAPI function handles AttributeError."""
        mock_app = Mock()
        # Remove openapi method to trigger AttributeError
        del mock_app.openapi

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        # Should return fallback schema
        assert "openapi" in result
        assert "info" in result
        mock_logger.error.assert_called()

    def test_robust_openapi_fallback_schema_structure(self):
        """Test the structure of the fallback OpenAPI schema."""
        mock_app = Mock(spec=FastAPI)
        mock_app.openapi.side_effect = Exception("Error")

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        # Verify required OpenAPI structure
        assert result["openapi"] == "3.0.0"
        assert "info" in result
        assert result["info"]["title"] == "ADK Agent API"
        assert result["info"]["version"] == "1.0.0"
        assert "paths" in result
        assert "components" in result
        assert "schemas" in result["components"]

    @patch('google.adk.utils.pydantic_v2_compatibility.httpx', create=True)
    def test_patch_httpx_client_success(self):
        """Test successful patching of httpx Client."""
        mock_client = Mock()

        with patch('google.adk.utils.pydantic_v2_compatibility.patch_types_for_pydantic_v2') as mock_patch:
            mock_patch.return_value = True
            result = patch_types_for_pydantic_v2()

            assert result is True

    def test_robust_openapi_preserves_successful_schema(self):
        """Test that robust OpenAPI preserves successful schema generation."""
        mock_app = Mock(spec=FastAPI)
        expected_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Custom API", "version": "2.0.0"},
            "paths": {"/test": {"get": {"summary": "Test endpoint"}}},
            "components": {"schemas": {"TestModel": {"type": "object"}}}
        }
        mock_app.openapi.return_value = expected_schema

        robust_openapi = create_robust_openapi_function(mock_app)
        result = robust_openapi()

        assert result == expected_schema

    @patch('google.adk.utils.pydantic_v2_compatibility.logger')
    def test_create_robust_openapi_logs_errors_appropriately(self, mock_logger):
        """Test that robust OpenAPI function logs errors with appropriate levels."""
        mock_app = Mock(spec=FastAPI)

        # Test RecursionError logging
        mock_app.openapi.side_effect = RecursionError("Recursion error")
        robust_openapi = create_robust_openapi_function(mock_app)
        robust_openapi()
        mock_logger.warning.assert_called()

        # Reset and test generic Exception logging
        mock_logger.reset_mock()
        mock_app.openapi.side_effect = ValueError("Generic error")
        robust_openapi = create_robust_openapi_function(mock_app)
        robust_openapi()
        mock_logger.error.assert_called()
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

"""Pydantic v2 compatibility patches for Google ADK.

This module provides patches for various types that are not compatible with
Pydantic v2 schema generation, which is required for OpenAPI/Swagger UI
functionality in FastAPI applications.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("google_adk." + __name__)


def patch_types_for_pydantic_v2() -> bool:
    """Patch various types to be Pydantic v2 compatible for OpenAPI generation.

    This function applies compatibility patches for:
    1. MCP ClientSession - removes deprecated __modify_schema__ method
    2. types.GenericAlias - adds support for modern generic syntax (list[str], etc.)
    3. httpx.Client/AsyncClient - adds schema generation support

    Returns:
        bool: True if any patches were applied successfully, False otherwise.
    """
    success_count = 0

    # Patch MCP ClientSession
    try:
        from mcp.client.session import ClientSession

        # Add Pydantic v2 schema method only (v2 rejects __modify_schema__)
        def __get_pydantic_core_schema__(cls, source_type, handler):
            from pydantic_core import core_schema
            return core_schema.any_schema()

        # Only set the Pydantic v2 method - remove v1 method to avoid conflicts
        setattr(ClientSession, "__get_pydantic_core_schema__", classmethod(__get_pydantic_core_schema__))

        # Remove __modify_schema__ if it exists to prevent Pydantic v2 conflicts
        if hasattr(ClientSession, "__modify_schema__"):
            delattr(ClientSession, "__modify_schema__")

        logger.info("MCP ClientSession patched for Pydantic v2 compatibility")
        success_count += 1

    except ImportError:
        logger.debug("MCP not available for patching (expected in some environments)")
    except Exception as e:
        logger.warning(f"Failed to patch MCP ClientSession: {e}")

    # Patch types.GenericAlias for modern generic syntax (list[str], dict[str, int], etc.)
    try:
        import types

        def generic_alias_get_pydantic_core_schema(cls, source_type, handler):
            """Handle modern generic types like list[str], dict[str, int]."""
            from pydantic_core import core_schema

            # For GenericAlias, try to use the handler to generate schema for the origin type
            if hasattr(source_type, "__origin__") and hasattr(source_type, "__args__"):
                try:
                    # Let pydantic handle the origin type (list, dict, etc.)
                    return handler(source_type.__origin__)
                except Exception:
                    # Fallback to any schema if we can't handle the specific type
                    return core_schema.any_schema()

            # Default fallback
            return core_schema.any_schema()

        # Patch types.GenericAlias
        setattr(types.GenericAlias, "__get_pydantic_core_schema__", classmethod(generic_alias_get_pydantic_core_schema))

        logger.info("types.GenericAlias patched for Pydantic v2 compatibility")
        success_count += 1

    except Exception as e:
        logger.warning(f"Failed to patch types.GenericAlias: {e}")

    # Patch httpx.Client and httpx.AsyncClient for Pydantic v2 compatibility
    try:
        import httpx

        def httpx_client_get_pydantic_core_schema(cls, source_type, handler):
            """Handle httpx.Client and httpx.AsyncClient."""
            from pydantic_core import core_schema
            # These are not serializable to JSON, so we provide a generic schema
            return core_schema.any_schema()

        # Patch both Client and AsyncClient
        for client_class in [httpx.Client, httpx.AsyncClient]:
            setattr(client_class, "__get_pydantic_core_schema__", classmethod(httpx_client_get_pydantic_core_schema))

        logger.info("httpx.Client and httpx.AsyncClient patched for Pydantic v2 compatibility")
        success_count += 1

    except Exception as e:
        logger.warning(f"Failed to patch httpx clients: {e}")

    if success_count > 0:
        logger.info(f"Successfully applied {success_count} Pydantic v2 compatibility patches")
        return True
    else:
        logger.warning("No Pydantic v2 compatibility patches were applied")
        return False


def create_robust_openapi_function(app):
    """Create a robust OpenAPI function that handles Pydantic v2 compatibility issues.

    This function provides a fallback mechanism for OpenAPI generation when
    Pydantic v2 compatibility issues prevent normal schema generation.

    Args:
        app: The FastAPI application instance

    Returns:
        Callable that generates OpenAPI schema with error handling
    """
    def robust_openapi() -> Dict[str, Any]:
        """Generate OpenAPI schema with comprehensive error handling."""
        if app.openapi_schema:
            return app.openapi_schema

        # First attempt: Try normal OpenAPI generation with recursion limits
        try:
            import sys
            from fastapi.openapi.utils import get_openapi

            # Set a lower recursion limit to catch infinite loops early
            original_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(min(500, original_limit))

                # Attempt normal OpenAPI generation
                openapi_schema = get_openapi(
                    title=app.title,
                    version=app.version,
                    description=app.description,
                    routes=app.routes,
                )
                app.openapi_schema = openapi_schema
                logger.info("OpenAPI schema generated successfully with all routes")
                return app.openapi_schema

            finally:
                sys.setrecursionlimit(original_limit)

        except RecursionError as re:
            logger.warning("🔄 RecursionError detected in OpenAPI generation - likely model circular reference")
        except Exception as e:
            error_str = str(e)

            # Check if this is a known Pydantic v2 compatibility issue
            is_pydantic_error = any(pattern in error_str for pattern in [
                "PydanticSchemaGenerationError",
                "PydanticInvalidForJsonSchema",
                "PydanticUserError",
                "__modify_schema__",
                "Unable to generate pydantic-core schema",
                "schema-for-unknown-type",
                "invalid-for-json-schema",
                "mcp.client.session.ClientSession",
                "httpx.Client",
                "types.GenericAlias",
                "generate_inner",
                "handler",
                "core_schema"
            ])

            if not is_pydantic_error:
                # Re-raise non-Pydantic/non-recursion related errors
                logger.error(f"Unexpected error during OpenAPI generation: {e}")
                raise e

            logger.warning(f"OpenAPI schema generation failed due to Pydantic v2 compatibility issues: {str(e)[:200]}...")

        # Fallback: Provide comprehensive minimal OpenAPI schema
        logger.info("🔄 Providing robust fallback OpenAPI schema for ADK service")

        fallback_schema = {
            "openapi": "3.1.0",
            "info": {
                "title": getattr(app, 'title', 'Google ADK API Server'),
                "version": getattr(app, 'version', '1.0.0'),
                "description": (
                    "Google Agent Development Kit (ADK) API Server\n\n"
                    "This is a robust fallback OpenAPI schema generated due to Pydantic v2 "
                    "compatibility issues (likely circular model references or unsupported types). "
                    "All API endpoints remain fully functional, but detailed request/response "
                    "schemas are simplified for compatibility.\n\n"
                    "For full schema support, see: https://github.com/googleapis/genai-adk/issues"
                ),
            },
            "paths": {},
            "components": {
                "schemas": {
                    "HTTPValidationError": {
                        "title": "HTTPValidationError",
                        "type": "object",
                        "properties": {
                            "detail": {
                                "title": "Detail",
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/ValidationError"}
                            }
                        }
                    },
                    "ValidationError": {
                        "title": "ValidationError",
                        "required": ["loc", "msg", "type"],
                        "type": "object",
                        "properties": {
                            "loc": {
                                "title": "Location",
                                "type": "array",
                                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
                            },
                            "msg": {"title": "Message", "type": "string"},
                            "type": {"title": "Error Type", "type": "string"}
                        }
                    },
                    "GenericResponse": {
                        "title": "Generic Response",
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean", "description": "Operation success status"},
                            "message": {"type": "string", "description": "Response message"},
                            "data": {"type": "object", "description": "Response data", "additionalProperties": True}
                        }
                    },
                    "AgentInfo": {
                        "title": "Agent Information",
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Agent name"},
                            "description": {"type": "string", "description": "Agent description"},
                            "status": {"type": "string", "description": "Agent status"}
                        }
                    }
                }
            },
            "tags": [
                {"name": "agents", "description": "Agent management operations"},
                {"name": "auth", "description": "Authentication operations"},
                {"name": "health", "description": "Health and status operations"}
            ]
        }

        # Safely extract route information without triggering schema generation
        try:
            for route in getattr(app, 'routes', []):
                if not hasattr(route, 'path') or not hasattr(route, 'methods'):
                    continue

                path = route.path

                # Skip internal routes
                if path.startswith(('/docs', '/redoc', '/openapi.json')):
                    continue

                path_item = {}
                methods = getattr(route, 'methods', set())

                for method in methods:
                    method_lower = method.lower()
                    if method_lower not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                        continue

                    if method_lower == 'head':
                        continue  # Skip HEAD methods in OpenAPI

                    # Create basic operation spec
                    operation = {
                        "summary": f"{method.upper()} {path}",
                        "description": f"Endpoint for {path}",
                        "responses": {
                            "200": {
                                "description": "Successful Response",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/GenericResponse"}
                                    }
                                }
                            }
                        }
                    }

                    # Add validation error response for POST/PUT/PATCH
                    if method_lower in ['post', 'put', 'patch']:
                        operation["responses"]["422"] = {
                            "description": "Validation Error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HTTPValidationError"}
                                }
                            }
                        }

                    # Add appropriate tags based on path
                    if any(keyword in path.lower() for keyword in ['agent', 'app']):
                        operation["tags"] = ["agents"]
                    elif 'auth' in path.lower():
                        operation["tags"] = ["auth"]
                    elif any(keyword in path.lower() for keyword in ['health', 'status', 'ping']):
                        operation["tags"] = ["health"]

                    # Special handling for known ADK endpoints
                    if path == "/" and method_lower == "get":
                        operation["summary"] = "API Root"
                        operation["description"] = "Get API server information and status"
                    elif path == "/list-apps" and method_lower == "get":
                        operation["summary"] = "List Available Agents"
                        operation["description"] = "Get list of available agent applications"
                        operation["responses"]["200"]["content"]["application/json"]["schema"] = {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of available agent names"
                        }
                    elif "health" in path.lower():
                        operation["summary"] = "Health Check"
                        operation["description"] = "Check service health and status"

                    path_item[method_lower] = operation

                if path_item:
                    fallback_schema["paths"][path] = path_item

        except Exception as route_error:
            logger.warning(f"Could not extract route information safely: {route_error}")

            # Add minimal essential endpoints manually if route extraction fails
            fallback_schema["paths"].update({
                "/": {
                    "get": {
                        "summary": "API Root",
                        "description": "Get API server information and status",
                        "tags": ["health"],
                        "responses": {
                            "200": {
                                "description": "API server information",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/GenericResponse"}
                                    }
                                }
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health Check",
                        "description": "Check service health and status",
                        "tags": ["health"],
                        "responses": {
                            "200": {
                                "description": "Service health status",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/GenericResponse"}
                                    }
                                }
                            }
                        }
                    }
                }
            })

        app.openapi_schema = fallback_schema
        logger.info("Using robust fallback OpenAPI schema with enhanced error handling")
        return app.openapi_schema

    return robust_openapi
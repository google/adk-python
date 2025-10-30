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

from __future__ import annotations

import argparse
import json
import logging
from typing import Any
from typing import Dict
from typing import List

# Google API client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logger = logging.getLogger("google_adk." + __name__)


class GoogleApiToOpenApiConverter:
  """Converts Google API Discovery documents to OpenAPI v3 format."""

  def __init__(self, api_name: str, api_version: str):
    """Initialize the converter with the API name and version.

    Args:
        api_name: The name of the Google API (e.g., "calendar")
        api_version: The version of the API (e.g., "v3")
    """
    self._api_name = api_name
    self._api_version = api_version
    self._google_api_resource = None
    self._google_api_spec = None
    self._openapi_spec = {
        "openapi": "3.0.0",
        "info": {},
        "servers": [],
        "paths": {},
        "components": {"schemas": {}, "securitySchemes": {}},
    }

  def fetch_google_api_spec(self) -> None:
    """Fetches the Google API specification using discovery service."""
    try:
      logger.info(
          "Fetching Google API spec for %s %s",
          self._api_name,
          self._api_version,
      )
      # Build a resource object for the specified API
      self._google_api_resource = build(self._api_name, self._api_version)

      # Access the underlying API discovery document
      self._google_api_spec = self._google_api_resource._rootDesc

      if not self._google_api_spec:
        raise ValueError("Failed to retrieve API specification")

      logger.info("Successfully fetched %s API specification", self._api_name)
    except HttpError as e:
      logger.error("HTTP Error: %s", e)
      raise
    except Exception as e:
      logger.error("Error fetching API spec: %s", e)
      raise

  def convert(self) -> Dict[str, Any]:
    """Convert the Google API spec to OpenAPI v3 format.

    Returns:
        Dict containing the converted OpenAPI v3 specification
    """
    if not self._google_api_spec:
      self.fetch_google_api_spec()

    # Convert basic API information
    self._convert_info()

    # Convert server information
    self._convert_servers()

    # Convert authentication/authorization schemes
    self._convert_security_schemes()

    # Convert schemas (models)
    self._convert_schemas()

    # Convert endpoints/paths
    self._convert_resources(self._google_api_spec.get("resources", {}))

    # Convert top-level methods, if any
    self._convert_methods(self._google_api_spec.get("methods", {}), "/")

    return self._openapi_spec

  def _convert_info(self) -> None:
    """Convert basic API information."""
    self._openapi_spec["info"] = {
        "title": self._google_api_spec.get("title", f"{self._api_name} API"),
        "description": self._google_api_spec.get("description", ""),
        "version": self._google_api_spec.get("version", self._api_version),
        "contact": {},
        "termsOfService": self._google_api_spec.get("documentationLink", ""),
    }

    # Add documentation links if available
    docs_link = self._google_api_spec.get("documentationLink")
    if docs_link:
      self._openapi_spec["externalDocs"] = {
          "description": "API Documentation",
          "url": docs_link,
      }

  def _convert_servers(self) -> None:
    """Convert server information."""
    base_url = self._google_api_spec.get(
        "rootUrl", ""
    ) + self._google_api_spec.get("servicePath", "")

    # Remove trailing slash if present
    if base_url.endswith("/"):
      base_url = base_url[:-1]

    self._openapi_spec["servers"] = [{
        "url": base_url,
        "description": f"{self._api_name} {self._api_version} API",
    }]

  def _convert_security_schemes(self) -> None:
    """Convert authentication and authorization schemes."""
    auth = self._google_api_spec.get("auth", {})
    oauth2 = auth.get("oauth2", {})

    if oauth2:
      # Handle OAuth2
      scopes = oauth2.get("scopes", {})
      formatted_scopes = {}

      for scope, scope_info in scopes.items():
        formatted_scopes[scope] = scope_info.get("description", "")

      self._openapi_spec["components"]["securitySchemes"]["oauth2"] = {
          "type": "oauth2",
          "description": "OAuth 2.0 authentication",
          "flows": {
              "authorizationCode": {
                  "authorizationUrl": (
                      "https://accounts.google.com/o/oauth2/auth"
                  ),
                  "tokenUrl": "https://oauth2.googleapis.com/token",
                  "scopes": formatted_scopes,
              }
          },
      }

    # Add API key authentication (most Google APIs support this)
    self._openapi_spec["components"]["securitySchemes"]["apiKey"] = {
        "type": "apiKey",
        "in": "query",
        "name": "key",
        "description": "API key for accessing this API",
    }

    # Create global security requirement
    self._openapi_spec["security"] = [
        {"oauth2": list(formatted_scopes.keys())} if oauth2 else {},
        {"apiKey": []},
    ]

  def _convert_schemas(self) -> None:
    """Convert schema definitions (models)."""
    schemas = self._google_api_spec.get("schemas", {})

    for schema_name, schema_def in schemas.items():
      converted_schema = self._convert_schema_object(schema_def)
      self._openapi_spec["components"]["schemas"][
          schema_name
      ] = converted_schema

  def _convert_schema_object(
      self, schema_def: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Recursively convert a Google API schema object to OpenAPI schema.

    Args:
        schema_def: Google API schema definition

    Returns:
        Converted OpenAPI schema object
    """
    result = {}

    # Convert the type
    if "type" in schema_def:
      gtype = schema_def["type"]
      if gtype == "object":
        result["type"] = "object"

        # Handle properties
        if "properties" in schema_def:
          result["properties"] = {}
          for prop_name, prop_def in schema_def["properties"].items():
            result["properties"][prop_name] = self._convert_schema_object(
                prop_def
            )

        # Handle required fields
        required_fields = []
        for prop_name, prop_def in schema_def.get("properties", {}).items():
          if prop_def.get("required", False):
            required_fields.append(prop_name)
        if required_fields:
          result["required"] = required_fields

      elif gtype == "array":
        result["type"] = "array"
        if "items" in schema_def:
          result["items"] = self._convert_schema_object(schema_def["items"])

      elif gtype == "any":
        # OpenAPI doesn't have direct "any" type
        # Use oneOf with multiple options as alternative
        result["oneOf"] = [
            {"type": "object"},
            {"type": "array"},
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]

      else:
        # Handle other primitive types
        result["type"] = gtype

    # Handle references
    if "$ref" in schema_def:
      ref = schema_def["$ref"]
      # Google refs use "#" at start, OpenAPI uses "#/components/schemas/"
      if ref.startswith("#"):
        ref = ref.replace("#", "#/components/schemas/")
      else:
        ref = "#/components/schemas/" + ref
      result["$ref"] = ref

    # Handle format
    if "format" in schema_def:
      result["format"] = schema_def["format"]

    # Handle enum values
    if "enum" in schema_def:
      result["enum"] = schema_def["enum"]

    # Handle description
    if "description" in schema_def:
      result["description"] = schema_def["description"]

    # Handle pattern
    if "pattern" in schema_def:
      result["pattern"] = schema_def["pattern"]

    # Handle default value
    if "default" in schema_def:
      result["default"] = schema_def["default"]

    return result

  def _convert_resources(
      self, resources: Dict[str, Any], parent_path: str = ""
  ) -> None:
    """Recursively convert all resources and their methods.

    Args:
        resources: Dictionary of resources from the Google API spec
        parent_path: The parent path prefix for nested resources
    """
    for resource_name, resource_data in resources.items():
      # Process methods for this resource
      resource_path = f"{parent_path}/{resource_name}"
      methods = resource_data.get("methods", {})
      self._convert_methods(methods, resource_path)

      # Process nested resources recursively
      nested_resources = resource_data.get("resources", {})
      if nested_resources:
        self._convert_resources(nested_resources, resource_path)

  def _convert_methods(
      self, methods: Dict[str, Any], resource_path: str
  ) -> None:
    """Convert methods for a specific resource path.

    Args:
        methods: Dictionary of methods from the Google API spec
        resource_path: The path of the resource these methods belong to
    """
    for method_name, method_data in methods.items():
      http_method = method_data.get("httpMethod", "GET").lower()

      # Determine the actual endpoint path
      # Google often has the format something like 'users.messages.list'
      # flatPath is preferred as it provides the actual path, while path
      # might contain variables like {+projectId}
      rest_path = method_data.get("flatPath", method_data.get("path", "/"))
      if not rest_path.startswith("/"):
        rest_path = "/" + rest_path

      path_params = self._extract_path_parameters(rest_path)

      # Create path entry if it doesn't exist
      if rest_path not in self._openapi_spec["paths"]:
        self._openapi_spec["paths"][rest_path] = {}

      # Add the operation for this method
      self._openapi_spec["paths"][rest_path][http_method] = (
          self._convert_operation(method_data, path_params)
      )

  def _extract_path_parameters(self, path: str) -> List[str]:
    """Extract path parameters from a URL path.

    Args:
        path: The URL path with path parameters

    Returns:
        List of parameter names
    """
    params = []
    segments = path.split("/")

    for segment in segments:
      # Google APIs often use {param} format for path parameters
      if segment.startswith("{") and segment.endswith("}"):
        param_name = segment[1:-1]
        params.append(param_name)

    return params

  def _convert_operation(
      self, method_data: Dict[str, Any], path_params: List[str]
  ) -> Dict[str, Any]:
    """Convert a Google API method to an OpenAPI operation.

    Args:
        method_data: Google API method data
        path_params: List of path parameter names

    Returns:
        OpenAPI operation object
    """
    operation = {
        "operationId": method_data.get("id", ""),
        "summary": method_data.get("description", ""),
        "description": method_data.get("description", ""),
        "parameters": [],
        "responses": {
            "200": {"description": "Successful operation"},
            "400": {"description": "Bad request"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden"},
            "404": {"description": "Not found"},
            "500": {"description": "Server error"},
        },
    }

    # Add path parameters
    for param_name in path_params:
      param = {
          "name": param_name,
          "in": "path",
          "required": True,
          "schema": {"type": "string"},
      }
      operation["parameters"].append(param)

    # Add query parameters
    for param_name, param_data in method_data.get("parameters", {}).items():
      # Skip parameters already included in path
      if param_name in path_params:
        continue

      param = {
          "name": param_name,
          "in": param_data.get("location", "query"),
          "description": param_data.get("description", ""),
          "required": param_data.get("required", False),
          "schema": self._convert_parameter_schema(param_data),
      }
      operation["parameters"].append(param)

    # Handle request body
    if "request" in method_data:
      request_ref = method_data.get("request", {}).get("$ref", "")
      if request_ref:
        if request_ref.startswith("#"):
          # Convert Google's reference format to OpenAPI format
          openapi_ref = request_ref.replace("#", "#/components/schemas/")
        else:
          openapi_ref = "#/components/schemas/" + request_ref
        operation["requestBody"] = {
            "description": "Request body",
            "content": {"application/json": {"schema": {"$ref": openapi_ref}}},
            "required": True,
        }

    # Handle response body
    if "response" in method_data:
      response_ref = method_data.get("response", {}).get("$ref", "")
      if response_ref:
        if response_ref.startswith("#"):
          # Convert Google's reference format to OpenAPI format
          openapi_ref = response_ref.replace("#", "#/components/schemas/")
        else:
          openapi_ref = "#/components/schemas/" + response_ref
        operation["responses"]["200"]["content"] = {
            "application/json": {"schema": {"$ref": openapi_ref}}
        }

    # Add scopes if available
    scopes = method_data.get("scopes", [])
    if scopes:
      # Add method-specific security requirement if different from global
      operation["security"] = [{"oauth2": scopes}]

    return operation

  def _convert_parameter_schema(
      self, param_data: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Convert a parameter definition to an OpenAPI schema.

    Args:
        param_data: Google API parameter data

    Returns:
        OpenAPI schema for the parameter
    """
    schema = {}

    # Convert type
    param_type = param_data.get("type", "string")
    schema["type"] = param_type

    # Handle enum values
    if "enum" in param_data:
      schema["enum"] = param_data["enum"]

    # Handle format
    if "format" in param_data:
      schema["format"] = param_data["format"]

    # Handle default value
    if "default" in param_data:
      schema["default"] = param_data["default"]

    # Handle pattern
    if "pattern" in param_data:
      schema["pattern"] = param_data["pattern"]

    return schema

  def save_openapi_spec(self, output_path: str) -> None:
    """Save the OpenAPI specification to a file.

    Args:
        output_path: Path where the OpenAPI spec should be saved
    """
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(self._openapi_spec, f, indent=2)
    logger.info("OpenAPI specification saved to %s", output_path)


def main():
  """Command line interface for the converter."""
  parser = argparse.ArgumentParser(
      description=(
          "Convert Google API Discovery documents to OpenAPI v3 specifications"
      )
  )
  parser.add_argument(
      "api_name", help="Name of the Google API (e.g., 'calendar')"
  )
  parser.add_argument("api_version", help="Version of the API (e.g., 'v3')")
  parser.add_argument(
      "--output",
      "-o",
      default="openapi_spec.json",
      help="Output file path for the OpenAPI specification",
  )

  args = parser.parse_args()

  try:
    # Create and run the converter
    converter = GoogleApiToOpenApiConverter(args.api_name, args.api_version)
    converter.convert()
    converter.save_openapi_spec(args.output)
    logger.info(
        "Successfully converted %s %s to OpenAPI v3",
        args.api_name,
        args.api_version,
    )
    logger.info("Output saved to %s", args.output)
  except Exception as e:
    logger.error("Conversion failed: %s", e)
    return 1

  return 0


if __name__ == "__main__":
  main()

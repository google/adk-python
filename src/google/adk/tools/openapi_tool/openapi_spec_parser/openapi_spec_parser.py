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

from __future__ import annotations

import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from fastapi.openapi.models import Operation
from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from ..._gemini_schema_util import _to_snake_case
from ..common.common import ApiParameter
from .operation_parser import OperationParser

# Valid JSON Schema types as per OpenAPI 3.0/3.1 specification.
#
# These are the only types accepted by Pydantic 2.11+ for Schema.type.
_VALID_SCHEMA_TYPES: Set[str] = frozenset({
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
})

_SCHEMA_CONTAINER_KEYS: Set[str] = frozenset({"schema", "schemas"})


class OperationEndpoint(BaseModel):
  base_url: str
  path: str
  method: str


class ParsedOperation(BaseModel):
  name: str
  description: str
  endpoint: OperationEndpoint
  operation: Operation
  parameters: List[ApiParameter]
  return_value: ApiParameter
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None
  additional_context: Optional[Any] = None


class OpenApiSpecParser:
  """Generates Python code, JSON schema, and callables for an OpenAPI operation.

  This class takes an OpenApiOperation object and provides methods to generate:
  1. A string representation of a Python function that handles the operation.
  2. A JSON schema representing the input parameters of the operation.
  3. A callable Python object (a function) that can execute the operation.
  """

  def parse(self, openapi_spec_dict: Dict[str, Any]) -> List[ParsedOperation]:
    """Extracts an OpenAPI spec dict into a list of ParsedOperation objects.

    ParsedOperation objects are further used for generating RestApiTool.

    Args:
        openapi_spec_dict: A dictionary representing the OpenAPI specification.

    Returns:
        A list of ParsedOperation objects.
    """

    openapi_spec_dict = self._resolve_references(openapi_spec_dict)
    openapi_spec_dict = self._sanitize_schema_types(openapi_spec_dict)
    operations = self._collect_operations(openapi_spec_dict)
    return operations

  def _sanitize_schema_types(
      self, openapi_spec: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Recursively sanitizes schema types in an OpenAPI specification.

    Pydantic 2.11+ strictly validates that schema types are one of:
    'array', 'boolean', 'integer', 'null', 'number', 'object', 'string'.

    External APIs (like Google Integration Connectors) may return schemas
    with non-standard types like 'Any'. This method removes or converts
    such invalid types to ensure compatibility.

    Args:
        openapi_spec: A dictionary representing the OpenAPI specification.

    Returns:
        A dictionary with invalid schema types removed or sanitized.
    """
    openapi_spec = copy.deepcopy(openapi_spec)

    def sanitize_type_field(schema_dict: Dict[str, Any]) -> None:
      if "type" not in schema_dict:
        return

      type_value = schema_dict["type"]
      if isinstance(type_value, str):
        normalized_type = type_value.lower()
        if normalized_type in _VALID_SCHEMA_TYPES:
          schema_dict["type"] = normalized_type
          return

        del schema_dict["type"]
        return

      if isinstance(type_value, list):
        valid_types = []
        for entry in type_value:
          if not isinstance(entry, str):
            continue

          normalized_entry = entry.lower()
          if normalized_entry not in _VALID_SCHEMA_TYPES:
            continue

          if normalized_entry not in valid_types:
            valid_types.append(normalized_entry)

        if valid_types:
          schema_dict["type"] = valid_types
        else:
          del schema_dict["type"]

    def sanitize_recursive(obj: Any, *, in_schema: bool) -> Any:
      if isinstance(obj, dict):
        if in_schema:
          sanitize_type_field(obj)

        # Recursively process all values in the dict
        for key, value in obj.items():
          obj[key] = sanitize_recursive(
              value,
              in_schema=in_schema or key in _SCHEMA_CONTAINER_KEYS,
          )
        return obj
      elif isinstance(obj, list):
        return [sanitize_recursive(item, in_schema=in_schema) for item in obj]
      else:
        return obj

    return sanitize_recursive(openapi_spec, in_schema=False)

  def _collect_operations(
      self, openapi_spec: Dict[str, Any]
  ) -> List[ParsedOperation]:
    """Collects operations from an OpenAPI spec."""
    operations = []

    # Taking first server url, or default to empty string if not present
    base_url = ""
    if openapi_spec.get("servers"):
      base_url = openapi_spec["servers"][0].get("url", "")

    # Get global security scheme (if any)
    global_scheme_name = None
    if openapi_spec.get("security"):
      # Use first scheme by default.
      scheme_names = list(openapi_spec["security"][0].keys())
      global_scheme_name = scheme_names[0] if scheme_names else None

    auth_schemes = openapi_spec.get("components", {}).get("securitySchemes", {})

    for path, path_item in openapi_spec.get("paths", {}).items():
      if path_item is None:
        continue

      for method in (
          "get",
          "post",
          "put",
          "delete",
          "patch",
          "head",
          "options",
          "trace",
      ):
        operation_dict = path_item.get(method)
        if operation_dict is None:
          continue

        # Append path-level parameters
        operation_dict["parameters"] = operation_dict.get(
            "parameters", []
        ) + path_item.get("parameters", [])

        # If operation ID is missing, assign an operation id based on path
        # and method
        if "operationId" not in operation_dict:
          temp_id = _to_snake_case(f"{path}_{method}")
          operation_dict["operationId"] = temp_id

        url = OperationEndpoint(base_url=base_url, path=path, method=method)
        operation = Operation.model_validate(operation_dict)
        operation_parser = OperationParser(operation)

        # Check for operation-specific auth scheme
        auth_scheme_name = operation_parser.get_auth_scheme_name()
        auth_scheme_name = (
            auth_scheme_name if auth_scheme_name else global_scheme_name
        )
        auth_scheme = (
            auth_schemes.get(auth_scheme_name) if auth_scheme_name else None
        )

        parsed_op = ParsedOperation(
            name=operation_parser.get_function_name(),
            description=operation.description or operation.summary or "",
            endpoint=url,
            operation=operation,
            parameters=operation_parser.get_parameters(),
            return_value=operation_parser.get_return_value(),
            auth_scheme=auth_scheme,
            auth_credential=None,  # Placeholder
            additional_context={},
        )
        operations.append(parsed_op)

    return operations

  def _resolve_references(self, openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolves all $ref references in an OpenAPI specification.

    Handles circular references correctly.

    Args:
        openapi_spec: A dictionary representing the OpenAPI specification.

    Returns:
        A dictionary representing the OpenAPI specification with all references
        resolved.
    """

    openapi_spec = copy.deepcopy(openapi_spec)  # Work on a copy
    resolved_cache = {}  # Cache resolved references

    def resolve_ref(ref_string, current_doc):
      """Resolves a single $ref string."""
      parts = ref_string.split("/")
      if parts[0] != "#":
        raise ValueError(f"External references not supported: {ref_string}")

      current = current_doc
      for part in parts[1:]:
        if part in current:
          current = current[part]
        else:
          return None  # Reference not found
      return current

    def recursive_resolve(obj, current_doc, seen_refs=None):
      """Recursively resolves references, handling circularity.

      Args:
          obj: The object to traverse.
          current_doc:  Document to search for refs.
          seen_refs: A set to track already-visited references (for circularity
            detection).

      Returns:
          The resolved object.
      """
      if seen_refs is None:
        seen_refs = set()  # Initialize the set if it's the first call

      if isinstance(obj, dict):
        if "$ref" in obj and isinstance(obj["$ref"], str):
          ref_string = obj["$ref"]

          # Check for circularity
          if ref_string in seen_refs and ref_string not in resolved_cache:
            # Circular reference detected! Return a *copy* of the object,
            # but *without* the $ref.  This breaks the cycle while
            # still maintaining the overall structure.
            return {k: v for k, v in obj.items() if k != "$ref"}

          seen_refs.add(ref_string)  # Add the reference to the set

          # Check if we have a cached resolved value
          if ref_string in resolved_cache:
            return copy.deepcopy(resolved_cache[ref_string])

          resolved_value = resolve_ref(ref_string, current_doc)
          if resolved_value is not None:
            # Recursively resolve the *resolved* value,
            # passing along the 'seen_refs' set
            resolved_value = recursive_resolve(
                resolved_value, current_doc, seen_refs
            )
            resolved_cache[ref_string] = resolved_value
            return copy.deepcopy(resolved_value)  # return the cached result
          else:
            return obj  # return original if no resolved value.

        else:
          new_dict = {}
          for key, value in obj.items():
            new_dict[key] = recursive_resolve(value, current_doc, seen_refs)
          return new_dict

      elif isinstance(obj, list):
        return [recursive_resolve(item, current_doc, seen_refs) for item in obj]
      else:
        return obj

    return recursive_resolve(openapi_spec, openapi_spec)

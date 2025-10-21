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

"""ADK AgentConfig schema query tool for dynamic schema information access."""

from typing import Any
from typing import Dict
from typing import Optional

from ..utils import load_agent_config_schema


async def query_schema(
    query_type: str,
    component: Optional[str] = None,
    field_path: Optional[str] = None,
) -> Dict[str, Any]:
  """Dynamically query ADK AgentConfig schema for specific information.

  This tool provides on-demand access to ADK AgentConfig schema details without
  embedding
  the full schema in context. It's designed for "query" mode where
  agents need specific schema information without the memory overhead
  of the complete schema.

  Args:
    query_type: Type of schema query to perform. Supported values: - "overview":
      Get high-level schema structure and main properties - "component": Get
      detailed info about a specific top-level component - "field": Get details
      about a specific field using dot notation - "properties": Get flat list of
      all available properties
    component: Component name to explore (required for "component" query_type).
              Examples: "name", "instruction", "tools", "model", "memory"
    field_path: Dot-separated path to specific field (required for "field"
      query_type).
               Examples: "tools.function_tool.function_path", "model.name"

  Returns:
    Dict containing schema exploration results:
      Always included:
        - query_type: type of query performed
        - success: bool indicating if exploration succeeded

      Success cases vary by query_type:
        overview: schema title, description, main properties list
        component: component details, nested properties, type info
        field: field traversal path, type, description, constraints
        properties: complete flat property list with types

      Error cases only (success=False):
        - error: descriptive error message
        - supported_queries: list of valid query types and usage

  Examples:
    Get schema overview:
      result = await query_schema("overview")

    Explore tools component:
      result = await query_schema("component", component="tools")

    Get specific field details:
      result = await query_schema("field", field_path="model.name")
  """
  try:
    schema = load_agent_config_schema(raw_format=False)

    if query_type == "overview":
      return _get_schema_overview(schema)
    elif query_type == "component" and component:
      return _get_component_details(schema, component)
    elif query_type == "field" and field_path:
      return _get_field_details(schema, field_path)
    elif query_type == "properties":
      return _get_all_properties(schema)
    else:
      return {
          "error": (
              f"Invalid query_type '{query_type}' or missing required"
              " parameters"
          ),
          "supported_queries": [
              "overview - Get high-level schema structure",
              (
                  "component - Get details for specific component (requires"
                  " component parameter)"
              ),
              (
                  "field - Get details for specific field (requires field_path"
                  " parameter)"
              ),
              "properties - Get all available properties",
          ],
      }

  except Exception as e:
    return {"error": f"Schema exploration failed: {str(e)}"}


def _get_schema_overview(schema: Dict[str, Any]) -> Dict[str, Any]:
  """Get high-level overview of schema structure."""
  overview = {
      "title": schema.get("title", "ADK Agent Configuration"),
      "description": schema.get("description", ""),
      "schema_version": schema.get("$schema", ""),
      "main_properties": [],
  }

  properties = schema.get("properties", {})
  for prop_name, prop_details in properties.items():
    overview["main_properties"].append({
        "name": prop_name,
        "type": prop_details.get("type", "unknown"),
        "description": prop_details.get("description", ""),
        "required": prop_name in schema.get("required", []),
    })

  return overview


def _get_component_details(
    schema: Dict[str, Any], component: str
) -> Dict[str, Any]:
  """Get detailed information about a specific component."""
  properties = schema.get("properties", {})

  if component not in properties:
    return {
        "error": f"Component '{component}' not found",
        "available_components": list(properties.keys()),
    }

  component_schema = properties[component]

  result = {
      "component": component,
      "type": component_schema.get("type", "unknown"),
      "description": component_schema.get("description", ""),
      "required": component in schema.get("required", []),
  }

  # Add nested properties if it's an object
  if component_schema.get("type") == "object":
    nested_props = component_schema.get("properties", {})
    result["properties"] = {}
    for prop_name, prop_details in nested_props.items():
      result["properties"][prop_name] = {
          "type": prop_details.get("type", "unknown"),
          "description": prop_details.get("description", ""),
          "required": prop_name in component_schema.get("required", []),
      }

  # Add array item details if it's an array
  if component_schema.get("type") == "array":
    items = component_schema.get("items", {})
    result["items"] = {
        "type": items.get("type", "unknown"),
        "description": items.get("description", ""),
    }
    if items.get("type") == "object":
      result["items"]["properties"] = items.get("properties", {})

  return result


def _get_field_details(
    schema: Dict[str, Any], field_path: str
) -> Dict[str, Any]:
  """Get details for a specific field using dot notation."""
  path_parts = field_path.split(".")
  current = schema.get("properties", {})

  result = {"field_path": field_path, "path_traversal": []}

  for i, part in enumerate(path_parts):
    if not isinstance(current, dict) or part not in current:
      return {
          "error": f"Field path '{field_path}' not found at '{part}'",
          "traversed": ".".join(path_parts[:i]),
          "available_at_level": (
              list(current.keys()) if isinstance(current, dict) else []
          ),
      }

    field_info = current[part]
    result["path_traversal"].append({
        "field": part,
        "type": field_info.get("type", "unknown"),
        "description": field_info.get("description", ""),
    })

    # Navigate deeper based on type
    if field_info.get("type") == "object":
      current = field_info.get("properties", {})
    elif (
        field_info.get("type") == "array"
        and field_info.get("items", {}).get("type") == "object"
    ):
      current = field_info.get("items", {}).get("properties", {})
    else:
      # End of navigable path
      result["final_field"] = field_info
      break

  return result


def _get_all_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
  """Get a flat list of all properties in the schema."""
  properties = {}

  def extract_properties(obj: Dict[str, Any], prefix: str = ""):
    if not isinstance(obj, dict):
      return

    for key, value in obj.items():
      if key == "properties" and isinstance(value, dict):
        for prop_name, prop_details in value.items():
          full_path = f"{prefix}.{prop_name}" if prefix else prop_name
          properties[full_path] = {
              "type": prop_details.get("type", "unknown"),
              "description": prop_details.get("description", ""),
          }

          # Recurse into object properties
          if prop_details.get("type") == "object":
            extract_properties(prop_details, full_path)
          # Recurse into array item properties
          elif (
              prop_details.get("type") == "array"
              and prop_details.get("items", {}).get("type") == "object"
          ):
            extract_properties(prop_details.get("items", {}), full_path)

  extract_properties(schema)

  return {"total_properties": len(properties), "properties": properties}

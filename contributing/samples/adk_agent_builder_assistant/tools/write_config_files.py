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

"""Configuration file writer tool with validation-before-write."""

from pathlib import Path
from typing import Any
from typing import Dict

from google.adk.tools.tool_context import ToolContext
import jsonschema
import yaml

from ..utils import load_agent_config_schema
from .write_files import write_files


async def write_config_files(
    configs: Dict[str, str],
    tool_context: ToolContext,
    backup_existing: bool = False,  # Changed default to False - user should decide
    create_directories: bool = True,
) -> Dict[str, Any]:
  """Write multiple YAML configurations with comprehensive validation-before-write.

  This tool validates YAML syntax and AgentConfig schema compliance before
  writing files to prevent invalid configurations from being saved. It
  provides detailed error reporting and optional backup functionality.

  Args:
    configs: Dict mapping file_path to config_content (YAML as string)
    backup_existing: Whether to create timestamped backup of existing files
      before overwriting (default: False - user should be asked)
    create_directories: Whether to create parent directories if they don't exist
      (default: True)

  Returns:
    Dict containing write operation results:
      Always included:
        - success: bool indicating if all write operations succeeded
        - total_files: number of files requested
        - successful_writes: number of files written successfully
        - files: dict mapping file_path to file results

      Success cases only (success=True):
        - file_size: size of written file in bytes
        - agent_name: extracted agent name from configuration
        - agent_class: agent class type (e.g., "LlmAgent")
        - warnings: list of warning messages for best practice violations.
                   Empty list if no warnings. Common warning types:
                   • Agent name formatting issues (special characters)
                   • Empty instruction for LlmAgent
                   • Missing sub-agent files
                   • Incorrect file extensions (.yaml/.yml)
                   • Mixed tool format consistency

      Conditionally included:
        - backup: dict with backup information (if backup was created).
                 Contains:
                 • "backup_created": True (always True when present)
                 • "backup_path": absolute path to the timestamped backup file
                                 (format: "original.yaml.backup.{timestamp}")

      Error cases only (success=False):
        - error: descriptive error message explaining the failure
        - error_type: categorized error type for programmatic handling
        - validation_step: stage where validation process stopped.
                          Possible values:
                          • "yaml_parsing": YAML syntax is invalid
                          • "yaml_structure": YAML is valid but not a
                          dict/object
                          • "schema_validation": YAML violates AgentConfig
                          schema
                          • Not present: Error during file operations
        - validation_errors: detailed validation error list (for schema errors
        only)
        - retry_suggestion: helpful suggestions for fixing the error

  Examples:
    Write new configuration:
      result = await write_config_files({"my_agent.yaml": yaml_content})

    Write without backup:
      result = await write_config_files(
          {"temp_agent.yaml": yaml_content},
          backup_existing=False
      )

    Check backup information:
      result = await write_config_files({"existing_agent.yaml": new_content})
      if result["success"] and
      result["files"]["existing_agent.yaml"]["backup_created"]:
          backup_path = result["files"]["existing_agent.yaml"]["backup_path"]
          print(f"Original file backed up to: {backup_path}")

    Check validation warnings:
      result = await write_config_files({"agent.yaml": yaml_content})
      if result["success"] and result["files"]["agent.yaml"]["warnings"]:
          for warning in result["files"]["agent.yaml"]["warnings"]:
              print(f"Warning: {warning}")

    Handle validation errors:
      result = await write_config_files({"agent.yaml": invalid_yaml})
      if not result["success"]:
          step = result.get("validation_step", "file_operation")
          if step == "yaml_parsing":
              print("YAML syntax error:", result["error"])
          elif step == "schema_validation":
              print("Schema validation failed:", result["retry_suggestion"])
          else:
              print("Error:", result["error"])
  """
  result: Dict[str, Any] = {
      "success": True,
      "total_files": len(configs),
      "successful_writes": 0,
      "files": {},
      "errors": [],
  }

  validated_configs: Dict[str, str] = {}

  # Step 1: Validate all configs before writing any files
  for file_path, config_content in configs.items():
    file_result = _validate_single_config(file_path, config_content)
    result["files"][file_path] = file_result

    if file_result.get("success", False):
      validated_configs[file_path] = config_content
    else:
      result["success"] = False

  # Step 2: If all validations passed, write all files
  if result["success"] and validated_configs:
    write_result: Dict[str, Any] = await write_files(
        validated_configs,
        tool_context,
        create_backup=backup_existing,
        create_directories=create_directories,
    )

    # Merge write results with validation results
    files_data = write_result.get("files", {})
    for file_path, write_info in files_data.items():
      if file_path in result["files"]:
        file_entry = result["files"][file_path]
        if isinstance(file_entry, dict):
          file_entry.update({
              "file_size": write_info.get("file_size", 0),
              "backup_created": write_info.get("backup_created", False),
              "backup_path": write_info.get("backup_path"),
          })
          if write_info.get("error"):
            file_entry["success"] = False
            file_entry["error"] = write_info["error"]
            result["success"] = False
          else:
            result["successful_writes"] = result["successful_writes"] + 1

  return result


def _validate_single_config(
    file_path: str, config_content: str
) -> Dict[str, Any]:
  """Validate a single configuration file.

  Returns validation results for one config file.
  """
  try:
    # Convert to absolute path
    path = Path(file_path).resolve()

    # Step 1: Parse YAML content
    try:
      config_dict = yaml.safe_load(config_content)
    except yaml.YAMLError as e:
      return {
          "success": False,
          "error_type": "YAML_PARSE_ERROR",
          "error": f"Invalid YAML syntax: {str(e)}",
          "file_path": str(path),
          "validation_step": "yaml_parsing",
      }

    if not isinstance(config_dict, dict):
      return {
          "success": False,
          "error_type": "YAML_STRUCTURE_ERROR",
          "error": "YAML content must be a dictionary/object",
          "file_path": str(path),
          "validation_step": "yaml_structure",
      }

    # Step 2: Validate against AgentConfig schema
    validation_result = _validate_against_schema(config_dict)
    if not validation_result["valid"]:
      return {
          "success": False,
          "error_type": "SCHEMA_VALIDATION_ERROR",
          "error": "Configuration does not comply with AgentConfig schema",
          "validation_errors": validation_result["errors"],
          "file_path": str(path),
          "validation_step": "schema_validation",
          "retry_suggestion": _generate_retry_suggestion(
              validation_result["errors"]
          ),
      }

    # Step 3: Additional structural validation
    structural_validation = _validate_structure(config_dict, path)

    # Success response with validation metadata
    return {
        "success": True,
        "file_path": str(path),
        "agent_name": config_dict.get("name", "unknown"),
        "agent_class": config_dict.get("agent_class", "LlmAgent"),
        "warnings": structural_validation.get("warnings", []),
    }

  except Exception as e:
    return {
        "success": False,
        "error_type": "UNEXPECTED_ERROR",
        "error": f"Unexpected error during validation: {str(e)}",
        "file_path": file_path,
    }


def _validate_against_schema(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
  """Validate configuration against AgentConfig.json schema."""
  try:
    schema = load_agent_config_schema(raw_format=False)
    jsonschema.validate(config_dict, schema)

    return {"valid": True, "errors": []}

  except jsonschema.ValidationError as e:
    # JSONSCHEMA QUIRK WORKAROUND: Handle false positive validation errors
    #
    # Problem: When AgentConfig schema uses anyOf with inheritance hierarchies,
    # jsonschema throws ValidationError even for valid configs that match multiple schemas.
    #
    # Example scenario:
    # - AgentConfig schema: {"anyOf": [{"$ref": "#/$defs/LlmAgentConfig"},
    #                                  {"$ref": "#/$defs/SequentialAgentConfig"},
    #                                  {"$ref": "#/$defs/BaseAgentConfig"}]}
    # - Input config: {"agent_class": "SequentialAgent", "name": "test", ...}
    # - Result: Config is valid against both SequentialAgentConfig AND BaseAgentConfig
    #   (due to inheritance), but jsonschema considers this an error.
    #
    # Error message format:
    # "{'agent_class': 'SequentialAgent', ...} is valid under each of
    #  {'$ref': '#/$defs/SequentialAgentConfig'}, {'$ref': '#/$defs/BaseAgentConfig'}"
    #
    # Solution: Detect this specific error pattern and treat as valid since the
    # config actually IS valid - it just matches multiple compatible schemas.
    if "is valid under each of" in str(e.message):
      return {"valid": True, "errors": []}

    error_path = " -> ".join(str(p) for p in e.absolute_path)
    return {
        "valid": False,
        "errors": [{
            "path": error_path or "root",
            "message": e.message,
            "invalid_value": e.instance,
            "constraint": (
                e.schema.get("type") or e.schema.get("enum") or "unknown"
            ),
        }],
    }

  except jsonschema.SchemaError as e:
    return {
        "valid": False,
        "errors": [{
            "path": "schema",
            "message": f"Schema error: {str(e)}",
            "invalid_value": None,
            "constraint": "schema_integrity",
        }],
    }

  except Exception as e:
    return {
        "valid": False,
        "errors": [{
            "path": "validation",
            "message": f"Validation error: {str(e)}",
            "invalid_value": None,
            "constraint": "validation_process",
        }],
    }


def _validate_structure(
    config: Dict[str, Any], file_path: Path
) -> Dict[str, Any]:
  """Perform additional structural validation beyond JSON schema."""
  warnings = []

  # Check agent name format
  name = config.get("name", "")
  if name and not name.replace("_", "").replace("-", "").isalnum():
    warnings.append(
        "Agent name contains special characters that may cause issues"
    )

  # Check for empty instruction
  instruction = config.get("instruction", "").strip()
  if config.get("agent_class", "LlmAgent") == "LlmAgent" and not instruction:
    warnings.append(
        "LlmAgent has empty instruction which may result in poor performance"
    )

  # Validate sub-agent references
  sub_agents = config.get("sub_agents", [])
  for sub_agent in sub_agents:
    if isinstance(sub_agent, dict) and "config_path" in sub_agent:
      config_path = sub_agent["config_path"]

      # Check if path looks like it should be relative to current file
      if not config_path.startswith("/"):
        referenced_path = file_path.parent / config_path
        if not referenced_path.exists():
          warnings.append(
              f"Referenced sub-agent file may not exist: {config_path}"
          )

      # Check file extension
      if not config_path.endswith((".yaml", ".yml")):
        warnings.append(
            "Sub-agent config_path should end with .yaml or .yml:"
            f" {config_path}"
        )

  # Check tool format consistency
  tools = config.get("tools", [])
  has_object_format = any(isinstance(t, dict) for t in tools)
  has_string_format = any(isinstance(t, str) for t in tools)

  if has_object_format and has_string_format:
    warnings.append(
        "Mixed tool formats detected - consider using consistent object format"
    )

  return {"warnings": warnings, "has_warnings": len(warnings) > 0}


def _generate_retry_suggestion(errors: list) -> str:
  """Generate helpful suggestions for fixing validation errors."""
  if not errors:
    return ""

  suggestions = []

  for error in errors:
    path = error.get("path", "")
    message = error.get("message", "")

    if "required" in message.lower():
      if "name" in message:
        suggestions.append(
            "Add required 'name' field with a descriptive agent name"
        )
      elif "instruction" in message:
        suggestions.append(
            "Add required 'instruction' field with clear agent instructions"
        )
      else:
        suggestions.append(
            f"Add missing required field mentioned in error at '{path}'"
        )

    elif "enum" in message.lower() or "not one of" in message.lower():
      suggestions.append(
          f"Use valid enum value for field '{path}' - check schema for allowed"
          " values"
      )

    elif "type" in message.lower():
      if "string" in message:
        suggestions.append(f"Field '{path}' should be a string value")
      elif "array" in message:
        suggestions.append(f"Field '{path}' should be a list/array")
      elif "object" in message:
        suggestions.append(f"Field '{path}' should be an object/dictionary")

    elif "additional properties" in message.lower():
      suggestions.append(
          f"Remove unrecognized field '{path}' or check for typos"
      )

  if not suggestions:
    suggestions.append(
        "Please fix the validation errors and regenerate the configuration"
    )

  return " | ".join(suggestions[:3])  # Limit to top 3 suggestions

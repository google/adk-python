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

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

from google.adk.tools.tool_context import ToolContext
import jsonschema
import yaml

from ..utils import load_agent_config_schema
from ..utils.resolve_root_directory import resolve_file_path
from .write_files import write_files

INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
PARSED_CONFIG_KEY = "_parsed_config"
WORKFLOW_AGENT_CLASSES = frozenset({
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
})
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CALLBACK_FIELD_NAMES = (
    "before_agent_callbacks",
    "after_agent_callbacks",
    "before_model_callbacks",
    "after_model_callbacks",
    "before_tool_callbacks",
    "after_tool_callbacks",
)


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
    backup_existing: Whether to create timest   amped backup of existing files
      before overwriting (default: False, User should decide)
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
        - target_file_path: normalized path used for writing the config
        - rename_applied: whether the file name was changed to match agent name
        - written_file_path: absolute path that was ultimately written

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

  validated_config_dicts: Dict[str, Dict[str, Any]] = {}
  normalized_path_to_original: Dict[str, str] = {}
  canonical_path_to_original: Dict[str, str] = {}
  rename_map: Dict[str, str] = {}

  session_state = None
  session = getattr(tool_context, "session", None)
  if session is not None:
    session_state = getattr(session, "state", None)
  project_folder_name: Optional[str] = None
  if session_state is not None:
    try:
      project_root = resolve_file_path(".", session_state)
      project_folder_name = project_root.name or None
    except Exception:
      project_folder_name = None

  # Step 1: Validate all configs before writing any files
  for file_path, config_content in configs.items():
    file_result = _validate_single_config(
        file_path, config_content, project_folder_name
    )
    result["files"][file_path] = file_result

    if file_result.get("success", False):
      parsed_config = file_result.pop(PARSED_CONFIG_KEY, None)
      if parsed_config is None:
        file_result["success"] = False
        file_result["error_type"] = "INTERNAL_VALIDATION_ERROR"
        file_result["error"] = "Failed to parse configuration content."
        result["success"] = False
        continue

      agent_name = file_result.get("agent_name")
      (
          target_path,
          rename_applied,
          sanitized_name,
          rename_warning,
      ) = _determine_target_file_path(file_path, agent_name)

      file_result["target_file_path"] = target_path
      file_result["rename_applied"] = rename_applied
      if rename_warning:
        warnings = file_result.get("warnings", [])
        warnings.append(rename_warning)
        file_result["warnings"] = warnings

      if rename_applied and sanitized_name and sanitized_name != agent_name:
        warnings = file_result.get("warnings", [])
        warnings.append(
            "Agent name normalized for filesystem compatibility:"
            f" '{agent_name}' -> '{sanitized_name}'"
        )
        file_result["warnings"] = warnings

      normalized_key = target_path
      if normalized_key in normalized_path_to_original:
        conflict_source = normalized_path_to_original[normalized_key]
        file_result["success"] = False
        file_result["error_type"] = "FILE_PATH_CONFLICT"
        file_result["error"] = (
            "Multiple agent configs target the same file path after"
            f" normalization: '{conflict_source}' and '{file_path}'"
        )
        result["success"] = False
        continue
      normalized_path_to_original[normalized_key] = file_path

      canonical_key = _canonical_path_key(normalized_key, session_state)
      if canonical_key in canonical_path_to_original:
        conflict_source = canonical_path_to_original[canonical_key]
        file_result["success"] = False
        file_result["error_type"] = "FILE_PATH_CONFLICT"
        file_result["error"] = (
            "Multiple agent configs resolve to the same file path after"
            f" normalization: '{conflict_source}' and '{file_path}'"
        )
        result["success"] = False
        continue
      canonical_path_to_original[canonical_key] = file_path

      if normalized_key != file_path:
        rename_map[file_path] = normalized_key

      validated_config_dicts[normalized_key] = parsed_config
    else:
      result["success"] = False

  if result["success"] and validated_config_dicts:
    if rename_map:
      reference_map = _build_reference_map(rename_map)
      for config_dict in validated_config_dicts.values():
        _update_sub_agent_references(config_dict, reference_map)

    validated_configs: Dict[str, str] = {}
    for normalized_path, config_dict in validated_config_dicts.items():
      validated_configs[normalized_path] = yaml.safe_dump(
          config_dict,
          sort_keys=False,
      )

    write_result: Dict[str, Any] = await write_files(
        validated_configs,
        tool_context,
        create_backup=backup_existing,
        create_directories=create_directories,
    )

    # Merge write results with validation results
    files_data = write_result.get("files", {})
    for written_path, write_info in files_data.items():
      canonical_written_key = _canonical_path_key(written_path, session_state)
      original_key = canonical_path_to_original.get(canonical_written_key)

      if original_key and original_key in result["files"]:
        file_entry = result["files"][original_key]
        if isinstance(file_entry, dict):
          file_entry.update({
              "file_size": write_info.get("file_size", 0),
              "backup_created": write_info.get("backup_created", False),
              "backup_path": write_info.get("backup_path"),
              "written_file_path": written_path,
          })
          if write_info.get("error"):
            file_entry["success"] = False
            file_entry["error"] = write_info["error"]
            result["success"] = False
          else:
            result["successful_writes"] = result["successful_writes"] + 1

  return result


def _build_reference_map(rename_map: Dict[str, str]) -> Dict[str, str]:
  """Build lookup for updating sub-agent config paths after renames."""
  reference_map: Dict[str, str] = {}
  for original, target in rename_map.items():
    original_path = Path(original)
    target_path = Path(target)

    candidates = {
        original: target,
        str(original_path): str(target_path),
        original_path.as_posix(): target_path.as_posix(),
        original_path.name: target_path.name,
    }

    # Ensure Windows-style separators are covered when running on POSIX.
    candidates.setdefault(
        str(original_path).replace("\\", "/"),
        str(target_path).replace("\\", "/"),
    )

    for candidate, replacement in candidates.items():
      reference_map[candidate] = replacement

  return reference_map


def _update_sub_agent_references(
    config_dict: Dict[str, Any], reference_map: Dict[str, str]
) -> None:
  """Update sub-agent config_path entries based on rename map."""
  if not reference_map:
    return

  sub_agents = config_dict.get("sub_agents")
  if not isinstance(sub_agents, list):
    return

  for sub_agent in sub_agents:
    if not isinstance(sub_agent, dict):
      continue

    config_path = sub_agent.get("config_path")
    if not isinstance(config_path, str):
      continue

    new_path = reference_map.get(config_path)
    if new_path is None:
      try:
        normalized = str(Path(config_path))
        new_path = reference_map.get(normalized)
      except (OSError, ValueError):
        normalized = None

    if new_path is None and normalized is not None:
      new_path = reference_map.get(Path(normalized).as_posix())

    if new_path is None:
      try:
        base_name = Path(config_path).name
        new_path = reference_map.get(base_name)
      except (OSError, ValueError):
        new_path = None

    if new_path:
      sub_agent["config_path"] = new_path


def _canonical_path_key(
    path: str, session_state: Optional[Dict[str, Any]]
) -> str:
  """Create a canonical absolute path string for consistent lookups."""
  try:
    resolved_path = resolve_file_path(path, session_state)
  except (OSError, ValueError, RuntimeError):
    resolved_path = Path(path)

  try:
    return str(resolved_path.resolve())
  except (OSError, RuntimeError):
    return str(resolved_path)


def _validate_single_config(
    file_path: str,
    config_content: str,
    project_folder_name: Optional[str] = None,
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
    # TODO: b/455645705 - Remove once the frontend performs these validations before calling
    # this tool.
    name_warning = _normalize_agent_name_field(config_dict, path)
    structural_validation = _validate_structure(config_dict, path)
    warnings = list(structural_validation.get("warnings", []))
    warnings.extend(_strip_workflow_agent_fields(config_dict))
    if name_warning:
      warnings.append(name_warning)
    name_validation_error = _require_valid_agent_name(config_dict, path)
    if name_validation_error is not None:
      return name_validation_error
    model_validation_error = _require_llm_agent_model(config_dict, path)
    if model_validation_error is not None:
      return model_validation_error
    project_scope_result = _enforce_project_scoped_references(
        config_dict, project_folder_name, path
    )
    warnings.extend(project_scope_result.get("warnings", []))
    project_scope_error = project_scope_result.get("error")
    if project_scope_error is not None:
      return project_scope_error

    # Success response with validation metadata
    return {
        "success": True,
        "file_path": str(path),
        "agent_name": config_dict.get("name", "unknown"),
        "agent_class": config_dict.get("agent_class", "LlmAgent"),
        "warnings": warnings,
        PARSED_CONFIG_KEY: config_dict,
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


def _generate_retry_suggestion(
    errors: Sequence[Mapping[str, Any]],
) -> str:
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


def _require_llm_agent_model(
    config: Dict[str, Any], file_path: Path
) -> Optional[Dict[str, Any]]:
  """Ensure every LlmAgent configuration declares a model."""
  agent_class = config.get("agent_class", "LlmAgent")
  if agent_class != "LlmAgent":
    return None

  model = config.get("model")
  if isinstance(model, str) and model.strip():
    return None

  agent_name = config.get("name", "unknown")
  return {
      "success": False,
      "error_type": "LLM_AGENT_MODEL_REQUIRED",
      "error": (
          f"LlmAgent '{agent_name}' in '{file_path}' must define a 'model' "
          "field. LlmAgents cannot rely on implicit defaults."
      ),
      "file_path": str(file_path),
      "validation_step": "structure_validation",
      "retry_suggestion": (
          "Add a 'model' field with the user-confirmed model "
          "(for example, 'model: gemini-2.5-flash')."
      ),
  }


def _require_valid_agent_name(
    config: Dict[str, Any], file_path: Path
) -> Optional[Dict[str, Any]]:
  """Ensure agent names are valid identifiers."""
  agent_name = config.get("name")
  if isinstance(agent_name, str) and IDENTIFIER_PATTERN.match(agent_name):
    return None

  return {
      "success": False,
      "error_type": "INVALID_AGENT_NAME",
      "error": (
          f"Found invalid agent name: `{agent_name}` in '{file_path}'. "
          "Names must start with a letter or underscore and contain only "
          "letters, digits, or underscores."
      ),
      "file_path": str(file_path),
      "validation_step": "structure_validation",
      "retry_suggestion": (
          "Rename the agent using only letters, digits, and underscores "
          "(e.g., 'Paper_Analyzer')."
      ),
  }


def _normalize_agent_name_field(
    config: Dict[str, Any], file_path: Path
) -> Optional[str]:
  """Normalize agent name to snake_case and update the config in-place."""
  agent_name = config.get("name")
  if not isinstance(agent_name, str):
    return None

  sanitized_name, normalization_warning = _sanitize_agent_name_for_filename(
      agent_name
  )
  if not sanitized_name:
    return normalization_warning

  if sanitized_name != agent_name:
    config["name"] = sanitized_name
    return (
        "Agent name normalized to snake_case in "
        f"'{file_path.name}': '{agent_name}' -> '{sanitized_name}'"
    )

  return normalization_warning


def _strip_workflow_agent_fields(config: Dict[str, Any]) -> List[str]:
  """Remove fields that workflow agents must not define."""
  warnings: List[str] = []
  agent_class = config.get("agent_class")
  if agent_class not in WORKFLOW_AGENT_CLASSES:
    return warnings

  removed_fields = []
  for field in ("model", "tools", "instruction"):
    if field in config:
      config.pop(field, None)
      removed_fields.append(field)

  if removed_fields:
    removed_fields_str = ", ".join(removed_fields)
    agent_name = config.get("name", "unknown")
    warnings.append(
        "Removed "
        f"{removed_fields_str}"
        f" from workflow agent '{agent_name}'. "
        "Workflow agents orchestrate sub-agents and must not define these "
        "fields."
    )

  return warnings


def _enforce_project_scoped_references(
    config: Dict[str, Any],
    project_folder_name: Optional[str],
    file_path: Path,
) -> Dict[str, Any]:
  """Ensure callback/tool references are scoped to the project package."""
  if not project_folder_name:
    return {"warnings": [], "error": None}

  prefix = f"{project_folder_name}."
  warnings: List[str] = []
  errors: List[str] = []

  def _normalize_reference_value(
      value: str, descriptor: str
  ) -> Tuple[str, List[str], List[str]]:
    local_warnings: List[str] = []
    local_errors: List[str] = []
    new_value = value

    if not isinstance(value, str) or "." not in value:
      return new_value, local_warnings, local_errors

    if value.startswith(prefix):
      return new_value, local_warnings, local_errors

    if value.lower().startswith(prefix.lower()):
      local_errors.append(
          f"{descriptor} '{value}' must use exact-case prefix '{prefix}'."
      )
      return new_value, local_warnings, local_errors

    if value.startswith("callbacks.") or value.startswith("tools."):
      new_value = prefix + value
      local_warnings.append(
          f"{descriptor} '{value}' updated to '{new_value}' to include project "
          "prefix."
      )
      return new_value, local_warnings, local_errors

    if ".callbacks." in value or ".tools." in value:
      local_errors.append(f"{descriptor} '{value}' must start with '{prefix}'.")

    return new_value, local_warnings, local_errors

  tools = config.get("tools")
  if isinstance(tools, list):
    for index, tool in enumerate(tools):
      if isinstance(tool, str):
        updated, local_warnings, local_errors = _normalize_reference_value(
            tool, "Tool reference"
        )
        if updated != tool:
          tools[index] = updated
        warnings.extend(local_warnings)
        errors.extend(local_errors)
      elif isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
          updated, local_warnings, local_errors = _normalize_reference_value(
              name, "Tool reference"
          )
          if updated != name:
            tool["name"] = updated
          warnings.extend(local_warnings)
          errors.extend(local_errors)

  for field_name in CALLBACK_FIELD_NAMES:
    callbacks_field = config.get(field_name)
    if not callbacks_field:
      continue

    items = (
        callbacks_field
        if isinstance(callbacks_field, list)
        else [callbacks_field]
    )

    for idx, item in enumerate(items):
      if isinstance(item, str):
        updated, local_warnings, local_errors = _normalize_reference_value(
            item, f"{field_name} entry"
        )
        if updated != item:
          if isinstance(callbacks_field, list):
            callbacks_field[idx] = updated
          else:
            config[field_name] = updated
        warnings.extend(local_warnings)
        errors.extend(local_errors)
      elif isinstance(item, dict):
        name = item.get("name")
        if isinstance(name, str):
          updated, local_warnings, local_errors = _normalize_reference_value(
              name, f"{field_name} entry"
          )
          if updated != name:
            item["name"] = updated
          warnings.extend(local_warnings)
          errors.extend(local_errors)

  if errors:
    return {
        "warnings": warnings,
        "error": {
            "success": False,
            "error_type": "PROJECT_REFERENCE_ERROR",
            "error": " | ".join(errors),
            "file_path": str(file_path),
            "retry_suggestion": (
                "Ensure all callback/tool references start with "
                f"'{prefix}' and that referenced directories contain "
                "__init__.py files (only for the package directories such as "
                "'callbacks/' or 'tools/') so they form importable packages."
            ),
        },
    }

  return {"warnings": warnings, "error": None}


def _determine_target_file_path(
    file_path: str, agent_name: Optional[str]
) -> Tuple[str, bool, Optional[str], Optional[str]]:
  """Determine desired file path based on agent name."""
  if not agent_name or not agent_name.strip():
    return file_path, False, None, None

  original_path = Path(file_path)

  # Preserve root_agent.yaml naming convention for root workflows.
  if original_path.stem == "root_agent":
    return file_path, False, None, None

  sanitized_name, sanitize_warning = _sanitize_agent_name_for_filename(
      agent_name
  )
  if not sanitized_name:
    return (
        file_path,
        False,
        None,
        (
            "Agent name could not be converted into a valid file name; original"
            " path preserved"
        ),
    )

  suffix = original_path.suffix or ".yaml"
  target_name = f"{sanitized_name}{suffix}"
  target_path = str(original_path.with_name(target_name))
  rename_applied = original_path.name != target_name

  return target_path, rename_applied, sanitized_name, sanitize_warning


def _to_snake_case(value: str) -> str:
  """Convert arbitrary text to snake_case."""
  value = value.strip()
  if not value:
    return ""

  value = re.sub(r"[\s\-]+", "_", value)
  value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
  value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
  value = re.sub(r"[^A-Za-z0-9_]", "_", value)
  value = re.sub(r"_+", "_", value)
  return value.lower().strip("_")


def _sanitize_agent_name_for_filename(
    agent_name: str,
) -> Tuple[str, Optional[str]]:
  """Sanitize agent name so it can be safely used as a filename."""
  trimmed_name = agent_name.strip()
  if not trimmed_name:
    return "", "Agent name is empty after trimming whitespace"

  snake_case = _to_snake_case(trimmed_name)
  if not snake_case:
    return "", "Agent name is empty after normalization"

  sanitized_chars = []
  replacements_made = snake_case != trimmed_name

  for char in snake_case:
    if char in INVALID_FILENAME_CHARACTERS:
      sanitized_chars.append("_")
      replacements_made = True
    elif char.isalnum() or char == "_":
      sanitized_chars.append(char)
    else:
      sanitized_chars.append("_")
      replacements_made = True

  sanitized_name = "".join(sanitized_chars)
  sanitized_name = re.sub(r"_+", "_", sanitized_name).strip("_")
  if not sanitized_name:
    return "", "Agent name is empty after removing unsupported characters"

  if sanitized_name[0].isdigit():
    sanitized_name = f"_{sanitized_name}"
    replacements_made = True

  warning = None
  if replacements_made:
    warning = (
        "Agent name normalized to snake_case: "
        f"'{agent_name}' -> '{sanitized_name}'"
    )

  return sanitized_name, warning

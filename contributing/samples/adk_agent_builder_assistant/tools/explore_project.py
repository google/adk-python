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

"""Project explorer tool for analyzing structure and suggesting file paths."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from google.adk.tools.tool_context import ToolContext

from ..utils.resolve_root_directory import resolve_file_path


async def explore_project(tool_context: ToolContext) -> Dict[str, Any]:
  """Analyze project structure and suggest optimal file paths for ADK agents.

  This tool performs comprehensive project analysis to understand the existing
  structure and recommend appropriate locations for new agent configurations,
  tools, and related files following ADK best practices.

  The tool automatically determines the project directory from session state.

  Returns:
    Dict containing analysis results with ALL PATHS RELATIVE TO PROJECT FOLDER:
      Always included:
        - success: bool indicating if exploration succeeded

      Success cases only (success=True):
        - project_info: dict with basic project metadata. Contains:
                       • "name": project directory name
                       • "absolute_path": full path to project root
                       • "is_empty": bool indicating if directory is empty
                       • "total_files": count of all files in project
                       • "total_directories": count of all subdirectories
                       • "has_python_files": bool indicating presence of .py
                       files
                       • "has_yaml_files": bool indicating presence of
                       .yaml/.yml files
                       • "has_tools_directory": bool indicating if tools/ exists
                       • "has_callbacks_directory": bool indicating if
                       callbacks/ exists
        - existing_configs: list of dicts for found YAML configuration files.
                           Each dict contains:
                           • "filename": name of the config file
                           • "relative_path": path relative to project folder
                           • "size": file size in bytes
                           • "is_valid_yaml": bool indicating if YAML parses
                           correctly
                           • "agent_name": extracted agent name (or None)
                           • "agent_class": agent class type (default:
                           "LlmAgent")
                           • "has_sub_agents": bool indicating if config has
                           sub_agents
                           • "has_tools": bool indicating if config has tools
        - directory_structure: dict with hierarchical project tree view
        - suggestions: dict with recommended paths for new components. Contains:
                      • "root_agent_configs": list of suggested main agent
                      filenames
                      • "sub_agent_patterns": list of naming pattern templates
                      • "directories": dict with tool/callback directory info
                      • "naming_examples": dict with example agent sets by
                      domain
        - conventions: dict with ADK naming and organization best practices

      Error cases only (success=False):
        - error: descriptive error message explaining the failure

  Examples:
    Basic project exploration:
      result = await explore_project(tool_context)

    Check project structure:
      if result["project_info"]["has_tools_directory"]:
          print("Tools directory already exists")

    Analyze existing configs:
      for config in result["existing_configs"]:
          if config["is_valid_yaml"]:
              print(f"Found agent: {config['agent_name']}")

    Get path suggestions:
      suggestions = result["suggestions"]["root_agent_configs"]
      directories = result["suggestions"]["directories"]["tools"]
  """
  try:
    # Resolve root directory using session state (use "." as current project directory)
    session_state = tool_context._invocation_context.session.state
    resolved_path = resolve_file_path(".", session_state)
    root_path = resolved_path.resolve()

    if not root_path.exists():
      return {
          "success": False,
          "error": f"Project directory does not exist: {root_path}",
      }

    if not root_path.is_dir():
      return {
          "success": False,
          "error": f"Path is not a directory: {root_path}",
      }

    # Analyze project structure
    project_info = _analyze_project_info(root_path)
    existing_configs = _find_existing_configs(root_path)
    directory_structure = _build_directory_tree(root_path)
    suggestions = _generate_path_suggestions(root_path, existing_configs)
    conventions = _get_naming_conventions()

    return {
        "success": True,
        "project_info": project_info,
        "existing_configs": existing_configs,
        "directory_structure": directory_structure,
        "suggestions": suggestions,
        "conventions": conventions,
    }

  except PermissionError:
    return {
        "success": False,
        "error": "Permission denied accessing project directory",
    }
  except Exception as e:
    return {
        "success": False,
        "error": f"Error exploring project: {str(e)}",
    }


def _analyze_project_info(root_path: Path) -> Dict[str, Any]:
  """Analyze basic project information."""
  info = {
      "name": root_path.name,
      "absolute_path": str(root_path),
      "is_empty": not any(root_path.iterdir()),
      "total_files": 0,
      "total_directories": 0,
      "has_python_files": False,
      "has_yaml_files": False,
      "has_tools_directory": False,
      "has_callbacks_directory": False,
  }

  try:
    for item in root_path.rglob("*"):
      if item.is_file():
        info["total_files"] += 1
        suffix = item.suffix.lower()

        if suffix == ".py":
          info["has_python_files"] = True
        elif suffix in [".yaml", ".yml"]:
          info["has_yaml_files"] = True

      elif item.is_dir():
        info["total_directories"] += 1

        if item.name == "tools" and item.parent == root_path:
          info["has_tools_directory"] = True
        elif item.name == "callbacks" and item.parent == root_path:
          info["has_callbacks_directory"] = True

  except Exception:
    # Continue with partial information if traversal fails
    pass

  return info


def _find_existing_configs(root_path: Path) -> List[Dict[str, Any]]:
  """Find existing YAML configuration files in the project."""
  configs = []

  try:
    # Look for YAML files in root directory (ADK convention)
    for yaml_file in root_path.glob("*.yaml"):
      if yaml_file.is_file():
        config_info = _analyze_config_file(yaml_file, root_path)
        configs.append(config_info)

    for yml_file in root_path.glob("*.yml"):
      if yml_file.is_file():
        config_info = _analyze_config_file(yml_file, root_path)
        configs.append(config_info)

    # Sort by name for consistent ordering
    configs.sort(key=lambda x: x["filename"])

  except Exception:
    # Return partial results if scanning fails
    pass

  return configs


def _analyze_config_file(config_path: Path, root_path: Path) -> Dict[str, Any]:
  """Analyze a single configuration file."""
  # Compute relative path from project root
  try:
    relative_path = config_path.relative_to(root_path)
  except ValueError:
    # Fallback if not relative to root_path
    relative_path = config_path.name

  info = {
      "filename": config_path.name,
      "relative_path": str(relative_path),
      "size": 0,
      "is_valid_yaml": False,
      "agent_name": None,
      "agent_class": None,
      "has_sub_agents": False,
      "has_tools": False,
  }

  try:
    info["size"] = config_path.stat().st_size

    # Try to parse YAML to extract basic info
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
      content = yaml.safe_load(f)

    if isinstance(content, dict):
      info["is_valid_yaml"] = True
      info["agent_name"] = content.get("name")
      info["agent_class"] = content.get("agent_class", "LlmAgent")
      info["has_sub_agents"] = bool(content.get("sub_agents"))
      info["has_tools"] = bool(content.get("tools"))

  except Exception:
    # File exists but couldn't be parsed
    pass

  return info


def _build_directory_tree(
    root_path: Path, max_depth: int = 3
) -> Dict[str, Any]:
  """Build a directory tree representation."""

  def build_tree_recursive(
      path: Path, current_depth: int = 0
  ) -> Dict[str, Any]:
    if current_depth > max_depth:
      return {"truncated": True}

    tree = {
        "name": path.name,
        "type": "directory" if path.is_dir() else "file",
        "path": str(path.relative_to(root_path)),
    }

    if path.is_dir():
      children = []
      try:
        for child in sorted(path.iterdir()):
          # Skip hidden files and common ignore patterns
          if not child.name.startswith(".") and child.name not in [
              "__pycache__",
              "node_modules",
          ]:
            children.append(build_tree_recursive(child, current_depth + 1))
        tree["children"] = children
      except PermissionError:
        tree["error"] = "Permission denied"
    else:
      tree["size"] = path.stat().st_size if path.exists() else 0

    return tree

  return build_tree_recursive(root_path)


def _generate_path_suggestions(
    root_path: Path, existing_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
  """Generate suggested file paths for new components."""

  # Suggest main agent names if none exist
  root_agent_suggestions = []
  if not any(
      config.get("agent_class") != "LlmAgent"
      or not config.get("has_sub_agents", False)
      for config in existing_configs
  ):
    root_agent_suggestions = [
        "root_agent.yaml",
    ]

  # Directory suggestions (relative paths)
  directories = {
      "tools": {
          "path": "tools",
          "exists": (root_path / "tools").exists(),
          "purpose": "Custom tool implementations",
          "example_files": [
              "custom_email.py",
              "database_connector.py",
          ],
      },
      "callbacks": {
          "path": "callbacks",
          "exists": (root_path / "callbacks").exists(),
          "purpose": "Custom callback functions",
          "example_files": ["logging.py", "security.py"],
      },
  }

  return {
      "root_agent_configs": root_agent_suggestions,
      "sub_agent_patterns": [
          "{purpose}_agent.yaml",
          "{domain}_{action}_agent.yaml",
          "{workflow_step}_agent.yaml",
      ],
      "directories": directories,
  }


def _get_naming_conventions() -> Dict[str, Any]:
  """Get ADK naming conventions and best practices."""
  return {
      "agent_files": {
          "format": "snake_case with .yaml extension",
          "examples": ["main_agent.yaml", "email_processor.yaml"],
          "location": "Root directory of the project",
          "avoid": ["camelCase.yaml", "spaces in names.yaml", "UPPERCASE.yaml"],
      },
      "agent_names": {
          "format": "snake_case, descriptive, no spaces",
          "examples": ["customer_service_coordinator", "email_classifier"],
          "avoid": ["Agent1", "my agent", "CustomerServiceAgent"],
      },
      "directory_structure": {
          "recommended": {
              "root": "All .yaml agent configuration files",
              "tools/": "Custom tool implementations (.py files)",
              "callbacks/": "Custom callback functions (.py files)",
          }
      },
  }

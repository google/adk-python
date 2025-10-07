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

"""Configuration file reader tool for existing YAML configs."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from google.adk.tools.tool_context import ToolContext
import yaml

from .read_files import read_files


async def read_config_files(
    file_paths: List[str], tool_context: ToolContext
) -> Dict[str, Any]:
  """Read multiple YAML configuration files and extract metadata.

  Args:
    file_paths: List of absolute or relative paths to YAML configuration files

  Returns:
    Dict containing:
      - success: bool indicating if all files were processed
      - total_files: number of files requested
      - successful_reads: number of files read successfully
      - files: dict mapping file_path to file analysis:
        - success: bool for this specific file
        - file_path: absolute path to the file
        - file_size: size of file in characters
        - line_count: number of lines in file
        - content: parsed YAML content as dict (success only)
        - agent_info: extracted agent metadata (success only)
        - sub_agents: list of referenced sub-agent files (success only)
        - tools: list of tools used by the agent (success only)
        - error: error message (failure only)
        - raw_yaml: original YAML string (parsing errors only)
      - errors: list of general error messages
  """
  # Read all files using the file_manager read_files tool
  read_result = await read_files(file_paths, tool_context)

  result = {
      "success": True,
      "total_files": len(file_paths),
      "successful_reads": 0,
      "files": {},
      "errors": [],
  }

  for file_path, file_info in read_result["files"].items():
    file_analysis = {
        "success": False,
        "file_path": file_path,
        "file_size": file_info.get("file_size", 0),
        "line_count": 0,
        "error": None,
    }

    # Check if file was read successfully
    if file_info.get("error"):
      file_analysis["error"] = file_info["error"]
      result["files"][file_path] = file_analysis
      result["success"] = False
      continue

    # Check if it's a YAML file
    path = Path(file_path)
    if path.suffix.lower() not in [".yaml", ".yml"]:
      file_analysis["error"] = f"File is not a YAML file: {file_path}"
      result["files"][file_path] = file_analysis
      result["success"] = False
      continue

    raw_yaml = file_info.get("content", "")
    file_analysis["line_count"] = len(raw_yaml.split("\n"))

    # Parse YAML
    try:
      content = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
      file_analysis["error"] = f"Invalid YAML syntax: {str(e)}"
      file_analysis["raw_yaml"] = raw_yaml
      result["files"][file_path] = file_analysis
      result["success"] = False
      continue

    if not isinstance(content, dict):
      file_analysis["error"] = "YAML content is not a valid object/dictionary"
      file_analysis["raw_yaml"] = raw_yaml
      result["files"][file_path] = file_analysis
      result["success"] = False
      continue

    # Extract agent metadata
    try:
      agent_info = _extract_agent_info(content)
      sub_agents = _extract_sub_agents(content)
      tools = _extract_tools(content)

      file_analysis.update({
          "success": True,
          "content": content,
          "agent_info": agent_info,
          "sub_agents": sub_agents,
          "tools": tools,
      })

      result["successful_reads"] += 1

    except Exception as e:
      file_analysis["error"] = f"Error extracting metadata: {str(e)}"
      result["success"] = False

    result["files"][file_path] = file_analysis

  return result


# Legacy functions removed - use read_config_files directly


def _extract_agent_info(content: Dict[str, Any]) -> Dict[str, Any]:
  """Extract basic agent information from configuration."""
  return {
      "name": content.get("name", "unknown"),
      "agent_class": content.get("agent_class", "LlmAgent"),
      "description": content.get("description", ""),
      "model": content.get("model", ""),
      "has_instruction": bool(content.get("instruction", "").strip()),
      "instruction_length": len(content.get("instruction", "")),
      "has_memory": bool(content.get("memory")),
      "has_state": bool(content.get("state")),
  }


def _extract_sub_agents(content: Dict[str, Any]) -> list:
  """Extract sub-agent references from configuration."""
  sub_agents = content.get("sub_agents", [])

  if not isinstance(sub_agents, list):
    return []

  extracted = []
  for sub_agent in sub_agents:
    if isinstance(sub_agent, dict):
      agent_ref = {
          "config_path": sub_agent.get("config_path", ""),
          "code": sub_agent.get("code", ""),
          "type": "config_path" if "config_path" in sub_agent else "code",
      }

      # Check if referenced file exists (for config_path refs)
      if agent_ref["config_path"]:
        agent_ref["file_exists"] = _check_file_exists(agent_ref["config_path"])

      extracted.append(agent_ref)
    elif isinstance(sub_agent, str):
      # Simple string reference
      extracted.append({
          "config_path": sub_agent,
          "code": "",
          "type": "config_path",
          "file_exists": _check_file_exists(sub_agent),
      })

  return extracted


def _extract_tools(content: Dict[str, Any]) -> list:
  """Extract tool information from configuration."""
  tools = content.get("tools", [])

  if not isinstance(tools, list):
    return []

  extracted = []
  for tool in tools:
    if isinstance(tool, dict):
      tool_info = {
          "name": tool.get("name", ""),
          "type": "object",
          "has_args": bool(tool.get("args")),
          "args_count": len(tool.get("args", [])),
          "raw": tool,
      }
    elif isinstance(tool, str):
      tool_info = {
          "name": tool,
          "type": "string",
          "has_args": False,
          "args_count": 0,
          "raw": tool,
      }
    else:
      continue

    extracted.append(tool_info)

  return extracted


def _check_file_exists(config_path: str) -> bool:
  """Check if a configuration file path exists."""
  try:
    if not config_path:
      return False

    path = Path(config_path)

    # If it's not absolute, check relative to current working directory
    if not path.is_absolute():
      # Try relative to current directory
      current_dir_path = Path.cwd() / config_path
      if current_dir_path.exists():
        return True

      # Try common agent directory patterns
      for potential_dir in [".", "./agents", "../agents"]:
        potential_path = Path(potential_dir) / config_path
        if potential_path.exists():
          return True

    return path.exists()

  except (OSError, ValueError):
    return False

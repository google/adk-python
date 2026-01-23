"""File system tools for the coding agent."""

from pathlib import Path
import random
from typing import Optional

# Global plan state
_current_plan: dict | None = None


def read_file(file_path: str) -> dict:
  """Read the contents of a file.

  Args:
      file_path: Path to the file to read (relative or absolute)

  Returns:
      A dictionary containing the file contents and metadata, or error info
  """
  try:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
      return {
          "error": "file_not_found",
          "message": f"File not found: {file_path}",
          "path": file_path,
      }

    if not path.is_file():
      return {
          "error": "not_a_file",
          "message": f"Path is not a file: {file_path}",
          "path": file_path,
      }

    try:
      with open(path, "r", encoding="utf-8") as f:
        content = f.read()

      return {
          "success": True,
          "path": str(path),
          "content": content,
          "size_bytes": path.stat().st_size,
          "lines": len(content.splitlines()),
      }
    except UnicodeDecodeError:
      # Try to read as binary if UTF-8 fails
      with open(path, "rb") as f:
        content_bytes = f.read()

      return {
          "success": True,
          "path": str(path),
          "content": f"<binary file, {len(content_bytes)} bytes>",
          "size_bytes": len(content_bytes),
          "lines": None,
          "is_binary": True,
      }
  except PermissionError as e:
    return {
        "error": "permission_denied",
        "message": f"Permission denied: {file_path}",
        "path": file_path,
    }
  except Exception as e:
    return {"error": "unexpected_error", "message": str(e), "path": file_path}


def write_file(file_path: str, content: str, create_dirs: bool = True) -> dict:
  """Write content to a file, creating it if it doesn't exist.

  Args:
      file_path: Path to the file to write (relative or absolute)
      content: The content to write to the file
      create_dirs: If True, create parent directories if they don't exist

  Returns:
      A dictionary with the operation result or error info
  """
  try:
    path = Path(file_path).expanduser().resolve()

    # Create parent directories if needed
    if create_dirs and not path.parent.exists():
      path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists
    existed = path.exists()

    # Write the file
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)

    return {
        "success": True,
        "path": str(path),
        "operation": "updated" if existed else "created",
        "size_bytes": path.stat().st_size,
        "lines": len(content.splitlines()),
    }
  except PermissionError:
    return {
        "error": "permission_denied",
        "message": f"Permission denied: {file_path}",
        "path": file_path,
    }
  except Exception as e:
    return {"error": "unexpected_error", "message": str(e), "path": file_path}


def update_file(file_path: str, old_text: str, new_text: str) -> dict:
  """Update a file by replacing old text with new text.

  Args:
      file_path: Path to the file to update (relative or absolute)
      old_text: The text to find and replace
      new_text: The text to replace with

  Returns:
      A dictionary with the operation result or error info
  """
  try:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
      return {
          "error": "file_not_found",
          "message": f"File not found: {file_path}",
          "path": file_path,
      }

    # Read current content
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()

    # Check if old_text exists
    if old_text not in content:
      return {
          "error": "text_not_found",
          "message": f"Text not found in file: {old_text[:50]}...",
          "path": str(path),
      }

    # Count occurrences
    occurrences = content.count(old_text)

    # Replace text
    new_content = content.replace(old_text, new_text)

    # Write back
    with open(path, "w", encoding="utf-8") as f:
      f.write(new_content)

    return {
        "success": True,
        "path": str(path),
        "operation": "updated",
        "replacements": occurrences,
        "size_bytes": path.stat().st_size,
        "lines": len(new_content.splitlines()),
    }
  except PermissionError:
    return {
        "error": "permission_denied",
        "message": f"Permission denied: {file_path}",
        "path": file_path,
    }
  except Exception as e:
    return {"error": "unexpected_error", "message": str(e), "path": file_path}


def list_directory(
    directory_path: str = ".",
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> dict:
  """List files and directories in a directory.

  Args:
      directory_path: Path to the directory to list (default: current directory)
      pattern: Optional glob pattern to filter files (e.g., "*.py", "**/*.txt")
      recursive: If True, list files recursively (default: False)

  Returns:
      A dictionary containing lists of files and directories or error info
  """
  try:
    path = Path(directory_path).expanduser().resolve()

    if not path.exists():
      return {
          "error": "directory_not_found",
          "message": f"Directory not found: {directory_path}",
          "path": directory_path,
      }

    if not path.is_dir():
      return {
          "error": "not_a_directory",
          "message": f"Path is not a directory: {directory_path}",
          "path": directory_path,
      }

    files = []
    directories = []

    if pattern:
      # Use glob with pattern
      glob_pattern = f"**/{pattern}" if recursive else pattern
      for item in path.glob(glob_pattern):
        rel_path = item.relative_to(path)
        if item.is_file():
          files.append(
              {"name": str(rel_path), "size_bytes": item.stat().st_size}
          )
        elif item.is_dir():
          directories.append(str(rel_path))
    else:
      # List all items
      if recursive:
        for item in path.rglob("*"):
          rel_path = item.relative_to(path)
          if item.is_file():
            files.append(
                {"name": str(rel_path), "size_bytes": item.stat().st_size}
            )
          elif item.is_dir():
            directories.append(str(rel_path))
      else:
        for item in path.iterdir():
          if item.is_file():
            files.append({"name": item.name, "size_bytes": item.stat().st_size})
          elif item.is_dir():
            directories.append(item.name)

    # Sort for consistent output
    files.sort(key=lambda x: x["name"])
    directories.sort()

    return {
        "success": True,
        "path": str(path),
        "files": files,
        "directories": directories,
        "file_count": len(files),
        "directory_count": len(directories),
    }
  except PermissionError:
    return {
        "error": "permission_denied",
        "message": f"Permission denied: {directory_path}",
        "path": directory_path,
    }
  except Exception as e:
    return {
        "error": "unexpected_error",
        "message": str(e),
        "path": directory_path,
    }


def print_affirming_message() -> str:
  """Prints a random affirming message."""
  messages = [
      "You're doing great!",
      "Keep up the excellent work!",
      "You're making fantastic progress!",
      "Awesome job!",
      "You're a star!",
      "Incredible effort!",
      "Keep shining!",
      "You're truly amazing!",
      "Well done!",
      "Fantastic work!",
  ]
  message = random.choice(messages)
  print(f"Affirmation: {message}")
  return message


# ============================================================================
# Planning Tools
# ============================================================================


def create_plan(title: str, tasks: list[str]) -> dict:
  """Create a new plan with a list of tasks.

  Args:
      title: The title/goal of the plan
      tasks: List of task descriptions

  Returns:
      The created plan with task statuses
  """
  global _current_plan

  _current_plan = {
      "title": title,
      "tasks": [
          {"id": i, "description": task, "completed": False}
          for i, task in enumerate(tasks)
      ],
  }

  return {
      "status": "plan_created",
      "title": _current_plan["title"],
      "total_tasks": len(_current_plan["tasks"]),
      "plan": _current_plan,
  }


def update_plan(task_id: int, completed: bool) -> dict:
  """Update the completion status of a task in the plan.

  Args:
      task_id: The ID of the task to update (0-based index)
      completed: Whether the task is completed

  Returns:
      Updated plan information or error info
  """
  global _current_plan

  if _current_plan is None:
    return {
        "error": "no_plan",
        "message": "No plan exists. Create a plan first using create_plan.",
    }

  if task_id < 0 or task_id >= len(_current_plan["tasks"]):
    return {
        "error": "invalid_task_id",
        "message": (
            f"Invalid task_id: {task_id}. Must be between 0 and"
            f" {len(_current_plan['tasks']) - 1}"
        ),
        "task_id": task_id,
        "valid_range": f"0-{len(_current_plan['tasks']) - 1}",
    }

  _current_plan["tasks"][task_id]["completed"] = completed

  completed_count = sum(1 for t in _current_plan["tasks"] if t["completed"])
  total_count = len(_current_plan["tasks"])

  return {
      "status": "plan_updated",
      "task_id": task_id,
      "task_description": _current_plan["tasks"][task_id]["description"],
      "completed": completed,
      "progress": f"{completed_count}/{total_count}",
      "plan": _current_plan,
  }


def reset_plan() -> dict:
  """Reset/clear the current plan.

  Returns:
      Confirmation of plan reset
  """
  global _current_plan

  had_plan = _current_plan is not None
  _current_plan = None

  return {"status": "plan_reset", "had_plan": had_plan}


def get_plan() -> dict:
  """Get the current plan.

  Returns:
      The current plan or None if no plan exists
  """
  global _current_plan

  if _current_plan is None:
    return {"status": "no_plan", "plan": None}

  completed_count = sum(1 for t in _current_plan["tasks"] if t["completed"])
  total_count = len(_current_plan["tasks"])

  return {
      "status": "plan_exists",
      "title": _current_plan["title"],
      "progress": f"{completed_count}/{total_count}",
      "plan": _current_plan,
  }

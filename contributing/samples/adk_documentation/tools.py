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

from datetime import datetime
import os
import subprocess
from subprocess import CompletedProcess
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from adk_documentation.settings import GITHUB_BASE_URL
from adk_documentation.utils import error_response
from adk_documentation.utils import get_paginated_request
from adk_documentation.utils import get_request
from adk_documentation.utils import patch_request
from adk_documentation.utils import post_request
import requests


def list_releases(repo_owner: str, repo_name: str) -> Dict[str, Any]:
  """Lists all releases for a repository.

  This function retrieves all releases and for each one, returns its ID,
  creation time, publication time, and associated tag name. It handles
  pagination to ensure all releases are fetched.

  Args:
      repo_owner: The name of the repository owner.
      repo_name: The name of the repository.

  Returns:
      A dictionary containing the status and a list of releases.
  """
  # The initial URL for the releases endpoint
  # per_page=100 is used to reduce the number of API calls
  url = (
      f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/releases?per_page=100"
  )

  try:
    all_releases_data = get_paginated_request(url)

    # Format the response to include only the requested fields
    formatted_releases = []
    for release in all_releases_data:
      formatted_releases.append({
          "id": release.get("id"),
          "tag_name": release.get("tag_name"),
          "created_at": release.get("created_at"),
          "published_at": release.get("published_at"),
      })

    return {"status": "success", "releases": formatted_releases}
  except requests.exceptions.HTTPError as e:
    return error_response(f"HTTP Error: {e}")
  except requests.exceptions.RequestException as e:
    return error_response(f"Request Error: {e}")


def get_changed_files_between_releases(
    repo_owner: str, repo_name: str, start_tag: str, end_tag: str
) -> Dict[str, Any]:
  """Gets changed files and their modifications between two release tags.

  Args:
      repo_owner: The name of the repository owner.
      repo_name: The name of the repository.
      start_tag: The older tag (base) for the comparison.
      end_tag: The newer tag (head) for the comparison.

  Returns:
      A dictionary containing the status and a list of changed files.
      Each file includes its name, status (added, removed, modified),
      and the patch/diff content.
  """
  # The 'basehead' parameter is specified as 'base...head'.
  url = f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/compare/{start_tag}...{end_tag}"

  try:
    comparison_data = get_request(url)

    # The API returns a 'files' key with the list of changed files.
    changed_files = comparison_data.get("files", [])

    # Extract just the information we need for a cleaner output
    formatted_files = []
    for file_data in changed_files:
      formatted_files.append({
          "relative_path": file_data.get("filename"),
          "status": file_data.get("status"),
          "additions": file_data.get("additions"),
          "deletions": file_data.get("deletions"),
          "changes": file_data.get("changes"),
          "patch": file_data.get(
              "patch", "No patch available."
          ),  # The diff content
      })
    return {"status": "success", "changed_files": formatted_files}
  except requests.exceptions.HTTPError as e:
    return error_response(f"HTTP Error: {e}")
  except requests.exceptions.RequestException as e:
    return error_response(f"Request Error: {e}")


def clone_or_pull_repo(
    repo_owner: str,
    repo_name: str,
    local_path: str,
) -> Dict[str, Any]:
  """Clones a GitHub repository to a local folder using owner and repo name.

  If the folder already exists and is a valid Git repository, it pulls the
  latest changes instead.

  Args:
      repo_owner: The username or organization that owns the repository.
      repo_name: The name of the repository.
      local_path: The local directory path where the repository should be cloned
        or updated.

  Returns:
      A dictionary indicating the status of the operation, output message, and
      the head commit hash.
  """
  repo_url = f"git@github.com:{repo_owner}/{repo_name}.git"

  try:
    # Check local path and decide to clone or pull
    if os.path.exists(local_path):
      git_dir_path = os.path.join(local_path, ".git")
      if os.path.isdir(git_dir_path):
        print(f"Repository exists at '{local_path}'. Pulling latest changes...")
        try:
          output = _get_pull(local_path)
        except subprocess.CalledProcessError as e:
          return error_response(f"git pull failed: {e.stderr}")
      else:
        return error_response(
            f"Path '{local_path}' exists but is not a Git repository."
        )
    else:
      print(f"Cloning from {repo_owner}/{repo_name} into '{local_path}'...")
      try:
        output = _get_clone(repo_url, local_path)
      except subprocess.CalledProcessError as e:
        return error_response(f"git clone failed: {e.stderr}")
    head_commit_sha = _find_head_commit_sha(local_path)
  except FileNotFoundError:
    return error_response("Error: 'git' command not found. Is Git installed?")
  except subprocess.TimeoutExpired as e:
    return error_response(f"Command timeout: {e}")
  except (subprocess.CalledProcessError, OSError, ValueError) as e:
    return error_response(f"An unexpected error occurred: {e}")

  return {
      "status": "success",
      "output": output,
      "head_commit_sha": head_commit_sha,
  }


def read_local_git_repo_file_content(file_path: str) -> Dict[str, Any]:
  """Reads the content of a specified file in a local Git repository.

  Args:
      file_path: The full, absolute path to the file.

  Returns:
      A dictionary containing the status, content of the file, and the head
      commit hash.
  """
  print(f"Attempting to read file from path: {file_path}")
  dir_path = os.path.dirname(file_path)
  head_commit_sha = _find_head_commit_sha(dir_path)

  try:
    # Open and read the file content
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()

      # Add line numbers to the content
      lines = content.splitlines()
      numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
      numbered_content = "\n".join(numbered_lines)

    return {
        "status": "success",
        "file_path": file_path,
        "content": numbered_content,
        "head_commit_sha": head_commit_sha,
    }
  except FileNotFoundError:
    return error_response(f"Error: File not found at {file_path}")
  except IOError as e:
    return error_response(f"An unexpected error occurred: {e}")


def list_directory_contents(directory_path: str) -> Dict[str, Any]:
  """Recursively lists all files and directories within a specified directory.

  Args:
      directory_path: The full, absolute path to the directory.

  Returns:
      A dictionary containing the status and a map where keys are directory
      paths relative to the initial directory_path, and values are lists of
      their contents.
      Returns an error message if the directory cannot be accessed.
  """
  print(
      f"Attempting to recursively list contents of directory: {directory_path}"
  )
  if not os.path.isdir(directory_path):
    return error_response(f"Error: Directory not found at {directory_path}")

  directory_map = {}
  try:
    for root, dirs, files in os.walk(directory_path):
      # Filter out hidden directories from traversal and from the result
      dirs[:] = [d for d in dirs if not d.startswith(".")]
      # Filter out hidden files
      non_hidden_files = [f for f in files if not f.startswith(".")]

      relative_path = os.path.relpath(root, directory_path)
      directory_map[relative_path] = dirs + non_hidden_files
    return {
        "status": "success",
        "directory_path": directory_path,
        "directory_map": directory_map,
    }
  except (IOError, OSError) as e:
    return error_response(f"An unexpected error occurred: {e}")


def search_local_git_repo(
    directory_path: str,
    pattern: str,
    extensions: Optional[List[str]] = None,
    ignored_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Searches a local Git repository for a pattern.

  Args:
      directory_path: The absolute path to the local Git repository.
      pattern: The search pattern (can be a simple string or regex for git
        grep).
      extensions: The list of file extensions to search, e.g. ["py", "md"]. If
        None, all extensions will be searched.
      ignored_dirs: The list of directories to ignore, e.g. ["tests"]. If None,
        no directories will be ignored.

  Returns:
      A dictionary containing the status, and a list of match details (relative
      file path to the directory_path, line number, content).
  """
  print(
      f"Attempting to search for pattern: {pattern} in directory:"
      f" {directory_path}, with extensions: {extensions}"
  )
  try:
    grep_process = _git_grep(directory_path, pattern, extensions, ignored_dirs)
    if grep_process.returncode > 1:
      return error_response(f"git grep failed: {grep_process.stderr}")

    matches = []
    if grep_process.stdout:
      for line in grep_process.stdout.strip().split("\n"):
        try:
          file_path, line_number_str, line_content = line.split(":", 2)
          matches.append({
              "file_path": file_path,
              "line_number": int(line_number_str),
              "line_content": line_content.strip(),
          })
        except ValueError:
          return error_response(
              f"Error: Failed to parse line: {line} from git grep output."
          )
    return {
        "status": "success",
        "matches": matches,
    }
  except FileNotFoundError:
    return error_response(f"Directory not found: {directory_path}")
  except subprocess.CalledProcessError as e:
    return error_response(f"git grep failed: {e.stderr}")
  except (IOError, OSError, ValueError) as e:
    return error_response(f"An unexpected error occurred: {e}")


def create_pull_request_from_changes(
    repo_owner: str,
    repo_name: str,
    local_path: str,
    base_branch: str,
    changes: Dict[str, str],
    commit_message: str,
    pr_title: str,
    pr_body: str,
) -> Dict[str, Any]:
  """Creates a new branch, applies file changes, commits, pushes, and creates a PR.

  Args:
      repo_owner: The username or organization that owns the repository.
      repo_name: The name of the repository.
      local_path: The local absolute path to the cloned repository.
      base_branch: The name of the branch to merge the changes into (e.g.,
        "main").
      changes: A dictionary where keys are file paths relative to the repo root
        and values are the new and full content for those files.
      commit_message: The message for the git commit.
      pr_title: The title for the pull request.
      pr_body: The body/description for the pull request.

  Returns:
      A dictionary containing the status and the pull request object on success,
      or an error message on failure.
  """
  try:
    # Step 0: Ensure we are on the base branch and it's up to date.
    _run_git_command(["checkout", base_branch], local_path)
    _run_git_command(["pull", "origin", base_branch], local_path)

    # Step 1: Create a new, unique branch from the base branch.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_branch = f"agent-changes-{timestamp}"
    _run_git_command(["checkout", "-b", new_branch], local_path)
    print(f"Created and switched to new branch: {new_branch}")

    # Step 2: Apply the file changes.
    if not changes:
      return error_response("No changes provided to apply.")

    for relative_path, new_content in changes.items():
      full_path = os.path.join(local_path, relative_path)
      os.makedirs(os.path.dirname(full_path), exist_ok=True)
      with open(full_path, "w", encoding="utf-8") as f:
        f.write(new_content)
      print(f"Applied changes to {relative_path}")

    # Step 3: Stage the changes.
    _run_git_command(["add", "."], local_path)
    print("Staged all changes.")

    # Step 4: Commit the changes.
    _run_git_command(["commit", "-m", commit_message], local_path)
    print(f"Committed changes with message: '{commit_message}'")

    # Step 5: Push the new branch to the remote repository.
    _run_git_command(["push", "-u", "origin", new_branch], local_path)
    print(f"Pushed branch '{new_branch}' to origin.")

    # Step 6: Create the pull request via GitHub API.
    url = f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/pulls"
    payload = {
        "title": pr_title,
        "body": pr_body,
        "head": new_branch,
        "base": base_branch,
    }
    pr_response = post_request(url, payload)
    print(f"Successfully created pull request: {pr_response.get('html_url')}")

    return {"status": "success", "pull_request": pr_response}

  except subprocess.CalledProcessError as e:
    return error_response(f"A git command failed: {e.stderr}")
  except requests.exceptions.RequestException as e:
    return error_response(f"GitHub API request failed: {e}")
  except (IOError, OSError) as e:
    return error_response(f"A file system error occurred: {e}")


def get_issue(
    repo_owner: str, repo_name: str, issue_number: int
) -> Dict[str, Any]:
  """Get the details of the specified issue number.

  Args:
      repo_owner: The name of the repository owner.
      repo_name: The name of the repository.
      issue_number: issue number of the GitHub issue.

  Returns:
    The status of this request, with the issue details when successful.
  """
  url = (
      f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
  )
  try:
    response = get_request(url)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {"status": "success", "issue": response}


def create_issue(
    repo_owner: str,
    repo_name: str,
    title: str,
    body: str,
) -> Dict[str, Any]:
  """Create a new issue in the specified repository.

  Args:
      repo_owner: The name of the repository owner.
      repo_name: The name of the repository.
      title: The title of the issue.
      body: The body of the issue.

  Returns:
      The status of this request, with the issue details when successful.
  """
  url = f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/issues"
  payload = {"title": title, "body": body, "labels": ["docs updates"]}
  try:
    response = post_request(url, payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {"status": "success", "issue": response}


def update_issue(
    repo_owner: str,
    repo_name: str,
    issue_number: int,
    title: str,
    body: str,
) -> Dict[str, Any]:
  """Update an existing issue in the specified repository.

  Args:
      repo_owner: The name of the repository owner.
      repo_name: The name of the repository.
      issue_number: The number of the issue to update.
      title: The title of the issue.
      body: The body of the issue.

  Returns:
      The status of this request, with the issue details when successful.
  """
  url = (
      f"{GITHUB_BASE_URL}/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
  )
  payload = {"title": title, "body": body}
  try:
    response = patch_request(url, payload)
  except requests.exceptions.RequestException as e:
    return error_response(f"Error: {e}")
  return {"status": "success", "issue": response}


def _run_git_command(command: List[str], cwd: str) -> CompletedProcess[str]:
  """A helper to run a git command and raise an exception on error."""
  base_command = ["git"]
  process = subprocess.run(
      base_command + command,
      cwd=cwd,
      capture_output=True,
      text=True,
      check=True,  # This will raise CalledProcessError if the command fails
  )
  return process


def _find_head_commit_sha(repo_path: str) -> str:
  """Checks the head commit hash of a Git repository."""
  head_sha_command = ["git", "rev-parse", "HEAD"]
  head_sha_process = subprocess.run(
      head_sha_command,
      cwd=repo_path,
      capture_output=True,
      text=True,
      check=True,
  )
  current_commit_sha = head_sha_process.stdout.strip()
  return current_commit_sha


def _get_pull(repo_path: str) -> str:
  """Pulls the latest changes from a Git repository."""
  pull_process = subprocess.run(
      ["git", "pull"],
      cwd=repo_path,
      capture_output=True,
      text=True,
      check=True,
  )
  return pull_process.stdout.strip()


def _get_clone(repo_url: str, repo_path: str) -> str:
  """Clones a Git repository to a local folder."""
  clone_process = subprocess.run(
      ["git", "clone", repo_url, repo_path],
      capture_output=True,
      text=True,
      check=True,
  )
  return clone_process.stdout.strip()


def _git_grep(
    repo_path: str,
    pattern: str,
    extensions: Optional[List[str]] = None,
    ignored_dirs: Optional[List[str]] = None,
) -> subprocess.CompletedProcess[Any]:
  """Uses 'git grep' to find all matching lines in a Git repository."""
  grep_command = [
      "git",
      "grep",
      "-n",
      "-I",
      "-E",
      "--ignore-case",
      "-e",
      pattern,
  ]
  pathspecs = []
  if extensions:
    pathspecs.extend([f"*.{ext}" for ext in extensions])
  if ignored_dirs:
    pathspecs.extend([f":(exclude){d}" for d in ignored_dirs])

  if pathspecs:
    grep_command.append("--")
    grep_command.extend(pathspecs)

  grep_process = subprocess.run(
      grep_command,
      cwd=repo_path,
      capture_output=True,
      text=True,
      check=False,  # Don't raise error on non-zero exit code (1 means no match)
  )
  return grep_process

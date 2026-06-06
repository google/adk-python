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

"""Runner script to execute prompts/commands via the Antigravity SDK."""

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from typing import Any

try:
  from google.antigravity import Agent
  from google.antigravity import CapabilitiesConfig
  from google.antigravity import LocalAgentConfig
  from google.antigravity.hooks import policy
  from google.antigravity.types import Text
  from google.antigravity.types import Thought
  from google.antigravity.types import ToolCall
  from google.antigravity.types import ToolResult
except ImportError:
  print(
      "Error: google-antigravity package is not installed. Run 'pip install"
      " google-antigravity'",
      file=sys.stderr,
  )
  sys.exit(1)


def _is_safe_command(args: dict[str, Any]) -> bool:
  """Validates if the command is a safe 'gh' or 'git' execution with no shell injections."""
  cmd = (args.get("command_line") or args.get("CommandLine") or "").strip()
  if not cmd:
    return False

  # Forbid shell metacharacters and control characters
  forbidden_chars = {
      ";",
      "&",
      "|",
      "$",
      "`",
      "<",
      ">",
      "\n",
      "\r",
      "(",
      ")",
      "\\",
      "{",
      "}",
  }
  if any(char in cmd for char in forbidden_chars):
    return False

  try:
    tokens = shlex.split(cmd)
  except ValueError:
    return False

  if not tokens:
    return False

  return tokens[0] in {"gh", "git"}


def fetch_github_issue(issue_number: int) -> str:
  """Fetches the details of a GitHub issue from the google/adk-python repository.

  Args:
    issue_number: The issue number (e.g. 5949).
  """
  # Use curl to fetch the issue details.
  # This supports running it outside of the gh CLI environment (e.g. without login/remotes setup).
  cmd = [
      "curl",
      "-s",
  ]
  token = os.environ.get("GITHUB_TOKEN")
  if token:
    cmd.extend(["-H", f"Authorization: token {token}"])

  issue_cmd = cmd + [
      f"https://api.github.com/repos/google/adk-python/issues/{issue_number}"
  ]
  comments_cmd = cmd + [
      f"https://api.github.com/repos/google/adk-python/issues/{issue_number}/comments"
  ]

  try:
    res = subprocess.run(issue_cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
      return (
          f"Error: Failed to fetch issue {issue_number}: {res.stderr.strip()}"
      )

    try:
      issue_data = json.loads(res.stdout.strip())
    except json.JSONDecodeError as e:
      return (
          f"Error: Failed to parse issue JSON response: {e}. Output:"
          f" {res.stdout.strip()}"
      )

    res_comments = subprocess.run(
        comments_cmd, capture_output=True, text=True, check=False
    )
    if res_comments.returncode == 0:
      try:
        comments_data = json.loads(res_comments.stdout.strip())
        issue_data["comments_data"] = comments_data
      except json.JSONDecodeError:
        pass

    return json.dumps(issue_data, indent=2)
  except Exception as e:
    return f"Error: Failed to run curl command: {e}"


def fetch_github_pr(pr_number: int) -> str:
  """Fetches the details of a GitHub Pull Request from the google/adk-python repository.

  Args:
    pr_number: The PR number (e.g. 5956).
  """
  # Use curl to fetch the PR details.
  # This supports running it outside of the gh CLI environment (e.g. without login/remotes setup).
  cmd = [
      "curl",
      "-s",
  ]
  token = os.environ.get("GITHUB_TOKEN")
  if token:
    cmd.extend(["-H", f"Authorization: token {token}"])

  pr_cmd = cmd + [
      f"https://api.github.com/repos/google/adk-python/pulls/{pr_number}"
  ]
  issue_comments_cmd = cmd + [
      f"https://api.github.com/repos/google/adk-python/issues/{pr_number}/comments"
  ]
  review_comments_cmd = cmd + [
      f"https://api.github.com/repos/google/adk-python/pulls/{pr_number}/comments"
  ]

  try:
    res = subprocess.run(pr_cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
      return f"Error: Failed to fetch PR {pr_number}: {res.stderr.strip()}"

    try:
      pr_data = json.loads(res.stdout.strip())
    except json.JSONDecodeError as e:
      return (
          f"Error: Failed to parse PR JSON response: {e}. Output:"
          f" {res.stdout.strip()}"
      )

    # Fetch issue-level comments
    res_issue_comments = subprocess.run(
        issue_comments_cmd, capture_output=True, text=True, check=False
    )
    if res_issue_comments.returncode == 0:
      try:
        pr_data["comments_data"] = json.loads(res_issue_comments.stdout.strip())
      except json.JSONDecodeError:
        pass

    # Fetch review/inline comments
    res_review_comments = subprocess.run(
        review_comments_cmd, capture_output=True, text=True, check=False
    )
    if res_review_comments.returncode == 0:
      try:
        pr_data["review_comments_data"] = json.loads(
            res_review_comments.stdout.strip()
        )
      except json.JSONDecodeError:
        pass

    return json.dumps(pr_data, indent=2)
  except Exception as e:
    return f"Error: Failed to run curl command: {e}"


async def main():
  parser = argparse.ArgumentParser(
      description=(
          "Runner script to execute prompts/commands via the Antigravity SDK."
      )
  )
  parser.add_argument(
      "--show-steps",
      action="store_true",
      help="Show intermediate thoughts, tool calls, and tool results.",
  )
  parser.add_argument(
      "prompt",
      nargs="+",
      help="The prompt to send to the Antigravity Agent.",
  )
  parsed_args = parser.parse_args()

  show_steps = parsed_args.show_steps
  prompt = " ".join(parsed_args.prompt)

  # Ensure GEMINI_API_KEY is set (using GOOGLE_API_KEY as fallback)
  if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

  if "GEMINI_API_KEY" not in os.environ:
    print(
        "Error: GEMINI_API_KEY environment variable is not set.",
        file=sys.stderr,
    )
    sys.exit(1)

  skills_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..", ".agents", "skills")
  )
  config = LocalAgentConfig(
      capabilities=CapabilitiesConfig(),
      tools=[fetch_github_issue, fetch_github_pr],
      policies=[
          policy.deny(
              "run_command",
              when=lambda args: not _is_safe_command(args),
              name="only_allow_gh_and_git",
          ),
      ],
      skills_paths=[skills_dir],
  )

  try:
    async with Agent(config) as agent:
      response = await agent.chat(prompt)
      if show_steps:
        in_thinking = False
        async for chunk in response.chunks:
          if isinstance(chunk, Thought):
            if not in_thinking:
              sys.stdout.write("[Thinking...]\n")
              in_thinking = True
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
          elif isinstance(chunk, ToolCall):
            if in_thinking:
              sys.stdout.write("\n[End of Thinking]\n")
              in_thinking = False
            print(
                f"\n[Calling Tool: {chunk.name} with args: {chunk.args}]",
                flush=True,
            )
          elif isinstance(chunk, ToolResult):
            status = f"Error: {chunk.error}" if chunk.error else "Success"
            print(f"\n[Tool {chunk.name} finished: {status}]", flush=True)
          elif isinstance(chunk, Text):
            if in_thinking:
              sys.stdout.write("\n[End of Thinking]\n")
              in_thinking = False
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        if in_thinking:
          sys.stdout.write("\n[End of Thinking]\n")
      else:
        async for token in response:
          sys.stdout.write(token)
          sys.stdout.flush()
      print()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"\nError running Antigravity Agent: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  asyncio.run(main())

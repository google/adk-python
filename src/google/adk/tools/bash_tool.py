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

"""Tool to execute bash commands."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import pathlib
import shlex
import signal
from typing import Any
from typing import Optional

from google.genai import types

from .. import features
from .base_tool import BaseTool
from .tool_context import ToolContext


@dataclasses.dataclass(frozen=True)
class BashToolPolicy:
  """Configuration for allowed bash commands based on prefix matching.

  Set allowed_command_prefixes to ("*",) to allow all commands (default),
  or explicitly list allowed prefixes.
  """

  allowed_command_prefixes: tuple[str, ...] = ("*",)


def _validate_command(command: str, policy: BashToolPolicy) -> Optional[str]:
  """Validates a bash command against the permitted prefixes."""
  stripped = command.strip()
  if not stripped:
    return "Command is required."

  if "*" in policy.allowed_command_prefixes:
    return None

  for prefix in policy.allowed_command_prefixes:
    if stripped.startswith(prefix):
      return None

  allowed = ", ".join(policy.allowed_command_prefixes)
  return f"Command blocked. Permitted prefixes are: {allowed}"


@features.experimental(features.FeatureName.SKILL_TOOLSET)
class ExecuteBashTool(BaseTool):
  """Tool to execute a validated bash command within a workspace directory."""

  def __init__(
      self,
      *,
      workspace: pathlib.Path | None = None,
      policy: Optional[BashToolPolicy] = None,
  ):
    if workspace is None:
      workspace = pathlib.Path.cwd()
    policy = policy or BashToolPolicy()
    allowed_hint = (
        "any command"
        if "*" in policy.allowed_command_prefixes
        else (
            "commands matching prefixes:"
            f" {', '.join(policy.allowed_command_prefixes)}"
        )
    )
    super().__init__(
        name="execute_bash",
        description=(
            "Executes a bash command with the working directory set to the"
            f" workspace. Allowed: {allowed_hint}. All commands require user"
            " confirmation."
        ),
    )
    self._workspace = workspace
    self._policy = policy

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    command = args.get("command")
    if not command:
      return {"error": "Command is required."}

    # Static validation.
    error = _validate_command(command, self._policy)
    if error:
      return {"error": error}

    # Always request user confirmation.
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=f"Please approve or reject the bash command: {command}",
      )
      tool_context.actions.skip_summarization = True
      return {
          "error": (
              "This tool call requires confirmation, please approve or reject."
          )
      }
    elif not tool_context.tool_confirmation.confirmed:
      return {"error": "This tool call is rejected."}

    stdout = None
    stderr = None
    try:
      process = await asyncio.create_subprocess_exec(
          *shlex.split(command),
          cwd=str(self._workspace),
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
          start_new_session=True,
      )

      try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=30
        )
      except asyncio.TimeoutError:
        try:
          os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
        stdout, stderr = await process.communicate()
        return {
            "error": "Command timed out after 30 seconds.",
            "stdout": (
                stdout.decode(errors="replace")
                if stdout
                else "<no stdout captured>"
            ),
            "stderr": (
                stderr.decode(errors="replace")
                if stderr
                else "<no stderr captured>"
            ),
            "returncode": process.returncode,
        }
      finally:
        try:
          if process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass

      return {
          "stdout": (
              stdout.decode(errors="replace")
              if stdout
              else "<no stdout captured>"
          ),
          "stderr": (
              stderr.decode(errors="replace")
              if stderr
              else "<no stderr captured>"
          ),
          "returncode": process.returncode,
      }
    except Exception as e:  # pylint: disable=broad-except
      logger = logging.getLogger("google_adk." + __name__)
      logger.exception("ExecuteBashTool execution failed")

      stdout_res = (
          stdout.decode(errors="replace") if stdout else "<no stdout captured>"
      )
      stderr_res = (
          stderr.decode(errors="replace") if stderr else "<no stderr captured>"
      )

      return {
          "error": f"Execution failed: {str(e)}",
          "stdout": stdout_res,
          "stderr": stderr_res,
      }

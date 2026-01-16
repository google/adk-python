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

"""ScriptExecutor for safely executing scripts from Agent Skills bundles.

This module provides sandboxed execution of Python, Bash, and JavaScript
scripts bundled with skills.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental

logger = logging.getLogger("google_adk." + __name__)


class ScriptExecutionResult(BaseModel):
  """Result from script execution.

  Contains the output, errors, and execution metadata from running
  a script.
  """

  model_config = ConfigDict(extra="forbid")

  success: bool = Field(
      description="Whether the script executed successfully (return code 0).",
  )
  stdout: str = Field(
      default="",
      description="Standard output from the script.",
  )
  stderr: str = Field(
      default="",
      description="Standard error from the script.",
  )
  return_code: int = Field(
      default=0,
      description="Exit code from the script.",
  )
  execution_time_ms: float = Field(
      default=0.0,
      description="Execution time in milliseconds.",
  )
  timed_out: bool = Field(
      default=False,
      description="Whether execution was terminated due to timeout.",
  )


class ScriptExecutionError(Exception):
  """Raised when script execution fails."""

  def __init__(
      self, message: str, result: Optional[ScriptExecutionResult] = None
  ):
    super().__init__(message)
    self.result = result


@experimental
class ScriptExecutor(BaseModel):
  """Executes scripts from Agent Skills bundles.

  Provides sandboxed execution of Python, Bash, and JavaScript
  scripts bundled with skills.

  Security features:
  - Execution timeout
  - Working directory isolation
  - Environment variable filtering
  - Optional container sandboxing (requires Docker)

  Example:
    ```python
    executor = ScriptExecutor(
        timeout_seconds=30.0,
        allow_network=False,
    )

    result = await executor.execute_script(
        script_path=Path("/path/to/skill/scripts/extract.py"),
        args=["--input", "file.pdf"],
        working_dir=Path("/tmp/workspace"),
    )

    if result.success:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")
    ```
  """

  model_config = ConfigDict(extra="forbid")

  timeout_seconds: float = Field(
      default=60.0,
      description="Maximum execution time in seconds.",
      gt=0,
  )
  allow_network: bool = Field(
      default=False,
      description="Whether to allow network access (container mode only).",
  )
  memory_limit_mb: int = Field(
      default=256,
      description="Memory limit in megabytes (container mode only).",
      gt=0,
  )
  use_container: bool = Field(
      default=False,
      description="Use container isolation (requires Docker).",
  )
  allowed_env_vars: List[str] = Field(
      default_factory=lambda: [
          "PATH",
          "HOME",
          "LANG",
          "LC_ALL",
          "PYTHONPATH",
          "PYTHONIOENCODING",
      ],
      description="Environment variables to pass through to scripts.",
  )
  max_output_size: int = Field(
      default=1024 * 1024,  # 1MB
      description="Maximum size of stdout/stderr in bytes.",
      gt=0,
  )

  # Mapping of file extensions to interpreters
  INTERPRETERS: Dict[str, List[str]] = {
      ".py": ["python3", "python"],
      ".sh": ["bash", "sh"],
      ".bash": ["bash"],
      ".js": ["node"],
      ".mjs": ["node"],
  }

  async def execute_script(
      self,
      script_path: Union[str, Path],
      args: Optional[List[str]] = None,
      working_dir: Optional[Union[str, Path]] = None,
      env: Optional[Dict[str, str]] = None,
      stdin: Optional[str] = None,
  ) -> ScriptExecutionResult:
    """Execute a script file.

    Args:
      script_path: Path to the script file.
      args: Command-line arguments to pass to the script.
      working_dir: Working directory for execution. Defaults to script's
        parent directory.
      env: Additional environment variables to set.
      stdin: Optional input to pass to script's stdin.

    Returns:
      ScriptExecutionResult with stdout, stderr, and status.

    Raises:
      FileNotFoundError: If script doesn't exist.
      ValueError: If script type is not supported.
    """
    script_path = Path(script_path).resolve()
    args = args or []
    start_time = time.time()

    # Validate script exists
    if not script_path.exists():
      raise FileNotFoundError(f"Script not found: {script_path}")

    if not script_path.is_file():
      raise ValueError(f"Not a file: {script_path}")

    # Determine interpreter
    interpreter = self._get_interpreter(script_path)

    # Build command
    cmd = [interpreter, str(script_path)] + args

    # Build safe environment
    safe_env = self._build_safe_env(env)

    # Set working directory
    cwd = Path(working_dir).resolve() if working_dir else script_path.parent

    if not cwd.exists():
      raise FileNotFoundError(f"Working directory not found: {cwd}")

    logger.debug(
        "Executing script: %s with args: %s in %s",
        script_path.name,
        args,
        cwd,
    )

    try:
      if self.use_container:
        result = await self._execute_in_container(cmd, cwd, safe_env, stdin)
      else:
        result = await self._execute_subprocess(cmd, cwd, safe_env, stdin)

      execution_time = (time.time() - start_time) * 1000
      result.execution_time_ms = execution_time

      logger.debug(
          "Script %s completed: success=%s, return_code=%d, time=%.2fms",
          script_path.name,
          result.success,
          result.return_code,
          execution_time,
      )

      return result

    except asyncio.TimeoutError:
      execution_time = (time.time() - start_time) * 1000
      logger.warning(
          "Script %s timed out after %.2fs",
          script_path.name,
          self.timeout_seconds,
      )
      return ScriptExecutionResult(
          success=False,
          stderr=f"Execution timed out after {self.timeout_seconds}s",
          return_code=-1,
          execution_time_ms=execution_time,
          timed_out=True,
      )

    except Exception as e:
      execution_time = (time.time() - start_time) * 1000
      logger.error("Script %s failed: %s", script_path.name, e)
      return ScriptExecutionResult(
          success=False,
          stderr=str(e),
          return_code=-1,
          execution_time_ms=execution_time,
      )

  def _get_interpreter(self, script_path: Path) -> str:
    """Determine the interpreter for a script.

    Args:
      script_path: Path to the script.

    Returns:
      Interpreter command.

    Raises:
      ValueError: If script type is not supported.
    """
    suffix = script_path.suffix.lower()

    if suffix not in self.INTERPRETERS:
      supported = ", ".join(self.INTERPRETERS.keys())
      raise ValueError(
          f"Unsupported script type: {suffix}. Supported: {supported}"
      )

    # Find available interpreter
    for interpreter in self.INTERPRETERS[suffix]:
      if shutil.which(interpreter):
        return interpreter

    raise ValueError(
        f"No interpreter found for {suffix}. Tried: {self.INTERPRETERS[suffix]}"
    )

  def _build_safe_env(
      self, additional_env: Optional[Dict[str, str]] = None
  ) -> Dict[str, str]:
    """Build a safe environment for script execution.

    Args:
      additional_env: Additional environment variables to include.

    Returns:
      Filtered environment dictionary.
    """
    safe_env = {}

    # Only pass allowed environment variables
    for var in self.allowed_env_vars:
      if var in os.environ:
        safe_env[var] = os.environ[var]

    # Ensure UTF-8 encoding
    safe_env.setdefault("PYTHONIOENCODING", "utf-8")
    safe_env.setdefault("LANG", "en_US.UTF-8")

    # Add additional environment variables
    if additional_env:
      safe_env.update(additional_env)

    return safe_env

  async def _execute_subprocess(
      self,
      cmd: List[str],
      cwd: Path,
      env: Dict[str, str],
      stdin: Optional[str] = None,
  ) -> ScriptExecutionResult:
    """Execute script using subprocess.

    Args:
      cmd: Command and arguments.
      cwd: Working directory.
      env: Environment variables.
      stdin: Optional stdin input.

    Returns:
      ScriptExecutionResult with output.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE if stdin else None,
        cwd=str(cwd),
        env=env,
    )

    try:
      stdin_bytes = stdin.encode("utf-8") if stdin else None
      stdout, stderr = await asyncio.wait_for(
          proc.communicate(input=stdin_bytes),
          timeout=self.timeout_seconds,
      )

      # Truncate output if too large
      stdout_str = self._truncate_output(
          stdout.decode("utf-8", errors="replace")
      )
      stderr_str = self._truncate_output(
          stderr.decode("utf-8", errors="replace")
      )

      return ScriptExecutionResult(
          success=proc.returncode == 0,
          stdout=stdout_str,
          stderr=stderr_str,
          return_code=proc.returncode or 0,
      )

    except asyncio.TimeoutError:
      # Kill the process on timeout
      try:
        proc.kill()
        await proc.wait()
      except ProcessLookupError:
        pass
      raise

  async def _execute_in_container(
      self,
      cmd: List[str],
      cwd: Path,
      env: Dict[str, str],
      stdin: Optional[str] = None,
  ) -> ScriptExecutionResult:
    """Execute script in a Docker container.

    Args:
      cmd: Command and arguments.
      cwd: Working directory (mounted as volume).
      env: Environment variables.
      stdin: Optional stdin input.

    Returns:
      ScriptExecutionResult with output.

    Raises:
      ValueError: If Docker is not available.
    """
    # Check Docker availability
    if not shutil.which("docker"):
      raise ValueError(
          "Docker not found. Install Docker or set use_container=False."
      )

    # Build Docker command
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        f"--memory={self.memory_limit_mb}m",
        "--cpus=1",
        f"-v",
        f"{cwd}:/workspace:rw",
        "-w",
        "/workspace",
    ]

    # Add network isolation if required
    if not self.allow_network:
      docker_cmd.extend(["--network", "none"])

    # Add environment variables
    for key, value in env.items():
      docker_cmd.extend(["-e", f"{key}={value}"])

    # Determine base image based on interpreter
    interpreter = cmd[0]
    if interpreter in ("python3", "python"):
      image = "python:3.11-slim"
    elif interpreter in ("node",):
      image = "node:20-slim"
    else:
      image = "ubuntu:22.04"

    docker_cmd.append(image)
    docker_cmd.extend(cmd)

    logger.debug("Docker command: %s", " ".join(docker_cmd))

    # Execute in container
    return await self._execute_subprocess(
        docker_cmd, cwd, os.environ.copy(), stdin
    )

  def _truncate_output(self, output: str) -> str:
    """Truncate output if it exceeds max size.

    Args:
      output: Output string.

    Returns:
      Truncated output with notice if truncated.
    """
    if len(output) <= self.max_output_size:
      return output

    truncated = output[: self.max_output_size]
    return (
        truncated
        + f"\n\n... (output truncated, exceeded {self.max_output_size} bytes)"
    )

  def check_interpreter_available(self, script_suffix: str) -> bool:
    """Check if an interpreter is available for a script type.

    Args:
      script_suffix: File extension (e.g., ".py").

    Returns:
      True if an interpreter is available.
    """
    suffix = script_suffix.lower()
    if suffix not in self.INTERPRETERS:
      return False

    for interpreter in self.INTERPRETERS[suffix]:
      if shutil.which(interpreter):
        return True

    return False

  def get_available_interpreters(self) -> Dict[str, str]:
    """Get available interpreters for all supported script types.

    Returns:
      Dictionary mapping extensions to available interpreters.
    """
    available = {}
    for suffix, interpreters in self.INTERPRETERS.items():
      for interpreter in interpreters:
        if shutil.which(interpreter):
          available[suffix] = interpreter
          break
    return available

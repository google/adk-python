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

"""Agent Engine sandbox environment."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import time
from typing import Any
from typing import TYPE_CHECKING
import uuid

from google.genai.errors import APIError
from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...features import experimental
from ...features import FeatureName
from ._sandbox_manager import SandboxManager

if TYPE_CHECKING:
  import agentplatform

logger = logging.getLogger("google_adk." + __name__)

# Working directory of the shell sandbox container.
_WORKSPACE = "/workspace"

# Exit code the read command uses to report a missing file.
_FILE_NOT_FOUND_EXIT_CODE = 66

# Exit code the read command uses to report a file too large to read in one
# command, with the file size on stdout.
_FILE_TOO_LARGE_EXIT_CODE = 67

# Files up to this size are read in one command; larger files are read in
# slices of this size, to keep each response small.
_READ_CHUNK_BYTES = 1024 * 1024

# A command reaches the sandbox as one `bash -c` argument, which Linux caps at
# 128 KiB, so a write sends at most this many base64 characters per command and
# sends larger files in several commands.
_WRITE_CHUNK_BYTES = 96 * 1024

# Largest file write_file() accepts. A write takes one command per chunk, so
# this bounds how long a single write can take.
_MAX_WRITE_BYTES = 64 * 1024 * 1024

_SANDBOX_STATE_RUNNING = "STATE_RUNNING"

# How long to keep resending a command the service could not deliver to the
# sandbox, for example while a new or resumed sandbox is still starting,
# counted from the first failed delivery. The service itself retries for about
# 30 seconds before it reports a sandbox unreachable, so each attempt can take
# that long.
_UNREACHABLE_RETRY_SECONDS = 60
_UNREACHABLE_RETRY_INTERVAL_SECONDS = 2.0


@experimental(FeatureName.AGENT_ENGINE_SANDBOX_ENVIRONMENT)
class AgentEngineEnvironment(BaseEnvironment):
  """A remote workspace backed by an Agent Engine sandbox.

  Runs shell commands and reads and writes files in an Agent Engine shell
  sandbox, a gVisor-isolated container on GKE with a persistent filesystem.

  One sandbox is created on ``initialize()`` from a shell sandbox template,
  which is created once per agent engine and reused. ``close()`` pauses the
  sandbox rather than deleting it, and the next ``initialize()`` resumes it
  with its files intact. If the sandbox no longer exists by then, for example
  because its time to live ran out while it was paused, a new sandbox is
  created and the old sandbox's files are lost. A command sent to a paused
  sandbox, for example one paused by another process, waits for the sandbox to
  resume and then runs.

  Each instance owns one sandbox, so every session that uses the instance
  shares that sandbox and its files.

  Example:
    ```python
    from google.adk.integrations.vmaas import AgentEngineEnvironment
    from google.adk.tools.environment import EnvironmentToolset

    environment = AgentEngineEnvironment(project_id="my-project")
    agent = Agent(tools=[EnvironmentToolset(environment=environment)], ...)
    ```
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str = "us-central1",
      agent_engine_name: str | None = None,
      sandbox_name: str | None = None,
      sandbox_template_name: str | None = None,
      sandbox_ttl_seconds: int | None = None,
      vertexai_client: agentplatform.Client | None = None,
  ):
    """Creates an Agent Engine environment.

    Args:
      project_id: Google Cloud project ID. If None, the Application Default
        Credentials project is used.
      location: Agent Engine location.
      agent_engine_name: Agent engine to create the template and sandbox under.
        Format: projects/{project}/locations/{location}/reasoningEngines/{id}.
          If None, one is created on first use.
      sandbox_name: Existing shell sandbox to use instead of creating one.
        Format:
          projects/{project}/locations/{location}/reasoningEngines/{id}/sandboxEnvironments/{id}
      sandbox_template_name: Shell sandbox template to create the sandbox from.
        If None, a template is found or created under the agent engine.
        Format:
          projects/{project}/locations/{location}/reasoningEngines/{id}/sandboxEnvironmentTemplates/{id}
      sandbox_ttl_seconds: Time to live of a created sandbox. If None, the
        service default applies.
      vertexai_client: Agent Platform client. If None, one is created on first
        use from ``project_id`` and ``location``.
    """
    self._manager = SandboxManager(
        project_id=project_id,
        location=location,
        agent_engine_name=agent_engine_name,
        client=vertexai_client,
    )
    self._sandbox_name = sandbox_name
    # Whether this environment created the sandbox, and so may replace it.
    self._owns_sandbox = False
    self._sandbox_template_name = sandbox_template_name
    self._sandbox_ttl_seconds = sandbox_ttl_seconds
    # Serializes initialize() and close(), so concurrent callers create one
    # sandbox and never pause it while it is being set up.
    self._lifecycle_lock = asyncio.Lock()

  @property
  def sandbox_name(self) -> str | None:
    """The resource name of the sandbox, once there is one."""
    return self._sandbox_name

  @property
  @override
  def working_dir(self) -> Path:
    if not self._is_initialized:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    return Path(_WORKSPACE)

  @override
  async def initialize(self) -> None:
    async with self._lifecycle_lock:
      if self._is_initialized:
        return
      if (
          self._sandbox_name is not None
          and not await self._manager.sandbox_exists(self._sandbox_name)
      ):
        if not self._owns_sandbox:
          raise RuntimeError(f"Sandbox {self._sandbox_name} does not exist.")
        # The sandbox expired, for example while it was paused.
        logger.warning(
            "Sandbox %s no longer exists; creating a new one. Files in the old"
            " sandbox are lost.",
            self._sandbox_name,
        )
        self._sandbox_name = None
      if self._sandbox_name is None:
        template_name = (
            self._sandbox_template_name
            or await self._manager.get_shell_template_name()
        )
        self._sandbox_name = await self._manager.create_sandbox(
            template_name=template_name,
            display_name="adk-shell-sandbox",
            ttl_seconds=self._sandbox_ttl_seconds,
        )
        self._owns_sandbox = True
      else:
        await self._manager.resume_sandbox_if_paused(self._sandbox_name)
      self._is_initialized = True

  @override
  async def close(self) -> None:
    async with self._lifecycle_lock:
      if not self._is_initialized or self._sandbox_name is None:
        return
      await self._manager.pause_sandbox(self._sandbox_name)
      self._is_initialized = False

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
  ) -> ExecutionResult:
    timeout_seconds = math.ceil(timeout) if timeout is not None else None
    try:
      result = await self._execute_bash(command, timeout=timeout_seconds)
    except APIError as e:
      if _is_timeout(e):
        return ExecutionResult(exit_code=-1, timed_out=True)
      raise
    return ExecutionResult(
        exit_code=result.get("returncode", -1),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
    )

  @override
  async def read_file(self, path: str | os.PathLike[str]) -> bytes:
    resolved = self._resolve_path(path)
    quoted = shlex.quote(resolved)
    # pipefail keeps a base64 failure, such as reading a directory, from being
    # masked by tr's exit code.
    result = await self._execute_bash(
        "set -o pipefail;"
        f" [ -e {quoted} ] || exit {_FILE_NOT_FOUND_EXIT_CODE};"
        f" size=$(stat -L -c %s -- {quoted}) || exit 1;"
        f' if [ "$size" -gt {_READ_CHUNK_BYTES} ]; then'
        f' echo "$size"; exit {_FILE_TOO_LARGE_EXIT_CODE}; fi;'
        f" base64 < {quoted} | tr -d '\\n'"
    )
    returncode = result.get("returncode")
    if returncode == _FILE_NOT_FOUND_EXIT_CODE:
      raise FileNotFoundError(resolved)
    if returncode == _FILE_TOO_LARGE_EXIT_CODE:
      size = int(result.get("stdout", ""))
      return await self._read_file_in_slices(resolved, size)
    if returncode != 0:
      raise OSError(f"Failed to read {resolved}: {result.get('stderr', '')}")
    return base64.b64decode(result.get("stdout", ""))

  async def _read_file_in_slices(self, path: str, size: int) -> bytes:
    """Reads a file one slice per command, for files too large for one."""
    quoted = shlex.quote(path)
    slices = []
    for index in range(math.ceil(size / _READ_CHUNK_BYTES)):
      result = await self._execute_bash(
          f"set -o pipefail; dd if={quoted} bs={_READ_CHUNK_BYTES}"
          f" skip={index} count=1 iflag=fullblock status=none | base64 |"
          " tr -d '\\n'"
      )
      if result.get("returncode") != 0:
        raise OSError(f"Failed to read {path}: {result.get('stderr', '')}")
      slices.append(base64.b64decode(result.get("stdout", "")))
    content = b"".join(slices)
    if len(content) != size:
      raise OSError(
          f"Failed to read {path}: expected {size} bytes, got {len(content)}"
      )
    return content

  @override
  async def write_file(
      self, path: str | os.PathLike[str], content: str | bytes
  ) -> None:
    if isinstance(content, str):
      content = content.encode("utf-8")
    resolved = PurePosixPath(self._resolve_path(path))
    if len(content) > _MAX_WRITE_BYTES:
      raise ValueError(
          f"Cannot write {len(content)} bytes to {resolved}: files larger than"
          f" {_MAX_WRITE_BYTES} bytes are not supported."
      )
    encoded = base64.b64encode(content).decode("ascii")
    parent = shlex.quote(str(resolved.parent))
    target = shlex.quote(str(resolved))
    if len(encoded) <= _WRITE_CHUNK_BYTES:
      await self._run_write_command(
          f"mkdir -p -- {parent} && printf %s {encoded} | base64 -d > {target}",
          resolved,
      )
      return
    # Upload the base64 text to a temporary file next to the target in
    # several commands, then decode it to a second temporary file and rename
    # that over the target, so the target is replaced only once all of the
    # content has arrived. The name is truncated to stay within the file
    # name length limit.
    prefix = resolved.name.encode("utf-8")[:100].decode(
        "utf-8", errors="ignore"
    )
    temp_path = resolved.parent / f".{prefix}.adk-{uuid.uuid4().hex}"
    temp = shlex.quote(str(temp_path))
    decoded = shlex.quote(f"{temp_path}.new")
    chunks = [
        encoded[i : i + _WRITE_CHUNK_BYTES]
        for i in range(0, len(encoded), _WRITE_CHUNK_BYTES)
    ]
    try:
      await self._run_write_command(
          f"mkdir -p -- {parent} && printf %s {chunks[0]} > {temp}", resolved
      )
      for chunk in chunks[1:]:
        await self._run_write_command(f"printf %s {chunk} >> {temp}", resolved)
      await self._run_write_command(
          f"base64 -d {temp} > {decoded} &&"
          f" {{ chmod --reference={target} -- {decoded} 2>/dev/null;"
          f" mv -fT -- {decoded} {target}; }};"
          f" rc=$?; rm -f -- {temp} {decoded}; exit $rc",
          resolved,
      )
    except Exception:
      try:
        await self._execute_bash(f"rm -f -- {temp} {decoded}")
      except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Failed to remove temporary files %s*", temp_path, exc_info=True
        )
      raise

  async def _run_write_command(self, command: str, path: PurePosixPath) -> None:
    """Runs one command of a file write, raising OSError if it fails."""
    result = await self._execute_bash(command)
    if result.get("returncode") != 0:
      raise OSError(f"Failed to write {path}: {result.get('stderr', '')}")

  async def _execute_bash(
      self, command: str, *, timeout: int | None = None
  ) -> dict[str, Any]:
    """Runs a command in the sandbox, resuming the sandbox if it is paused."""
    sandbox_name = self._sandbox_name
    if not self._is_initialized or sandbox_name is None:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    resumed = False
    unreachable_deadline: float | None = None
    while True:
      try:
        return await self._manager.execute_bash(
            sandbox_name, command, timeout=timeout
        )
      except APIError as e:
        # The command never reached the shell server, so sending it again
        # cannot run it twice.
        if _is_unreachable(e):
          if unreachable_deadline is None:
            unreachable_deadline = time.monotonic() + _UNREACHABLE_RETRY_SECONDS
          if time.monotonic() < unreachable_deadline:
            logger.info(
                "Sandbox %s is not reachable yet; retrying", sandbox_name
            )
            await asyncio.sleep(_UNREACHABLE_RETRY_INTERVAL_SECONDS)
            continue
          raise
        # A paused or resuming sandbox rejects commands: the service answers
        # UNAVAILABLE after it starts resuming a paused sandbox, and
        # FAILED_PRECONDITION while the sandbox is not running. The service
        # also reports some errors from a running sandbox, such as a command
        # timeout, as FAILED_PRECONDITION, so the command is retried only if
        # it did not time out and the sandbox is not running; otherwise it
        # would run twice.
        if (
            resumed
            or _is_timeout(e)
            or (e.code != 503 and e.status != "FAILED_PRECONDITION")
        ):
          raise
        state = await self._manager.get_sandbox_state(sandbox_name)
        if state == _SANDBOX_STATE_RUNNING:
          raise
        logger.info("Sandbox %s is %s; resuming", sandbox_name, state)
      await self._manager.wait_until_running(sandbox_name)
      resumed = True
      unreachable_deadline = None

  def _resolve_path(self, path: str | os.PathLike[str]) -> str:
    """Resolves a relative path against the sandbox working directory."""
    pure = PurePosixPath(os.fspath(path))
    if pure.is_absolute():
      return str(pure)
    return str(PurePosixPath(_WORKSPACE) / pure)


def _is_timeout(error: APIError) -> bool:
  """Returns whether an error reports a command the sandbox killed for time."""
  # The service reports the shell server's timeout as FAILED_PRECONDITION with
  # DEADLINE_EXCEEDED in the message, rather than as a 504.
  return (
      error.code == 504
      or error.status == "DEADLINE_EXCEEDED"
      or "DEADLINE_EXCEEDED" in (error.message or "")
  )


def _is_unreachable(error: APIError) -> bool:
  """Returns whether the service could not deliver a command to the sandbox."""
  return "Unable to reach the sandbox environment" in (error.message or "")

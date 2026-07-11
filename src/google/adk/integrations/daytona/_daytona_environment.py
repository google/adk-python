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

"""Daytona sandbox code execution environment."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...features import experimental
from ...features import FeatureName

if TYPE_CHECKING:
  from daytona import AsyncDaytona
  from daytona import AsyncSandbox
  from daytona import ExecuteResponse
  from daytona import Image

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_TIMEOUT = 300
_SANDBOX_HOME = "/workspaces"


@experimental(FeatureName.DAYTONA_ENVIRONMENT)
class DaytonaEnvironment(BaseEnvironment):
  """A persistent remote workspace backed by a Daytona sandbox.

  Provides file CRUD and shell execution inside an isolated remote sandbox.
  One sandbox is created on ``initialize()`` and killed on ``close()``.

  Requires the ``daytona`` extra: ``pip install google-adk[daytona]``.
  """

  def __init__(
      self,
      *,
      image: str | Image | None = None,
      timeout: int = _DEFAULT_TIMEOUT,
      api_key: str | None = None,
      api_url: str | None = None,
      env_vars: dict[str, str] | None = None,
  ):
    """Create a Daytona environment.

    Args:
      image: Daytona template/image name used to create the sandbox.
      timeout: Sandbox time-to-live / timeout in seconds.
      api_key: Daytona API key. If ``None``, the environment variable is used.
      api_url: Daytona API URL. If ``None``, defaults to Daytona Cloud API.
      env_vars: Environment variables set inside the sandbox.
    """
    self._image = image
    self._timeout = timeout
    self._api_key = api_key
    self._api_url = api_url
    self._env_vars = env_vars
    self._sandbox: AsyncSandbox | None = None
    self._client: AsyncDaytona | None = None  # To hold AsyncDaytona instance

  @property
  @override
  def working_dir(self) -> Path:
    if self._sandbox is None:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    return Path(_SANDBOX_HOME)

  @override
  async def initialize(self) -> None:
    if self._sandbox is not None:
      return
    self._sandbox = await self._create_sandbox()
    self._is_initialized = True

  @override
  async def close(self) -> None:
    if self._sandbox is not None:
      await self._sandbox.delete()
      self._sandbox = None
      self._client = None
      self._is_initialized = False

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
  ) -> ExecutionResult:
    sandbox = await self._ensure_sandbox()

    # timeout needs to be int for daytona SDK
    timeout_int = int(timeout) if timeout is not None else self._timeout

    try:
      response: ExecuteResponse = await sandbox.process.exec(
          command=command,
          timeout=timeout_int,
      )
    except Exception as e:
      # If Daytona has specific Timeout exceptions, they should be handled,
      # but a generic fallback is catching and logging/translating.
      from daytona import DaytonaError

      if isinstance(e, DaytonaError) and "timeout" in str(e).lower():
        return ExecutionResult(exit_code=-1, timed_out=True)
      # Otherwise raise
      raise e

    return ExecutionResult(
        exit_code=response.exit_code or 0,
        stdout=response.artifacts.stdout if response.artifacts else "",
        # Daytona process.exec combines stdout and stderr into stdout.
        stderr="",
    )

  @override
  async def read_file(self, path: str | os.PathLike[str]) -> bytes:
    sandbox = await self._ensure_sandbox()
    resolved = self._resolve_path(path)
    try:
      content = await sandbox.fs.download_file(resolved)
      if content is None:
        raise FileNotFoundError(resolved)
      return bytes(content)
    except Exception as e:
      from daytona import DaytonaNotFoundError

      if isinstance(e, DaytonaNotFoundError):
        raise FileNotFoundError(resolved) from e
      raise e

  @override
  async def write_file(
      self, path: str | os.PathLike[str], content: str | bytes
  ) -> None:
    sandbox = await self._ensure_sandbox()
    resolved = self._resolve_path(path)

    # Create parent directory recursively to prevent upload failures
    resolved_path = PurePosixPath(resolved)
    parent = resolved_path.parent
    if parent and parent != PurePosixPath("/"):
      parts = parent.parts
      for i in range(2, len(parts) + 1):
        current = PurePosixPath(*parts[:i])
        try:
          await sandbox.fs.create_folder(str(current), mode="755")
        except Exception as e:
          # If folder already exists, Daytona may raise DaytonaConflictError
          # or similar. We check the message or type if we can, but safely
          # ignoring is fine since the ultimate upload will fail if it's a
          # real issue.
          from daytona import DaytonaConflictError

          if (
              isinstance(e, DaytonaConflictError)
              or "already exists" in str(e).lower()
          ):
            continue

    # Daytona's upload_file accepts bytes directly
    if isinstance(content, str):
      content_bytes = content.encode("utf-8")
    else:
      content_bytes = content

    await sandbox.fs.upload_file(content_bytes, resolved)

  async def _create_sandbox(self) -> AsyncSandbox:
    try:
      from daytona import AsyncDaytona
      from daytona import CreateSandboxFromImageParams
      from daytona import CreateSandboxFromSnapshotParams
      from daytona import DaytonaConfig
    except ImportError as e:
      raise ImportError(
          "The daytona package is required to use DaytonaEnvironment. Install"
          " it with `pip install google-adk[daytona]`."
      ) from e

    config_args = {}
    if self._api_key:
      config_args["api_key"] = self._api_key
    if self._api_url:
      config_args["api_url"] = self._api_url

    config = DaytonaConfig(**config_args) if config_args else None
    self._client = AsyncDaytona(config=config)

    auto_stop_interval_mins = self._timeout // 60
    if self._timeout > 0 and auto_stop_interval_mins == 0:
      auto_stop_interval_mins = 1

    if self._image:
      params = CreateSandboxFromImageParams(
          image=self._image,
          env_vars=self._env_vars or {},
          auto_stop_interval=auto_stop_interval_mins,
          auto_delete_interval=0,
      )
    else:
      params = CreateSandboxFromSnapshotParams(
          language="python",
          env_vars=self._env_vars or {},
          auto_stop_interval=auto_stop_interval_mins,
          auto_delete_interval=0,
      )

    return await self._client.create(params)

  async def _ensure_sandbox(self) -> AsyncSandbox:
    sandbox = self._sandbox
    if sandbox is None:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    await sandbox.refresh_activity()
    return sandbox

  def _resolve_path(self, path: str | os.PathLike[str]) -> str:
    """Resolve a relative path against the sandbox working directory."""
    pure = PurePosixPath(os.fspath(path))
    if pure.is_absolute():
      return str(pure)
    return str(PurePosixPath(_SANDBOX_HOME) / pure)

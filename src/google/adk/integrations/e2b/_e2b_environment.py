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

"""E2B sandbox code execution environment."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...utils.feature_decorator import experimental

if TYPE_CHECKING:
  from e2b import AsyncSandbox

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_IMAGE = 'base'
_DEFAULT_TIMEOUT = 300
_SANDBOX_HOME = '/home/user'


@experimental
class E2BEnvironment(BaseEnvironment):
  """A persistent remote workspace backed by an E2B sandbox.

  Provides file CRUD, shell execution, and on-demand software installs
  (e.g. ``pip install``, ``apt install``) inside an isolated remote
  sandbox.

  One sandbox is created on ``initialize()`` and killed on ``close()``.
  The sandbox has a bounded time-to-live (``timeout``) to cap credit
  usage.  Every operation extends the TTL so an actively used workspace
  never expires mid-use; once it does expire after genuine idle, the next
  operation transparently recreates a fresh sandbox (workspace state such
  as installs and files is lost).

  Requires the ``e2b`` extra: ``pip install google-adk[e2b]``.
  """

  def __init__(
      self,
      *,
      image: str = _DEFAULT_IMAGE,
      timeout: int = _DEFAULT_TIMEOUT,
      api_key: Optional[str] = None,
      env_vars: Optional[dict[str, str]] = None,
  ):
    """Create an E2B environment.

    Args:
      image: E2B template name or ID used to create the sandbox.  Defaults
        to E2B's public ``base`` template, available to every user.
      timeout: Sandbox time-to-live in seconds.  The TTL is reset on every
        operation.  Defaults to 300 seconds.
      api_key: E2B API key.  If ``None``, the ``E2B_API_KEY`` environment
        variable is used.
      env_vars: Environment variables set inside the sandbox.
    """
    self._image = image
    self._timeout = timeout
    self._api_key = api_key
    self._env_vars = env_vars
    self._sandbox: Optional[AsyncSandbox] = None

  @property
  @override
  def working_dir(self) -> Path:
    if self._sandbox is None:
      raise RuntimeError('Sandbox is not started. Call initialize() first.')
    return Path(_SANDBOX_HOME)

  @override
  async def initialize(self) -> None:
    if self._sandbox is not None:
      return
    self._sandbox = await self._create_sandbox()

  @override
  async def close(self) -> None:
    if self._sandbox is not None:
      await self._sandbox.kill()
      self._sandbox = None

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
  ) -> ExecutionResult:
    from e2b import CommandExitException
    from e2b import TimeoutException

    sandbox = await self._ensure_sandbox()
    try:
      result = await sandbox.commands.run(command, timeout=timeout)
    except CommandExitException as e:
      # A non-zero exit code is a normal result, not a failure.
      return ExecutionResult(
          exit_code=e.exit_code,
          stdout=e.stdout,
          stderr=e.stderr,
      )
    except TimeoutException:
      return ExecutionResult(exit_code=-1, timed_out=True)

    return ExecutionResult(
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
    )

  @override
  async def read_file(self, path: str | os.PathLike[str]) -> bytes:
    from e2b import FileNotFoundException

    sandbox = await self._ensure_sandbox()
    resolved = self._resolve_path(path)
    try:
      content = await sandbox.files.read(resolved, format='bytes')
    except FileNotFoundException as e:
      raise FileNotFoundError(resolved) from e
    return bytes(content)

  @override
  async def write_file(
      self, path: str | os.PathLike[str], content: str | bytes
  ) -> None:
    sandbox = await self._ensure_sandbox()
    resolved = self._resolve_path(path)
    await sandbox.files.write(resolved, content)

  async def _create_sandbox(self) -> AsyncSandbox:
    try:
      from e2b import AsyncSandbox
    except ImportError as e:
      raise ImportError(
          'The e2b package is required to use E2BEnvironment. Install it with'
          ' `pip install google-adk[e2b]`.'
      ) from e

    return await AsyncSandbox.create(
        template=self._image,
        timeout=self._timeout,
        envs=self._env_vars,
        api_key=self._api_key,
    )

  async def _ensure_sandbox(self) -> AsyncSandbox:
    if self._sandbox is None:
      raise RuntimeError('Sandbox is not started. Call initialize() first.')

    if await self._sandbox.is_running():
      # Keepalive: extend the TTL while the workspace is actively used.
      await self._sandbox.set_timeout(self._timeout)
    else:
      logger.warning(
          'E2B sandbox expired; recreating a fresh sandbox. Workspace state'
          ' (installed packages and files) has been lost.'
      )
      self._sandbox = await self._create_sandbox()
    return self._sandbox

  def _resolve_path(self, path: str | os.PathLike[str]) -> str:
    """Resolve a relative path against the sandbox working directory."""
    pure = PurePosixPath(os.fspath(path))
    if pure.is_absolute():
      return str(pure)
    return str(PurePosixPath(_SANDBOX_HOME) / pure)

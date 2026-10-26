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

"""OpenSandbox remote code execution environment."""

from __future__ import annotations

from datetime import timedelta
import logging
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...features import experimental
from ...features import FeatureName

if TYPE_CHECKING:
  from opensandbox import Sandbox
  from opensandbox.config import ConnectionConfig
  from opensandbox.models.execd import Execution
  from opensandbox.models.sandboxes import NetworkPolicy
  from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_IMAGE = "python:3.11"
_DEFAULT_TIMEOUT = 300
_DEFAULT_READY_TIMEOUT = 30
_MIN_TIMEOUT = 60
_COMMAND_TTL_GRACE = 30
_SANDBOX_HOME = "/workspace"
# OpenSandbox modes use decimal digits, not Python's octal literal form.
_DIRECTORY_MODE = 755
_FILE_MODE = 644
_HTTP_NOT_FOUND = 404
_TIMEOUT_EXIT_CODE = -1


@experimental(FeatureName.OPENSANDBOX_ENVIRONMENT)
class OpenSandboxEnvironment(BaseEnvironment):
  """A persistent remote workspace backed by OpenSandbox.

  By default, ``initialize()`` creates a sandbox and ``close()`` destroys it.
  When ``sandbox_id`` is supplied, the environment attaches to that sandbox;
  ``close()`` then releases only local SDK resources and leaves the remote
  sandbox running.

  Each command execution on an owned sandbox renews its configured lifetime so
  an actively used workspace never expires mid-use; once it does expire after
  genuine idle, the next ``execute()`` call transparently recreates a fresh
  sandbox (workspace state such as installs and files is lost). Commands
  without an explicit timeout renew only the configured sandbox lifetime. Pass
  a timeout when a command may run longer than that lifetime.

  Requires the ``opensandbox`` extra:
  ``pip install google-adk[opensandbox]``.
  """

  def __init__(
      self,
      *,
      image: str | SandboxImageSpec | None = None,
      snapshot_id: str | None = None,
      sandbox_id: str | None = None,
      timeout: int = _DEFAULT_TIMEOUT,
      ready_timeout: float = _DEFAULT_READY_TIMEOUT,
      env_vars: dict[str, str] | None = None,
      metadata: dict[str, str] | None = None,
      connection_config: ConnectionConfig | None = None,
      network_policy: NetworkPolicy | None = None,
      secure_access: bool = False,
  ):
    """Create an OpenSandbox environment.

    Args:
      image: Container image for a newly created sandbox. Defaults to
        ``python:3.11``. Mutually exclusive with ``snapshot_id``.
      snapshot_id: Snapshot used instead of an image for a new sandbox.
      sandbox_id: Existing sandbox to attach to instead of creating one.
      timeout: Owned sandbox lifetime in seconds. Must be at least 60 seconds.
        The lifetime is renewed before each command execution.
      ready_timeout: Maximum seconds to wait for create or connect readiness.
      env_vars: Environment variables applied to new sandboxes and commands.
      metadata: Additional metadata for a newly created sandbox.
      connection_config: OpenSandbox SDK connection configuration. When
        omitted, honors an explicit scheme in ``OPEN_SANDBOX_DOMAIN`` or
        defaults protocol to HTTP for localhost or unspecified domain and HTTPS
        for remote hosts, reading ``OPEN_SANDBOX_DOMAIN`` and
        ``OPEN_SANDBOX_API_KEY``.
      network_policy: Optional network policy for a newly created sandbox.
      secure_access: Whether to enable secure access mode for a newly created
        sandbox.

    Raises:
      ValueError: If configuration values are invalid or mutually exclusive.
    """
    if image is not None and snapshot_id is not None:
      raise ValueError("image and snapshot_id are mutually exclusive")
    if sandbox_id is not None and (
        image is not None
        or snapshot_id is not None
        or network_policy is not None
        or secure_access
    ):
      raise ValueError(
          "image, snapshot_id, network_policy, and secure_access cannot be"
          " used when attaching by sandbox_id"
      )
    if timeout < _MIN_TIMEOUT:
      raise ValueError("timeout must be at least 60 seconds")
    if ready_timeout <= 0:
      raise ValueError("ready_timeout must be positive")

    self._image = image
    self._snapshot_id = snapshot_id
    self._requested_sandbox_id = sandbox_id
    self._timeout = timedelta(seconds=timeout)
    self._ready_timeout = timedelta(seconds=ready_timeout)
    self._env_vars = dict(env_vars) if env_vars is not None else None
    self._metadata = dict(metadata) if metadata is not None else None
    self._connection_config = connection_config
    self._network_policy = network_policy
    self._secure_access = secure_access
    self._sandbox: Sandbox | None = None
    self._owns_sandbox = False

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

    self._sandbox, self._owns_sandbox = await self._create_and_prepare_sandbox()
    self._is_initialized = True

  async def _create_and_prepare_sandbox(self) -> tuple[Sandbox, bool]:
    sandbox, owns_sandbox = await self._open_sandbox()
    try:
      await self._prepare_working_directory(sandbox)
    except BaseException:
      try:
        await self._cleanup_sandbox(sandbox, owns_sandbox=owns_sandbox)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Failed to clean up OpenSandbox after initialization failure",
            exc_info=True,
        )
      raise
    return sandbox, owns_sandbox

  @override
  async def close(self) -> None:
    sandbox = self._sandbox
    if sandbox is None:
      return

    try:
      await self._cleanup_sandbox(sandbox, owns_sandbox=self._owns_sandbox)
    finally:
      self._sandbox = None
      self._owns_sandbox = False
      self._is_initialized = False

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
  ) -> ExecutionResult:
    if timeout is not None and timeout < 0:
      raise ValueError("timeout must be non-negative or None")

    from opensandbox.models.execd import RunCommandOpts

    sandbox = await self._get_sandbox(
        renewal_timeout=self._command_renewal_timeout(timeout)
    )
    started_at = monotonic()
    execution = await sandbox.commands.run(
        command,
        opts=RunCommandOpts(
            working_directory=_SANDBOX_HOME,
            timeout=self._command_timeout(timeout),
            envs=self._env_vars,
        ),
    )
    elapsed = monotonic() - started_at

    stdout = self._stdout(execution)
    stderr = self._stderr(execution)

    exit_code = execution.exit_code
    if exit_code is None:
      exit_code = _TIMEOUT_EXIT_CODE
      error = execution.error
      detail = (
          f"{error.name}: {error.value}"
          if error is not None
          else "command completed without an exit code"
      )
      stderr = f"{stderr.rstrip()}\n{detail}" if stderr else detail

    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=(
            timeout is not None
            and exit_code == _TIMEOUT_EXIT_CODE
            and elapsed >= timeout
        ),
    )

  @override
  async def read_file(self, path: str | os.PathLike[str]) -> bytes:
    from opensandbox.exceptions import SandboxApiException

    sandbox = await self._get_sandbox()
    resolved = self._resolve_path(path)
    try:
      return bytes(await sandbox.files.read_bytes(resolved))
    except SandboxApiException as e:
      if e.status_code == _HTTP_NOT_FOUND:
        raise FileNotFoundError(resolved) from e
      raise

  @override
  async def write_file(
      self, path: str | os.PathLike[str], content: str | bytes
  ) -> None:
    from opensandbox.models.filesystem import WriteEntry

    sandbox = await self._get_sandbox()
    resolved = self._resolve_path(path)
    parent = PurePosixPath(resolved).parent
    if parent != PurePosixPath("/"):
      parts = parent.parts
      entries = [
          WriteEntry(path=str(PurePosixPath(*parts[:i])), mode=_DIRECTORY_MODE)
          for i in range(2, len(parts) + 1)
      ]
      await sandbox.files.create_directories(entries)
    await sandbox.files.write_file(resolved, content, mode=_FILE_MODE)

  def _resolve_connection_config(self) -> ConnectionConfig:
    if self._connection_config is not None:
      return self._connection_config

    from opensandbox.config import ConnectionConfig

    config = ConnectionConfig()
    domain = (config.get_domain() or "").strip()
    parsed = urlsplit(domain if "://" in domain else f"//{domain}")
    protocol = parsed.scheme or (
        "http"
        if not parsed.hostname
        or parsed.hostname in ("localhost", "127.0.0.1", "::1")
        else "https"
    )
    return ConnectionConfig(protocol=protocol)

  async def _open_sandbox(self) -> tuple[Sandbox, bool]:
    try:
      from opensandbox import Sandbox
    except ImportError as e:
      raise ImportError(
          "The opensandbox package is required to use OpenSandboxEnvironment. "
          "Install it with `pip install google-adk[opensandbox]`."
      ) from e

    connection_config = self._resolve_connection_config()
    if self._requested_sandbox_id is not None:
      sandbox = await Sandbox.connect(
          self._requested_sandbox_id,
          connection_config=connection_config,
          connect_timeout=self._ready_timeout,
      )
      return sandbox, False

    metadata = {
        "framework": "google-adk",
        "integration": "google-adk-opensandbox",
    }
    metadata.update(self._metadata or {})
    image = self._image
    if image is None and self._snapshot_id is None:
      image = _DEFAULT_IMAGE
    sandbox = await Sandbox.create(
        image,
        snapshot_id=self._snapshot_id,
        timeout=self._timeout,
        ready_timeout=self._ready_timeout,
        env=self._env_vars,
        metadata=metadata,
        connection_config=connection_config,
        network_policy=self._network_policy,
        secure_access=self._secure_access,
    )
    return sandbox, True

  async def _get_sandbox(
      self, *, renewal_timeout: timedelta | None = None
  ) -> Sandbox:
    sandbox = self._sandbox
    if sandbox is None:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    if self._owns_sandbox and renewal_timeout is not None:
      from opensandbox.exceptions import SandboxApiException

      try:
        await sandbox.renew(renewal_timeout)
      except SandboxApiException as e:
        if e.status_code == _HTTP_NOT_FOUND:
          logger.warning(
              "OpenSandbox expired; recreating a fresh sandbox. Workspace"
              " state (installed packages and files) has been lost."
          )
          await sandbox.close()
          self._sandbox, self._owns_sandbox = (
              await self._create_and_prepare_sandbox()
          )
          sandbox = self._sandbox
          if renewal_timeout > self._timeout:
            await sandbox.renew(renewal_timeout)
        else:
          raise
    return sandbox

  @staticmethod
  async def _cleanup_sandbox(sandbox: Sandbox, *, owns_sandbox: bool) -> None:
    if not owns_sandbox:
      await sandbox.close()
      return

    from opensandbox.exceptions import SandboxApiException

    try:
      await sandbox.destroy()
    except SandboxApiException as e:
      if e.status_code != _HTTP_NOT_FOUND:
        raise

  @staticmethod
  async def _prepare_working_directory(sandbox: Sandbox) -> None:
    from opensandbox.models.filesystem import WriteEntry

    await sandbox.files.create_directories(
        [WriteEntry(path=_SANDBOX_HOME, mode=_DIRECTORY_MODE)]
    )

  def _command_renewal_timeout(self, timeout: float | None) -> timedelta:
    if timeout is None:
      return self._timeout
    return max(
        self._timeout,
        timedelta(seconds=timeout + _COMMAND_TTL_GRACE),
    )

  @staticmethod
  def _resolve_path(path: str | os.PathLike[str]) -> str:
    pure = PurePosixPath(os.fspath(path))
    if pure.is_absolute():
      return str(pure)
    return str(PurePosixPath(_SANDBOX_HOME) / pure)

  @staticmethod
  def _command_timeout(timeout: float | None) -> timedelta | None:
    if timeout is None:
      return None
    return timedelta(milliseconds=max(1, math.ceil(timeout * 1000)))

  @staticmethod
  def _stdout(execution: Execution) -> str:
    chunks: list[str | None] = [
        message.text for message in execution.logs.stdout
    ]
    chunks.extend(result.text for result in execution.result)
    return "".join(chunk for chunk in chunks if chunk)

  @staticmethod
  def _stderr(execution: Execution) -> str:
    return "".join(
        message.text for message in execution.logs.stderr if message.text
    )

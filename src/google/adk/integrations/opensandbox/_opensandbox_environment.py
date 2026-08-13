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

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
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
  from collections.abc import AsyncIterator

  from opensandbox import Sandbox
  from opensandbox.config import ConnectionConfig
  from opensandbox.models.execd import Execution
  from opensandbox.models.sandboxes import SandboxImageSpec

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_IMAGE = "python:3.11"
_DEFAULT_SANDBOX_TIMEOUT = 300
_DEFAULT_READY_TIMEOUT = 30
_DEFAULT_REQUEST_TIMEOUT = 30
_MIN_SANDBOX_TIMEOUT = 60
_COMMAND_TTL_GRACE = timedelta(seconds=30)
_MAX_RENEW_INTERVAL_SECONDS = 60
_SANDBOX_HOME = "/workspace"
# OpenSandbox's WriteEntry schema uses decimal digits for Unix modes (for
# example, its SDK default is 755), rather than Python's raw octal bit value.
_FILE_MODE = 644
_HTTP_NOT_FOUND = 404
_TIMEOUT_EXIT_CODE = -1


@experimental(FeatureName.OPENSANDBOX_ENVIRONMENT)
class OpenSandboxEnvironment(BaseEnvironment):
  """A persistent remote workspace backed by OpenSandbox.

  By default, ``initialize()`` creates a sandbox and ``close()`` destroys it.
  When ``sandbox_id`` is supplied, the environment attaches to that sandbox;
  ``close()`` then releases only local SDK resources and leaves the caller-owned
  remote sandbox running.

  Requires the ``opensandbox`` extra:
  ``pip install google-adk[opensandbox]``.
  """

  def __init__(  # pylint: disable=too-many-arguments
      self,
      *,
      image: str | SandboxImageSpec | None = None,
      snapshot_id: str | None = None,
      sandbox_id: str | None = None,
      timeout: int | None = _DEFAULT_SANDBOX_TIMEOUT,
      ready_timeout: float = _DEFAULT_READY_TIMEOUT,
      working_dir: str | os.PathLike[str] = _SANDBOX_HOME,
      env_vars: dict[str, str] | None = None,
      metadata: dict[str, str] | None = None,
      connection_config: ConnectionConfig | None = None,
      api_key: str | None = None,
      domain: str | None = None,
      protocol: str | None = None,
      request_timeout: float | None = None,
      use_server_proxy: bool | None = None,
  ):
    """Create an OpenSandbox environment.

    Args:
      image: Container image for a newly created sandbox. Defaults to
        ``python:3.11``. Mutually exclusive with ``snapshot_id``.
      snapshot_id: Snapshot used instead of an image for a new sandbox.
      sandbox_id: Existing sandbox to attach to instead of creating one.
      timeout: Remote sandbox lifetime in seconds. Active, owned environments
        renew this lifetime before each operation. Must be at least 60 seconds;
        ``None`` disables expiry when supported by the runtime.
      ready_timeout: Maximum seconds to wait for create or connect readiness.
      working_dir: Absolute POSIX path used for commands and relative files.
      env_vars: Environment variables applied to created sandboxes and commands.
      metadata: Additional metadata for a newly created sandbox.
      connection_config: Complete OpenSandbox connection configuration.
      api_key: OpenSandbox API key. Falls back to ``OPEN_SANDBOX_API_KEY``.
      domain: OpenSandbox service domain. Falls back to
        ``OPEN_SANDBOX_DOMAIN``.
      protocol: Protocol used when ``domain`` has no URL scheme.
      request_timeout: HTTP request timeout in seconds.
      use_server_proxy: Route sandbox requests through the lifecycle server.

    Raises:
      TypeError: If ``connection_config`` is mixed with basic connection fields.
      ValueError: If configuration values are invalid or mutually exclusive.
    """
    if image is not None and snapshot_id is not None:
      raise ValueError("image and snapshot_id are mutually exclusive")
    if sandbox_id is not None and (
        image is not None or snapshot_id is not None
    ):
      raise ValueError(
          "image and snapshot_id cannot be used when attaching by sandbox_id"
      )
    if timeout is not None and timeout < _MIN_SANDBOX_TIMEOUT:
      raise ValueError("timeout must be at least 60 seconds or None")
    if ready_timeout <= 0:
      raise ValueError("ready_timeout must be positive")
    if request_timeout is not None and request_timeout <= 0:
      raise ValueError("request_timeout must be positive")

    pure_working_dir = PurePosixPath(os.fspath(working_dir))
    if not pure_working_dir.is_absolute():
      raise ValueError("working_dir must be an absolute POSIX path")

    basic_connection_fields = (
        api_key,
        domain,
        protocol,
        request_timeout,
        use_server_proxy,
    )
    if connection_config is not None and any(
        value is not None for value in basic_connection_fields
    ):
      raise TypeError(
          "connection_config cannot be combined with basic connection fields"
      )

    self._image = image
    self._snapshot_id = snapshot_id
    self._requested_sandbox_id = sandbox_id
    self._sandbox_timeout = (
        timedelta(seconds=timeout) if timeout is not None else None
    )
    self._ready_timeout = timedelta(seconds=ready_timeout)
    self._working_dir = pure_working_dir
    self._env_vars = dict(env_vars) if env_vars is not None else None
    self._metadata = dict(metadata) if metadata is not None else None
    self._connection_config = connection_config
    self._api_key = api_key
    self._domain = domain
    self._protocol = protocol
    self._request_timeout = request_timeout
    self._use_server_proxy = use_server_proxy
    self._sandbox: Sandbox | None = None
    self._owns_remote = False
    self._pending_destroy_id: str | None = None
    self._known_expiration: datetime | None = None
    self._renew_lock = asyncio.Lock()
    self._active_operations = 0
    self._closing = False
    self._close_task: asyncio.Task[None] | None = None
    self._lifecycle_condition = asyncio.Condition()

  @property
  @override
  def working_dir(self) -> Path:
    if self._sandbox is None:
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    return Path(str(self._working_dir))

  @override
  async def initialize(self) -> None:
    async with self._lifecycle_condition:
      await self._lifecycle_condition.wait_for(lambda: not self._closing)
      if self._sandbox is not None:
        return
      await self._retry_pending_destroy()

      sandbox, owns_remote = await self._open_sandbox()
      try:
        await self._prepare_working_directory(sandbox)
      except BaseException:
        cleanup_task = asyncio.create_task(
            self._cleanup_sandbox(sandbox, owns_remote=owns_remote)
        )
        try:
          await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
          try:
            await cleanup_task
          except BaseException:  # pylint: disable=broad-exception-caught
            if owns_remote:
              self._pending_destroy_id = sandbox.id
            logger.warning(
                "Failed to clean up OpenSandbox after initialization was "
                "cancelled",
                exc_info=True,
            )
          raise
        except BaseException:  # pylint: disable=broad-exception-caught
          if owns_remote:
            self._pending_destroy_id = sandbox.id
          logger.warning(
              "Failed to clean up OpenSandbox after initialization failure",
              exc_info=True,
          )
        raise

      self._sandbox = sandbox
      self._owns_remote = owns_remote
      self._known_expiration = None
      self._is_initialized = True

  @override
  async def close(self) -> None:
    async with self._lifecycle_condition:
      close_task = self._close_task
      if close_task is None:
        self._closing = True
        close_task = asyncio.create_task(self._close_once())
        close_task.add_done_callback(self._log_close_task_failure)
        self._close_task = close_task

    await asyncio.wait({close_task})
    close_task.result()

  @staticmethod
  def _log_close_task_failure(close_task: asyncio.Task[None]) -> None:
    if close_task.cancelled():
      return
    error = close_task.exception()
    if error is not None:
      logger.warning(
          "OpenSandbox cleanup task failed",
          exc_info=(type(error), error, error.__traceback__),
      )

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

    sdk_timeout = self._command_timeout(timeout)
    renewal_timeout = self._command_renewal_timeout(timeout)
    async with self._sandbox_operation(
        renewal_timeout=renewal_timeout
    ) as sandbox:
      started_at = monotonic()
      opts = RunCommandOpts(
          working_directory=str(self._working_dir),
          timeout=sdk_timeout,
          envs=self._env_vars,
      )
      execution = await sandbox.commands.run(command, opts=opts)
      elapsed = monotonic() - started_at

    if execution.exit_code is None:
      error = execution.error
      detail = (
          f"{error.name}: {error.value}"
          if error is not None
          else "no completion or error event"
      )
      raise RuntimeError(
          "OpenSandbox command completed without an exit code: " + detail
      )

    return ExecutionResult(
        exit_code=execution.exit_code,
        stdout=self._stdout(execution),
        stderr=self._stderr(execution),
        timed_out=(
            timeout is not None
            and execution.exit_code == _TIMEOUT_EXIT_CODE
            and elapsed >= timeout
        ),
    )

  @override
  async def read_file(self, path: str | os.PathLike[str]) -> bytes:
    from opensandbox.exceptions import SandboxApiException

    resolved = self._resolve_path(path)
    async with self._sandbox_operation() as sandbox:
      try:
        return await sandbox.files.read_bytes(resolved)
      except SandboxApiException as e:
        if e.status_code == _HTTP_NOT_FOUND:
          raise FileNotFoundError(resolved) from e
        raise

  @override
  async def write_file(
      self,
      path: str | os.PathLike[str],
      content: str | bytes,
  ) -> None:
    async with self._sandbox_operation() as sandbox:
      await sandbox.files.write_file(
          self._resolve_path(path), content, mode=_FILE_MODE
      )

  async def _open_sandbox(self) -> tuple[Sandbox, bool]:
    try:
      from opensandbox import Sandbox
    except ImportError as e:
      raise ImportError(
          "The opensandbox package is required to use OpenSandboxEnvironment. "
          "Install it with `pip install google-adk[opensandbox]`."
      ) from e

    connection_config = self._build_connection_config()
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
        timeout=self._sandbox_timeout,
        ready_timeout=self._ready_timeout,
        env=self._env_vars,
        metadata=metadata,
        connection_config=connection_config,
    )
    return sandbox, True

  async def _close_once(self) -> None:
    current_task = asyncio.current_task()
    try:
      async with self._lifecycle_condition:
        await self._lifecycle_condition.wait_for(
            lambda: self._active_operations == 0
        )
        sandbox = self._sandbox
        if sandbox is None:
          await self._retry_pending_destroy()
          return

        owns_remote = self._owns_remote
        sandbox_id = sandbox.id
        self._sandbox = None
        self._owns_remote = False
        self._known_expiration = None
        self._is_initialized = False

      try:
        await self._cleanup_sandbox(sandbox, owns_remote=owns_remote)
      except BaseException:  # pylint: disable=broad-exception-caught
        if owns_remote:
          self._pending_destroy_id = sandbox_id
        raise
    finally:
      async with self._lifecycle_condition:
        if self._close_task is current_task:
          self._close_task = None
        self._closing = False
        self._lifecycle_condition.notify_all()

  def _build_connection_config(self) -> ConnectionConfig:
    if self._connection_config is not None:
      return self._connection_config

    from opensandbox.config import ConnectionConfig

    return ConnectionConfig(
        api_key=self._api_key,
        domain=self._domain,
        protocol=self._resolve_protocol(),
        request_timeout=timedelta(
            seconds=self._request_timeout or _DEFAULT_REQUEST_TIMEOUT
        ),
        use_server_proxy=self._use_server_proxy or False,
    )

  async def _prepare_working_directory(self, sandbox: Sandbox) -> None:
    from opensandbox.models.filesystem import WriteEntry

    await sandbox.files.create_directories(
        [WriteEntry(path=str(self._working_dir), mode=755)]
    )

  @staticmethod
  async def _cleanup_sandbox(
      sandbox: Sandbox,
      *,
      owns_remote: bool,
  ) -> None:
    if not owns_remote:
      await sandbox.close()
      return

    from opensandbox.exceptions import SandboxApiException

    try:
      await sandbox.destroy()
    except SandboxApiException as e:
      if e.status_code != _HTTP_NOT_FOUND:
        raise

  @asynccontextmanager
  async def _sandbox_operation(
      self,
      *,
      renewal_timeout: timedelta | None = None,
  ) -> AsyncIterator[Sandbox]:
    async with self._lifecycle_condition:
      if self._closing:
        raise RuntimeError("Sandbox is closing. Wait for close() to finish.")
      sandbox = self._sandbox
      if sandbox is None:
        raise RuntimeError("Sandbox is not started. Call initialize() first.")
      owns_remote = self._owns_remote
      if renewal_timeout is None:
        renewal_timeout = self._sandbox_timeout
      self._active_operations += 1

    operation_failed = False
    stop: asyncio.Event | None = None
    heartbeat: asyncio.Task[None] | None = None
    try:
      if owns_remote and renewal_timeout is not None:
        await self._renew_sandbox(sandbox, renewal_timeout)
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._renew_until_stopped(sandbox, stop, renewal_timeout)
        )
      yield sandbox
    except BaseException:
      operation_failed = True
      raise
    finally:
      try:
        if stop is not None and heartbeat is not None:
          stop.set()
          try:
            await heartbeat
          except Exception:  # pylint: disable=broad-exception-caught
            if not operation_failed:
              raise
            logger.warning(
                "OpenSandbox renewal failed while an operation was exiting",
                exc_info=True,
            )
      finally:
        async with self._lifecycle_condition:
          self._active_operations -= 1
          if self._active_operations == 0:
            self._lifecycle_condition.notify_all()

  async def _renew_sandbox(
      self,
      sandbox: Sandbox,
      renewal_timeout: timedelta,
  ) -> None:
    async with self._renew_lock:
      requested_expiration = datetime.now(timezone.utc) + renewal_timeout
      if (
          self._known_expiration is not None
          and requested_expiration <= self._known_expiration
      ):
        return
      response = await sandbox.renew(renewal_timeout)
      self._known_expiration = response.expires_at

  async def _renew_until_stopped(
      self,
      sandbox: Sandbox,
      stop: asyncio.Event,
      renewal_timeout: timedelta,
  ) -> None:
    interval = min(
        renewal_timeout.total_seconds() / 2,
        _MAX_RENEW_INTERVAL_SECONDS,
    )
    while True:
      try:
        await asyncio.wait_for(stop.wait(), timeout=interval)
        return
      except asyncio.TimeoutError:
        await self._renew_sandbox(sandbox, renewal_timeout)

  async def _retry_pending_destroy(self) -> None:
    sandbox_id = self._pending_destroy_id
    if sandbox_id is None:
      return

    from opensandbox import Sandbox
    from opensandbox.exceptions import SandboxApiException

    try:
      sandbox = await Sandbox.connect(
          sandbox_id,
          connection_config=self._build_connection_config(),
          skip_health_check=True,
      )
      await self._cleanup_sandbox(sandbox, owns_remote=True)
    except SandboxApiException as e:
      if e.status_code != _HTTP_NOT_FOUND:
        raise
    self._pending_destroy_id = None

  def _resolve_protocol(self) -> str:
    domain = self._domain or os.getenv("OPEN_SANDBOX_DOMAIN")
    if domain is not None:
      scheme = urlsplit(domain).scheme.lower()
      if scheme in {"http", "https"}:
        return scheme
    return self._protocol or "http"

  def _command_renewal_timeout(self, timeout: float | None) -> timedelta | None:
    sandbox_timeout = self._sandbox_timeout
    if sandbox_timeout is None or timeout is None:
      return sandbox_timeout
    command_lifetime = timedelta(seconds=timeout) + _COMMAND_TTL_GRACE
    return max(sandbox_timeout, command_lifetime)

  def _resolve_path(self, path: str | os.PathLike[str]) -> str:
    pure = PurePosixPath(os.fspath(path))
    if pure.is_absolute():
      return str(pure)
    return str(self._working_dir / pure)

  @staticmethod
  def _command_timeout(timeout: float | None) -> timedelta | None:
    if timeout is None:
      return None
    timeout_ms = max(1, math.ceil(timeout * 1000))
    return timedelta(milliseconds=timeout_ms)

  @staticmethod
  def _stdout(execution: Execution) -> str:
    chunks = [message.text for message in execution.logs.stdout]
    chunks.extend(
        result.text for result in execution.result if result.text is not None
    )
    return "\n".join(chunk.rstrip("\r\n") for chunk in chunks)

  @staticmethod
  def _stderr(execution: Execution) -> str:
    return "\n".join(
        message.text.rstrip("\r\n") for message in execution.logs.stderr
    )

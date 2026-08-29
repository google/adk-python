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

"""Local subprocess code execution environment."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import shutil
import signal
import tempfile

from typing_extensions import override

from ..utils.feature_decorator import experimental
from ._base_environment import BaseEnvironment
from ._base_environment import ExecutionResult

logger = logging.getLogger('google_adk.' + __name__)

# How long to wait for a command to exit after SIGTERM before escalating to
# SIGKILL, and then for its output pipes to close, so that tearing a command
# down cannot itself block forever.
_TERMINATE_GRACE_SECONDS = 5


def _signal_group(group: int, sig: int) -> None:
  """Signals every process left in a group, tolerating an empty one."""
  if not hasattr(os, 'killpg'):
    return
  try:
    os.killpg(group, sig)
  except OSError:
    logger.debug('Could not signal the command process group.')


def _terminate(proc: asyncio.subprocess.Process) -> None:
  """Sends SIGTERM to a command, tolerating one that already exited."""
  try:
    proc.terminate()
  except ProcessLookupError:
    pass


def _kill(proc: asyncio.subprocess.Process) -> None:
  """Sends SIGKILL to a command, tolerating one that already exited."""
  try:
    proc.kill()
  except ProcessLookupError:
    pass


@experimental
class LocalEnvironment(BaseEnvironment):
  """Execute commands via local ``asyncio`` subprocesses.

  When ``working_dir`` is not specified, a temporary directory is
  created on ``initialize()`` and removed on ``close()``.
  """

  def __init__(
      self,
      *,
      working_dir: Path | None = None,
      env_vars: dict[str, str] | None = None,
  ):
    """Create a local environment.

    Args:
      working_dir: Absolute path to the workspace directory.  If
        ``None``, a temporary directory is created during
        ``initialize()``.
      env_vars: Extra environment variables merged into the subprocess
        environment.
    """
    self._working_dir = working_dir
    self._env_vars = env_vars
    self._auto_created = False
    self._is_initialized = False

  @property
  @override
  def working_dir(self) -> Path:
    if self._working_dir is None:
      raise RuntimeError('`working_dir` is not set. Call initialize() first.')
    return self._working_dir

  @override
  async def initialize(self) -> None:
    if self._working_dir is None:
      self._working_dir = Path(tempfile.mkdtemp(prefix='adk_workspace_'))
      self._auto_created = True
      logger.debug('Created temporary folder: %s', self._working_dir)
    else:
      os.makedirs(self._working_dir, exist_ok=True)
    self._is_initialized = True

  @override
  async def close(self) -> None:
    if self._auto_created and self._working_dir:
      shutil.rmtree(self._working_dir, ignore_errors=True)
      logger.debug('Removed temporary workspace: %s', self._working_dir)
      self._working_dir = None
    self._is_initialized = False

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
  ) -> ExecutionResult:
    if self._working_dir is None:
      raise RuntimeError('`working_dir` is not set. Call initialize() first.')

    proc_env = os.environ.copy()
    if self._env_vars:
      proc_env.update(self._env_vars)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=self._working_dir,
        env=proc_env,
        # Its own session, so a timeout can take down everything the command
        # started and nothing else. Ignored on Windows, as `os.killpg` is.
        start_new_session=True,
    )

    # One drain task for the whole call. The pipes stay open while any
    # descendant holds them, so a second `communicate()` after the deadline
    # would wait on a child the timeout never reached.
    drain = asyncio.ensure_future(proc.communicate())

    timed_out = False
    try:
      stdout_bytes, stderr_bytes = await asyncio.wait_for(
          asyncio.shield(drain), timeout=timeout
      )
    except asyncio.TimeoutError:
      timed_out = True
      stdout_bytes, stderr_bytes = await self._kill_command(proc, drain)
    except asyncio.CancelledError:
      await self._kill_command(proc, drain)
      raise

    return ExecutionResult(
        exit_code=proc.returncode or 0,
        stdout=stdout_bytes.decode('utf-8', errors='replace'),
        stderr=stderr_bytes.decode('utf-8', errors='replace'),
        timed_out=timed_out,
    )

  @staticmethod
  async def _kill_command(
      proc: asyncio.subprocess.Process,
      drain: asyncio.Future[tuple[bytes, bytes]],
  ) -> tuple[bytes, bytes]:
    """Kills a command and its descendants, returning what it wrote."""
    # The command leads its own process group, so its pid is the group that
    # holds whatever it spawned. `terminate` and `kill` below reach the command
    # itself on the platforms that have no group to signal.
    group = proc.pid

    # SIGTERM first, so the command and its children get a chance to exit
    # cleanly before anything is killed outright.
    _signal_group(group, signal.SIGTERM)
    _terminate(proc)
    done, _ = await asyncio.wait([drain], timeout=_TERMINATE_GRACE_SECONDS)

    if not done:
      # Escalate unconditionally: the command exiting says nothing about a
      # child of it that is ignoring SIGTERM, and such a child holds the
      # output pipes open besides.
      _signal_group(group, signal.SIGKILL)
      _kill(proc)
      done, _ = await asyncio.wait([drain], timeout=_TERMINATE_GRACE_SECONDS)

    if not done:
      # A descendant escaped the group (it made one of its own) and still
      # holds the pipes. Give up on its output rather than wait for it.
      drain.cancel()
      logger.warning('Gave up reading output from a killed command.')
      return b'', b''
    return drain.result()

  @override
  async def read_file(self, path: str | Path) -> bytes:
    if self._working_dir is None:
      raise RuntimeError('`working_dir` is not set. Call initialize() first.')

    resolved = self._resolve_path(path)
    return await asyncio.to_thread(self._sync_read, resolved)

  @override
  async def write_file(self, path: str | Path, content: str | bytes) -> None:
    if self._working_dir is None:
      raise RuntimeError('`working_dir` is not set. Call initialize() first.')

    resolved = self._resolve_path(path)
    return await asyncio.to_thread(self._sync_write, resolved, content)

  def _resolve_path(self, path: str | Path) -> Path:
    """Resolve a file path inside the working directory."""
    candidate = Path(path)
    working_dir = self.working_dir.resolve()
    if not candidate.is_absolute():
      candidate = working_dir / candidate

    resolved = candidate.resolve()
    if not resolved.is_relative_to(working_dir):
      raise ValueError(f'Path escapes working directory: {path}')
    return resolved

  @staticmethod
  def _sync_read(path: Path) -> bytes:
    with open(path, 'rb') as f:
      return f.read()

  @staticmethod
  def _sync_write(path: Path, content: str | bytes) -> None:
    os.makedirs(path.parent, exist_ok=True)
    if isinstance(content, str):
      with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    else:
      with open(path, 'wb') as f:
        f.write(content)

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
import tempfile
from typing import Optional

from typing_extensions import override

from ..utils.feature_decorator import experimental
from ._base_environment import BaseEnvironment
from ._base_environment import ExecutionResult

logger = logging.getLogger('google_adk.' + __name__)


@experimental
class LocalEnvironment(BaseEnvironment):
  """Execute commands via local ``asyncio`` subprocesses.

  When ``working_dir`` is not specified, a temporary directory is
  created on ``initialize()`` and removed on ``close()``.
  """

  def __init__(
      self,
      *,
      working_dir: Optional[Path] = None,
      env_vars: Optional[dict[str, str]] = None,
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

  @override
  async def close(self) -> None:
    if self._auto_created and self._working_dir:
      shutil.rmtree(self._working_dir, ignore_errors=True)
      logger.debug('Removed temporary workspace: %s', self._working_dir)
      self._working_dir = None

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
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
    )

    timed_out = False
    try:
      stdout_bytes, stderr_bytes = await asyncio.wait_for(
          proc.communicate(), timeout=timeout
      )
    except asyncio.TimeoutError:
      timed_out = True
      proc.kill()
      stdout_bytes, stderr_bytes = await proc.communicate()

    return ExecutionResult(
        exit_code=proc.returncode or 0,
        stdout=stdout_bytes.decode('utf-8', errors='replace'),
        stderr=stderr_bytes.decode('utf-8', errors='replace'),
        timed_out=timed_out,
    )

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

  def _resolve_path(self, path: str | Path) -> str:
    """Resolve a relative path against the working directory."""
    path = str(path)
    if os.path.isabs(path):
      return path
    return os.path.join(self._working_dir, path)

  @staticmethod
  def _sync_read(path: str) -> bytes:
    with open(path, 'rb') as f:
      return f.read()

  @staticmethod
  def _sync_write(path: str, content: str | bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'w' if isinstance(content, str) else 'wb'
    kwargs = {'encoding': 'utf-8'} if isinstance(content, str) else {}
    with open(path, mode, **kwargs) as f:
      f.write(content)

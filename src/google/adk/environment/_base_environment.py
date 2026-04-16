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

"""Base class for agent environments."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import dataclasses
from pathlib import Path
from typing import Optional

from ..utils.feature_decorator import experimental


@dataclasses.dataclass
class ExecutionResult:
  """Result of a command execution."""

  exit_code: int = 0
  """The exit code of the process."""

  stdout: str = ""
  """Standard output captured from the process."""

  stderr: str = ""
  """Standard error captured from the process."""

  timed_out: bool = False
  """Whether the execution exceeded the timeout."""


@experimental
class BaseEnvironment(ABC):
  """Abstract base class for code execution environments.

  An environment provides the ability to execute shell commands,
  read files, and write files within a working directory.  Concrete
  implementations include local subprocess execution, sandboxed
  execution, container environments, and cloud-hosted environments.

  Lifecycle:
    1. Construct the environment (``__init__``).
    2. Call ``initialize()`` before first use.
    3. Use ``execute``, ``read_file``, ``write_file``.
    4. Call ``close()`` when done.
  """

  async def initialize(self) -> None:
    """Initialize the environment (e.g. create working directory).

    Called before first use. The default implementation is a
    no-op. Sub-classes should ensure this method is idempotent.
    """

  async def close(self) -> None:
    """Release resources held by the environment.

    Called when the environment is no longer needed. The default
    implementation is a no-op. Sub-classes should ensure this method is
    idempotent.
    """

  @property
  @abstractmethod
  def working_dir(self) -> Path:
    """The absolute path to the environment's working directory."""

  @abstractmethod
  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
  ) -> ExecutionResult:
    """Execute a shell command in the working directory.

    Args:
      command: The shell command string to execute.
      timeout: Maximum execution time in seconds.  ``None`` means
        no limit.

    Returns:
      An ``ExecutionResult`` with exit code, stdout, stderr, and
      timeout status.
    """

  @abstractmethod
  async def read_file(self, path: Path) -> bytes:
    """Read a file from the environment filesystem.

    Args:
      path: Absolute or working-dir-relative path to the file.

    Returns:
      The raw file contents as bytes.

    Raises:
      FileNotFoundError: If the file does not exist.
    """

  @abstractmethod
  async def write_file(self, path: Path, content: str | bytes) -> None:
    """Write content to a file in the environment's filesystem.

    Parent directories are created automatically if they do not
    exist.

    Args:
      path: Absolute or working-dir-relative path to the file.
      content: The string or raw bytes to write.
    """

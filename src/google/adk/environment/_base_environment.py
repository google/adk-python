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
from collections.abc import AsyncGenerator
import dataclasses
import functools
import inspect
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
  from ..agents.context import Context

# Methods that callers pass a `ctx` keyword argument to.
_CONTEXT_METHODS = (
    "initialize",
    "execute",
    "read_file",
    "read_file_stream",
    "write_file",
)


def _accepts_context(method: Callable[..., Any]) -> bool:
  """Returns whether a method can be called with a `ctx` keyword."""
  try:
    parameters = inspect.signature(method).parameters.values()
  except (TypeError, ValueError):
    return True
  return any(
      p.kind is inspect.Parameter.VAR_KEYWORD
      or (p.name == "ctx" and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
      for p in parameters
  )


def _drop_context(method: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps a method written before `ctx` existed so callers can pass it."""
  if inspect.iscoroutinefunction(method):

    @functools.wraps(method)
    async def async_wrapper(
        self: Any, *args: Any, ctx: Any = None, **kwargs: Any
    ) -> Any:
      del ctx
      return await method(self, *args, **kwargs)

    return async_wrapper

  # Also covers async generators such as `read_file_stream`: the wrapper
  # returns the generator the method creates.
  @functools.wraps(method)
  def wrapper(self: Any, *args: Any, ctx: Any = None, **kwargs: Any) -> Any:
    del ctx
    return method(self, *args, **kwargs)

  return wrapper


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


class BaseEnvironment(ABC):
  """Abstract base class for code execution environments.

  An environment provides the ability to execute shell commands,
  read files, and write files within a working directory.  Concrete
  implementations include local subprocess execution, sandboxed
  execution, container environments, and cloud-hosted environments.

  Lifecycle:
    1. Construct the environment (``__init__``).
    2. Call ``initialize()`` before first use.
    3. Use ``execute``, ``read_file``, ``read_file_stream``, ``write_file``.
    4. Call ``close()`` when done.

  Callers inside an agent run pass the current context as ``ctx`` to
  ``initialize``, ``execute``, ``read_file``, ``read_file_stream`` and
  ``write_file``, so an implementation can key its resources off the
  session or user, read and write session state, or request credentials.
  ``ctx`` is ``None`` when the environment is used outside an agent run.
  The toolsets currently call ``initialize`` while building the model
  request rather than from a tool call, so state written through its
  ``ctx`` is not saved and ``ctx.request_credential`` raises there.
  """

  _is_initialized: bool = False

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    # Subclasses written before these methods took `ctx` would fail with a
    # TypeError once callers pass it. Wrap them so they keep working and the
    # context is dropped.
    # TODO: Remove the wrapping once overrides without `ctx` are no longer
    # supported.
    outdated = []
    for name in _CONTEXT_METHODS:
      method = cls.__dict__.get(name)
      if not inspect.isfunction(method) or _accepts_context(method):
        continue
      setattr(cls, name, _drop_context(method))
      outdated.append(name)
    if outdated:
      warnings.warn(
          f"{cls.__qualname__} overrides {', '.join(outdated)} without a"
          " `ctx` keyword argument. Add `*, ctx=None` to the"
          " signature; overrides without it will break when this"
          " compatibility support is removed in a future release.",
          DeprecationWarning,
          # Skip this method and ABCMeta.__new__ to point at the class body.
          stacklevel=3,
      )

  @property
  def is_initialized(self) -> bool:
    """Whether the environment has been initialized."""
    return self._is_initialized

  @is_initialized.setter
  def is_initialized(self, value: bool) -> None:
    self._is_initialized = value

  async def initialize(self, *, ctx: Context | None = None) -> None:
    """Initialize the environment (e.g. create working directory).

    Called before first use. The default implementation is a
    no-op. Sub-classes should ensure this method is idempotent.

    Args:
      ctx: Context of the agent run that triggered initialization, or
        ``None`` when called outside an agent run.
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
      timeout: float | None = None,
      ctx: Context | None = None,
  ) -> ExecutionResult:
    """Execute a shell command in the working directory.

    Args:
      command: The shell command string to execute.
      timeout: Maximum execution time in seconds.  ``None`` means
        no limit.
      ctx: Context of the calling agent run, or ``None`` when called
        outside an agent run.

    Returns:
      An ``ExecutionResult`` with exit code, stdout, stderr, and
      timeout status.
    """

  @abstractmethod
  async def read_file(self, path: Path, *, ctx: Context | None = None) -> bytes:
    """Read a file from the environment filesystem.

    Args:
      path: Absolute or working-dir-relative path to the file.
      ctx: Context of the calling agent run, or ``None`` when called
        outside an agent run.

    Returns:
      The raw file contents as bytes.

    Raises:
      FileNotFoundError: If the file does not exist.
    """

  async def read_file_stream(
      self, path: Path, *, ctx: Context | None = None
  ) -> AsyncGenerator[bytes, None]:
    """Yields the same bytes read_file(path) returns, in chunks.

    The default yields the whole file at once; override it to stream.

    Args:
      path: Absolute or working-dir-relative path to the file.
      ctx: Context of the calling agent run, or ``None`` when called
        outside an agent run.

    Yields:
      Consecutive chunks of the raw file contents.

    Raises:
      FileNotFoundError: If the file does not exist.
    """
    yield await self.read_file(path, ctx=ctx)

  @abstractmethod
  async def write_file(
      self,
      path: Path,
      content: str | bytes,
      *,
      ctx: Context | None = None,
  ) -> None:
    """Write content to a file in the environment's filesystem.

    Parent directories are created automatically if they do not
    exist.

    Args:
      path: Absolute or working-dir-relative path to the file.
      content: The string or raw bytes to write.
      ctx: Context of the calling agent run, or ``None`` when called
        outside an agent run.
    """

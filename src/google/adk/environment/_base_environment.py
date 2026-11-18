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
import os
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
import warnings

from ._environment_scope import EnvironmentScope

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

  @property
  def scope(self) -> EnvironmentScope:
    """Which agent runs share one resource of this environment.

    The environment and skill toolsets run ``initialize`` one call at a time
    within a scope: for the whole environment with ``SHARED``, per user with
    ``USER`` and per session with ``SESSION``. An environment that keeps a
    resource per user or per session overrides this to say so, which lets
    other users or sessions initialize at the same time. Such an environment
    must make ``initialize`` safe to run concurrently for different users or
    sessions, including from different threads. With every scope, a toolset
    closes the environment only while no ``initialize`` call is running.

    The toolsets serialize these calls within one process. When the agent
    runs as several instances, calls that reach different instances are not
    serialized.

    The value must not change during the life of the environment. The
    default implementation returns ``SHARED``.
    """
    return EnvironmentScope.SHARED

  async def initialize(self, *, ctx: Context | None = None) -> None:
    """Initialize the environment (e.g. create working directory).

    Called before first use. The default implementation is a
    no-op. Sub-classes should ensure this method is idempotent.

    The environment and skill toolsets call it before every tool call
    that uses the environment, with that call's ``ctx``, so it should
    return quickly once the environment is set up. An environment that
    keeps resources per session, such as a sandbox whose name it stores
    in ``ctx.state``, looks them up here and creates them when missing;
    ``is_initialized`` describes only the environment as a whole.

    An exception raised here is raised from the tool call that triggered
    it, so it reaches ``on_tool_error_callback``.

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
  def working_dir(self) -> Path | None:
    """The absolute path to the environment's working directory.

    ``None`` when the environment does not know its working directory yet,
    for example a temporary directory that ``initialize`` creates. An
    environment whose working directory is a fixed path returns it before
    ``initialize``. The default implementation returns ``None``.

    The property takes no context, so an environment with a ``USER`` or
    ``SESSION`` scope uses the same path in each of its resources.
    """
    return None

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


def resolve_sandbox_path(
    path: str | os.PathLike[str], workspace_root: str
) -> str:
  """Resolves a file path against a remote sandbox workspace root.

  Relative paths are anchored at ``workspace_root`` and normalized lexically;
  a relative path whose normalization escapes the root raises ``ValueError``,
  so model-supplied ``..`` segments cannot reach files outside the workspace.
  An absolute path passes through unchanged unless it starts with
  ``workspace_root``: such a path claims to be inside the workspace, so it is
  validated the same way and a ``..`` traversal spelled in absolute form is
  refused too. Normalization is lexical only: the filesystem that would
  resolve symlinks lives inside the sandbox, not in this process.

  Args:
    path: The model-supplied path to resolve.
    workspace_root: The absolute workspace root relative paths anchor at.

  Returns:
    The resolved POSIX path.

  Raises:
    ValueError: If a relative path escapes ``workspace_root``, or an absolute
      path under ``workspace_root`` normalizes outside it.
  """
  candidate = PurePosixPath(os.fspath(path))
  if candidate.is_absolute():
    absolute = str(candidate)
    if absolute != workspace_root and not absolute.startswith(
        workspace_root + "/"
    ):
      return absolute
    candidate = PurePosixPath(posixpath.normpath(absolute))
  else:
    candidate = PurePosixPath(
        posixpath.normpath(str(PurePosixPath(workspace_root) / candidate))
    )
  if not candidate.is_relative_to(PurePosixPath(workspace_root)):
    raise ValueError(f"Path escapes working directory: {path}")
  return str(candidate)

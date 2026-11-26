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

"""Plimsoll self-hosted sandbox environment."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import concurrent.futures
import contextlib
from collections.abc import Mapping
import logging
import math
import os
from pathlib import Path
from pathlib import PurePath
import re
import threading
from typing import Any
from typing import TYPE_CHECKING
from typing import TypeVar

from typing_extensions import override

from ...environment._base_environment import BaseEnvironment
from ...environment._base_environment import ExecutionResult
from ...features import experimental
from ...features import FeatureName

if TYPE_CHECKING:
  from plimsoll_client import Client
  from plimsoll_client import Session
  from plimsoll_client import Workspace

  from ...agents.context import Context

logger = logging.getLogger("google_adk." + __name__)

_T = TypeVar("_T")

_DEFAULT_URL = "http://127.0.0.1:8746"
_DEFAULT_TIMEOUT = 120.0
_TIERS = ("process", "container", "kernel", "vm")
# The names the Plimsoll workspace accepts for a command's variables.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Seconds: the daemon counts a command's budget in whole milliseconds.
_SHORTEST_BUDGET = 0.001
_CLOSED = (
    "This PlimsollEnvironment was closed, and its sandbox and files with it;"
    " it does not open an empty sandbox in their place. ADK closes an"
    " environment when a Runner that holds it closes, an AgentTool's nested"
    " Runner included: give each Runner that closes on its own a"
    " PlimsollEnvironment of its own, or pass reopen_after_close=True."
)


@experimental(FeatureName.PLIMSOLL_ENVIRONMENT)
class PlimsollEnvironment(BaseEnvironment):
  """A persistent workspace in a self-hosted Plimsoll sandbox.

  ``initialize()`` opens one Plimsoll session: a sandbox kept for many calls,
  at or above ``minimum_isolation``. Every ``execute``, ``read_file`` and
  ``write_file`` is a call of that session, so files persist between calls.
  ``close()`` closes the session, and the sandbox and its files are gone.

  The isolation floor is checked twice: the daemon refuses to open a session
  below it, and the environment checks the tier the session states before
  any call. ``kernel`` (the default) refuses a plain Docker daemon and accepts
  Docker under gVisor; ``vm`` needs a microVM provider. Pass ``container`` only
  for code you trust on a machine you own.

  Each command runs in a new ``sh`` in the working directory: files persist
  from one command to the next, a ``cd`` or an ``export`` does not. When the
  session has ended (its lifetime, or the daemon restarted), every call
  raises ``plimsoll_client.SessionEndedError``; the environment never opens a
  fresh sandbox in its place, because the agent would take an empty sandbox
  for the one holding its files. For the same reason ``initialize()`` after
  ``close()`` raises ``RuntimeError`` unless ``reopen_after_close`` is set:
  ADK initializes before every tool call, and closes the environment with any
  Runner that holds it. So don't share one environment between Runners that
  close on their own, such as an agent and the agent an ``AgentTool`` wraps.

  One environment holds one sandbox for every agent run that uses it. Give
  each end user an environment of their own.

  Requires the ``plimsoll`` extra: ``pip install google-adk[plimsoll]``.
  """

  def __init__(
      self,
      *,
      url: str | None = None,
      token: str | None = None,
      client: Client | None = None,
      minimum_isolation: str = "kernel",
      timeout: float = _DEFAULT_TIMEOUT,
      env_vars: Mapping[str, str] | None = None,
      lifetime: float | None = None,
      idle_timeout: float | None = None,
      owner: str | None = None,
      reopen_after_close: bool = False,
  ):
    """Create a Plimsoll environment. Nothing is contacted until initialize().

    Args:
      url: The Plimsoll daemon's address. Defaults to the ``PLIMSOLL_URL``
        environment variable, then ``http://127.0.0.1:8746``.
      token: The caller token. Defaults to the ``PLIMSOLL_TOKEN`` environment
        variable.
      client: A ``plimsoll_client.Client`` to use instead of ``url`` and
        ``token``, for example one with a custom TLS context.
      minimum_isolation: The weakest isolation tier the sandbox may have:
        ``process``, ``container``, ``kernel`` or ``vm``.
      timeout: Seconds a file call may take, capped at the daemon's own
        limit. A command run with no timeout of its own gets the daemon's
        limit, or this when the daemon states none.
      env_vars: Variables added to every command's environment. They enter
        the sandbox, so never put a credential here.
      lifetime: Seconds the session may live. May only shorten the daemon's.
      idle_timeout: Seconds the session may sit unused. May only shorten the
        daemon's.
      owner: The end user the session is for, which a daemon can cap.
      reopen_after_close: Whether ``initialize()`` after ``close()`` opens a
        new, empty sandbox instead of raising ``RuntimeError``.

    Raises:
      ValueError: If a configuration value is invalid.
    """
    if client is not None and (url is not None or token is not None):
      raise ValueError("client cannot be combined with url or token")
    if minimum_isolation not in _TIERS:
      raise ValueError(f"minimum_isolation must be one of {', '.join(_TIERS)}")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
      raise ValueError("timeout must be a positive number of seconds")
    variables = dict(env_vars or {})
    for name, value in variables.items():
      if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ValueError(f"env_vars: {name!r} is not a shell variable name")
      if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"env_vars: the value of {name} must be text")

    self._url = url
    self._token = token
    self._client = client
    self._minimum_isolation = minimum_isolation
    self._timeout = float(timeout)
    self._env_vars = variables
    self._open_options: dict[str, Any] = {
        "lifetime": lifetime,
        "idle_timeout": idle_timeout,
        "owner": owner,
    }
    self._session: Session | None = None
    self._workspace: Workspace | None = None
    self._working_dir: Path | None = None
    self._ceiling: float | None = None
    self._reopen_after_close = reopen_after_close
    self._closed = False
    # Sessions whose close has not yet succeeded.
    self._closing: list[Session] = []
    # The open in flight, which every initialize() call meanwhile waits for.
    self._opening: _Opening | None = None
    # Moved on by every close(), so an open that a close overtook closes what
    # it opens instead of publishing it.
    self._generation = 0
    # Guards the fields above and the open session's handles, which an open's
    # thread publishes while the event loop reads them.
    self._lock = threading.Lock()

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    del state["_lock"]
    state["_opening"] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    self.__dict__.update(state)
    self._lock = threading.Lock()

  @property
  @override
  def working_dir(self) -> Path | None:
    """The sandbox's working directory (``/work`` on Docker).

    ``None`` before ``initialize()``: the daemon's provider decides the path.
    """
    return self._working_dir

  @property
  def isolation(self) -> str | None:
    """The isolation tier the open session states, ``None`` without one.

    The tier is the provider's configuration and evidence plus the daemon's
    startup check, not a runtime attestation.
    """
    session = self._session
    return session.isolation if session is not None else None

  @override
  async def initialize(self, *, ctx: Context | None = None) -> None:
    """Opens the session; returns at once while one is open.

    Calls made while an open is in flight wait for that open rather than
    starting another, and a ``close()`` made meanwhile closes what it opens.
    """
    del ctx
    while True:
      with self._lock:
        if self._workspace is not None:
          return
        if self._closed and not self._reopen_after_close:
          raise RuntimeError(_CLOSED)
        opening = self._opening
        if opening is None and not self._closing:
          opening = self._opening = _Opening(self._generation)
          threading.Thread(
              target=self._open_in_thread,
              args=(opening,),
              name="PlimsollEnvironment open",
          ).start()
        if opening is not None:
          opening.waiters += 1
      if opening is None:
        # A sandbox whose close failed still holds one of the daemon's
        # places, so close it before opening another.
        await self._finish_close()
        continue
      try:
        # A cancelled caller leaves the open to its thread, which closes what
        # it opens when no caller waits for it any more, so no session holds
        # one of the daemon's places until its idle limit. The thread
        # decides, not this event loop: asyncio.run cancels everything when
        # it ends, and the open may finish after the loop has closed.
        await asyncio.shield(asyncio.wrap_future(opening.done))
      finally:
        with self._lock:
          opening.waiters -= 1

  def _open_in_thread(self, opening: _Opening) -> None:
    """Opens a session and publishes it, or closes it when no one wants it."""
    try:
      session, workspace, root, ceiling = self._open()
    except BaseException as e:  # pylint: disable=broad-exception-caught
      with self._lock:
        if self._opening is opening:
          self._opening = None
      opening.done.set_exception(e)
      return
    with self._lock:
      keep = opening.waiters > 0 and opening.generation == self._generation
      if keep:
        self._session = session
        self._ceiling = ceiling
        self._working_dir = Path(root)
        self._workspace = workspace
        self._is_initialized = True
        self._opening = None
    if not keep:
      # Still the open in flight while it closes, so a close() waits for it.
      self._close_or_keep(session)
      with self._lock:
        if self._opening is opening:
          self._opening = None
    opening.done.set_result(None)

  def _open(self) -> tuple[Session, Workspace, str, float | None]:
    try:
      import plimsoll_client
    except ImportError as e:
      raise ImportError(
          "PlimsollEnvironment requires the plimsoll-client package. "
          "Install it with `pip install google-adk[plimsoll]`."
      ) from e

    client = self._client
    if client is None:
      client = plimsoll_client.Client(
          self._url or os.environ.get("PLIMSOLL_URL") or _DEFAULT_URL,
          token=self._token or os.environ.get("PLIMSOLL_TOKEN"),
      )
      self._client = client
    info = client.describe()
    if not info.supports_sessions:
      raise plimsoll_client.UnsupportedError(
          f"the Plimsoll daemon ({info.provider}) offers no sessions, which"
          " PlimsollEnvironment needs; nothing ran",
          not_dispatched="unsupported",
      )
    limit_ms = info.session_environment.max_timeout_ms
    ceiling = limit_ms / 1000 if limit_ms > 0 else None
    session = client.open_session(
        minimum_isolation=self._minimum_isolation, **self._open_options
    )
    try:
      if not plimsoll_client.meets(session.isolation, self._minimum_isolation):
        raise plimsoll_client.InsufficientIsolationError(
            f"the session states {session.isolation or 'no tier'}, below the"
            f" required {self._minimum_isolation}; nothing ran",
            not_dispatched="isolation",
        )
      workspace = plimsoll_client.Workspace(
          session, timeout=self._timeout, max_timeout=ceiling, confined=False
      )
      root = workspace.root
    except BaseException:
      _close_quietly(session)
      raise
    return session, workspace, root, ceiling

  @override
  async def close(self) -> None:
    """Closes the session: the sandbox and its files are gone. Idempotent.

    No call reaches the sandbox once a close has started. A close that fails
    keeps the session, and ``is_initialized`` stays true, so the next
    ``close()`` tries again.
    """
    with self._lock:
      session = self._session
      opening = self._opening
      self._session = None
      self._workspace = None
      self._working_dir = None
      self._ceiling = None
      self._generation += 1
      if session is not None or (opening is not None and opening.waiters):
        # Set before the delete, so a delete that fails still refuses a
        # reopen. An open no caller waits for any more opened nothing for
        # anyone, as a close before any open.
        self._closed = True
      if session is not None:
        self._closing.append(session)
    if opening is not None:
      # The open's thread closes what it opens, since the generation moved
      # on; one that failed opened nothing.
      with contextlib.suppress(Exception):
        await asyncio.shield(asyncio.wrap_future(opening.done))
    await self._finish_close()

  async def _finish_close(self) -> None:
    """Closes the sessions closes took; raises if one fails, keeping it."""
    while True:
      with self._lock:
        if not self._closing:
          if self._session is None and self._opening is None:
            self._is_initialized = False
          return
        session = self._closing[0]
      try:
        await asyncio.to_thread(session.close)
      except Exception as e:
        if not _close_is_final(e):
          raise
        # Raised once: the session is no longer this environment's to close.
        self._drop_closing(session)
        raise
      self._drop_closing(session)

  def _drop_closing(self, session: Session) -> None:
    with self._lock:
      self._closing = [s for s in self._closing if s is not session]

  @override
  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
      ctx: Context | None = None,
  ) -> ExecutionResult:
    """Runs ``command`` in a new ``sh`` in the working directory.

    A non-zero exit is a result, not an error. ``timeout`` ``None`` gets the
    daemon's longest budget; ``0`` sends the shortest budget the daemon
    takes, one millisecond, as ``OpenSandboxEnvironment`` does. When the daemon
    cut a stream at its output limit, a ``plimsoll:`` line at the end of
    ``stderr`` says so.

    Raises:
      ValueError: If ``timeout`` is negative, not finite, or not a number.
    """
    del ctx
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
      raise ValueError("timeout must be a non-negative number of seconds")
    workspace = self._require()
    if timeout is None:
      budget = self._ceiling if self._ceiling is not None else self._timeout
    else:
      budget = max(float(timeout), _SHORTEST_BUDGET)
    out = await self._call(
        workspace.run, ["sh", "-c", command], env=self._env_vars, timeout=budget
    )
    stderr = out.stderr_text
    notes = [
        f"plimsoll: the daemon cut {name} at its output limit"
        for name, cut in (
            ("stdout", out.stdout_truncated),
            ("stderr", out.stderr_truncated),
        )
        if cut
    ]
    if notes:
      if stderr and not stderr.endswith("\n"):
        stderr += "\n"
      stderr += "\n".join(notes) + "\n"
    record = out.result.record
    logger.debug(
        "Plimsoll command ran behind %s, run record %s",
        out.result.isolation,
        record.sha256 if record is not None else "none",
    )
    return ExecutionResult(
        exit_code=out.exit_code,
        stdout=out.stdout_text,
        stderr=stderr,
        timed_out=out.timed_out,
    )

  @override
  async def read_file(
      self, path: Path | str, *, ctx: Context | None = None
  ) -> bytes:
    del ctx
    workspace = self._require()
    return await self._call(workspace.read_whole, _posix(path))

  @override
  async def write_file(
      self,
      path: Path | str,
      content: str | bytes,
      *,
      ctx: Context | None = None,
  ) -> None:
    del ctx
    if isinstance(content, str):
      content = content.encode("utf-8")
    elif not isinstance(content, (bytes, bytearray)):
      raise TypeError(
          f"content must be str or bytes, not {type(content).__name__}"
      )
    workspace = self._require()
    await self._call(workspace.write, _posix(path), bytes(content))

  def _require(self) -> Workspace:
    workspace = self._workspace
    if workspace is None:
      if self._closed and not self._reopen_after_close:
        raise RuntimeError(_CLOSED)
      raise RuntimeError("Sandbox is not started. Call initialize() first.")
    return workspace

  async def _call(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Runs one blocking Plimsoll call in a worker thread.

    Cancelling the await cuts the request in flight. A call that came back
    without a usable answer is raised as a RuntimeError naming only its
    outcome, never the sandbox's text: the command may have run.
    """
    from plimsoll_client import CancelHandle
    from plimsoll_client import WorkspaceCallError

    handle = CancelHandle()
    try:
      return await asyncio.to_thread(fn, *args, cancel=handle, **kwargs)
    except asyncio.CancelledError:
      handle.cancel()
      raise
    except WorkspaceCallError as e:
      raise RuntimeError(
          f"the Plimsoll call ended {e.result.outcome} without a usable"
          " answer; it may have run, so check its effects before repeating"
          " it"
      ) from None

  def _close_or_keep(self, session: Session) -> None:
    """Closes ``session`` from a worker thread, where nothing can be raised to.

    A close that may succeed when tried again is kept for the next
    ``close()``.
    """
    try:
      session.close()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to close the Plimsoll session", exc_info=True)
      if not _close_is_final(e):
        with self._lock:
          self._closing.append(session)


def _close_is_final(error: Exception) -> bool:
  """Whether a close that raised ``error`` would raise it again, unsent.

  The client raises ``ChainError`` when the daemon answered the close (the
  session is closed there, and its count shows a call this client did not
  see, as a cut call may be), and ``StaleOwnerError`` when another worker
  holds the session. Either is final; any other error, such as a close that
  got no answer, is sent again by the next close.
  """
  import plimsoll_client

  return isinstance(
      error, (plimsoll_client.ChainError, plimsoll_client.StaleOwnerError)
  )


def _posix(path: Path | str) -> str:
  """The path as POSIX text, so a Windows host's separators never reach it."""
  if isinstance(path, PurePath):
    return path.as_posix()
  if isinstance(path, str):
    return path
  raise TypeError(f"path must be str or a path, not {type(path).__name__}")


def _close_quietly(session: Session) -> None:
  try:
    session.close()
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning("Failed to close the Plimsoll session", exc_info=True)


class _Opening:
  """An open in flight, which its own thread publishes or closes."""

  def __init__(self, generation: int) -> None:
    self.done: concurrent.futures.Future[None] = concurrent.futures.Future()
    # Running from the start, so nothing can cancel it: only the thread
    # settles it.
    self.done.set_running_or_notify_cancel()
    # The environment's generation when the open started.
    self.generation = generation
    # The initialize() calls waiting for it, counted under the environment's
    # lock; none left means no one wants what it opens.
    self.waiters = 0

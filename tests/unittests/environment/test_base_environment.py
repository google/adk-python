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

"""Tests for BaseEnvironment."""

from pathlib import Path
from typing import Any
import warnings

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
import pytest


def _define_outdated_environment() -> type[BaseEnvironment]:
  """Defines an environment whose methods predate the `ctx` argument."""

  class _OutdatedEnvironment(BaseEnvironment):
    """Records each call it receives."""

    def __init__(self):
      self.calls: list[tuple[str, tuple[Any, ...]]] = []

    @property
    def working_dir(self) -> Path:
      return Path("/workspace")

    async def initialize(self) -> None:
      self.calls.append(("initialize", ()))

    async def execute(
        self, command: str, *, timeout: float | None = None
    ) -> ExecutionResult:
      self.calls.append(("execute", (command, timeout)))
      return ExecutionResult(stdout="ok")

    async def read_file(self, path: Path) -> bytes:
      self.calls.append(("read_file", (path,)))
      return b"data"

    async def write_file(self, path: Path, content: str | bytes) -> None:
      self.calls.append(("write_file", (path, content)))

  return _OutdatedEnvironment


def _define_current_environment() -> type[BaseEnvironment]:
  """Defines an environment whose methods take `ctx` or `**kwargs`."""

  class _CurrentEnvironment(BaseEnvironment):
    """Records the context each call receives."""

    def __init__(self):
      self.contexts: list[Any] = []

    @property
    def working_dir(self) -> Path:
      return Path("/workspace")

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        ctx: Any = None,
    ) -> ExecutionResult:
      self.contexts.append(ctx)
      return ExecutionResult()

    async def read_file(self, path: Path, **kwargs: Any) -> bytes:
      self.contexts.append(kwargs.get("ctx"))
      return b""

    async def write_file(
        self, path: Path, content: str | bytes, *, ctx: Any = None
    ) -> None:
      self.contexts.append(ctx)

  return _CurrentEnvironment


def test_outdated_override_warns_at_class_definition():
  """Overrides without `ctx` raise one warning naming each of them."""
  with pytest.warns(DeprecationWarning) as record:
    _define_outdated_environment()

  assert len(record) == 1
  message = str(record[0].message)
  assert "initialize, execute, read_file, write_file" in message
  assert record[0].filename == __file__


@pytest.mark.asyncio
async def test_outdated_override_accepts_and_drops_context():
  """Callers can pass `ctx` to an override that does not take it."""
  with pytest.warns(DeprecationWarning):
    env = _define_outdated_environment()()
  ctx = object()

  await env.initialize(ctx=ctx)
  result = await env.execute("ls", timeout=5, ctx=ctx)
  data = await env.read_file(Path("a.txt"), ctx=ctx)
  await env.write_file(Path("a.txt"), "x", ctx=ctx)

  assert result.stdout == "ok"
  assert data == b"data"
  assert env.calls == [
      ("initialize", ()),
      ("execute", ("ls", 5)),
      ("read_file", (Path("a.txt"),)),
      ("write_file", (Path("a.txt"), "x")),
  ]


@pytest.mark.asyncio
async def test_outdated_override_still_works_without_context():
  """Calls that leave `ctx` out reach the override unchanged."""
  with pytest.warns(DeprecationWarning):
    env = _define_outdated_environment()()

  await env.execute("ls")

  assert env.calls == [("execute", ("ls", None))]


@pytest.mark.asyncio
async def test_current_override_receives_context_without_warning():
  """Overrides that take `ctx` or `**kwargs` are left alone."""
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    env = _define_current_environment()()
  ctx = object()

  await env.execute("ls", ctx=ctx)
  await env.read_file(Path("a.txt"), ctx=ctx)
  await env.write_file(Path("a.txt"), "x", ctx=ctx)

  assert env.contexts == [ctx, ctx, ctx]


@pytest.mark.asyncio
async def test_outdated_read_file_stream_override_accepts_and_drops_context():
  """An async generator override without `ctx` is wrapped like the others."""
  with pytest.warns(DeprecationWarning, match="read_file_stream"):

    class _OutdatedStreamEnvironment(BaseEnvironment):
      """Streams two chunks without accepting `ctx`."""

      @property
      def working_dir(self) -> Path:
        return Path("/workspace")

      async def execute(
          self, command: str, *, timeout: float | None = None, ctx: Any = None
      ) -> ExecutionResult:
        return ExecutionResult()

      async def read_file(self, path: Path, *, ctx: Any = None) -> bytes:
        return b""

      async def read_file_stream(self, path: Path):
        yield b"a"
        yield b"b"

      async def write_file(
          self, path: Path, content: str | bytes, *, ctx: Any = None
      ) -> None:
        pass

  env = _OutdatedStreamEnvironment()

  chunks = [c async for c in env.read_file_stream(Path("a.txt"), ctx=object())]

  assert chunks == [b"a", b"b"]


@pytest.mark.asyncio
async def test_default_read_file_stream_passes_context_to_read_file():
  """The default read_file_stream reads through read_file with the same ctx."""
  env = _define_current_environment()()
  ctx = object()

  chunks = [c async for c in env.read_file_stream(Path("a.txt"), ctx=ctx)]

  assert chunks == [b""]
  assert env.contexts == [ctx]


def test_outdated_override_keeps_abstract_methods_enforced():
  """A subclass missing an abstract method still cannot be instantiated."""
  with pytest.warns(DeprecationWarning):

    class _Incomplete(BaseEnvironment):
      """Lacks read_file and write_file."""

      @property
      def working_dir(self) -> Path:
        return Path("/workspace")

      async def execute(self, command: str) -> ExecutionResult:
        return ExecutionResult()

  with pytest.raises(TypeError, match="abstract"):
    _Incomplete()

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

"""Unit tests for utilities in cli."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
import types
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import click
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
import google.adk.cli.cli as cli
from google.adk.cli.utils.service_factory import create_artifact_service_from_options
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


# Helpers
class _Recorder:
  """Callable that records every invocation."""

  def __init__(self) -> None:
    self.calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

  def __call__(self, *args: Any, **kwargs: Any) -> None:
    self.calls.append((args, kwargs))


# Fixtures
@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Silence click output in every test."""
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)
  monkeypatch.setattr(click, "secho", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _patch_types_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
  """Replace google.genai.types and Runner with lightweight fakes."""

  # Dummy Part / Content
  class _Part:

    def __init__(self, text: str | None = "") -> None:
      self.text = text

  class _Content:

    def __init__(self, role: str, parts: List[_Part]) -> None:
      self.role = role
      self.parts = parts

  monkeypatch.setattr(cli.types, "Part", _Part)
  monkeypatch.setattr(cli.types, "Content", _Content)

  # Fake Runner yielding a single assistant echo
  class _FakeRunner:

    def __init__(self, *a: Any, **k: Any) -> None:
      ...

    async def run_async(self, *a: Any, **k: Any):
      message = a[2] if len(a) >= 3 else k["new_message"]
      text = message.parts[0].text if message.parts else ""
      response = _Content("assistant", [_Part(f"echo:{text}")])
      yield types.SimpleNamespace(author="assistant", content=response)

    async def close(self, *a: Any, **k: Any) -> None:
      ...

  monkeypatch.setattr(cli, "Runner", _FakeRunner)


@pytest.fixture()
def fake_agent(tmp_path: Path):
  """Create a minimal importable agent package and patch importlib."""

  parent_dir = tmp_path / "agents"
  parent_dir.mkdir()
  agent_dir = parent_dir / "fake_agent"
  agent_dir.mkdir()
  # __init__.py exposes root_agent with .name
  (agent_dir / "__init__.py").write_text(dedent("""
    from google.adk.agents.base_agent import BaseAgent
    class FakeAgent(BaseAgent):
      def __init__(self, name):
        super().__init__(name=name)

    root_agent = FakeAgent(name="fake_root")
    """))

  return parent_dir, "fake_agent"


@pytest.fixture()
def fake_app_agent(tmp_path: Path):
  """Create an agent package that exposes an App."""

  parent_dir = tmp_path / "agents"
  parent_dir.mkdir()
  agent_dir = parent_dir / "fake_app_agent"
  agent_dir.mkdir()
  (agent_dir / "__init__.py").write_text(dedent("""
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.apps.app import App
    class FakeAgent(BaseAgent):
      def __init__(self, name):
        super().__init__(name=name)

    root_agent = FakeAgent(name="fake_root")
    app = App(name="custom_cli_app", root_agent=root_agent)
    """))

  return parent_dir, "fake_app_agent", "custom_cli_app"


# _run_input_file
@pytest.mark.asyncio
async def test_run_input_file_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_input_file should echo user & assistant messages and return a populated session."""
  recorder: List[str] = []

  def _echo(msg: str) -> None:
    recorder.append(msg)

  monkeypatch.setattr(click, "echo", _echo)

  input_json = {
      "state": {"foo": "bar"},
      "queries": ["hello world"],
  }
  input_path = tmp_path / "input.json"
  input_path.write_text(json.dumps(input_json))

  artifact_service = InMemoryArtifactService()
  session_service = InMemorySessionService()
  credential_service = InMemoryCredentialService()
  dummy_root = BaseAgent(name="root")

  session = await cli.run_input_file(
      app_name="app",
      user_id="user",
      agent_or_app=dummy_root,
      artifact_service=artifact_service,
      session_service=session_service,
      credential_service=credential_service,
      input_path=str(input_path),
  )

  assert session.state["foo"] == "bar"
  assert any("[user]:" in line for line in recorder)
  assert any("[assistant]:" in line for line in recorder)


# _run_cli (input_file branch)
@pytest.mark.asyncio
async def test_run_cli_with_input_file(fake_agent, tmp_path: Path) -> None:
  """run_cli should process an input file without raising and without saving."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": ["ping"]}
  input_path = tmp_path / "in.json"
  input_path.write_text(json.dumps(input_json))

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
  )


@pytest.mark.asyncio
async def test_run_cli_loads_services_module(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should load custom services from the agents directory."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": ["ping"]}
  input_path = tmp_path / "input.json"
  input_path.write_text(json.dumps(input_json))

  loaded_dirs: list[str] = []
  monkeypatch.setattr(
      cli, "load_services_module", lambda path: loaded_dirs.append(path)
  )

  agent_root = parent_dir / folder_name

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
  )

  assert loaded_dirs == [str(agent_root.resolve())]


@pytest.mark.asyncio
async def test_run_cli_app_uses_app_name_for_sessions(
    fake_app_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should honor the App-provided name when creating sessions."""
  parent_dir, folder_name, app_name = fake_app_agent
  created_app_names: List[str] = []

  class _SpySessionService(InMemorySessionService):

    async def create_session(self, *, app_name: str, **kwargs: Any) -> Any:
      created_app_names.append(app_name)
      return await super().create_session(app_name=app_name, **kwargs)

  spy_session_service = _SpySessionService()

  def _session_factory(**_: Any) -> InMemorySessionService:
    return spy_session_service

  monkeypatch.setattr(
      cli, "create_session_service_from_options", _session_factory
  )

  input_json = {"state": {}, "queries": ["ping"]}
  input_path = tmp_path / "input_app.json"
  input_path.write_text(json.dumps(input_json))

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
  )

  assert created_app_names
  assert all(name == app_name for name in created_app_names)


# _run_cli (interactive + save session branch)
@pytest.mark.asyncio
async def test_run_cli_save_session(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should save a session file when save_session=True."""
  parent_dir, folder_name = fake_agent

  # Simulate user typing 'exit' followed by session id 'sess123'
  responses = iter(["exit", "sess123"])
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(responses))

  session_file = Path(parent_dir) / folder_name / "sess123.session.json"
  if session_file.exists():
    session_file.unlink()

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=None,
      saved_session_file=None,
      save_session=True,
  )

  assert session_file.exists()
  data = json.loads(session_file.read_text())
  # The saved JSON should at least contain id and events keys
  assert "id" in data and "events" in data


def test_create_artifact_service_defaults_to_file(tmp_path: Path) -> None:
  """Service factory should default to FileArtifactService when URI is unset."""
  service = create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(service, FileArtifactService)
  expected_root = Path(tmp_path) / ".adk" / "artifacts"
  assert service.root_dir == expected_root
  assert expected_root.exists()


def test_create_artifact_service_uses_shared_root(
    tmp_path: Path,
) -> None:
  """Artifact service should use a single file artifact service."""
  service = create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(service, FileArtifactService)
  expected_root = Path(tmp_path) / ".adk" / "artifacts"
  assert service.root_dir == expected_root
  assert expected_root.exists()


def test_create_artifact_service_respects_memory_uri(tmp_path: Path) -> None:
  """Service factory should honor memory:// URIs."""
  service = create_artifact_service_from_options(
      base_dir=tmp_path, artifact_service_uri="memory://"
  )
  assert isinstance(service, InMemoryArtifactService)


def test_create_artifact_service_accepts_file_uri(tmp_path: Path) -> None:
  """Service factory should allow custom local roots via file:// URIs."""
  custom_root = tmp_path / "custom_artifacts"
  service = create_artifact_service_from_options(
      base_dir=tmp_path, artifact_service_uri=custom_root.as_uri()
  )
  assert isinstance(service, FileArtifactService)
  assert service.root_dir == custom_root
  assert custom_root.exists()


@pytest.mark.asyncio
async def test_run_cli_accepts_memory_scheme(
    fake_agent, tmp_path: Path
) -> None:
  """run_cli should allow configuring in-memory services via memory:// URIs."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": []}
  input_path = tmp_path / "noop.json"
  input_path.write_text(json.dumps(input_json))

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
      session_service_uri="memory://",
      artifact_service_uri="memory://",
  )


@pytest.mark.asyncio
async def test_run_interactively_whitespace_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_interactively should skip blank input, echo once, then exit."""
  # make a session that belongs to dummy agent
  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")
  artifact_service = InMemoryArtifactService()
  credential_service = InMemoryCredentialService()
  root_agent = BaseAgent(name="root")

  # fake user input: blank -> 'hello' -> 'exit'
  answers = iter(["  ", "hello", "exit"])
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))

  # capture assisted echo
  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg: echoed.append(msg))

  await cli.run_interactively(
      root_agent, artifact_service, sess, session_service, credential_service
  )

  # verify: assistant echoed once with 'echo:hello'
  assert any("echo:hello" in m for m in echoed)

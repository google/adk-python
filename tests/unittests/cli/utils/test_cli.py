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

import asyncio
import json
from pathlib import Path
import sys
from textwrap import dedent
import time
import types
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from unittest import mock

import click
from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App
from google.adk.artifacts.file_artifact_service import FileArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
import google.adk.cli.cli as cli
from google.adk.cli.utils.local_storage import PerAgentFileArtifactService
from google.adk.cli.utils.service_factory import create_artifact_service_from_options
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
import pytest
from watchdog.events import FileClosedNoWriteEvent
from watchdog.events import FileModifiedEvent
from watchdog.events import FileOpenedEvent


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
      ev = types.SimpleNamespace(
          author="assistant",
          content=response,
          node_info=None,
          long_running_tool_ids=[],
      )
      yield ev

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


@pytest.mark.asyncio
async def test_run_cli_save_session_raises_when_session_is_gone(
    fake_agent, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli raises a clear error if the session vanished before saving."""
  parent_dir, folder_name = fake_agent
  responses = iter(["exit", "sess123"])
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(responses))

  class _ForgetfulSessionService(InMemorySessionService):

    async def get_session(self, **kwargs: Any) -> Session | None:
      return None

  monkeypatch.setattr(
      cli,
      "create_session_service_from_options",
      lambda **_: _ForgetfulSessionService(),
  )

  with pytest.raises(ValueError, match="not found"):
    await cli.run_cli(
        agent_parent_dir=str(parent_dir),
        agent_folder_name=folder_name,
        input_file=None,
        saved_session_file=None,
        save_session=True,
    )


@pytest.mark.asyncio
async def test_run_cli_resume_strips_internal_metadata_and_marks_events_restored(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Events loaded with --resume lose ADK-internal keys and are marked."""
  from google.adk.events.event import Event

  parent_dir, folder_name = fake_agent
  saved = Session(id="saved", app_name=folder_name, user_id="u")
  saved.events.append(
      Event(
          author="user",
          custom_metadata={
              "keep": 1,
              INTERNAL_METADATA_PREFIX + "planted": "x",
              RESTORED_EVENT_KEY: False,
          },
      )
  )
  saved_path = tmp_path / "saved.session.json"
  saved_path.write_text(saved.model_dump_json())
  captured = {}

  async def _capture_session(*args: Any, **kwargs: Any) -> None:
    captured["session"] = kwargs.get(
        "session", args[2] if len(args) > 2 else None
    )

  monkeypatch.setattr(cli, "run_interactively", _capture_session)

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      saved_session_file=str(saved_path),
      save_session=False,
      in_memory=True,
  )

  events = captured["session"].events
  assert len(events) == 1
  assert events[0].custom_metadata == {"keep": 1, RESTORED_EVENT_KEY: True}


@pytest.mark.asyncio
async def test_run_cli_save_session_omits_internal_metadata(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A resumed session is saved without the restored marker."""
  from google.adk.events.event import Event

  parent_dir, folder_name = fake_agent
  saved = Session(id="saved", app_name=folder_name, user_id="u")
  saved.events.append(Event(author="user", custom_metadata={"keep": 1}))
  saved_path = tmp_path / "saved.session.json"
  saved_path.write_text(saved.model_dump_json())

  async def _no_interaction(*args: Any, **kwargs: Any) -> None:
    del args, kwargs

  monkeypatch.setattr(cli, "run_interactively", _no_interaction)
  monkeypatch.setattr("builtins.input", lambda *_a, **_k: "resaved")

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      saved_session_file=str(saved_path),
      save_session=True,
      in_memory=True,
  )

  data = json.loads(
      (Path(parent_dir) / folder_name / "resaved.session.json").read_text()
  )
  assert [e.get("customMetadata") for e in data["events"]] == [{"keep": 1}]


@pytest.mark.asyncio
async def test_create_artifact_service_isolates_artifacts_per_agent(
    tmp_path: Path,
) -> None:
  """Each agent's artifacts should land in its own .adk/artifacts folder."""
  (tmp_path / "agent_a").mkdir()
  (tmp_path / "agent_b").mkdir()
  service = create_artifact_service_from_options(
      base_dir=tmp_path,
      use_local_storage=True,
  )
  assert isinstance(service, PerAgentFileArtifactService)

  # Touching each agent provisions its own per-agent .adk/artifacts folder.
  for app_name in ("agent_a", "agent_b"):
    await service.list_artifact_keys(
        app_name=app_name, user_id="user", session_id="session"
    )

  assert (tmp_path / "agent_a" / ".adk" / "artifacts").exists()
  assert (tmp_path / "agent_b" / ".adk" / "artifacts").exists()
  assert not (tmp_path / ".adk").exists()


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
      memory_service_uri="memory://",
  )


@pytest.mark.asyncio
async def test_run_cli_invalid_memory_uri_surfaces_value_error(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should let ValueError propagate for invalid memory service URIs."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": []}
  input_path = tmp_path / "invalid_memory_uri.json"
  input_path.write_text(json.dumps(input_json))

  def _raise_invalid_memory_uri(
      *,
      base_dir: Path | str,
      memory_service_uri: str | None = None,
  ) -> object:
    del base_dir, memory_service_uri
    raise ValueError("Unsupported memory service URI: unknown://x")

  monkeypatch.setattr(
      cli, "create_memory_service_from_options", _raise_invalid_memory_uri
  )

  with pytest.raises(ValueError, match="Unsupported memory service URI"):
    await cli.run_cli(
        agent_parent_dir=str(parent_dir),
        agent_folder_name=folder_name,
        input_file=str(input_path),
        saved_session_file=None,
        save_session=False,
        memory_service_uri="unknown://x",
    )


@pytest.mark.asyncio
async def test_run_cli_passes_memory_service_to_input_file(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should construct and pass the configured memory service."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": []}
  input_path = tmp_path / "memory_input.json"
  input_path.write_text(json.dumps(input_json))

  memory_service_sentinel = object()
  captured_factory_args: dict[str, Any] = {}
  captured_memory_service: dict[str, Any] = {}

  def _memory_factory(
      *,
      base_dir: Path | str,
      memory_service_uri: str | None = None,
  ) -> object:
    captured_factory_args["base_dir"] = base_dir
    captured_factory_args["memory_service_uri"] = memory_service_uri
    return memory_service_sentinel

  async def _run_input_file(
      app_name: str,
      user_id: str,
      agent_or_app: BaseAgent | App,
      artifact_service: Any,
      session_service: Any,
      credential_service: InMemoryCredentialService,
      input_path: str,
      memory_service: Any = None,
  ) -> object:
    del app_name, user_id, agent_or_app, artifact_service
    del session_service, credential_service, input_path
    captured_memory_service["value"] = memory_service
    return object()

  monkeypatch.setattr(
      cli, "create_memory_service_from_options", _memory_factory
  )
  monkeypatch.setattr(cli, "run_input_file", _run_input_file)

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
      memory_service_uri="memory://",
  )

  assert Path(captured_factory_args["base_dir"]) == parent_dir.resolve()
  assert captured_factory_args["memory_service_uri"] == "memory://"
  assert captured_memory_service["value"] is memory_service_sentinel


@pytest.mark.asyncio
async def test_run_cli_loads_dotenv_before_memory_service_creation(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should load agent .env values before creating memory service."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": []}
  input_path = tmp_path / "dotenv_order_input.json"
  input_path.write_text(json.dumps(input_json))

  call_order: list[str] = []

  def _load_dotenv_for_agent(agent_name: str, agents_dir: str) -> None:
    del agent_name, agents_dir
    call_order.append("load_dotenv")

  def _memory_factory(
      *,
      base_dir: Path | str,
      memory_service_uri: str | None = None,
  ) -> object:
    del base_dir, memory_service_uri
    call_order.append("create_memory")
    return object()

  monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "0")
  monkeypatch.setattr(cli.envs, "load_dotenv_for_agent", _load_dotenv_for_agent)
  monkeypatch.setattr(
      cli, "create_memory_service_from_options", _memory_factory
  )

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
      memory_service_uri="memory://",
  )

  assert "create_memory" in call_order
  assert "load_dotenv" in call_order
  assert call_order.index("load_dotenv") < call_order.index("create_memory")


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


def test_print_event_omits_internal_metadata_in_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """_print_event does not print ADK-internal custom_metadata."""
  from google.adk.events.event import Event

  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg: echoed.append(msg))
  event = Event.model_validate({
      "author": "agent",
      "custom_metadata": {"keep": 1, INTERNAL_METADATA_PREFIX + "stamp": "x"},
  })

  cli._print_event(event, jsonl=True)

  assert json.loads(echoed[0])["customMetadata"] == {"keep": 1}


def test_print_event_preserves_non_ascii_in_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """_print_event should output unescaped UTF-8 characters when jsonl=True."""
  from google.adk.events.event import Event

  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg: echoed.append(msg))

  # Built from a dict rather than types.Content/types.Part: the autouse
  # _patch_types_and_runner fixture swaps those for fakes that Event's pydantic
  # validation rejects.
  event = Event.model_validate({
      "author": "agent",
      "content": {"role": "model", "parts": [{"text": "日本語の回答"}]},
  })

  cli._print_event(event, jsonl=True)

  assert len(echoed) == 1
  assert "日本語の回答" in echoed[0]
  assert "\\u" not in echoed[0]


@pytest.mark.asyncio
async def test_run_cli_in_memory_flag_sets_memory_service_uri(
    fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli with in_memory=True should configure memory_service_uri='memory://'."""
  parent_dir, folder_name = fake_agent
  input_json = {"state": {}, "queries": []}
  input_path = tmp_path / "in_memory_input.json"
  input_path.write_text(json.dumps(input_json))

  captured_factory_args: dict[str, Any] = {}

  def _memory_factory(
      *,
      base_dir: Path | str,
      memory_service_uri: str | None = None,
  ) -> object:
    captured_factory_args["memory_service_uri"] = memory_service_uri
    return object()

  monkeypatch.setattr(
      cli, "create_memory_service_from_options", _memory_factory
  )

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      input_file=str(input_path),
      saved_session_file=None,
      save_session=False,
      in_memory=True,
  )

  assert captured_factory_args["memory_service_uri"] == "memory://"


@pytest.mark.asyncio
async def test_run_interactively_reload_agents_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_interactively should reload the agent when reload_event is set."""

  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")
  artifact_service = InMemoryArtifactService()
  credential_service = InMemoryCredentialService()
  root_agent = BaseAgent(name="root")

  reload_event = asyncio.Event()
  sys_stdin_readline_calls = []

  main_loop = asyncio.get_running_loop()

  def mock_readline():
    sys_stdin_readline_calls.append(True)
    if len(sys_stdin_readline_calls) == 1:
      # Return a normal query first
      return "hello\n"
    elif len(sys_stdin_readline_calls) == 2:
      # Sleep a bit to allow the loop to run, then trigger the reload
      time.sleep(0.1)
      # In tests, we need to set the event thread-safely
      main_loop.call_soon_threadsafe(reload_event.set)
      time.sleep(0.1)
      return "exit\n"
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg, **kw: echoed.append(msg))
  monkeypatch.setattr(click, "secho", lambda msg, **kw: echoed.append(msg))

  class DummyAgentLoader:
    removed = False
    reloaded = False

    def remove_agent_from_cache(self, name):
      self.removed = True

    def load_agent(self, name):
      self.reloaded = True
      return BaseAgent(name="reloaded_root")

  loader = DummyAgentLoader()

  await cli.run_interactively(
      root_agent,
      artifact_service,
      sess,
      session_service,
      credential_service,
      reload_agents=True,
      reload_event=reload_event,
      agent_loader=loader,
      agent_folder_name="dummy_folder",
  )

  # Check that the agent handled the first message, without the newline
  assert "[assistant]: echo:hello" in echoed
  # Check that the reload message was printed
  assert any("reloading agent..." in m for m in echoed)
  # Check that the loader cache was manipulated
  assert loader.removed is True
  assert loader.reloaded is True


@pytest.mark.asyncio
async def test_run_interactively_reload_agents_does_not_drop_input_on_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """run_interactively should not discard user input queued during a reload."""
  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")
  artifact_service = InMemoryArtifactService()
  credential_service = InMemoryCredentialService()
  root_agent = BaseAgent(name="root")

  reload_event = asyncio.Event()

  queries = ["first\n", "second\n", "exit\n"]
  call_count = 0

  def mock_readline():
    nonlocal call_count
    if call_count < len(queries):
      line = queries[call_count]
      call_count += 1
      return line
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg, **kw: echoed.append(msg))
  monkeypatch.setattr(click, "secho", lambda msg, **kw: echoed.append(msg))

  orig_runner = cli.Runner

  class _DelayedRunner(orig_runner):

    async def run_async(self, *a: Any, **k: Any):
      message = a[2] if len(a) >= 3 else k["new_message"]
      text = message.parts[0].text if message.parts else ""
      if text.strip() == "first":
        await asyncio.sleep(0.05)
        reload_event.set()
        await asyncio.sleep(0.05)
      async for ev in super().run_async(*a, **k):
        yield ev

  monkeypatch.setattr(cli, "Runner", _DelayedRunner)

  class DummyAgentLoader:

    def remove_agent_from_cache(self, name):
      pass

    def load_agent(self, name):
      return BaseAgent(name="reloaded_root")

  await cli.run_interactively(
      root_agent,
      artifact_service,
      sess,
      session_service,
      credential_service,
      reload_agents=True,
      reload_event=reload_event,
      agent_loader=DummyAgentLoader(),
      agent_folder_name="dummy_folder",
  )

  assert any("echo:second" in m for m in echoed)


@pytest.mark.asyncio
async def test_run_interactively_reload_agents_reloads_when_input_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """run_interactively reloads the agent even when input is already queued."""
  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")
  artifact_service = InMemoryArtifactService()
  credential_service = InMemoryCredentialService()
  root_agent = BaseAgent(name="root")

  reload_event = asyncio.Event()

  queries = ["first\n", "second\n", "exit\n"]
  call_count = 0

  def mock_readline():
    nonlocal call_count
    if call_count < len(queries):
      line = queries[call_count]
      call_count += 1
      return line
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  echoed: list[str] = []
  monkeypatch.setattr(click, "echo", lambda msg, **kw: echoed.append(msg))
  monkeypatch.setattr(click, "secho", lambda msg, **kw: echoed.append(msg))

  orig_runner = cli.Runner

  class _DelayedRunner(orig_runner):

    async def run_async(self, *a: Any, **k: Any):
      message = a[2] if len(a) >= 3 else k["new_message"]
      text = message.parts[0].text if message.parts else ""
      if text.strip() == "first":
        await asyncio.sleep(0.05)
        reload_event.set()
        await asyncio.sleep(0.05)
      async for ev in super().run_async(*a, **k):
        yield ev

  monkeypatch.setattr(cli, "Runner", _DelayedRunner)

  class DummyAgentLoader:
    removed = False
    reloaded = False

    def remove_agent_from_cache(self, name):
      self.removed = True

    def load_agent(self, name):
      self.reloaded = True
      return BaseAgent(name="reloaded_root")

  loader = DummyAgentLoader()

  await cli.run_interactively(
      root_agent,
      artifact_service,
      sess,
      session_service,
      credential_service,
      reload_agents=True,
      reload_event=reload_event,
      agent_loader=loader,
      agent_folder_name="dummy_folder",
  )

  assert loader.reloaded is True
  assert any("reloading agent..." in m for m in echoed)


@pytest.mark.asyncio
async def test_agent_reload_handler_ignores_read_only_events() -> None:
  """Opening or reading a watched file must not trigger a reload."""
  reload_event = asyncio.Event()
  handler = cli._AgentReloadHandler(asyncio.get_running_loop(), reload_event)

  handler.on_any_event(FileOpenedEvent("/agent/root_agent.yaml"))
  handler.on_any_event(FileClosedNoWriteEvent("/agent/root_agent.yaml"))
  await asyncio.sleep(0)
  assert not reload_event.is_set()

  handler.on_any_event(FileModifiedEvent("/agent/root_agent.yaml"))
  await asyncio.sleep(0)
  assert reload_event.is_set()


@pytest.mark.asyncio
async def test_agent_reload_handler_reloads_on_yml_change() -> None:
  """Editing a .yml agent config triggers a reload."""
  reload_event = asyncio.Event()
  handler = cli._AgentReloadHandler(asyncio.get_running_loop(), reload_event)

  handler.on_any_event(FileModifiedEvent("/agent/sub_agent.yml"))
  await asyncio.sleep(0)
  assert reload_event.is_set()


@pytest.mark.asyncio
async def test_agent_reload_handler_ignores_files_under_dot_adk() -> None:
  """Saving a .py artifact into the agent's .adk storage must not reload."""
  reload_event = asyncio.Event()
  handler = cli._AgentReloadHandler(asyncio.get_running_loop(), reload_event)

  handler.on_any_event(
      FileModifiedEvent("/agent/.adk/artifacts/s1/tool.py/versions/0/tool.py")
  )
  await asyncio.sleep(0)
  assert not reload_event.is_set()

  handler.on_any_event(FileModifiedEvent("/agent/agent.py"))
  await asyncio.sleep(0)
  assert reload_event.is_set()


@pytest.mark.asyncio
async def test_run_interactively_reload_agents_prompts_after_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """A timed-out turn in reload mode prints the next [user]: prompt."""
  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")

  lines = iter(["slow\n", "exit\n"])
  monkeypatch.setattr(sys.stdin, "readline", lambda: next(lines, ""))
  monkeypatch.setattr(cli, "_parse_timeout", lambda _: 0.01)

  class _SlowRunner:

    def __init__(self, *a: Any, **k: Any) -> None:
      ...

    async def run_async(self, *a: Any, **k: Any):
      await asyncio.sleep(10)
      yield

    async def close(self, *a: Any, **k: Any) -> None:
      ...

  monkeypatch.setattr(cli, "Runner", _SlowRunner)

  await cli.run_interactively(
      BaseAgent(name="root"),
      InMemoryArtifactService(),
      sess,
      session_service,
      InMemoryCredentialService(),
      timeout="1s",
      reload_agents=True,
      reload_event=asyncio.Event(),
  )

  assert capsys.readouterr().out.count("[user]: ") == 2


@pytest.mark.asyncio
async def test_run_cli_reload_agents_stops_observer_on_exit(
    fake_agent, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli should stop and join the file observer when exiting."""
  parent_dir, folder_name = fake_agent

  calls = 0

  def mock_readline() -> str:
    nonlocal calls
    calls += 1
    if calls == 1:
      return "exit\n"
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  stopped = False
  joined = False
  orig_observer = cli.Observer
  created_observers: list[cli.Observer] = []

  class _InstrumentedObserver(orig_observer):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      super().__init__(*args, **kwargs)
      created_observers.append(self)

    def stop(self) -> None:
      nonlocal stopped
      stopped = True
      super().stop()

    def join(self, timeout: float | None = None) -> None:
      nonlocal joined
      joined = True
      super().join(timeout)

  monkeypatch.setattr(cli, "Observer", _InstrumentedObserver)

  stopped_by_run_cli = False
  joined_by_run_cli = False

  try:
    await cli.run_cli(
        agent_parent_dir=str(parent_dir),
        agent_folder_name=folder_name,
        saved_session_file=None,
        save_session=False,
        reload_agents=True,
    )
    stopped_by_run_cli = stopped
    joined_by_run_cli = joined
  finally:
    for obs in created_observers:
      if obs.is_alive():
        obs.stop()
        obs.join()

  assert stopped_by_run_cli is True
  assert joined_by_run_cli is True


@pytest.mark.asyncio
async def test_run_interactively_reload_agents_hitl_consumes_queued_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """run_interactively with reload_agents reads HITL input from the queue."""
  session_service = InMemorySessionService()
  sess = await session_service.create_session(app_name="dummy", user_id="u")
  artifact_service = InMemoryArtifactService()
  credential_service = InMemoryCredentialService()
  root_agent = BaseAgent(name="root")

  reload_event = asyncio.Event()

  queries = ["book flight\n", "yes\n", "exit\n"]
  call_count = 0

  def mock_readline() -> str:
    nonlocal call_count
    if call_count < len(queries):
      line = queries[call_count]
      call_count += 1
      return line
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  class _PartWithFunctionResponse:

    def __init__(
        self, text: str | None = "", function_response: Any = None
    ) -> None:
      self.text = text
      self.function_response = function_response

  monkeypatch.setattr(cli.types, "Part", _PartWithFunctionResponse)

  orig_runner = cli.Runner
  hitl_handled = False

  class _HitlRunner(orig_runner):

    async def run_async(self, *a: Any, **k: Any):
      nonlocal hitl_handled
      message = a[2] if len(a) >= 3 else k["new_message"]
      for part in getattr(message, "parts", []):
        fr = getattr(part, "function_response", None)
        if fr and getattr(fr, "name", None) == cli._REQUEST_CONFIRMATION:
          hitl_handled = True
          yield types.SimpleNamespace(
              author="assistant",
              content=cli.types.Content(
                  "assistant", [_PartWithFunctionResponse("Confirmed!")]
              ),
              node_info=None,
              long_running_tool_ids=[],
          )
          return

      fc = types.SimpleNamespace(
          id="call_1",
          name=cli._REQUEST_CONFIRMATION,
          args={"toolConfirmation": {"hint": "Confirm booking?"}},
      )
      part = _PartWithFunctionResponse("Confirm?")
      part.function_call = fc
      yield types.SimpleNamespace(
          author="assistant",
          content=cli.types.Content("assistant", [part]),
          node_info=None,
          long_running_tool_ids=["call_1"],
      )

  monkeypatch.setattr(cli, "Runner", _HitlRunner)

  await cli.run_interactively(
      root_agent,
      artifact_service,
      sess,
      session_service,
      credential_service,
      reload_agents=True,
      reload_event=reload_event,
  )

  assert hitl_handled is True


@pytest.mark.asyncio
async def test_run_cli_reload_agents_save_session_prompts(
    fake_agent, monkeypatch: pytest.MonkeyPatch
) -> None:
  """run_cli with reload_agents prompts for a session id and saves it."""
  parent_dir, folder_name = fake_agent

  queries = ["exit\n", "sess_reload\n"]
  calls = 0

  def mock_readline() -> str:
    nonlocal calls
    if calls < len(queries):
      line = queries[calls]
      calls += 1
      return line
    return ""

  monkeypatch.setattr(sys.stdin, "readline", mock_readline)

  session_file = Path(parent_dir) / folder_name / "sess_reload.session.json"
  if session_file.exists():
    session_file.unlink()

  await cli.run_cli(
      agent_parent_dir=str(parent_dir),
      agent_folder_name=folder_name,
      saved_session_file=None,
      save_session=True,
      reload_agents=True,
  )

  assert session_file.exists()

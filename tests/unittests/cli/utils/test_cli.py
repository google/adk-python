# Copyright 2025 Google LLC
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

import click
import json
import pytest
import sys
import types

import google.adk.cli.cli as cli

from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        def __init__(self, *a: Any, **k: Any) -> None: ...

        async def run_async(self, *a: Any, **k: Any):
            message = a[2] if len(a) >= 3 else k["new_message"]
            text = message.parts[0].text if message.parts else ""
            response = _Content("assistant", [_Part(f"echo:{text}")])
            yield types.SimpleNamespace(author="assistant", content=response)

    monkeypatch.setattr(cli, "Runner", _FakeRunner)


@pytest.fixture()
def fake_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a minimal importable agent package and patch importlib."""

    parent_dir = tmp_path / "agents"
    parent_dir.mkdir()
    agent_dir = parent_dir / "fake_agent"
    agent_dir.mkdir()
    # __init__.py exposes root_agent with .name
    (agent_dir / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "root_agent = SimpleNamespace(name='fake_root')\n"
    )

    # Ensure importable via sys.path
    sys.path.insert(0, str(parent_dir))

    import importlib

    module = importlib.import_module("fake_agent")
    fake_module = types.SimpleNamespace(agent=module)

    monkeypatch.setattr(importlib, "import_module", lambda n: fake_module)
    monkeypatch.setattr(cli.envs, "load_dotenv_for_agent", lambda *a, **k: None)

    yield parent_dir, "fake_agent"

    # Cleanup
    sys.path.remove(str(parent_dir))
    del sys.modules["fake_agent"]


# _run_input_file
@pytest.mark.asyncio
async def test_run_input_file_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    artifact_service = cli.InMemoryArtifactService()
    session_service = cli.InMemorySessionService()
    dummy_root = types.SimpleNamespace(name="root")

    session = await cli.run_input_file(
        app_name="app",
        user_id="user",
        root_agent=dummy_root,
        artifact_service=artifact_service,
        session_service=session_service,
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


# _run_cli (interactive + save session branch)
@pytest.mark.asyncio
async def test_run_cli_save_session(fake_agent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
async def test_run_interactively_whitespace_and_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_interactively should skip blank input, echo once, then exit."""
    # make a session that belongs to dummy agent
    svc = cli.InMemorySessionService()
    sess = svc.create_session(app_name="dummy", user_id="u")
    artifact_service = cli.InMemoryArtifactService()
    root_agent = types.SimpleNamespace(name="root")

    # fake user input: blank -> 'hello' -> 'exit'
    answers = iter(["  ", "hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))

    # capture assisted echo
    echoed: list[str] = []
    monkeypatch.setattr(click, "echo", lambda msg: echoed.append(msg))

    await cli.run_interactively(root_agent, artifact_service, sess, svc)

    # verify: assistant echoed once with 'echo:hello'
    assert any("echo:hello" in m for m in echoed)


# run_cli (resume branch)
@pytest.mark.asyncio
async def test_run_cli_resume_saved_session(tmp_path: Path, fake_agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_cli should load previous session, print its events, then re-enter interactive mode."""
    parent_dir, folder = fake_agent

    # stub Session.model_validate_json to return dummy session with two events
    user_content = types.SimpleNamespace(parts=[types.SimpleNamespace(text="hi")])
    assistant_content = types.SimpleNamespace(parts=[types.SimpleNamespace(text="hello!")])
    dummy_session = types.SimpleNamespace(
        id="sess",
        app_name=folder,
        user_id="u",
        events=[
            types.SimpleNamespace(author="user", content=user_content, partial=False),
            types.SimpleNamespace(author="assistant", content=assistant_content, partial=False),
        ],
    )
    monkeypatch.setattr(cli.Session, "model_validate_json", staticmethod(lambda _s: dummy_session))
    monkeypatch.setattr(cli.InMemorySessionService, "append_event", lambda *_a, **_k: None)
    # interactive inputs: immediately 'exit'
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "exit")

    # collect echo output
    captured: list[str] = []
    monkeypatch.setattr(click, "echo", lambda m: captured.append(m))

    saved_path = tmp_path / "prev.session.json"
    saved_path.write_text("{}")  # contents not used – patched above

    await cli.run_cli(
        agent_parent_dir=str(parent_dir),
        agent_folder_name=folder,
        input_file=None,
        saved_session_file=str(saved_path),
        save_session=False,
    )

    # ④ ensure both historical messages were printed
    assert any("[user]: hi" in m for m in captured)
    assert any("[assistant]: hello!" in m for m in captured)

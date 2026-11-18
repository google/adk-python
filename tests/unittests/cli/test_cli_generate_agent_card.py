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

from __future__ import annotations

from collections.abc import Iterator
import json
import os
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

from a2a.types import AgentCard
from click.testing import CliRunner
from google.adk.a2a import _compat
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.agents.base_agent import BaseAgent
from google.adk.cli._cli_generate_agent_card import generate_agent_card
import pytest


@pytest.fixture
def runner() -> CliRunner:
  try:
    return CliRunner(mix_stderr=False)
  except TypeError:
    # Click 8.2+ removed mix_stderr and separates stderr by default.
    return CliRunner()


@pytest.fixture
def mock_agent_card_builder() -> Iterator[MagicMock]:
  with patch(
      "google.adk.a2a.utils.agent_card_builder.AgentCardBuilder",
      wraps=AgentCardBuilder,
  ) as mock:
    yield mock


def test_generate_agent_card_missing_a2a(runner: CliRunner) -> None:
  # Simulate the module being missing from the environment
  with patch.dict(
      "sys.modules", {"google.adk.a2a.utils.agent_card_builder": None}
  ):
    result = runner.invoke(generate_agent_card)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Error: 'a2a' package is required for this command." in result.stderr


@patch(
    "google.adk.a2a._compat.a2a_to_dict",
    wraps=_compat.a2a_to_dict,
)
@patch("google.adk.cli.utils.agent_loader.AgentLoader")
def test_generate_agent_card_success_no_file(
    mock_loader_cls: MagicMock,
    mock_to_dict: MagicMock,
    mock_agent_card_builder: MagicMock,
    runner: CliRunner,
) -> None:
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1"]
  mock_agent = BaseAgent(name="agent1", description="test")
  mock_loader.load_agent.return_value = mock_agent

  result = runner.invoke(
      generate_agent_card,
      ["--protocol", "http", "--host", "localhost", "--port", "9000"],
  )

  assert result.exit_code == 0
  output = json.loads(result.output)
  assert len(output) == 1
  assert output[0]["name"] == "agent1"
  card = _compat.parse_agent_card(output[0])
  assert _compat.agent_card_url(card) == "http://localhost:9000/a2a/agent1"

  mock_loader.list_agents.assert_called_once()
  mock_loader.load_agent.assert_called_with("agent1")
  mock_agent_card_builder.assert_called_with(
      agent=mock_agent, rpc_url="http://localhost:9000/a2a/agent1"
  )
  assert isinstance(mock_to_dict.call_args.args[0], AgentCard)


@patch(
    "google.adk.a2a._compat.a2a_to_dict",
    wraps=_compat.a2a_to_dict,
)
@patch("google.adk.cli.utils.agent_loader.AgentLoader")
def test_generate_agent_card_success_create_file(
    mock_loader_cls: MagicMock,
    mock_to_dict: MagicMock,
    mock_agent_card_builder: MagicMock,
    runner: CliRunner,
) -> None:
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1"]
  mock_agent = BaseAgent(name="agent1", description="test")
  mock_loader.load_agent.return_value = mock_agent

  with runner.isolated_filesystem():
    mock_loader.agents_dir = os.getcwd()
    os.mkdir("agent1")

    result = runner.invoke(generate_agent_card, ["--create-file"])

    assert result.exit_code == 0

    agent_json = os.path.join("agent1", "agent.json")
    assert os.path.exists(agent_json)
    with open(agent_json, "r") as f:
      content = json.load(f)
      assert content["name"] == "agent1"
      card = _compat.parse_agent_card(content)
      assert _compat.agent_card_url(card) == "http://127.0.0.1:8000/a2a/agent1"
  assert isinstance(mock_to_dict.call_args.args[0], AgentCard)


@patch(
    "google.adk.a2a._compat.a2a_to_dict",
    wraps=_compat.a2a_to_dict,
)
def test_generate_agent_card_single_agent_mode_create_file(
    mock_to_dict: MagicMock,
    mock_agent_card_builder: MagicMock,
    runner: CliRunner,
) -> None:
  original_sys_path = sys.path.copy()
  try:
    with runner.isolated_filesystem() as root_dir:
      agent_dir = os.path.join(root_dir, "weather")
      os.mkdir(agent_dir)
      with open(
          os.path.join(agent_dir, "agent.py"), "w", encoding="utf-8"
      ) as f:
        f.write(
            "from google.adk.agents.base_agent import BaseAgent\n"
            "root_agent = BaseAgent(name='weather', description='test')\n"
        )
      os.chdir(agent_dir)

      result = runner.invoke(generate_agent_card, ["--create-file"])

      assert result.exit_code == 0
      agent_json = os.path.join(agent_dir, "agent.json")
      assert os.path.exists(agent_json)
      assert not os.path.exists(os.path.join(agent_dir, "weather"))
      with open(agent_json, "r", encoding="utf-8") as f:
        content = json.load(f)
        assert content["name"] == "weather"
        card = _compat.parse_agent_card(content)
        assert (
            _compat.agent_card_url(card) == "http://127.0.0.1:8000/a2a/weather"
        )
  finally:
    sys.path[:] = original_sys_path
    sys.modules.pop("weather", None)
    sys.modules.pop("weather.agent", None)
  assert isinstance(mock_to_dict.call_args.args[0], AgentCard)


@patch(
    "google.adk.a2a._compat.a2a_to_dict",
    wraps=_compat.a2a_to_dict,
)
@patch("google.adk.cli.utils.agent_loader.AgentLoader")
def test_generate_agent_card_creates_dir_if_missing(
    mock_loader_cls: MagicMock,
    mock_to_dict: MagicMock,
    mock_agent_card_builder: MagicMock,
    runner: CliRunner,
) -> None:
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1"]
  mock_agent = BaseAgent(name="agent1", description="test")
  mock_loader.load_agent.return_value = mock_agent

  with runner.isolated_filesystem():
    mock_loader.agents_dir = os.getcwd()
    result = runner.invoke(generate_agent_card, ["--create-file"])

    assert result.exit_code == 0

    agent_json = os.path.join("agent1", "agent.json")
    assert os.path.exists(agent_json)
    with open(agent_json, "r") as f:
      content = json.load(f)
      assert content["name"] == "agent1"
  assert isinstance(mock_to_dict.call_args.args[0], AgentCard)


@patch(
    "google.adk.a2a._compat.a2a_to_dict",
    wraps=_compat.a2a_to_dict,
)
@patch("google.adk.cli.utils.agent_loader.AgentLoader")
def test_generate_agent_card_agent_error(
    mock_loader_cls: MagicMock,
    mock_to_dict: MagicMock,
    mock_agent_card_builder: MagicMock,
    runner: CliRunner,
) -> None:
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1", "agent2"]

  mock_agent2 = BaseAgent(name="agent2", description="test")

  def side_effect(name: str) -> BaseAgent:
    if name == "agent1":
      raise Exception("Load error")
    return mock_agent2

  mock_loader.load_agent.side_effect = side_effect

  result = runner.invoke(generate_agent_card)

  assert result.exit_code == 0
  assert "Error processing agent agent1: Load error" in result.stderr

  output = json.loads(result.stdout)
  assert len(output) == 1
  assert output[0]["name"] == "agent2"
  assert isinstance(mock_to_dict.call_args.args[0], AgentCard)


def test_importing_cli_tools_click_defers_runtime_modules() -> None:
  """Importing cli_tools_click does not eagerly load heavy runtime modules."""
  import subprocess
  import sys
  import tempfile

  forbidden = (
      "fastapi",
      "google.adk.agents.run_config",
      "google.adk.cli.cli",
      "google.adk.evaluation.agent_evaluator",
      "google.genai",
      "mcp",
      "uvicorn",
  )
  source = f"""
import importlib
import sys

importlib.import_module('google.adk.cli.cli_tools_click')
forbidden = {forbidden!r}
loaded = [
    prefix
    for prefix in forbidden
    if any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    )
]
assert not loaded, loaded
"""
  with tempfile.TemporaryDirectory() as isolated_cwd:
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=isolated_cwd,
        capture_output=True,
        text=True,
        check=False,
    )
  assert result.returncode == 0, result.stderr

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

import json
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from click.testing import CliRunner
from google.adk.cli.cli_generate_agent_card import generate_agent_card
import pytest


@pytest.fixture
def runner():
  return CliRunner()


@pytest.fixture
def mock_agent_loader():
  with patch("google.adk.cli.utils.agent_loader.AgentLoader") as mock:
    yield mock


@pytest.fixture
def mock_agent_card_builder():
  mock_module = MagicMock()
  with patch.dict(
      "sys.modules", {"google.adk.a2a.utils.agent_card_builder": mock_module}
  ):
    yield mock_module.AgentCardBuilder


def test_generate_agent_card_missing_a2a(runner):
  # Simulate the module being missing from the environment
  with patch.dict("sys.modules", {"google.adk.a2a.utils.agent_card_builder": None}):
    result = runner.invoke(generate_agent_card)
    
    assert result.exit_code != 0
    assert "Error: 'a2a' package is required for this command." in result.stderr


def test_generate_agent_card_import_error(runner):
  # Simulate a generic ImportError during import
  with patch.dict("sys.modules", {"google.adk.a2a.utils.agent_card_builder": None}):
     result = runner.invoke(generate_agent_card)
     
     assert result.exit_code != 0
     assert isinstance(result.exception, SystemExit)


@patch("google.adk.cli.cli_generate_agent_card.AgentLoader")
def test_generate_agent_card_success_no_file(
    mock_loader_cls, mock_agent_card_builder, runner
):
  # Setup mocks
  mock_builder_cls = mock_agent_card_builder
  # Setup mocks
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1"]
  mock_agent = MagicMock()
  del mock_agent.root_agent
  mock_loader.load_agent.return_value = mock_agent

  mock_builder = mock_builder_cls.return_value
  mock_card = MagicMock()
  mock_card.model_dump.return_value = {"name": "agent1", "description": "test"}
  mock_builder.build = AsyncMock(return_value=mock_card)

  # Run command
  result = runner.invoke(
      generate_agent_card,
      ["--protocol", "http", "--host", "localhost", "--port", "9000"],
  )

  assert result.exit_code == 0
  output = json.loads(result.output)
  assert len(output) == 1
  assert output[0]["name"] == "agent1"

  # Verify calls
  mock_loader.list_agents.assert_called_once()
  mock_loader.load_agent.assert_called_with("agent1")
  mock_builder_cls.assert_called_with(
      agent=mock_agent, rpc_url="http://localhost:9000/agent1"
  )
  mock_builder.build.assert_called_once()


@patch("google.adk.cli.cli_generate_agent_card.AgentLoader")
def test_generate_agent_card_success_create_file(
    mock_loader_cls, mock_agent_card_builder, runner
):
  # Setup mocks
  mock_builder_cls = mock_agent_card_builder
  
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1"]
  mock_agent = MagicMock()
  mock_loader.load_agent.return_value = mock_agent

  mock_builder = mock_builder_cls.return_value
  mock_card = MagicMock()
  mock_card.model_dump.return_value = {"name": "agent1", "description": "test"}
  mock_builder.build = AsyncMock(return_value=mock_card)

  with runner.isolated_filesystem():
    os.mkdir("agent1")

    # Run command
    result = runner.invoke(generate_agent_card, ["--create-file"])

    assert result.exit_code == 0

    # Verify file creation
    agent_json = os.path.join("agent1", "agent.json")
    assert os.path.exists(agent_json)
    with open(agent_json, "r") as f:
      content = json.load(f)
      assert content["name"] == "agent1"


@patch("google.adk.cli.cli_generate_agent_card.AgentLoader")
def test_generate_agent_card_agent_error(
    mock_loader_cls, mock_agent_card_builder, runner
):
  # Setup mocks
  mock_builder_cls = mock_agent_card_builder
  # Setup mocks
  mock_loader = mock_loader_cls.return_value
  mock_loader.list_agents.return_value = ["agent1", "agent2"]

  # agent1 fails, agent2 succeeds
  mock_agent1 = MagicMock()
  mock_agent2 = MagicMock()

  def side_effect(name):
    if name == "agent1":
      raise Exception("Load error")
    return mock_agent2

  mock_loader.load_agent.side_effect = side_effect

  mock_builder = mock_builder_cls.return_value
  mock_card = MagicMock()
  mock_card.model_dump.return_value = {"name": "agent2"}
  mock_builder.build = AsyncMock(return_value=mock_card)

  # Run command
  result = runner.invoke(generate_agent_card)

  assert result.exit_code == 0
  # stderr should contain error for agent1
  assert "Error processing agent agent1: Load error" in result.stderr

  # stdout should contain json for agent2
  output = json.loads(result.stdout)
  assert len(output) == 1
  assert output[0]["name"] == "agent2"

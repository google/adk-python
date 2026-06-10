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

import json
import os
from pathlib import Path

from click.testing import CliRunner
from google.adk.cli import cli_tools_click
import pytest


def test_cli_run_integration(tmp_path: Path) -> None:
  """Integration test for `adk run` with query using CliRunner (No mocks)."""
  # Arrange
  agent_dir = tmp_path / "dummy_agent"
  agent_dir.mkdir()

  # Create __init__.py
  with open(agent_dir / "__init__.py", "w") as f:
    f.write("from . import agent\n")

  # Create agent.py
  agent_code = """
from google.adk.agents import Agent
from google.adk.events.event import Event
from typing import AsyncGenerator

class DummyAgent(Agent):
  async def run_async(self, ctx) -> AsyncGenerator[Event, None]:
    # Yield a text response
    from google.adk.events.event import Event
    text = ctx.user_content.parts[0].text if ctx.user_content and ctx.user_content.parts else "No input"
    yield Event(author="dummy", output=f"Echo: {text}")

root_agent = DummyAgent(name="dummy", model="gemini-2.5-flash")
"""
  with open(agent_dir / "agent.py", "w") as f:
    f.write(agent_code)

  runner = CliRunner()

  # Act
  result = runner.invoke(
      cli_tools_click.main,
      ["run", "--jsonl", str(agent_dir), "hello world"],
  )

  # Assert
  assert result.exit_code == 0

  # Check stdout
  stdout = result.stdout
  assert stdout, "Stdout should not be empty"

  # Parse JSONL lines
  lines = stdout.strip().split("\n")
  assert len(lines) > 0

  # The last line should be the final output or event
  last_line = lines[-1]
  try:
    data = json.loads(last_line)
    assert "output" in data
    assert "Echo: hello world" in data["output"]
  except json.JSONDecodeError:
    pytest.fail(f"Stdout contained non-JSON line: {last_line}")

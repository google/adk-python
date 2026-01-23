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

"""Unit tests for dotenv loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

import google.adk.cli.utils.envs as envs
import pytest


@pytest.fixture(autouse=True)
def _clear_explicit_env_cache() -> None:
  envs._get_explicit_env_keys.cache_clear()


def test_load_dotenv_for_agent_preserves_explicit_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  agents_dir = tmp_path / "agents"
  agent_dir = agents_dir / "agent1"
  agent_dir.mkdir(parents=True)

  explicit_key = "ADK_TEST_EXPLICIT_ENV"
  from_dotenv_key = "ADK_TEST_FROM_DOTENV"

  monkeypatch.setenv(explicit_key, "explicit")
  monkeypatch.delenv(from_dotenv_key, raising=False)
  envs._get_explicit_env_keys.cache_clear()

  (agent_dir / ".env").write_text(
      f"{explicit_key}=from_dotenv\n{from_dotenv_key}=from_dotenv\n"
  )

  envs.load_dotenv_for_agent("agent1", str(agents_dir))

  assert os.environ[explicit_key] == "explicit"
  assert os.environ[from_dotenv_key] == "from_dotenv"


def test_load_dotenv_for_agent_overrides_previous_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  agents_dir = tmp_path / "agents"
  agent1_dir = agents_dir / "agent1"
  agent2_dir = agents_dir / "agent2"
  agent1_dir.mkdir(parents=True)
  agent2_dir.mkdir(parents=True)

  key = "ADK_TEST_DOTENV_OVERRIDE"
  monkeypatch.delenv(key, raising=False)

  (agent1_dir / ".env").write_text(f"{key}=one\n")
  envs.load_dotenv_for_agent("agent1", str(agents_dir))
  assert os.environ[key] == "one"

  (agent2_dir / ".env").write_text(f"{key}=two\n")
  envs.load_dotenv_for_agent("agent2", str(agents_dir))
  assert os.environ[key] == "two"


def test_load_dotenv_for_agent_respects_disable_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  agents_dir = tmp_path / "agents"
  agent_dir = agents_dir / "agent1"
  agent_dir.mkdir(parents=True)

  key = "ADK_TEST_DISABLE_DOTENV"
  monkeypatch.delenv(key, raising=False)
  monkeypatch.setenv("ADK_DISABLE_LOAD_DOTENV", "1")
  envs._get_explicit_env_keys.cache_clear()

  (agent_dir / ".env").write_text(f"{key}=from_dotenv\n")

  envs.load_dotenv_for_agent("agent1", str(agents_dir))

  assert key not in os.environ

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

import os
from pathlib import Path
import sys
import tempfile
from textwrap import dedent

from google.adk.cli.utils._nested_agent_loader import NestedAgentLoader
import pytest


class TestNestedAgentLoader:
  """Comprehensive tests for NestedAgentLoader focusing on list_agents."""

  def test_list_agents_in_single_agent_mode_returns_only_single_agent(self):
    """Single agent mode returns list containing only the single agent's name."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_file = temp_path / "my_agent.py"
      agent_file.write_text("")

      loader = NestedAgentLoader(str(agent_file))
      agents = loader.list_agents()

      assert agents == ["my_agent"]

  def test_list_agents_with_nonexistent_directory_returns_empty_list(self):
    """Constructing with a nonexistent directory path returns an empty list."""
    loader = NestedAgentLoader("/nonexistent/directory/path")
    agents = loader.list_agents()

    assert agents == []

  def test_list_agents_finds_nested_agents_recursively(self):
    """Recursive directory walk discovers all agents with nested namespaces."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent_one/agent.py
      dir1 = temp_path / "agent_one"
      dir1.mkdir()
      (dir1 / "agent.py").write_text("")

      # Create sub_dir/agent_two/root_agent.yaml
      dir2 = temp_path / "sub_dir" / "agent_two"
      dir2.mkdir(parents=True)
      (dir2 / "root_agent.yaml").write_text("")

      # Create sub_dir/sub_sub_dir/agent_three/agent.py
      dir3 = temp_path / "sub_dir" / "sub_sub_dir" / "agent_three"
      dir3.mkdir(parents=True)
      (dir3 / "agent.py").write_text("")

      loader = NestedAgentLoader(str(temp_path))
      agents = loader.list_agents()

      assert agents == [
          "agent_one",
          "sub_dir.agent_two",
          "sub_dir.sub_sub_dir.agent_three",
      ]

  def test_list_agents_ignores_hidden_and_special_directories(self):
    """Discovered agents exclude hidden directories, pycache, and temp folders."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Valid agent
      dir_valid = temp_path / "valid_agent"
      dir_valid.mkdir()
      (dir_valid / "agent.py").write_text("")

      # Hidden agent (starts with dot)
      dir_hidden = temp_path / ".hidden_agent"
      dir_hidden.mkdir()
      (dir_hidden / "agent.py").write_text("")

      # Special directory __pycache__
      dir_pycache = temp_path / "__pycache__"
      dir_pycache.mkdir()
      (dir_pycache / "agent.py").write_text("")

      # Special directory tmp
      dir_tmp = temp_path / "tmp"
      dir_tmp.mkdir()
      (dir_tmp / "agent.py").write_text("")

      loader = NestedAgentLoader(str(temp_path))
      agents = loader.list_agents()

      assert agents == ["valid_agent"]

  def test_single_agent_directory_with_subfolders_remains_single_agent(self):
    """A directory with an agent.py at root remains in single-agent mode even with subfolders."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir) / "my_single_agent"
      temp_path.mkdir()

      # Create agent.py at the root of my_single_agent directory
      (temp_path / "agent.py").write_text("")

      # Create a nested subfolder containing an agent
      dir_nested = temp_path / "sub_agent"
      dir_nested.mkdir()
      (dir_nested / "agent.py").write_text("")

      loader = NestedAgentLoader(str(temp_path))
      assert loader.is_single_agent is True
      assert loader.single_agent_name == "my_single_agent"
      assert loader.list_agents() == ["my_single_agent"]

  def test_list_agents_sorts_discovered_agents_alphabetically(self):
    """Returned list of discovered agents is sorted alphabetically."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agents in non-alphabetical order
      for name in ["z_agent", "a_agent", "m_agent"]:
        agent_dir = temp_path / name
        agent_dir.mkdir()
        (agent_dir / "agent.py").write_text("")

      loader = NestedAgentLoader(str(temp_path))
      agents = loader.list_agents()

      assert agents == ["a_agent", "m_agent", "z_agent"]

  def test_remove_agent_from_cache_clears_module_keys_and_internal_cache(self):
    """remove_agent_from_cache removes agent from internal cache and deletes its sys.modules keys."""
    loader = NestedAgentLoader("/dummy/path")

    # Arrange: populate internal cache (with slash-separated path) and sys.modules (with dot-separated path)
    loader._agent_cache["foo/bar/my_agent"] = "dummy_agent_instance"
    sys.modules["foo.bar.my_agent"] = "dummy_module_instance"
    sys.modules["foo.bar.my_agent.submodule"] = "dummy_submodule_instance"

    # Act: remove the agent from cache
    loader.remove_agent_from_cache("foo/bar/my_agent")

    # Assert: cleared from loader's internal cache
    assert "foo/bar/my_agent" not in loader._agent_cache

    # Assert: cleared from sys.modules (both the module and its submodules)
    assert "foo.bar.my_agent" not in sys.modules
    assert "foo.bar.my_agent.submodule" not in sys.modules

  def test_nested_wrong_type_root_agent_raises_targeted_error(self):
    """A non-agent `root_agent` in nested path raises a type-mismatch error."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_dir = temp_path / "parent" / "mistyped_agent"
      agent_dir.mkdir(parents=True)
      agent_file = agent_dir / "agent.py"
      agent_file.write_text(dedent("""
                root_agent = "I am a string, not an agent"
            """))

      loader = NestedAgentLoader(str(temp_path))

      with pytest.raises(ValueError) as exc_info:
        loader.load_agent("parent.mistyped_agent")

      message = str(exc_info.value)
      assert "parent.mistyped_agent.agent.root_agent" in message
      assert "builtins.str" in message
      assert "No root_agent found" not in message

  def test_nested_app_exported_as_root_agent_suggests_app_name(self):
    """Exporting an App as `root_agent` in nested path points the user at 'app'."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_dir = temp_path / "parent" / "app_agent"
      agent_dir.mkdir(parents=True)
      agent_file = agent_dir / "agent.py"
      agent_file.write_text(dedent("""
                from google.adk.agents.base_agent import BaseAgent
                from google.adk.apps.app import App


                class MyAgent(BaseAgent):

                    def __init__(self):
                        super().__init__(name="my_agent")


                root_agent = App(name="app_agent", root_agent=MyAgent())
            """))

      loader = NestedAgentLoader(str(temp_path))

      with pytest.raises(ValueError) as exc_info:
        loader.load_agent("parent.app_agent")

      message = str(exc_info.value)
      assert "google.adk.apps.app.App" in message
      assert "under the name 'app'" in message

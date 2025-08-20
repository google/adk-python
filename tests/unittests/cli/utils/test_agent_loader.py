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

import os
from pathlib import Path
import sys
import tempfile
from textwrap import dedent

from google.adk.cli.utils.agent_loader import AgentLoader
from pydantic import ValidationError
import pytest


class TestAgentLoader:
  """Unit tests for AgentLoader focusing on interface behavior."""

  @pytest.fixture(autouse=True)
  def cleanup_sys_path(self):
    """Ensure sys.path is restored after each test."""
    original_path = sys.path.copy()
    original_env = os.environ.copy()
    yield
    sys.path[:] = original_path
    # Restore environment variables
    os.environ.clear()
    os.environ.update(original_env)

  def create_agent_structure(
      self, temp_dir: Path, agent_name: str, structure_type: str
  ):
    """Create different agent structures for testing.

    Args:
        temp_dir: The temporary directory to create the agent in
        agent_name: Name of the agent
        structure_type: One of 'module', 'package_with_root', 'package_with_agent_module'
    """
    if structure_type == "module":
      # Structure: agents_dir/agent_name.py
      agent_file = temp_dir / f"{agent_name}.py"
      agent_file.write_text(dedent(f"""
                import os
                from google.adk.agents.base_agent import BaseAgent
                from typing import Any

                class {agent_name.title()}Agent(BaseAgent):
                    agent_id: Any = None
                    config: Any = None

                    def __init__(self):
                        super().__init__(name="{agent_name}")
                        self.agent_id = id(self)
                        self.config = os.environ.get("AGENT_CONFIG", "default")

                root_agent = {agent_name.title()}Agent()


            """))

    elif structure_type == "package_with_root":
      # Structure: agents_dir/agent_name/__init__.py (with root_agent)
      agent_dir = temp_dir / agent_name
      agent_dir.mkdir()
      init_file = agent_dir / "__init__.py"
      init_file.write_text(dedent(f"""
                import os
                from google.adk.agents.base_agent import BaseAgent
                from typing import Any

                class {agent_name.title()}Agent(BaseAgent):
                    agent_id: Any = None
                    config: Any = None

                    def __init__(self):
                        super().__init__(name="{agent_name}")
                        self.agent_id = id(self)
                        self.config = os.environ.get("AGENT_CONFIG", "default")

                root_agent = {agent_name.title()}Agent()
            """))

    elif structure_type == "package_with_agent_module":
      # Structure: agents_dir/agent_name/agent.py
      agent_dir = temp_dir / agent_name
      agent_dir.mkdir()

      # Create __init__.py
      init_file = agent_dir / "__init__.py"
      init_file.write_text("")

      # Create agent.py with root_agent
      agent_file = agent_dir / "agent.py"
      agent_file.write_text(dedent(f"""
                import os
                from google.adk.agents.base_agent import BaseAgent
                from typing import Any

                class {agent_name.title()}Agent(BaseAgent):
                    agent_id: Any = None
                    config: Any = None

                    def __init__(self):
                        super().__init__(name="{agent_name}")
                        self.agent_id = id(self)
                        self.config = os.environ.get("AGENT_CONFIG", "default")

                root_agent = {agent_name.title()}Agent()
            """))

  def create_env_file(self, temp_dir: Path, agent_name: str, env_vars: dict):
    """Create a .env file for the agent."""
    env_file = temp_dir / agent_name / ".env"
    env_file.parent.mkdir(exist_ok=True)

    env_content = "\n".join(
        [f"{key}={value}" for key, value in env_vars.items()]
    )
    env_file.write_text(env_content)

  def test_load_agent_as_module(self):
    """Test loading an agent structured as a single module file."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent as module
      self.create_agent_structure(temp_path, "module_agent", "module")

      # Load the agent
      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent("module_agent")

      # Assert agent was loaded correctly
      assert agent.name == "module_agent"
      assert hasattr(agent, "agent_id")
      assert agent.config == "default"

  def test_load_agent_as_package_with_root_agent(self):
    """Test loading an agent structured as a package with root_agent in __init__.py."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent as package
      self.create_agent_structure(
          temp_path, "package_agent", "package_with_root"
      )

      # Load the agent
      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent("package_agent")

      # Assert agent was loaded correctly
      assert agent.name == "package_agent"
      assert hasattr(agent, "agent_id")

  def test_load_agent_as_package_with_agent_module(self):
    """Test loading an agent structured as a package with separate agent.py module."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent as package with agent.py
      self.create_agent_structure(
          temp_path, "modular_agent", "package_with_agent_module"
      )

      # Load the agent
      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent("modular_agent")

      # Assert agent was loaded correctly
      assert agent.name == "modular_agent"
      assert hasattr(agent, "agent_id")

  def test_agent_caching_returns_same_instance(self):
    """Test that loading the same agent twice returns the same instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent
      self.create_agent_structure(temp_path, "cached_agent", "module")

      # Load the agent twice
      loader = AgentLoader(str(temp_path))
      agent1 = loader.load_agent("cached_agent")
      agent2 = loader.load_agent("cached_agent")

      # Assert same instance is returned
      assert agent1 is agent2
      assert agent1.agent_id == agent2.agent_id

  def test_env_loading_for_agent(self):
    """Test that .env file is loaded for the agent."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent and .env file
      self.create_agent_structure(temp_path, "env_agent", "package_with_root")
      self.create_env_file(
          temp_path,
          "env_agent",
          {"AGENT_CONFIG": "production", "AGENT_SECRET": "test_secret_123"},
      )

      # Load the agent
      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent("env_agent")

      # Assert environment variables were loaded
      assert agent.config == "production"
      assert os.environ.get("AGENT_SECRET") == "test_secret_123"

  def test_loading_order_preference(self):
    """Test that module/package is preferred over agent.py in a sub-package."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "order_test_agent"

      # Create structure 1: agents_dir/agent_name.py (expected to be loaded)
      agent_module_file = temp_path / f"{agent_name}.py"
      agent_module_file.write_text(dedent(f"""
                from google.adk.agents.base_agent import BaseAgent
                class ModuleAgent(BaseAgent):
                    def __init__(self):
                        super().__init__(name="{agent_name}_module_version")
                root_agent = ModuleAgent()
            """))

      # Create structure 2: agents_dir/agent_name/agent.py (should be ignored)
      agent_package_dir = temp_path / agent_name
      agent_package_dir.mkdir()
      agent_submodule_file = agent_package_dir / "agent.py"
      agent_submodule_file.write_text(dedent(f"""
                from google.adk.agents.base_agent import BaseAgent
                class SubmoduleAgent(BaseAgent):
                    def __init__(self):
                        super().__init__(name="{agent_name}_submodule_version")
                root_agent = SubmoduleAgent()
            """))

      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent(agent_name)

      # Assert that the module version was loaded due to the new loading order
      assert agent.name == f"{agent_name}_module_version"

  def test_load_multiple_different_agents(self):
    """Test loading multiple different agents."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create multiple agents with different structures
      self.create_agent_structure(temp_path, "agent_one", "module")
      self.create_agent_structure(temp_path, "agent_two", "package_with_root")
      self.create_agent_structure(
          temp_path, "agent_three", "package_with_agent_module"
      )

      # Load all agents
      loader = AgentLoader(str(temp_path))
      agent1 = loader.load_agent("agent_one")
      agent2 = loader.load_agent("agent_two")
      agent3 = loader.load_agent("agent_three")

      # Assert all agents were loaded correctly and are different instances
      assert agent1.name == "agent_one"
      assert agent2.name == "agent_two"
      assert agent3.name == "agent_three"
      assert agent1 is not agent2
      assert agent2 is not agent3
      assert agent1.agent_id != agent2.agent_id != agent3.agent_id

  def test_agent_not_found_error(self):
    """Test that appropriate error is raised when agent is not found."""
    with tempfile.TemporaryDirectory() as temp_dir:
      loader = AgentLoader(temp_dir)
      agents_dir = temp_dir  # For use in the expected message string

      # Try to load non-existent agent
      with pytest.raises(ValueError) as exc_info:
        loader.load_agent("nonexistent_agent")

      expected_msg_part_1 = "No root_agent found for 'nonexistent_agent'."
      expected_msg_part_2 = (
          "Searched in 'nonexistent_agent.agent.root_agent',"
          " 'nonexistent_agent.root_agent' and"
          " 'nonexistent_agent/root_agent.yaml'."
      )
      expected_msg_part_3 = (
          f"Ensure '{agents_dir}/nonexistent_agent' is structured correctly"
      )

      assert expected_msg_part_1 in str(exc_info.value)
      assert expected_msg_part_2 in str(exc_info.value)
      assert expected_msg_part_3 in str(exc_info.value)

  def test_agent_without_root_agent_error(self):
    """Test that appropriate error is raised when agent has no root_agent."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent without root_agent
      agent_file = temp_path / "broken_agent.py"
      agent_file.write_text(dedent("""
                class BrokenAgent:
                    def __init__(self):
                        self.name = "broken"

                # Note: No root_agent defined
            """))

      loader = AgentLoader(str(temp_path))

      # Try to load agent without root_agent
      with pytest.raises(ValueError) as exc_info:
        loader.load_agent("broken_agent")

      assert "No root_agent found for 'broken_agent'" in str(exc_info.value)

  def test_agent_internal_module_not_found_error(self):
    """Test error when an agent tries to import a non-existent module."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "importer_agent"

      # Create agent that imports a non-existent module
      agent_file = temp_path / f"{agent_name}.py"
      agent_file.write_text(dedent(f"""
                from google.adk.agents.base_agent import BaseAgent
                import non_existent_module  # This will fail

                class {agent_name.title()}Agent(BaseAgent):
                    def __init__(self):
                        super().__init__(name="{agent_name}")

                root_agent = {agent_name.title()}Agent()
            """))

      loader = AgentLoader(str(temp_path))
      with pytest.raises(ModuleNotFoundError) as exc_info:
        loader.load_agent(agent_name)

      assert f"Fail to load '{agent_name}' module." in str(exc_info.value)
      assert "No module named 'non_existent_module'" in str(exc_info.value)

  def test_agent_internal_syntax_error(self):
    """Test other import errors within an agent's code (e.g., SyntaxError)."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "syntax_error_agent"

      # Create agent with a syntax error (which leads to ImportError)
      agent_file = temp_path / f"{agent_name}.py"
      agent_file.write_text(dedent(f"""
                from google.adk.agents.base_agent import BaseAgent

                # Invalid syntax
                this is not valid python code

                class {agent_name.title()}Agent(BaseAgent):
                    def __init__(self):
                        super().__init__(name="{agent_name}")

                root_agent = {agent_name.title()}Agent()
            """))

      loader = AgentLoader(str(temp_path))
      # SyntaxError is a subclass of Exception, and importlib might wrap it
      # The loader is expected to prepend its message and re-raise.
      with pytest.raises(
          SyntaxError
      ) as exc_info:  # Or potentially ImportError depending on Python version specifics with importlib
        loader.load_agent(agent_name)

      assert str(exc_info.value).startswith(
          f"Fail to load '{agent_name}' module."
      )
      # Check for part of the original SyntaxError message
      assert "invalid syntax" in str(exc_info.value).lower()

  def test_agent_internal_name_error(self):
    """Test other import errors within an agent's code (e.g., SyntaxError)."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "name_error_agent"

      # Create agent with a syntax error (which leads to ImportError)
      agent_file = temp_path / f"{agent_name}.py"
      agent_file.write_text(dedent(f"""
                from google.adk.agents.base_agent import BaseAgent

                # name is not defined
                print(non_existing_name)

                class {agent_name.title()}Agent(BaseAgent):
                    def __init__(self):
                        super().__init__(name="{agent_name}")

                root_agent = {agent_name.title()}Agent()
            """))

      loader = AgentLoader(str(temp_path))
      # SyntaxError is a subclass of Exception, and importlib might wrap it
      # The loader is expected to prepend its message and re-raise.
      with pytest.raises(
          NameError
      ) as exc_info:  # Or potentially ImportError depending on Python version specifics with importlib
        loader.load_agent(agent_name)

      assert str(exc_info.value).startswith(
          f"Fail to load '{agent_name}' module."
      )
      # Check for part of the original SyntaxError message
      assert "is not defined" in str(exc_info.value).lower()

  def test_sys_path_modification(self):
    """Test that agents_dir is added to sys.path correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Create agent
      self.create_agent_structure(temp_path, "path_agent", "module")

      # Check sys.path before
      assert str(temp_path) not in sys.path

      loader = AgentLoader(str(temp_path))

      # Path should not be added yet - only added during load
      assert str(temp_path) not in sys.path

      # Load agent - this should add the path
      agent = loader.load_agent("path_agent")

      # Now assert path was added
      assert str(temp_path) in sys.path
      assert agent.name == "path_agent"

  def create_yaml_agent_structure(
      self, temp_dir: Path, agent_name: str, yaml_content: str
  ):
    """Create an agent structure with YAML configuration.

    Args:
        temp_dir: The temporary directory to create the agent in
        agent_name: Name of the agent
        yaml_content: YAML content for the root_agent.yaml file
    """
    agent_dir = temp_dir / agent_name
    agent_dir.mkdir()

    # Create root_agent.yaml file
    yaml_file = agent_dir / "root_agent.yaml"
    yaml_file.write_text(yaml_content)

  def test_load_agent_from_yaml_config(self):
    """Test loading an agent from YAML configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "yaml_agent"

      # Create YAML configuration
      yaml_content = dedent("""
        agent_class: LlmAgent
        name: yaml_test_agent
        model: gemini-2.0-flash
        instruction: You are a test agent loaded from YAML configuration.
        description: A test agent created from YAML config
      """)

      self.create_yaml_agent_structure(temp_path, agent_name, yaml_content)

      # Load the agent
      loader = AgentLoader(str(temp_path))
      agent = loader.load_agent(agent_name)

      # Assert agent was loaded correctly
      assert agent.name == "yaml_test_agent"
      # Check if it's an LlmAgent before accessing model and instruction
      from google.adk.agents.llm_agent import LlmAgent

      if isinstance(agent, LlmAgent):
        assert agent.model == "gemini-2.0-flash"
        # Handle instruction which can be string or InstructionProvider
        instruction_text = str(agent.instruction)
        assert "test agent loaded from YAML" in instruction_text

  def test_yaml_agent_caching_returns_same_instance(self):
    """Test that loading the same YAML agent twice returns the same instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "cached_yaml_agent"

      # Create YAML configuration
      yaml_content = dedent("""
        agent_class: LlmAgent
        name: cached_yaml_test_agent
        model: gemini-2.0-flash
        instruction: You are a cached test agent.
      """)

      self.create_yaml_agent_structure(temp_path, agent_name, yaml_content)

      # Load the agent twice
      loader = AgentLoader(str(temp_path))
      agent1 = loader.load_agent(agent_name)
      agent2 = loader.load_agent(agent_name)

      # Assert same instance is returned
      assert agent1 is agent2
      assert agent1.name == agent2.name

  def test_yaml_agent_not_found_error(self):
    """Test that appropriate error is raised when YAML agent is not found."""
    with tempfile.TemporaryDirectory() as temp_dir:
      loader = AgentLoader(temp_dir)
      agents_dir = temp_dir  # For use in the expected message string

      # Try to load non-existent YAML agent
      with pytest.raises(ValueError) as exc_info:
        loader.load_agent("nonexistent_yaml_agent")

      expected_msg_part_1 = "No root_agent found for 'nonexistent_yaml_agent'."
      expected_msg_part_2 = (
          "Searched in 'nonexistent_yaml_agent.agent.root_agent',"
          " 'nonexistent_yaml_agent.root_agent' and"
          " 'nonexistent_yaml_agent/root_agent.yaml'."
      )
      expected_msg_part_3 = (
          f"Ensure '{agents_dir}/nonexistent_yaml_agent' is structured"
          " correctly"
      )

      assert expected_msg_part_1 in str(exc_info.value)
      assert expected_msg_part_2 in str(exc_info.value)
      assert expected_msg_part_3 in str(exc_info.value)

  def test_yaml_agent_invalid_yaml_error(self):
    """Test that appropriate error is raised when YAML is invalid."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)
      agent_name = "invalid_yaml_agent"

      # Create invalid YAML content with wrong field name
      invalid_yaml_content = dedent("""
        not_exist_field: invalid_yaml_test_agent
        model: gemini-2.0-flash
        instruction: You are a test agent with invalid YAML
      """)

      self.create_yaml_agent_structure(
          temp_path, agent_name, invalid_yaml_content
      )

      loader = AgentLoader(str(temp_path))

      # Try to load agent with invalid YAML
      with pytest.raises(ValidationError) as exc_info:
        loader.load_agent(agent_name)

      # Should raise some form of YAML parsing error
      assert "Extra inputs are not permitted" in str(exc_info.value)

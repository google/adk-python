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

from unittest import mock
import warnings

from google.adk.cli.utils import agent_loader
from google.adk.cli.utils.agent_change_handler import AgentChangeEventHandler
from google.adk.cli.utils.shared_value import SharedValue
import pytest
from watchdog.events import FileModifiedEvent


class TestAgentChangeEventHandler:
  """Unit tests for AgentChangeEventHandler file extension filtering."""

  @pytest.fixture
  def mock_agent_loader(self):
    """Create a mock AgentLoader constrained to the public API."""
    return mock.create_autospec(
        agent_loader.AgentLoader, instance=True, spec_set=True
    )

  @pytest.fixture
  def agents_dir(self, tmp_path):
    """Create a temporary agents directory structure."""
    d = tmp_path / "agents"
    d.mkdir()
    (d / "test_agent").mkdir()
    (d / "test_agent" / "agent.py").write_text("")
    (d / "test_agent" / "config.yaml").write_text("")
    (d / "test_agent" / "config.yml").write_text("")
    (d / "team" / "support").mkdir(parents=True)
    (d / "team" / "support" / "agent.py").write_text("")
    (d / "sub_dir" / "sub_sub_dir" / "agent_three").mkdir(parents=True)
    (
        d / "sub_dir" / "sub_sub_dir" / "agent_three" / "root_agent.yaml"
    ).write_text("")
    (d / "standalone_agent.py").write_text("")
    return d

  @pytest.fixture
  def handler(self, mock_agent_loader, agents_dir):
    """Create an AgentChangeEventHandler with mocked dependencies."""
    runners_to_clean = set()
    return AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

  @pytest.mark.parametrize(
      "rel_path,expected_agent_name",
      [
          pytest.param(
              "test_agent/agent.py",
              "test_agent",
              id="python_file",
          ),
          pytest.param(
              "test_agent/config.yaml",
              "test_agent",
              id="yaml_file",
          ),
          pytest.param(
              "test_agent/config.yml",
              "test_agent",
              id="yml_file",
          ),
          pytest.param(
              "team/support/agent.py",
              "team.support",
              id="nested_python_file",
          ),
          pytest.param(
              "sub_dir/sub_sub_dir/agent_three/root_agent.yaml",
              "sub_dir.sub_sub_dir.agent_three",
              id="deeply_nested_yaml_file",
          ),
          pytest.param(
              "standalone_agent.py",
              "standalone_agent",
              id="standalone_python_file",
          ),
      ],
  )
  def test_on_modified_triggers_reload_for_supported_extensions(
      self,
      handler,
      mock_agent_loader,
      agents_dir,
      rel_path,
      expected_agent_name,
  ):
    """Verify that .py, .yaml, and .yml files trigger agent reload."""
    file_path = agents_dir / rel_path
    event = FileModifiedEvent(src_path=str(file_path))

    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_any_call(
        expected_agent_name
    )
    assert (
        expected_agent_name in handler.runners_to_clean
    ), f"Expected '{expected_agent_name}' in runners_to_clean for {file_path}"

  @pytest.mark.parametrize(
      "filename",
      [
          pytest.param("file.json", id="json_file"),
          pytest.param("file.txt", id="txt_file"),
          pytest.param("file.md", id="markdown_file"),
          pytest.param("file.toml", id="toml_file"),
          pytest.param(".gitignore", id="gitignore_file"),
          pytest.param("file", id="no_extension"),
      ],
  )
  def test_on_modified_ignores_unsupported_extensions(
      self, handler, mock_agent_loader, agents_dir, filename
  ):
    """Verify that non-py/yaml/yml files do not trigger reload."""
    file_path = agents_dir / "test_agent" / filename
    file_path.write_text("")
    event = FileModifiedEvent(src_path=str(file_path))

    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_not_called()
    assert not handler.runners_to_clean, (
        f"Expected runners_to_clean to be empty for {file_path}, "
        f"got {handler.runners_to_clean}"
    )

  def test_on_modified_subfolder_file_derives_enclosing_agent_name(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that editing a file in an agent subfolder resolves to the enclosing agent."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    support_dir = agents_dir / "team" / "support"
    support_dir.mkdir(parents=True)
    (support_dir / "agent.py").write_text("")
    tools_dir = support_dir / "tools"
    tools_dir.mkdir()
    custom_tool = tools_dir / "custom_tool.py"
    custom_tool.write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(custom_tool))
    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_has_calls(
        [
            mock.call("team.support"),
            mock.call("team"),
        ],
        any_order=False,
    )
    assert handler.runners_to_clean == {"team.support", "team"}

  def test_on_modified_nested_agent_evicts_all_enclosing_agents(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that editing a sub-agent evicts both the sub-agent and its enclosing parent agent."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    parent_dir = agents_dir / "parent_agent"
    parent_dir.mkdir()
    (parent_dir / "agent.py").write_text("")
    greeter_dir = parent_dir / "sub_agents" / "greeter"
    greeter_dir.mkdir(parents=True)
    (greeter_dir / "agent.py").write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(greeter_dir / "agent.py"))
    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_has_calls(
        [
            mock.call("parent_agent.sub_agents.greeter"),
            mock.call("parent_agent"),
        ],
        any_order=False,
    )
    assert mock_agent_loader.remove_agent_from_cache.call_count == 2
    assert handler.runners_to_clean == {
        "parent_agent.sub_agents.greeter",
        "parent_agent",
    }

  def test_on_modified_non_agent_directory_ignores_event(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that modifying a file outside any agent directory does not trigger reload."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    helpers_dir = agents_dir / "helpers"
    helpers_dir.mkdir()
    helper_file = helpers_dir / "helper.py"
    helper_file.write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(helper_file))
    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_not_called()
    assert not handler.runners_to_clean

  def test_on_modified_outside_agents_dir_ignores_event(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that modifying a file outside the agents_dir does not trigger reload."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    outside_file = tmp_path / "outside.py"
    outside_file.write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(outside_file))
    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_not_called()
    assert not handler.runners_to_clean

  def test_on_modified_nested_agent_file_evicts_top_level_agent(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that editing a nested agent file evicts the top-level agent for flat loader."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    support_dir = agents_dir / "team" / "support"
    support_dir.mkdir(parents=True)
    (support_dir / "agent.py").write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(support_dir / "agent.py"))
    handler.on_modified(event)

    assert "team" in handler.runners_to_clean
    mock_agent_loader.remove_agent_from_cache.assert_any_call("team")

  def test_on_modified_package_style_agent_evicts_agent(
      self, mock_agent_loader, tmp_path
  ):
    """Verify that editing a package-style agent (__init__.py) triggers eviction."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    pkg_dir = agents_dir / "pkg_agent"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")

    runners_to_clean = set()
    handler = AgentChangeEventHandler(
        agent_loader=mock_agent_loader,
        runners_to_clean=runners_to_clean,
        agents_dir=str(agents_dir),
    )

    event = FileModifiedEvent(src_path=str(pkg_dir / "__init__.py"))
    handler.on_modified(event)

    assert "pkg_agent" in handler.runners_to_clean
    mock_agent_loader.remove_agent_from_cache.assert_any_call("pkg_agent")

  def test_init_with_deprecated_current_app_name_ref_warns(
      self, mock_agent_loader
  ):
    """Verify that passing current_app_name_ref emits a DeprecationWarning."""
    runners_to_clean = set()
    current_app_name_ref = SharedValue(value="test_agent")
    with pytest.deprecated_call():
      handler = AgentChangeEventHandler(
          agent_loader=mock_agent_loader,
          runners_to_clean=runners_to_clean,
          current_app_name_ref=current_app_name_ref,
      )
    assert handler.current_app_name_ref is current_app_name_ref
    assert handler.agents_dir is None

  def test_init_with_deprecated_current_app_name_ref_positional_warns(
      self, mock_agent_loader
  ):
    """Verify that passing SharedValue positionally emits a DeprecationWarning."""
    runners_to_clean = set()
    current_app_name_ref = SharedValue(value="test_agent")
    with pytest.deprecated_call():
      handler = AgentChangeEventHandler(
          mock_agent_loader,
          runners_to_clean,
          current_app_name_ref,
      )
    assert handler.current_app_name_ref is current_app_name_ref
    assert handler.agents_dir is None

  def test_on_modified_with_deprecated_current_app_name_ref_cleans_agent(
      self, mock_agent_loader
  ):
    """Verify that on_modified cleans the agent specified in current_app_name_ref."""
    runners_to_clean = set()
    current_app_name_ref = SharedValue(value="test_agent")
    with pytest.deprecated_call():
      handler = AgentChangeEventHandler(
          agent_loader=mock_agent_loader,
          runners_to_clean=runners_to_clean,
          current_app_name_ref=current_app_name_ref,
      )
    event = FileModifiedEvent(src_path="/some/path/agent.py")
    handler.on_modified(event)

    mock_agent_loader.remove_agent_from_cache.assert_called_once_with(
        "test_agent"
    )
    assert "test_agent" in handler.runners_to_clean

  def test_init_with_agents_dir_positional(self, mock_agent_loader, agents_dir):
    """Verify that passing agents_dir as a string positionally works without warning."""
    runners_to_clean = set()
    with warnings.catch_warnings():
      warnings.simplefilter("error", DeprecationWarning)
      handler = AgentChangeEventHandler(
          mock_agent_loader,
          runners_to_clean,
          str(agents_dir),
      )
    assert handler.agents_dir == agents_dir.resolve()
    assert handler.current_app_name_ref is None

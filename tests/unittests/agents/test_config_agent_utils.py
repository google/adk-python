import pytest

from google.adk.agents.config_agent_utils import check_config_for_blocked_keys


def test_check_config_for_blocked_keys_rejects_args_key():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [
          {
              "name": "some_project.tools.create_tool",
              "args": [],
          }
      ],
  }

  with pytest.raises(ValueError, match="Blocked key 'args'"):
    check_config_for_blocked_keys(config, "root_agent.yaml")


def test_check_config_for_blocked_keys_rejects_blocked_tool_module():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [
          {
              "name": "subprocess.run",
          }
      ],
  }

  with pytest.raises(ValueError, match="Blocked code reference 'subprocess.run'"):
    check_config_for_blocked_keys(config, "root_agent.yaml")


def test_check_config_for_blocked_keys_allows_non_blocked_tool_reference():
  config = {
      "name": "test_agent",
      "model": "gemini-2.0-flash",
      "tools": [
          {
              "name": "my_project.tools.echo",
          }
      ],
  }

  check_config_for_blocked_keys(config, "root_agent.yaml")
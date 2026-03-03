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

import ntpath
import os
from pathlib import Path
from textwrap import dedent
from typing import Literal
from typing import Type
from unittest import mock

from google.adk.agents import config_agent_utils
from google.adk.agents.agent_config import AgentConfig
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.common_configs import AgentRefConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
import pytest
import yaml


def test_agent_config_discriminator_default_is_llm_agent(tmp_path: Path):
  yaml_content = """\
name: search_agent
model: gemini-2.0-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert config.root.agent_class == "LlmAgent"


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "LlmAgent",
        "google.adk.agents.LlmAgent",
        "google.adk.agents.llm_agent.LlmAgent",
    ],
)
def test_agent_config_discriminator_llm_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: search_agent
model: gemini-2.0-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "LoopAgent",
        "google.adk.agents.LoopAgent",
        "google.adk.agents.loop_agent.LoopAgent",
    ],
)
def test_agent_config_discriminator_loop_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LoopAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "ParallelAgent",
        "google.adk.agents.ParallelAgent",
        "google.adk.agents.parallel_agent.ParallelAgent",
    ],
)
def test_agent_config_discriminator_parallel_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, ParallelAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "SequentialAgent",
        "google.adk.agents.SequentialAgent",
        "google.adk.agents.sequential_agent.SequentialAgent",
    ],
)
def test_agent_config_discriminator_sequential_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, SequentialAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    ("agent_class_value", "expected_agent_type"),
    [
        ("LoopAgent", LoopAgent),
        ("google.adk.agents.LoopAgent", LoopAgent),
        ("google.adk.agents.loop_agent.LoopAgent", LoopAgent),
        ("ParallelAgent", ParallelAgent),
        ("google.adk.agents.ParallelAgent", ParallelAgent),
        ("google.adk.agents.parallel_agent.ParallelAgent", ParallelAgent),
        ("SequentialAgent", SequentialAgent),
        ("google.adk.agents.SequentialAgent", SequentialAgent),
        ("google.adk.agents.sequential_agent.SequentialAgent", SequentialAgent),
    ],
)
def test_agent_config_discriminator_with_sub_agents(
    agent_class_value: str, expected_agent_type: Type[BaseAgent], tmp_path: Path
):
  # Create sub-agent config files
  sub_agent_dir = tmp_path / "sub_agents"
  sub_agent_dir.mkdir()
  sub_agent_config = """\
name: sub_agent_{index}
model: gemini-2.0-flash
description: a sub agent
instruction: sub agent instruction
"""
  (sub_agent_dir / "sub_agent1.yaml").write_text(
      sub_agent_config.format(index=1)
  )
  (sub_agent_dir / "sub_agent2.yaml").write_text(
      sub_agent_config.format(index=2)
  )
  yaml_content = f"""\
agent_class: {agent_class_value}
name: main_agent
description: main agent with sub agents
sub_agents:
  - config_path: sub_agents/sub_agent1.yaml
  - config_path: sub_agents/sub_agent2.yaml
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, expected_agent_type)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    ("agent_class_value", "expected_agent_type"),
    [
        ("LlmAgent", LlmAgent),
        ("google.adk.agents.LlmAgent", LlmAgent),
        ("google.adk.agents.llm_agent.LlmAgent", LlmAgent),
    ],
)
def test_agent_config_discriminator_llm_agent_with_sub_agents(
    agent_class_value: str, expected_agent_type: Type[BaseAgent], tmp_path: Path
):
  # Create sub-agent config files
  sub_agent_dir = tmp_path / "sub_agents"
  sub_agent_dir.mkdir()
  sub_agent_config = """\
name: sub_agent_{index}
model: gemini-2.0-flash
description: a sub agent
instruction: sub agent instruction
"""
  (sub_agent_dir / "sub_agent1.yaml").write_text(
      sub_agent_config.format(index=1)
  )
  (sub_agent_dir / "sub_agent2.yaml").write_text(
      sub_agent_config.format(index=2)
  )
  yaml_content = f"""\
agent_class: {agent_class_value}
name: main_agent
model: gemini-2.0-flash
description: main agent with sub agents
instruction: main agent instruction
sub_agents:
  - config_path: sub_agents/sub_agent1.yaml
  - config_path: sub_agents/sub_agent2.yaml
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, expected_agent_type)
  assert config.root.agent_class == agent_class_value


def test_agent_config_litellm_model_with_custom_args(tmp_path: Path):
  yaml_content = """\
name: managed_api_agent
description: Agent using LiteLLM managed endpoint
instruction: Respond concisely.
model_code:
  name: google.adk.models.lite_llm.LiteLlm
  args:
    - name: model
      value: kimi/k2
    - name: api_base
      value: https://proxy.litellm.ai/v1
"""
  config_file = tmp_path / "litellm_agent.yaml"
  config_file.write_text(yaml_content)

  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert isinstance(agent.model, LiteLlm)
  assert agent.model.model == "kimi/k2"
  assert agent.model._additional_args.get("api_base") == (
      "https://proxy.litellm.ai/v1"
  )


def test_agent_config_legacy_model_mapping_still_supported(tmp_path: Path):
  yaml_content = """\
name: managed_api_agent
description: Agent using LiteLLM managed endpoint
instruction: Respond concisely.
model:
  name: google.adk.models.lite_llm.LiteLlm
  args:
    - name: model
      value: kimi/k2
"""
  config_file = tmp_path / "legacy_litellm_agent.yaml"
  config_file.write_text(yaml_content)

  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert isinstance(agent.model, LiteLlm)
  assert agent.model.model == "kimi/k2"


def test_agent_config_discriminator_custom_agent():
  class MyCustomAgentConfig(BaseAgentConfig):
    agent_class: Literal["mylib.agents.MyCustomAgent"] = (
        "mylib.agents.MyCustomAgent"
    )
    other_field: str

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
other_field: other value
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  # pylint: disable=unidiomatic-typecheck Needs exact class matching.
  assert type(config.root) is BaseAgentConfig
  assert config.root.agent_class == "mylib.agents.MyCustomAgent"
  assert config.root.model_extra == {"other_field": "other value"}

  my_custom_config = MyCustomAgentConfig.model_validate(
      config.root.model_dump()
  )
  assert my_custom_config.other_field == "other value"


@pytest.mark.parametrize(
    ("config_rel_path", "child_rel_path", "child_name", "instruction"),
    [
        (
            Path("main.yaml"),
            Path("sub_agents/child.yaml"),
            "child_agent",
            "I am a child agent",
        ),
        (
            Path("level1/level2/nested_main.yaml"),
            Path("sub/nested_child.yaml"),
            "nested_child",
            "I am nested",
        ),
    ],
)
def test_resolve_agent_reference_resolves_relative_paths(
    config_rel_path: Path,
    child_rel_path: Path,
    child_name: str,
    instruction: str,
    tmp_path: Path,
):
  """Verify resolve_agent_reference resolves relative sub-agent paths."""
  config_file = tmp_path / config_rel_path
  config_file.parent.mkdir(parents=True, exist_ok=True)

  child_config_path = config_file.parent / child_rel_path
  child_config_path.parent.mkdir(parents=True, exist_ok=True)
  child_config_path.write_text(dedent(f"""
          agent_class: LlmAgent
          name: {child_name}
          model: gemini-2.0-flash
          instruction: {instruction}
          """).lstrip())

  config_file.write_text(dedent(f"""
          agent_class: LlmAgent
          name: main_agent
          model: gemini-2.0-flash
          instruction: I am the main agent
          sub_agents:
            - config_path: {child_rel_path.as_posix()}
          """).lstrip())

  ref_config = AgentRefConfig(config_path=child_rel_path.as_posix())
  agent = config_agent_utils.resolve_agent_reference(
      ref_config, str(config_file)
  )

  assert agent.name == child_name

  config_dir = os.path.dirname(str(config_file.resolve()))
  assert config_dir == str(config_file.parent.resolve())

  expected_child_path = os.path.join(config_dir, *child_rel_path.parts)
  assert os.path.exists(expected_child_path)


def test_resolve_agent_reference_uses_windows_dirname():
  """Ensure Windows-style config references resolve via os.path.dirname."""
  ref_config = AgentRefConfig(config_path="sub\\child.yaml")
  recorded: dict[str, str] = {}

  def fake_from_config(path: str):
    recorded["path"] = path
    return "sentinel"

  with (
      mock.patch.object(
          config_agent_utils,
          "from_config",
          autospec=True,
          side_effect=fake_from_config,
      ),
      mock.patch.object(config_agent_utils.os, "path", ntpath),
  ):
    referencing = r"C:\workspace\agents\main.yaml"
    result = config_agent_utils.resolve_agent_reference(ref_config, referencing)

  expected_path = ntpath.join(
      ntpath.dirname(referencing), ref_config.config_path
  )
  assert result == "sentinel"
  assert recorded["path"] == expected_path

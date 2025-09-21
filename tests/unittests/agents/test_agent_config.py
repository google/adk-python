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

from pathlib import Path
from typing import Literal
from typing import Type

from google.adk.agents import config_agent_utils
from google.adk.agents.agent_config import AgentConfig
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
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

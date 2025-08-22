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

from typing import Literal

from google.adk.agents.agent_config import AgentConfig
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.llm_agent_config import LlmAgentConfig
from google.adk.agents.loop_agent_config import LoopAgentConfig
from google.adk.agents.parallel_agent_config import ParallelAgentConfig
from google.adk.agents.sequential_agent_config import SequentialAgentConfig
import yaml


def test_agent_config_discriminator_default_is_llm_agent():
  yaml_content = """\
name: search_agent
model: gemini-2.0-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  assert isinstance(config.root, LlmAgentConfig)
  assert config.root.agent_class == "LlmAgent"


def test_agent_config_discriminator_llm_agent():
  yaml_content = """\
agent_class: LlmAgent
name: search_agent
model: gemini-2.0-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  assert isinstance(config.root, LlmAgentConfig)
  assert config.root.agent_class == "LlmAgent"


def test_agent_config_discriminator_loop_agent():
  yaml_content = """\
agent_class: LoopAgent
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents:
  - config_path: sub_agents/code_writer_agent.yaml
  - config_path: sub_agents/code_reviewer_agent.yaml
  - config_path: sub_agents/code_refactorer_agent.yaml
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  assert isinstance(config.root, LoopAgentConfig)
  assert config.root.agent_class == "LoopAgent"


def test_agent_config_discriminator_parallel_agent():
  yaml_content = """\
agent_class: ParallelAgent
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents:
  - config_path: sub_agents/code_writer_agent.yaml
  - config_path: sub_agents/code_reviewer_agent.yaml
  - config_path: sub_agents/code_refactorer_agent.yaml
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  assert isinstance(config.root, ParallelAgentConfig)
  assert config.root.agent_class == "ParallelAgent"


def test_agent_config_discriminator_sequential_agent():
  yaml_content = """\
agent_class: SequentialAgent
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents:
  - config_path: sub_agents/code_writer_agent.yaml
  - config_path: sub_agents/code_reviewer_agent.yaml
  - config_path: sub_agents/code_refactorer_agent.yaml
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  assert isinstance(config.root, SequentialAgentConfig)
  assert config.root.agent_class == "SequentialAgent"


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

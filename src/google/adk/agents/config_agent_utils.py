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

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ..utils.feature_decorator import working_in_progress
from .agent_config import AgentConfig
from .base_agent import BaseAgent
from .llm_agent import LlmAgent
from .llm_agent import LlmAgentConfig
from .loop_agent import LoopAgent
from .loop_agent import LoopAgentConfig
from .parallel_agent import ParallelAgent
from .parallel_agent import ParallelAgentConfig
from .sequential_agent import SequentialAgent
from .sequential_agent import SequentialAgentConfig


@working_in_progress("from_config is not ready for use.")
def from_config(config_path: str) -> BaseAgent:
  """Build agent from a configfile path.

  Args:
    config: the path to a YAML config file.

  Returns:
    The created agent instance.

  Raises:
    FileNotFoundError: If config file doesn't exist.
    ValidationError: If config file's content is invalid YAML.
    ValueError: If agent type is unsupported.
  """
  abs_path = os.path.abspath(config_path)
  config = _load_config_from_path(abs_path)

  if isinstance(config.root, LlmAgentConfig):
    return LlmAgent.from_config(config.root)
  elif isinstance(config.root, LoopAgentConfig):
    return LoopAgent.from_config(config.root)
  elif isinstance(config.root, ParallelAgentConfig):
    return ParallelAgent.from_config(config.root)
  elif isinstance(config.root, SequentialAgentConfig):
    return SequentialAgent.from_config(config.root)
  else:
    raise ValueError("Unsupported config type")


@working_in_progress("_load_config_from_path is not ready for use.")
def _load_config_from_path(config_path: str) -> AgentConfig:
  """Load an agent's configuration from a YAML file.

  Args:
    config_path: Path to the YAML config file. Both relative and absolute
      paths are accepted.

  Returns:
    The loaded and validated AgentConfig object.

  Raises:
    FileNotFoundError: If config file doesn't exist.
    ValidationError: If config file's content is invalid YAML.
  """
  config_path = Path(config_path)

  if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

  with open(config_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

  return AgentConfig.model_validate(config_data)

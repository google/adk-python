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

import importlib
import os
from typing import Any
from typing import List

import yaml

from ..utils.feature_decorator import working_in_progress
from .agent_config import AgentConfig
from .base_agent import BaseAgent
from .base_agent import SubAgentConfig
from .common_configs import CodeConfig
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
    return LlmAgent.from_config(config.root, abs_path)
  elif isinstance(config.root, LoopAgentConfig):
    return LoopAgent.from_config(config.root, abs_path)
  elif isinstance(config.root, ParallelAgentConfig):
    return ParallelAgent.from_config(config.root, abs_path)
  elif isinstance(config.root, SequentialAgentConfig):
    return SequentialAgent.from_config(config.root, abs_path)
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
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

  with open(config_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

  return AgentConfig.model_validate(config_data)


@working_in_progress("build_sub_agent is not ready for use.")
def build_sub_agent(
    sub_config: SubAgentConfig, parent_agent_folder_path: str
) -> BaseAgent:
  """Build a sub-agent from configuration.

  Args:
    sub_config: The sub-agent configuration (SubAgentConfig).
    parent_agent_folder_path: The folder path to the parent agent's YAML config.

  Returns:
    The created sub-agent instance.
  """
  if sub_config.config:
    if os.path.isabs(sub_config.config):
      return from_config(sub_config.config)
    else:
      return from_config(
          os.path.join(parent_agent_folder_path, sub_config.config)
      )
  elif sub_config.code:
    return _resolve_sub_agent_code_reference(sub_config.code)
  else:
    raise ValueError("SubAgentConfig must have either 'code' or 'config'")


@working_in_progress("_resolve_sub_agent_code_reference is not ready for use.")
def _resolve_sub_agent_code_reference(code: str) -> Any:
  """Resolve a code reference to an actual agent object.

  Args:
    code: The code reference to the sub-agent.

  Returns:
    The resolved agent object.

  Raises:
    ValueError: If the code reference cannot be resolved.
  """
  if "." not in code:
    raise ValueError(f"Invalid code reference: {code}")

  module_path, obj_name = code.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  if callable(obj):
    raise ValueError(f"Invalid code reference to a callable: {code}")

  return obj


@working_in_progress("resolve_code_reference is not ready for use.")
def resolve_code_reference(code_config: CodeConfig) -> Any:
  """Resolve a code reference to actual Python object.

  Args:
    code_config: The code configuration (CodeConfig).

  Returns:
    The resolved Python object.

  Raises:
    ValueError: If the code reference cannot be resolved.
  """
  if not code_config or not code_config.name:
    raise ValueError("Invalid CodeConfig.")

  module_path, obj_name = code_config.name.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  if code_config.args and callable(obj):
    kwargs = {arg.name: arg.value for arg in code_config.args if arg.name}
    positional_args = [arg.value for arg in code_config.args if not arg.name]

    return obj(*positional_args, **kwargs)
  else:
    return obj


@working_in_progress("resolve_callbacks is not ready for use.")
def resolve_callbacks(callbacks_config: List[CodeConfig]) -> Any:
  """Resolve callbacks from configuration.

  Args:
    callbacks_config: List of callback configurations (CodeConfig objects).

  Returns:
    List of resolved callback objects.
  """
  return [resolve_code_reference(config) for config in callbacks_config]

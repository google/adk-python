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
import inspect
import os
from typing import Any
from typing import List

import yaml

from ..utils.feature_decorator import working_in_progress
from .agent_config import AgentConfig
from .base_agent import BaseAgent
from .base_agent_config import BaseAgentConfig
from .common_configs import AgentRefConfig
from .common_configs import CodeConfig


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
  agent_config = config.root

  # pylint: disable=unidiomatic-typecheck Needs exact class matching.
  if type(agent_config) is BaseAgentConfig:
    # Resolve the concrete agent config for user-defined agent classes.
    agent_class = _resolve_agent_class(agent_config.agent_class)
    agent_config = agent_class.config_type.model_validate(
        agent_config.model_dump()
    )
    return agent_class.from_config(agent_config, abs_path)
  else:
    # For built-in agent classes, no need to re-validate.
    agent_class = _resolve_agent_class(agent_config.agent_class)
    return agent_class.from_config(agent_config, abs_path)


def _resolve_agent_class(agent_class: str) -> type[BaseAgent]:
  """Resolve the agent class from its fully qualified name."""
  agent_class_name = agent_class or "LlmAgent"
  if "." not in agent_class_name:
    agent_class_name = f"google.adk.agents.{agent_class_name}"

  agent_class = _resolve_fully_qualified_name(agent_class_name)
  if inspect.isclass(agent_class) and issubclass(agent_class, BaseAgent):
    return agent_class

  raise ValueError(
      f"Invalid agent class `{agent_class_name}`. It must be a subclass of"
      " BaseAgent."
  )


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


@working_in_progress("_resolve_fully_qualified_name is not ready for use.")
def _resolve_fully_qualified_name(name: str) -> Any:
  try:
    module_path, obj_name = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
  except Exception as e:
    raise ValueError(f"Invalid fully qualified name: {name}") from e


@working_in_progress("resolve_agent_reference is not ready for use.")
def resolve_agent_reference(
    ref_config: AgentRefConfig, referencing_agent_config_abs_path: str
) -> BaseAgent:
  """Build an agent from a reference.

  Args:
    ref_config: The agent reference configuration (AgentRefConfig).
    referencing_agent_config_abs_path: The absolute path to the agent config
    that contains the reference.

  Returns:
    The created agent instance.
  """
  if ref_config.config_path:
    if os.path.isabs(ref_config.config_path):
      return from_config(ref_config.config_path)
    else:
      return from_config(
          os.path.join(
              referencing_agent_config_abs_path.rsplit("/", 1)[0],
              ref_config.config_path,
          )
      )
  elif ref_config.code:
    return _resolve_agent_code_reference(ref_config.code)
  else:
    raise ValueError("AgentRefConfig must have either 'code' or 'config_path'")


@working_in_progress("_resolve_agent_code_reference is not ready for use.")
def _resolve_agent_code_reference(code: str) -> Any:
  """Resolve a code reference to an actual agent instance.

  Args:
    code: The fully-qualified path to an agent instance.

  Returns:
    The resolved agent instance.

  Raises:
    ValueError: If the agent reference cannot be resolved.
  """
  if "." not in code:
    raise ValueError(f"Invalid code reference: {code}")

  module_path, obj_name = code.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  if callable(obj):
    raise ValueError(f"Invalid agent reference to a callable: {code}")

  if not isinstance(obj, BaseAgent):
    raise ValueError(f"Invalid agent reference to a non-agent instance: {code}")

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
